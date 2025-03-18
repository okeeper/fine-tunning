#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估脚本，用于评估微调后模型的性能、过拟合和灾难性遗忘情况
"""

import os
import sys
import json
import logging
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    pipeline
)
from peft import PeftModel, PeftConfig
from datasets import load_dataset, load_from_disk, Dataset
import evaluate

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    evaluate_perplexity,
    analyze_overfitting,
    load_training_args
)

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估微调后的模型")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="微调后模型的路径"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="基础模型的路径，用于对比评估"
    )
    parser.add_argument(
        "--eval_data_dir",
        type=str,
        default="./data/processed",
        help="评估数据集目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="评估结果输出目录"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="评估批次大小"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="每个任务的最大评估样本数"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="评估设备"
    )
    parser.add_argument(
        "--eval_original_tasks",
        action="store_true",
        help="是否评估原始任务性能（用于检测灾难性遗忘）"
    )
    parser.add_argument(
        "--perplexity_eval",
        action="store_true",
        help="是否计算困惑度"
    )
    parser.add_argument(
        "--knowledge_eval",
        action="store_true",
        help="是否进行知识保留测试"
    )
    parser.add_argument(
        "--overfitting_analysis",
        action="store_true",
        help="是否进行过拟合分析"
    )
    
    return parser.parse_args()

def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    load_in_4bit: bool = True
) -> Tuple:
    """
    加载模型和分词器
    
    Args:
        model_path: 模型路径
        device: 设备
        load_in_4bit: 是否以4位精度加载
        
    Returns:
        模型和分词器
    """
    logger.info(f"加载模型: {model_path}")
    
    # 检查是否为PEFT模型
    is_peft_model = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    
    if is_peft_model:
        # 加载PEFT配置
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_path = peft_config.base_model_name_or_path
        
        # 配置量化参数
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            bnb_config = None
        
        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # 加载PEFT适配器
        model = PeftModel.from_pretrained(model, model_path)
    else:
        # 直接加载模型
        if load_in_4bit and device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            bnb_config = None
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path if not is_peft_model else base_model_path)
    
    # 确保分词器有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def evaluate_finetuned_task(
    model,
    tokenizer,
    eval_dataset: Dataset,
    batch_size: int = 4,
    device: str = "cuda",
    max_samples: Optional[int] = None
) -> Dict:
    """
    评估微调任务性能
    
    Args:
        model: 模型
        tokenizer: 分词器
        eval_dataset: 评估数据集
        batch_size: 批次大小
        device: 设备
        max_samples: 最大样本数
        
    Returns:
        评估结果
    """
    logger.info("评估微调任务性能")
    
    # 限制样本数量
    if max_samples is not None and len(eval_dataset) > max_samples:
        eval_dataset = eval_dataset.select(range(max_samples))
    
    # 计算困惑度
    perplexity = evaluate_perplexity(model, tokenizer, eval_dataset, batch_size, device)
    
    # 创建生成管道
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    # 评估生成质量
    correct = 0
    total = 0
    generated_texts = []
    
    for i in range(0, len(eval_dataset), batch_size):
        batch = eval_dataset[i:min(i+batch_size, len(eval_dataset))]
        
        for item in batch:
            instruction = item["instruction"] if "instruction" in item else item["input"] if "input" in item else item["question"]
            expected_response = item["response"] if "response" in item else item["output"] if "output" in item else item["answer"]
            
            # 生成回复
            prompt = f"<s>[INST] {instruction} [/INST]"
            generated = generator(prompt, max_new_tokens=256)[0]["generated_text"]
            
            # 提取生成的回复
            response = generated.split("[/INST]")[-1].strip()
            
            # 简单的精确匹配评估（实际应用中应使用更复杂的评估方法）
            if response == expected_response:
                correct += 1
            
            total += 1
            generated_texts.append({
                "instruction": instruction,
                "expected_response": expected_response,
                "generated_response": response
            })
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        "perplexity": perplexity,
        "accuracy": accuracy,
        "generated_texts": generated_texts
    }

def evaluate_squad(
    model,
    tokenizer,
    dataset_path: str,
    batch_size: int = 4,
    device: str = "cuda",
    max_samples: Optional[int] = None
) -> Dict:
    """
    评估SQuAD问答任务
    
    Args:
        model: 模型
        tokenizer: 分词器
        dataset_path: 数据集路径
        batch_size: 批次大小
        device: 设备
        max_samples: 最大样本数
        
    Returns:
        评估结果
    """
    logger.info("评估SQuAD问答任务")
    
    # 加载数据集
    dataset = load_from_disk(dataset_path)
    
    # 使用验证集
    eval_dataset = dataset["validation"]
    
    # 限制样本数量
    if max_samples is not None and len(eval_dataset) > max_samples:
        eval_dataset = eval_dataset.select(range(max_samples))
    
    # 创建问答管道
    qa_pipeline = pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1
    )
    
    # 评估问答性能
    exact_match = 0
    f1_scores = []
    
    for i in range(0, len(eval_dataset), batch_size):
        batch = eval_dataset[i:min(i+batch_size, len(eval_dataset))]
        
        for item in batch:
            question = item["question"]
            context = item["context"]
            expected_answer = item["answer"]
            
            # 生成回答
            result = qa_pipeline(question=question, context=context)
            predicted_answer = result["answer"]
            
            # 计算精确匹配
            if predicted_answer.lower() == expected_answer.lower():
                exact_match += 1
            
            # 计算F1分数
            f1 = compute_f1(predicted_answer, expected_answer)
            f1_scores.append(f1)
    
    exact_match_score = exact_match / len(eval_dataset)
    avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    return {
        "exact_match": exact_match_score,
        "f1": avg_f1_score
    }

def evaluate_glue(
    model,
    tokenizer,
    dataset_path: str,
    task_name: str,
    batch_size: int = 4,
    device: str = "cuda",
    max_samples: Optional[int] = None
) -> Dict:
    """
    评估GLUE分类任务
    
    Args:
        model: 模型
        tokenizer: 分词器
        dataset_path: 数据集路径
        task_name: 任务名称
        batch_size: 批次大小
        device: 设备
        max_samples: 最大样本数
        
    Returns:
        评估结果
    """
    logger.info(f"评估GLUE {task_name}任务")
    
    # 加载数据集
    dataset = load_from_disk(dataset_path)
    
    # 使用验证集
    eval_dataset = dataset["validation"]
    
    # 限制样本数量
    if max_samples is not None and len(eval_dataset) > max_samples:
        eval_dataset = eval_dataset.select(range(max_samples))
    
    # 创建文本分类管道
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1
    )
    
    # 评估分类性能
    correct = 0
    
    for i in range(0, len(eval_dataset), batch_size):
        batch = eval_dataset[i:min(i+batch_size, len(eval_dataset))]
        
        for item in batch:
            if task_name == "sst2":
                text = item["sentence"]
                expected_label = item["label"]
            elif task_name in ["mnli", "rte"]:
                text = f"{item['premise']} [SEP] {item['hypothesis']}"
                expected_label = item["label"]
            else:
                continue
            
            # 生成预测
            result = classifier(text)
            predicted_label = int(result[0]["label"].split("_")[-1])
            
            # 计算准确率
            if predicted_label == expected_label:
                correct += 1
    
    accuracy = correct / len(eval_dataset)
    
    return {
        "accuracy": accuracy
    }

def evaluate_truthful_qa(
    model,
    tokenizer,
    dataset_path: str,
    batch_size: int = 4,
    device: str = "cuda",
    max_samples: Optional[int] = None
) -> Dict:
    """
    评估TruthfulQA知识保留
    
    Args:
        model: 模型
        tokenizer: 分词器
        dataset_path: 数据集路径
        batch_size: 批次大小
        device: 设备
        max_samples: 最大样本数
        
    Returns:
        评估结果
    """
    logger.info("评估TruthfulQA知识保留")
    
    # 加载数据集
    dataset = load_from_disk(dataset_path)
    
    # 使用验证集
    eval_dataset = dataset["validation"]
    
    # 限制样本数量
    if max_samples is not None and len(eval_dataset) > max_samples:
        eval_dataset = eval_dataset.select(range(max_samples))
    
    # 创建生成管道
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
        max_new_tokens=64,
        do_sample=False
    )
    
    # 评估知识保留
    correct = 0
    
    for i in range(0, len(eval_dataset), batch_size):
        batch = eval_dataset[i:min(i+batch_size, len(eval_dataset))]
        
        for item in batch:
            question = item["question"]
            correct_answers = item["correct_answers"]
            incorrect_answers = item["incorrect_answers"]
            
            # 生成回答
            prompt = f"<s>[INST] {question} [/INST]"
            generated = generator(prompt, max_new_tokens=64)[0]["generated_text"]
            
            # 提取生成的回答
            answer = generated.split("[/INST]")[-1].strip()
            
            # 检查是否与正确答案匹配
            is_correct = False
            for correct_answer in correct_answers:
                if correct_answer.lower() in answer.lower():
                    is_correct = True
                    break
            
            # 检查是否与错误答案匹配
            is_incorrect = False
            for incorrect_answer in incorrect_answers:
                if incorrect_answer.lower() in answer.lower():
                    is_incorrect = True
                    break
            
            # 只有匹配正确答案且不匹配错误答案才算正确
            if is_correct and not is_incorrect:
                correct += 1
    
    accuracy = correct / len(eval_dataset)
    
    return {
        "accuracy": accuracy
    }

def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    计算F1分数
    
    Args:
        prediction: 预测文本
        ground_truth: 真实文本
        
    Returns:
        F1分数
    """
    prediction_tokens = prediction.lower().split()
    ground_truth_tokens = ground_truth.lower().split()
    
    common = set(prediction_tokens) & set(ground_truth_tokens)
    
    # 如果没有共同词，则F1为0
    if len(common) == 0:
        return 0
    
    precision = len(common) / len(prediction_tokens)
    recall = len(common) / len(ground_truth_tokens)
    
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1

def compare_models(
    finetuned_results: Dict,
    base_results: Dict,
    output_dir: str
) -> Dict:
    """
    比较微调模型和基础模型的性能
    
    Args:
        finetuned_results: 微调模型结果
        base_results: 基础模型结果
        output_dir: 输出目录
        
    Returns:
        比较结果
    """
    logger.info("比较模型性能")
    
    comparison = {}
    
    # 比较各项指标
    for task, metrics in finetuned_results.items():
        if task in base_results:
            task_comparison = {}
            
            for metric, value in metrics.items():
                if metric in base_results[task] and metric != "generated_texts":
                    base_value = base_results[task][metric]
                    difference = value - base_value
                    relative_change = difference / base_value if base_value != 0 else float('inf')
                    
                    task_comparison[metric] = {
                        "finetuned": value,
                        "base": base_value,
                        "difference": difference,
                        "relative_change": relative_change
                    }
            
            comparison[task] = task_comparison
    
    # 绘制比较图表
    plot_comparison(comparison, output_dir)
    
    return comparison

def plot_comparison(comparison: Dict, output_dir: str) -> None:
    """
    绘制模型比较图表
    
    Args:
        comparison: 比较结果
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制任务性能比较图
    plt.figure(figsize=(12, 8))
    
    tasks = []
    finetuned_scores = []
    base_scores = []
    
    for task, metrics in comparison.items():
        # 使用第一个指标作为主要指标
        main_metric = list(metrics.keys())[0]
        
        tasks.append(task)
        finetuned_scores.append(metrics[main_metric]["finetuned"])
        base_scores.append(metrics[main_metric]["base"])
    
    x = np.arange(len(tasks))
    width = 0.35
    
    plt.bar(x - width/2, finetuned_scores, width, label="微调模型")
    plt.bar(x + width/2, base_scores, width, label="基础模型")
    
    plt.xlabel("任务")
    plt.ylabel("性能")
    plt.title("微调模型与基础模型性能比较")
    plt.xticks(x, tasks)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    plt.close()
    
    # 绘制相对变化图
    plt.figure(figsize=(12, 8))
    
    tasks = []
    relative_changes = []
    colors = []
    
    for task, metrics in comparison.items():
        # 使用第一个指标作为主要指标
        main_metric = list(metrics.keys())[0]
        
        tasks.append(task)
        relative_change = metrics[main_metric]["relative_change"]
        relative_changes.append(relative_change)
        
        # 性能提升为绿色，下降为红色
        colors.append("green" if relative_change > 0 else "red")
    
    plt.bar(tasks, relative_changes, color=colors)
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    
    plt.xlabel("任务")
    plt.ylabel("相对变化")
    plt.title("微调模型相对于基础模型的性能变化")
    plt.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "relative_change.png"))
    plt.close()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载微调模型和分词器
    finetuned_model, finetuned_tokenizer = load_model_and_tokenizer(
        args.model_path,
        args.device,
        load_in_4bit=True if args.device == "cuda" else False
    )
    
    # 加载基础模型和分词器（用于对比）
    base_model, base_tokenizer = load_model_and_tokenizer(
        args.base_model_path,
        args.device,
        load_in_4bit=True if args.device == "cuda" else False
    )
    
    # 评估结果
    finetuned_results = {}
    base_results = {}
    
    # 评估微调任务性能
    if os.path.exists(os.path.join(args.eval_data_dir, "instruction_dataset")):
        # 加载微调数据集
        eval_dataset = load_from_disk(os.path.join(args.eval_data_dir, "instruction_dataset"))["validation"]
        
        # 评估微调模型
        finetuned_results["finetuned_task"] = evaluate_finetuned_task(
            finetuned_model,
            finetuned_tokenizer,
            eval_dataset,
            args.batch_size,
            args.device,
            args.max_samples
        )
        
        # 评估基础模型
        base_results["finetuned_task"] = evaluate_finetuned_task(
            base_model,
            base_tokenizer,
            eval_dataset,
            args.batch_size,
            args.device,
            args.max_samples
        )
    
    # 评估原始任务性能（检测灾难性遗忘）
    if args.eval_original_tasks:
        # 评估SQuAD
        squad_path = os.path.join(args.eval_data_dir, "eval", "squad")
        if os.path.exists(squad_path):
            finetuned_results["squad"] = evaluate_squad(
                finetuned_model,
                finetuned_tokenizer,
                squad_path,
                args.batch_size,
                args.device,
                args.max_samples
            )
            
            base_results["squad"] = evaluate_squad(
                base_model,
                base_tokenizer,
                squad_path,
                args.batch_size,
                args.device,
                args.max_samples
            )
        
        # 评估GLUE/SST-2
        sst2_path = os.path.join(args.eval_data_dir, "eval", "glue_sst2")
        if os.path.exists(sst2_path):
            finetuned_results["glue_sst2"] = evaluate_glue(
                finetuned_model,
                finetuned_tokenizer,
                sst2_path,
                "sst2",
                args.batch_size,
                args.device,
                args.max_samples
            )
            
            base_results["glue_sst2"] = evaluate_glue(
                base_model,
                base_tokenizer,
                sst2_path,
                "sst2",
                args.batch_size,
                args.device,
                args.max_samples
            )
    
    # 评估知识保留
    if args.knowledge_eval:
        truthful_qa_path = os.path.join(args.eval_data_dir, "eval", "truthful_qa")
        if os.path.exists(truthful_qa_path):
            finetuned_results["truthful_qa"] = evaluate_truthful_qa(
                finetuned_model,
                finetuned_tokenizer,
                truthful_qa_path,
                args.batch_size,
                args.device,
                args.max_samples
            )
            
            base_results["truthful_qa"] = evaluate_truthful_qa(
                base_model,
                base_tokenizer,
                truthful_qa_path,
                args.batch_size,
                args.device,
                args.max_samples
            )
    
    # 比较模型性能
    comparison = compare_models(finetuned_results, base_results, args.output_dir)
    
    # 过拟合分析
    if args.overfitting_analysis and os.path.exists(os.path.join(args.model_path, "trainer_state.json")):
        # 加载训练状态
        with open(os.path.join(args.model_path, "trainer_state.json"), "r") as f:
            trainer_state = json.load(f)
        
        # 提取训练和验证损失
        train_losses = []
        eval_losses = []
        
        for log in trainer_state["log_history"]:
            if "loss" in log and "eval_loss" not in log:
                train_losses.append(log["loss"])
            if "eval_loss" in log:
                eval_losses.append(log["eval_loss"])
        
        # 分析过拟合
        overfitting_results = analyze_overfitting(train_losses, eval_losses, args.output_dir)
        finetuned_results["overfitting_analysis"] = overfitting_results
    
    # 保存评估结果
    with open(os.path.join(args.output_dir, "finetuned_results.json"), "w") as f:
        # 移除不可序列化的对象
        for task in finetuned_results:
            if "generated_texts" in finetuned_results[task]:
                del finetuned_results[task]["generated_texts"]
        json.dump(finetuned_results, f, indent=2)
    
    with open(os.path.join(args.output_dir, "base_results.json"), "w") as f:
        # 移除不可序列化的对象
        for task in base_results:
            if "generated_texts" in base_results[task]:
                del base_results[task]["generated_texts"]
        json.dump(base_results, f, indent=2)
    
    with open(os.path.join(args.output_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"评估完成，结果已保存到: {args.output_dir}")

if __name__ == "__main__":
    main() 