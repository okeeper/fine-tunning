"""
工具函数，包含数据处理和评估所需的函数
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel
)

def format_instruction(sample: Dict) -> str:
    """
    将样本格式化为指令格式
    
    Args:
        sample: 包含指令和响应的样本
        
    Returns:
        格式化后的指令字符串
    """
    if "instruction" in sample and "response" in sample:
        return f"<s>[INST] {sample['instruction']} [/INST] {sample['response']}</s>"
    elif "input" in sample and "output" in sample:
        return f"<s>[INST] {sample['input']} [/INST] {sample['output']}</s>"
    elif "question" in sample and "answer" in sample:
        return f"<s>[INST] {sample['question']} [/INST] {sample['answer']}</s>"
    elif "prompt" in sample and "completion" in sample:
        return f"<s>[INST] {sample['prompt']} [/INST] {sample['completion']}</s>"
    else:
        raise ValueError(f"未知的样本格式: {sample.keys()}")

def tokenize_function(examples: Dict, tokenizer: PreTrainedTokenizer, max_length: int) -> Dict:
    """
    对样本进行分词处理
    
    Args:
        examples: 样本字典
        tokenizer: 分词器
        max_length: 最大序列长度
        
    Returns:
        分词后的样本
    """
    # 格式化指令
    texts = [format_instruction({"instruction": instruction, "response": response}) 
             for instruction, response in zip(examples["instruction"], examples["response"])]
    
    # 分词
    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    # 设置标签
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    
    return tokenized_inputs

def prepare_dataset(
    dataset_name: Optional[str] = None,
    dataset_config_name: Optional[str] = None,
    train_file: Optional[str] = None,
    validation_file: Optional[str] = None,
    test_file: Optional[str] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    max_length: int = 512,
    streaming: bool = False,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None
) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    """
    准备数据集
    
    Args:
        dataset_name: Hugging Face数据集名称
        dataset_config_name: 数据集配置名称
        train_file: 训练数据文件路径
        validation_file: 验证数据文件路径
        test_file: 测试数据文件路径
        tokenizer: 分词器
        max_length: 最大序列长度
        streaming: 是否使用流式加载
        max_train_samples: 最大训练样本数量
        max_eval_samples: 最大评估样本数量
        
    Returns:
        训练集、验证集和测试集（如果有）
    """
    data_files = {}
    if train_file is not None:
        data_files["train"] = train_file
    if validation_file is not None:
        data_files["validation"] = validation_file
    if test_file is not None:
        data_files["test"] = test_file
        
    # 加载数据集
    if dataset_name is not None:
        # 从Hugging Face加载数据集
        raw_datasets = load_dataset(
            dataset_name,
            dataset_config_name,
            streaming=streaming
        )
    else:
        # 从本地文件加载数据集
        extension = train_file.split(".")[-1] if train_file else "json"
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            streaming=streaming
        )
    
    # 处理训练集
    if "train" in raw_datasets:
        train_dataset = raw_datasets["train"]
        if max_train_samples is not None:
            train_dataset = train_dataset.select(range(max_train_samples))
        if tokenizer is not None:
            train_dataset = train_dataset.map(
                lambda examples: tokenize_function(examples, tokenizer, max_length),
                batched=True,
                remove_columns=train_dataset.column_names
            )
    else:
        train_dataset = None
        
    # 处理验证集
    if "validation" in raw_datasets:
        eval_dataset = raw_datasets["validation"]
        if max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        if tokenizer is not None:
            eval_dataset = eval_dataset.map(
                lambda examples: tokenize_function(examples, tokenizer, max_length),
                batched=True,
                remove_columns=eval_dataset.column_names
            )
    else:
        eval_dataset = None
        
    # 处理测试集
    if "test" in raw_datasets:
        test_dataset = raw_datasets["test"]
        if tokenizer is not None:
            test_dataset = test_dataset.map(
                lambda examples: tokenize_function(examples, tokenizer, max_length),
                batched=True,
                remove_columns=test_dataset.column_names
            )
    else:
        test_dataset = None
        
    return train_dataset, eval_dataset, test_dataset

def compute_metrics(eval_preds: Tuple) -> Dict:
    """
    计算评估指标
    
    Args:
        eval_preds: 评估预测结果
        
    Returns:
        评估指标字典
    """
    preds, labels = eval_preds
    
    # 计算困惑度
    loss = torch.nn.CrossEntropyLoss()(
        torch.tensor(preds).view(-1, preds.shape[-1]),
        torch.tensor(labels).view(-1)
    )
    perplexity = torch.exp(loss).item()
    
    return {"perplexity": perplexity}

def plot_training_curve(
    train_losses: List[float],
    eval_losses: List[float],
    output_dir: str = "./output"
) -> None:
    """
    绘制训练曲线
    
    Args:
        train_losses: 训练损失列表
        eval_losses: 评估损失列表
        output_dir: 输出目录
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(eval_losses, label="Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "training_curve.png"))
    plt.close()
    
def evaluate_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int = 4,
    device: str = "cuda"
) -> float:
    """
    评估模型在数据集上的困惑度
    
    Args:
        model: 预训练模型
        tokenizer: 分词器
        dataset: 数据集
        batch_size: 批次大小
        device: 设备
        
    Returns:
        困惑度
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            inputs = {k: torch.tensor(v).to(device) for k, v in batch.items() if k != "labels"}
            labels = torch.tensor(batch["labels"]).to(device)
            
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item() * len(batch["input_ids"])
            total_samples += len(batch["input_ids"])
    
    avg_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity

def save_training_args(args: Dict, output_dir: str) -> None:
    """
    保存训练参数
    
    Args:
        args: 参数字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_args.json"), "w") as f:
        json.dump(args, f, indent=2)
        
def load_training_args(output_dir: str) -> Dict:
    """
    加载训练参数
    
    Args:
        output_dir: 输出目录
        
    Returns:
        参数字典
    """
    with open(os.path.join(output_dir, "training_args.json"), "r") as f:
        return json.load(f)
        
def analyze_overfitting(
    train_losses: List[float],
    eval_losses: List[float],
    output_dir: str = "./output"
) -> Dict:
    """
    分析过拟合情况
    
    Args:
        train_losses: 训练损失列表
        eval_losses: 评估损失列表
        output_dir: 输出目录
        
    Returns:
        过拟合分析结果
    """
    # 计算训练集和验证集的最终损失差异
    final_train_loss = train_losses[-1]
    final_eval_loss = eval_losses[-1]
    loss_gap = final_train_loss - final_eval_loss
    
    # 计算验证集损失的最小值和最终值之间的差异
    min_eval_loss = min(eval_losses)
    min_eval_loss_idx = eval_losses.index(min_eval_loss)
    eval_loss_increase = final_eval_loss - min_eval_loss
    
    # 绘制过拟合分析图
    plt.figure(figsize=(12, 8))
    
    # 训练和验证损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(eval_losses, label="Validation Loss")
    plt.axvline(x=min_eval_loss_idx, color='r', linestyle='--', label=f"Min Val Loss at Step {min_eval_loss_idx}")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    
    # 训练和验证损失差异
    plt.subplot(2, 1, 2)
    loss_gaps = [t - e for t, e in zip(train_losses, eval_losses)]
    plt.plot(loss_gaps, label="Train-Val Loss Gap")
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel("Steps")
    plt.ylabel("Loss Gap")
    plt.title("Difference Between Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "overfitting_analysis.png"))
    plt.close()
    
    # 判断是否存在过拟合
    is_overfitting = eval_loss_increase > 0.1 and min_eval_loss_idx < len(eval_losses) - 10
    
    return {
        "final_train_loss": final_train_loss,
        "final_eval_loss": final_eval_loss,
        "loss_gap": loss_gap,
        "min_eval_loss": min_eval_loss,
        "min_eval_loss_step": min_eval_loss_idx,
        "eval_loss_increase": eval_loss_increase,
        "is_overfitting": is_overfitting,
        "overfitting_analysis_plot": os.path.join(output_dir, "overfitting_analysis.png")
    } 