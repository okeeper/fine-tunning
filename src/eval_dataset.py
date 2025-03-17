#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估数据集准备脚本，用于准备评估灾难性遗忘的数据集
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, Dataset, DatasetDict

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="准备评估灾难性遗忘的数据集")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/eval",
        help="评估数据集的输出目录"
    )
    parser.add_argument(
        "--eval_datasets",
        type=str,
        nargs="+",
        default=["squad", "glue/sst2", "truthful_qa"],
        help="用于评估的数据集列表"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="每个数据集的最大样本数量"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    return parser.parse_args()

def prepare_squad_dataset(
    output_dir: str,
    max_samples: Optional[int] = None,
    seed: int = 42
) -> None:
    """
    准备SQuAD数据集（问答任务）
    
    Args:
        output_dir: 输出目录
        max_samples: 最大样本数量
        seed: 随机种子
    """
    print("准备SQuAD数据集...")
    
    # 加载数据集
    dataset = load_dataset("squad")
    
    # 提取问题和答案
    def extract_qa(example):
        return {
            "question": example["question"],
            "context": example["context"],
            "answer": example["answers"]["text"][0] if example["answers"]["text"] else ""
        }
    
    dataset = dataset.map(extract_qa)
    
    # 限制样本数量
    if max_samples is not None:
        for split in dataset:
            if len(dataset[split]) > max_samples:
                dataset[split] = dataset[split].select(range(max_samples))
    
    # 保存数据集
    dataset_output_dir = os.path.join(output_dir, "squad")
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    dataset.save_to_disk(dataset_output_dir)
    
    # 保存为JSON文件（便于查看）
    for split in dataset:
        with open(os.path.join(dataset_output_dir, f"{split}.json"), "w", encoding="utf-8") as f:
            json.dump(dataset[split].to_dict(), f, ensure_ascii=False, indent=2)
    
    print(f"SQuAD数据集已保存到: {dataset_output_dir}")
    for split in dataset:
        print(f"  {split} 大小: {len(dataset[split])}")

def prepare_glue_dataset(
    config_name: str,
    output_dir: str,
    max_samples: Optional[int] = None,
    seed: int = 42
) -> None:
    """
    准备GLUE数据集（文本分类任务）
    
    Args:
        config_name: GLUE任务名称
        output_dir: 输出目录
        max_samples: 最大样本数量
        seed: 随机种子
    """
    print(f"准备GLUE/{config_name}数据集...")
    
    # 加载数据集
    dataset = load_dataset("glue", config_name)
    
    # 根据不同的GLUE任务处理数据
    if config_name == "mnli":
        def format_mnli(example):
            return {
                "premise": example["premise"],
                "hypothesis": example["hypothesis"],
                "label": example["label"],
                "label_text": ["矛盾", "中立", "蕴含"][example["label"]]
            }
        dataset = dataset.map(format_mnli)
        
    elif config_name == "sst2":
        def format_sst2(example):
            return {
                "sentence": example["sentence"],
                "label": example["label"],
                "label_text": ["负面", "正面"][example["label"]]
            }
        dataset = dataset.map(format_sst2)
        
    elif config_name == "cola":
        def format_cola(example):
            return {
                "sentence": example["sentence"],
                "label": example["label"],
                "label_text": ["不合语法", "合语法"][example["label"]]
            }
        dataset = dataset.map(format_cola)
        
    elif config_name == "rte":
        def format_rte(example):
            return {
                "premise": example["sentence1"],
                "hypothesis": example["sentence2"],
                "label": example["label"],
                "label_text": ["不蕴含", "蕴含"][example["label"]]
            }
        dataset = dataset.map(format_rte)
    
    # 限制样本数量
    if max_samples is not None:
        for split in dataset:
            if len(dataset[split]) > max_samples:
                dataset[split] = dataset[split].select(range(max_samples))
    
    # 保存数据集
    dataset_output_dir = os.path.join(output_dir, f"glue_{config_name}")
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    dataset.save_to_disk(dataset_output_dir)
    
    # 保存为JSON文件（便于查看）
    for split in dataset:
        with open(os.path.join(dataset_output_dir, f"{split}.json"), "w", encoding="utf-8") as f:
            json.dump(dataset[split].to_dict(), f, ensure_ascii=False, indent=2)
    
    print(f"GLUE/{config_name}数据集已保存到: {dataset_output_dir}")
    for split in dataset:
        print(f"  {split} 大小: {len(dataset[split])}")

def prepare_truthful_qa_dataset(
    output_dir: str,
    max_samples: Optional[int] = None,
    seed: int = 42
) -> None:
    """
    准备TruthfulQA数据集（知识评估）
    
    Args:
        output_dir: 输出目录
        max_samples: 最大样本数量
        seed: 随机种子
    """
    print("准备TruthfulQA数据集...")
    
    # 加载数据集
    dataset = load_dataset("truthful_qa", "multiple_choice")
    
    # 提取问题和答案
    def extract_truthful_qa(example):
        return {
            "question": example["question"],
            "correct_answers": example["correct_answers"],
            "incorrect_answers": example["incorrect_answers"],
            "mc1_targets": example["mc1_targets"]
        }
    
    dataset = dataset.map(extract_truthful_qa)
    
    # 限制样本数量
    if max_samples is not None:
        for split in dataset:
            if len(dataset[split]) > max_samples:
                dataset[split] = dataset[split].select(range(max_samples))
    
    # 保存数据集
    dataset_output_dir = os.path.join(output_dir, "truthful_qa")
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    dataset.save_to_disk(dataset_output_dir)
    
    # 保存为JSON文件（便于查看）
    for split in dataset:
        with open(os.path.join(dataset_output_dir, f"{split}.json"), "w", encoding="utf-8") as f:
            json.dump(dataset[split].to_dict(), f, ensure_ascii=False, indent=2)
    
    print(f"TruthfulQA数据集已保存到: {dataset_output_dir}")
    for split in dataset:
        print(f"  {split} 大小: {len(dataset[split])}")

def prepare_mmlu_dataset(
    output_dir: str,
    max_samples: Optional[int] = None,
    seed: int = 42
) -> None:
    """
    准备MMLU数据集（多任务语言理解）
    
    Args:
        output_dir: 输出目录
        max_samples: 最大样本数量
        seed: 随机种子
    """
    print("准备MMLU数据集...")
    
    # 加载数据集
    dataset = load_dataset("cais/mmlu", "all")
    
    # 提取问题和答案
    def format_mmlu(example):
        choices = [example["choices"][i] for i in range(4)]
        return {
            "question": example["question"],
            "choices": choices,
            "answer_index": example["answer"],
            "answer": choices[example["answer"]],
            "subject": example["subject"]
        }
    
    dataset = dataset.map(format_mmlu)
    
    # 限制样本数量
    if max_samples is not None:
        for split in dataset:
            if len(dataset[split]) > max_samples:
                dataset[split] = dataset[split].select(range(max_samples))
    
    # 保存数据集
    dataset_output_dir = os.path.join(output_dir, "mmlu")
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    dataset.save_to_disk(dataset_output_dir)
    
    # 保存为JSON文件（便于查看）
    for split in dataset:
        with open(os.path.join(dataset_output_dir, f"{split}.json"), "w", encoding="utf-8") as f:
            json.dump(dataset[split].to_dict(), f, ensure_ascii=False, indent=2)
    
    print(f"MMLU数据集已保存到: {dataset_output_dir}")
    for split in dataset:
        print(f"  {split} 大小: {len(dataset[split])}")

def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 准备评估数据集
    for dataset_name in args.eval_datasets:
        if dataset_name == "squad":
            prepare_squad_dataset(
                output_dir=args.output_dir,
                max_samples=args.max_samples,
                seed=args.seed
            )
        elif dataset_name.startswith("glue/"):
            config_name = dataset_name.split("/")[1]
            prepare_glue_dataset(
                config_name=config_name,
                output_dir=args.output_dir,
                max_samples=args.max_samples,
                seed=args.seed
            )
        elif dataset_name == "truthful_qa":
            prepare_truthful_qa_dataset(
                output_dir=args.output_dir,
                max_samples=args.max_samples,
                seed=args.seed
            )
        elif dataset_name == "mmlu":
            prepare_mmlu_dataset(
                output_dir=args.output_dir,
                max_samples=args.max_samples,
                seed=args.seed
            )
        else:
            print(f"不支持的数据集: {dataset_name}")

if __name__ == "__main__":
    main() 