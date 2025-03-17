#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据准备脚本，专门用于处理LCCC中文对话数据集
"""

import os
import argparse
import random
import logging
import json
from typing import Tuple, Optional, Dict, Any
from datasets import load_dataset, Dataset

def load_config_file(config_file: str) -> Dict[str, Any]:
    """
    从配置文件加载参数
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        配置参数字典
    """
    if not os.path.exists(config_file):
        logger.warning(f"配置文件 {config_file} 不存在")
        return {}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"从配置文件 {config_file} 加载参数成功")
        return config
    except Exception as e:
        logger.error(f"加载配置文件 {config_file} 失败: {e}")
        return {}

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="准备LCCC中文对话数据集")
    
    # 添加配置文件参数
    parser.add_argument(
        "--config_file",
        type=str,
        default="../config/finetune_config.json",
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="数据集名称，默认为LCCC-base"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="处理后数据集的输出目录"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=None,
        help="验证集比例"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大样本数量，用于调试"
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=None,
        help="对话最大轮次"
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 从配置文件加载参数
    config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), args.config_file))
    config = load_config_file(config_file)
    
    # 设置默认参数
    default_args = {
        "dataset_name": "thu-coai/lccc",
        "output_dir": "./data/processed",
        "val_size": 0.1,
        "seed": 42,
        "max_samples": None,
        "max_turns": 3
    }
    
    # 从配置文件中提取数据参数
    config_args = {}
    if "data_args" in config:
        data_args = config["data_args"]
        config_args = {
            "dataset_name": data_args.get("dataset_name"),
            "output_dir": data_args.get("output_dir", "./data/processed"),
            "val_size": data_args.get("val_size", 0.1),
            "seed": data_args.get("seed", 42),
            "max_samples": data_args.get("max_train_samples"),
            "max_turns": data_args.get("max_turns", 3)
        }
    
    # 合并参数，优先级：命令行参数 > 配置文件参数 > 默认参数
    final_args = default_args.copy()
    
    # 更新配置文件参数
    for key, value in config_args.items():
        if value is not None:
            final_args[key] = value
            logger.info(f"参数 {key} 从配置文件设置为: {value}")
    
    # 更新命令行参数
    for key, value in vars(args).items():
        if key != "config_file" and value is not None:
            if key in final_args:
                final_args[key] = value
                logger.info(f"参数 {key} 从命令行设置为: {value}")
    
    # 创建最终参数对象
    class Args:
        pass
    
    final_args_obj = Args()
    for key, value in final_args.items():
        setattr(final_args_obj, key, value)
    
    return final_args_obj

def prepare_lccc_dataset(
    dataset_name: str = "thu-coai/lccc",
    output_dir: str = "./data/processed",
    val_size: float = 0.1,
    seed: int = 42,
    max_samples: Optional[int] = None,
    max_turns: int = 3
) -> Tuple[Dataset, Dataset]:
    """
    准备LCCC中文对话数据集
    
    Args:
        dataset_name: 数据集名称，thu-coai/lccc
        output_dir: 输出目录
        val_size: 验证集比例
        seed: 随机种子
        max_samples: 最大样本数量，用于调试
        max_turns: 对话最大轮次
        
    Returns:
        训练集和验证集
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 第一次加载数据集后，保存数据集，目录名根据数据集名称命名
    dataset_path = os.path.join(output_dir, dataset_name.split("/")[-1])
    if not os.path.exists(dataset_path):    
        logger.info(f"加载数据集: {dataset_name}")
        dataset = load_dataset(dataset_name)
        dataset.save_to_disk(dataset_path)
    else:
        logger.info(f"从本地加载数据集: {dataset_path}")
        dataset = load_dataset(dataset_path)
    
    def convert_conversation_to_instruction(example):
        """将对话转换为指令格式"""
        conversation = example["conversations"]
        
        # 过滤掉过长的对话
        if len(conversation) > max_turns * 2:
            conversation = conversation[:max_turns * 2]
        
        # 确保对话是偶数轮次（每轮包含用户和助手）
        if len(conversation) % 2 != 0:
            conversation = conversation[:-1]
        
        # 如果对话为空或只有一轮，则跳过
        if len(conversation) < 2:
            return {
                "instruction": "",
                "response": "",
                "input": ""
            }
        
        # 构建指令和回复
        instruction_parts = []
        response_parts = []
        
        for i in range(0, len(conversation), 2):
            if i == 0:
                # 第一轮对话作为主要指令
                instruction = conversation[i]
                response = conversation[i+1]
            else:
                # 后续轮次作为对话历史
                instruction_parts.append(f"用户: {conversation[i-2]}\n助手: {conversation[i-1]}\n用户: {conversation[i]}")
                response_parts.append(conversation[i+1])
        
        # 合并所有指令部分
        if instruction_parts:
            full_instruction = f"{instruction}\n\n对话历史:\n" + "\n".join(instruction_parts)
        else:
            full_instruction = instruction
            
        # 合并所有回复部分
        if response_parts:
            full_response = f"{response}\n" + "\n".join(response_parts)
        else:
            full_response = response
            
        return {
            "instruction": full_instruction,
            "response": full_response,
            "input": ""  # LCCC没有额外输入
        }
    
    # 转换数据集
    logger.info("转换对话为指令格式")
    dataset = dataset.map(
        convert_conversation_to_instruction, 
        remove_columns=["conversations"],
        num_proc=os.cpu_count()
    )
    
    # 过滤掉空对话
    logger.info("过滤空对话")
    dataset = dataset.filter(lambda x: len(x["instruction"]) > 0 and len(x["response"]) > 0)
    
    # 如果指定了最大样本数，则截取
    if max_samples is not None:
        logger.info(f"限制样本数量为 {max_samples}")
        dataset["train"] = dataset["train"].select(range(min(max_samples, len(dataset["train"]))))
    
    # 分割数据集
    if "validation" not in dataset:
        logger.info(f"从训练集分割验证集，验证集比例: {val_size}")
        split_dataset = dataset["train"].train_test_split(
            test_size=val_size,
            seed=seed
        )
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]
    else:
        logger.info("使用数据集自带的验证集")
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
    
    # 保存处理后的数据集
    logger.info(f"保存处理后的数据集到 {output_dir}")
    train_dataset.save_to_disk(os.path.join(output_dir, "train"))
    val_dataset.save_to_disk(os.path.join(output_dir, "validation"))
    
    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 打印几个样本示例
    logger.info("样本示例:")
    for i in range(min(3, len(train_dataset))):
        logger.info(f"示例 {i+1}:")
        logger.info(f"指令: {train_dataset[i]['instruction']}")
        logger.info(f"回复: {train_dataset[i]['response']}")
        logger.info("---")
    
    return train_dataset, val_dataset

def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 检查是否为LCCC数据集
    if not args.dataset_name.lower().startswith("thu-coai/lccc"):
        logger.error(f"不支持的数据集: {args.dataset_name}")
        raise ValueError(f"不支持的数据集: {args.dataset_name}，请使用LCCC数据集")
    
    logger.info(f"准备处理LCCC数据集: {args.dataset_name}")
    # 准备LCCC数据集
    train_dataset, val_dataset = prepare_lccc_dataset(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        val_size=args.val_size,
        seed=args.seed,
        max_samples=args.max_samples,
        max_turns=args.max_turns
    )
    
    logger.info("数据集准备完成!")

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    
    main() 