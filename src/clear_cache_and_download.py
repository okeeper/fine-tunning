#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
清除Hugging Face数据集缓存并重新下载LCCC数据集
"""

import os
import shutil
import argparse
import logging
import json
import sys
import subprocess
from datasets import load_dataset, config
from typing import List, Dict, Any

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="清除Hugging Face数据集缓存并重新下载LCCC数据集")
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="thu-coai/lccc-base",
        help="数据集名称，可选 thu-coai/lccc-base 或 thu-coai/lccc"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="处理后数据集的输出目录"
    )
    parser.add_argument(
        "--clear_all_cache",
        action="store_true",
        help="是否清除所有Hugging Face缓存（包括模型等）"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大样本数量，用于调试"
    )
    parser.add_argument(
        "--use_data_script",
        action="store_true",
        help="使用data/prepare_dataset.py脚本处理数据"
    )
    
    return parser.parse_args()

def get_hf_cache_dirs() -> Dict[str, str]:
    """获取Hugging Face所有缓存目录"""
    # 获取默认缓存目录
    cache_dirs = {
        "datasets_cache": config.HF_DATASETS_CACHE,
        "datasets_modules": os.path.join(config.HF_DATASETS_CACHE, "modules"),
        "datasets_downloads": os.path.join(config.HF_DATASETS_CACHE, "downloads"),
        "hub_cache": os.environ.get("HUGGINGFACE_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")),
        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface/transformers"))
    }
    
    # 添加可能的其他位置
    home = os.path.expanduser("~")
    additional_paths = [
        os.path.join(home, ".cache", "huggingface"),
        os.path.join(home, ".huggingface"),
        "/tmp/huggingface"
    ]
    
    for path in additional_paths:
        if os.path.exists(path) and path not in cache_dirs.values():
            cache_dirs[f"additional_{len(cache_dirs)}"] = path
    
    return cache_dirs

def clear_specific_dataset_cache(dataset_name: str) -> List[str]:
    """清除特定数据集的缓存"""
    # 获取Hugging Face数据集缓存目录
    datasets_cache = config.HF_DATASETS_CACHE
    datasets_modules = os.path.join(datasets_cache, "modules")
    datasets_downloads = os.path.join(datasets_cache, "downloads")
    
    # 构造数据集ID的各种可能形式
    dataset_id_parts = dataset_name.split("/")
    dataset_id = "-".join(dataset_id_parts) if len(dataset_id_parts) > 1 else dataset_name
    dataset_id_alt = "--".join(dataset_id_parts) if len(dataset_id_parts) > 1 else dataset_name
    dataset_id_with_underscore = "_".join(dataset_id_parts) if len(dataset_id_parts) > 1 else dataset_name
    
    # 各种可能的数据集ID
    dataset_ids = [dataset_name, dataset_id, dataset_id_alt, dataset_id_with_underscore]
    
    # 记录删除的目录
    removed_paths = []
    
    # 删除模块缓存
    for root, dirs, files in os.walk(datasets_modules):
        for dir_name in dirs:
            for ds_id in dataset_ids:
                if ds_id in dir_name:
                    dir_path = os.path.join(root, dir_name)
                    logger.info(f"删除模块缓存: {dir_path}")
                    try:
                        shutil.rmtree(dir_path)
                        removed_paths.append(dir_path)
                    except Exception as e:
                        logger.warning(f"删除 {dir_path} 失败: {e}")
    
    # 删除数据集缓存
    for root, dirs, files in os.walk(datasets_cache):
        for dir_name in dirs:
            for ds_id in dataset_ids:
                if ds_id.replace("/", "--") in dir_name or ds_id.replace("/", "-") in dir_name:
                    dir_path = os.path.join(root, dir_name)
                    logger.info(f"删除数据集缓存: {dir_path}")
                    try:
                        shutil.rmtree(dir_path)
                        removed_paths.append(dir_path)
                    except Exception as e:
                        logger.warning(f"删除 {dir_path} 失败: {e}")
    
    # 删除下载缓存
    for root, dirs, files in os.walk(datasets_downloads):
        for file_name in files:
            for ds_id in dataset_ids:
                if ds_id.replace("/", "_") in file_name or ds_id.replace("/", "-") in file_name:
                    file_path = os.path.join(root, file_name)
                    logger.info(f"删除下载缓存: {file_path}")
                    try:
                        os.remove(file_path)
                        removed_paths.append(file_path)
                    except Exception as e:
                        logger.warning(f"删除 {file_path} 失败: {e}")
    
    return removed_paths

def clear_all_hf_cache() -> List[str]:
    """清除所有Hugging Face缓存"""
    cache_dirs = get_hf_cache_dirs()
    removed_paths = []
    
    for name, path in cache_dirs.items():
        if os.path.exists(path):
            logger.info(f"删除{name}缓存: {path}")
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                removed_paths.append(path)
            except Exception as e:
                logger.warning(f"删除 {path} 失败: {e}")
    
    return removed_paths

def download_and_prepare_lccc(dataset_name: str, output_dir: str, max_samples: int = None, use_data_script: bool = False) -> bool:
    """下载并准备LCCC数据集"""
    try:
        if use_data_script:
            # 使用data/prepare_dataset.py脚本
            cmd = [
                "python", "data/prepare_dataset.py",
                "--dataset_name", dataset_name,
                "--output_dir", output_dir,
            ]
            
            if max_samples:
                cmd.extend(["--max_samples", str(max_samples)])
            
            logger.info(f"运行命令: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        else:
            # 直接使用Hugging Face datasets API
            logger.info(f"加载数据集: {dataset_name}")
            dataset = load_dataset(dataset_name, download_mode="force_redownload")
            
            logger.info(f"数据集加载成功: {dataset}")
            
            if max_samples is not None and "train" in dataset:
                logger.info(f"限制样本数量为 {max_samples}")
                dataset["train"] = dataset["train"].select(range(min(max_samples, len(dataset["train"]))))
            
            # 将数据集转换为对话形式
            logger.info("将数据集转换为对话形式")
            
            # 检测字段名
            sample = dataset["train"][0] if "train" in dataset and len(dataset["train"]) > 0 else None
            conversation_field = None
            
            if sample:
                for field in ["conversations", "dialog", "dialogue"]:
                    if field in sample:
                        conversation_field = field
                        break
            
            if not conversation_field:
                logger.warning("无法检测到对话字段，尝试使用'conversations'")
                conversation_field = "conversations"
            
            logger.info(f"使用字段: {conversation_field}")
            
            # 转换数据集
            def convert_format(example):
                conversation = example[conversation_field]
                if len(conversation) < 2:
                    return {"instruction": "", "response": "", "input": ""}
                
                return {
                    "instruction": conversation[0],
                    "response": conversation[1],
                    "input": ""
                }
            
            try:
                dataset = dataset.map(convert_format, remove_columns=[conversation_field])
            except Exception as e:
                logger.error(f"转换数据集失败: {e}")
                logger.warning("尝试继续处理，但可能会出现问题")
            
            # 过滤空对话
            dataset = dataset.filter(lambda x: len(x["instruction"]) > 0 and len(x["response"]) > 0)
            
            # 分割数据集
            if "validation" not in dataset:
                logger.info("从训练集分割验证集")
                split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
                train_dataset = split_dataset["train"]
                val_dataset = split_dataset["test"]
            else:
                train_dataset = dataset["train"]
                val_dataset = dataset["validation"]
            
            # 保存处理后的数据集
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"保存数据集到 {output_dir}")
            train_dataset.save_to_disk(os.path.join(output_dir, "train"))
            val_dataset.save_to_disk(os.path.join(output_dir, "validation"))
            
            logger.info(f"处理完成! 训练集: {len(train_dataset)}个样本, 验证集: {len(val_dataset)}个样本")
        
        return True
    except Exception as e:
        logger.error(f"下载和准备数据集失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """主函数"""
    try:
        args = parse_args()
        
        # 清除缓存
        if args.clear_all_cache:
            logger.info("清除所有Hugging Face缓存...")
            removed = clear_all_hf_cache()
            logger.info(f"共清除 {len(removed)} 个缓存目录/文件")
        else:
            logger.info(f"清除数据集 {args.dataset_name} 的缓存...")
            removed = clear_specific_dataset_cache(args.dataset_name)
            logger.info(f"共清除 {len(removed)} 个缓存目录/文件")
        
        # 重新下载并准备数据集
        logger.info(f"重新下载并准备数据集: {args.dataset_name}...")
        success = download_and_prepare_lccc(
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            use_data_script=args.use_data_script
        )
        
        if success:
            logger.info("=" * 60)
            logger.info(f"数据集 {args.dataset_name} 已成功下载并准备到 {args.output_dir}")
            logger.info("您现在可以运行微调脚本:")
            logger.info(f"python src/finetune.py --data_dir {args.output_dir}")
            logger.info("=" * 60)
        else:
            logger.error("数据准备失败，尝试使用小型本地数据集:")
            logger.error("python src/prepare_small_dataset.py --output_dir ./data/processed")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 