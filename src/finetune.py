#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
微调训练脚本，实现对Llama-2-7b-chat模型的微调
这是一个简化版本，专为初学者设计，便于理解和使用
"""

import os
import sys
import json
import logging
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_from_disk
import wandb

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Llama-2-7b-chat模型微调")
    
    # 配置文件参数
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/finetune_config.json",
        help="配置文件路径"
    )
    
    # 模型参数
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="预训练模型的路径或标识符"
    )
    
    # 数据参数
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/processed",
        help="处理好的数据集目录"
    )
    
    # 训练参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="模型输出目录"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=None,
        help="训练轮数"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="学习率"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=None,
        help="每个设备的训练批次大小"
    )
    
    # LoRA参数
    parser.add_argument(
        "--lora_r",
        type=int,
        default=None,
        help="LoRA适应器的秩"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="LoRA alpha参数"
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 加载配置文件
    if os.path.exists(args.config_file):
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"从配置文件 {args.config_file} 加载参数")
    else:
        logger.warning(f"配置文件 {args.config_file} 不存在，使用默认参数")
        config = {}
    
    # 合并参数，优先级：命令行参数 > 配置文件参数 > 默认参数
    final_args = {}
    
    # 模型参数
    final_args["model_name_or_path"] = args.model_name_or_path or config.get("model_args", {}).get("model_name_or_path", "meta-llama/Llama-2-7b-chat-hf")
    final_args["use_auth_token"] = config.get("model_args", {}).get("use_auth_token", True)
    final_args["torch_dtype"] = config.get("model_args", {}).get("torch_dtype", "float16")
    
    # 数据参数
    final_args["data_dir"] = args.data_dir
    
    # 训练参数
    final_args["output_dir"] = args.output_dir or config.get("training_args", {}).get("output_dir", "./output/llama2-7b-chat-lccc")
    final_args["num_train_epochs"] = args.num_train_epochs or config.get("training_args", {}).get("num_train_epochs", 3.0)
    final_args["learning_rate"] = args.learning_rate or config.get("training_args", {}).get("learning_rate", 2e-5)
    final_args["per_device_train_batch_size"] = args.per_device_train_batch_size or config.get("training_args", {}).get("per_device_train_batch_size", 4)
    final_args["per_device_eval_batch_size"] = config.get("training_args", {}).get("per_device_eval_batch_size", 4)
    final_args["gradient_accumulation_steps"] = config.get("training_args", {}).get("gradient_accumulation_steps", 8)
    final_args["warmup_ratio"] = config.get("training_args", {}).get("warmup_ratio", 0.03)
    final_args["logging_steps"] = config.get("training_args", {}).get("logging_steps", 10)
    final_args["save_steps"] = config.get("training_args", {}).get("save_steps", 100)
    final_args["save_total_limit"] = config.get("training_args", {}).get("save_total_limit", 3)
    
    # LoRA参数
    final_args["lora_r"] = args.lora_r or config.get("lora_args", {}).get("lora_r", 8)
    final_args["lora_alpha"] = args.lora_alpha or config.get("lora_args", {}).get("lora_alpha", 16)
    final_args["lora_dropout"] = config.get("lora_args", {}).get("lora_dropout", 0.05)
    final_args["target_modules"] = config.get("lora_args", {}).get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # 创建参数对象
    class Args:
        pass
    
    args_obj = Args()
    for key, value in final_args.items():
        setattr(args_obj, key, value)
    
    return args_obj

def setup_wandb(args):
    """设置Weights & Biases"""
    wandb.init(
        project="llama2-finetune",
        name=os.path.basename(args.output_dir),
        config={
            "model": args.model_name_or_path,
            "learning_rate": args.learning_rate,
            "epochs": args.num_train_epochs,
            "batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha
        }
    )

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(42)
    
    # 设置Weights & Biases
    setup_wandb(args)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载分词器
    logger.info(f"加载分词器: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_auth_token=args.use_auth_token
    )
    
    # 确保分词器有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 配置量化参数
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # 加载模型
    logger.info(f"加载模型: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        use_auth_token=args.use_auth_token,
        torch_dtype=getattr(torch, args.torch_dtype)
    )
    
    # 准备模型进行量化训练
    model = prepare_model_for_kbit_training(model)
    
    # 配置LoRA
    logger.info("配置LoRA适配器")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=args.target_modules
    )
    
    # 获取PEFT模型
    model = get_peft_model(model, lora_config)
    
    # 打印模型参数统计
    model.print_trainable_parameters()
    
    # 加载处理好的数据集
    logger.info(f"加载处理好的数据集: {args.data_dir}")
    try:
        train_dataset = load_from_disk(os.path.join(args.data_dir, "train"))
        logger.info(f"训练集大小: {len(train_dataset)}")
        
        if os.path.exists(os.path.join(args.data_dir, "validation")):
            eval_dataset = load_from_disk(os.path.join(args.data_dir, "validation"))
            logger.info(f"验证集大小: {len(eval_dataset)}")
        else:
            eval_dataset = None
            logger.warning("未找到验证集")
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        logger.error("请先运行 python data/prepare_dataset.py 准备数据集")
        sys.exit(1)
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="wandb",
        remove_unused_columns=False
    )
    
    # 创建Trainer
    logger.info("创建Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    
    # 保存训练参数
    with open(os.path.join(args.output_dir, "training_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)
    
    # 训练模型
    logger.info("开始训练")
    train_result = trainer.train()
    
    # 保存最终模型
    logger.info("保存最终模型")
    trainer.save_model()
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # 评估模型
    if eval_dataset is not None:
        logger.info("评估模型")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    
    # 关闭wandb
    wandb.finish()
    
    logger.info(f"训练完成！模型保存在: {args.output_dir}")
    logger.info("可以使用以下命令评估模型:")
    logger.info(f"python src/evaluate.py --model_path {args.output_dir}")

if __name__ == "__main__":
    main() 