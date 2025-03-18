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
import glob

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="对LLaMA模型进行微调")
    
    # 配置文件参数
    parser.add_argument(
        "--config_file",
        type=str,
        default="../config/finetune_config.json",
        help="配置文件路径"
    )
    
    # 模型参数
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="模型名称或路径（优先级高于配置文件）"
    )
    
    # 数据参数
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/processed",
        help="预处理数据的目录"
    )
    
    # 训练参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（优先级高于配置文件）"
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
        help="学习率（优先级高于配置文件）"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=None,
        help="每个设备的训练批次大小（优先级高于配置文件）"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=None,
        help="每个设备的评估批次大小（优先级高于配置文件）"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="梯度累积步数（优先级高于配置文件）"
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
    
    # 新增：CPU模式参数
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="是否使用CPU进行训练（不使用量化）"
    )
    
    # 新增：PyTorch数据类型参数
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="PyTorch数据类型"
    )
    
    # 新增：Hugging Face认证令牌参数
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="是否使用Hugging Face认证令牌"
    )
    
    # 新增：本地模型参数
    parser.add_argument(
        "--local_model",
        action="store_true",
        help="指定使用的是本地模型路径（不是Hugging Face模型ID）"
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
    final_args["local_model"] = args.local_model or config.get("model_args", {}).get("local_model", False)
    final_args["use_auth_token"] = args.use_auth_token or config.get("model_args", {}).get("use_auth_token", True)
    final_args["torch_dtype"] = args.torch_dtype or config.get("model_args", {}).get("torch_dtype", "float16")
    
    # 数据参数
    final_args["data_dir"] = args.data_dir
    
    # 训练参数
    final_args["output_dir"] = args.output_dir or config.get("training_args", {}).get("output_dir", "./output/llama2-7b-chat-lccc")
    final_args["num_train_epochs"] = args.num_train_epochs or config.get("training_args", {}).get("num_train_epochs", 3.0)
    final_args["learning_rate"] = args.learning_rate or config.get("training_args", {}).get("learning_rate", 2e-5)
    final_args["per_device_train_batch_size"] = args.per_device_train_batch_size or config.get("training_args", {}).get("per_device_train_batch_size", 4)
    final_args["per_device_eval_batch_size"] = args.per_device_eval_batch_size or config.get("training_args", {}).get("per_device_eval_batch_size", 4)
    final_args["gradient_accumulation_steps"] = args.gradient_accumulation_steps or config.get("training_args", {}).get("gradient_accumulation_steps", 8)
    final_args["warmup_ratio"] = config.get("training_args", {}).get("warmup_ratio", 0.03)
    final_args["logging_steps"] = config.get("training_args", {}).get("logging_steps", 10)
    final_args["save_steps"] = config.get("training_args", {}).get("save_steps", 100)
    final_args["save_total_limit"] = config.get("training_args", {}).get("save_total_limit", 3)
    
    # LoRA参数
    final_args["lora_r"] = args.lora_r or config.get("lora_args", {}).get("lora_r", 8)
    final_args["lora_alpha"] = args.lora_alpha or config.get("lora_args", {}).get("lora_alpha", 16)
    final_args["lora_dropout"] = config.get("lora_args", {}).get("lora_dropout", 0.05)
    final_args["target_modules"] = config.get("lora_args", {}).get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # CPU模式参数
    final_args["use_cpu"] = args.use_cpu
    
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
    
    # 检查模型路径是否正确（优先使用Hugging Face模型ID）
    if args.model_name_or_path and args.model_name_or_path.startswith("/opt/llama/"):
        logger.warning(f"检测到本地模型路径: {args.model_name_or_path}")
        
        # 检查用户是否明确指定使用本地路径
        if args.local_model:
            logger.info("用户指定使用本地模型路径")
            
            # 检查本地路径是否存在并包含模型文件
            if os.path.exists(args.model_name_or_path):
                # 检查模型路径下是否有必要的文件
                config_exists = os.path.exists(os.path.join(args.model_name_or_path, "config.json"))
                
                # 检查是否有任何一种模型权重文件存在
                model_files = [
                    "pytorch_model.bin", "model.safetensors", "tf_model.h5", 
                    "model.ckpt.index", "flax_model.msgpack"
                ]
                model_exists = any(os.path.exists(os.path.join(args.model_name_or_path, f)) for f in model_files)
                
                # 检查是否有分片模型文件
                if not model_exists:
                    pytorch_shards = glob.glob(os.path.join(args.model_name_or_path, "pytorch_model-*.bin"))
                    safetensors_shards = glob.glob(os.path.join(args.model_name_or_path, "model-*.safetensors"))
                    model_exists = len(pytorch_shards) > 0 or len(safetensors_shards) > 0
                
                if config_exists and model_exists:
                    logger.info(f"本地模型路径验证成功: {args.model_name_or_path}")
                else:
                    error_messages = []
                    if not config_exists:
                        error_messages.append("缺少config.json文件")
                    if not model_exists:
                        error_messages.append("未找到任何模型权重文件")
                    
                    error_str = ", ".join(error_messages)
                    logger.warning(f"本地模型路径验证失败: {error_str}")
                    logger.warning("建议检查模型文件是否完整，或运行 ./check_model.sh 进行详细检查")
                    
                    if not args.use_cpu:
                        logger.warning("使用本地模型路径可能需要CPU模式，自动切换到CPU模式")
                        args.use_cpu = True
            else:
                logger.error(f"本地模型路径不存在: {args.model_name_or_path}")
                logger.warning("将自动替换为Hugging Face模型ID")
                args.model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
                logger.info(f"已自动替换模型路径为: {args.model_name_or_path}")
        else:
            # 提示但不自动替换
            logger.warning("推荐使用Hugging Face模型ID: meta-llama/Llama-2-7b-chat-hf")
            logger.warning("如果确定要使用本地路径，请添加参数 --local_model")
    
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
    
    # 根据是否使用CPU模式决定加载方式
    if args.use_cpu:
        logger.info(f"使用CPU模式加载模型（不使用量化）: {args.model_name_or_path}")
        try:
            # CPU模式：使用float32或float16，不使用量化
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=getattr(torch, args.torch_dtype),
                use_auth_token=args.use_auth_token,
                trust_remote_code=True,  # 增加此参数以支持自定义模型代码
                device_map="cpu"   # 明确指定使用CPU
            )
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"CPU模式加载模型失败: {e}")
            
            if "No file named" in str(e) and args.model_name_or_path.startswith("/"):
                logger.error("本地模型文件路径有问题，请检查目录结构")
                logger.error("建议运行: ./check_model.sh 检查模型文件")
                logger.error("或使用: ./fix_model_path.sh 自动修复模型路径")
            elif "No file named" in str(e):
                logger.error("Hugging Face模型ID有问题，请检查是否拼写正确")
                logger.error("请确认您已经登录Hugging Face账号: huggingface-cli login")
            
            logger.error("模型加载失败，程序退出")
            sys.exit(1)
    else:
        try:
            logger.info(f"尝试使用4bit量化加载模型: {args.model_name_or_path}")
            # 检查是否存在GPU
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA不可用，无法使用量化加载")
                
            # 使用BitsAndBytes进行4bit量化
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, args.torch_dtype)
            )
            
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                quantization_config=bnb_config,
                use_auth_token=args.use_auth_token,
                trust_remote_code=True,  # 增加此参数以支持自定义模型代码
                device_map="auto"
            )
            logger.info("模型量化加载成功")
        except Exception as e:
            logger.warning(f"量化加载失败: {e}")
            logger.info(f"回退到CPU模式加载模型: {args.model_name_or_path}")
            
            try:
                # 回退到CPU模式
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    torch_dtype=getattr(torch, args.torch_dtype),
                    use_auth_token=args.use_auth_token,
                    trust_remote_code=True,  # 增加此参数以支持自定义模型代码
                    device_map="cpu",
                    # 禁用量化以避免bitsandbytes相关错误
                    load_in_8bit=False,
                    load_in_4bit=False
                )
                args.use_cpu = True
                logger.info("CPU模式加载模型成功")
            except Exception as e2:
                logger.error(f"CPU模式加载也失败: {e2}")
                
                if "No file named" in str(e2) and args.model_name_or_path.startswith("/"):
                    logger.error("本地模型文件路径有问题，请检查目录结构")
                    logger.error("建议运行: ./check_model.sh 检查模型文件")
                    logger.error("或使用: ./fix_model_path.sh 自动修复模型路径")
                elif "No file named" in str(e2):
                    logger.error("Hugging Face模型ID有问题，请检查是否拼写正确")
                    logger.error("请确认您已经登录Hugging Face账号: huggingface-cli login")
                
                logger.error("模型加载失败，程序退出")
                sys.exit(1)
    
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
    
    # 如果使用CPU模式，调整批次大小和梯度累积步数
    if args.use_cpu:
        logger.warning("CPU模式下训练速度会非常慢，建议减小批次大小和数据量")
        # 如果批次大小大于1，则减小到1
        if args.per_device_train_batch_size > 1:
            logger.info(f"CPU模式下将批次大小从 {args.per_device_train_batch_size} 减小到 1")
            args.per_device_train_batch_size = 1
        
        # 增加梯度累积步数
        if args.gradient_accumulation_steps < 16:
            logger.info(f"CPU模式下将梯度累积步数从 {args.gradient_accumulation_steps} 增加到 16")
            args.gradient_accumulation_steps = 16
    
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
        remove_unused_columns=False,
        # 如果使用CPU，禁用fp16
        fp16=not args.use_cpu
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
    try:
        main()
    except Exception as e:
        # 捕获所有未处理的异常并提供有用的建议
        logger.error(f"运行时错误: {e}")
        
        # 根据错误类型提供不同的建议
        error_str = str(e)
        if "CUDA" in error_str or "GPU" in error_str:
            logger.error("GPU相关错误。建议尝试:")
            logger.error("1. 使用CPU模式: bash run_finetune.sh --use_cpu")
            logger.error("2. 检查NVIDIA驱动兼容性: ./check_nvidia.sh")
        elif "GLIBCXX" in error_str:
            logger.error("GLIBCXX库错误。建议尝试:")
            logger.error("1. 修复GLIBCXX: ./fix_glibcxx.sh")
            logger.error("2. 使用综合解决方案: ./fix_and_run.sh --fix_glibcxx")
        elif "No file named" in error_str:
            logger.error("模型文件错误。建议尝试:")
            logger.error("1. 检查模型目录: ./check_model.sh")
            logger.error("2. 修复模型路径: ./fix_model_path.sh")
            logger.error("3. 使用Hugging Face模型ID: ./fix_model_path.sh --use_hf")
        else:
            logger.error("未知错误。建议尝试:")
            logger.error("1. 使用综合解决方案: ./fix_and_run.sh --all")
            logger.error("2. 使用CPU模式并减少样本数: bash run_finetune.sh --use_cpu --max_samples 1000")
        
        # 打印完整的堆栈跟踪用于调试
        import traceback
        logger.debug("完整错误堆栈:")
        logger.debug(traceback.format_exc())
        
        sys.exit(1) 