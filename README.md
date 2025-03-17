# Llama-2-7b-chat 基于LCCC中文对话数据集的微调

本项目提供了一个简单易用的框架，用于使用LCCC中文对话数据集对Llama-2-7b-chat模型进行微调训练，以提升模型的中文对话能力。项目专为微调初学者设计，流程简单清晰。

## 快速开始

### 1. 环境设置

```bash
# 创建并激活虚拟环境
conda create -n llama_ft_env python=3.10
conda activate llama_ft_env

# 安装依赖
pip install -r requirements.txt

# 登录Weights & Biases（用于实验跟踪）
wandb login
```

### 2. 一键微调

只需运行以下命令即可完成数据准备和模型微调：

```bash
bash run_finetune.sh
```

这将使用默认配置（LCCC-base数据集，LoRA微调）进行训练。训练完成后，模型将保存在`./output/llama2-7b-chat-lccc`目录中。

### 3. 测试模型

训练完成后，可以使用以下命令测试模型：

```bash
python src/evaluate.py --model_path ./output/llama2-7b-chat-lccc
```

## 项目结构

```
.
├── README.md                 # 项目说明文档
├── requirements.txt          # 项目依赖
├── run_finetune.sh           # 一键运行脚本
├── config/                   # 配置文件目录
│   └── finetune_config.json  # 微调配置文件
├── data/                     # 数据目录
│   └── prepare_dataset.py    # LCCC数据集准备脚本
└── src/                      # 源代码
    ├── finetune.py           # 微调训练脚本
    ├── evaluate.py           # 评估脚本
    ├── utils.py              # 工具函数
    └── config.py             # 配置类定义
```

## 详细使用说明

### 数据准备

数据准备步骤会自动将LCCC对话数据集转换为适合Llama-2模型的指令格式：

```bash
python data/prepare_dataset.py
```

参数说明：
- `--dataset_name`: 数据集名称，可选 `thu-coai/lccc-base`（小规模）或 `thu-coai/lccc`（大规模）
- `--max_samples`: 最大样本数量，用于调试或限制数据集大小
- `--max_turns`: 对话最大轮次，控制每个对话样本的长度
- `--output_dir`: 处理后数据集的输出目录，默认为 `./data/processed`
- `--val_size`: 验证集比例，默认为 0.1

### 微调训练

微调训练使用LoRA技术，只训练少量参数，大大减少了计算资源需求：

```bash
# 基本用法
bash run_finetune.sh

# 使用自定义配置文件
bash run_finetune.sh --config_file config/my_custom_config.json

# 跳过数据准备步骤（如果数据已准备好）
bash run_finetune.sh --skip_data_prep

# 自定义参数
bash run_finetune.sh --learning_rate 1e-5 --num_train_epochs 5
```

### 配置文件

`config/finetune_config.json`文件包含所有训练参数，您可以根据需要修改：

```json
{
    "model_args": {
        "model_name_or_path": "meta-llama/Llama-2-7b-chat-hf",
        "torch_dtype": "float16"
    },
    "data_args": {
        "dataset_name": "thu-coai/lccc-base",
        "max_train_samples": 10000,
        "max_turns": 3,
        "val_size": 0.1
    },
    "training_args": {
        "output_dir": "./output/llama2-7b-chat-lccc",
        "num_train_epochs": 3.0,
        "learning_rate": 2e-5
    },
    "lora_args": {
        "lora_r": 8,
        "lora_alpha": 16
    }
}
```

## LCCC数据集说明

LCCC（Large-scale Cleaned Chinese Conversation）是一个大规模的中文对话数据集，由清华大学开发：

- **LCCC-base**：小规模版本，约600万对话，1200万个utterance
- **LCCC**：大规模版本，约1200万对话，3600万个utterance

数据集经过严格清洗，移除了不良内容、无意义对话等，质量较高。

在本项目中，对话数据被转换为以下格式：
- **指令**：对话的第一个utterance，加上后续对话历史
- **回复**：模型应生成的回复

## 微调方法说明

本项目使用LoRA（Low-Rank Adaptation）技术进行参数高效微调，这种方法的优势：

1. **计算效率高**：仅训练少量参数（约0.1%的模型参数），可在消费级GPU上运行
2. **降低过拟合风险**：减少可训练参数数量，降低模型过拟合风险
3. **减轻灾难性遗忘**：保留原始模型大部分知识，减轻灾难性遗忘问题

## 常见问题

### Q: 运行时出现CUDA内存不足错误怎么办？
A: 尝试减小批次大小，在`run_finetune.sh`中添加参数：`--per_device_train_batch_size 1 --gradient_accumulation_steps 16`

### Q: 如何使用自己的对话数据集？
A: 目前脚本专为LCCC数据集设计，如需使用自己的数据集，需要修改`data/prepare_dataset.py`中的数据加载和处理逻辑。

### Q: 训练多久才能看到效果？
A: 使用默认配置，在约10000个样本上训练3个epoch，通常可以看到明显的中文对话能力提升。

### Q: 如何在训练后使用模型？
A: 可以使用`src/evaluate.py`脚本进行交互式测试，或者使用Hugging Face的`pipeline`加载模型。

## 参考资料

- [Llama 2 论文](https://arxiv.org/abs/2307.09288)
- [PEFT 库文档](https://huggingface.co/docs/peft/index)
- [LCCC数据集论文](https://arxiv.org/abs/2008.03946)
- [LCCC数据集](https://github.com/thu-coai/CDial-GPT) 