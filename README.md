# Llama-2-7b-chat 基于LCCC中文对话数据集的微调

本项目提供了一个简单易用的框架，用于使用LCCC中文对话数据集对Llama-2-7b-chat模型进行微调训练，以提升模型的中文对话能力。项目专为微调初学者设计，流程简单清晰。

## 环境要求

- Python 3.8+
- PyTorch 1.10+ (适配不同CUDA版本)
- Transformers 4.30+
- CUDA 10.2+ (可选，也支持CPU模式)
- 16GB+ 内存
- 磁盘空间: 至少30GB用于模型和数据存储

## 快速开始

### 1. 环境设置

```bash
# 创建并激活虚拟环境
conda create -n llama_ft_env python=3.12
conda activate llama_ft_env

# 安装依赖
pip install -r requirements.txt

# 登录Weights & Biases（用于实验跟踪）
wandb login
```

### 2. 一键微调

只需运行以下命令即可完成数据准备和模型微调：

```bash
# GPU环境（推荐）
bash run_finetune.sh

# CPU环境（训练速度较慢）
bash run_finetune.sh --use_cpu --max_samples 1000
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
├── git_push.sh               # Git快速提交和推送脚本
├── check_nvidia.sh           # 检查NVIDIA驱动兼容性脚本
├── check_model.sh            # 检查模型目录结构脚本
├── fix_and_run.sh            # 综合解决方案脚本
├── fix_model_path.sh         # 修复模型路径脚本
├── fix_glibcxx.sh            # 修复GLIBCXX错误的脚本
├── fix_pytorch.sh            # 修复PyTorch与CUDA兼容性脚本
├── download_model.sh         # 下载LLaMA模型脚本
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

## Git快速提交和推送

项目提供了一个便捷的Git操作脚本`git_push.sh`，用于快速提交和推送代码到远程仓库：

```bash
# 基本用法
./git_push.sh "提交信息"

# 指定分支
./git_push.sh "提交信息" 分支名称
```

该脚本会自动执行以下操作：
1. 添加所有变更文件（`git add .`）
2. 提交变更（`git commit -m "提交信息"`）
3. 推送到远程仓库（`git push origin 分支名称`）

使用此脚本可以大大简化Git操作流程，提高开发效率。

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
# 基本用法（GPU环境）
bash run_finetune.sh

# CPU环境（适用于没有GPU的情况）
bash run_finetune.sh --use_cpu --max_samples 1000

# 使用自定义配置文件
bash run_finetune.sh --config_file config/my_custom_config.json

# 跳过数据准备步骤（如果数据已准备好）
bash run_finetune.sh --skip_data_prep

# 自定义参数
bash run_finetune.sh --learning_rate 1e-5 --num_train_epochs 5

# 查看所有可用选项
bash run_finetune.sh --help
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

### 修复常见错误

项目提供了一个修复GLIBCXX错误的脚本`fix_glibcxx.sh`：

```bash
# 运行修复脚本
./fix_glibcxx.sh
```

该脚本会自动执行以下操作：
1. 备份原始库文件
2. 安装更新的库文件
3. 验证修复是否成功

如果您遇到"GLIBCXX_3.4.32 not found"错误，可以尝试运行此脚本进行修复。

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

### Q: 遇到"CUDA is required but not available for bitsandbytes"错误怎么办？
A: 这表明您的环境中没有可用的CUDA或CUDA驱动版本过旧。您可以：
   1. 使用CPU模式运行：`bash run_finetune.sh --use_cpu --max_samples 1000`
   2. 更新NVIDIA驱动程序：访问 http://www.nvidia.com/Download/index.aspx
   3. 检查CUDA安装：运行 `nvidia-smi` 查看CUDA版本

### Q: 遇到"GLIBCXX_3.4.32 not found"错误怎么办？
A: 这是由于系统库版本不匹配导致的。您可以：
   1. 运行修复脚本：`./fix_glibcxx.sh`
   2. 使用CPU模式运行：`bash run_finetune.sh --use_cpu`
   3. 手动更新系统库：`conda install -c conda-forge libstdcxx-ng`
   4. 或者在Docker容器中运行，以确保环境一致

### Q: 遇到"Error no file named pytorch_model.bin..."错误怎么办？
A: 这表明模型文件路径有问题。您可以：
   1. 运行修复模型路径脚本：`./fix_model_path.sh`
   2. 运行综合解决方案脚本：`./fix_and_run.sh --all`
   3. 检查模型目录结构：`./check_model.sh`
   4. 在运行命令中明确指定模型ID：`bash run_finetune.sh --use_cpu --model_name_or_path meta-llama/Llama-2-7b-chat-hf`
   5. 如果您要使用本地模型，确保添加`--local_model`参数：`bash run_finetune.sh --model_name_or_path /path/to/model --local_model`

### Q: 如何使用本地已下载的LLaMA模型？
A: 要使用本地模型，请按照以下步骤操作：
   1. 确保您的模型目录包含所有必要文件：
      ```bash
      ./check_model.sh --path /opt/llama/Llama-2-7b-chat
      ```
   2. 使用`fix_model_path.sh`脚本更新配置文件：
      ```bash
      ./fix_model_path.sh --model_path /opt/llama/Llama-2-7b-chat
      ```
   3. 或者在运行微调时直接指定：
      ```bash
      bash run_finetune.sh --model_name_or_path /opt/llama/Llama-2-7b-chat --local_model --use_cpu
      ```

   注意：使用本地模型路径时，添加`--local_model`参数非常重要，这将确保脚本正确处理本地路径。
   
### Q: 本地模型目录应该包含哪些文件？
A: 一个有效的LLaMA模型目录应该包含以下文件：
   - `config.json`：模型配置文件
   - 模型权重文件：`pytorch_model.bin`或分片的`pytorch_model-00001-of-00003.bin`等
   - 或者使用SafeTensors格式：`model.safetensors`或分片的`model-00001-of-00003.safetensors`
   - 分词器文件：`tokenizer_config.json`、`tokenizer.model`、`special_tokens_map.json`
   
   您可以使用`check_model.sh`脚本检查您的模型目录是否包含所有必要文件。

### Q: 如何下载LLaMA模型到本地使用？
A: 我们提供了一个下载脚本，可以帮助您获取LLaMA模型：
   ```bash
   # 下载默认的7B聊天模型
   ./download_model.sh
   
   # 下载指定模型到自定义目录
   ./download_model.sh --model_id meta-llama/Llama-2-13b-chat-hf --output_dir /custom/path
   ```
   
   下载完成后，您可以：
   1. 更新配置文件：`./fix_model_path.sh --model_path /path/to/downloaded/model`
   2. 或在运行时指定：`bash run_finetune.sh --model_name_or_path /path/to/downloaded/model --local_model`
   
   注意：下载LLaMA模型需要您有访问权限，请确保已登录Hugging Face账号：`huggingface-cli login`

## 参考资料

- [Llama 2 论文](https://arxiv.org/abs/2307.09288)
- [PEFT 库文档](https://huggingface.co/docs/peft/index)
- [LCCC数据集论文](https://arxiv.org/abs/2008.03946)
- [LCCC数据集](https://github.com/thu-coai/CDial-GPT)

## 综合修复方案

如果您遇到多个问题，可以使用我们的综合解决方案脚本:

```bash
# 运行全部修复步骤
./fix_and_run.sh --all

# 或指定特定步骤
./fix_and_run.sh --check_nvidia --fix_glibcxx --fix_model_path --use_cpu
```

## 针对特定环境的推荐配置

### 低内存环境 (16GB以下)

```bash
bash run_finetune.sh --use_cpu --max_samples 1000 --per_device_train_batch_size 1 --gradient_accumulation_steps 16
```

### CUDA 10.2环境

```bash
# 1. 修复PyTorch安装
./fix_pytorch.sh
# 选择适合CUDA 10.2的安装方法

# 2. 修复模型路径
./fix_model_path.sh --model_path /opt/llama/Llama-2-7b-chat

# 3. 运行微调
bash run_finetune.sh --local_model --max_samples 1000
```

### 无GPU环境

```bash
# 1. 安装CPU版PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. 运行微调
bash run_finetune.sh --use_cpu --max_samples 1000 --per_device_train_batch_size 1 --gradient_accumulation_steps 16
```

## 贡献

欢迎提交问题和贡献代码，帮助改进这个项目。

## 许可证

本项目使用MIT许可证。请注意，LLaMA模型有其自己的许可条款，使用前请查阅。

## 使用指南

### 1. 模型选择

本项目支持多种方式加载LLaMA模型：

#### 使用Hugging Face模型ID (推荐)

推荐使用Hugging Face模型ID，这种方式会自动下载模型文件：

```bash
# 在配置文件中设置
"model_name_or_path": "meta-llama/Llama-2-7b-chat-hf"

# 或者通过命令行指定
bash run_finetune.sh --model_name_or_path meta-llama/Llama-2-7b-chat-hf
```

注意：使用Hugging Face模型ID需要登录Hugging Face账号：

```bash
huggingface-cli login
```

#### 使用本地Hugging Face格式模型

如果您已经下载了HF格式的LLaMA模型，可以直接使用本地路径：

```bash
# 在配置文件中设置
"model_name_or_path": "/path/to/your/llama/model"

# 或者通过命令行指定
bash run_finetune.sh --model_name_or_path /path/to/your/llama/model --local_model
```

注意：使用本地路径时，请确保模型目录结构正确，包含以下文件：
- `config.json`
- 模型权重文件（`pytorch_model.bin`或分片的`pytorch_model-*.bin`文件）
- 分词器文件（`tokenizer_config.json`、`tokenizer.model`等）

您可以使用我们提供的脚本检查模型目录结构：

```bash
./check_model.sh --path /path/to/your/llama/model
```

#### 使用原始Meta格式LLaMA模型

如果您有原始Meta格式的LLaMA模型（包含`consolidated.00.pth`、`params.json`等文件），您需要先将其转换为Hugging Face格式：

```bash
# 转换Meta格式模型为Hugging Face格式
./convert_meta_to_hf.sh --input_dir /path/to/meta/llama --output_dir /path/to/hf/llama --chat_model
```

转换完成后，您可以使用转换后的模型路径：

```bash
# 更新配置文件
./fix_model_path.sh --model_path /path/to/hf/llama

# 运行微调
bash run_finetune.sh --local_model
```

### 2. 数据准备

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

## 常见问题和修复方案

### 模型相关问题

#### 问题: 找不到模型文件
```
Error no file named pytorch_model.bin, model.safetensors, tf_model.h5, model.ckpt.index or flax_model.msgpack found in directory /opt/llama/Llama-2-7b-chat.
```

**解决方案:**
1. 检查模型目录:
   ```bash
   ./check_model.sh --path /opt/llama/Llama-2-7b-chat
   ```
   
2. 修复模型路径:
   ```bash
   ./fix_model_path.sh --model_path /opt/llama/Llama-2-7b-chat
   ```
   
3. 使用Hugging Face模型ID:
   ```bash
   ./fix_model_path.sh --use_hf
   ```
   
4. 下载模型到本地:
   ```bash
   ./download_model.sh --output_dir /opt/llama/Llama-2-7b-chat
   ```

5. 如果您有原始Meta格式的模型（带有consolidated.00.pth文件），请转换为Hugging Face格式:
   ```bash
   ./convert_meta_to_hf.sh --input_dir /opt/llama/Llama-2-7b-chat --output_dir /opt/llama/Llama-2-7b-chat-hf --chat_model
   ```

### Meta格式LLaMA模型问题

#### 问题: 模型是Meta原始格式（包含consolidated.00.pth而非pytorch_model.bin）
```
ls -la /opt/llama/Llama-2-7b-chat
-rw-r--r-- 1 root root         100 Jul 20  2023 checklist.chk
-rw-r--r-- 1 root root 13476925163 Jul 20  2023 consolidated.00.pth
-rw-r--r-- 1 root root        7020 Jul 20  2023 LICENSE.txt
-rw-r--r-- 1 root root         102 Jul 20  2023 params.json
-rw-r--r-- 1 root root       10352 Jul 20  2023 README.md
-rw-r--r-- 1 root root     1253223 Jul 20  2023 Responsible-Use-Guide.pdf
-rw-r--r-- 1 root root          50 Jul 20  2023 tokenizer_checklist.chk
-rw-r--r-- 1 root root      499723 Jul 20  2023 tokenizer.model
-rw-r--r-- 1 root root        4766 Jul 20  2023 USE_POLICY.md
```

**解决方案:**
1. 使用我们的转换脚本将Meta格式转换为Hugging Face格式:
   ```bash
   # 对于聊天模型
   ./convert_meta_to_hf.sh --input_dir /opt/llama/Llama-2-7b-chat --output_dir /opt/llama/Llama-2-7b-chat-hf --chat_model
   
   # 对于基础模型
   ./convert_meta_to_hf.sh --input_dir /opt/llama/Llama-2-7b --output_dir /opt/llama/Llama-2-7b-hf
   ```

2. 使用转换后的模型:
   ```bash
   ./fix_model_path.sh --model_path /opt/llama/Llama-2-7b-chat-hf
   bash run_finetune.sh --local_model
   ```

3. 如果转换过程中遇到内存问题，可以尝试:
   ```bash
   # 使用较小的批处理大小
   bash run_finetune.sh --local_model --per_device_train_batch_size 1 --gradient_accumulation_steps 16
   ```

#### 问题: 转换时遇到"Trying to create tensor with negative dimension"错误

错误信息:
```
RuntimeError: Trying to create tensor with negative dimension -1: [-1, 4096]
```

**解决方案:**
1. 显式指定词表大小:
   ```bash
   # 使用32000作为词表大小(常用值)
   ./convert_meta_to_hf.sh --input_dir /opt/llama/Llama-2-7b-chat --output_dir /opt/llama/Llama-2-7b-chat-hf --chat_model --vocab_size 32000
   ```

2. 使用更高版本的transformers库:
   ```bash
   pip install -U transformers
   ```

3. 如果问题依然存在，尝试使用默认参数:
   ```bash
   # 使用默认模型ID
   ./fix_model_path.sh --use_hf
   bash run_finetune.sh --use_cpu
   ```

#### 问题: 转换过程中出现错误或不完整

**解决方案:**
1. 检查您的Python环境是否安装了所有必要的依赖:
   ```bash
   pip install transformers torch numpy sentencepiece
   ```

2. 确保有足够的磁盘空间和内存:
   ```bash
   # 检查磁盘空间
   df -h
   
   # 检查内存
   free -h
   ```

3. 对于大型模型，可能需要调整转换脚本的参数:
   ```bash
   # 对于13B模型
   ./convert_meta_to_hf.sh --input_dir /opt/llama/Llama-2-13b-chat --output_dir /opt/llama/Llama-2-13b-chat-hf --model_size 13B --chat_model --vocab_size 32000
   ``` 