# Llama-2-7b-chat 微调训练与评估项目总结

本项目提供了一个完整的框架，用于对Llama-2-7b-chat模型进行微调训练，并评估其性能、过拟合情况和灾难性遗忘风险。

## 项目结构

```
.
├── README.md                 # 项目说明文档
├── SUMMARY.md                # 项目总结文档
├── requirements.txt          # 项目依赖
├── run_finetune.sh           # 微调训练运行脚本
├── run_evaluate.sh           # 评估运行脚本
├── config/                   # 配置文件目录
│   └── finetune_config.json  # 微调训练配置文件
├── data/                     # 数据目录
│   ├── prepare_dataset.py    # 数据集准备脚本
│   └── eval_dataset.py       # 评估数据集准备脚本
├── src/                      # 源代码
│   ├── config.py             # 配置类定义
│   ├── finetune.py           # 微调训练脚本
│   ├── evaluate.py           # 评估脚本
│   └── utils.py              # 工具函数
└── notebooks/                # Jupyter笔记本
    ├── data_exploration.py   # 数据探索脚本
    └── results_analysis.ipynb # 结果分析笔记本
```

## 主要功能

1. **数据准备**：
   - 支持从Hugging Face下载公开数据集
   - 支持处理本地数据文件
   - 自动分割训练集和验证集
   - 准备评估数据集（用于检测灾难性遗忘）

2. **微调训练**：
   - 使用LoRA（Low-Rank Adaptation）进行参数高效微调
   - 支持4位量化训练，降低显存需求
   - 支持配置文件驱动的训练参数设置
   - 自动记录训练过程和指标

3. **模型评估**：
   - 评估微调任务性能
   - 评估原始任务性能（检测灾难性遗忘）
   - 评估知识保留情况
   - 分析过拟合情况
   - 比较微调模型和基础模型的性能差异

4. **结果分析**：
   - 生成性能对比图表
   - 分析灾难性遗忘情况
   - 分析过拟合情况
   - 提供改进建议

## 使用方法

### 1. 环境设置

```bash
# 创建并激活虚拟环境
python -m venv llama_ft_env
source llama_ft_env/bin/activate  # Linux/Mac
# 或
.\llama_ft_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据集

```bash
# 准备微调数据集和评估数据集
python data/prepare_dataset.py --prepare_eval_datasets
```

### 3. 微调训练

```bash
# 使用默认配置进行微调训练
./run_finetune.sh

# 或使用自定义配置文件
./run_finetune.sh path/to/your/config.json
```

### 4. 评估模型

```bash
# 评估微调后的模型
./run_evaluate.sh path/to/your/model

# 或指定基础模型路径
./run_evaluate.sh path/to/your/model path/to/base/model
```

### 5. 分析结果

```bash
# 运行数据探索脚本
python notebooks/data_exploration.py

# 使用Jupyter笔记本分析结果
jupyter notebook notebooks/results_analysis.ipynb
```

## 防止过拟合和灾难性遗忘的策略

本项目实现了多种策略来防止过拟合和灾难性遗忘：

1. **防止过拟合**：
   - 使用LoRA进行参数高效微调，减少可训练参数数量
   - 实现dropout正则化
   - 支持权重衰减
   - 提供早停策略的实现
   - 分析训练和验证损失曲线，检测过拟合

2. **减轻灾难性遗忘**：
   - 使用LoRA适配器，保留大部分原始权重不变
   - 评估原始任务性能，监控能力保留情况
   - 支持在训练数据中混合原始任务样本
   - 提供知识保留测试

## 注意事项

1. 使用Llama-2-7b-chat模型需要在Hugging Face上申请访问权限
2. 微调训练需要至少16GB显存的GPU（使用4位量化）
3. 评估过程可能需要较长时间，特别是在评估多个任务时
4. 对于中文数据集，可能需要调整分词器和评估方法

## 未来改进方向

1. 支持更多的微调方法，如QLoRA、Prefix Tuning等
2. 增加更多的评估指标和任务
3. 实现多任务学习和持续学习策略
4. 优化训练和评估的性能
5. 添加模型部署和服务功能 