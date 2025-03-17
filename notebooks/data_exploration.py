#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据探索脚本，用于分析微调数据集的特点
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, load_from_disk
from collections import Counter
from wordcloud import WordCloud
import jieba

# 设置绘图样式
plt.style.use('ggplot')
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="分析微调数据集")
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/processed",
        help="数据集目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data_analysis",
        help="分析结果输出目录"
    )
    
    return parser.parse_args()

def load_instruction_dataset(data_dir):
    """加载指令微调数据集"""
    dataset_path = os.path.join(data_dir, "instruction_dataset")
    
    if os.path.exists(dataset_path):
        # 从磁盘加载数据集
        dataset = load_from_disk(dataset_path)
    else:
        # 尝试从JSON文件加载
        train_file = os.path.join(data_dir, "train.json")
        val_file = os.path.join(data_dir, "validation.json")
        
        if os.path.exists(train_file) and os.path.exists(val_file):
            with open(train_file, "r", encoding="utf-8") as f:
                train_data = json.load(f)
            
            with open(val_file, "r", encoding="utf-8") as f:
                val_data = json.load(f)
            
            dataset = {
                "train": pd.DataFrame(train_data),
                "validation": pd.DataFrame(val_data)
            }
        else:
            raise FileNotFoundError(f"未找到数据集: {dataset_path}")
    
    return dataset

def analyze_text_length(dataset, output_dir):
    """分析文本长度分布"""
    print("分析文本长度分布...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算指令和响应的长度
    train_inst_lengths = [len(text) for text in dataset["train"]["instruction"]]
    train_resp_lengths = [len(text) for text in dataset["train"]["response"]]
    
    val_inst_lengths = [len(text) for text in dataset["validation"]["instruction"]]
    val_resp_lengths = [len(text) for text in dataset["validation"]["response"]]
    
    # 绘制长度分布图
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    sns.histplot(train_inst_lengths, kde=True)
    plt.title("训练集指令长度分布")
    plt.xlabel("长度")
    plt.ylabel("频率")
    
    plt.subplot(2, 2, 2)
    sns.histplot(train_resp_lengths, kde=True)
    plt.title("训练集响应长度分布")
    plt.xlabel("长度")
    plt.ylabel("频率")
    
    plt.subplot(2, 2, 3)
    sns.histplot(val_inst_lengths, kde=True)
    plt.title("验证集指令长度分布")
    plt.xlabel("长度")
    plt.ylabel("频率")
    
    plt.subplot(2, 2, 4)
    sns.histplot(val_resp_lengths, kde=True)
    plt.title("验证集响应长度分布")
    plt.xlabel("长度")
    plt.ylabel("频率")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "text_length_distribution.png"))
    plt.close()
    
    # 计算统计信息
    stats = {
        "train_instruction": {
            "min": min(train_inst_lengths),
            "max": max(train_inst_lengths),
            "mean": np.mean(train_inst_lengths),
            "median": np.median(train_inst_lengths),
            "std": np.std(train_inst_lengths)
        },
        "train_response": {
            "min": min(train_resp_lengths),
            "max": max(train_resp_lengths),
            "mean": np.mean(train_resp_lengths),
            "median": np.median(train_resp_lengths),
            "std": np.std(train_resp_lengths)
        },
        "val_instruction": {
            "min": min(val_inst_lengths),
            "max": max(val_inst_lengths),
            "mean": np.mean(val_inst_lengths),
            "median": np.median(val_inst_lengths),
            "std": np.std(val_inst_lengths)
        },
        "val_response": {
            "min": min(val_resp_lengths),
            "max": max(val_resp_lengths),
            "mean": np.mean(val_resp_lengths),
            "median": np.median(val_resp_lengths),
            "std": np.std(val_resp_lengths)
        }
    }
    
    # 保存统计信息
    with open(os.path.join(output_dir, "text_length_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    return stats

def analyze_word_frequency(dataset, output_dir, top_n=50):
    """分析词频分布"""
    print("分析词频分布...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 合并所有文本
    all_instructions = " ".join(dataset["train"]["instruction"] + dataset["validation"]["instruction"])
    all_responses = " ".join(dataset["train"]["response"] + dataset["validation"]["response"])
    
    # 分词
    inst_words = jieba.lcut(all_instructions)
    resp_words = jieba.lcut(all_responses)
    
    # 统计词频
    inst_word_counts = Counter(inst_words)
    resp_word_counts = Counter(resp_words)
    
    # 过滤停用词
    stopwords = set(["的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"])
    inst_word_counts = {word: count for word, count in inst_word_counts.items() if word not in stopwords and len(word) > 1}
    resp_word_counts = {word: count for word, count in resp_word_counts.items() if word not in stopwords and len(word) > 1}
    
    # 获取top_n词频
    top_inst_words = dict(sorted(inst_word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n])
    top_resp_words = dict(sorted(resp_word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    # 绘制词频条形图
    plt.figure(figsize=(12, 16))
    
    plt.subplot(2, 1, 1)
    plt.barh(list(reversed(list(top_inst_words.keys()))), list(reversed(list(top_inst_words.values()))))
    plt.title(f"指令中出现频率最高的{top_n}个词")
    plt.xlabel("频率")
    
    plt.subplot(2, 1, 2)
    plt.barh(list(reversed(list(top_resp_words.keys()))), list(reversed(list(top_resp_words.values()))))
    plt.title(f"响应中出现频率最高的{top_n}个词")
    plt.xlabel("频率")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "word_frequency.png"))
    plt.close()
    
    # 生成词云
    inst_wordcloud = WordCloud(width=800, height=400, background_color="white", font_path="SimHei.ttf").generate_from_frequencies(inst_word_counts)
    resp_wordcloud = WordCloud(width=800, height=400, background_color="white", font_path="SimHei.ttf").generate_from_frequencies(resp_word_counts)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(inst_wordcloud, interpolation="bilinear")
    plt.title("指令词云")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(resp_wordcloud, interpolation="bilinear")
    plt.title("响应词云")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "word_cloud.png"))
    plt.close()
    
    return {"top_instruction_words": top_inst_words, "top_response_words": top_resp_words}

def analyze_sample_examples(dataset, output_dir, n_samples=10):
    """分析样本示例"""
    print("分析样本示例...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 随机选择样本
    train_samples = dataset["train"].sample(n_samples)
    val_samples = dataset["validation"].sample(n_samples)
    
    # 保存样本示例
    with open(os.path.join(output_dir, "sample_examples.md"), "w", encoding="utf-8") as f:
        f.write("# 数据集样本示例\n\n")
        
        f.write("## 训练集样本\n\n")
        for i, sample in enumerate(train_samples.itertuples(), 1):
            f.write(f"### 样本 {i}\n\n")
            f.write(f"**指令**：\n\n{sample.instruction}\n\n")
            f.write(f"**响应**：\n\n{sample.response}\n\n")
            f.write("---\n\n")
        
        f.write("## 验证集样本\n\n")
        for i, sample in enumerate(val_samples.itertuples(), 1):
            f.write(f"### 样本 {i}\n\n")
            f.write(f"**指令**：\n\n{sample.instruction}\n\n")
            f.write(f"**响应**：\n\n{sample.response}\n\n")
            f.write("---\n\n")

def main():
    """主函数"""
    args = parse_args()
    
    # 加载数据集
    print(f"加载数据集: {args.data_dir}")
    dataset = load_instruction_dataset(args.data_dir)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 分析文本长度
    length_stats = analyze_text_length(dataset, args.output_dir)
    
    # 分析词频分布
    word_freq = analyze_word_frequency(dataset, args.output_dir)
    
    # 分析样本示例
    analyze_sample_examples(dataset, args.output_dir)
    
    print(f"分析完成，结果已保存到: {args.output_dir}")

if __name__ == "__main__":
    main() 