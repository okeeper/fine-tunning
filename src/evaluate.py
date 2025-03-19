#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估微调模型的效果，检测灾难性遗忘问题并提供可视化指标
支持使用标准测试集评估灾难性遗忘，使用微调数据测试集评估微调效果
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except Exception:
    print("警告: 未能设置中文字体，可视化图表中的中文可能无法正确显示")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估微调模型效果")
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="基础模型路径"
    )
    parser.add_argument(
        "--finetuned_model_path",
        type=str,
        required=True,
        help="微调后模型路径"
    )
    # 添加支持多种测试集的参数
    parser.add_argument(
        "--finetune_test_file",
        type=str,
        help="微调数据的测试集文件路径，用于评估微调效果"
    )
    parser.add_argument(
        "--standard_test_file",
        type=str,
        help="标准测试集文件路径，用于评估灾难性遗忘"
    )
    parser.add_argument(
        "--prepared_test_dir",
        type=str,
        help="prepare_dataset.py准备的测试数据目录"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./eval_results",
        help="结果保存目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行设备 (cuda/cpu)"
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str, device: str) -> Tuple:
    """加载模型和分词器"""
    print(f"正在加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True,
        use_cache=True
    )
    model.eval()
    return model, tokenizer


def load_test_data(test_file: str) -> List[Dict]:
    """加载测试数据"""
    print(f"正在加载测试数据: {test_file}")
    
    if test_file.endswith('.json'):
        with open(test_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif test_file.endswith('.jsonl'):
        data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    else:
        try:
            # 尝试使用datasets库加载
            dataset = load_dataset(test_file)
            if 'test' in dataset:
                return dataset['test']
            else:
                # 如果没有test分割，使用第一个可用的分割
                split_name = list(dataset.keys())[0]
                return dataset[split_name]
        except Exception as e:
            raise ValueError(f"不支持的数据格式: {test_file}, 错误: {e}")


def load_prepared_test_data(test_dir: str) -> List[Dict]:
    """加载prepare_dataset.py准备的测试数据"""
    print(f"正在加载prepare_dataset准备的测试数据: {test_dir}")
    
    all_data = []
    # 查找测试目录中的所有JSON文件
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.json') or file.endswith('.jsonl'):
                file_path = os.path.join(root, file)
                try:
                    all_data.extend(load_test_data(file_path))
                except Exception as e:
                    print(f"加载文件 {file_path} 出错: {e}")
    
    return all_data


def generate_response(model, tokenizer, prompt: str, device: str) -> str:
    """使用模型生成回复"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 移除输入的prompt部分
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    return response


def calculate_metrics(base_model_outputs: List[str], 
                    finetuned_model_outputs: List[str], 
                    reference_outputs: List[str]) -> Dict:
    """计算评估指标"""
    # 这里使用简单的字符匹配率作为指标
    # 实际应用中可以替换为更复杂的指标如BLEU、ROUGE等
    
    base_match_scores = []
    tuned_match_scores = []
    
    for base_out, tuned_out, ref in zip(base_model_outputs, finetuned_model_outputs, reference_outputs):
        # 计算简单的字符匹配比例
        base_match = len(set(base_out) & set(ref)) / max(len(set(ref)), 1)
        tuned_match = len(set(tuned_out) & set(ref)) / max(len(set(ref)), 1)
        
        base_match_scores.append(base_match)
        tuned_match_scores.append(tuned_match)
    
    # 计算平均分数
    avg_base_score = np.mean(base_match_scores)
    avg_tuned_score = np.mean(tuned_match_scores)
    
    # 计算灾难性遗忘的指标 (基础能力保留率)
    # 如果微调后模型在非目标任务上表现降低太多，可能存在灾难性遗忘
    retention_rate = avg_tuned_score / max(avg_base_score, 1e-10)
    
    return {
        "base_model_score": avg_base_score,
        "finetuned_model_score": avg_tuned_score,
        "improvement": avg_tuned_score - avg_base_score,
        "improvement_percentage": (avg_tuned_score - avg_base_score) / max(avg_base_score, 1e-10) * 100,
        "retention_rate": retention_rate,
        "base_scores": base_match_scores,
        "tuned_scores": tuned_match_scores
    }


def visualize_results(metrics: Dict, result_dir: str, test_type: str = ""):
    """可视化评估结果"""
    # 创建特定类型测试的结果目录
    if test_type:
        result_dir = os.path.join(result_dir, test_type)
    
    os.makedirs(result_dir, exist_ok=True)
    
    # 1. 整体性能对比图
    plt.figure(figsize=(10, 6))
    models = ['基础模型', '微调模型']
    scores = [metrics['base_model_score'], metrics['finetuned_model_score']]
    
    plt.bar(models, scores, color=['blue', 'orange'])
    plt.ylabel('匹配得分')
    title = '模型性能对比'
    if test_type:
        title += f' ({test_type})'
    plt.title(title)
    for i, v in enumerate(scores):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'performance_comparison.png'))
    plt.close()
    
    # 2. 样本级别的性能对比散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics['base_scores'], metrics['tuned_scores'], alpha=0.6)
    
    # 添加对角线
    max_score = max(max(metrics['base_scores']), max(metrics['tuned_scores']))
    plt.plot([0, max_score], [0, max_score], 'r--')
    
    plt.xlabel('基础模型得分')
    plt.ylabel('微调模型得分')
    title = '样本级性能对比'
    if test_type:
        title += f' ({test_type})'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'sample_performance.png'))
    plt.close()
    
    # 3. 性能改进直方图
    improvements = np.array(metrics['tuned_scores']) - np.array(metrics['base_scores'])
    
    plt.figure(figsize=(10, 6))
    plt.hist(improvements, bins=20, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('性能改进')
    plt.ylabel('样本数量')
    title = '微调改进分布'
    if test_type:
        title += f' ({test_type})'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'improvement_distribution.png'))
    plt.close()
    
    # 保存指标数据
    with open(os.path.join(result_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    # 生成简单的评估报告
    report_title = "模型评估报告"
    if test_type:
        report_title += f" ({test_type})"
    
    report = f"""
    # {report_title}

    ## 整体性能
    - 基础模型得分: {metrics['base_model_score']:.4f}
    - 微调模型得分: {metrics['finetuned_model_score']:.4f}
    - 性能提升: {metrics['improvement']:.4f} ({metrics['improvement_percentage']:.2f}%)
    
    ## 灾难性遗忘分析
    - 能力保留率: {metrics['retention_rate']:.4f}
    
    ## 结论
    {'微调模型性能有显著提升' if metrics['improvement'] > 0.05 else '微调效果不明显'}
    {'可能存在灾难性遗忘问题' if metrics['retention_rate'] < 0.9 else '未观察到严重的灾难性遗忘问题'}
    """
    
    with open(os.path.join(result_dir, 'report.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"评估结果已保存至: {result_dir}")


def evaluate_model_on_dataset(base_model, base_tokenizer, finetuned_model, finetuned_tokenizer, 
                             test_data: List[Dict], device: str, result_dir: str, test_type: str):
    """在特定数据集上评估模型"""
    # 收集结果
    base_outputs = []
    finetuned_outputs = []
    reference_outputs = []
    
    # 为当前评估类型创建子目录
    sub_dir = os.path.join(result_dir, test_type)
    os.makedirs(sub_dir, exist_ok=True)
    
    # 对每个测试样本进行评估
    print(f"开始{test_type}评估 ({len(test_data)} 个测试样本)...")
    for i, item in enumerate(test_data):
        # 这里假设测试数据的格式为 {"prompt": "...", "response": "..."}
        # 可以根据实际数据格式进行调整
        prompt = item.get("prompt", item.get("input", item.get("instruction", "")))
        reference = item.get("response", item.get("output", item.get("completion", "")))
        
        if not prompt or not reference:
            print(f"警告: 样本 {i} 缺少prompt或response字段，已跳过")
            continue
            
        print(f"正在处理{test_type}样本 {i+1}/{len(test_data)}...")
        
        # 使用基础模型生成
        base_output = generate_response(base_model, base_tokenizer, prompt, device)
        
        # 使用微调模型生成
        finetuned_output = generate_response(finetuned_model, finetuned_tokenizer, prompt, device)
        
        base_outputs.append(base_output)
        finetuned_outputs.append(finetuned_output)
        reference_outputs.append(reference)
        
        # 保存每个样本的结果
        sample_result = {
            "prompt": prompt,
            "reference": reference,
            "base_model_output": base_output,
            "finetuned_model_output": finetuned_output
        }
        
        with open(os.path.join(sub_dir, f"sample_{i}.json"), "w", encoding="utf-8") as f:
            json.dump(sample_result, f, ensure_ascii=False, indent=2)
    
    # 计算指标
    print(f"计算{test_type}评估指标...")
    metrics = calculate_metrics(base_outputs, finetuned_outputs, reference_outputs)
    
    # 可视化结果
    print(f"生成{test_type}可视化结果...")
    visualize_results(metrics, result_dir, test_type)
    
    return metrics


def generate_summary_report(metrics_dict: Dict[str, Dict[str, Any]], result_dir: str):
    """生成汇总报告，比较不同测试集的结果"""
    print("生成汇总报告...")
    
    # 创建汇总表格数据
    test_types = list(metrics_dict.keys())
    base_scores = [metrics_dict[t]["base_model_score"] for t in test_types]
    tuned_scores = [metrics_dict[t]["finetuned_model_score"] for t in test_types]
    improvements = [metrics_dict[t]["improvement"] for t in test_types]
    retention_rates = [metrics_dict[t]["retention_rate"] for t in test_types]
    
    # 生成汇总图
    plt.figure(figsize=(12, 8))
    bar_width = 0.35
    index = np.arange(len(test_types))
    
    plt.bar(index, base_scores, bar_width, label='基础模型')
    plt.bar(index + bar_width, tuned_scores, bar_width, label='微调模型')
    
    plt.ylabel('评分')
    plt.title('不同测试集上的模型性能对比')
    plt.xticks(index + bar_width / 2, test_types)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'summary_performance.png'))
    plt.close()
    
    # 绘制性能提升对比图
    plt.figure(figsize=(10, 6))
    plt.bar(test_types, improvements, color='green', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.ylabel('性能提升')
    plt.title('不同测试集上的性能提升对比')
    for i, v in enumerate(improvements):
        plt.text(i, v + 0.005, f'{v:.4f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'summary_improvements.png'))
    plt.close()
    
    # 绘制保留率对比图 (用于评估灾难性遗忘)
    plt.figure(figsize=(10, 6))
    plt.bar(test_types, retention_rates, color='purple', alpha=0.7)
    plt.axhline(y=1.0, color='r', linestyle='--')  # 理想情况下保留率为1
    plt.ylabel('能力保留率')
    plt.title('不同测试集上的能力保留率对比 (评估灾难性遗忘)')
    for i, v in enumerate(retention_rates):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'summary_retention_rates.png'))
    plt.close()
    
    # 生成汇总报告
    summary_report = f"""
    # 模型评估汇总报告
    
    ## 性能概览

    | 测试集 | 基础模型分数 | 微调模型分数 | 性能提升 | 能力保留率 |
    |--------|------------|------------|---------|----------|
    """
    
    for t in test_types:
        m = metrics_dict[t]
        summary_report += f"| {t} | {m['base_model_score']:.4f} | {m['finetuned_model_score']:.4f} | {m['improvement']:.4f} ({m['improvement_percentage']:.2f}%) | {m['retention_rate']:.4f} |\n"
    
    summary_report += f"""
    ## 结论分析
    
    ### 微调效果
    """
    
    # 找出最大和最小提升的测试集
    max_improvement_test = test_types[np.argmax(improvements)]
    min_improvement_test = test_types[np.argmin(improvements)]
    
    summary_report += f"""
    - 最大性能提升出现在 {max_improvement_test} 测试集，提升了 {max(improvements):.4f} ({metrics_dict[max_improvement_test]['improvement_percentage']:.2f}%)
    - 最小性能提升出现在 {min_improvement_test} 测试集，提升了 {min(improvements):.4f} ({metrics_dict[min_improvement_test]['improvement_percentage']:.2f}%)
    
    ### 灾难性遗忘分析
    """
    
    # 分析保留率
    min_retention_test = test_types[np.argmin(retention_rates)]
    if min(retention_rates) < 0.9:
        summary_report += f"""
    - 检测到可能的灾难性遗忘问题，特别是在 {min_retention_test} 测试集上，能力保留率仅为 {min(retention_rates):.4f}
    - 建议调整微调参数，或增加相关领域的训练数据，以减轻灾难性遗忘
        """
    else:
        summary_report += f"""
    - 未检测到严重的灾难性遗忘问题，所有测试集的能力保留率都在接受范围内
    - 最低的能力保留率出现在 {min_retention_test} 测试集，为 {min(retention_rates):.4f}
        """
    
    # 提供一些建议
    summary_report += """
    ## 改进建议
    
    1. 如果微调效果不明显，可以考虑：
       - 增加训练数据的数量和质量
       - 调整学习率或训练轮次
       - 检查训练数据与目标任务的相关性
    
    2. 如果存在灾难性遗忘问题，可以考虑：
       - 使用更保守的学习率
       - 应用正则化技术
       - 考虑使用混合训练数据集，包含一部分原始能力的数据
    """
    
    with open(os.path.join(result_dir, 'summary_report.md'), 'w', encoding='utf-8') as f:
        f.write(summary_report)
    
    print(f"汇总报告已保存至: {os.path.join(result_dir, 'summary_report.md')}")


def main():
    """主函数"""
    args = parse_args()
    
    # 确保结果目录存在
    os.makedirs(args.result_dir, exist_ok=True)
    
    # 加载两个模型
    base_model, base_tokenizer = load_model_and_tokenizer(args.base_model_path, args.device)
    finetuned_model, finetuned_tokenizer = load_model_and_tokenizer(args.finetuned_model_path, args.device)
    
    # 用于存储不同测试集的评估结果
    all_metrics = {}
    
    # 1. 如果提供了微调数据测试集，评估微调效果
    if args.finetune_test_file:
        finetune_test_data = load_test_data(args.finetune_test_file)
        finetune_metrics = evaluate_model_on_dataset(
            base_model, base_tokenizer, 
            finetuned_model, finetuned_tokenizer,
            finetune_test_data, args.device, args.result_dir, 
            "微调数据测试"
        )
        all_metrics["微调数据测试"] = finetune_metrics
    
    # 2. 如果提供了标准测试集，评估灾难性遗忘
    if args.standard_test_file:
        standard_test_data = load_test_data(args.standard_test_file)
        standard_metrics = evaluate_model_on_dataset(
            base_model, base_tokenizer, 
            finetuned_model, finetuned_tokenizer,
            standard_test_data, args.device, args.result_dir, 
            "标准测试集"
        )
        all_metrics["标准测试集"] = standard_metrics
    
    # 3. 如果提供了prepare_dataset.py准备的测试目录，使用它们评估
    if args.prepared_test_dir:
        prepared_test_data = load_prepared_test_data(args.prepared_test_dir)
        prepared_metrics = evaluate_model_on_dataset(
            base_model, base_tokenizer, 
            finetuned_model, finetuned_tokenizer,
            prepared_test_data, args.device, args.result_dir, 
            "预处理测试数据"
        )
        all_metrics["预处理测试数据"] = prepared_metrics
    
    # 如果有多个测试集，生成汇总报告
    if len(all_metrics) > 1:
        generate_summary_report(all_metrics, args.result_dir)
    
    print(f"""
评估完成!

评估结果已保存在: {args.result_dir}
请查看该目录下的报告和可视化结果。
    """)


if __name__ == "__main__":
    main()
