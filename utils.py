"""
Utility functions for retrosynthesis evaluation
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_smiles(smiles: str) -> str:
    """
    标准化 SMILES 字符串，用于模糊匹配
    将点号分隔的反应物排序，以忽略反应物顺序差异
    
    Args:
        smiles: SMILES 字符串，可能包含多个反应物（用点号分隔）
    
    Returns:
        标准化后的 SMILES 字符串（反应物按字典序排序）
    
    Example:
        >>> normalize_smiles("CCO.CC(=O)Cl")
        "CC(=O)Cl.CCO"
    """
    if not smiles:
        return ""
    
    # 去除首尾空格
    smiles = smiles.strip()
    
    # 分割点号分隔的 SMILES
    parts = smiles.split('.')
    
    # 去除每个部分的空格并排序
    parts = [part.strip() for part in parts if part.strip()]
    parts.sort()
    
    # 重新组合
    return '.'.join(parts)


def extract_answer_from_result_file(file_path: str) -> Tuple[str, str, bool]:
    """
    从结果文件中提取预测答案、正确答案和原始判断结果
    
    Args:
        file_path: 结果文件路径（如 results/llm_retro_claude/1.txt）
    
    Returns:
        (predicted_answer, correct_answer, original_passed) 元组
    
    Example file format:
        ...
        Final Answer:CC(=O)Cl.c1ccc2c(ccn2C(=O)OC(C)(C)C)c1
        
        Correct Answer:
        CC(=O)c1ccc2[nH]ccc2c1.CC(C)(C)OC(=O)OC(=O)OC(C)(C)C
        
        Passed:
        False
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取 Final Answer
    final_answer_pattern = r'Final Answer:\s*([^\n]+)'
    final_match = re.search(final_answer_pattern, content)
    predicted = final_match.group(1).strip() if final_match else None
    
    # 提取 Correct Answer
    correct_answer_pattern = r'Correct Answer:\s*([^\n]+)'
    correct_match = re.search(correct_answer_pattern, content)
    correct = correct_match.group(1).strip() if correct_match else None
    
    # 提取原始 Passed 结果
    passed_pattern = r'Passed:\s*(True|False)'
    passed_match = re.search(passed_pattern, content)
    original_passed = passed_match.group(1) == "True" if passed_match else False
    
    return predicted, correct, original_passed


def fuzzy_match_smiles(predicted: str, correct: str) -> bool:
    """
    模糊匹配两个 SMILES 字符串，忽略反应物顺序
    
    Args:
        predicted: 预测的 SMILES 字符串
        correct: 正确的 SMILES 字符串
    
    Returns:
        是否匹配（True/False）
    
    Example:
        >>> fuzzy_match_smiles("CCO.CC(=O)Cl", "CC(=O)Cl.CCO")
        True
    """
    if predicted is None or correct is None:
        return False
    
    # 标准化并比较
    norm_pred = normalize_smiles(predicted)
    norm_correct = normalize_smiles(correct)
    
    return norm_pred == norm_correct


def evaluate_retro_results_fuzzy(results_dir: str, verbose: bool = True) -> Dict:
    """
    对逆合成结果文件夹进行模糊匹配评估
    忽略反应物顺序差异，返回真正的准确率
    
    Args:
        results_dir: 结果文件夹路径（如 "results/llm_retro_claude"）
        verbose: 是否打印详细信息
    
    Returns:
        包含评估结果的字典，包括：
        - total: 总样本数
        - strict_correct: 严格匹配正确数
        - fuzzy_correct: 模糊匹配正确数
        - strict_accuracy: 严格匹配准确率
        - fuzzy_accuracy: 模糊匹配准确率
        - improved_count: 模糊匹配额外纠正的样本数
        - details: 每个样本的详细结果列表
    
    Example:
        >>> results = evaluate_retro_results_fuzzy("results/llm_retro_claude")
        >>> print(f"Fuzzy Accuracy: {results['fuzzy_accuracy']:.2%}")
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        raise ValueError(f"结果文件夹不存在: {results_dir}")
    
    # 收集所有 .txt 文件（按数字排序）
    txt_files = list(results_path.glob("*.txt"))
    txt_files.sort(key=lambda x: int(x.stem) if x.stem.isdigit() else 0)
    
    if not txt_files:
        raise ValueError(f"未找到任何 .txt 文件在: {results_dir}")
    
    total = 0
    strict_correct = 0
    fuzzy_correct = 0
    details = []
    
    if verbose:
        print(f"📁 正在评估: {results_dir}")
        print(f"📊 发现 {len(txt_files)} 个结果文件\n")
        print("=" * 100)
    
    for txt_file in txt_files:
        try:
            predicted, correct, original_passed = extract_answer_from_result_file(txt_file)
            
            if predicted is None or correct is None:
                if verbose:
                    print(f"⚠️  样本 {txt_file.stem}: 无法提取答案，跳过")
                continue
            
            total += 1
            
            # 严格匹配（原始评估）
            strict_match = original_passed
            if strict_match:
                strict_correct += 1
            
            # 模糊匹配（忽略顺序）
            fuzzy_match = fuzzy_match_smiles(predicted, correct)
            if fuzzy_match:
                fuzzy_correct += 1
            
            # 记录详细结果
            detail = {
                'sample_id': txt_file.stem,
                'predicted': predicted,
                'correct': correct,
                'strict_match': strict_match,
                'fuzzy_match': fuzzy_match,
                'improved': fuzzy_match and not strict_match  # 模糊匹配纠正的
            }
            details.append(detail)
            
            # 打印详细信息
            if verbose:
                if fuzzy_match:
                    if strict_match:
                        status = "✓✓"  # 严格和模糊都对
                    else:
                        status = "✓*"  # 仅模糊匹配对（顺序不同）
                else:
                    status = "✗✗"  # 都错
                
                print(f"{status} 样本 {txt_file.stem:>3}")
                print(f"  预测: {predicted}")
                print(f"  真实: {correct}")
                
                if detail['improved']:
                    print(f"  💡 模糊匹配纠正（反应物顺序不同）")
                    print(f"  标准化预测: {normalize_smiles(predicted)}")
                    print(f"  标准化真实: {normalize_smiles(correct)}")
                
                print()
        
        except Exception as e:
            if verbose:
                print(f"❌ 处理文件 {txt_file} 时出错: {e}")
            continue
    
    # 计算准确率
    strict_accuracy = strict_correct / total if total > 0 else 0
    fuzzy_accuracy = fuzzy_correct / total if total > 0 else 0
    improved_count = fuzzy_correct - strict_correct
    
    # 汇总结果
    summary = {
        'total': total,
        'strict_correct': strict_correct,
        'fuzzy_correct': fuzzy_correct,
        'strict_accuracy': strict_accuracy,
        'fuzzy_accuracy': fuzzy_accuracy,
        'improved_count': improved_count,
        'details': details
    }
    
    if verbose:
        print("=" * 100)
        print("\n📊 评估结果汇总:\n")
        print(f"  总样本数: {total}")
        print(f"  严格匹配正确数: {strict_correct}")
        print(f"  严格匹配准确率: {strict_accuracy:.2%}")
        print(f"\n  模糊匹配正确数: {fuzzy_correct}")
        print(f"  模糊匹配准确率: {fuzzy_accuracy:.2%}")
        print(f"\n  💡 模糊匹配额外纠正: {improved_count} 个样本")
        print(f"  准确率提升: {(fuzzy_accuracy - strict_accuracy):.2%}")
        
        # 打印正确的题号列表
        strict_correct_ids = [d['sample_id'] for d in details if d['strict_match']]
        fuzzy_correct_ids = [d['sample_id'] for d in details if d['fuzzy_match']]
        improved_ids = [d['sample_id'] for d in details if d['improved']]
        
        print(f"\n📋 正确样本列表:")
        print(f"  严格匹配正确: {strict_correct_ids}")
        print(f"  模糊匹配正确: {fuzzy_correct_ids}")
        if improved_ids:
            print(f"  💡 模糊匹配额外纠正: {improved_ids}")
    
    return summary


def save_evaluation_report(summary: Dict, output_path: str = None):
    """
    保存评估报告到文件
    
    Args:
        summary: evaluate_retro_results_fuzzy() 返回的结果字典
        output_path: 输出文件路径，默认为 "evaluation_report.txt"
    """
    if output_path is None:
        output_path = "evaluation_report.txt"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("逆合成结果模糊匹配评估报告\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("总体统计:\n")
        f.write(f"  总样本数: {summary['total']}\n")
        f.write(f"  严格匹配正确数: {summary['strict_correct']}\n")
        f.write(f"  严格匹配准确率: {summary['strict_accuracy']:.2%}\n")
        f.write(f"  模糊匹配正确数: {summary['fuzzy_correct']}\n")
        f.write(f"  模糊匹配准确率: {summary['fuzzy_accuracy']:.2%}\n")
        f.write(f"  额外纠正样本数: {summary['improved_count']}\n")
        f.write(f"  准确率提升: {(summary['fuzzy_accuracy'] - summary['strict_accuracy']):.2%}\n\n")
        
        # 添加正确样本列表
        strict_correct_ids = [d['sample_id'] for d in summary['details'] if d['strict_match']]
        fuzzy_correct_ids = [d['sample_id'] for d in summary['details'] if d['fuzzy_match']]
        improved_ids = [d['sample_id'] for d in summary['details'] if d['improved']]
        
        f.write("正确样本列表:\n")
        f.write(f"  严格匹配正确 ({len(strict_correct_ids)} 个): {strict_correct_ids}\n")
        f.write(f"  模糊匹配正确 ({len(fuzzy_correct_ids)} 个): {fuzzy_correct_ids}\n")
        if improved_ids:
            f.write(f"  💡 模糊匹配额外纠正 ({len(improved_ids)} 个): {improved_ids}\n")
        f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("详细结果:\n\n")
        
        for detail in summary['details']:
            status = "✓" if detail['fuzzy_match'] else "✗"
            f.write(f"{status} 样本 {detail['sample_id']}\n")
            f.write(f"  预测: {detail['predicted']}\n")
            f.write(f"  真实: {detail['correct']}\n")
            f.write(f"  严格匹配: {detail['strict_match']}\n")
            f.write(f"  模糊匹配: {detail['fuzzy_match']}\n")
            
            if detail['improved']:
                f.write(f"  💡 模糊匹配纠正（反应物顺序不同）\n")
            
            f.write("\n")
    
    print(f"✅ 评估报告已保存到: {output_path}")


# 命令行接口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="评估逆合成结果（支持模糊匹配）")
    parser.add_argument("results_dir", type=str, help="结果文件夹路径，如 results/llm_retro_claude")
    parser.add_argument("--quiet", action="store_true", help="静默模式，不打印详细信息")
    parser.add_argument("--output", type=str, default=None, help="输出报告文件路径")
    
    args = parser.parse_args()
    
    # 执行评估
    summary = evaluate_retro_results_fuzzy(args.results_dir, verbose=not args.quiet)
    
    # 保存报告
    if args.output:
        save_evaluation_report(summary, args.output)
    else:
        # 默认保存到结果文件夹内
        results_path = Path(args.results_dir)
        output_path = results_path / "fuzzy_evaluation_report.txt"
        save_evaluation_report(summary, str(output_path))
