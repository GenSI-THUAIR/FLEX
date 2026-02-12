import argparse
import asyncio
import time
import re
import json
from datetime import datetime
from actor import Actor
from memory_retriever import create_memory_retriever, create_memory_retrieval_tool
from smolagents import OpenAIServerModel
import os

# SmolagentsInstrumentor 相关导入将在需要时动态导入

def save_problem_details(problem_idx, query, response, ground_truth, predicted_answer, is_correct, results_dir):
    """保存单个问题的详细信息到文件"""
    try:
        # 创建结果目录
        os.makedirs(results_dir, exist_ok=True)
        
        # 文件路径
        file_path = os.path.join(results_dir, f"{problem_idx}.txt")
        
        # 准备文件内容，确保所有值都是字符串
        content = f"""Problem {problem_idx}
{'='*50}

Query:
{query}

{'='*50}

Agent Response:
{response}

{'='*50}

Ground Truth Answer: {str(ground_truth) if ground_truth is not None else 'None'}
Predicted Answer: {str(predicted_answer) if predicted_answer is not None else 'None'}
Correct: {'✅ Yes' if is_correct else '❌ No'}

{'='*50}
Saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        if args.timer:
            print(f"💾 问题 {problem_idx} 详细信息已保存到: {file_path}")
            
    except Exception as e:
        print(f"⚠️ 保存问题 {problem_idx} 详细信息时出错: {e}")

parser = argparse.ArgumentParser()
parser.add_argument('--actor', default='google/gemini-2.5-pro')
parser.add_argument('--task_type', choices=['math','retro'], default='math', help='Task type: math or retro (single-step retrosynthesis)')
parser.add_argument('--memory_path', default=None, help='Path to memory file or directory (defaults vary by task_type)')
parser.add_argument('--data_path', default=None, help='Path to dataset directory (defaults vary by task_type)')
parser.add_argument('--split', default=None, help='test split name without .jsonl (defaults vary by task_type)')
parser.add_argument('--samples', type=int, default=None, help='Number of samples to process. If not specified, use all samples.')
parser.add_argument('--batch_size', type=int, default=10, help='The number of samples processed concurrently as a batch.')
parser.add_argument('--pass_at_n', type=int, default=1, help='Pass@N: number of attempts per problem (default: 1)')
parser.add_argument('--max_concurrent', type=int, default=10, help='Maximum concurrent requests (default: 5)')
parser.add_argument('--no-timer', dest='timer', action='store_false', default=True, help='Disable detailed timing (default: enabled)')
parser.add_argument('--no-retrieve', dest='retrieve', action='store_false', default=True, help='Disable memory retrieval (default: enabled)')
parser.add_argument('--no-telemetry', dest='telemetry', action='store_false', default=True, help='Disable SmolagentsInstrumentor telemetry (default: enabled)')
parser.add_argument('--results_dir', default=None, help='Directory to save problem details (defaults vary by task_type)')

args = parser.parse_args()

# Derive defaults based on task_type when not explicitly provided
if args.task_type == 'retro':
    if args.memory_path is None:
        args.memory_path = './exps/retro_50.json'
    if args.data_path is None:
        args.data_path = './data/uspto50k/'
    if args.split is None:
        args.split = 'test'
    if args.results_dir is None:
        args.results_dir = 'results/agent_mem_retro_gemini'
else:
    if args.memory_path is None:
        args.memory_path = './exps/aime25_gemini.json'
    if args.data_path is None:
        args.data_path = './data/AIME/'
    if args.split is None:
        args.split = 'aime25'
    if args.results_dir is None:
        args.results_dir = 'results/agent_mem_aime_gemini_tts'

def load_data(file_path):
    """加载JSONL测试数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_answer(response_text):
    """从模型响应中提取答案"""
    # 英文答案格式 (优先级较高)
    english_patterns = [
        r"Answer:\s*(\d+)",              # Answer: 123
        r"The answer is\s*(\d+)",        # The answer is 123
        r"Final answer:\s*(\d+)",        # Final answer: 123
        r"Therefore,?\s*the answer is\s*(\d+)",  # Therefore, the answer is 123
        r"So,?\s*the answer is\s*(\d+)", # So, the answer is 123
        r"Thus,?\s*the answer is\s*(\d+)", # Thus, the answer is 123
        r"Hence,?\s*the answer is\s*(\d+)", # Hence, the answer is 123
        r"We get\s*(\d+)",               # We get 123
        r"We have\s*(\d+)",              # We have 123
        r"This gives us\s*(\d+)",        # This gives us 123
        r"The result is\s*(\d+)",        # The result is 123
        r"The solution is\s*(\d+)",      # The solution is 123
        r"(?:Therefore|Thus|Hence|So),?\s*(\d+)\.?$",  # Therefore, 123.
        r"(?:^|\n)\s*Answer\s*=\s*(\d+)", # Answer = 123
        r"(?:^|\n)\s*=\s*(\d+)\s*$",     # = 123
    ]
    
    # 中文答案格式 (保留原有的)
    chinese_patterns = [
        r"答案是?\s*(\d+)",               # 答案是 123 或 答案 123
        r"最终答案是?\s*(\d+)",           # 最终答案是 123
        r"因此答案是?\s*(\d+)",           # 因此答案是 123
        r"所以答案是?\s*(\d+)",           # 所以答案是 123
        r"答案为\s*(\d+)",               # 答案为 123
        r"结果是\s*(\d+)",               # 结果是 123
        r"得到\s*(\d+)",                 # 得到 123
        r"为\s*(\d+)",                   # 为 123
    ]
    
    # 通用数字格式
    general_patterns = [
        r"(?:^|\n)\s*(\d+)\s*\.?\s*(?:\n|$)",  # 单独一行的数字，可能有句号
        r"\$(\d+)\$",                    # $123$ (LaTeX格式)
        r"\\boxed\{(\d+)\}",            # \boxed{123} (LaTeX答案框)
    ]
    
    # 按优先级顺序搜索
    all_patterns = english_patterns + chinese_patterns + general_patterns
    
    for pattern in all_patterns:
        matches = re.findall(pattern, response_text, re.MULTILINE | re.IGNORECASE)
        if matches:
            return matches[-1]  # 返回最后一个匹配的答案
    
    # 如果没有找到明确的答案格式，尝试提取最后出现的数字
    # 但要排除一些明显不是答案的数字（如年份、页码等）
    all_numbers = re.findall(r'\b\d+\b', response_text)
    if all_numbers:
        # 过滤掉可能的年份（1900-2100）、页码等
        filtered_numbers = []
        for num in all_numbers:
            num_int = int(num)
            # 排除明显不是答案的数字
            if not (1900 <= num_int <= 2100 or num_int in [1, 2, 3] and len(all_numbers) > 5):
                filtered_numbers.append(num)
        
        if filtered_numbers:
            return filtered_numbers[-1]
        else:
            return all_numbers[-1]  # 如果过滤后没有数字，返回原来的最后一个
    
    return None

# 为pass@n功能导入AsyncOpenAI
try:
    from openai import AsyncOpenAI
    async_client = AsyncOpenAI(
        base_url=os.getenv("BASE_URL", ""),
        api_key=os.getenv("API_KEY", "")
    )
except ImportError:
    print("Warning: AsyncOpenAI not available, pass@n mode will be limited")
    async_client = None


class Timer:
    def __init__(self, name, enabled=True):
        self.name = name
        self.enabled = enabled
        self.start_time = None
        
    def __enter__(self):
        if self.enabled:
            self.start_time = time.time()
            print(f"🕐 [{datetime.now().strftime('%H:%M:%S')}] 开始: {self.name}")
        return self
        
    def __exit__(self, *args):
        if self.enabled and self.start_time is not None:
            end_time = time.time()
            duration = end_time - self.start_time
            print(f"✅ [{datetime.now().strftime('%H:%M:%S')}] 完成: {self.name} - 耗时: {duration:.2f}秒")

def extract_answer_from_response(response_text):
    """从模型响应中提取答案 (复用test_llm.py的逻辑)"""
    # 英文答案格式 (优先级较高)
    english_patterns = [
        r"Answer:\s*(\d+)",              # Answer: 123
        r"The answer is\s*(\d+)",        # The answer is 123
        r"Final answer:\s*(\d+)",        # Final answer: 123
        r"Therefore,?\s*the answer is\s*(\d+)",  # Therefore, the answer is 123
        r"So,?\s*the answer is\s*(\d+)", # So, the answer is 123
        r"Thus,?\s*the answer is\s*(\d+)", # Thus, the answer is 123
        r"Hence,?\s*the answer is\s*(\d+)", # Hence, the answer is 123
        r"We get\s*(\d+)",               # We get 123
        r"We have\s*(\d+)",              # We have 123
        r"This gives us\s*(\d+)",        # This gives us 123
        r"The result is\s*(\d+)",        # The result is 123
        r"The solution is\s*(\d+)",      # The solution is 123
        r"(?:Therefore|Thus|Hence|So),?\s*(\d+)\.?$",  # Therefore, 123.
        r"(?:^|\n)\s*Answer\s*=\s*(\d+)", # Answer = 123
        r"(?:^|\n)\s*=\s*(\d+)\s*$",     # = 123
    ]
    
    # 中文答案格式
    chinese_patterns = [
        r"答案是?\s*(\d+)",               # 答案是 123 或 答案 123
        r"最终答案是?\s*(\d+)",           # 最终答案是 123
        r"因此答案是?\s*(\d+)",           # 因此答案是 123
        r"所以答案是?\s*(\d+)",           # 所以答案是 123
        r"答案为\s*(\d+)",               # 答案为 123
        r"结果是\s*(\d+)",               # 结果是 123
    ]
    
    # 通用数字格式
    general_patterns = [
        r"(?:^|\n)\s*(\d+)\s*\.?\s*(?:\n|$)",  # 单独一行的数字
        r"\$(\d+)\$",                    # $123$ (LaTeX格式)
        r"\\boxed\{(\d+)\}",            # \boxed{123} (LaTeX答案框)
    ]
    
    # 按优先级顺序搜索
    all_patterns = english_patterns + chinese_patterns + general_patterns
    
    for pattern in all_patterns:
        matches = re.findall(pattern, response_text, re.MULTILINE | re.IGNORECASE)
        if matches:
            return matches[-1]  # 返回最后一个匹配的答案
    
    # 如果没有找到明确的答案格式，尝试提取最后出现的数字
    all_numbers = re.findall(r'\b\d+\b', response_text)
    if all_numbers:
        return all_numbers[-1]
    
    return None

async def test_single_sample(sample, memory_retrieval_tool, sample_idx=None, actor_model=None, retrieve=True, results_dir="results"):
    """测试单个样本，为每个样本创建独立的Actor"""
    sample_name = f"样本 {sample_idx+1}" if sample_idx is not None else "单个样本"
    
    with Timer(sample_name, args.timer):
        query = sample['question']
        gt = sample['answer']

        # 为每个样本创建独立的actor
        log_path = os.path.join(args.results_dir + "_process", str(sample_idx) + ".txt")
        actor = Actor(actor_model, memory_retrieval_tool, retrieve=retrieve, log_path=log_path, setting=args.task_type)
        
        # 使用配置好的actor解题
        with Timer(f"{sample_name} - Actor", args.timer):
            response = await actor.act(query)
            
            # 提取答案
            if args.task_type == "math":
                predicted_answer = extract_answer_from_response(str(response))
            else:
                predicted_answer = response
            
        if args.timer and sample_idx is not None:
            print(f'样本{sample_idx+1} 预测答案: {predicted_answer} 真实答案: {gt}')
        else:
            print('预测答案:', predicted_answer, '真实答案:', gt)

    # 判断正确性 - 根据任务类型选择匹配方式
    is_correct = False
    if predicted_answer is not None and gt is not None:
        if args.task_type == "math":
            # 数学题使用精确匹配
            is_correct = str(predicted_answer) == str(gt)
        else:
            # 逆合成使用模糊匹配（忽略反应物顺序）
            from utils import fuzzy_match_smiles
            is_correct = fuzzy_match_smiles(str(predicted_answer), str(gt))
    
    # 保存问题详细信息到文件（只在有sample_idx时保存）
    if sample_idx is not None:
        save_problem_details(
            problem_idx=sample_idx + 1,  # 题号从1开始
            query=query,
            response=response,
            ground_truth=gt,
            predicted_answer=predicted_answer,
            is_correct=is_correct,
            results_dir=results_dir
        )
    
    return is_correct, predicted_answer, response

async def test_single_sample_passat(sample, memory_retrieval_tool, sample_idx, attempt_num=1, actor_model=None, retrieve=True):
    """Pass@N模式的单次尝试，为每次尝试创建独立的Actor"""
    query = sample['question']
    
    try:
        start_time = time.time()
        
        # 为每次尝试创建独立的actor
        log_path = os.path.join(args.results_dir + "_process", str(sample_idx) + ".txt")
        actor = Actor(actor_model, memory_retrieval_tool, retrieve=retrieve, log_path=log_path)
        
        # 使用配置好的actor解题
        response = await actor.act(query)
        end_time = time.time()
        
        # 提取答案
        predicted_answer = extract_answer_from_response(response)
        
        return {
            'sample_idx': sample_idx,
            'attempt_num': attempt_num,
            'predicted_answer': predicted_answer,
            'solve_time': end_time - start_time,
            'response': response,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'sample_idx': sample_idx,
            'attempt_num': attempt_num,
            'predicted_answer': None,
            'solve_time': 0,
            'response': str(e),
            'success': False,
            'error': str(e)
        }

async def run_passat_test(data, memory_retrieval_tool, pass_at_n, max_concurrent, actor_model, retrieve):
    """运行Pass@N测试"""
    print(f"🔄 Pass@{pass_at_n} 模式启动")
    print(f"⚡ 最大并发数: {max_concurrent}")
    print(f"📊 总共需要 {len(data) * pass_at_n} 次API调用")
    
    # 初始化pass@n结果列表
    pass_results = [0] * len(data)  # 0表示未通过，1表示通过
    detailed_results = []
    
    # 创建所有任务
    all_tasks = []
    for i, sample in enumerate(data):
        for attempt in range(pass_at_n):
            task = test_single_sample_passat(sample, memory_retrieval_tool, i, attempt + 1, actor_model, retrieve)
            all_tasks.append((task, i, sample))
    
    print(f"🚀 开始并发执行 {len(all_tasks)} 个任务...")
    
    # 使用信号量控制并发数
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_task(task, sample_idx, sample):
        async with semaphore:
            return await task, sample_idx, sample
    
    # 分批执行以避免过多并发
    batch_size = max_concurrent * 2
    all_results = []
    
    start_time = time.time()
    
    for i in range(0, len(all_tasks), batch_size):
        batch = all_tasks[i:i + batch_size]
        if args.timer:
            print(f"🔄 执行批次 {i//batch_size + 1}/{(len(all_tasks) + batch_size - 1)//batch_size} "
                  f"({len(batch)} 个任务)")
        
        # 执行当前批次
        batch_tasks = [limited_task(task, idx, sample) for task, idx, sample in batch]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        all_results.extend(batch_results)
        
        # 批次间短暂休息
        if i + batch_size < len(all_tasks):
            await asyncio.sleep(1)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"✅ 所有任务完成，总耗时: {total_time:.2f}秒")
    
    # 处理结果
    print("📊 处理结果...")
    
    # 用于跟踪每个问题是否已保存详细信息
    problem_saved = set()
    
    for result_data in all_results:
        if isinstance(result_data, Exception):
            print(f"❌ 任务执行异常: {result_data}")
            continue
            
        result, sample_idx, sample = result_data
        correct_answer = extract_answer(sample['answer'])
        
        # 检查答案是否正确 - 根据任务类型选择匹配方式
        is_correct = False
        if result['success'] and result['predicted_answer'] is not None:
            if args.task_type == "math":
                # 数学题使用精确匹配
                is_correct = str(result['predicted_answer']) == str(correct_answer)
            else:
                # 逆合成使用模糊匹配（忽略反应物顺序）
                from utils import fuzzy_match_smiles
                is_correct = fuzzy_match_smiles(str(result['predicted_answer']), str(correct_answer))
            
            # 如果答对了，更新pass@n结果
            if is_correct:
                pass_results[sample_idx] = 1
        
        # 保存问题详细信息（Pass@N模式：只保存第一次成功的尝试，或最后一次尝试）
        problem_id = sample_idx + 1
        should_save = False
        
        if problem_id not in problem_saved:
            if is_correct:
                # 如果答对了，立即保存
                should_save = True
                problem_saved.add(problem_id)
            else:
                # 如果答错了，检查这是否是该问题的最后一次尝试
                # 统计这个问题总共有多少次尝试
                total_attempts_for_this_problem = pass_at_n
                if result['attempt_num'] == total_attempts_for_this_problem:
                    should_save = True
                    problem_saved.add(problem_id)
        
        if should_save:
            save_problem_details(
                problem_idx=problem_id,
                query=sample['question'],
                response=result['response'],
                ground_truth=correct_answer,
                predicted_answer=result['predicted_answer'],
                is_correct=is_correct,
                results_dir=args.results_dir
            )
        
        # 记录详细结果
        detailed_results.append({
            'problem_id': sample_idx + 1,
            'attempt_num': result['attempt_num'],
            'question': sample['question'],
            'correct_answer': correct_answer,
            'predicted_answer': result['predicted_answer'],
            'is_correct': is_correct,
            'solve_time': result['solve_time'],
            'full_response': result['response'],
            'success': result['success'],
            'error': result['error']
        })
    
    # 计算Pass@N统计
    pass_count = sum(pass_results)
    pass_rate = pass_count / len(data) * 100
    
    # 收集通过的题目题号
    passed_problem_ids = [i + 1 for i, passed in enumerate(pass_results) if passed]
    
    # 显示统计结果
    print(f"\n📊 Pass@{pass_at_n} 统计:")
    print(f"   总题目数: {len(data)}")
    print(f"   通过题目: {pass_count}")
    print(f"   Pass@{pass_at_n} 率: {pass_rate:.2f}%")
    print(f"   总API调用: {len(all_tasks)}")
    print(f"   总用时: {total_time:.2f}秒")
    print(f"   平均每题用时: {total_time / len(data):.2f}秒")
    print(f"\n✅ 通过的题目题号 ({len(passed_problem_ids)}题):")
    print(f"   {passed_problem_ids}")
    
    # 分析每题的尝试情况
    if args.timer:
        print(f"\n📋 每题尝试详情:")
        for i, (sample, passed) in enumerate(zip(data, pass_results)):
            attempts_for_this_problem = [r for r in detailed_results 
                                       if r['problem_id'] == i + 1]
            correct_attempts = [r for r in attempts_for_this_problem if r['is_correct']]
            
            status = "✅" if passed else "❌"
            print(f"   题目 {i + 1}: {status} "
                  f"({len(correct_attempts)}/{pass_at_n} 正确)")
    
    return pass_rate, detailed_results, pass_results, total_time

async def test_batch(batch, memory_retrieval_tool, batch_idx=None, batch_size=None, actor_model=None, retrieve=True, results_dir="results"):
    """并发测试一批样本，为每个样本创建独立的Actor"""
    batch_name = f"Batch {batch_idx+1}" if batch_idx is not None else "批次测试"
    
    with Timer(batch_name, args.timer):
        tasks = []
        for i, sample in enumerate(batch):
            if batch_idx is not None and batch_size is not None:
                sample_global_idx = batch_idx * batch_size + i
                task = test_single_sample(sample, memory_retrieval_tool, sample_global_idx, actor_model, retrieve, results_dir)
            else:
                task = test_single_sample(sample, memory_retrieval_tool, None, actor_model, retrieve, results_dir)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        correct_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Sample {i} failed: {result}")
            else:
                is_correct, predicted_answer, response = result
                if is_correct:
                    correct_count += 1
        
        return correct_count, results

async def main():
    # 根据命令行参数决定是否设置 SmolagentsInstrumentor 来监视 agent 执行过程
    if args.telemetry:
        try:
            # 动态导入 telemetry 相关模块
            from phoenix.otel import register
            from openinference.instrumentation.smolagents import SmolagentsInstrumentor
            
            register(
                project_name="MemAgent (test: test_2.jsonl)",
            )  # 注册 Phoenix OTEL
            SmolagentsInstrumentor().instrument()  # 启用 Smolagents instrumentation
            print("🔍 SmolagentsInstrumentor 已启用，将监视 agent 执行过程")
            print("📊 Telemetry traces 将发送到 Phoenix 默认端点")
        except ImportError as e:
            print(f"⚠️  无法导入 telemetry 模块: {e}")
            print("💡 请确保已安装相关依赖: uv pip install 'smolagents[telemetry,toolkit]'")
        except Exception as e:
            print(f"⚠️  无法启用 SmolagentsInstrumentor: {e}")
    
    with Timer("整个测试程序", args.timer):
        with Timer("数据加载", args.timer):
            data_path = args.data_path + args.split + '.jsonl'
            data = load_data(data_path)
            
            # 根据args.samples决定使用多少样本
            if args.samples is None:
                num_samples = len(data)
                print(f"📚 使用全部测试样本: {num_samples} 个")
            else:
                num_samples = min(args.samples, len(data))
                data = data[:num_samples]
                print(f"📚 加载了 {num_samples} 个测试样本 (共 {len(load_data(data_path))} 个可用)")

        with Timer("组件初始化", args.timer):
            # 首先初始化内存检索系统
            memory_retrieval_tool = None
            memory_stats = {}
            
            if args.retrieve:
                with Timer("内存检索系统初始化", args.timer):
                    # 创建语言模型用于内存检索器
                    memory_model = OpenAIServerModel(
                        model_id=args.actor,  # 使用与actor相同的模型
                        api_base=os.getenv("BASE_URL"),
                        api_key=os.getenv("API_KEY")
                    )
                    
                    # 根据任务类型选择不同的 Memory 系统提示
                    if args.task_type == 'retro':
                        system_prompt_path = 'prompts/memory_retro.yaml'
                    else:
                        system_prompt_path = 'prompts/memory_aime.yaml'

                    print(f"🧩 使用Memory系统提示: {system_prompt_path}")

                    # 创建持久化内存检索agent（带系统提示路径）
                    memory_retriever = create_memory_retriever(memory_model, args.memory_path, system_prompt_path)
                    
                    # 创建绑定的内存检索工具
                    memory_retrieval_tool = create_memory_retrieval_tool(memory_retriever)
                    
                    # 获取内存统计信息
                    total_memories = sum(len(partition) for partition in memory_retriever.memory_data.values())
                    memory_stats = {
                        'total_memories': total_memories,
                        'golden_count': len(memory_retriever.memory_data.get('golden', [])),
                        'warning_count': len(memory_retriever.memory_data.get('warning', [])),
                        'mixed_count': len(memory_retriever.memory_data.get('mixed', []))
                    }
                    
                    print(f"🧠 内存检索系统已启用，使用内存文件: {args.memory_path}")
                    print(f"📖 Memory统计: 总计{total_memories}条经验 (Golden: {memory_stats['golden_count']}, Warning: {memory_stats['warning_count']}, Mixed: {memory_stats['mixed_count']})")
            else:
                print("🚫 内存检索系统未启用")
                memory_stats = {'total_memories': 0, 'golden_count': 0, 'warning_count': 0, 'mixed_count': 0}
            
            if args.timer:
                print("🚀 测试组件初始化完成")

        # 测试流程 - 根据pass_at_n选择模式
        if args.pass_at_n > 1:
            # Pass@N 模式
            with Timer("Pass@N测试过程", args.timer):
                print(f"\n🔄 使用Pass@{args.pass_at_n}模式")
                print(f"🧠 Memory检索状态: {'启用' if args.retrieve else '禁用'}")
                
                pass_rate, detailed_results, pass_results, total_time = await run_passat_test(
                    data, memory_retrieval_tool, args.pass_at_n, args.max_concurrent, args.actor, args.retrieve
                )
        else:
            # 传统模式 (Pass@1)
            with Timer("传统测试过程", args.timer):
                batch_size = args.batch_size
                total_correct = 0
                num_batches = (len(data) + batch_size - 1) // batch_size
                all_results = []
                
                if args.timer:
                    print(f"\n📈 开始传统测试, 共 {num_batches} 个batch")
                    print(f"🧠 Memory检索状态: {'启用' if args.retrieve else '禁用'}")
                
                for i in range(0, len(data), batch_size):
                    batch = data[i:i+batch_size]
                    batch_idx = i // batch_size
                    
                    if not args.timer:
                        print(f'Testing batch {batch_idx + 1}/{num_batches}')
                    
                    batch_correct, batch_results = await test_batch(batch, memory_retrieval_tool, batch_idx, batch_size, args.actor, args.retrieve, args.results_dir)
                    total_correct += batch_correct
                    all_results.extend(batch_results)
                    
                    # 批次间等待（除了最后一个批次）
                    if i + batch_size < len(data):
                        if args.timer:
                            print("⏸️  批次间等待 3秒...")
                        await asyncio.sleep(3)
                    
                # 传统模式统计
                accuracy = total_correct / num_samples * 100
                
                # 收集正确题目的题号
                correct_problem_ids = []
                for i, result in enumerate(all_results):
                    if not isinstance(result, Exception):
                        is_correct, _, _ = result
                        if is_correct:
                            correct_problem_ids.append(i + 1)  # 题号从1开始
                
                if args.timer:
                    print(f"\n🎯 测试结果: 准确率 = {accuracy:.2f}% ({total_correct}/{num_samples})")
                    print(f"🧠 Memory使用状态: {'已启用' if args.retrieve else '已禁用'}")
                    print(f"\n✅ 答对的题目题号 ({len(correct_problem_ids)}题):")
                    print(f"   {correct_problem_ids}")
                else:
                    print(f'Test Accuracy = {accuracy:.2f}%')
                    print(f'Correct problems: {correct_problem_ids}')

if __name__ == "__main__":
    if args.timer:
        print("🧪 Memory测试程序开始...")
        print(f"📋 配置: Actor={args.actor}, Memory={'启用' if args.retrieve else '禁用'}, Telemetry={'启用' if args.telemetry else '禁用'}")
        if args.pass_at_n > 1:
            print(f"🔄 Pass@{args.pass_at_n} 模式, 最大并发={args.max_concurrent}")
        else:
            print(f"📊 传统模式 (Pass@1), batch_size={args.batch_size}")
    
    asyncio.run(main())
    
    if args.timer:
        print("🎉 Memory测试完成！")
