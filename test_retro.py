import json
import asyncio
import os
import re
import argparse
from actor import Actor
from openai import AsyncOpenAI
from typing import Optional

def parse_tag_block(text: str, tag: str, *, first: bool = True) -> Optional[str]:
    """Extract content enclosed by a given XML-like tag.

    Parameters
    ----------
    text : str
        Source text.
    tag : str
        Tag name without angle brackets.
    first : bool, default True
        If True return first match; else return the last match (useful when the
        model duplicates sections).

    Returns
    -------
    Optional[str]
        Inner content or None if not found.
    """
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text or "")
    if not matches:
        return ""
    return matches[0].strip() if first else matches[-1].strip()

# Full dataset test (removed previous slicing of last 30 questions)
async def process_question(i, item, use_actor, total, model, llm_dir, client, temperature=0):
    """Process a single question and save the result.

    Args:
        i: 1-based question index.
        item: Data sample dict.
        use_actor: Whether to route through Actor or raw model.
        total: Total number of questions (for progress display).
        model: Model name/ID to use.
        llm_dir: Directory to save results.
        client: AsyncOpenAI client instance.
        temperature: Model temperature (default: 0).
    """
    print(f"Starting question {i}/{total}...")
    
    # Log path
    filename = f"{i}.txt"
    filepath = os.path.join(llm_dir, filename)
    
    product = item['question']
    correct_answer = item['answer']

    if use_actor:
        actor = Actor(
            model=model,
            temperature=temperature,
            retrieve=False,
            log_path=filepath
        )

        prompt = f"""
<SMILES> {product} </SMILES>
You are given the product's SMILES string. Predict the SMILES string(s) of the reactant(s) for a **single-step** retrosynthesis using your knowledge of chemical retrosynthesis.
Please reason step by step, and provide your final answer enclosed within the final_answer() tool.
**Important:** Only include SMILES notation inside the final_answer tool. If there is more than one reactant, separate them with a dot ('.'). Do not add any explanations or comments there.
Example:
final_answer("CCOc1ccc(Oc2ncnc3c2cnn3C2CCNCC2)c(F)c1.C(=O)(Cl)OC1CCCC1") # Put your final answer in the final_answer tool! 
Now it's your turn.
"""
        final_answer = await actor.act(prompt)
        with open(filepath, "r") as f: 
            response = f.read().strip()
    else:
        prompt = f"""
<SMILES> {product} </SMILES>
You are given the product's SMILES string. Predict the SMILES string(s) of the reactant(s) for a **single-step** retrosynthesis using your knowledge of chemical retrosynthesis.
Please reason step by step, and provide your final answer enclosed within the <answer> and </answer> tags.
**Important:** Only include SMILES notation inside the <answer> and </answer> tags. If there is more than one reactant, separate them with a dot ('.'). Do not add any explanations or comments there.
Example:
<answer> CCOc1ccc(Oc2ncnc3c2cnn3C2CCNCC2)c(F)c1.C(=O)(Cl)OC1CCCC1 </answer>
Now it's your turn.
"""
        # Load system prompt from file
        with open("prompts/llm_system_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
        
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )
        response = response.choices[0].message.content.strip()
        print(f"Sample {i}, response: {response}")
        final_answer = parse_tag_block(response, "answer", first=True)
        print(f"Sample {i}, final answer: {final_answer}")

    is_passed = (final_answer.strip() == correct_answer.strip())
    
    # Create content for the file
    content = f"Question {i}:\n{prompt}\n\nResponse:\n{response}\n\nFinal Answer:{final_answer}\n\nCorrect Answer:\n{correct_answer}\n\nPassed:\n{is_passed}"
    
    # Save to llm directory
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Completed question {i}/{total} - saved to {os.path.basename(llm_dir)}/{filename}")
    return is_passed

async def main():
    """Run full evaluation over the entire loaded dataset."""
    
    # ============================================================================
    # Configuration Section - All configurable parameters
    # ============================================================================
    parser = argparse.ArgumentParser(description='Retrosynthesis Testing Script')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='x-ai/grok-4',
                        help='Model name/ID to use')
    parser.add_argument('--temperature', type=float, default=0,
                        help='Model temperature')
    parser.add_argument('--use_actor', action='store_true', default=True,
                        help='Use Actor wrapper (default: False, direct API)')
    
    # Data configuration
    parser.add_argument('--data_path', type=str, default='data/uspto50k/test.jsonl',
                        help='Path to test data file')
    
    # Output configuration
    parser.add_argument('--results_dir', type=str, default='results/agent_retro_grok',
                        help='Directory to save results')
    
    # Batch processing configuration
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of questions to process in each batch')
    parser.add_argument('--batch_pause', type=float, default=2.0,
                        help='Pause duration (seconds) between batches')

    # Telemetry
    parser.add_argument('--no-telemetry', dest='telemetry', action='store_false', default=True, help='Disable SmolagentsInstrumentor telemetry (default: enabled)')
    
    args = parser.parse_args()
    
    # ============================================================================
    # Initialize components based on configuration
    # ============================================================================
    if args.telemetry:
        try:
            # 动态导入 telemetry 相关模块
            from phoenix.otel import register
            from openinference.instrumentation.smolagents import SmolagentsInstrumentor
            
            register(
                project_name="default",
            )  # 注册 Phoenix OTEL
            SmolagentsInstrumentor().instrument()  # 启用 Smolagents instrumentation
            print("🔍 SmolagentsInstrumentor 已启用，将监视 agent 执行过程")
            print("📊 Telemetry traces 将发送到 Phoenix 默认端点")
        except ImportError as e:
            print(f"⚠️  无法导入 telemetry 模块: {e}")
            print("💡 请确保已安装相关依赖: uv pip install 'smolagents[telemetry,toolkit]'")
        except Exception as e:
            print(f"⚠️  无法启用 SmolagentsInstrumentor: {e}")

    # Initialize OpenAI client
    client = AsyncOpenAI(
        api_key=os.getenv("API_KEY", ""),
        base_url=os.getenv("BASE_URL", "")
    )
    
    # Load data
    print(f"Loading data from: {args.data_path}")
    data = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Results will be saved to: {args.results_dir}")
    
    # Print configuration summary
    print(f"\n{'='*80}")
    print(f"Configuration Summary:")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Use Actor: {args.use_actor}")
    print(f"Data path: {args.data_path}")
    print(f"Results directory: {args.results_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batch pause: {args.batch_pause}s")
    print(f"Total questions: {len(data)}")
    print(f"{'='*80}\n")
    
    # ============================================================================
    # Main testing loop
    # ============================================================================

    total = len(data)
    batch_size = args.batch_size
    print(f"Starting full test over {total} questions (batch size: {batch_size})...")
    
    all_results = []
    
    # Process data in batches
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_num = (batch_start // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size
        
        print(f"\n{'='*80}")
        print(f"Processing batch {batch_num}/{total_batches} (questions {batch_start + 1}-{batch_end})")
        print(f"{'='*80}")
        
        # Create tasks for current batch
        batch_data = data[batch_start:batch_end]
        tasks = [
            process_question(
                batch_start + i + 1, 
                item, 
                use_actor=args.use_actor, 
                total=total, 
                model=args.model,
                llm_dir=args.results_dir,
                client=client,
                temperature=args.temperature
            ) 
            for i, item in enumerate(batch_data)
        ]
        
        # Process current batch
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)
        
        # Show batch statistics
        batch_passed = sum(1 for result in batch_results if result)
        batch_accuracy = (batch_passed / len(batch_results)) * 100 if batch_results else 0
        print(f"\nBatch {batch_num} completed: {batch_passed}/{len(batch_results)} passed ({batch_accuracy:.2f}%)")
        
        # Brief pause between batches to prevent overload
        if batch_end < total:
            print(f"Pausing {args.batch_pause} seconds before next batch...")
            await asyncio.sleep(args.batch_pause)

    # Calculate overall accuracy
    passed_count = sum(1 for result in all_results if result)
    accuracy = (passed_count / total) * 100 if total > 0 else 0
    
    # Collect correct problem IDs
    correct_problem_ids = [i + 1 for i, result in enumerate(all_results) if result]
    
    print(f"\n{'='*80}")
    print(f"All questions completed. Results saved to {args.results_dir}")
    print(f"{'='*80}")
    print(f"Total questions: {total}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {total - passed_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"\n✅ Correct problem IDs ({len(correct_problem_ids)} problems):")
    print(f"   {correct_problem_ids}")
    print(f"{'='*80}")

# Run the main function
asyncio.run(main())