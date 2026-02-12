import json
import asyncio
import os
import argparse
from actor import Actor
from openai import AsyncOpenAI

async def process_question(i, item, use_actor, total, model, llm_dir, client, temperature=0.1, retrieve=False):
    """Process a single question and save the result.

    Args:
        i: 1-based question index.
        item: Data sample dict.
        use_actor: Whether to route through Actor or raw model.
        total: Total number of questions (for progress display).
        model: Model name/ID to use.
        llm_dir: Directory to save results.
        client: AsyncOpenAI client instance.
        temperature: Model temperature (default: 0.1).
        retrieve: Whether to retrieve from experience library (default: False).
    """
    print(f"Starting question {i}/{total}...")
    
    query = item['question']
    correct_answer = item['answer']
    
    if use_actor:
        actor = Actor(
            model=model,
            temperature=temperature,
            retrieve=retrieve
        )
        response = await actor.act(query)
    else:
        # Load system prompt from file
        with open("prompts/llm_system_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
        
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=temperature,
        )
        print(f"Idx: {i}, response:\n{response}")
        response = response.choices[0].message.content.strip()
    
    # Create content for the file
    content = f"Question {i}:\n{query}\n\nAgent Response:\n{response}\n\nCorrect Answer:\n{correct_answer}\n"
    
    # Save to llm directory
    filename = f"{i}.txt"
    filepath = os.path.join(llm_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Completed question {i}/{total} - saved to {os.path.basename(llm_dir)}/{filename}")
    return i

async def main():
    """Run full evaluation over the entire loaded dataset."""
    
    # ============================================================================
    # Configuration Section - All configurable parameters
    # ============================================================================
    parser = argparse.ArgumentParser(description='AIME Testing Script')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='z-ai/glm-4.5',
                        help='Model name/ID to use')
    parser.add_argument('--temperature', type=float, default=0,
                        help='Model temperature')
    parser.add_argument('--use_actor', action='store_true', default=True,
                        help='Use Actor wrapper (default: False, direct API)')
    parser.add_argument('--retrieve', action='store_true', default=False,
                        help='Retrieve from experience library (only used with --use_actor)')
    
    # Data configuration
    parser.add_argument('--data_path', type=str, default='data/AIME/aime25.jsonl',
                        help='Path to test data file')
    
    # Output configuration
    parser.add_argument('--results_dir', type=str, default='results/agent_aime_glm',
                        help='Directory to save results')
    
    # Batch processing configuration
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Number of questions to process in each batch')
    parser.add_argument('--batch_pause', type=float, default=0.5,
                        help='Pause duration (seconds) between batches')

    # Telemetry
    parser.add_argument('--no-telemetry', dest='telemetry', action='store_false', default=True, 
                        help='Disable SmolagentsInstrumentor telemetry (default: enabled)')
    
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

    # Initialize OpenAI client with 10 minute timeout
    client = AsyncOpenAI(
        api_key=os.getenv("API_KEY", ""),
        base_url=os.getenv("BASE_URL", ""),
        timeout=1200.0  # 20 minutes timeout
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
    print(f"Retrieve: {args.retrieve}")
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
                temperature=args.temperature,
                retrieve=args.retrieve
            ) 
            for i, item in enumerate(batch_data)
        ]
        
        # Process current batch
        await asyncio.gather(*tasks)
        
        print(f"\nBatch {batch_num} completed")
        
        # Brief pause between batches to prevent overload
        if batch_end < total:
            print(f"Pausing {args.batch_pause} seconds before next batch...")
            await asyncio.sleep(args.batch_pause)

    print(f"\n{'='*80}")
    print(f"All questions completed. Results saved to {args.results_dir}")
    print(f"{'='*80}")

# Run the main function
asyncio.run(main())