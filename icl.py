import os
import json
import asyncio
import argparse
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
import re


def load_jsonl(path: str) -> List[Dict[str, Any]]:
	data: List[Dict[str, Any]] = []
	with open(path, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			data.append(json.loads(line))
	return data


def extract_answer_math(answer_text: str) -> str:
	# Mimic tts.py's extract_answer: numeric at end or after indicators
	patterns = [
		r"(?:answer is|answer:|the answer is)\s*([+-]?\d+(?:\.\d+)?)",
		r"([+-]?\d+(?:\.\d+)?)(?:\s*$|\s*\.$)",
	]
	s = str(answer_text)
	for pattern in patterns:
		m = re.search(pattern, s.lower())
		if m:
			return m.group(1)
	nums = re.findall(r'([+-]?\d+(?:\.\d+)?)', s)
	return nums[0] if nums else "0"


def parse_tag_block(text: str, tag: str, *, first: bool = True) -> str:
	pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
	matches = pattern.findall(text or "")
	if not matches:
		return ""
	return matches[0].strip() if first else matches[-1].strip()


def build_icl_context(
	examples: List[Dict[str, Any]],
	task_type: str,
	max_examples: int,
) -> str:
	lines: List[str] = []
	lines.append("Below are solved examples. Learn the input-output pattern and answer the new problem accordingly.\n")
	sel = examples[:max_examples] if max_examples else examples
	for idx, ex in enumerate(sel, start=1):
		q = str(ex.get('question', '')).strip()
		a = str(ex.get('answer', '')).strip()
		if task_type == 'retro':
			lines.append(f"Example {idx}:")
			lines.append(f"Product: {q}")
			lines.append(f"<answer> {a} </answer>")
			lines.append("")
		else:
			# math: show final numeric only to reduce tokens
			final_num = extract_answer_math(a)
			lines.append(f"Example {idx}:")
			lines.append(f"Problem: {q}")
			lines.append(f"Final Answer: {final_num}")
			lines.append("")
	return "\n".join(lines)


async def evaluate_sample(
	client: AsyncOpenAI,
	model: str,
	temperature: float,
	system_prompt: str,
	icl_context: str,
	sample: Dict[str, Any],
	task_type: str,
) -> Dict[str, Any]:
	q = str(sample.get('question', '')).strip()
	gt_raw = str(sample.get('answer', '')).strip()

	if task_type == 'retro':
		user_prompt = (
			f"{icl_context}\n"
			"Now solve the new problem.\n"
			f"Product: {q}\n"
			"Provide your final reactant SMILES strictly inside <answer> and </answer> tags, dot-separated if multiple."
		)
	else:
		user_prompt = (
			f"{icl_context}\n"
			"Now solve the new problem.\n"
			f"Problem: {q}\n"
			"Respond with reasoning if needed, but clearly state the final numeric value in the last line as 'Final Answer: X'."
		)

	resp = await client.chat.completions.create(
		model=model,
		messages=[
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		],
		temperature=temperature,
	)
	content = resp.choices[0].message.content or ""

	if task_type == 'retro':
		pred = parse_tag_block(content, 'answer', first=True)
		gt = gt_raw
		is_correct = (pred.strip() == gt.strip())
	else:
		# extract final numeric from model content and GT
		pred = extract_answer_math(content)
		gt = extract_answer_math(gt_raw)
		is_correct = (pred.strip() == gt.strip())

	return {
		"question": q,
		"ground_truth": gt,
		"prediction": pred,
		"raw_output": content,
		"correct": is_correct,
	}


async def main():
	parser = argparse.ArgumentParser(description='ICL evaluation script (math | retro)')
	parser.add_argument('--type', choices=['math', 'retro'], default='retro', help='Task type')
	parser.add_argument('--model', type=str, default='openai/gpt-5', help='Model name/ID')
	parser.add_argument('--temperature', type=float, default=0.0, help='Generation temperature')
	parser.add_argument('--data_path', type=str, default="data/uspto50k/train.jsonl", help='Path to context examples (JSONL with question/answer)')
	parser.add_argument('--eval_path', type=str, default="data/uspto50k/test.jsonl", help='Optional eval set path; defaults to data_path if not provided')
	parser.add_argument('--max_context_examples', type=int, default=50, help='Max examples to include in ICL context')
	parser.add_argument('--max_eval_samples', type=int, default=None, help='Limit number of eval samples')
	parser.add_argument('--results_dir', type=str, default='results/icl_retro_gpt', help='Directory to save per-sample results')
	parser.add_argument('--batch_size', type=int, default=5, help='Number of concurrent eval requests to run per batch')
	parser.add_argument('--batch_pause', type=float, default=0.5, help='Pause seconds between batches to avoid rate limits')
	parser.add_argument('--base_url', type=str, default=os.getenv('BASE_URL', ''), help='OpenAI-compatible base URL')
	parser.add_argument('--api_key', type=str, default=os.getenv('API_KEY', ''), help='API key')
	args = parser.parse_args()

	# Load examples for context
	context_data = load_jsonl(args.data_path)
	if args.eval_path:
		eval_data = load_jsonl(args.eval_path)
	else:
		eval_data = context_data

	if args.max_eval_samples:
		eval_data = eval_data[: args.max_eval_samples]

	# System prompt by task
	if args.type == 'retro':
		# Prefer retro-specific system prompt if available
		system_prompt_path = 'prompts/llm_system_prompt_retro.txt'
		if os.path.exists(system_prompt_path):
			with open(system_prompt_path, 'r', encoding='utf-8') as f:
				system_prompt = f.read()
		else:
			system_prompt = (
				"You are a chemistry assistant specialized in single-step retrosynthesis. "
				"Given a product SMILES, output ONLY the reactant SMILES inside <answer></answer> tags."
			)
	else:
		# math
		with open('prompts/llm_system_prompt.txt', 'r', encoding='utf-8') as f:
			system_prompt = f.read()

	# Build ICL context block (few-shot)
	icl_context = build_icl_context(context_data, args.type, args.max_context_examples)

	# Init client
	client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)

	# Prepare results dir
	os.makedirs(args.results_dir, exist_ok=True)

	print(f"ICL eval | type={args.type} | model={args.model} | context_examples={min(args.max_context_examples, len(context_data))}/{len(context_data)} | eval={len(eval_data)}")

	correct = 0
	correct = 0
	total = len(eval_data)
	batch_size = max(1, int(args.batch_size))
	for start in range(0, total, batch_size):
		end = min(start + batch_size, total)
		batch_items = eval_data[start:end]
		tasks = [
			evaluate_sample(
				client, args.model, args.temperature, system_prompt, icl_context, item, args.type
			)
			for item in batch_items
		]
		batch_results = await asyncio.gather(*tasks)

		# Save results and update counts
		for idx, result in enumerate(batch_results):
			global_id = start + idx + 1
			if result['correct']:
				correct += 1
			with open(os.path.join(args.results_dir, f"{global_id}.json"), 'w', encoding='utf-8') as f:
				json.dump(result, f, ensure_ascii=False, indent=2)
			status = '✅' if result['correct'] else '❌'
			print(f"{status} [{global_id}/{total}] Correct={result['correct']}")

		# Batch summary
		batch_pass = sum(1 for r in batch_results if r['correct'])
		print(f"Batch {start//batch_size + 1}: {batch_pass}/{len(batch_results)} correct")
		if end < total and args.batch_pause > 0:
			print(f"Pausing {args.batch_pause:.1f}s before next batch...")
			await asyncio.sleep(args.batch_pause)

	acc = (correct / total) * 100 if total else 0.0
	print(f"\nSummary: {correct}/{total} correct ({acc:.2f}%)")

	acc = (correct / len(eval_data)) * 100 if eval_data else 0.0
	print(f"\nSummary: {correct}/{len(eval_data)} correct ({acc:.2f}%)")


if __name__ == '__main__':
	asyncio.run(main())

