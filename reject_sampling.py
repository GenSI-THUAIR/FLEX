# This file implements a reject sampling workflow:
# The LLM will try to answer a query and a critic will judge on the result and 
# generate feedback for the next round of execution.
# This loop will not stop until the critic is satisfied with the LLM's output.

import asyncio
import os
import json
from rate_limiter import RateLimitedLLMClient
from critic import Critic
from explib import ExpLib

async def process_task(
    query: str,
    ground_truth: str,
    client: RateLimitedLLMClient,
    llm: str,
    verifier: str,
    critic: str,
    max_rounds: int = 5,
    max_rollouts: int = 3,
    task_type: str = "math",
    llm_system_prompt_path: str | None = None,
    verifier_prompt_path: str | None = None,
    with_gt: bool = True,
):
    """
    Implement reject sampling for the given task (query) with maximum `max_rounds` of llm+verifier loop and `max_rollouts` generations.

    Args:
        query (str): The given task.
        ground_truth (str): The ground truth for the query.
        client (RateLimitedLLMClient): The llm client that send the API call.
        llm (str): The model used to generate the answer.
        verifier (str): The model used to provide feedback on the llm's answer during reject sampling.
        critic (str): The model used to extract memory from the final trajectory.
        max_rounds (int): The workflow for one rollout will be a loop of llm generation followed by the verifier's feedback. 
                          This argument determines the maximum rounds of the loop.
        max_rollouts (int): For a task (query), there are only at most `max_rollouts` tries. If in one rollout, the verifier is 
                            satisfied with the llm's answer, there will not be more tries.
        task_type (str): Type of task, either "math" or "retro".
        llm_system_prompt_path (str | None): Path to custom LLM system prompt file.
        verifier_prompt_path (str | None): Path to custom verifier prompt file.
        with_gt (bool): Whether to provide ground truth to the verifier. Default is True.
    """

    all_histories = [] # Contains all rollouts
    for rollout in range(max_rollouts):
        is_correct = False
        responses = [] # Contains all llm generations
        history = [] # Contains both llm generations and verifier feedbacks
        for r in range(max_rounds):
            # Each round contains a llm generation and a verifier feedback
            # Load the system prompt depending on task type
            if task_type == "retro":
                # Try reading retro system prompt from provided path; fallback to built-in minimal retro prompt
                if llm_system_prompt_path:
                    try:
                        with open(llm_system_prompt_path, "r", encoding="utf-8") as f:
                            llm_system_prompt = f.read()
                    except Exception:
                        llm_system_prompt = (
                            "You are a chemistry assistant specialized in single-step retrosynthesis. "
                            "Given a product SMILES, think step by step and output ONLY the reactant SMILES inside <answer></answer> tags."
                        )
                else:
                    llm_system_prompt = (
                        "You are a chemistry assistant specialized in single-step retrosynthesis. "
                        "Given a product SMILES, think step by step and output ONLY the reactant SMILES inside <answer></answer> tags."
                    )
            else:
                # math
                with open("prompts/llm_system_prompt.txt", "r") as f:
                    llm_system_prompt = f.read()

            # Generate the messages
            if task_type == "retro":
                if responses:
                    processed_query = f"""
## Single-Step Retrosynthesis (Reject Sampling)

Your task is to predict the reactant SMILES for a single-step retrosynthesis.

- Product SMILES: {query}
- Round: {r + 1} of Rollout {rollout + 1}
- Attempts remaining in this rollout: {max_rounds - r}
- Total rollouts allowed: {max_rollouts}

Previous attempt:
{responses[-1]}

Verifier feedback on previous attempt:
{history[-1]}

What to do now:
1. Address verifier's comments precisely.
2. Reconsider the disconnection strategy and common reaction types.
3. Ensure the final answer is ONLY the reactant SMILES in <answer>...</answer>. If multiple, separate with a dot.
4. Do not include any text inside <answer> other than SMILES.

Provide reasoning first (outside the tags), then the final SMILES inside <answer>...</answer>.
"""
                    question = processed_query
                else:
                    question = f"""
## Single-Step Retrosynthesis (Reject Sampling)

You are given the product's SMILES string. Predict the SMILES string(s) of the reactant(s) for a single-step retrosynthesis.

- Product SMILES: {query}
- Expected answer format: canonical SMILES string(s) of reactants, dot-separated, enclosed within <answer> and </answer> tags.

Instructions:
1. Think step by step about plausible disconnections and reaction classes.
2. Ensure dataset conventions for reactant sets are followed.
3. The final answer must be ONLY SMILES inside <answer> tags; no extra text.

Example:
<answer> CCO.CC(=O)Cl </answer>

Now solve this instance. Provide your reasoning, then output the final answer inside <answer> tags.
"""
            else:
                if responses:
                    processed_query = f"""
## Reject Sampling Problem-Solving Task

**Your Mission:** You are participating in a reject sampling process where you need to solve a mathematical problem through iterative refinement. Your goal is to provide a correct, well-reasoned solution that will satisfy a strict verifier.

### Problem to Solve:
{query}

### Current Context:
- This is Round {r + 1} of Rollout {rollout + 1}
- You have {max_rounds - r} attempts remaining in this rollout
- Total rollouts allowed: {max_rollouts}

### Your Previous Attempt:
{responses[-1]}

### Verifier's Feedback on Your Previous Attempt:
{history[-1]}

### What You Need to Do:
1. **Carefully analyze the verifier's feedback** - Identify specific issues mentioned
2. **Reflect on your previous approach** - What went wrong? What was missing?
3. **Develop an improved strategy** - How can you address the feedback?
4. **Provide a complete solution** - Show all work clearly and systematically
5. **Double-check your work** - Verify each step before finalizing

### Success Criteria:
- Mathematical accuracy and logical soundness
- Clear, step-by-step reasoning
- Proper justification for each step
- Correct final answer
- Address all issues raised in the feedback

Please provide your improved solution below:
"""
                    question = processed_query
                else:
                    question = f"""
## Reject Sampling Problem-Solving Task

**Your Mission:** You are participating in a reject sampling process where you need to solve a mathematical problem. Your solution will be evaluated by a strict verifier, so accuracy and clarity are essential.

### Problem to Solve:
{query}

### Current Context:
- This is Round 1 of Rollout {rollout + 1}
- You have {max_rounds} attempts in this rollout
- Total rollouts allowed: {max_rollouts}
- Expected answer format: Numerical value (will be verified against ground truth)

### Your Task:
1. **Analyze the problem thoroughly** - Understand what is being asked
2. **Plan your approach** - Choose the most appropriate method
3. **Execute step-by-step** - Show all mathematical work clearly
4. **Verify your solution** - Check for errors and logical consistency
5. **Present your final answer** - State it clearly and unambiguously

### Success Criteria:
- Mathematical accuracy and rigor
- Clear, logical progression of steps
- Proper mathematical notation and reasoning
- Correct final numerical answer
- Well-organized presentation

### Quality Standards:
- Each step should be justified
- Calculations should be shown explicitly
- Assumptions should be stated clearly
- The solution should be complete and self-contained

Please solve the problem systematically and provide your complete solution:
"""
            llm_messages = [
                {"role": "system", "content": llm_system_prompt},
                {"role": "user", "content": question}
            ]

            # Collect the llm's generation
            print(f"\n🧠 [LLM GENERATOR] Calling {llm} - Rollout {rollout + 1}, Round {r + 1}")
            preview_label = "Product SMILES" if task_type == "retro" else "Query"
            print(f"📝 [LLM GENERATOR] {preview_label} preview: {str(query)[:100]}...")
            response = await client.chat_completion(
                model=llm,
                messages=llm_messages,
                temperature=0.5
            )
            # Ensure response is a string
            response_str = str(response) if response else ""
            print(f"✅ [LLM GENERATOR] Response received ({len(response_str)} chars)")
            print(f"📄 [LLM GENERATOR] Full Response:")
            print(f"{response_str}")
            print(f"🔚 [LLM GENERATOR] Response end")
            responses.append(response_str)
            history.append(response_str)

            # Collect the verifier's feedback
            if task_type == "retro":
                # Retro verifier system prompt
                if verifier_prompt_path:
                    try:
                        with open(verifier_prompt_path, "r", encoding="utf-8") as f:
                            verifier_system_prompt = f.read()
                    except Exception:
                        verifier_system_prompt = (
                            "You are a strict retrosynthesis verifier. Extract ONLY from <answer>...</answer> and compare to ground truth string."
                        )
                else:
                    verifier_system_prompt = (
                        "You are a strict retrosynthesis verifier. Extract ONLY from <answer>...</answer> and compare to ground truth string."
                    )

                if with_gt:
                    # With ground truth: compare against GT
                    processed_response = f"""
## Retrosynthesis Verification Task

Role: You are a strict retrosynthesis verifier in a reject sampling process. Evaluate the LLM's solution and decide whether to accept or continue.

Original Product SMILES: {query}
Current Round: {r + 1} of {max_rounds} in Rollout {rollout + 1}
Remaining Rounds: {max_rounds - r}
Ground Truth (canonical reactant SMILES): {ground_truth}

LLM's Solution to Evaluate:
{response_str}

Verification Instructions:
1. Extract the predicted reactant SMILES ONLY from <answer>...</answer> tags.
2. Strip whitespace and compare EXACT STRING EQUALITY with the Ground Truth provided above.
3. If equal and the reasoning is acceptable, output <next_step>end</next_step>.
4. Otherwise, provide precise feedback and output <next_step>continue</next_step>.

Provide your evaluation and the decision tag at the end.
"""
                else:
                    # Without ground truth: evaluate based on chemical validity and reasoning
                    processed_response = f"""
## Retrosynthesis Verification Task (No Ground Truth)

Role: You are a strict retrosynthesis verifier in a reject sampling process. Evaluate the LLM's solution based on chemical validity and reasoning quality.

Original Product SMILES: {query}
Current Round: {r + 1} of {max_rounds} in Rollout {rollout + 1}
Remaining Rounds: {max_rounds - r}

LLM's Solution to Evaluate:
{response_str}

Verification Instructions (WITHOUT Ground Truth):
1. Extract the predicted reactant SMILES from <answer>...</answer> tags.
2. Evaluate chemical validity:
   - Are the SMILES strings valid?
   - Is the retrosynthetic disconnection chemically reasonable?
   - Are the functional groups and stereochemistry preserved appropriately?
3. Evaluate reasoning quality:
   - Is the disconnection strategy clearly explained?
   - Are reaction mechanisms and precedents properly considered?
   - Is the reasoning logically sound?
4. Decision criteria:
   - If the solution is chemically valid, well-reasoned, and represents a plausible retrosynthetic step, output <next_step>end</next_step>
   - Otherwise, provide specific chemical feedback and output <next_step>continue</next_step>

Provide your evaluation and the decision tag at the end.
"""
            else:
                with open("prompts/verifier.txt", "r") as f:
                    verifier_system_prompt = f.read()
                
                if with_gt:
                    # With ground truth: compare against GT
                    processed_response = f"""
## Reject Sampling Verification Task

**Your Role:** You are a strict mathematical verifier in a reject sampling process. Your job is to rigorously evaluate the LLM's solution, provide actionable feedback to help improve the solution quality and help the LLM arrive at the correct answer.

### Problem Context:
**Original Query:** {query}
**Current Round:** {r + 1} of {max_rounds} in Rollout {rollout + 1}
**Remaining Rounds:** {max_rounds - r}
**Ground Truth (numeric):** {ground_truth}

### LLM's Solution to Evaluate:
{response_str}

### Your Evaluation Tasks:

#### 1. Mathematical Accuracy Assessment
- Check all calculations for errors
- Verify logical flow and reasoning steps
- Ensure mathematical notation is correct
- Validate that the approach is sound

#### 2. Completeness Analysis
- Is the solution complete from start to finish?
- Are all necessary steps included?
- Is the final answer clearly stated?
- Are any crucial steps missing or unclear?

#### 3. Clarity and Rigor Evaluation
- Is the reasoning easy to follow?
- Are assumptions stated clearly?
- Is the mathematical work shown explicitly?
- Would another mathematician be able to verify this solution?

#### 4. Decision Making (STRICT)
**You must decide:** Should this solution be accepted or does it need improvement?

First, EXTRACT the LLM's final numeric answer (e.g., using patterns like "The answer is X").
Then, COMPARE it to the provided Ground Truth exactly (allowing minor whitespace/formatting differences only).

**Accept (use `<next_step>end</next_step>`)** ONLY IF:
- The extracted final numeric answer EQUALS the Ground Truth, and
- The mathematical reasoning is sound and complete.

**Continue (use `<next_step>continue</next_step>`)** if:
- Mathematical errors or logical flaws exist
- Important steps are missing or inadequately explained
- The reasoning is unclear or unjustified
- The approach seems fundamentally flawed
- The extracted final numeric answer DOES NOT MATCH the Ground Truth

### Your Feedback Format:
1. **Summary Assessment:** Brief overall evaluation
2. **Extracted Final Answer:** State the number you extracted from the LLM's solution
3. **Truth Comparison:** Explicitly state whether it matches the Ground Truth and why
4. **Specific Issues:** Point out exact problems (if any)
5. **Suggestions:** Concrete improvement recommendations (if continuing)
6. **Decision:** Your final verdict with appropriate next_step tag

Please provide your thorough evaluation and decision:
"""
                else:
                    # Without ground truth: evaluate based on mathematical rigor only
                    processed_response = f"""
## Reject Sampling Verification Task (No Ground Truth)

**Your Role:** You are a strict mathematical verifier in a reject sampling process. Evaluate the LLM's solution based on mathematical rigor, logical soundness, and completeness—WITHOUT access to ground truth.

### Problem Context:
**Original Query:** {query}
**Current Round:** {r + 1} of {max_rounds} in Rollout {rollout + 1}
**Remaining Rounds:** {max_rounds - r}

### LLM's Solution to Evaluate:
{response_str}

### Your Evaluation Tasks:

#### 1. Mathematical Accuracy Assessment
- Check all calculations for arithmetic errors
- Verify each step follows logically from the previous
- Ensure mathematical notation and terminology are used correctly
- Validate that the mathematical approach is valid

#### 2. Completeness Analysis
- Is the solution complete from problem statement to final answer?
- Are all necessary steps included and explained?
- Is the final answer clearly stated?
- Are there any logical gaps in the reasoning?

#### 3. Rigor and Clarity Evaluation
- Is the reasoning rigorous and mathematically sound?
- Are assumptions explicitly stated and justified?
- Is the mathematical work shown explicitly (not just results)?
- Would another mathematician be able to verify and follow this solution?
- Are edge cases and special conditions properly addressed?

#### 4. Internal Consistency
- Do all the steps fit together coherently?
- Are there any contradictions in the reasoning?
- Does the final answer make sense given the problem constraints?

#### 5. Decision Making (Based on Process Quality)
**You must decide:** Should this solution be accepted based on its mathematical rigor and completeness?

**Accept (use `<next_step>end</next_step>`)** if:
- All mathematical steps are correct and verifiable
- The reasoning is complete, rigorous, and logically sound
- No arithmetic or logical errors are present
- The solution methodology is appropriate for the problem
- All work is clearly shown and justified

**Continue (use `<next_step>continue</next_step>`)** if:
- Mathematical errors or logical flaws exist
- Important steps are missing or inadequately explained
- The reasoning has gaps or lacks rigor
- Calculations are not shown or verified
- The approach seems inappropriate or fundamentally flawed
- Assumptions are not justified

### Your Feedback Format:
1. **Summary Assessment:** Brief overall evaluation
2. **Mathematical Rigor:** Evaluate correctness of all steps
3. **Logical Soundness:** Assess the reasoning flow and completeness
4. **Specific Issues:** Point out exact problems (if any)
5. **Suggestions:** Concrete improvement recommendations (if continuing)
6. **Decision:** Your final verdict with appropriate next_step tag

**Important:** Since you don't have ground truth, focus on the PROCESS quality, not whether the final answer is "correct". A well-reasoned, rigorous solution should be accepted even if you cannot verify the final numerical answer against a reference.

Please provide your thorough evaluation and decision:
"""

            verifier_messages = [
                {"role": "system", "content": verifier_system_prompt},
                {"role": "user", "content": processed_response}
            ]

            print(f"\n🔍 [VERIFIER] Calling {verifier} - Rollout {rollout + 1}, Round {r + 1}")
            print(f"📋 [VERIFIER] Evaluating LLM response ({len(response_str)} chars)")
            feedback = await client.chat_completion(
                model=verifier,
                messages=verifier_messages
            )
            # Ensure feedback is a string before processing
            feedback_str = str(feedback) if feedback else ""
            print(f"✅ [VERIFIER] Feedback received ({len(feedback_str)} chars)")
            print(f"📄 [VERIFIER] Full Feedback:")
            print(f"{feedback_str}")
            raw_feedback, next_step = extract_next_step(feedback_str)
            print(f"🎯 [VERIFIER] Decision: {next_step}")
            print(f"💬 [VERIFIER] Raw feedback (after extracting next_step):")
            print(f"{raw_feedback}")
            history.append(raw_feedback)

            if next_step == "end":
                is_correct = True
                break
        
        all_histories.append(
            {
                "history": history,
                "correctness": is_correct
            }
        )
        # If the last rollout is correct, no more tries.
        if is_correct:
            print(f"🎉 [TASK] Rollout {rollout + 1} succeeded! Stopping early.")
            break
        else:
            print(f"❌ [TASK] Rollout {rollout + 1} failed. Continuing to next rollout.")
    
    print(f"\n📈 [TASK] All rollouts completed. Total: {len(all_histories)}")
    print(f"🏆 [TASK] Success rate: {sum(1 for h in all_histories if h['correctness'])}/{len(all_histories)}")
    
    # For each rollout, generate memory using an object from Critic
    # Create a Critic instance for memory extraction
    memory_critic = Critic(critic, temperature=0)
    
    # Process ALL rollouts to generate memories, not just the best one
    memory_data_list = []
    
    for rollout_idx, trajectory in enumerate(all_histories):
        # Extract the final response (last LLM generation in the history)
        history = trajectory["history"]

        # Get the last LLM response (LLM responses are at even indices)
        final_response = history[-2]
        
        if final_response:
            # Generate memory using critic for each rollout
            is_success = trajectory["correctness"]
            print(f"\n🧩 [CRITIC] Processing Rollout {rollout_idx + 1} (Success: {is_success})")
            print(f"📊 [CRITIC] Final response length: {len(final_response)} chars")
            print(f"🎯 [CRITIC] Ground truth: {ground_truth}")
            response = await memory_critic.feedback(
                query=query, 
                all_histories=[trajectory],  # Pass current trajectory for context
                ground_truth=ground_truth,
                is_success=is_success
            )
            print(f"✅ [CRITIC] Feedback generated ({len(str(response))} chars)")
            print(f"📄 [CRITIC] Full Critic Feedback:")
            print(f"{str(response)}")
            memory_data = memory_critic.extract(
                response=response,
                is_success=is_success,
                problem_category=("retrosynthesis" if task_type == "retro" else "math"),  # category by task
                priority=None  # Let critic auto-determine
            )
            print(f"🗂️ [CRITIC] Memory extraction result: {type(memory_data).__name__}")
            if memory_data:
                print(f"📝 [CRITIC] Memory keys: {list(memory_data.keys()) if isinstance(memory_data, dict) else 'Not a dict'}")
            else:
                print("⚠️ [CRITIC] No memory data extracted!")
            
            if memory_data:
                memory_data_list.append({
                    "rollout_idx": rollout_idx,
                    "memory_data": memory_data,
                    "is_success": is_success
                })
    
    # Find best trajectory for result reporting
    # Since we break immediately on success, the last trajectory is always the best
    # (either successful, or the final attempt if all failed)
    best_trajectory = all_histories[-1] if all_histories else None
            
    return {
        "query": query,
        "ground_truth": ground_truth,
        "all_histories": all_histories,
        "best_trajectory": best_trajectory,
        "memory_data_list": memory_data_list,  # Now contains all rollout memories
        "success": any(h["correctness"] for h in all_histories)
    }
    

def extract_next_step(feedback: str):
    """
    Extract the next step from the critic's feedback. The critic will wrap the next step (continue|end) 
    with '<next_step></next_step>' tag at the end of the feedback.
    This function will extract the raw comments without the <next_step> at the end and the wrapped content
    in <next_step>.
    """
    import re
    
    # Pattern to match <next_step>content</next_step>
    pattern = r'<next_step>(.*?)</next_step>'
    
    # Search for the next_step tag
    match = re.search(pattern, feedback, re.DOTALL)
    
    if match:
        # Extract the next step content
        next_step = match.group(1).strip()
        
        # Remove the entire <next_step>...</next_step> section from feedback
        raw_comments = re.sub(pattern, '', feedback, flags=re.DOTALL).strip()
        
        return raw_comments, next_step
    else:
        # If no next_step tag found, return original feedback and default to continue
        return feedback.strip(), "continue"

def extract_answer_from_response(response):
    """Extract the final answer from LLM response"""
    # Simple extraction - look for common answer patterns
    import re
    
    # Look for patterns like "The answer is X" or "Answer: X" or final number
    patterns = [
        r"(?:the answer is|answer is|answer:|final answer is|final answer:)\s*([+-]?\d+(?:\.\d+)?)",
        r"(?:therefore|thus|so|hence),?\s*(?:the answer is)?\s*([+-]?\d+(?:\.\d+)?)",
        r"([+-]?\d+(?:\.\d+)?)(?:\s*$|\s*\.$)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response.lower())
        if match:
            return match.group(1)
    
    # If no pattern found, try to extract the last number in the response
    numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', response)
    return numbers[-1] if numbers else "0"

async def process_batch(
    batch,
    client,
    llm_model,
    verifier_model,
    critic_model,
    exp_lib,
    use_updater=True,
    updater=None,
    max_rounds=5,
    max_rollouts=3,
    task_type: str = "math",
    llm_system_prompt_path: str | None = None,
    verifier_prompt_path: str | None = None,
    with_gt: bool = True,
):
    """
    Process a batch of tasks in parallel using reject sampling.
    
    Args:
        batch: List of samples, each with 'question' and 'answer' fields
        client: RateLimitedLLMClient instance
        llm_model: LLM model name for generation
        verifier_model: Verifier model name for feedback during reject sampling
        critic_model: Critic model name for memory extraction
        exp_lib: ExpLib instance for memory management
        use_updater: Whether to use updater for intelligent memory management
        updater: Updater instance (required if use_updater=True)
        max_rounds: Maximum rounds per rollout
        max_rollouts: Maximum rollouts per task
    
    Returns:
        List of results for each task in the batch
    """
    
    # Create tasks for parallel processing
    tasks = []
    for sample in batch:
        query = sample['question']
        if task_type == "retro":
            ground_truth = str(sample['answer']).strip()
        else:
            ground_truth = extract_answer(sample['answer'])

        task = process_task(
            query,
            ground_truth,
            client,
            llm_model,
            verifier_model,
            critic_model,
            max_rounds,
            max_rollouts,
            task_type,
            llm_system_prompt_path,
            verifier_prompt_path,
            with_gt,
        )
        tasks.append(task)
    
    # Execute all tasks in parallel
    print(f"🚀 [BATCH] Starting {len(tasks)} tasks in parallel...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print(f"✅ [BATCH] All tasks completed. Processing results...")
    
    # Process results and update memory
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"❌ [BATCH] Task {i} failed: {result}")
            print(f"🔍 [BATCH] Exception type: {type(result).__name__}")
            processed_results.append({
                "success": False,
                "error": str(result),
                "sample": batch[i]
            })
        else:
            # Update memory for ALL rollouts if we have valid memory data
            if result.get("memory_data_list"):
                print(f"🧠 [MEMORY] Task {i}: Processing {len(result['memory_data_list'])} memory entries")
                try:
                    total_memories_added = 0
                    for memory_entry in result["memory_data_list"]:
                        memory_data = memory_entry["memory_data"]
                        is_success = memory_entry["is_success"]
                        rollout_idx = memory_entry["rollout_idx"]
                        
                        print(f"📝 [MEMORY] Processing rollout {rollout_idx} memory (success: {is_success})")
                        print(f"🔍 [MEMORY] Memory data type: {type(memory_data).__name__}")
                        
                        if use_updater and updater:
                            # 使用 Updater 进行智能记忆管理
                            # 为 updater 添加记忆元数据
                            enhanced_memory_data = memory_data.copy()
                            enhanced_memory_data["memory_metadata"] = {
                                "suggested_type": "golden" if is_success else "warning",
                                "problem_category": ("retrosynthesis" if task_type == "retro" else "math"),
                                "priority": 5 if is_success else 4
                            }
                            
                            print(f"🧠 [UPDATER] Calling updater for analysis...")
                            await updater.analyse_and_update(enhanced_memory_data)
                            print(f"Task {i}, Rollout {rollout_idx}: Updater processed {'success' if is_success else 'failure'} memory")
                            
                        else:
                            # 直接添加到分区（快速但可能冗余）
                            memory_type = "golden" if is_success else "warning"
                            priority = 5 if is_success else 4
                            
                            print(f"💾 [DIRECT_ADD] Adding {memory_type} memory with priority {priority}")
                            exp_lib.add_memory(
                                data=memory_data,
                                memory_type=memory_type,
                                problem_category=("retrosynthesis" if task_type == "retro" else "math"),
                                priority=priority
                            )
                            print(f"Task {i}, Rollout {rollout_idx}: Direct add {memory_type} memory")
                        
                        total_memories_added += 1
                    
                    print(f"✅ [MEMORY] Task {i}: {total_memories_added} memories processed successfully")
                    
                except Exception as e:
                    print(f"❌ [MEMORY] Task {i}: Memory update failed: {e}")
                    import traceback
                    print(f"🔍 [MEMORY] Full traceback:")
                    traceback.print_exc()
            else:
                print(f"⚠️ [MEMORY] Task {i}: No memory data to process")
            
            processed_results.append(result)
    
    return processed_results

def load_data(data_path):
    """Load data from JSONL file"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def extract_answer(answer_text):
    """Extract numerical answer from ground truth text"""
    import re
    
    # Look for number at the end or after common answer indicators
    patterns = [
        r"(?:answer is|answer:|the answer is)\s*([+-]?\d+(?:\.\d+)?)",
        r"([+-]?\d+(?:\.\d+)?)(?:\s*$|\s*\.$)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, str(answer_text).lower())
        if match:
            return match.group(1)
    
    # Fallback: extract first number found
    numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', str(answer_text))
    return numbers[0] if numbers else "0"

async def main():
    """Main training workflow"""
    import argparse
    import time
    from datetime import datetime
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Reject Sampling Training Workflow (Unified: math | retro)")
    parser.add_argument('--llm_model', default='openai/gpt-5', help='LLM model for generation')
    parser.add_argument('--verifier_model', default='openai/gpt-5', help='Verifier model for feedback during reject sampling')
    parser.add_argument('--critic_model', default='openai/gpt-5', help='Critic model for memory extraction')
    parser.add_argument('--updater_model', default='openai/gpt-5', help='Updater model for memory updates')
    
    parser.add_argument('--task_type', choices=['math', 'retro'], default='retro', help='Task type: math or retro (single-step retrosynthesis)')
    # Defaults depend on task_type; we'll adjust after parsing
    parser.add_argument('--data_path', default="data/uspto50k/train.jsonl", help='Path to training data (JSONL)')
    parser.add_argument('--load_path', default='./exps/retro_gpt.json', help='Path to load existing memory')
    parser.add_argument('--store_path', default='./exps/retro_gpt.json', help='Path to save updated memory')
    parser.add_argument('--exp_path', default='./exps', help='Experiment directory')
    
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for parallel processing')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to process')
    parser.add_argument('--max_rounds', type=int, default=5, help='Maximum rounds per rollout')
    parser.add_argument('--max_rollouts', type=int, default=3, help='Maximum rollouts per task')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    
    # Memory management strategy
    parser.add_argument('--use_updater', action='store_true', default=True, help='Use intelligent updater for memory management')
    parser.add_argument('--direct_add', action='store_true', help='Use direct memory addition (faster but may create duplicates)')
    
    # Ground truth control
    parser.add_argument('--with_gt', type=lambda x: x.lower() == 'true', default=True, help='Provide ground truth to verifier for evaluation (True/False, default: False)')
    
    # Retro prompts are optional; used when task_type=retro (fallback built-ins used if files missing)
    parser.add_argument('--llm_system_prompt', default='prompts/llm_system_prompt_retro.txt', help='(retro) Path to LLM system prompt')
    parser.add_argument('--verifier_prompt', default='prompts/verifier_retro.txt', help='(retro) Path to verifier system prompt')

    args = parser.parse_args()
    
    # Resolve memory strategy
    if args.direct_add:
        args.use_updater = False
    
    print("🚀 Starting Reject Sampling Training Workflow")
    print(f"📋 Configuration:")
    print(f"   LLM Model: {args.llm_model}")
    print(f"   Verifier Model: {args.verifier_model}")
    print(f"   Critic Model: {args.critic_model}")
    print(f"   Task Type: {args.task_type}")
    if args.use_updater:
        print(f"   Updater Model: {args.updater_model}")
    print(f"   Memory Strategy: {'Intelligent Updater' if args.use_updater else 'Direct Addition'}")
    print(f"   Ground Truth for Verifier: {'Yes' if args.with_gt else 'No (Process-based evaluation only)'}")
    # Set default data_path based on task
    if args.data_path is None:
        args.data_path = './data/uspto50k/test.jsonl' if args.task_type == 'retro' else './data/AIME/aime_full.jsonl'
    print(f"   Data Path: {args.data_path}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Max Rounds: {args.max_rounds}")
    print(f"   Max Rollouts: {args.max_rollouts}")
    if args.task_type == 'retro':
        print(f"   LLM System Prompt (retro): {args.llm_system_prompt}")
        print(f"   Verifier Prompt (retro): {args.verifier_prompt}")
    
    # Load data
    print("\n📚 Loading data...")
    data = load_data(args.data_path)
    if args.max_samples:
        data = data[:args.max_samples]
    print(f"   Loaded {len(data)} samples")
    print(f"🔍 [DEBUG] First sample preview:")
    if data:
        first_sample = data[0]
        if args.task_type == 'retro':
            print(f"   Product (question): {first_sample.get('question', 'N/A')[:100]}...")
            print(f"   Canonical Reactants (answer): {first_sample.get('answer', 'N/A')}")
        else:
            print(f"   Question: {first_sample.get('question', 'N/A')[:100]}...")
            print(f"   Answer: {first_sample.get('answer', 'N/A')}")
            ground_truth_preview = extract_answer(first_sample.get('answer', ''))
            print(f"   Extracted GT: {ground_truth_preview}")
    
    # Initialize components
    print("\n🔧 Initializing components...")
    client = RateLimitedLLMClient()
    print(f"✅ [DEBUG] RateLimitedLLMClient initialized")
    exp_lib = ExpLib(path=args.exp_path, load_path=args.load_path, save_path=args.store_path)
    print(f"✅ [DEBUG] ExpLib initialized")
    
    # Initialize updater if needed
    updater = None
    if args.use_updater:
        from updater import Updater
        updater = Updater(args.updater_model, exp_lib)
        print("   Updater initialized for intelligent memory management")
    
    print("   Components initialized successfully")
    print(f"   Loaded memory statistics: {exp_lib.get_statistics()}")
    
    # Training loop
    total_success = 0
    total_samples = 0
    all_failed_indices = []  # Track all failed sample indices
    
    for epoch in range(args.epochs):
        print(f"\n📈 Starting Epoch {epoch + 1}/{args.epochs}")
        epoch_start_time = time.time()
        
        epoch_success = 0
        epoch_samples = 0
        epoch_failed_indices = []  # Track failed indices for this epoch
        
        # Process data in batches
        for i in range(0, len(data), args.batch_size):
            batch = data[i:i + args.batch_size]
            batch_idx = i // args.batch_size + 1
            total_batches = (len(data) + args.batch_size - 1) // args.batch_size
            
            print(f"\n🔄 Processing Batch {batch_idx}/{total_batches} (samples {i+1}-{min(i+args.batch_size, len(data))})")
            batch_start_time = time.time()
            
            try:
                results = await process_batch(
                    batch,
                    client,
                    args.llm_model,
                    args.verifier_model,
                    args.critic_model,
                    exp_lib,
                    args.use_updater,
                    updater,
                    args.max_rounds,
                    args.max_rollouts,
                    args.task_type,
                    args.llm_system_prompt if args.task_type == 'retro' else None,
                    args.verifier_prompt if args.task_type == 'retro' else None,
                    args.with_gt,
                )
                
                # Calculate batch statistics
                batch_success = sum(1 for r in results if r.get("success", False))
                batch_samples = len(results)
                
                # Track failed indices for this batch
                batch_failed_indices = []
                for j, r in enumerate(results):
                    if not r.get("success", False):
                        # Calculate the global index for this sample
                        global_index = batch_idx * args.batch_size + j
                        batch_failed_indices.append(global_index)
                        epoch_failed_indices.append(global_index)
                
                epoch_success += batch_success
                epoch_samples += batch_samples
                
                batch_time = time.time() - batch_start_time
                print(f"   ✅ Batch {batch_idx} completed in {batch_time:.2f}s")
                print(f"   📊 Batch success rate: {batch_success}/{batch_samples} ({batch_success/batch_samples*100:.1f}%)")
                
                # Save memory after each batch
                print(f"💾 [SAVE] Attempting to save memory to {args.store_path}")
                try:
                    exp_lib.save(args.store_path)
                    print(f"✅ [SAVE] Memory saved successfully to {args.store_path}")
                    # Verify the save
                    import os
                    if os.path.exists(args.store_path):
                        file_size = os.path.getsize(args.store_path)
                        print(f"📊 [SAVE] File size: {file_size} bytes")
                    else:
                        print(f"⚠️ [SAVE] Warning: File doesn't exist after save!")
                except Exception as save_error:
                    print(f"❌ [SAVE] Save failed: {save_error}")
                    import traceback
                    traceback.print_exc()
                
                # Brief pause between batches
                if i + args.batch_size < len(data):
                    print("   ⏸️  Pausing 5 seconds between batches...")
                    await asyncio.sleep(5)
                    
            except Exception as e:
                print(f"   ❌ Batch {batch_idx} failed: {e}")
                continue
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        epoch_accuracy = epoch_success / epoch_samples if epoch_samples > 0 else 0
        all_failed_indices.extend(epoch_failed_indices)  # Add failed indices from this epoch
        
        total_success += epoch_success
        total_samples += epoch_samples
        
        print(f"\n🎯 Epoch {epoch + 1} Summary:")
        print(f"   Success Rate: {epoch_success}/{epoch_samples} ({epoch_accuracy*100:.1f}%)")
        print(f"   Time: {epoch_time:.2f}s")
        
        # Get updated memory statistics
        memory_stats = exp_lib.get_statistics()
        print(f"   Memory Size: {memory_stats['total_memories']} entries")
        print(f"   Memory Distribution: Golden={memory_stats['by_type']['golden']}, Warning={memory_stats['by_type']['warning']}, Mixed={memory_stats['by_type']['mixed']}")
    
    # Final summary
    final_accuracy = total_success / total_samples if total_samples > 0 else 0
    final_stats = exp_lib.get_statistics()
    print(f"\n🏁 Training Complete!")
    print(f"   Overall Success Rate: {total_success}/{total_samples} ({final_accuracy*100:.1f}%)")
    print(f"   Failed Sample Indices: {all_failed_indices}")
    print(f"   Final Memory Size: {final_stats['total_memories']} entries")
    print(f"   Final Distribution: {final_stats['by_type']}")
    print(f"   Memory saved to: {args.store_path}")
    
    return final_accuracy, all_failed_indices

if __name__ == "__main__":
    accuracy, failed_indices = asyncio.run(main())
    print(f"\n📊 Final Results:")
    print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   Failed indices: {failed_indices}")