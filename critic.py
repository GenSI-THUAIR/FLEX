import os
import re
import json
import asyncio
from typing import Dict, Optional
from rate_limiter import RateLimitedLLMClient

class Critic:
    """Critic组件: 对过程和答案进行semantic feedback 提取事实和方法论"""
    ##TODO: refine the prompt template; more robust parsing
    ##TODO: how to critique on the chosen exp?
    ##      从而在exp refinement和performance improvement之间建立连接

    def __init__(self, model, temperature=0):
        self.model = model
        self.temperature = temperature
        self.llm_client = RateLimitedLLMClient()

    async def llm(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        return await self.llm_client.chat_completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
    
    async def feedback(self, query: str, all_histories: list, ground_truth: str, is_success: bool) -> str:
        """
        Analyzes the problem-solving histories and generates structured feedback for memory extraction.
        It uses different prompts based on whether the final outcome was successful or not.
        Now includes memory type determination for the new partitioned memory system.
        """
        # Prepare a string representation of all histories for the prompt
        history_str = ""
        for i, rollout in enumerate(all_histories):
            history_str += f"--- Rollout {i+1} (Success: {rollout.get('correctness', False)}) ---\n"
            # The history is a list of strings: [llm_gen_1, verifier_fb_1, llm_gen_2, ...]
            for j, turn in enumerate(rollout.get("history", [])):
                if j % 2 == 0:
                    history_str += f"Round {j//2 + 1} - LLM Generation:\n{turn}\n\n"
                else:
                    history_str += f"Round {j//2 + 1} - Verifier Feedback:\n{turn}\n\n"
            history_str += "---\n\n"

        # Select the correct prompt based on the is_success flag
        if is_success:
            # Prompt for successful rollouts: Focus on generalizing the "golden path"
            prompt = f"""
You are a professional expert analyzing a solver's successful problem-solving process. Your goal is to extract a generalizable "recipe for success" from this experience.

**Original Query:** {query}
**Correct Answer:** {ground_truth}
**Full Problem-Solving History:**
{history_str}

The final attempt was successful. Please provide your evaluation with a focus on **refining and generalizing the successful strategy**.

---
### 1. Method Summary (The "Golden Path")

**Your Goal:** To abstract the successful approach into a high-level, reusable algorithm for this entire class of problems. This is the most important piece of memory.

**Instructions:**
- First, categorize the problem type (e.g., "Geometry problem involving optimization", "Multi-step algebraic simplification").
- Then, outline the successful workflow as a series of general steps.

**Example Output:**
"**Problem Category:** Multi-step algebraic simplification.
**General Method:**
1. Identify the core structure of the expression (e.g., nested fractions, repeated terms).
2. Use substitution to simplify the most complex or repeated part first.
3. Sequentially substitute back and simplify at each step, avoiding expanding large expressions until the end.
4. Perform a final verification of the simplified expression."

---
### 2. General Rules and Best Practices

**Your Goal:** To capture the key insights, "tricks," or crucial good habits that were pivotal for success and turn them into actionable advice.

**Instructions:**
- Extract positive, generalizable principles.
- For each rule, provide a brief, specific snippet from the trajectory that demonstrates the rule in action.

**Example Output:**
- **Rule:** "When faced with a high-degree polynomial, always check for simple integer roots (like -1, 0, 1) before attempting complex factorization methods."
  - **Example from trajectory:** "The solver correctly tested `x=1` and found it was a root, which immediately simplified the problem."

---
### 3. Revised Trajectory with Annotations

**Your Goal:** To reinforce *why* certain steps were good, creating a positive feedback loop for the agent.

**Instructions:**
- Take the **final successful solving process**.
- Add `<comments>...</comments>` tags around **excellent or pivotal** parts.
- Inside the comments, explain *why* this step represents a good practice or a key insight.

**Example Output:**
"...the solver wrote `let y = x^2`. <comments>This is an excellent use of substitution. It transforms a complex quartic equation into a simple quadratic one, which is much easier to solve. This technique is applicable to any equation with symmetric powers.</comments>"

---
Please format your complete response as a single JSON block inside ```json ... ```:

```json
{{
    "method": "A complete methodology description starting with problem categorization and including the general steps as a single comprehensive string.",
    "rules": [
        {{
            "rule": "A general, actionable guideline or best practice derived from the success.",
            "examples": ["Specific snippet from the successful process that illustrates the rule."]
        }}
    ],
    "revised_trajectory": "The final successful solving process with <comments> highlighting excellent parts."
}}
```"""
        else:
            # Prompt for failed rollouts: Focus on root cause analysis
            prompt = f"""
You are a professional expert diagnosing a solver's failed problem-solving process. Your goal is to perform a deep root cause analysis to create a "what not to do" guide and prevent future failures.

**Original Query:** {query}
**Correct Answer:** {ground_truth}
**Full Problem-Solving History (all attempts failed):**
{history_str}

All attempts were unsuccessful. Please provide your evaluation with a focus on **diagnosing the root cause of the failure**.

---
### 1. Root Cause Analysis and General Rules

**Your Goal:** To identify the single most critical error and other mistakes, turning them into strong, memorable warnings for the future.

**Instructions:**
- Identify the core reason for the failure (e.g., conceptual misunderstanding, logical fallacy, calculation trap).
- Formulate this as a primary "what to avoid" rule.
- Identify any other misleading paths or cognitive traps the solver fell into.
- For each rule, provide a specific snippet from a failed process that clearly shows the error.

**Example Output:**
- **Rule:** "CRITICAL: Never assume a variable is non-zero when dividing. Always check for the case where the variable could be zero."
  - **Example from trajectory:** "The solver divided the equation by `x-2` without considering that `x=2` might be a valid solution, thus losing a correct answer."
- **Rule:** "Avoid premature expansion of complex expressions. Look for structural simplifications first."
  - **Example from trajectory:** "The solver immediately expanded `(a+b+c)^3`, leading to a calculation mess. The structure allowed for a simpler substitution."

---
### 2. Revised Trajectory with Annotations

**Your Goal:** To provide a clear, step-by-step correction of the final failed attempt, explaining not just *what* was wrong but *why*.

**Instructions:**
- Take the **last failed solving process**.
- Add `<comments>...</comments>` tags around **problematic** parts.
- Inside the comments, explain the error, its root cause, and the correct approach.

**Example Output:**
"...the solver stated that `sqrt(a^2 + b^2) = a + b`. <comments>This is a fundamental algebraic error. The square root of a sum is not the sum of the square roots. The correct approach is to leave the expression as is or look for other ways to simplify the entire equation.</comments>"

---
### 3. Method Summary (Corrected Approach)

**Your Goal:** To outline the correct high-level strategy that *should have been* used, providing a clear path forward.

**Instructions:**
- First, categorize the problem type.
- Then, describe the correct general method or workflow that would have led to success.

**Example Output:**
"**Problem Category:** Equation Solving with Radicals.
**Correct General Method:**
1. Isolate the radical term on one side of the equation.
2. Square both sides to eliminate the radical.
3. Solve the resulting polynomial equation.
4. CRUCIAL: Check all potential solutions in the original equation to eliminate extraneous roots introduced by squaring."

---
Please format your complete response as a single JSON block inside ```json ... ```:

```json
{{
    "method": "A complete description of the corrected methodology that should be used for this problem category, including problem categorization and general steps as a single comprehensive string.",
    "rules": [
        {{
            "rule": "A critical 'what to avoid' guideline based on the root cause of failure.",
            "examples": ["Specific snippet from a failed process that shows the error."]
        }}
    ],
    "revised_trajectory": "The last failed solving process with <comments> explaining the errors and corrections."
}}
```
"""

        response = await self.llm(prompt)
        return response
    
    def extract(self, response, is_success: bool = None, problem_category: str = "", priority: int = None):
        """
        Extract structured data from critic response and prepare for partitioned memory storage.
        
        Args:
            response: LLM response containing the feedback
            is_success: Whether the problem-solving was successful (determines memory type)
            problem_category: Category of the problem (e.g., "geometry", "algebra")
            priority: Priority level for the memory (1-5, will auto-determine if not provided)
        """
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    
        if json_match:
            json_str = json_match.group(1)
            try:
                data = json.loads(json_str)
                print(f"Type of data: {type(data).__name__}")
                
                # Enhance data with partitioned memory metadata
                enhanced_data = {
                    **data,  # Original critic extraction: method, rules, revised_trajectory
                    "memory_metadata": {
                        "suggested_type": self._determine_memory_type(data, is_success),
                        "problem_category": problem_category,
                        "priority": priority or self._determine_priority(data, is_success),
                        "is_success": is_success
                    }
                }

                print("[CRITIC] Extraction finished successfully!")
                
                return enhanced_data
                
            except json.JSONDecodeError as e:
                print(f"Critic JSON解析错误: {e}")
                print(f"原始JSON字符串: {json_str}")
                # 返回空字典，避免程序崩溃
                return {}

        return {}
    
    def _determine_memory_type(self, data: Dict, is_success: bool) -> str:
        """
        Determine the appropriate memory partition based on success and content analysis.
        
        Returns:
            "golden": For successful experiences with clear best practices
            "warning": For failed experiences with clear error patterns  
            "mixed": For experiences showing failure-to-success transitions
        """
        print("Enter _determine_memory_type")
        if is_success is None:
            return "mixed"  # Default when success status is unclear
            
        # Analyze content for mixed signals (failure-to-success narratives)
        # Method field is now a simple string
        method_text = str(data.get("method", ""))
        trajectory_content = str(data.get("revised_trajectory", ""))
        
        # Handle rules - they should be a list of dicts
        rules_content = ""
        rules = data.get("rules", [])
        if isinstance(rules, list):
            rules_content = " ".join([str(rule.get("rule", "")) if isinstance(rule, dict) else str(rule) for rule in rules])
        
        content_text = " ".join([
            method_text,
            trajectory_content,
            rules_content
        ]).lower()
        
        # Look for indicators of learning from failure
        mixed_indicators = [
            "initially", "at first", "originally", "wrong", "error", "corrected",
            "revised", "learned", "mistake", "improved", "then realized", "后来",
            "最初", "错误", "修正", "改进", "发现问题"
        ]
        
        has_mixed_narrative = any(indicator in content_text for indicator in mixed_indicators)

        print("_determine_memory_type finished successfully!")
        
        if is_success and has_mixed_narrative:
            return "mixed"  # Success that learned from failure
        elif is_success:
            return "golden"  # Pure success experience
        else:
            return "warning"  # Failed experience for warnings
    
    def _determine_priority(self, data: Dict, is_success: bool) -> int:
        """
        Auto-determine priority based on content richness and success status.
        
        Returns:
            1-5 priority level (5 being highest)
        """
        print("Enter _determine_priority")
        base_priority = 3  # Default medium priority
        
        # Successful experiences get higher base priority
        if is_success:
            base_priority = 4
        else:
            base_priority = 3
            
        # Increase priority for rich content
        rules_count = len(data.get("rules", []))
        if rules_count >= 3:
            base_priority += 1
        elif rules_count >= 2:
            base_priority += 0  # No change
        else:
            base_priority -= 1
            
        # Increase priority for detailed method descriptions
        # Method field is now a simple string
        method_content = str(data.get("method", ""))
        method_length = len(method_content)
            
        if method_length > 200:
            base_priority += 1
        elif method_length < 50:
            base_priority -= 1
            
        print("_determine_priority finished succesfully!")
        # Clamp to valid range
        return max(1, min(5, base_priority))