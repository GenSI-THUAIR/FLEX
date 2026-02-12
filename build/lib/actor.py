import os
import re
import asyncio
import yaml
import sys
import io
import threading
from contextlib import redirect_stdout
from smolagents import ToolCallingAgent, OpenAIServerModel, CodeAgent
from memory_retriever import agentic_memory_retrieval

def strip_ansi_colors(text):
    """Remove ANSI color codes from text"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

class TeeOutput:
    """A file-like object that writes to both stdout and a file"""
    def __init__(self, file_handle):
        self.file = file_handle
        self.stdout = sys.stdout
    
    def write(self, text):
        # Write to both stdout (with colors) and file (without colors)
        self.stdout.write(text)
        self.stdout.flush()
        # Strip ANSI colors before writing to file
        clean_text = strip_ansi_colors(text)
        self.file.write(clean_text)
        self.file.flush()
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()

class Actor:
    """Actor组件: 对给定问题进行思考解答 可调用ExpLib 给出推理过程和答案"""
    def __init__(self, model, memory_retrieval_tool=None, temperature=0, retrieve=True, log_path=None, setting="aime"):
        self.model = model
        self.temperature = temperature
        self.log_path = log_path
        self.retrieve = retrieve
        self.setting = setting
        # lock to avoid concurrent replay stdout mix when multiple coroutines call act
        self._log_lock = threading.Lock()

        # Load the system prompt
        if retrieve and setting == "aime":
            yaml_path = "prompts/actor_aime.yaml"
        elif retrieve and setting == "retro":
            yaml_path = "prompts/actor_retro.yaml"
        else:
            yaml_path = "prompts/actor.yaml"
        try:
            if os.path.exists(yaml_path):
                with open(yaml_path, "r", encoding="utf-8") as f:
                    prompt_template = yaml.safe_load(f)
        except Exception as e:
            print(f"[Actor] Failed to load system prompt from {yaml_path}: {e}")
        
        # Construct the tool
        if retrieve:
            tools = [memory_retrieval_tool]
        else:
            tools = []

        # Initialize smolagents ToolCallingAgent using OpenAI-compatible endpoint from env
        self.agent = CodeAgent(
            model=OpenAIServerModel(
                model_id=self.model,
                api_key=os.getenv("API_KEY", ""),
                api_base=os.getenv("BASE_URL", ""),
                temperature=self.temperature,
            ),
            tools=tools,
            max_steps=50,
            additional_authorized_imports=["memory"],
            prompt_templates=prompt_template,
        )

    
    async def act(self, query) -> str:
        """
        Process a problem and provide a solution.
        
        Args:
            query: The problem to solve
        
        Returns:
            The solution process and final answer
        """
        # The memory retrieval tools are now available as agent tools
        if self.retrieve and self.setting == "aime":
            prompt = f"""
            Problem to solve: {query}
            
            Please think step by step, use the available memory tools as needed following the memory retrieval strategy described in your system prompt, and provide the answer to this problem.
            
            Remember to provide your final answer in the format: "The answer is <your result>"
            """
        elif self.retrieve and self.setting == "retro":
            prompt = f"""
<SMILES> {query} </SMILES>
You are given the product's SMILES string. Predict the SMILES string(s) of the reactant(s) for a **single-step** retrosynthesis using your knowledge of chemical retrosynthesis.
Please reason step by step, and provide your final answer enclosed within the final_answer() tool.
**Important:** Only include SMILES notation inside the final_answer tool. If there is more than one reactant, separate them with a dot ('.'). Do not add any explanations or comments there.
Example:
final_answer("CCOc1ccc(Oc2ncnc3c2cnn3C2CCNCC2)c(F)c1.C(=O)(Cl)OC1CCCC1") # Put your final answer in the final_answer tool! 
Now it's your turn.
"""
        else:
            prompt = query
        
        # Run agent normally - output goes to terminal
        result = await asyncio.to_thread(self.agent.run, prompt)
        
        # If log_path is provided, save replay output to file
        if self.log_path:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True) if os.path.dirname(self.log_path) else None
                
                # Capture replay output
                replay_buffer = io.StringIO()
                with self._log_lock:
                    with redirect_stdout(replay_buffer):
                        try:
                            self.agent.replay()
                        except Exception as e:
                            print(f"[Actor] replay error: {e}")
                
                # Write replay output to log file
                with open(self.log_path, "w", encoding="utf-8") as log_file:
                    replay_output = strip_ansi_colors(replay_buffer.getvalue())
                    log_file.write(replay_output)
                    
            except Exception as e:
                print(f"[Actor] failed to write log to {self.log_path}: {e}", file=sys.stderr)

        return result
    
    def extract_traj(self, response):
        if not response or not response.strip():
            return "", ""
        
        lines = response.strip().split('\n')
        if not lines:
            return "", ""
        
        # 最后一行包含答案
        last_line = lines[-1] if lines else ""
        # 使用正则表达式提取答案
        answer_match = re.search(r"The answer is (\d+)", last_line)
        ans = answer_match.group(1).strip() if answer_match else ""
        
        # trajectory是除最后一行外的所有内容
        if len(lines) > 1:
            traj = '\n'.join(lines[:-1]).strip()
        else:
            traj = ""

        return traj, ans

    async def retrieve_exp(self, query, retrieve=False, exp_lib = None):
        ##TODO: retrieve relevant exp from exp_lib by LLM？
        ## 可以考虑先用BM25粗筛，再用LLM精筛
        ##TODO: the prompt template requires refinement
        if exp_lib == None:
            return ""

        if retrieve==False:
            return exp_lib

        # 获取所有methods
        library = exp_lib.get_library()
        if not library:
            return ""
        
        # 构建methods列表供LLM选择
        methods_list = []
        for i, entry in enumerate(library):
            method = entry.get("method", "")
            if method:
                methods_list.append(f"Method {i}: {method}")
        
        methods_str = "\n".join(methods_list)
        
        prompt = f"""
        You are a general problem plans seeking assistant who excels at seeking the most appropriate plans for a task among different candidate plans. You need to select the most relevant methods from an experience library to help solve a given problem.

        Problem to solve:
        {query}
        
        Available methods in the experience library:
        {methods_str}
        
        Please select up to 5 methods that are most relevant to solving this problem. Consider:
        - Which methods address similar problem types
        - Which approaches would be most helpful for this specific problem
        - Select methods that complement each other if possible
        
        Return your selection in the following JSON format:
        ```json
        [method_index_1, method_index_2, method_index_3, ...]
        ```
        
        For example, if methods 0, 2, and 4 are most relevant, return:
        ```json
        [0, 2, 4]
        ```
        
        If no methods are relevant, return an empty array:
        ```json
        []
        ```
        """
        
        # Run ToolCallingAgent synchronously in a worker thread
        result = await asyncio.to_thread(self.agent.run, prompt)
        return self.extract_exp(result, exp_lib)

    def extract_exp(self, response, exp_lib):
        ##从LLM响应中提取method indices并格式化返回
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        
        if not json_match:
            return ""
        
        try:
            import json
            selected_indices = json.loads(json_match.group(1))
            
            if not isinstance(selected_indices, list) or not selected_indices:
                return ""
            
            # 获取选中的methods并格式化
            library = exp_lib.get_library()
            formatted_methods = []
            
            for i, idx in enumerate(selected_indices):
                if 0 <= idx < len(library):
                    entry = library[idx]
                    method = entry.get("method", "")
                    rules = entry.get("rules", [])
                    
                    # 格式化method
                    method_text = f"General Method {i+1}: {method}"
                    
                    # 格式化rules with examples
                    if rules:
                        rules_text = "Important Rules:"
                        for j, rule in enumerate(rules):
                            rule_content = rule.get("rule", "")
                            if rule_content:
                                rules_text += f"\n  Rule {j}: {rule_content}"
                                
                                # 添加rule的examples（最多两个）
                                examples = rule.get("examples", [])
                                if examples:
                                    rules_text += f"\n    Examples:"
                                    # 最多取两个例子
                                    for k, example in enumerate(examples[:2]):
                                        if example:
                                            rules_text += f"\n      Example {k+1}: {example}"
                    else:
                        rules_text = "Important Rules: None"
                    
                    formatted_method = f"{method_text}\n\n{rules_text}"
                    formatted_methods.append(formatted_method)
            
            return "\n\n" + "="*50 + "\n\n".join(formatted_methods) if formatted_methods else ""
            
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"Extract exp error: {e}")
            return ""
