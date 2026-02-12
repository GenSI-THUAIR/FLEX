<div align="center">
    <table style="width: 100%; border: none;">
        <tr>
            <td align="left" style="width: 50%; border: none;">
                <img src="asset/seed_logo.png" height="20">
            </td>
            <td align="right" style="width: 50%; border: none;">
                <img src="asset/AIRlogo.png" height="20" style="margin-right: 20px;">
                <img src="asset/gensi_logo_black.png" height="20">
            </td>
        </tr>
    </table>
</div>

---

<div align="center">

# FLEX: Continuous Agent Evolution via Forward Learning from Experience

[![arXiv](https://img.shields.io/badge/arXiv-2511.06449-b31b1b.svg)](https://arxiv.org/abs/2511.06449)

</div>

The official codebase for our paper, FLEX: Continuous Agent Evolution via Forward Learning from Experience.

---

## 🔥 News

- **[2025/11]** We release our paper on arXiv! Check out [FLEX: Continuous Agent Evolution via Forward Learning from Experience](https://arxiv.org/abs/2511.06449).
<!-- - **[2025/11]** Code, datasets, and trained experience libraries are now available! -->

---

## Introduction
Welcome to **F**orward **L**earning from **Ex**perience (FLEX), a novel learning paradigm that shifts learning from modifying model parameters to constructing and leveraging an evolvable experience library.
By continuously expanding and refining this library, agents can progressively acquire deeper insights and knowledge, enhancing their cognitive capabilities with accumulated experiences.

We conduct extensive experiments across diverse challenging scientific domains, including Olympiad-level mathematics (AIME25), chemical retrosynthesis (USPTO50k), and protein fitness prediction (ProteinGym). FLEX demonstrates substantial and consistent improvements on these tasks, from 40\% to 63\% on AIME25 and 20\% to 30\% on USPTO50k, exhibiting great enhancement in the capacity of reasoning and knowledge leverage.

<img src="asset/front_pic.png" width="100%">

The following picture exhibits the differences between gradient-based learning and FLEX, highlighting the interaction among the actor $\pi$, updater $\mu$, and experience library $\mathcal{E}$ of FLEX.

<img src="asset/method_pic.png" width="100%">

We have also discovered **two exciting features** of FLEX:
1. **The scaling law** for the experience library: agent performance scales predictably with accumulated knowledge and revealing a path towards a collaborative experience ecosystem.
2. **Intelligence Inheritance**: Distilled experience can be transferred between agents in a plug-and-play manner, enabling instant knowledge assimilation and bypassing redundant learning

## Getting Started

This repo contains the reference implementation of FLEX used in our paper. The pipeline is split into two main stages:

- **Training / Experience Collection**: run `reject_sampling.py` to generate multi‑round trajectories with an LLM + verifier + critic loop, extract experiences, and write them into an experience library (`exp‑lib`, usually a single JSON file under `exps/`).
- **Testing / Evaluation**: run the testing scripts (`test_aime.py`, `test_retro.py`, `test_flex.py`) to evaluate (i) vanilla LLM / vanilla ReAct agents and (ii) FLEX agents that leverage the learned experience library.

Below we briefly describe how to set up the environment and run each stage.

### 1. Environment setup

1. Install dependencies (we use `uv` in our scripts, but you can use any PEP 517 backend):

```bash
cd FLEX
uv pip install -e .
```

2. Configure your LLM endpoint via environment variables (for OpenAI‑compatible APIs):

```bash
export API_KEY="<YOUR_API_KEY>"
export BASE_URL="<YOUR_OPENAI_COMPATIBLE_BASE_URL>"  # e.g. https://api.openai.com/v1
```

3. Prepare data and prompts (default paths):

- AIME25: 
   - train: `data/AIME/train.jsonl`
   - test: `data/AIME/test.jsonl`
- USPTO‑50k retrosynthesis: 
   - train: `data/uspto50k/train.jsonl`
   - test: `data/uspto50k/test.jsonl`
- Actor agent system prompt: 
   - aime: `prompts/actor_aime.txt`
   - retro: `prompts/actor_retro.txt`
   - default: `prompts/actor_default.txt`
- Memory agent system prompt (managing experience retrieval): 
   - aime: `prompts/memory_aime.txt`
   - retro: `prompts/memory_retro.txt`
   - default: `prompts/memory_default.txt`
- LLM system prompt (for task solving):
   - aime: `prompts/llm_system_prompt.txt`
   - retro: `prompts/llm_system_prompt_retro.txt`
- LLM verifier prompt (in reject sampling): 
   - aime: `prompts/verifier.txt`
   - retro: `prompts/verifier_retro.txt`

You can customize these by passing corresponding CLI flags.

### 2. Stage I – Experience collection via reject sampling

The **training** stage is implemented in `reject_sampling.py`. The core function `process_task(...)` runs a multi‑round loop:

1. **Actor (`actor.py`)**: an LLM agent attempts the task (math or retrosynthesis) under a task‑specific system prompt.
2. **Verifier**: a second LLM checks the latest attempt (using `prompts/verifier.txt` for math, or a built‑in verifier prompt for retro) and decides whether to continue (`<next_step>continue</next_step>`) or stop (`<next_step>end</next_step>`).
3. **Critic (`critic.py`)**: after a completed trajectory (one rollout), the critic agent extracts structured experiences from the full conversation trace.
4. **Updater (`updater.py`) + Experience Library (`explib.py`)**: the updater inspects current experience entries and critic outputs, performs de‑duplication / filtering, and writes accepted experiences into the experience library JSON (e.g. `exps/aime_claude.json`, `exps/retro_claude.json`).

`reject_sampling.py` is written as a library entry point and is typically wrapped by your own driver script or notebook. At a high level you:

```python
from rate_limiter import RateLimitedLLMClient
from critic import Critic
from explib import ExpLib
from reject_sampling import process_task

client = RateLimitedLLMClient(api_key=..., base_url=...)
critic_model = Critic(model_name="...")
exp_lib = ExpLib(path="exps/aime_claude.json")

history = await process_task(
        query=question,
        ground_truth=answer,
        client=client,
        llm="<actor_model>",
        verifier="<verifier_model>",
        critic="<critic_model>",
        max_rounds=5,
        max_rollouts=3,
        task_type="math",  # or "retro"
)

# Then pass `history` to `critic` and `updater` to update `exp_lib`.
```

In practice, you will iterate this over the whole training split (e.g. AIME train, USPTO train) and periodically save the updated experience library JSON under `exps/`.

Key `process_task` arguments:

- `task_type`: `"math"` or `"retro"` (selects prompts and formatting).
- `max_rounds`: maximum LLM + verifier iterations per rollout.
- `max_rollouts`: maximum independent rollouts per problem.
- `llm_system_prompt_path`: optional path to override the default actor system prompt.
- `verifier_prompt_path`: optional path to override the default verifier system prompt.
- `with_gt`: whether to reveal ground truth to the verifier (for supervised reject sampling).

### 3. Stage II – Vanilla agent evaluation

We provide task‑specific evaluation scripts to measure vanilla LLM / ReAct agents **without** using FLEX experiences.

#### 3.1 AIME25 evaluation – `test_aime.py`

This script evaluates on AIME25 math problems. It supports two modes:

- **Direct LLM**: call the base model directly.
- **Actor**: route through `Actor` in `actor.py` (e.g. ReAct‑style reasoning, tools, etc.).

Run with default settings:

```bash
cd flex_refined
python test_aime.py \
    --model z-ai/glm-4.5 \
    --use_actor \
    --data_path data/AIME/test.jsonl \
    --results_dir results/agent_aime_glm
```

Important CLI flags:

- `--model`: base model name / ID.
- `--temperature`: sampling temperature (default 0).
- `--use_actor`: if set, use `Actor` instead of raw API (default `True` in code).
- `--retrieve`: whether to enable experience retrieval inside `Actor` (FLEX mode; only effective when `--use_actor`).
- `--data_path`: path to the AIME JSONL file (`{"question": ..., "answer": ...}` per line).
- `--results_dir`: directory to store per‑problem `.txt` logs.
- `--batch_size`, `--batch_pause`: control async batch size and pause between batches.
- `--no-telemetry`: disable optional Smolagents / Phoenix tracing.

Each problem is processed by `process_question(...)`, which:

1. Builds an `Actor` (or raw API call) with the specified model.
2. Solves the problem and logs the full trajectory.
3. Saves question, response, and ground‑truth answer to `results_dir/<idx>.txt`.

#### 3.2 USPTO‑50k retrosynthesis evaluation – `test_retro.py`

This script evaluates single‑step retrosynthesis on USPTO‑50k. It also supports `Actor` and direct LLM modes.

Run with typical settings:

```bash
python test_retro.py \
    --model x-ai/grok-4 \
    --use_actor \
    --data_path data/uspto50k/test.jsonl \
    --results_dir results/agent_retro_grok
```

Key CLI flags:

- `--model`, `--temperature`, `--use_actor`: same semantics as `test_aime.py`.
- `--data_path`: path to the USPTO‑50k test split (`{"question": <product SMILES>, "answer": <reactant SMILES>}`).
- `--results_dir`: per‑sample logs (prompt, response, extracted final answer, ground truth, pass/fail).
- `--batch_size`, `--batch_pause`, `--no-telemetry`: same as above.

Evaluation metrics:

- For vanilla evaluation, the script compares the extracted `<answer>...</answer>` block (or the `Actor` final tool call) against ground truth.
- For more robust chemistry evaluation (ignoring reactant order), you can post‑process results with `utils.evaluate_retro_results_fuzzy(results_dir)`.

### 4. Stage III – FLEX agent evaluation with experience library

To evaluate FLEX (actor + experience library), use `test_flex.py`. This script:

1. Loads an experience library JSON via `memory_retriever.py` and `explib.py`.
2. Wraps it into a retrieval tool (`create_memory_retrieval_tool`) accessible to the `Actor`.
3. Runs either math (AIME25) or retrosynthesis (USPTO‑50k) depending on `--task_type`.
4. Logs detailed per‑problem results and optional timing.

Example: AIME25 + FLEX with math experiences

```bash
python test_flex.py \
    --task_type math \
    --actor deepseek/deepseek-v3.1-terminus \
    --memory_path exps/aime_claude.json \
    --data_path data/AIME \
    --split test \
    --batch_size 10 \
    --results_dir results/agent_mem_aime_deepseek_mem_claude
```

Example: USPTO‑50k + FLEX with retrosynthesis experiences

```bash
python test_flex.py \
    --task_type retro \
    --actor deepseek/deepseek-v3.1-terminus \
    --memory_path exps/retro_claude.json \
    --data_path data/uspto50k \
    --split test \
    --batch_size 10 \
    --results_dir results/agent_mem_retro_gemini_mem_claude
```

Important CLI flags in `test_flex.py`:

- `--task_type`: `math` or `retro`, controls dataset, prompts and evaluation logic.
- `--actor`: the LLM model used inside `Actor`.
- `--memory_path`: path to the experience library JSON (file or directory). If omitted, sensible defaults are chosen based on `task_type`:
    - math: `./exps/aime_claude.json`
    - retro: `./exps/retro_claude.json`
- `--data_path`: dataset root directory (default `./data/AIME/` or `./data/uspto50k/`).
- `--split`: split name without `.jsonl` (e.g. `aime25` or `test`).
- `--samples`: limit number of samples for quick debugging.
- `--batch_size`, `--max_concurrent`: control async concurrency.
- `--no-retrieve`: disable retrieval even if a memory library is provided (ablation).
- `--results_dir`: directory for detailed per‑problem logs; a `_process` subfolder records raw agent traces.
- `--pass_at_n`: enable Pass@N‑style evaluation where each problem is tried multiple times.
- `--no-telemetry`: disable Smolagents / Phoenix tracing.

Under the hood, `test_flex.py` creates a per‑sample `Actor` with a `memory_retrieval_tool`:

1. `memory_retriever.create_memory_retriever(...)` loads the JSON experience library and builds a retriever.
2. `create_memory_retrieval_tool(...)` wraps it as a tool that can be called by the `Actor`.
3. `Actor` queries this tool during reasoning (when `retrieve=True`) to load relevant experiences, which are then incorporated into the solution.

### 5. Core components

- `actor.py`: task‑solving agent (LLM + tools + optional memory retrieval). Used in both training and testing.
- `critic.py`: experience extraction agent; reads full rollouts and produces structured experience entries.
- `explib.py`: experience library abstraction; manages reading, writing and indexing experience entries.
- `updater.py`: decides which critic‑generated experiences to accept, preventing excessive duplication in `exp‑lib`.
- `memory_retriever.py`: creates retrievers and tools for querying `explib` during inference.
- `rate_limiter.py`: wraps an async OpenAI‑compatible client with rate limiting and robust logging.

## Citation

If you use FLEX in your research, please cite our paper:

```bibtex
@misc{cai2025flexcontinuousagentevolution,
      title={FLEX: Continuous Agent Evolution via Forward Learning from Experience},
      author={Zhicheng Cai and Xinyuan Guo and Yu Pei and JiangTao Feng and Jiangjie Chen and Ya-Qin Zhang and Wei-Ying Ma and Mingxuan Wang and Hao Zhou},
      year={2025},
      eprint={2511.06449},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.06449},
}
```
