def run_amix_tts_experiment(
    initial_sequence: str,
    init_data: str,
    eval_task_weights: str,
    exp_dir: str,
    rounds: int = 10,
    num_seqs: int = 30,
    top_k: int = 10,
    infer_step: int = 100,
    init_t: int = 99,
    eval_filter: str = "",
    target_ec: str = "",
    target_temp: float = None,
    target_ph: float = None,
) -> str:
    """
    Performs a directed evolution experiment using a Test-Time Scaling (TTS) algorithm.
    It starts with an initial protein sequence, iteratively generates, evaluates,
    and selects the best candidates over several rounds based on specified metrics.

    Parameters
    ----------
    initial_sequence : str
        The starting protein sequence for the evolution process. Must be a valid amino acid sequence.
    init_data: str
        The directory containing the sequence (.a3m) and structural (.pdb) files of the original sequence.
    eval_task_weights : str
        Comma-separated list of 'metric:weight' pairs for selection (e.g., 'TM_score:1.0,progen_nll:-0.5').
        A positive weight means higher is better; a negative weight means lower is better.
    exp_dir : str
        Path to the directory where all experiment results and logs will be saved.
    rounds : int, optional
        The number of generation-evaluation-selection cycles to perform (default is 5).
    num_seqs : int, optional
        The number of new candidate sequences to generate in each round (default is 10).
    top_k : int, optional
        The number of best sequences to select from the candidates to proceed to the next round (default is 2).
    infer_step: int, optional
        The infer steps of the underlying flow model.
    init_t: int, optional
        The initial time point (discrete) of the trajectories of the underlying flow model (from 0 to 100).
    eval_filter : str, optional
        Comma-separated list of filtering conditions to apply before selection (e.g., 'pLDDT>80,TM_score>0.7').
    target_ec : str, optional
        Target EC number for the enzyme (e.g., '4.2.3.113').
    target_temp : float, optional
        Target temperature for the reaction environment.
    target_ph : float, optional
        Target pH for the reaction environment.

    Returns
    -------
    str
        A summary report of the experiment, including the top sequences and their scores from the final round.
    """
    import os
    import re
    import json
    import subprocess
    # --- 1. Input Validation and Setup ---
    if not re.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", initial_sequence.upper()):
        return "Error: `initial_sequence` contains invalid characters. Please provide a valid amino acid sequence."

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        print(f"Created experiment directory: {exp_dir}")

    # --- 2. Construct the Command ---
    # python_executable = "/root/miniconda3/envs/tts/bin/python"
    script_path = "/AIRvePFS/ai4science/users/wanghao/tts_dev/run_tts_biomni.sh"

    # Define fixed and user-provided parameters
    cmd = [
        "bash",
        script_path,
        "--exp-dir", str(exp_dir),
        "--init-data", init_data,
        "--rounds", str(rounds),
        "--top-k", str(top_k),
        "--num-seqs", str(num_seqs),
        "--eval-task-weights", str(eval_task_weights),
        "--infer-step", str(infer_step),
        "--init-t", str(init_t),
        "--filter-window", str(rounds),
        # Hardcoded internal parameters for consistency
        "--ckpt-file", "/ai4science/wanghao/zhoujiang/zhoujiang_vepfs/mnt/workspace/fengjiangtao/checkpoints/public/bfn_checkpoint_45_460000.pt",
        "--beta1", "1.6",
        "--beta-time-order", "1.0",
        "--mbcltbf", "1",
        "--batch-size", "600",
        "--infer-type", "profile",
        "--esm-fold-gpus", "1",
        "--mutation-ratio", "0.1",
        "--sort-by", "weighted_score",
    ]

    # Add optional parameters if they are provided
    if eval_filter:
        cmd.extend(["--eval-filter", str(eval_filter)])
    if target_ec:
        cmd.extend(["--target-ec", str(target_ec)])
    if target_temp is not None:
        cmd.extend(["--target-temp", str(target_temp)])
    if target_ph is not None:
        cmd.extend(["--target-ph", str(target_ph)])

    # --- 3. Execute the Experiment ---
    try:
        print(f"Executing command: {' '.join(cmd)}")
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
        )
        print("Experiment script executed successfully.")
        print("STDOUT:\n" + process.stdout)

    except subprocess.CalledProcessError as e:
        error_message = (
            f"TTS experiment failed with exit code {e.returncode}.\n"
            f"Error log (stderr):\n{e.stderr}\n"
            f"Output log (stdout):\n{e.stdout}"
        )
        return error_message

    # --- 4. Parse and Format Results ---
    final_round_num = rounds
    result_file = os.path.join(exp_dir, f"round_{final_round_num}", f"filtered_sequences_round_{final_round_num}.json")

    if not os.path.exists(result_file):
        return (f"Error: Result file '{result_file}' not found. The experiment may have failed to complete the final round. "
                f"Check logs in '{exp_dir}'.")

    with open(result_file, 'r') as f:
        result_data = json.load(f)

    # Assuming a single sample input as per the problem description
    if not result_data:
        return f"Error: The result file '{result_file}' is empty."

    final_sequences = result_data[0]['sequences']

    # Build the report string
    report = [
        f"✅ TTS Directed Evolution Experiment Completed Successfully!",
        f"   - Experiment Directory: {exp_dir}",
        f"\nResults from Final Round ({final_round_num}):",
        f"-------------------------------------------------",
    ]

    if not final_sequences:
        report.append("No sequences passed the filtering criteria in the final round.")
    else:
        for i, seq_data in enumerate(final_sequences):
            report.append(f"🔹 Rank {i+1} Sequence (Top {top_k}):")
            report.append(f"   - Sequence: {seq_data['seq'][:30]}...{seq_data['seq'][-10:]} (Length: {len(seq_data['seq'])})")
            scores_str = ", ".join([f"{k}: {v:.4f}" for k, v in seq_data.get('score', {}).items()])
            report.append(f"   - Scores: {scores_str}")
            report.append(f"   - Weighted Score: {seq_data.get('weighted_score', 'N/A'):.4f}")

    report.append("\nPlots and detailed logs are available in the experiment directory.")
    return "\n".join(report)