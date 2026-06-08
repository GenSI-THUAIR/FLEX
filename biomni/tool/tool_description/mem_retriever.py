description = [
    {
        "name": "mem_retriever",
        "description": "Performs a directed evolution experiment using a Test-Time Scaling (TTS) algorithm. It starts with an initial protein sequence, iteratively generates, evaluates, and selects the best candidates over several rounds based on specified metrics such as TM_score, pLDDT, etc. Returns the top sequences and their scores from the final round. IMPORTANT: To access final results, read the data.json file from the exp_dir directory (exp_dir/data.json). The data.json file contains experiment results with structure: {sample_meta: {id, eval_task, higher_better}, rounds: {'0': [...], '1': [...], ..., 'N': [...]} where N is the final round number}. For example, if you run 10 rounds, the data will contain rounds '0' through '10' (11 total entries), where round '10' contains the final optimized sequences.",
        "required_parameters": [
            {
                "name": "initial_sequence",
                "type": "str",
                "description": "The starting protein sequence for the evolution process. Must be a valid amino acid sequence using standard 20 amino acid letters (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y). The sequence will be used as the seed for generating variants through the TTS algorithm."
            },
            {
                "name": "init_data",
                "type": "str",
                "description": "The directory containing the sequence (.a3m) and structural (.pdb) files of the original sequence. This directory should contain the initial data files needed for the experiment setup."
            },
            {
                "name": "eval_task_weights",
                "type": "str",
                "description": "Comma-separated list of 'metric:weight' pairs for weighted scoring and selection (e.g., 'TM_score:1.0,progen_nll:-0.5'). WEIGHT INTERPRETATION: Positive weights mean 'higher is better', negative weights mean 'lower is better'. Available metrics include: progen_nll (negative log-likelihood, lower is better), TM_score (structure similarity, 0-1), rosetta_energy (energy score, lower is better), repeatness (repetitive pattern score), novelty (sequence novelty score), diversity (sequence diversity score), CLEAN_EC (enzyme classification), identical (sequence identity score), clipzyme (enzyme activity prediction), CLEAN_Distance (distance metric), seq2phopt (sequence to phenotype optimization), seq2topt (sequence to target optimization). The weighted score is calculated as: sum(weight_i * metric_i) across all specified metrics."
            },
            {
                "name": "exp_dir",
                "type": "str",
                "description": "Absolute path to the directory where all experiment results and logs will be saved. The directory will be created if it doesn't exist. Results include: generated sequences, evaluation scores, structure predictions, and detailed logs for each round."
            }
        ],
        "optional_parameters": [
            {
                "name": "rounds",
                "type": "int",
                "default": 5,
                "description": "The number of generation-evaluation-selection cycles to perform. Each round consists of: 1) Generate new sequences from current best sequences, 2) Evaluate all sequences using specified metrics, 3) Filter and select top sequences for next round. Typical range: 3-10 rounds. More rounds allow better optimization but increase computational cost."
            },
            {
                "name": "num_seqs",
                "type": "int",
                "default": 10,
                "description": "The number of new candidate sequences to generate in each round using the TTS algorithm. These are generated from the best sequences of the previous round through controlled mutations. Typical range: 10-100 sequences. Higher values increase diversity but require more computational resources."
            },
            {
                "name": "top_k",
                "type": "int",
                "default": 2,
                "description": "The number of best sequences to select from the candidates to proceed to the next round. These sequences serve as parents for the next generation. IMPORTANT: top_k should be <= num_seqs and typically 20-50% of num_seqs to maintain diversity while ensuring quality."
            },
            {
                "name": "infer_step",
                "type": "int",
                "default": 50,
                "description": "The infer steps of the underlying flow model. This parameter controls the number of inference steps used by the generative model during sequence generation."
            },
            {
                "name": "init_t",
                "type": "int",
                "default": 99,
                "description": "The initial time point (discrete) of the trajectories of the underlying flow model (from 0 to 100). This parameter controls the starting point of the diffusion process in the generative model."
            },
            {
                "name": "eval_filter",
                "type": "str",
                "default": "",
                "description": "⚠️ CRITICAL PARAMETER - Comma-separated list of filtering conditions to apply before selection (e.g., 'pLDDT>80,TM_score>0.7'). Supported operators: >, <, >=, <=, ==, !=. Available metrics for filtering: progen_nll (negative log-likelihood, lower is better), TM_score (0-1, structure similarity), rosetta_energy (energy score, lower is better), repeatness (repetitive pattern score), novelty (sequence novelty score), diversity (sequence diversity score), CLEAN_EC (enzyme classification score), identical (sequence identity score), clipzyme (enzyme activity prediction), CLEAN_Distance (distance metric), seq2phopt (sequence to phenotype optimization), seq2topt (sequence to target optimization). ⚠️ WARNING: Overly strict filters may eliminate ALL sequences in a round, causing the experiment to fail. If this occurs, try relaxing the thresholds (e.g., change 'progen_nll<-5' to 'progen_nll<-3', or 'TM_score>0.8' to 'TM_score>0.5'). Recommended starting values: 'TM_score>0.5' or 'progen_nll<-2' for initial experiments."
            },
            {
                "name": "target_ec",
                "type": "str",
                "default": "",
                "description": "Target Enzyme Commission (EC) number for the desired enzymatic activity (e.g., '4.2.3.113' for a specific lyase). The EC number follows the format 'X.Y.Z.W' where: X=enzyme class (1-7), Y=subclass, Z=sub-subclass, W=serial number. This parameter guides the evolution towards specific enzymatic functions. Common EC classes: 1=oxidoreductases, 2=transferases, 3=hydrolases, 4=lyases, 5=isomerases, 6=ligases, 7=translocases."
            },
            {
                "name": "target_temp",
                "type": "float",
                "default": None,
                "description": "Target temperature (°C) for the reaction environment where the evolved enzyme should be active. Common values: 25°C (room temperature), 37°C (human body), 50-80°C (thermophilic applications), 4°C (cold storage). This parameter influences stability and activity predictions during evaluation."
            },
            {
                "name": "target_ph",
                "type": "float",
                "default": None,
                "description": "Target pH for the reaction environment where the evolved enzyme should function optimally. Common ranges: 6.0-8.0 (physiological), 2.0-4.0 (acidic), 8.0-10.0 (alkaline). This parameter affects enzyme stability, substrate binding, and catalytic efficiency during evaluation."
            }
        ],
        "return_value": {
            "type": "str",
            "description": "JSON string containing the results of the directed evolution experiment. Includes: 1) top_sequences: list of best evolved sequences with their scores, 2) evolution_metrics: progression of scores across rounds, 3) final_statistics: summary of improvements achieved, 4) experiment_metadata: parameters used and computational details. CRITICAL: For complete results analysis, read the data.json file from exp_dir/data.json which contains detailed round-by-round data including all sequences, scores, and rankings for each optimization round."
        },
        "common_issues": [
            {
                "issue": "All sequences filtered out",
                "cause": "eval_filter parameters too strict",
                "solution": "Relax filtering thresholds (e.g., pLDDT>90 → pLDDT>70)"
            },
            {
                "issue": "Poor convergence",
                "cause": "Insufficient diversity or rounds",
                "solution": "Increase num_seqs, reduce top_k ratio, or add more rounds"
            },
            {
                "issue": "Memory/computation errors",
                "cause": "Too many sequences or complex evaluation",
                "solution": "Reduce num_seqs, simplify eval_task_weights, or use smaller batch sizes"
            }
        ],
        "best_practices": [
            "Start with relaxed eval_filter parameters and tighten them in subsequent experiments",
            "Use 2-3 complementary metrics in eval_task_weights (e.g., structure + function)",
            "Monitor intermediate results to adjust parameters if needed",
            "For initial experiments, use rounds=3-5, num_seqs=20-50, top_k=5-10"
        ]
    }
]