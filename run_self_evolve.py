
import os
import argparse
import pandas as pd
from datetime import datetime

import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed


PROFILE_PATH = "/ai4science-a100/yupei/data/ProteinGym/ProteinGym_split_63_filtered/protein_profile.csv"





def run_cmd(cmd):
    print(f"🚀 Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main(args):

    data_profile_df = pd.read_csv(PROFILE_PATH)
    data_profile_dict = data_profile_df.set_index('target').T.to_dict()

    time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    # time_str = "20250927092408"
    output_subdir = os.path.join(args.output_dir, f"self_evolve_corr_{time_str}")


    with ProcessPoolExecutor(max_workers=min(len(os.listdir(args.input_dir)), os.cpu_count())) as executor:
        futures = []
        for subdir in os.listdir(args.input_dir):
            if("." in subdir):
                continue

            subdir_path = os.path.join(args.input_dir, subdir)
            output_subdir_path = os.path.join(output_subdir, subdir)

            test_fpath = os.path.join(subdir_path, f"{subdir}_left100.csv")


            # for fold in range(args.fold):
            fold = 0

            fold_dir = os.path.join(subdir_path, f"{fold}_fold")
            output_fold_dir = os.path.join(output_subdir_path, f"{fold}_fold")


            cmd = f"python ./forward_learning/{args.setting}/evolve_long.py --input_dir {fold_dir} --record_dir {output_fold_dir} --target_name {subdir} --strategy_nums {args.strategy_nums} --add_samples_num {args.add_samples_num} --model {args.model} --iters {args.iters} --train_ratio {args.train_ratio} --top_k {args.top_k}"

            futures.append(executor.submit(run_cmd, cmd))


        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"❌ Error: {e}")

            

    # gather result
    end_cmd = f"python gather_evolve_result.py --input_dir {args.input_dir}"
    os.system(end_cmd)
    # os.system(f"echo {end_cmd} | python experiments/sendtask2queue.py -q amix_evolve -p 1")


 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/AIRvePFS/ai4science/users/yupei/data/ProteinGym/ProteinGym_test/")
    parser.add_argument("--output_dir", type=str, default="/AIRvePFS/ai4science/users/wanghao/Biomni_dev/test_cases/2025.9.29-10.8/")
    parser.add_argument("--setting", type=str, default="zero-shot")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--model", type=str, default="claude4")
    parser.add_argument("--strategy_nums", type=int, default=25)
    parser.add_argument("--add_samples_num", type=int, default=20)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()
    main(args)




# python run_self_evolve.py --strategy_nums 5 --input_dir /ai4science-a100/yupei/data/ProteinGym/ProteinGym_split_63_filtered_test/ --output_dir ./logs/ --model gpt-4o --mode biomni --iters 5 --train_ratio 0.75 --top_k 5 --fold 1 --setting zero-shot 