
import os
import sys
import argparse
from os.path import exists
from numpy import ndarray
import numpy as np
import pandas as pd
from datetime import datetime
import shutil
import json
from tabulate import tabulate
import joblib
from copy import deepcopy

from biomni.agent import A1
from biomni.agent import react

PROFILE_PATH = "/ai4science-a100/yupei/data/ProteinGym/ProteinGym_split_63_filtered/protein_profile.csv"


def add_md(df, md_fpath, n):
    top_features = df["feature"].head(n)

    header = f"## Top features of Valid Dataset\n"
    feature_lines = "".join(f"{idx+1}. **{f}**\n" for idx, f in enumerate(top_features))
    new_section = header + feature_lines + "\n"

    try:
        with open(md_fpath, "r", encoding="utf-8") as f:
            old_content = f.read()
    except FileNotFoundError:
        old_content = ""

    updated_content = new_section + old_content

    with open(md_fpath, "w", encoding="utf-8") as f:
        f.write(updated_content)

def add_proxy_metric_column(x_df: pd.DataFrame, test_df: pd.DataFrame, model_path, proxy_name, columns) -> pd.DataFrame:

    model = joblib.load(model_path)

    drop_cols = [c for c in ["mutant", "mutated_sequence", "DMS_score", "DMS_score_bin"] if c in test_df.columns]
    test_df = test_df.drop(columns=drop_cols)

    print(f"selected_columns = {columns}")
    X = test_df[columns].values

    preds = model.predict(X)
    preds = np.asarray(preds)
    print(f"pred = {preds}")

    x_df[proxy_name] = preds
    return x_df



def main(args):

    data_profile_df = pd.read_csv(PROFILE_PATH)
    data_profile_dict = data_profile_df.set_index('target').T.to_dict()

    # 对于测试集中每一个数据，进行策略的self-evolving

    # pujiang
    # agent = A1(
    #     path='./data', 
    #     llm='claude-sonnet-4-20250514', 
    #     base_url="https://api.boyuerichdata.opensphereai.com/", 
    #     api_key="sk-xgYFUgHgSy1PlOaqXqWYpqZoB6nR6uYIPCBMhpWen9l896QJ",
    #     timeout_seconds=172800,
    #     # self_critic=True
    # )

    # # tsinghua
    # agent = A1(
    #     path='./data', 
    #     llm='anthropic/claude-sonnet-4', 
    #     base_url="http://103.242.175.254:20008/v1", 
    #     api_key="sk-audit-wv7iFmrhKCapmecwx1VwvhRTbpPpJw30",
    #     timeout_seconds=172800,
    #     # self_critic=True
    # )


    if args.model == "gpt_oss":
        if(args.mode == "react"):
            agent = react(
                path='./data',
                llm='/data/xyguo/gpt-oss-120b',   
                base_url="http://103.242.175.254:20011/v1",
                api_key="not-needed",
                timeout_seconds=172800,
            )
        else:
            agent = A1(
                path='./data',
                llm='/data/xyguo/gpt-oss-120b',   # <- 如果你的服务需要完整路径就用这个；否则尝试 'gpt-oss-120b'
                base_url="http://103.242.175.254:20011/v1",
                api_key="not-needed",
                timeout_seconds=172800,
            )
    else :
        agent = A1(
                    path='./data', 
                    llm='gpt-4o', 
                    base_url="http://14.103.213.146/suwen/v1",
                    api_key="sk-audit-8Hn797H64m7iHiWsLnjjFs0EYY1QcQDy",
                    timeout_seconds=172800,
                )

    subdir = args.target_name

    fold_dir = args.input_dir
    record_dir = args.record_dir

    test_fpath = os.path.join(fold_dir, f"../wogt_{subdir}_left100.csv")
    test_df = pd.read_csv(test_fpath)
    test_columns = test_df.columns
    train_fpath = os.path.join(fold_dir, f"../{subdir}_train100.csv")
    

    steps = range(1, args.iters + 1)
    
    # 制造 n-fold 数据
    for i, step in enumerate(steps):
        step_dir = os.path.join(record_dir, f"step{step}")
        os.makedirs( step_dir, exist_ok=True )
        df = pd.read_csv(train_fpath)

        for j in range(args.n_fold):
            valid_fold_dir = os.path.join(step_dir, "valid_fold")
            os.makedirs( valid_fold_dir, exist_ok=True )

            df = df.sample(frac=1, random_state=( (i+1) * (j+1) + 1)).reset_index(drop=True)
            ratio = float(len(df) / (len(test_df) + len(df) ) )
            # if(ratio < 0.2):
            ratio = 0.2

            # 按比例划分训练/测试集
            train_size = int(len(df) * ratio)
            df_train = df.iloc[:train_size, :]
            df_test = df.iloc[train_size:, :]

            df.to_csv( os.path.join(valid_fold_dir, f"{subdir}_train100.csv"), index=False )

            df_train.to_csv(os.path.join(valid_fold_dir, f"{subdir}_train{step}_fold{j}.csv"), index=False)
            df_test.to_csv(os.path.join(valid_fold_dir, f"{subdir}_test{step}_fold{j}.csv"), index=False)

    os.makedirs(os.path.join(record_dir, "step0"), exist_ok=True)
    start_exp_fpath = os.path.join(record_dir, "step0", "experience.md")
    os.system(f"cp /AIRvePFS/ai4science/users/yupei/Biomni_dev/forward_learning/experience.md {start_exp_fpath}")

    
    for i, step in enumerate(steps):
        step_dir = os.path.join(record_dir, f"step{step}")

        train_data = os.path.join(step_dir, "valid_fold", f"{subdir}_train100.csv")
        valid_train_data = os.path.join(step_dir, "valid_fold", f"{subdir}_train{step}_fold0.csv")
        valid_test_data = os.path.join(step_dir, "valid_fold", f"{subdir}_test{step}_fold0.csv")
        vaild_test_df = pd.read_csv(valid_test_data)
        model_dir = os.path.join(step_dir, "models")
        script_dir = os.path.join(step_dir, "proxy_script")
        predict_dir = os.path.join(step_dir, "predict_dir")
        gather_dir = os.path.join(step_dir, "gather_info")

        prompt = None

        past_exp_fpath = os.path.join(record_dir, f"step{step-1}", f"experience.md")
        next_test_data = os.path.join(step_dir, f"{subdir}_test{step}.csv")

        
        # train and propose strategy
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(predict_dir, exist_ok=True)
        os.makedirs(script_dir, exist_ok=True)



        # select column and operator
        select_check = False
        while(not select_check):
            os.makedirs(gather_dir, exist_ok=True)
            prompt = f"""
            [Task Objective]  
                A dataset is stored in {valid_train_data}. After reading it into a pd.DataFrame, the file contains the following columns:
                - mutant: please ignore this column  
                - mutated_sequence: please ignore this column  
                - mut_num: please ignore this column  
                - DMS_score_bin: please ignore this column  
                - DMS_score: this is the target variable y to be fitted  
                - other columns([PDA-Pred-ddG,PDA-Pred-Kd,GEMME,ll_means,Saprot_values,esm2_cmp,esmc_cmp,esm3_cmp,proteinix_plddt_mean_wo_msa, progen2-base-nll, proteinix_plddt_max_wo_msa,proteinix_plddt_mean_w_msa,proteinix_plddt_max_w_msa,proteinix_plddt_mean_w_dna,proteinix_plddt_max_w_dna,ProSST-4096,mutated_sequences,ProSST-2048,ProSST-1024,ProSST-512,ProSST-128,ProSST-20,S3F,VenusREM_searched_msa,VenusREM_wetlab_msa,proteinglm-10b-mlm_mlm_score, Boltz-2_Affinity_Avg,Boltz-2_Affinity_Max,Boltz-2_Affinity_Binary_Avg,Boltz-2_Affinity_Binary_Max]): the remaining columns are experimentally measured floating-point values, which can be used as optional x features.

                Based on all x feature columns, construct new computational experimental metrics (proxy_metric) expressed as formulas, 
                such that the Spearman correlation coefficient between proxy_metric and DMS_score is as high as possible.  
                Note: proxy_metric must be computed **only** from x feature columns, and cannot use the y column (DMS_score).

            [Execution Steps]  
                Propose {args.strategy_nums} different fitting strategies, that each include:
                    1. Column Selection
                    Based on the past experience in {past_exp_fpath}, select 3 columns that may affect DMS_score, then randomly add 0 new columns in the rest of dataset to explore new areas.
                    e.g. [esm3_cmp, ..., SaProt_values]; 

                    2. Regression Algorithm Selection
                    Based on the past experience in {past_exp_fpath}, decide and describe an Regression Algorithm within 50 words, which use selected features and target y in training set to fit a regression model.
                    note: don't include training-free method like weighted sum.

                please use different columns and algorithm, that will result in {args.strategy_nums} different json files generated.
                
            [Storage Requirement]  
                save selected column and selected algorithm to file {os.path.join(gather_dir, f"calculate_proxy_metric_step{step}_i.json")} , i=0,1,....,{args.strategy_nums-1}
                e.g. 
                    {{  
                        \"selected_columns\": [\"esm3_cmp\", \"esmc_600M_cmp\", ... , \"Saprot_values\"],
                        \"selected_algorithm\": \"The Algorithm is ...\"
                    }}


            """

            agent.go(prompt)

            # check execute and produce model
            select_check = True
            for j in range(args.strategy_nums):
            
                gather_info_path = os.path.join(gather_dir, f"calculate_proxy_metric_step{step}_{j}.json")

                if(not os.path.exists(gather_info_path)  ):
                    select_check = False
                    shutil.rmtree(gather_dir)
                    break

        # for generated json file, check and filter in-valid columns
        for j in range(args.strategy_nums):

            json_fpath = os.path.join(gather_dir, f"calculate_proxy_metric_step{step}_{j}.json")
            json_dict = None
            with open(json_fpath, "r") as f:
                json_dict = json.load(f)
            json_dict["selected_columns"] = [col for col in json_dict["selected_columns"] if(col in test_columns)]
            with open(json_fpath, "w") as f:
                json.dump(json_dict, f)


        for j in range(args.strategy_nums):

            json_fpath = os.path.join(gather_dir, f"calculate_proxy_metric_step{step}_{j}.json")
            json_dict = None
            with open(json_fpath, "r") as f:
                json_dict = json.load(f)


            # gen code and execute
            gen_check = False
            while(not gen_check):
                
                prompt = f"""
                [Task Objective]  
                    Complete a python script based on guidance tips.
      
                [Execution Steps]  
                    Script Generation
                    Based on selected column and selected algorithm from json file, complete a script based on the following templete, which uses features x and target values y to fit a regression model: 
                    
                    import argparse
                    import pandas as pd
                    import joblib
                    from copy import deepcopy
                    # note: import packages that contain selected regression method

                    def calc_algo(selected_features, target_values) :
                        # TODO : {json_dict["selected_algorithm"]}
                        

                        # save trained model checkpoint in ckpt_path
                        joblib.dump(model, ckpt_path)


                    def calculate_proxy_metric(x_df: pd.DataFrame = None, ckpt_path: str) -> None:
                        df = deepcopy(x_df)
                        y = df["DMS_score"].values
                        drop_cols = [c for c in ["mutant", "mutated_sequence", "DMS_score", "DMS_score_bin", "mut_num"] if c in df.columns]
                        df = df.drop(columns=drop_cols)

                        selected_columns = {json_dict["selected_columns"]}
                        X_selected = df[selected_cols].values

                        calc_algo(X_selected, y)

                
                    if __name__ == "__main__":
                        parser = argparse.ArgumentParser()
                        parser.add_argument("--input_csv", type=str, default=f"")
                        parser.add_argument("--ckpt_path", type=str, default=f"")
                        args = parser.parse_args()

                        input_df = pd.read_csv(args.input_csv)
                        calculate_proxy_metric(input_df, args.ckpt_path)


                [Storage Requirement] 

                    save each script in file {os.path.join(record_dir, f"step{step}", "proxy_script", f"calculate_proxy_metric_step{step}_{j}.py")}.

                """

                agent.go(prompt)

                # check execute and produce model
                gen_check = True
                proxy_name = f"calculate_proxy_metric_step{step}_{j}"
                proxy_script_path = os.path.join(step_dir, "proxy_script", f"{proxy_name}.py")
                trained_model_path = os.path.join(step_dir, "models", f"{proxy_name}.joblib")
                try_train_exit_code = os.system(f"python {proxy_script_path} --input_csv {valid_train_data} --ckpt_path {trained_model_path}")
                    # check model predict
                predict_check = False
                if(os.path.exists(trained_model_path)):
                    added_df = deepcopy(vaild_test_df )
                    try:
                        added_df = add_proxy_metric_column(added_df, vaild_test_df, trained_model_path, proxy_name, json_dict["selected_columns"])
                    except:
                        predict_check = False
                    if(proxy_name in added_df.columns):
                        predict_check = True


                if(not ( (predict_check) and (try_train_exit_code == 0) and (os.path.exists(trained_model_path) )  and ( os.path.exists(proxy_script_path) ))):
                    gen_check = False
                    if(os.path.exists(trained_model_path)):
                        os.remove(trained_model_path)
                    if(os.path.exists(proxy_script_path)):
                        os.remove(proxy_script_path)

          
        # try evaluator manually
        os.system(f"python forward_learning/few-shot/reject_sampling_fewshot.py --input_dir {step_dir} --top_k {args.top_k} --n_fold {args.n_fold}")

        # gather into result df : [method, spearman, calc_operator, selected_columns]
        valid_summary_fpath = os.path.join(step_dir, f"top_{args.top_k}_valid.csv")
        exp_fpath = os.path.join(step_dir, f"experience.md")
        valid_summary_df = pd.read_csv(valid_summary_fpath)
        oracle_value_df = valid_summary_df[valid_summary_df["target"] == "test_best_col"].copy()

        valid_summary_df = valid_summary_df[valid_summary_df["target"] != "test_best_col"].copy()
        valid_positive_df = valid_summary_df[valid_summary_df["status_gap"] >= (-0.02)].copy()
        valid_negative_df = valid_summary_df[valid_summary_df["status_gap"] < (-0.02)].copy()

            # concret test oracle row
        oracle_value_df.loc[0, 'selected_algorithm'] = 'You may keep trying more any other algorithm'
        valid_positive_df = pd.concat([valid_positive_df, oracle_value_df], ignore_index=True)


        valid_positive_df = valid_positive_df.drop(columns=['Unnamed: 0'])
        valid_negative_df = valid_negative_df.drop(columns=['Unnamed: 0'])

        valid_positive_str = tabulate(valid_positive_df, headers='keys', tablefmt='grid', showindex=False)
        valid_negative_str = tabulate(valid_negative_df, headers='keys', tablefmt='grid', showindex=False)

 
        valid_dataset_summary_fpath = os.path.join(step_dir, "valid_fold", f"{subdir}_test{step}_spearman_merged.csv")
        valid_dataset_summary_df = pd.read_csv(valid_dataset_summary_fpath)


        gather_check = False
        while(not gather_check):
            
            prompt = f"""

        [Task Objective]  
            Extract know-hows from added DataFrame, complete the following form:

            note: please don't include training-free in your answer, like weighted sum, median, average and etc.

            Given well-performed trails:
                {valid_positive_str}

            [Top 2 performance algorithms that combines features]
                TODO


            after finished summary above, conclude meta level guidance as follows:

            [Ways for discovering features for improving Spearman correlation]
                TODO

            [Directions for finding more beneficial algorithms]
                TODO
            
            please write each aspect within 100 words, then save it to file {exp_fpath}

            """

            agent.go(prompt)

            # check execute and produce model
            gather_check = True
            if(not  ( os.path.exists(exp_fpath) ) ):
                gather_check = False

        add_md(valid_dataset_summary_df , exp_fpath, 2)








        # # test and modify strategy
        # top_script_dir = os.path.join(record_dir, f"step{step}", "top_proxy_metrics")
        # while(not (os.path.exists(top_script_dir) and (len(os.listdir(top_script_dir)) == 3) and os.path.exists(os.path.join(record_dir, f"step{step}", f"test_data_with_proxy_metrics.csv"))) ):
        #     if(os.path.exists(top_script_dir)):
        #         shutil.rmtree(top_script_dir)

        #     prompt = f"""
        #     [Known Data]  

        #         1. A dataset is stored in {next_test_data}. After reading it into a pd.DataFrame, the file contains the following columns:
        #         - mutant: please ignore this column  
        #         - mutated_sequence: please ignore this column  
        #         - DMS_score_bin: please ignore this column  
        #         - DMS_score: this is the target variable y calculating spearman correlation with 
        #         - other columns: the remaining columns are experimentally measured floating-point values, which can be used as optional x features.

        #         2. Python files in directory {os.path.join(record_dir, f"step{step}", "proxy_metrics")} stored several method to compte proxy metric using x features.
        #         You may execute the file and call all functions in this file to saperately calculate different proxy metrics for each sample.

        #     [Execution Steps]  
        #         1. For each python script, get proxy metrics based on all x feature columns, each metric became a new column of DataFrame, save the whole DataFrame in file {os.path.join(record_dir, f"step{step}", f"test_data_with_proxy_metrics.csv")}.
        #         2. calculate spearman correlation between each proxy metric and DMS_score column in step 1, store those correlations in file {os.path.join(record_dir, f"step{step}", "test_correlation.csv")}.
        #         3. Copy python file of top 3 metrics into new folder {os.path.join(record_dir, f"step{step}", "top_proxy_metrics")}
        #         Note: proxy_metric must be computed **only** from x feature columns, and cannot use the y column (DMS_score).

        #     [Storage Requirement]  
        #         Use the directory {os.path.join(record_dir, f"step{step}")} as the working directory.

        #     """

        #     agent.go(prompt)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--record_dir", type=str)
    parser.add_argument("--model", type=str, default="claude4")
    parser.add_argument("--mode", type=str, default="biomni")
    parser.add_argument("--target_name", type=str)
    parser.add_argument("--strategy_nums", type=int)
    parser.add_argument("--add_samples_num", type=int)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--n_fold", type=int, default=1)
    args = parser.parse_args()
    main(args)


