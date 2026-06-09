
import os
import sys
import argparse
from os.path import exists
from numpy import ndarray
import pandas as pd
from datetime import datetime
import shutil
import json
from biomni.agent import A1
from biomni.agent import react
from tabulate import tabulate


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





def main(args):

    profile_path = os.path.join(args.input_dir, "protein_profile.csv")

    data_profile_df = pd.read_csv(profile_path)
    data_profile_dict = data_profile_df.set_index('target').T.to_dict()

    if(args.mode == "react"):
        agent = react(
            path='./data',
            llm=args.llm,  
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_seconds=172800,
        )
    else:
        agent = A1(
            path='./data',
            llm=args.llm,   
            base_url=args.base_url,
            api_key=args.api_key,
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
    os.system(f"cp ./forward_learning/experience.md {start_exp_fpath}")

    
    for i, step in enumerate(steps):
        step_dir = os.path.join(record_dir, f"step{step}")

        train_data = os.path.join(step_dir, "valid_fold", f"{subdir}_train100.csv")
        valid_train_data = os.path.join(step_dir, "valid_fold", f"{subdir}_train{step}_fold0.csv")
        valid_train_df = pd.read_csv(valid_train_data)
        if 'DMS_score' in valid_train_df.columns:
            valid_train_df = valid_train_df.drop(columns=['DMS_score'])

        wogt_valid_train_data = os.path.join(step_dir, "valid_fold", f"wogt_{subdir}_train{step}_fold0.csv")
        valid_train_df.to_csv(wogt_valid_train_data)


        valid_test_data = os.path.join(step_dir, "valid_fold", f"{subdir}_test{step}_fold0.csv")
        predict_dir = os.path.join(step_dir, "predict_dir")
        gather_dir = os.path.join(step_dir, "gather_info")
        script_dir = os.path.join(step_dir, "proxy_script")

        prompt = None

        past_exp_fpath = os.path.join(record_dir, f"step{step-1}", f"experience.md")
        next_test_data = os.path.join(step_dir, f"{subdir}_test{step}.csv")

        
        # train and propose strategy

        os.makedirs(predict_dir, exist_ok=True)
        os.makedirs(os.path.join(step_dir, "proxy_script"), exist_ok=True)


        valid_train_df = pd.read_csv(valid_train_data)
        valid_train_str = tabulate(valid_train_df, headers='keys', tablefmt='grid', showindex=False)

        # select column and operator
        select_check = False
        while(not select_check):
            os.makedirs(gather_dir, exist_ok=True)
            prompt = f"""
            [Task Objective]  

                A dataset is stored in {valid_train_data}
                
                After reading it into a pd.DataFrame, the file contains the following columns:
                - mutant: please ignore this column  
                - mutated_sequence: please ignore this column  
                - DMS_score_bin: please ignore this column  
                - mut_num: please ignore this column  
                - DMS_score: this is the target variable y to be fitted  
                - other columns([esm2_150m_cmp,esm2_650m_cmp,esm2_3b_cmp,esm3_cmp,esmc_300m_cmp,esmc_600m_cmp,ProSST-4096,ProSST-2048,ProSST-1024,ProSST-512,ProSST-20,progen2-base-nll,Saprot_values,esm1v_t33_650M_UR90S_1,esm1v_t33_650M_UR90S_2,esm1v_t33_650M_UR90S_3,esm1v_t33_650M_UR90S_4,esm1v_t33_650M_UR90S_5,Ensemble_ESM1v,proteinglm-100b-int4_clm_score,proteinglm-3b-mlm_mlm_score,proteinglm-10b-mlm_mlm_score,proteinglm-1b-mlm_mlm_score,S3F,VenusREM_wetlab_msa,VenusREM_searched_msa,ESCOTT_score,AIDO.Protein-RAG-16B-zeroshot_wetlab_msa,AIDO.Protein-RAG-16B-zeroshot_searched_msa]): the remaining columns are experimentally measured floating-point values, which can be used as optional x features.

                Based on all x feature columns, construct new computational experimental metrics (proxy_metric) expressed as formulas, 
                such that the Spearman correlation coefficient between proxy_metric and DMS_score is as high as possible.  
                Note: proxy_metric must be computed **only** from x feature columns, and cannot use the y column (DMS_score).

            [Execution Steps]  
                Propose {args.strategy_nums} different fitting strategies, that each include:
                    1. Column Selection
                    Based on the past experience in {past_exp_fpath}, select 3 columns that may affect DMS_score, then randomly add 0 new columns in the rest of dataset to explore new areas.
                    e.g. [esm2_650m_cmp, ProSST-2048, ... , AIDO.Protein-RAG-16B-zeroshot_searched_msa]; 

                    2. Algorithm Selection
                    Based on the past experience in {past_exp_fpath}, based on the past experience, decide and describe an Algorithm within 50 words which could combine all column scores selected.
                    note: don't include any regression method that needs training

                please use different columns and algorithm, that will result in {args.strategy_nums} different json files generated.
                
            [Storage Requirement]  
                save selected column and selected algorithm to file {os.path.join(gather_dir, f"calculate_proxy_metric_step{step}_i.json")} , i=0,1,....,{args.strategy_nums-1}
                e.g. 
                    {{  
                        \"selected_columns\": [\"esm2_650m_cmp\", \"ProSST-2048\", ... , \"AIDO.Protein-RAG-16B-zeroshot_searched_msa\"],
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
                    Based on selected column and selected algorithm from json file, complete a script constructing proxy_metrics based on the following templete, which uses features x to approximate a proxy metric instead of y: 
                    
                    import argparse
                    import pandas as pd
                    from copy import deepcopy
                    # note: don't include any regression method

                    def calc_algo(selected_features: numpy.ndarray) -> np.ndarray :
                        # TODO : {json_dict["selected_algorithm"]}


                    def calculate_proxy_metric(x_df: pd.DataFrame = None, proxy_name: str, output_fpath: str) -> None:
                        df = deepcopy(x_df)
                        drop_cols = [c for c in ["mutant", "mutated_sequence", "DMS_score", "DMS_score_bin", "mut_num"] if c in df.columns]
                        df = df.drop(columns=drop_cols)

                        selected_columns = {json_dict["selected_columns"]}
                        X_selected = df[selected_cols].values

                        proxy_metric = calc_algo(X_selected)

                        x_df[proxy_name] = proxy_metric
                        x_df.to_csv(output_fpath)
                        

                    if __name__ == "__main__":
                        parser = argparse.ArgumentParser()
                        parser.add_argument("--input_csv", type=str, default=f"")
                        parser.add_argument("--proxy_name", type=str, default=f"")
                        parser.add_argument("--output_csv", type=str, default=f"")
                        args = parser.parse_args()

                        input_df = pd.read_csv(args.input_csv)
                        calculate_proxy_metric(input_df, args.proxy_name, args.output_csv)


                [Storage Requirement] 

                    save each script in file {os.path.join(record_dir, f"step{step}", "proxy_script", f"calculate_proxy_metric_step{step}_{j}.py")}.

                """

                agent.go(prompt)

                # check execute and produce model
                gen_check = True
                proxy_name = f"calculate_proxy_metric_step{step}_{j}"
                proxy_script_path = os.path.join(step_dir, "proxy_script", f"{proxy_name}.py")

                predict_train_data = os.path.join(predict_dir, f"{subdir}_predict_step{step}_method{j}_fold0.csv")
                try_predict_exit_code = os.system(f"python {proxy_script_path} --input_csv {wogt_valid_train_data} --proxy_name {proxy_name} --output_csv {predict_train_data}")
                if(not ( (try_predict_exit_code == 0) and (os.path.exists(predict_train_data) )  and ( os.path.exists(proxy_script_path) ))):
                    gen_check = False
                    if(os.path.exists(predict_train_data)):
                        os.remove(predict_train_data)
                    if(os.path.exists(proxy_script_path)):
                        os.remove(proxy_script_path)

          
        # try evaluator manually
        os.system(f"python forward_learning/zero-shot/reject_sampling_zeroshot.py --input_dir {step_dir} --top_k {args.strategy_nums} --n_fold {args.n_fold}")




        # gather into result df : [method, spearman, calc_operator, selected_columns]
        valid_summary_fpath = os.path.join(step_dir, f"top_{args.strategy_nums}_valid.csv")
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

        valid_dataset_summary_fpath = os.path.join(step_dir, "valid_fold", f"{subdir}_test{step}_fold0_spearman.csv")
        valid_dataset_summary_df = pd.read_csv(valid_dataset_summary_fpath)


        gather_check = False
        while(not gather_check):
            
            prompt = f"""

        [Task Objective]  
            Extract know-hows from added DataFrame, complete the following form:

            note: please don't include following suggestions in your answer: random forest, xgboost, SVM, elastic net, ridge regression and other method that needs training.

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












if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--record_dir", type=str)
    parser.add_argument("--base_url", type=str)
    parser.add_argument("--api_key", type=str)
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


