
import os
import argparse
import pandas as pd
from datetime import datetime
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
import importlib.util
from uuid import uuid4
from copy import deepcopy
import joblib
import json

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

    summary = {
        "target_name": [],
        # "train_top5_mean_corr": [],
        "train_best_corr_zero_shot": [],
        "test_mean_corr": [],
        "test_best_corr_zero_shot": [],
        "test_best_corr_copy_train_zero_shot": [],
    }

    # find top_num
    top_num = args.top_num
    for i in range(0, top_num):
        summary[f"proxy_metric_{i+1}_name"] = []
        # summary[f"proxy_metric_{i+1}_train_corr"] = []
        summary[f"proxy_metric_{i+1}_test_corr"] = []



    # for proteingym zero-shot prediction
    ref_fpath = args.data_dir
    ref_df = pd.read_csv( os.path.join(ref_fpath, "summary.csv") )

    for subdir in tqdm( os.listdir(args.input_dir) ):
        if("." in subdir):
            continue

        done_exp_file = os.path.join(args.input_dir, subdir, "0_fold", f"step{args.step}", "experience.md")
        if(not os.path.exists(done_exp_file)):
            continue

        cross_valid_summary_fpath = os.path.join(args.input_dir, subdir, "0_fold", f"step{args.step}", f"top_{args.top_num}_valid.csv")
        if(not os.path.exists(cross_valid_summary_fpath)):
            continue

        print(f"processing {subdir}")

        summary["target_name"].append(subdir)

        # load test data with DMS score
        train_data_fpath = os.path.join(args.data_dir, subdir, f"{subdir}_train100.csv")
        test_data_fpath = os.path.join(args.data_dir, subdir, f"{subdir}_left100.csv")
        test_data_df = pd.read_csv(test_data_fpath)

        # get zero-shot correlation from ref_df
        summary["train_best_corr_zero_shot"].append( ref_df[ ref_df["target"]==subdir ]["train_best_corr"].values[0] )
        summary["test_best_corr_zero_shot"].append( ref_df[ ref_df["target"]==subdir ]["test_best_corr"].values[0] )
        summary["test_best_corr_copy_train_zero_shot"].append( ref_df[ ref_df["target"]==subdir ]["copy_best_corr"].values[0] )


        # for test score
        script_dir = os.path.join(args.input_dir, subdir, f"0_fold", f"step{args.step}", "proxy_script")
        model_dir = os.path.join(args.input_dir, subdir, f"0_fold", f"step{args.step}", "models")
        gather_dir = os.path.join(args.input_dir, subdir, f"0_fold", f"step{args.step}", "gather_info")
        fold_valid_dir = os.path.join(args.input_dir, subdir, f"0_fold", f"step{args.step}", "valid_fold")
        test_data_wogt_fpath = os.path.join(args.data_dir, subdir, f"wogt_{subdir}_left100.csv")
        test_input_df = pd.read_csv(test_data_wogt_fpath)
        orig_cols = set(test_input_df.columns.tolist())

        added_df = deepcopy(test_input_df)
        # print(f"added_df: {added_df.head()}")
        predicted_df_path = os.path.join(fold_valid_dir, f"{subdir}_testset{args.step}_fold0.csv")
        added_df.to_csv(predicted_df_path)

        # read valid sorted df
        valid_df = pd.read_csv(cross_valid_summary_fpath )
        top_metrics = valid_df["target"].to_list()
        top_metrics = [metric for metric in top_metrics if(metric != "test_best_col")]

        for top_metric in tqdm(top_metrics):
            python_fpath = os.path.join(script_dir, f"{top_metric}.py")
            model_fpath = os.path.join(model_dir, f"{top_metric}_test.joblib")
            json_fapth = os.path.join(gather_dir, f"{top_metric}.json")
            json_dict = None
            with open(json_fapth, "r") as f:
                json_dict = json.load(f)
            selected_columns = json_dict["selected_columns"]

            # model training
            os.system(f"python {python_fpath} --input_csv {train_data_fpath} --ckpt_path {model_fpath}")

            added_df = add_proxy_metric_column(added_df, test_input_df, model_fpath, top_metric, selected_columns)

        added_df["DMS_score"] = test_data_df["DMS_score"].values
        # TODO 2: for each column startswith proxy, calculate spearman correlation between this column and DMS_score column
        # sort those correlations from high to low value, then 
        # fill in form of summary[f"proxy_metric_{i+1}_name"] by i_th highest correlation produced by python script filename
        # fill i summary[f"proxy_metric_{i+1}_test_corr"] by correlation i_th highest value
        new_cols = top_metrics
        print(f"new_cols = {new_cols}")


        col_corrs = {}
        for c in new_cols:
            corr, _ = spearmanr(added_df[c], added_df["DMS_score"])
            col_corrs[c] = abs(corr)

        sorted_cols = list(col_corrs.items() )

        top_corr_values = [v for (_, v) in sorted_cols if not pd.isna(v)]
        summary["test_mean_corr"].append(np.mean(top_corr_values))
        print(f"sorted_cols: {sorted_cols}")


        for i in range(0, top_num):
            key_name = f"proxy_metric_{i+1}_name"
            key_test_corr = f"proxy_metric_{i+1}_test_corr"
            
            colname, corr_val = sorted_cols[i]
            summary[key_name].append(colname)
            summary[key_test_corr].append(corr_val)

    # add mean score
    summary["target_name"].append("mean")
    summary["train_best_corr_zero_shot"].append(np.nanmean(summary["train_best_corr_zero_shot"]))
    summary["test_best_corr_zero_shot"].append(np.nanmean(summary["test_best_corr_zero_shot"]))
    summary["test_best_corr_copy_train_zero_shot"].append(np.nanmean(summary["test_best_corr_copy_train_zero_shot"]))
    summary["test_mean_corr"].append(np.nanmean(summary["test_mean_corr"]))
    for i in range(top_num):
        summary[f"proxy_metric_{i+1}_name"].append("mean")
        # summary[f"proxy_metric_{i+1}_train_corr"].append(np.nanmean(summary[f"proxy_metric_{i+1}_train_corr"]))
        summary[f"proxy_metric_{i+1}_test_corr"].append(np.nanmean(summary[f"proxy_metric_{i+1}_test_corr"]))

    for key, value in summary.items():
        print(f"{key}: {len(value)}")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(args.input_dir, f"summary_step{args.step}.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/AIRvePFS/ai4science/users/yupei/Biomni_dev/forward_learning/few-shot/self_evolve_corr_20251021033337_edit10")
    parser.add_argument("--data_dir", type=str, default="/AIRvePFS/ai4science/users/yupei/data/ProteinGym/ProteinGym_split_63")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--top_num", type=int, default=10)
    args = parser.parse_args()
    main(args)