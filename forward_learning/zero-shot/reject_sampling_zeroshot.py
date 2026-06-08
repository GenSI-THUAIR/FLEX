
import os
import argparse
import pandas as pd
from datetime import datetime
import numpy as np
from tqdm import tqdm
import shutil
from scipy.stats import spearmanr
import importlib.util
from uuid import uuid4
import joblib
from copy import deepcopy
import json



def merge_and_average_csv(subdir, fold_valid_dir, step, n_fold):
    csv_files = []
    for i in range(n_fold):
        csv_files.append(os.path.join(fold_valid_dir, f"{subdir}_test{step}_fold{i}_spearman.csv"))

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="feature", how="inner", suffixes=("", "_dup"))

    corr_cols = [c for c in merged.columns if "spearman_correlation" in c]
    merged["spearman_correlation_avg"] = merged[corr_cols].mean(axis=1)

    final = merged[["feature", "spearman_correlation_avg"]]

    final = final.sort_values("spearman_correlation_avg", ascending=False)
    final = final.rename(columns={"spearman_correlation_avg": "spearman_correlation"})

    # 保存
    output_path = os.path.join(fold_valid_dir, f"{subdir}_test{step}_spearman_merged.csv")
    final.to_csv(output_path, index=False)
    print(f"Saved merged average CSV to: {output_path}")


def add_proxy_metric_column(x_df: pd.DataFrame, test_df: pd.DataFrame, model_path, proxy_name) -> pd.DataFrame:

    model = joblib.load(model_path)

    drop_cols = [c for c in ["mutant", "mutated_sequence", "DMS_score", "DMS_score_bin"] if c in test_df.columns]
    test_df = test_df.drop(columns=drop_cols)
    print(f"x columns: {len(test_df.columns)}")
    X = test_df.values

    preds = model.predict(X)
    preds = np.asarray(preds)

    x_df[proxy_name] = preds
    return x_df


def main(args):
    summary = {
        "target": [],
        "spearman_correlation": [],
        "selected_columns": [],
        "selected_algorithm": [],
    }

    base_dir = "".join(args.input_dir.split("/")[:-1])
    step = int( args.input_dir.split("/")[-1].split("step")[-1] )
    subdir = str(args.input_dir.split("/")[-3])

    exp_dir = os.path.join(base_dir, f"step{step-1}", "top_train_script")

    method_dir = os.path.join(args.input_dir, "proxy_script")
    gather_dir = os.path.join(args.input_dir, "gather_info")


    # cross-validation of n folds
    valid_spearmans = {}
    fold_valid_dir = os.path.join(args.input_dir, "valid_fold")

    test_best_col = None
    for i in range(args.n_fold):
        corr_summary = {
            "feature": [],
            "spearman_correlation": []
        }
        
        train_df_path = os.path.join(fold_valid_dir, f"{subdir}_train{step}_fold{i}.csv")
        train_df = pd.read_csv(train_df_path)
        test_df_path = os.path.join(fold_valid_dir, f"{subdir}_test{step}_fold{i}.csv")
        test_df = pd.read_csv(test_df_path)

        # calculate best zero-shot test score
        test_best_corr = 0.0
        exclude_cols = {"DMS_score", "mutant", "mutated_sequence", "DMS_score_bin", "DMS_bin_score", "mut_num", "SD"}

        for col in test_df.columns:
            if col in exclude_cols:
                continue

            corr, _ = spearmanr(test_df[col], test_df['DMS_score'])
            corr_summary["feature"].append(col)
            corr_summary["spearman_correlation"].append(corr)

            if abs(corr) > abs(test_best_corr):
                test_best_corr = abs(corr)
                test_best_col = col
        if(i == 0):
            valid_spearmans["test_best_col"] = [test_best_corr]
        else:
            valid_spearmans["test_best_col"].append(test_best_corr)

        corr_summary_df = pd.DataFrame(corr_summary)
        corr_summary_df = corr_summary_df.sort_values(by="spearman_correlation", ascending=False)
        corr_summary_df.to_csv(os.path.join(fold_valid_dir, f"{subdir}_test{step}_fold{i}_spearman.csv"), index=False )


        orig_cols = set(test_df.columns.tolist())

        # add proxy metric
        added_df = deepcopy(test_df )
        predicted_df_path = os.path.join(fold_valid_dir, f"{subdir}_predict{step}_fold{i}.csv")
        added_df.to_csv(predicted_df_path)
        for python_file in tqdm(os.listdir(method_dir)):
            if(not python_file.endswith(".py")):
                continue
            print(f"python_path: {python_file}")
            python_fpath = os.path.join(method_dir, python_file)

            proxy_name = python_file.split(".")[0]
            python_id = int(proxy_name.split("_")[-1] )

            os.system(f"python {python_fpath} --input_csv {predicted_df_path} --proxy_name {proxy_name} --output_csv {predicted_df_path}")

        added_df = pd.read_csv(predicted_df_path)
        added_df["DMS_score"] = test_df["DMS_score"].values
        added_df = added_df.drop(columns=['Unnamed: 0'])


        # find top-k
        new_cols = [c for c in added_df.columns if c not in orig_cols and c != "DMS_score"]
        new_cols = [c for c in new_cols if(c.startswith("calculate_proxy_metric"))]
        print(f"new_cols = {new_cols}")
        col_corrs = {}
        for c in new_cols:
            corr, _ = spearmanr(added_df[c], added_df["DMS_score"])
            col_corrs[c] = abs(corr)

        
        for column, cor_score in col_corrs.items():
            if(i == 0):
                valid_spearmans[column] = [cor_score]
            else:
                valid_spearmans[column].append(cor_score)

    # # process valid test_best_col
    merge_and_average_csv(subdir, fold_valid_dir, step, args.n_fold)

    for method_i, score_array_i in valid_spearmans.items():
        summary["target"].append(method_i)
        summary["spearman_correlation"].append(np.mean(score_array_i))

        # for method_i, find selcted column and selected operator
        if(method_i == "test_best_col"):
            summary["selected_columns"].append([test_best_col])
            summary["selected_algorithm"].append("You may keep trying more any other algorithm")
        else:
            gather_info_path = os.path.join(gather_dir, f"{method_i}.json")
            gather_info = None
            with open(gather_info_path, "r") as f:
                gather_info = json.load(f)
            summary["selected_columns"].append(gather_info["selected_columns"])
            summary["selected_algorithm"].append(gather_info["selected_algorithm"])

        
    valid_spearmans = pd.DataFrame(valid_spearmans)
    valid_spearmans.to_csv(os.path.join(args.input_dir, "cross_valid_record.csv"))


    summary_df = pd.DataFrame(summary)

    # set status gap
    oracle_value = summary_df.loc[summary_df["target"] == "test_best_col", "spearman_correlation"].iloc[0]
    summary_df["status"] = summary_df["spearman_correlation"].apply(
        lambda x: (("above_oracle" if x > oracle_value 
                else ("equal" if x == oracle_value else "below")))
    )
    summary_df["status_gap"] = summary_df["spearman_correlation"].apply(
        lambda x: (x - oracle_value)
    )

    summary_df = summary_df.sort_values(by="spearman_correlation", ascending=False)



    # for i in range(0, args.top_k):

    #     os.makedirs(os.path.join(args.input_dir, "top_proxy_metrics"), exist_ok=True)
        
    #     colname, corr_val = sorted_cols[i]
    #     summary["target"].append(colname)
    #     summary["spearman_correlation"].append(corr_val)

    #     # copy high-score method
    #     src_fpath = os.path.join(method_dir, f"{colname}.py")
    #     dst_fpath = os.path.join(args.input_dir, "top_proxy_metrics", f"{colname}.py")
    #     shutil.copy(src_fpath, dst_fpath)

    # summary_df = pd.DataFrame(summary)

    output_cross_valid_fpath = os.path.join(args.input_dir, f"top_{args.top_k}_valid.csv")
    summary_df.to_csv(output_cross_valid_fpath)
    print(f"[saved] {output_cross_valid_fpath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/AIRvePFS/ai4science/users/yupei/Biomni_dev/test_cases/2025.10.9-10.12/self_evolve_corr_20251018110122/A4GRB6_PSEAI_Chen_2020/0_fold/step2")
    parser.add_argument("--n_fold", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()
    main(args)