import os
import pandas as pd

ROOT_DIR = "./RLMC_final_预测结果"
SAVE_PATH = "./实验结果汇总/all_test_metrics.csv"

all_rows = []

for model_name in os.listdir(ROOT_DIR):
    model_path = os.path.join(ROOT_DIR, model_name)
    if not os.path.isdir(model_path):
        continue

    for pred_len in os.listdir(model_path):
        pred_path = os.path.join(model_path, pred_len)
        if not os.path.isdir(pred_path):
            continue

        csv_path = os.path.join(pred_path, "test_metrics.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        # 添加列
        df["pred_len"] = int(pred_len)
        df["model_name"] = model_name

        # ===== 数值保留位数 =====
        if "test_MAE" in df.columns:
            df["test_MAE"] = df["test_MAE"].astype(float).round(3)

        if "test_RMSE" in df.columns:
            df["test_RMSE"] = df["test_RMSE"].astype(float).round(3)

        if "test_IA" in df.columns:
            df["test_IA"] = df["test_IA"].astype(float).round(4)

        if "test_R2" in df.columns:
            df["test_R2"] = df["test_R2"].astype(float).round(4)

        all_rows.append(df)

# 拼接
final_df = pd.concat(all_rows, ignore_index=True)

# 排序（推荐）
final_df = final_df.sort_values(
    by=["model_name", "pred_len"]
).reset_index(drop=True)

# 调整列顺序
cols = ["model_name", "pred_len"] + \
       [c for c in final_df.columns if c not in ["model_name", "pred_len"]]
final_df = final_df[cols]

# 保存
final_df.to_csv(SAVE_PATH, index=False)

print("✅ 已生成:", SAVE_PATH)
print(final_df)
