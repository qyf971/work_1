import numpy as np
import os
import pandas as pd
from pathlib import Path

# ==========================================================
# 【仅读取原始数据，不做任何归一化】
# ==========================================================
def load_raw_data():
    data_list = []
    columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'WSPM']
    data_dir = './dataset/Beijing_12/cleaned_data'
    for filename in os.listdir(data_dir):
        data = pd.read_csv(os.path.join(data_dir, filename), usecols=columns).to_numpy()
        data_list.append(data)
    data_all = np.concatenate(data_list, axis=0)  # (总时间步, 特征数)
    num_nodes = 12  # 北京12个站点
    T, F = data_all.shape
    data_3d = data_all.reshape(num_nodes, T // num_nodes, F)  # (12, T, 12)
    return data_3d

# ==========================================================
# Max-Min 归一化工具
# ==========================================================
def max_min_normalize(data, min_vals, max_vals):
    return (data - min_vals) / (max_vals - min_vals)

# ==========================================================
# 生成序列 X, y（不变）
# ==========================================================
def generate_input_target(data, seq_len, predict_len, target):
    num_nodes, T, F = data.shape
    num_samples = T - seq_len - predict_len + 1
    X = np.zeros((num_samples, num_nodes, seq_len, F))
    y = np.zeros((num_samples, num_nodes, predict_len))
    target_index = 0 if target == 'PM25' else 1
    for i in range(num_samples):
        X[i] = data[:, i:i + seq_len]
        y[i] = data[:, i + seq_len:i + seq_len + predict_len, target_index]
    return X, y

# ==========================================================
# 划分数据集（时间序列必须按顺序，不能打乱！）
# ==========================================================
def split_data(X, y, train_ratio, val_ratio, test_ratio):
    num_samples = X.shape[0]
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    return X_train, y_train, X_val, y_val, X_test, y_test

# ==========================================================
# 主函数：正确流程：原始数据 → 构造序列 → 划分 → 训练集归一化
# ==========================================================
if __name__ == '__main__':
    predict_len_list = [1, 3, 6]
    seq_len = 24
    train_ratio, val_ratio, test_ratio = 0.5, 0.25, 0.25

    # 1. 读取【原始数据】（未归一化！）
    data_raw = load_raw_data()

    for predict_len in predict_len_list:
        Output_dir = Path('./dataset/Beijing_12/train_val_test_data') / f'{seq_len}_{predict_len}'
        Output_dir.mkdir(parents=True, exist_ok=True)

        print(f"input_len:{seq_len} predict_len:{predict_len} 数据处理开始")

        for target in ['PM25']:
            # 2. 构造序列（未归一化！）
            X, y = generate_input_target(data_raw, seq_len, predict_len, target)

            # 3. 划分训练/验证/测试（未归一化！）
            X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw = split_data(
                X, y, train_ratio, val_ratio, test_ratio
            )

            # ==========================================================
            # 4. ✅ 【核心】仅用训练集 X_train_raw 计算 min/max
            # ==========================================================
            train_flat = X_train_raw.reshape(-1, X_train_raw.shape[-1])
            feat_min = np.min(train_flat, axis=0)
            feat_max = np.max(train_flat, axis=0)

            # 取出 PM25 的参数（用于后续反归一化）
            PM25_min = feat_min[0]
            PM25_max = feat_max[0]

            # 5. 用训练集参数归一化所有数据
            X_train = max_min_normalize(X_train_raw, feat_min, feat_max)
            X_val = max_min_normalize(X_val_raw, feat_min, feat_max)
            X_test = max_min_normalize(X_test_raw, feat_min, feat_max)

            # y 也用同样参数归一化
            y_train = max_min_normalize(y_train_raw, PM25_min, PM25_max)
            y_val = max_min_normalize(y_val_raw, PM25_min, PM25_max)
            y_test = max_min_normalize(y_test_raw, PM25_min, PM25_max)

            # 6. 保存归一化参数
            np.save(Output_dir / 'scaler_PM25.npy', np.array([PM25_min, PM25_max]))

            # 7. 保存数据集
            np.savez_compressed(Output_dir / f'train_{target}.npz', X=X_train, y=y_train)
            np.savez_compressed(Output_dir / f'val_{target}.npz', X=X_val, y=y_val)
            np.savez_compressed(Output_dir / f'test_{target}.npz', X=X_test, y=y_test)

            print(f"{target}_train_X: {X_train.shape}  {target}_train_y: {y_train.shape}")
            print(f"{target}_val_X:   {X_val.shape}    {target}_val_y:   {y_val.shape}")
            print(f"{target}_test_X:  {X_test.shape}   {target}_test_y:  {y_test.shape}")

        print(f"input_len:{seq_len} predict_len:{predict_len} 数据处理完毕\n")