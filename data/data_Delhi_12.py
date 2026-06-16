import numpy as np
import os
import pandas as pd
from pathlib import Path

# ==========================================================
# 【全局配置】对应你的 Beijing_12 配置
# ==========================================================
columns = ['AQI']  # 只有 AQI 一个特征
data_dir = "./dataset/Delhi_12/AQI/delhi_aqi.csv"  # 你合并好的CSV
save_root = "./dataset/Delhi_12/train_val_test_data"        # 保存根目录


# ==========================================================
# 数据读取与格式转换（单独拆分，便于先划分数据集再归一化）
# ==========================================================
def load_and_reshape_data():
    # 读取合并好的 AQI 数据
    df = pd.read_csv(data_dir, index_col='datetime', parse_dates=True)
    df = df.sort_index()
    data = df.values  # shape: [T, N]

    # 格式转为 [N, T, F]  适配后续函数
    N = data.shape[1]
    T = data.shape[0]
    F = 1
    data_out = data.T.reshape(N, T, 1)  # [N, T, 1]

    return data_out


# ==========================================================
# 最大最小归一化（仅用训练集计算min/max，避免数据泄露）
# ==========================================================
def data_standardization_max_min_AQI(train_data, all_data):
    # 仅用训练集计算全局最大最小（关键修改）
    train_data_reshaped = train_data.reshape(-1, 1)  # 展平训练集，计算整体min/max
    data_min = np.min(train_data_reshaped)
    data_max = np.max(train_data_reshaped)

    # 归一化函数（避免分母为0）
    def norm(x):
        if data_max == data_min:
            return np.zeros_like(x)
        return (x - data_min) / (data_max - data_min)

    # 用训练集的min/max，归一化所有数据（训练集、验证集、测试集）
    data_scaled = norm(all_data)

    AQI_min = data_min
    AQI_max = data_max

    return data_scaled, AQI_min, AQI_max


# ==========================================================
# 生成 X, y（和你格式完全一致）
# X: [N, T, F]
# y: [N, T_out]
# ==========================================================
def generate_input_target_AQI(data, seq_len, predict_len):
    num_nodes, T, F = data.shape
    num_samples = T - seq_len - predict_len + 1

    X = np.zeros((num_samples, num_nodes, seq_len, F))
    y = np.zeros((num_samples, num_nodes, predict_len))

    for i in range(num_samples):
        X[i] = data[:, i:i + seq_len]
        y[i] = data[:, i + seq_len:i + seq_len + predict_len, 0]

    return X, y


# ==========================================================
# 数据集划分（和你格式完全一致）
# ==========================================================
def split_data(X, y, train_ratio, val_ratio, test_ratio, random_state=None):
    num_samples = X.shape[0]
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("比例和必须为1")

    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)

    if random_state is not None:
        np.random.seed(random_state)
        indices = np.random.permutation(num_samples)
        X = X[indices]
        y = y[indices]

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ==========================================================
# 主函数：完全对齐你的风格，关键修改「训练集归一化」逻辑
# ==========================================================
if __name__ == '__main__':
    predict_len_list = [1, 6, 12, 24]
    seq_len = 72
    train_ratio, val_ratio, test_ratio = 0.5, 0.25, 0.25

    # 1. 先读取并转换所有原始数据（未归一化）
    all_data = load_and_reshape_data()  # shape: [N, T, F]
    num_nodes, T, F = all_data.shape

    for predict_len in predict_len_list:
        Output_dir = Path(save_root) / f'{seq_len}_{predict_len}'
        Output_dir.mkdir(parents=True, exist_ok=True)

        # 2. 先构造所有样本（未归一化），再划分数据集
        X_all, y_all = generate_input_target_AQI(all_data, seq_len, predict_len)
        # 划分训练/验证/测试集（此时数据未归一化）
        X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw = split_data(
            X_all, y_all, train_ratio, val_ratio, test_ratio
        )

        # 3. 仅用训练集计算归一化参数，归一化所有数据集（关键步骤）
        # 提取训练集输入数据，用于计算min/max
        train_input_data = X_train_raw  # shape: [train_samples, N, seq_len, F]
        # 归一化所有数据（训练、验证、测试）
        X_train, AQI_min, AQI_max = data_standardization_max_min_AQI(train_input_data, X_train_raw)
        X_val = (X_val_raw - AQI_min) / (AQI_max - AQI_min) if AQI_max != AQI_min else np.zeros_like(X_val_raw)
        X_test = (X_test_raw - AQI_min) / (AQI_max - AQI_min) if AQI_max != AQI_min else np.zeros_like(X_test_raw)
        # 标签也用相同参数归一化
        y_train = (y_train_raw - AQI_min) / (AQI_max - AQI_min) if AQI_max != AQI_min else np.zeros_like(y_train_raw)
        y_val = (y_val_raw - AQI_min) / (AQI_max - AQI_min) if AQI_max != AQI_min else np.zeros_like(y_val_raw)
        y_test = (y_test_raw - AQI_min) / (AQI_max - AQI_min) if AQI_max != AQI_min else np.zeros_like(y_test_raw)

        # 4. 保存 scaler：[min, max]（训练集计算得到）
        scaler_AQI = np.array([AQI_min, AQI_max])
        np.save(Output_dir / 'scaler_AQI.npy', scaler_AQI)

        print(f"input_len:{seq_len} predict_len:{predict_len} 数据处理开始")

        target = 'AQI'
        # 保存归一化后的数据集
        np.savez_compressed(Output_dir / f'train_{target}.npz', X=X_train, y=y_train)
        np.savez_compressed(Output_dir / f'val_{target}.npz', X=X_val, y=y_val)
        np.savez_compressed(Output_dir / f'test_{target}.npz', X=X_test, y=y_test)

        print(f"{target}_train_X: {X_train.shape}  {target}_train_y: {y_train.shape}")
        print(f"{target}_val_X:   {X_val.shape}    {target}_val_y:   {y_val.shape}")
        print(f"{target}_test_X:  {X_test.shape}   {target}_test_y:  {y_test.shape}")
        print(f"归一化参数（训练集计算）：AQI_min={AQI_min:.2f}, AQI_max={AQI_max:.2f}")
        print(f"input_len:{seq_len} predict_len:{predict_len} 数据处理完毕\n")