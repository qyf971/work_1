import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def data_standardization():
    columns = ['PM25_Concentration', 'weather', 'temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction']
    data_dir = '../dataset/Beijing_36/stations_data'
    data_list = []
    for filename in os.listdir(data_dir):
        data = pd.read_csv(os.path.join(data_dir, filename), usecols=columns).to_numpy()
        data_list.append(data)
    data_all = np.concatenate(data_list, axis=0)
    scaler = StandardScaler()
    scaler.fit(data_all)

    PM25_mean = scaler.mean_[0]
    PM25_std = scaler.scale_[0]

    data_norm_list = []
    for data in data_list:
        data_norm = scaler.transform(data)
        data_norm_list.append(data_norm)
    data_norm = np.array(data_norm_list)

    return data_norm, PM25_mean, PM25_std

def generate_input_target(data, seq_len, predict_len, target):
    """
        生成输入和目标数据。

        参数:
        - data: 形状为 (num_nodes, T, F) 的三维数组。
        - seq_len: 输入序列长度。
        - predict_len: 预测序列长度。
        - target: 目标特征的名称，例如 'PM25' 或 'PM10'。

        返回:
        - X: 形状为 (num_samples, num_nodes, seq_len, F) 的输入数据。
        - y: 形状为 (num_samples, num_nodes, predict_len, 1) 的目标数据。
        """
    num_nodes, T, F = data.shape
    num_samples = T - seq_len - predict_len + 1

    X = np.zeros((num_samples, num_nodes, seq_len, F))
    y = np.zeros((num_samples, num_nodes, predict_len, 1))

    target_index = 0  # 默认为 PM25 的索引
    if target == 'PM10':
        target_index = 1  # 假设 PM10 是第二个特征

    for i in range(num_samples):
        X[i] = data[:, i:i + seq_len]
        y[i] = data[:, i + seq_len:i + seq_len + predict_len, target_index:target_index + 1]

    return X, y

def split_data(X, y, train_ratio, val_ratio, test_ratio, random_state=None):
    """
    按照指定的比例从前往后划分数据集。

    参数:
    - X: 输入数据，形状为 (num_samples, ...)。
    - y: 目标数据，形状为 (num_samples, ...)。
    - train_ratio: 训练集比例。
    - val_ratio: 验证集比例。
    - test_ratio: 测试集比例。
    - random_state: 随机种子，用于 reproducibility（可选）。

    返回:
    - X_train, y_train: 训练集的输入和目标数据。
    - X_val, y_val: 验证集的输入和目标数据。
    - X_test, y_test: 测试集的输入和目标数据。
    """
    # 计算样本数量
    num_samples = X.shape[0]

    # 检查比例之和是否为1
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must be 1.0")

    # 计算划分索引
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    test_size = num_samples - train_size - val_size

    # 如果提供了 random_state，先随机打乱数据
    if random_state is not None:
        np.random.seed(random_state)
        indices = np.random.permutation(num_samples)
        X = X[indices]
        y = y[indices]

    # 从前往后依次划分数据
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == '__main__':
    seq_len = 24
    predict_len_list = [1, 2, 3, 4, 5, 6]
    train_ratio, val_ratio, test_ratio = 0.7, 0.1, 0.2
    for predict_len in predict_len_list:
        Output_dir = Path('../dataset/Beijing_36/train_val_test_data') / f'{predict_len}'
        Output_dir.mkdir(parents=True, exist_ok=True)

        data_norm, PM25_mean, PM25_std = data_standardization()

        scaler_PM25 = np.array([PM25_mean, PM25_std])

        np.save(Output_dir / 'scaler_PM25.npy', scaler_PM25)

        print(f"input_len: {seq_len} predict_len: {predict_len} 数据处理开始")
        for target in ['PM25']:
            X, y = generate_input_target(data_norm, seq_len, predict_len, target)
            X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, train_ratio, val_ratio, test_ratio)
            np.savez_compressed(Output_dir / f'train_{target}.npz', X=X_train, y=y_train)
            np.savez_compressed(Output_dir / f'val_{target}.npz', X=X_val, y=y_val)
            np.savez_compressed(Output_dir / f'test_{target}.npz', X=X_test, y=y_test)
            print(f"{target}_train_X: {X_train.shape}  {target}_train_y: {y_train.shape}")
            print(f"{target}_val_X:   {X_val.shape}    {target}_val_y:   {y_val.shape}")
            print(f"{target}_test_X:  {X_test.shape}   {target}_test_y:  {y_test.shape}")
        print(f"input_len: {seq_len} predict_len: {predict_len} 数据处理完毕")