import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import entropy
import numpy as np
import math
import os


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    计算两个经纬度坐标之间的大圆距离（单位：千米）。
    """
    # 地球平均半径，单位为公里
    R = 6371.004

    # 将角度转换为弧度
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # 差值
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine 公式
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


def jensen_shannon_divergence(p, q):
    """
    计算两个概率分布 p 和 q 的 Jensen-Shannon 散度。
    """
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)


# 距离图
def calculate_the_distance_matrix(threshold):
    file_path = '../dataset/Beijing_12/location/location.csv'
    positions_data = pd.read_csv(file_path)

    stations = positions_data['station']
    longitudes = positions_data['Longitude'].values.astype('float32')
    latitudes = positions_data['Latitude'].values.astype('float32')

    # 创建一个 n×n 的距离矩阵
    num_stations = len(stations)
    distance_matrix = np.zeros((num_stations, num_stations))

    # 计算所有站点两两之间的距离
    for i in range(num_stations):
        for j in range(num_stations):
            if i == j:
                distance_matrix[i, j] = 0.0  # 同一站点之间的距离为0
            else:
                distance_matrix[i, j] = haversine_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])

    # 计算标准差
    std_deviation = np.std(distance_matrix.flatten())
    distance_matrix_exp = np.exp(-(distance_matrix ** 2) / (std_deviation ** 2))
    # for i in range(num_stations):
    #     distance_matrix_exp[i, i] = 0.0
    adj_matrix = np.array([[value if value > threshold else 0 for value in row] for row in distance_matrix_exp])
    mask = distance_matrix_exp > threshold
    edge_index = np.array(np.where(mask))
    edge_weights = distance_matrix_exp[mask]
    return adj_matrix, edge_index, edge_weights


# 邻居图
def calculate_the_neighbor_matrix():
    adj_matrix = pd.read_csv('../dataset/Beijing_12/neighbors/neighbors.csv', index_col=0).values.astype('float32')
    for i in range(len(adj_matrix)):
        adj_matrix[i, i] = 0.0
    mask = adj_matrix != 0
    edge_index = np.array(np.where(mask))
    return adj_matrix, edge_index


# 相似图
def calculate_the_similarity_matrix(threshold):
    folder_path = '../dataset/Beijing_12/AQI_processed/'
    file_names = [f'PRSA_Data_{i}.csv' for i in range(1, 13)]
    # 初始化结果矩阵
    num_files = len(file_names)
    js_divergence_matrix = np.zeros((num_files, num_files))
    # 遍历所有文件组合
    for i, file1 in enumerate(file_names):
        data1 = pd.read_csv(os.path.join(folder_path, file1))
        # 提取 PM2.5 列
        pm25_data1 = data1['PM2.5']
        for j, file2 in enumerate(file_names):
            if i == j:
                continue
            data2 = pd.read_csv(os.path.join(folder_path, file2))
            # 提取 PM2.5 列
            pm25_data2 = data2['PM2.5']
            # 计算两个 PM2.5 列的 JS 散度
            js_divergence = jensen_shannon_divergence(pm25_data1, pm25_data2)
            # 存储结果
            js_divergence_matrix[i, j] = js_divergence
            js_divergence_matrix[j, i] = js_divergence  # 因为 JS 散度是对称的
    # 计算标准差
    std_deviation = np.std(js_divergence_matrix.flatten())
    js_divergence_matrix_exp = np.exp(-(js_divergence_matrix ** 2) / (std_deviation ** 2))
    for i in range(len(js_divergence_matrix_exp)):
        js_divergence_matrix_exp[i, i] = 0.0
    adj_matrix = np.array([[value if value > threshold else 0 for value in row] for row in js_divergence_matrix_exp])
    mask = js_divergence_matrix_exp > threshold
    edge_index = np.array(np.where(mask))
    edge_weights = js_divergence_matrix_exp[mask]
    return adj_matrix, edge_index, edge_weights


# 相关图
def calculate_the_correlation_matrix(threshold):
    # 文件路径前缀和站点数量
    file_base_path = "../dataset/Beijing_12/AQI_processed/PRSA_Data_{}.csv"
    num_stations = 12

    # 加载所有站点的PM2.5数据
    pm25_data = []
    for i in range(1, num_stations + 1):
        file_path = file_base_path.format(i)
        df = pd.read_csv(file_path)
        pm25_data.append(df['PM2.5'].values)

    # 计算所有站点对之间的皮尔森相关系数
    correlation_matrix = np.zeros((num_stations, num_stations))
    for i in range(num_stations):
        for j in range(i, num_stations):
            corr, _ = pearsonr(pm25_data[i], pm25_data[j])
            correlation_matrix[i, j] = corr
            correlation_matrix[j, i] = corr

    # 计算标准差
    # std_deviation = np.std(correlation_matrix.flatten())
    # correlation_matrix_exp = np.exp(-(correlation_matrix ** 2) / (std_deviation ** 2))
    # mask = correlation_matrix_exp > threshold
    adj_matrix = np.array([[value if value > threshold else 0 for value in row] for row in correlation_matrix])
    for i in range(len(correlation_matrix)):
        correlation_matrix[i, i] = 0.0
    mask = correlation_matrix > threshold
    edge_index = np.array(np.where(mask))
    # edge_weights = correlation_matrix_exp[mask]
    edge_weights = correlation_matrix[mask]
    return adj_matrix, edge_index, edge_weights
