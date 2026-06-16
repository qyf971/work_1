import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import entropy
from scipy.stats import spearmanr
import numpy as np
import math
from dtaidistance import dtw
import os
from sklearn.metrics.pairwise import cosine_similarity


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

def jensen_shannon_distance(p, q):
    return math.sqrt(jensen_shannon_divergence(p, q))


def get_data(target):
    folder_path = './dataset/Beijing_12/AQI_processed/'
    file_names = [f'PRSA_Data_{i}.csv' for i in range(1, 13)]
    # 一次性读取所有文件并缓存数据
    data_list = []
    for file_name in file_names:
        data = pd.read_csv(os.path.join(folder_path, file_name))
        if target == 'PM25':
            target_data = data['PM2.5'].values
        elif target == 'PM10':
            target_data = data['PM10'].values
        else:
            raise ValueError(f"Invalid target: {target}")
        data_list.append(target_data)
    return data_list


def calculate_the_distance_matrix_STCN():
    file_path = './dataset/Delhi_12/location/location.csv'
    positions_data = pd.read_csv(file_path)
    longitudes = positions_data['Longitude'].values.astype('float32')
    latitudes = positions_data['Latitude'].values.astype('float32')
    
    num_sites = len(latitudes)
    distance_matrix = np.zeros((num_sites, num_sites))
    
    for i in range(num_sites):
        for j in range(i + 1, num_sites):
            dist = haversine_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    # 构造距离倒数邻接矩阵（论文原版）
    adj_matrix = np.zeros_like(distance_matrix)
    for i in range(num_sites):
        for j in range(num_sites):
            if i != j and distance_matrix[i, j] > 0:
                adj_matrix[i, j] = 1.0 / distance_matrix[i, j]

    # 全连接边
    mask = np.ones_like(adj_matrix).astype(bool)
    np.fill_diagonal(mask, False)
    
    edge_index = np.array(np.where(mask))
    edge_weights = adj_matrix[mask]
    
    return adj_matrix, edge_index, edge_weights

def calculate_the_distance_matrix_GC_LSTM(R):
    file_path = './dataset/Delhi_12/location/location.csv'
    positions_data = pd.read_csv(file_path)
    longitudes = positions_data['Longitude'].values.astype('float32')
    latitudes = positions_data['Latitude'].values.astype('float32')
    
    num_sites = len(latitudes)
    distance_matrix = np.zeros((num_sites, num_sites))
    
    # 1. 计算两两站点的Haversine距离
    for i in range(num_sites):
        for j in range(i + 1, num_sites):
            dist = haversine_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    # 2. 严格按照公式构造邻接矩阵
    adj_matrix = np.zeros_like(distance_matrix)
    for i in range(num_sites):
        for j in range(num_sites):
            if i != j and distance_matrix[i, j] < R:
                adj_matrix[i, j] = 1.0 / distance_matrix[i, j]
            else:
                adj_matrix[i, j] = 0.0

    # 3. 提取边索引和边权重（适配PyTorch Geometric等图学习框架）
    mask = (adj_matrix != 0)
    edge_index = np.array(np.where(mask))
    edge_weights = adj_matrix[mask]
    
    return adj_matrix, edge_index, edge_weights


# 距离图
def calculate_the_distance_matrix(threshold):
    file_path = './dataset/Beijing_12/location/location.csv'
    positions_data = pd.read_csv(file_path)
    stations = positions_data['site_name']
    longitudes = positions_data['Longitude'].values.astype('float32')
    latitudes = positions_data['Latitude'].values.astype('float32')
    # 创建一个 n×n 的距离矩阵
    num_sites = len(stations)
    distance_matrix = np.zeros((num_sites, num_sites))
    # 计算所有站点两两之间的距离
    for i in range(num_sites):
        for j in range(i + 1, num_sites):
            distance_matrix[i, j] = haversine_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
            distance_matrix[j, i] = distance_matrix[i, j]
    # 计算标准差和均值
    std_deviation = np.std(distance_matrix.flatten())
    mean_distance = np.mean(distance_matrix.flatten())
    print(f"Distance Matrix -  $\sigma$: 16.00, $\epsilon$: {threshold:.2f}")
    distance_matrix_exp = np.exp(-(distance_matrix ** 2) / (16 ** 2))
    adj_matrix = np.where(distance_matrix_exp >= threshold, distance_matrix_exp, 0.0)
    mask = distance_matrix_exp >= threshold
    edge_index = np.array(np.where(mask))
    edge_weights = distance_matrix_exp[mask]
    return adj_matrix, edge_index, edge_weights


# 邻居图
def calculate_the_neighbor_matrix(R):
    """
    根据经纬度计算 0-1 邻接矩阵（邻居矩阵），包含自环。
    :param R: 距离阈值（单位：千米）
    :return: 邻接矩阵 A, edge_index 和 edge_weights
    """
    file_path = './dataset/Beijing_12/location/location.csv'
    positions_data = pd.read_csv(file_path)
    longitudes = positions_data['Longitude'].values.astype('float32')
    latitudes = positions_data['Latitude'].values.astype('float32')

    n = len(latitudes)
    # 初始化全 0 矩阵
    A = np.zeros((n, n), dtype='float32')
    edge_index = []
    edge_weights = []

    for i in range(n):
        for j in range(n):
            # 1. 对角线元素：每个站点必然是自己的邻居
            if i == j:
                A[i, j] = 1.0
                edge_index.append([i, j])
                edge_weights.append(1.0)
            # 2. 非对角线元素：计算 Haversine 距离
            else:
                d_ij = haversine_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
                # 如果距离小于阈值 R，判定为邻居
                if d_ij <= R:
                    A[i, j] = 1.0
                    edge_index.append([i, j])
                    edge_weights.append(1.0)
    print(f"Neighbor Matrix -  $\epsilon$: {R:.2f}")
    # 转换为 PyG 所需的 Tensor 格式前，先转为 numpy 数组
    edge_index = np.array(edge_index).T  # 形状变为 (2, num_edges)
    edge_weights = np.array(edge_weights).astype('float32')

    return A, edge_index, edge_weights


# 分布相似图
def calculate_the_similarity_matrix(threshold, target):
    data_list = get_data(target)
    # 初始化结果矩阵
    num_sites = len(data_list)
    js_divergence_matrix = np.zeros((num_sites, num_sites))
    # 遍历所有文件组合，计算 JS 散度
    for i in range(num_sites):
        for j in range(i + 1, num_sites):
            js_divergence = jensen_shannon_distance(data_list[i], data_list[j])
            js_divergence_matrix[i, j] = js_divergence
            js_divergence_matrix[j, i] = js_divergence  # 因为 JS 散度是对称的
    # 计算标准差
    std_deviation = np.std(js_divergence_matrix.flatten())
    mean_distance = np.mean(js_divergence_matrix.flatten())
    print(std_deviation)
    print(f"Distributional Similarity Matrix - $\sigma$: 0.20, $\epsilon$: {threshold:.2f}")
    js_divergence_matrix_exp = np.exp(-(js_divergence_matrix ** 2) / (0.2 ** 2))
    # 应用阈值过滤
    adj_matrix = np.where(js_divergence_matrix_exp >= threshold, js_divergence_matrix_exp, 0.0)
    # 获取边的索引和权重
    mask = js_divergence_matrix_exp >= threshold
    edge_index = np.array(np.where(mask))
    edge_weights = js_divergence_matrix_exp[mask]
    return adj_matrix, edge_index, edge_weights


def calculate_poi_cosine_similarity(threshold=0.8):
    """
    计算站点POI余弦相似度矩阵，并根据阈值构建图邻接矩阵
    小于阈值的相似度置为 0，对角线保持为 1
    :return: numpy 格式的邻接矩阵 (12,12)
    """
    # 读取POI数据
    df = pd.read_csv("get_POI/beijing_12site_20poi.csv")

    # 提取20类POI特征
    poi_features = df.iloc[:, 3:].values

    # 计算余弦相似度
    adj_matrix_POI = cosine_similarity(poi_features)

    # 阈值处理
    adj_matrix_POI[adj_matrix_POI < threshold] = 0
    np.fill_diagonal(adj_matrix_POI, 1.0)

    return adj_matrix_POI