import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import os
from _Support.Graph_Construction_Beijing_12 import haversine_distance, jensen_shannon_divergence


def get_data():
    PM25_years = []
    PM10_years = []
    years = ['2020', '2021', '2022', '2023']   # 共四年
    for year in years:
        PM25 = []
        PM10 = []
        for filename in os.listdir('./dataset/Delhi/' + year):
            if filename.endswith(".csv"):
                df = pd.read_csv('./dataset/Delhi/2020/' + filename).iloc[:, 1:3]
                df = df.interpolate(method='linear')
                df = df.bfill()
                PM25.append(df.iloc[:, 0])
                PM10.append(df.iloc[:, 1])
        PM25 = np.array(PM25)
        PM10 = np.array(PM10)
        PM25_years.append(PM25)
        PM10_years.append(PM10)

    PM25 = np.concatenate(PM25_years, axis=1)
    PM10 = np.concatenate(PM10_years, axis=1)
    return PM25, PM10


def calculate_the_distance_matrix(threshold):
    file_path = './dataset/Delhi/location/location.csv'
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

    std_deviation = np.std(distance_matrix.flatten())   # 计算标准差
    distance_matrix_exp = np.exp(-(distance_matrix ** 2) / (std_deviation ** 2))
    np.fill_diagonal(distance_matrix_exp, 0.0)
    adj_matrix = np.where(distance_matrix_exp > threshold, distance_matrix_exp, 0.0)
    # 获取边的索引和权重
    mask = distance_matrix_exp > threshold
    edge_index = np.array(np.where(mask))
    edge_weights = distance_matrix_exp[mask]
    return adj_matrix, edge_index, edge_weights


# 邻居图
def calculate_the_neighbor_matrix():
    adj_matrix = pd.read_csv('./dataset/Beijing_12/neighbors/neighbors.csv', index_col=0).values.astype('float32')
    np.fill_diagonal(adj_matrix, 0.0)
    mask = adj_matrix != 0
    edge_index = np.array(np.where(mask))
    return adj_matrix, edge_index

# 相似图
def calculate_the_similarity_matrix(threshold, target):
    PM25, PM10 = get_data()
    data = PM25 if target == 'PM25' else PM10
    num_sites = data.shape[0]
    js_divergence_matrix = np.zeros((num_sites, num_sites))
    for i in range(num_sites):
        data1 = data[i]
        for j in range(i + 1, num_sites):
            data2 = data[j]
            js_divergence = jensen_shannon_divergence(data1, data2)
            js_divergence_matrix[i, j] = js_divergence
            js_divergence_matrix[j, i] = js_divergence  # 因为 JS 散度是对称的
    # 计算标准差
    std_deviation = np.std(js_divergence_matrix)
    js_divergence_matrix_exp = np.exp(-(js_divergence_matrix ** 2) / (std_deviation ** 2))
    np.fill_diagonal(js_divergence_matrix_exp, 0.0)
    adj_matrix = np.where(js_divergence_matrix_exp > threshold, js_divergence_matrix_exp, 0.0)
    # 获取边的索引和权重
    mask = js_divergence_matrix_exp > threshold
    edge_index = np.array(np.where(mask))
    edge_weights = js_divergence_matrix_exp[mask]
    return adj_matrix, edge_index, edge_weights


def calculate_the_correlation_matrix(threshold, target):
    PM25, PM10 = get_data()
    data = PM25 if target == 'PM25' else PM10
    num_sites = data.shape[0]
    correlation_matrix = np.zeros((num_sites, num_sites))
    # 计算所有站点两两之间的皮尔森相关系数
    for i in range(num_sites):
        for j in range(i, num_sites):
            if i == j:
                correlation_matrix[i, j] = 1.0  # 同一站点之间的相关系数为1
            else:
                corr, _ = pearsonr(data[i], data[j])
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr  # 相关系数矩阵是对称的
    np.fill_diagonal(correlation_matrix, 0.0)
    adj_matrix = np.where(correlation_matrix > threshold, correlation_matrix, 0.0)
    # 获取边的索引和权重
    mask = correlation_matrix > threshold
    edge_index = np.array(np.where(mask))
    edge_weights = correlation_matrix[mask]
    return adj_matrix, edge_index, edge_weights

