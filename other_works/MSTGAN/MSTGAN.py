import torch.nn as nn
from other_works.MSTGAN.ST_block import MST_block
import numpy as np
from scipy.sparse.linalg import eigs
import torch


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]  #35

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def calculate_laplacian_with_self_loop(matrix):
    matrix = matrix + torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian


import numpy as np
import pandas as pd
import math

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

def calculate_adjacency_matrix_and_distances(R):
    """
    根据经纬度计算邻接矩阵 A 和距离矩阵 D。
    :param R: 距离阈值（单位：千米）
    :return: 邻接矩阵 A 和距离矩阵 D
    """
    file_path = './dataset/Beijing_12/location/location.csv'
    positions_data = pd.read_csv(file_path)
    longitudes = positions_data['Longitude'].values.astype('float32')
    latitudes = positions_data['Latitude'].values.astype('float32')

    n = len(latitudes)
    A = np.zeros((n, n), dtype=float)  # 邻接矩阵
    D = np.zeros((n, n), dtype=float)  # 距离矩阵

    # 计算每对站点之间的 Haversine 距离
    for i in range(n):
        for j in range(n):
            if i != j:
                d_ij = haversine_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
                D[i, j] = d_ij  # 距离矩阵赋值
                if d_ij < R:
                    A[i, j] = 1 / d_ij  # 根据公式计算邻接矩阵的权重

    return A, D



class MSTAN(nn.Module):
    def __init__(self,input_dim,hiden_dim,out_channels,device,num_nodes,num_of_timesteps,num_for_predict,K,dropout,d_model, dataset):
        super(MSTAN,self).__init__()
        self.MST_Blocklist = nn.ModuleList()
        self.MST_Blocklist.append(MST_block(input_dim,hiden_dim,device,num_nodes,num_of_timesteps,K,dropout,d_model))
        self.MST_Blocklist.append(MST_block(hiden_dim,out_channels,device,num_nodes,num_of_timesteps,K,dropout,d_model))
        self.Predict_layer = nn.Conv2d(num_of_timesteps, num_for_predict, kernel_size=(1, out_channels))

        if dataset == "Beijing_12":
            adj, _ = calculate_adjacency_matrix_and_distances(45) # beijing_12: 45 Delhi: 12
        elif dataset == 'Delhi_12':
            adj, _ = calculate_adjacency_matrix_and_distances(12) # beijing_12: 45 Delhi: 12
        adj_mx = np.asmatrix(adj).astype(float)
        L_tilde = scaled_Laplacian(adj_mx)
        self.cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in cheb_polynomial(L_tilde, 3)]


    def forward(self,x):
        # x:[B,N,F,T]
        # Multi-Spatio-temporal Feature extraction
        x = x.transpose(2, 3)
        for block in self.MST_Blocklist:
            x= block(x,self.cheb_polynomials)
        # Prediction layer
        output = self.Predict_layer(x.permute(0, 3, 1, 2)).permute(0,2,3,1).squeeze(2)
        return output


