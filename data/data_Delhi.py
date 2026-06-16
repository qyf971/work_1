import os
import numpy as np
import pandas as pd


def preprocess_data():
    PM25_years = []
    PM10_years = []
    years = ['2020', '2021', '2022', '2023']   # 共四年
    for year in years:
        PM25 = []
        PM10 = []
        for filename in os.listdir('../dataset/Delhi/' + year):
            if filename.endswith(".csv"):
                df = pd.read_csv('../dataset/Delhi/2020/' + filename).iloc[:, 1:3]
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

    mean_PM25 = np.mean(PM25.flatten())
    std_PM25 = np.std(PM25.flatten())
    scaler_PM25 = np.array([mean_PM25, std_PM25])

    mean_POM10 = np.mean(PM10.flatten())
    std_PM10 = np.std(PM10.flatten())
    scaler_PM10 = np.array([mean_POM10, std_PM10])

    PM25_norm = (PM25 - mean_PM25) / std_PM25
    PM10_norm = (PM10 - mean_POM10) / std_PM10

    np.save('../dataset/Delhi/preprocessed_data/PM25_norm.npy', PM25_norm)
    np.save('../dataset/Delhi/preprocessed_data/scaler_PM25.npy', scaler_PM25)
    np.save('../dataset/Delhi/preprocessed_data/PM10_norm.npy', PM10_norm)
    np.save('../dataset/Delhi/preprocessed_data/scaler_PM10.npy', scaler_PM10)


if __name__ == '__main__':
    preprocess_data()