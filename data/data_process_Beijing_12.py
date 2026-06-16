import pandas as pd


def wind_direction_to_number(data):
    # 创建风向到数值的映射关系
    wd_mapping = {
        'N': 0,
        'NNE': 1,
        'NE': 2,
        'ENE': 3,
        'E': 4,
        'ESE': 5,
        'SE': 6,
        'SSE': 7,
        'S': 8,
        'SSW': 9,
        'SW': 10,
        'WSW': 11,
        'W': 12,
        'WNW': 13,
        'NW': 14,
        'NNW': 15
    }
    # 将'wd'列中的字符串值映射为数值
    data['wd'] = data['wd'].map(wd_mapping)
    return data


def data_fill(data):
    data = wind_direction_to_number(data)
    data['wd'] = data['wd'].interpolate(methon='nearest')
    data = data.interpolate(methon='linear')
    data = data.bfill()
    return data


def merge_date_columns(data):
    data['date'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
    data.insert(0, 'date', data.pop('date'))
    data.drop(['year', 'month', 'day', 'hour'], axis=1, inplace=True)
    # data.drop(['No', 'station', 'PRES', 'RAIN'], axis=1, inplace=True)
    data.drop(['No', 'PRES', 'RAIN'], axis=1, inplace=True)
    return data


def process_for_each_file():
    for i in range(12):
        df = pd.read_csv('../dataset/AQI/PRSA_Data_{}.csv'.format(i + 1))
        df = data_fill(df)
        df = merge_date_columns(df)
        # station = df['station'].iloc[0]
        df.to_csv('../dataset/AQI_processed/PRSA_Data_{}.csv'.format(i + 1), index=False)
        # df.drop(['station'], axis=1, inplace=True)
        # df.to_csv('../dataset/cleaned_data/{}.csv'.format(station), index=False)


def csv_load():  # 拼接数据集
    file_list = []
    for file_index in range(6):
        file_list.append('../dataset/AQI_processed/PRSA_Data_{}.csv'.format(file_index + 1))
    dfs = [pd.read_csv(file, encoding='utf-8') for file in file_list]
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv('../dataset/AQI_processed/PRSA_Data.csv', index=False)


if __name__ == '__main__':
    process_for_each_file()

    csv_load()
