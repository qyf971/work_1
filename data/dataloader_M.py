import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')


class Dataset_PRSA(Dataset):
    def __init__(self, flag, size, target, border1s, border2s):
        # size = (seq_len, pred_len)
        self.seq_len = size[0]
        self.pred_len = size[1]

        self.border1s = border1s
        self.border2s = border2s

        # init
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.type = type_map[flag]

        self.target = target
        self.__read_data__()

    def get_all_data(self):
        data_list = []
        for i in range(12):
            data = pd.read_csv(f'../dataset/AQI_processed/PRSA_Data_{i + 1}.csv').iloc[:, 1:11].to_numpy()
            data_list.append(data)
        data_all = np.concatenate(data_list, axis=0)
        return data_all

    def __read_data__(self):
        border1 = self.border1s[self.type]
        border2 = self.border2s[self.type]

        self.data_all = self.get_all_data()
        self.scaler = StandardScaler()
        self.scaler.fit(self.data_all)

        features_data = []
        target_data = []
        for i in range(12):
            data = pd.read_csv(f'../dataset/AQI_processed/PRSA_Data_{i + 1}.csv').iloc[:, 1:11].to_numpy()
            scaled = self.scaler.transform(data)
            features = scaled
            if self.target == 'PM25':
                target = scaled[:, 0:1]
            elif self.target == 'PM10':
                target = scaled[:, 1:2]
            else:
                raise ValueError(f"Unsupported target: {self.target}")
            features_data.append(features)
            target_data.append(target)
        features_data = np.array(features_data)
        target_data = np.array(target_data)
        self.data_x = features_data[:, border1:border2]
        self.data_y = target_data[:, border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        features = self.data_x[:, s_begin:s_end]
        target = self.data_y[:, r_begin:r_end]

        return features, target

    def __len__(self):
        return self.data_x.shape[1] - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if self.target == 'PM25':
            mean_ = self.scaler.mean_[0]
            std_ = self.scaler.scale_[0]
        elif self.target == 'PM10':
            mean_ = self.scaler.mean_[1]
            std_ = self.scaler.scale_[1]
        else:
            raise ValueError(f"Unsupported target: {self.target}")
        data_inverse = data * std_ + mean_
        return data_inverse


def dataloader_mutil_sites(flag, size, target, batch_size, num_workers, border1s, border2s):
    data_set = Dataset_PRSA(flag, size, target, border1s, border2s)
    if flag == 'train':
        shuffle_flag = True
        drop_last = True
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
    elif flag == 'test':
        shuffle_flag = False
        drop_last = False
    else:
        raise ValueError('Unsupported flag: {}'.format(flag))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last)
    print(flag + '数据准备完成')
    return data_set, data_loader
