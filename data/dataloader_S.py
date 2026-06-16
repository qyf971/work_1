import warnings

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')


class Dataset_PRSA(Dataset):
    def __init__(self, flag, size, target, data_path, border1s, border2s):
        # size [seq_len, pred_len]
        self.seq_len = size[0]
        self.pred_len = size[1]

        self.border1s = border1s
        self.border2s = border2s

        self.data_path = data_path

        # init
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.type = type_map[flag]

        self.target = target
        self.__read_data__()

    def __read_data__(self):
        border1 = self.border1s[self.type]
        border2 = self.border2s[self.type]

        data = pd.read_csv(self.data_path).iloc[:, 1:11].to_numpy()
        self.scaler = StandardScaler()
        self.scaler.fit(data)
        data_norm = self.scaler.transform(data)
        self.data_x = data_norm[border1:border2]
        if self.target == 'PM25':
            self.data_y = data_norm[:, 0:1][border1:border2]
        elif self.target == 'PM10':
            self.data_y = data_norm[:, 1:2][border1:border2]
        else:
            raise ValueError(f"Unsupported target: {self.target}")

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        features = self.data_x[s_begin:s_end]
        target = self.data_y[r_begin:r_end]

        return features, target

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

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


def dataloader_S(flag, size, target, batch_size, num_workers, data_path, border1s, border2s):
    data_set = Dataset_PRSA(flag, size, target, data_path, border1s, border2s)
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
        raise ValueError(f"Unsupported flag: {flag}")
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last)
    print(flag + '数据准备完成')
    return data_set, data_loader
