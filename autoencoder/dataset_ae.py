import pickle
import torch
from torch.utils.data import Dataset, DataLoader


class CcDatasetAeNoSeq(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)   # 3003 * 61

        self.batch_x = []
        self.batch_y = []        
        for j in range(len(data)):
            raw_batch_data = data[j]    # [rtt1, rtt2, ..., rtt20, loss1, loss2, ..., loss20, tp1, tp2, ..., tp20, tp21]
            for i in range(20):
                self.batch_x.append([raw_batch_data[i], raw_batch_data[i+20], raw_batch_data[i+40]/1000])
                self.batch_y.append([raw_batch_data[i], raw_batch_data[i+20], raw_batch_data[i+40]/1000])

        self.batch_x = torch.tensor(self.batch_x, dtype=torch.float32)
        self.batch_y = torch.tensor(self.batch_y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.batch_x)
    
    def __getitem__(self, idx):
        return self.batch_x[idx], self.batch_y[idx]


class CcDatasetAeWithSeq(Dataset):
    def __init__(self, data_path, window_size=20):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)   # 3003 * 61

        self.batch_x = []
        self.batch_y = []       
        for i in range(window_size-1, len(data)-1):
            x_one = []  # [[rtt1, loss1, tp1], [rtt2, loss2, tp2], ...,]  [window_size, 3]
            for j in range(window_size):  # 把当前和历史窗口的数据放到x_one中
                x_one.append(data[i-window_size+j+1])
            self.batch_x.append(x_one)
            self.batch_y.append(data[i])  # 当前网络状态是标签
        self.batch_x = torch.tensor(self.batch_x, dtype=torch.float32)
        self.batch_y = torch.tensor(self.batch_y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.batch_x)
    
    def __getitem__(self, idx):
        return self.batch_x[idx], self.batch_y[idx]


class CcDatasetAeWithSeqPredict(Dataset):
    def __init__(self, data_path, window_size=20):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)   # 3003 * 61

        self.batch_x = []
        self.batch_y = []       
        for i in range(window_size-1, len(data)-2):
            x_one = []  # [[rtt1, loss1, tp1], [rtt2, loss2, tp2], ...,]  [window_size, 3]
            for j in range(window_size):  # 把当前和历史窗口的数据放到x_one中
                x_one.append(data[i-window_size+j+1][:-1])
            self.batch_x.append(x_one)
            self.batch_y.append(data[i+1][:-1])  # 下一步的网络状态是标签
        self.batch_x = torch.tensor(self.batch_x, dtype=torch.float32)
        self.batch_y = torch.tensor(self.batch_y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.batch_x)
    
    def __getitem__(self, idx):
        return self.batch_x[idx], self.batch_y[idx]

class CcDatasetAeWithSeqOld(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)   # 3003 * 61

        self.batch_x = []
        self.batch_y = []        
        for j in range(len(data)):
            raw_batch_data = data[j]    # [rtt1, rtt2, ..., rtt20, loss1, loss2, ..., loss20, tp1, tp2, ..., tp20, tp21]
            x_one = []  # [[rtt1, loss1, tp1], [rtt2, loss2, tp2], ..., [rtt20, loss20, tp20]]
            y_one = []  # ➡️[rtt20, loss20, tp20]
            for i in range(20):
                x_one.append([raw_batch_data[i], raw_batch_data[i+20], raw_batch_data[i+40]/1000])
                if i == 19:
                    y_one = [raw_batch_data[i], raw_batch_data[i+20], raw_batch_data[i+40]/1000]
            self.batch_x.append(x_one)
            self.batch_y.append(y_one)
        self.batch_x = torch.tensor(self.batch_x, dtype=torch.float32)
        self.batch_y = torch.tensor(self.batch_y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.batch_x)
    
    def __getitem__(self, idx):
        return self.batch_x[idx], self.batch_y[idx]