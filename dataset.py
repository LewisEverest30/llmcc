import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from config import cfg

class CcDataset(Dataset):
    def __init__(self, data_path, seq_len=cfg.seq_len, window_size=20):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)   # len1 * 3  // [rtt1, loss1, tp1], [rtt2, loss2, tp2], ...
        
        # print('data length:', len(data))
        # print('data[0] length:', len(data[0]))

        self.batch_x = []
        self.batch_y = []       
        x_this_seq = []
        y_this_seq = [] 
        for i in range(window_size-1, len(data)-1):
            x_one = []  # [[rtt1, loss1, tp1], [rtt2, loss2, tp2], ...,]  [window_size, 3]
            y_one = []  # [tp]  [1]
            for j in range(window_size):  # 把当前和历史窗口的数据放到x_one中
                x_one.append(data[i-window_size+j+1])
            y_one = data[i+1][-1]  # 把下一个时间步的tp放到y_one中
            x_this_seq.append(x_one)
            y_this_seq.append(y_one)
            if len(x_this_seq) == seq_len and len(y_this_seq) == seq_len:
                self.batch_x.append(x_this_seq)  # [seq_len, window_size, 3]
                self.batch_y.append(y_this_seq)  # [seq_len, 1]
                x_this_seq = []
                y_this_seq = []
        self.batch_x = torch.tensor(self.batch_x, dtype=torch.float32)
        print('self.batch_x shape:', self.batch_x.shape)
        self.batch_y = torch.tensor(self.batch_y, dtype=torch.float32)
        print('self.batch_y shape:', self.batch_y.shape)
    

    # def __init__(self, data_path, seq_len=cfg.seq_len, window_size=20):
    #     with open(data_path, 'rb') as f:
    #         data = pickle.load(f)   # len1 * 3  // [rtt1, loss1, tp1], [rtt2, loss2, tp2], ...
        
    #     # print('data length:', len(data))
    #     # print('data[0] length:', len(data[0]))

    #     self.batch_x = []
    #     self.batch_y = []       
    #     x_this_seq = []
    #     y_this_seq = [] 
    #     for i in range(window_size-1, len(data)-1-seq_len):
    #         for k in range(seq_len):
    #             x_one = []  # [[rtt1, loss1, tp1], [rtt2, loss2, tp2], ...,]  [window_size, 3]
    #             y_one = []  # [tp]  [1]
    #             for j in range(window_size):  # 把当前和历史窗口的数据放到x_one中
    #                 x_one.append(data[i+k-window_size+j+1])
    #             y_one = data[i+k+1][-1]  # 把下一个时间步的tp放到y_one中
    #             x_this_seq.append(x_one)
    #             y_this_seq.append(y_one)
    #             if len(x_this_seq) == seq_len and len(y_this_seq) == seq_len:
    #                 self.batch_x.append(x_this_seq)  # [seq_len, window_size, 3]
    #                 self.batch_y.append(y_this_seq)  # [seq_len, 1]
    #                 x_this_seq = []
    #                 y_this_seq = []
    #     self.batch_x = torch.tensor(self.batch_x, dtype=torch.float32)
    #     print('self.batch_x shape:', self.batch_x.shape)
    #     self.batch_y = torch.tensor(self.batch_y, dtype=torch.float32)
    #     print('self.batch_y shape:', self.batch_y.shape)


    def __len__(self):
        return len(self.batch_x)
    
    def __getitem__(self, idx):
        return self.batch_x[idx], self.batch_y[idx]



class Stage2Dataset(Dataset):
    def __init__(self, data_path):
        saved_data = torch.load(data_path)
        self.net_states = saved_data['net_states']
        self.available_resources = saved_data['available_resources']
        self.label = saved_data['label']
    
    def __len__(self):
        return len(self.net_states)
    
    def __getitem__(self, idx):
        return self.net_states[idx], self.available_resources[idx], self.label[idx]