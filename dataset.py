import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from config import cfg

# 与nsdi26一样的数据集格式
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


# label改为了当前的tc限速
class CcDatasetUseTc(Dataset):
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
                x_one.append(data[i-window_size+j+1][:-1])  # 不要最后一个元素，即tc
            y_one = data[i][-1]  # 把当前的tc放到y_one中
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

    def __len__(self):
        return len(self.batch_x)
    
    def __getitem__(self, idx):
        return self.batch_x[idx], self.batch_y[idx]
    

# label改为了当前的tc限速，且适配全连接的编码器
class CcDatasetUseTcNoWindow(Dataset):
    def __init__(self, data_path, seq_len=cfg.seq_len, tp_thres_inc=200, tp_thres_dec=100):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)   # len1 * 3  // [rtt1, loss1, tp1], [rtt2, loss2, tp2], ...
        
        # print('data length:', len(data))
        # print('data[0] length:', len(data[0]))

        self.batch_x = []
        self.batch_y = []       
        for i in range(seq_len-1, len(data)-1):
            x_one = []  # [[rtt1, loss1, tp1], [rtt2, loss2, tp2], ...,]  [seq_len, 3]
            y_one = []  # [tp]  [1]
            for j in range(seq_len):  # 向前回溯seq_len个时间步的数据
                x_one.append(data[i-seq_len+j+1][:-1])  # 不要最后一个元素，即tc
            next_tc = data[i+1][-1]
            now_tp = data[i][2]
            if next_tc - now_tp >= tp_thres_inc:
                y_one = 1 # 表示当前吞吐量距离未来的tc还有一定距离，可以升高码率
            elif next_tc - now_tp < tp_thres_inc and next_tc - now_tp > tp_thres_dec:
                y_one = 0 # 表示当前吞吐量稍高于未来的tc，维持码率
            else:
                y_one = 2 # 表示当前吞吐量已经超过了未来的tc，需要下降码率
            self.batch_x.append(x_one)
            self.batch_y.append(y_one)
        self.batch_x = torch.tensor(self.batch_x, dtype=torch.float32)
        self.batch_y = torch.tensor(self.batch_y, dtype=torch.long)  # 分类任务，标签为整数
        print('self.batch_x shape:', self.batch_x.shape)
        print('self.batch_y shape:', self.batch_y.shape)


    def __len__(self):
        return len(self.batch_x)
    
    def __getitem__(self, idx):
        return self.batch_x[idx], self.batch_y[idx]
    