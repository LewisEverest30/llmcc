import torch
import math
import sys
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import time

from ae_model import Autoencoder
from dataset_ae import CcDatasetAeNoSeq, CcDatasetAeWithSeq, CcDatasetAeWithSeqOld, CcDatasetAeWithSeqPredict


# =================== 网络结构 ===================
input_dim = 3  # 输入维度 (延迟, 丢包, 带宽)
encoding_dim = 896  # 编码后的维度


# =================== 训练参数 ===================
num_epochs = 500
batch_size = 512
learning_rate = 0.0005
weight_decay = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================== 数据 ===================
# train_dataset_path = './data/data_nopac.pkl'
# train_dataset_path = './data/data_nopac_processed.pkl'
# train_dataset = gcc_dataset_ae_no_seq(train_dataset_path)
# train_dataset = gcc_dataset_ae_with_seq(train_dataset_path)
# train_dataset = gcc_dataset_ae_with_seq_old(train_dataset_path)   # 能快速收敛，且loss低

train_dataset_path_gcc = './data/gcc.pkl'
train_dataset = CcDatasetAeWithSeqPredict(train_dataset_path_gcc)

dataset_train , dataset_validation = torch.utils.data.dataset.random_split(train_dataset, [len(train_dataset)-math.floor(len(train_dataset)*0.1), math.floor(len(train_dataset)*0.1)])
print('train dataset length:', len(train_dataset))
print('valid dataset length:', len(dataset_validation))

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset_validation, batch_size=batch_size, shuffle=True)


# =================== 模型、损失函数、优化器 ===================
# 初始化模型、损失函数和优化器
model = Autoencoder(input_dim, encoding_dim).to(device)
criterion = nn.SmoothL1Loss(beta=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# ================== 训练模型 ===================
for epoch in range(num_epochs):
    start_time = time.time()
    for batch_idx, (net_states, decoded_label) in enumerate(train_loader):
        # if batch_idx == 0:
        #     print('net_states shape:', net_states.shape)
        #     print('decoded_label shape:', decoded_label.shape)
        #     for i in net_states[0]:
        #         print(i)
        #     for i in decoded_label[0]:
        #         print(i)
        optimizer.zero_grad()
        net_states = net_states.to(device) 
        decoded_label = decoded_label.to(device)
        reconstruction = model(net_states)

        # loss = criterion(reconstruction, net_states)  # 用ae输出和输入计算loss
        loss = criterion(reconstruction, decoded_label)  # 用ae输出和数据集的解码标签计算loss
        loss.backward()
        optimizer.step()
    end_time = time.time()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Time: {end_time-start_time:.2f} s')

    if (epoch+1) % 10 == 0:
        # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # 保存模型
        torch.save(model.state_dict(), f'autoencoder/checkpoint/model_{epoch+1}.pth')





