import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import os
import math
from torch.nn import CrossEntropyLoss, MSELoss, SmoothL1Loss
import torch.nn.functional as F
from tqdm import tqdm
import random

from autoencoder.ae_model import Autoencoder
from config import cfg
from reward_loss import RewardLoss
torch.autograd.set_detect_anomaly(True)

class LlmCC(torch.nn.Module):  
    def __init__(
            self, 
            net_state_autoencoder:Autoencoder,  # 网络状态编码器
            plm,                # 大模型
            plm_embed_size,     # 大模型嵌入大小
            available_resource_encode_size = 64,  # 可用资源编码长度q
            use_memory = True,  # 是否使用记忆池
            memory_pool_max_size = (cfg.seq_len+1)*20,  # 记忆池大小
            loss_threshold = 0.5,  # loss阈值（低于阈值的可以保存到memory）
            memory_save_interval = cfg.seq_len+5,  # 保存到memory的间隔
            plm_stop_layer_idx = -1,
            device = 'cuda' if torch.cuda.is_available() else 'cpu',
            ):  
        super(LlmCC, self).__init__()  
        self.device = device

        # 神经网络模块
        self.ae = net_state_autoencoder
        self.plm = plm

        # plm以外的网络，用于保存和加载
        self.modules_except_plm = nn.ModuleList([  # used to save and load modules except plm
            self.ae,

        ])

        # 记忆池相关参数
        self.use_memory = use_memory
        self.memory_pool_max_size = memory_pool_max_size
        self.loss_threshold = loss_threshold
        self.last_step_embeded_input = None
        self.last_step_predict = None   # 用于保存上一次的预测结果
        self.memory = None  # 记忆池，用于达到阈值的时候，将预测结果保存到记忆池中
        self.memory_save_interval = memory_save_interval
        self.memory_save_step = 0  # 放入一条记忆就要等待至少interval个step再考虑能否放入


    def forward(
            self, 
            net_states, 
            available_resource=None, 
            last_step_loss=None, 
            attention_mask=None
    ):          
        
        print('net_states:', net_states.shape)  # [batch_size, seq_len, windows_size, 3]
        # 通过ae编码得到词向量
        embeded_feature = self.ae.encode(net_states)  # [batch_size, seq_len, plm_embed_size]
        # 通过plm
        transformer_outputs = self.plm(
            inputs_embeds=embeded_feature,  # 使用嵌入后的向量直接传入plm模型
            output_hidden_states=True,
        )
        
        # 获取最后一个隐藏层状态，
        plm_last_hidden_state = transformer_outputs.hidden_states[-1]  # [16, 60, 896] [batch_size, seq_len, plm_embed_size]
        # 将最后一个时间步的隐藏状态保存到self.last_step_predict，形状为[batch_size, 1, plm_embed_size]
        self.last_step_predict = plm_last_hidden_state[:, -1:, :]

        # 实现方案1：只使用最后一个时间步，“最后一个时间步的预测任务”
        # 从最后一个隐藏层状态中提取出预测的词向量([batch_size, seq_len, plm_embed_size]), 取最后一个seq，也就是最后一个时间步的隐藏状态
        # plm_predict = plm_last_hidden_state[:, -1, :]    # [batch_size, 896]

        # 实现方案2：使用所有时间步的隐藏状态，“整个序列的预测任务”
        plm_predict = plm_last_hidden_state  # [batch_size, seq_len, plm_embed_size]
        
        # 通过ae解码
        predict = self.ae.decode(plm_predict)       # [batch_size, seq_len, 3]
        # print('predict:', predict.shape)
        # 只取tp的预测值
        predict = predict[: , : , -1]    # [batch_size, seq_len]
        # print('predict after:', predict.shape)
        

        return predict


    @classmethod
    def save_model(cls, model, save_dir, lora_rank):
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        if lora_rank > 0:
            # save lora weights
            model.plm.save_pretrained(save_dir)
            # save other modules except plm
            torch.save(model.modules_except_plm.state_dict(), os.path.join(save_dir, 'modules_except_plm.bin'))
        else:
            # lora is disabled, save whole model
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.bin'))

    @classmethod
    def load_model(cls, model, model_dir, lora_rank):
        if lora_rank > 0:
            # load lora weights
            model.plm.load_adapter(model_dir, adapter_name='default')
            # load other modules except plm
            model.modules_except_plm.load_state_dict(torch.load(os.path.join(model_dir, 'modules_except_plm.bin')))
        else:
            # lora is disabled, load whole model
            model.load_state_dict(torch.load(os.path.join(model_dir, 'model.bin')))
        return model


def train_LlmCC(
        model,
        train_dataset,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        lora_rank: int,
        continue_epoch: int = 0,
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        # output_loss = True,
        output_loss_dir = './output/train/',
        checkpoints_dir = './checkpoint/',
        save_checkpoint_epoch_interval = 1,
        validation_epoch_interval = 1,
):
    checkpoints_dir += f'batch_size_{batch_size}_lr_{lr}_weight_decay_{weight_decay}_lora_rank_{lora_rank}/'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if not os.path.exists(output_loss_dir):
        os.makedirs(output_loss_dir)
    output_predict_result_file = open(os.path.join(output_loss_dir, f'predict_result_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.txt'), 'w')
    
    # 切分训练数据集，分出来10%用于验证集
    dataset_train , dataset_validation = torch.utils.data.dataset.random_split(train_dataset, [len(train_dataset)-math.floor(len(train_dataset)*0.1), math.floor(len(train_dataset)*0.1)])
    print('train dataset length:', len(dataset_train))
    print('valid dataset length:', len(dataset_validation))
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset_validation, batch_size=batch_size, shuffle=True)
    
    
    # loss_fn = MSELoss()
    loss_fn = SmoothL1Loss(beta=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = model.to(device)
    model.train()
    for epoch in range(epochs):

        for batch_idx, (net_states, labels) in enumerate(train_loader):
            net_states = net_states.to(device)  # 
            labels = labels.to(device)

            optimizer.zero_grad()
            print('net_states:', net_states.shape)  # [batch_size, seq_len, windows_size, 3]
            output = model(net_states)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            # print('batch_idx:', batch_idx)
            # print('net_states:', net_states.shape)
            # print('labels:', labels.shape)
            # print('output:', output.shape)

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}, batch {batch_idx}, loss {loss.item()}")

        if (epoch+1) % save_checkpoint_epoch_interval ==  0:
            checkpoint_save_dir = os.path.join(checkpoints_dir, str(continue_epoch + epoch + 1))
            LlmCC.save_model(model, checkpoint_save_dir, lora_rank)
            print(f'Epoch {epoch+1}, checkpoint saved at:', checkpoint_save_dir)
    
        # 验证
        if (epoch+1) % validation_epoch_interval ==  0:
            model.eval()
            with torch.no_grad():
                valid_loss = 0
                for batch_idx, (net_states, labels) in enumerate(valid_loader):
                    net_states = net_states.to(device)
                    labels = labels.to(device)
                    output = model(1, net_states)
                    valid_loss += loss_fn(output, labels)
                    if epoch == epochs - 1:  # 最后一个epoch保存预测结果
                        for label, out in zip(labels, output):
                            # print("label:", label)
                            # print("output:", out)
                            for i in range(len(label)):
                                output_predict_result_file.write(f'{label[i]} {out[i]}\n')
                print(f"Epoch {epoch+1}, valid loss {valid_loss/len(valid_loader)}\n")
            model.train()


    output_predict_result_file.close()
    return model
    

def test_LlmCC(
        model,
        test_dataset,
        batch_size: int,
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_info = True,
        output_info_dir = './output/test/',
):

    output_predict_result_file = open(os.path.join(output_info_dir, f'predict_result_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.txt'), 'w')
    
    print('test dataset length:', len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 不能打乱顺序, 否则会导致时序混乱


    loss_fn = SmoothL1Loss()
    model = model.to(device)
    model.eval()
    total_inference_time = 0
    with torch.no_grad():
        test_loss = 0
        for batch_idx, (net_states, labels) in enumerate(tqdm(test_loader, desc='Testing')):
            net_states = net_states.to(device)
            labels = labels.to(device)
            start_time = datetime.now()
            output = model(1, net_states)
            end_time = datetime.now()
            total_inference_time += ((end_time - start_time).microseconds)/1000
            # print(f'{(end_time - start_time).microseconds} us')
            this_batch_loss = loss_fn(output, labels)
            test_loss += this_batch_loss
            
            for label, out in zip(labels, output):
                # print("label:", label)
                # print("output:", out)
                for i in range(len(label)):
                    output_predict_result_file.write(f'{label[i]} {out[i]}\n')
            # print(f"batch loss {this_batch_loss}")
        print(f"Total average test loss {test_loss/len(test_loader)}\n")
        print(f"Total average inference time {total_inference_time/len(test_loader)} ms\n")

    
    output_predict_result_file.close()
    return model