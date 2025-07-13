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
from data_server import DataServer
from encoder import NetStateEncoder
from head import BitrateRegressionHead, ActionClassificationHead


torch.autograd.set_detect_anomaly(True)

class LlmCC(torch.nn.Module):  
    def __init__(
            self, 
            # net_state_autoencoder:Autoencoder,  # 网络状态编码器
            plm,                # 大模型
            plm_embed_size,     # 大模型嵌入大小
            device = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        super(LlmCC, self).__init__()  
        self.device = device

        # 神经网络模块
        # self.ae = net_state_autoencoder
        self.net_states_encoder = NetStateEncoder(state_encoded_dim=plm_embed_size)
        self.plm = plm
        # self.head = BitrateRegressionHead(plm_embed_size=plm_embed_size)  # 输出码率的回归头
        self.head = ActionClassificationHead(plm_embed_size=plm_embed_size)  # 输出动作的分类头

        # plm以外的网络，用于保存和加载
        self.modules_except_plm = nn.ModuleList([  # used to save and load modules except plm
            # self.ae,
            self.net_states_encoder,
            self.head,

        ])

    def forward(
            self, 
            net_states, 
            available_resource=None, 
            last_step_loss=None, 
            attention_mask=None
    ):          
        
        # print('net_states:', net_states.shape)  # [batch_size, seq_len, 3]

        # # 通过ae编码得到词向量
        # embeded_feature = self.ae.encode(net_states)  # [batch_size, seq_len, plm_embed_size]

        # 通过encoder编码得到词向量
        embeded_feature = self.net_states_encoder(net_states)  # [batch_size, seq_len, plm_embed_size]

        # 通过plm
        transformer_outputs = self.plm(
            inputs_embeds=embeded_feature,  # 使用嵌入后的向量直接传入plm模型
            output_hidden_states=True,
        )
        
        # 获取最后一个隐藏层状态，的最后一个token的输出
        plm_last_hidden_state = transformer_outputs.hidden_states[-1]  # [batch_size, seq_len, plm_embed_size]
        plm_predict = plm_last_hidden_state[:, -1, :]  # [batch_size, plm_embed_size]

        # # 通过ae解码
        # predict = self.ae.decode(plm_predict)
        
        # 通过head得到预测码率
        predict = self.head(plm_predict)  # [batch_size, 1]

        # squeeze the output
        predict = predict.squeeze(dim=1)  # [batch_size]  去掉最后一维
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
        model: LlmCC,
        train_dataset,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        lora_rank: int,
        continue_epoch: int = 0,
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
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
    # loss_fn = SmoothL1Loss(beta=0.1)
    loss_fn = CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = model.to(device)
    model.train()
    for epoch in range(epochs):

        for batch_idx, (net_states, labels) in enumerate(train_loader):
            net_states = net_states.to(device)  # 
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(net_states)
            # print("output:", output)
            # print("labels:", labels)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

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
                    output = model(net_states)
                    valid_loss += loss_fn(output, labels)
                    if epoch == epochs - 1:  # 最后一个epoch保存预测结果
                        for label, out in zip(labels, output):
                            # print("label:", label)
                            # print("output:", out)
                            for i in range(len(label)):
                                output_predict_result_file.write(f'{label[i]} {out[i]}\n')
                        for label, out in zip(labels, output.argmax(dim=1)):
                            output_predict_result_file.write(f'{label} {out}\n')
                print(f"Epoch {epoch+1}, valid loss {valid_loss/len(valid_loader)}\n")
            model.train()


    output_predict_result_file.close()
    return model
    

def test_LlmCC_offline(
        model,
        test_dataset,
        batch_size: int,
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_info_dir = './output/test/',
):
    if not os.path.exists(output_info_dir):
        os.makedirs(output_info_dir)
    output_predict_result_file = open(os.path.join(output_info_dir, f'predict_result_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.txt'), 'w')
    
    print('test dataset length:', len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 不能打乱顺序, 否则会导致时序混乱


    # loss_fn = MSELoss()
    # loss_fn = SmoothL1Loss(beta=0.1)
    loss_fn = CrossEntropyLoss()
    model = model.to(device)
    model.eval()
    total_inference_time = 0
    with torch.no_grad():
        test_loss = 0
        for batch_idx, (net_states, labels) in enumerate(tqdm(test_loader, desc='Testing')):
            net_states = net_states.to(device)
            labels = labels.to(device)
            start_time = datetime.now()
            output = model(net_states)
            end_time = datetime.now()
            total_inference_time += ((end_time - start_time).microseconds)/1000
            # print(f'{(end_time - start_time).microseconds} us')
            this_batch_loss = loss_fn(output, labels)
            test_loss += this_batch_loss

            # for label, out in zip(labels, output):
            #     # print("label:", label)
            #     # print("output:", out)
            #     for i in range(len(label)):
            #         output_predict_result_file.write(f'{label[i]} {out[i]}\n')
            for label, out in zip(labels, output.argmax(dim=1)):
                output_predict_result_file.write(f'{label} {out}\n')
            # print(f"batch loss {this_batch_loss}")
        print(f"Total average test loss {test_loss/len(test_loader)}\n")
        print(f"Total average inference time {total_inference_time/len(test_loader)} ms\n")

    
    output_predict_result_file.close()
    return model


def test_LlmCC_online(
        data_server: DataServer,
        model: LlmCC,
        collect_epochs: int,
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_info_dir = './output/test/',
        batches_per_epoch = 10000,   # 由于没有数据集。一个epoch可以定义为某个固定的步数
        dataset_save_dir = './data/collect/',
        enable_torch_jit = True,  # 是否启用JIT编译优化
):
    if not os.path.exists(output_info_dir):
        os.makedirs(output_info_dir)
    output_predict_result_file = open(os.path.join(output_info_dir, f'predict_result_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.txt'), 'w')
    
    # loss_fn = SmoothL1Loss()
    loss_fn = CrossEntropyLoss()

    model = model.to(device)
    model.eval()
    

    net_states_buffer = None  # 网络数据的缓冲区
    collected_raw_dataset = []  # 用于保存数据集  [batches_per_epoch, 4] (rtt, net_loss, throughput, available_resources)


    for epoch in range(collect_epochs):
        print('Collecting Dataset Epoch:', epoch)

        with torch.no_grad():
            for batch_idx in tqdm(range(batches_per_epoch), desc='  Processing batches'):
                # 等待接收新的网络特征RTCP数据 （阻塞等待的时间视为上个周期的时间）
                recv_tuple = data_server.receive_net_features_and_available_resources_from_webrtc_sender()
                print(f'    接收数据：{recv_tuple}')
                # 提取各个网络特征
                if recv_tuple is None:
                    raise ValueError('  接收数据失败')
                else:
                    rtt, net_loss, throughput, available_resources_this_step= recv_tuple
                
                # 累积序列的网络数据--AE
                # new_net_states = torch.tensor([rtt, net_loss, throughput], dtype=torch.float32, device=device).view(1, 1, 3)  # [1, 1, 3] 直接在GPU上创建
                # if net_states_buffer is None:
                #     net_states_buffer = new_net_states
                # else:
                #     net_states_buffer = torch.cat([net_states_buffer, new_net_states], dim=1)
                # if net_states_buffer is not None and net_states_buffer.shape[1] > 2*cfg.seq_len-1: # 达到序列长度后，移除最旧的
                #     net_states_buffer = net_states_buffer[:, 1:, :]
                # if net_states_buffer.shape[1] < 2*cfg.seq_len-1:  
                #     # 不足2*seq_len-1，则继续累积数据
                #     # 因为要构造[bath_size, seq_len, seq_len, 3]的输入，所以要等到有2*seq_len-1个数据才能开始训练
                #     data_server.send_ack_to_webrtc_sender()  # 发ack使发送端不被recv阻塞
                #     continue

                # 累积序列的网络数据--Encoder
                new_net_states = torch.tensor([rtt, net_loss, throughput], dtype=torch.float32, device=device).view(1, 1, 3)  # [1, 1, 3] 直接在GPU上创建
                # print(f'    New net states: {new_net_states.shape}, {new_net_states}')  
                if net_states_buffer is None:
                    net_states_buffer = new_net_states
                else:
                    net_states_buffer = torch.cat([net_states_buffer, new_net_states], dim=1)
                    # print(f'    Updated net states buffer: {net_states_buffer.shape}, {net_states_buffer}')
                if net_states_buffer is not None and net_states_buffer.shape[1] > cfg.seq_len: # 达到序列长度后，移除最旧的
                    net_states_buffer = net_states_buffer[:, 1:, :]
                if net_states_buffer.shape[1] < cfg.seq_len:
                    data_server.send_ack_to_webrtc_sender()  # 发ack使发送端不被recv阻塞
                    continue


                # 输入到大模型
                before_inference_time = datetime.now()
                # print(f'    Inference input shape: {new_net_states.shape}')  # [1, seq_len, seq_len, 3]
                # predicted_bitrate = model(new_net_states)
                predicted_logits = model(new_net_states)
                print(f'    predicted_logits: {predicted_logits.shape}, {predicted_logits}')  # [1, 3] 预测的动作 logits
                # loss = loss_fn(predicted_bitrate, torch.tensor([available_resources_this_step], dtype=torch.float32).to(device))  # 计算损失
                after_inference_time = datetime.now()
                inference_time = (after_inference_time - before_inference_time).microseconds / 1000  # 转换为毫秒
                print(f'    Inference time: {inference_time} ms')

                # 发送预测码率到发送端，发送端会阻塞等待该码率的到来，然后再考虑发送下一个网络数据（条件是收到码率且有新的网络数据，当rtcp周期长于推理周期的时候，主要阻塞在等新的网络数据）
                # print(f"    Epoch {epoch+1}, Batch {batch_idx+1}, Predicted Bitrate: {predicted_bitrate.item()}, Loss: {loss.item()}")
                predicted_class = predicted_logits.argmax(dim=1)     # [1]
                predicted_class_value = predicted_class.item()       # 0、1 或 2

                # print(f"    Epoch {epoch+1}, Batch {batch_idx+1}, Predicted Bitrate: {predicted_bitrate.item()}")
                # data_server.send_predicted_bitrate_to_webrtc_sender(predicted_bitrate.item())  # 发送预测码率到发送端

                print(f"    Epoch {epoch+1}, Batch {batch_idx+1}, Predicted Class: {predicted_class_value}")
                data_server.send_predicted_bitrate_to_webrtc_sender(predicted_class_value)  # 发送预测动作到发送端

                # output_predict_result_file.write(f'Epoch:{epoch}\tBatch:{batch_idx}\tPredicted:{predicted_bitrate.item()}\tLoss:\n')
                output_predict_result_file.write(f'Epoch:{epoch}\tBatch:{batch_idx}\tPredicted:{predicted_class_value}\n')

                # 保存数据集
                if not(epoch == collect_epochs-1 and batch_idx >= batches_per_epoch-2):
                    collected_raw_dataset.append(
                        [
                            rtt, 
                            net_loss, 
                            throughput, 
                            available_resources_this_step, 
                        ]
                    )

    
    # 保存数据集  数据集长度为collect_epochs*batches_per_epoch-(2*seq_len-2)-2
    print('dataset length:', len(collected_raw_dataset))
    if not os.path.exists(dataset_save_dir):
        os.makedirs(dataset_save_dir)
    dataset_file_name = f'dataset_from_model{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
    with open(os.path.join(dataset_save_dir, dataset_file_name), 'wb') as f:
        pickle.dump(collected_raw_dataset, f)
    print('dataset saved at:', os.path.join(dataset_save_dir, dataset_file_name))
    return os.path.join(dataset_save_dir, dataset_file_name)