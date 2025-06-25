import torch
import torch.nn as nn


# 定义自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim=3, encoding_dim=896, lstm_hidden_size=64, lstm_bidirectional=True):
        super(Autoencoder, self).__init__()

        self.encoder_lstm1 = nn.LSTM(input_dim, lstm_hidden_size, batch_first=True, bidirectional=lstm_bidirectional)
        self.encoder_fc1 = nn.Linear(lstm_hidden_size*2 if lstm_bidirectional else lstm_hidden_size, encoding_dim)
        self.encoder_bn1 = nn.BatchNorm1d(encoding_dim)
        self.encoder_do1 = nn.Dropout(p=0.1)

        self.decoder_fc1 = nn.Linear(encoding_dim, 128)
        self.decoder_bn1 = nn.BatchNorm1d(128)
        self.decoder_do1 = nn.Dropout(0.1)
        self.decoder_fc2 = nn.Linear(128, input_dim)
        

    # 前向传播
    def forward(self, input):
        # input: [batch_size, windows_size, input_dim]
        # encoder part
        x, _  = self.encoder_lstm1(input)
        x = x[:, -1, :] # 取最后一个时间步的输出作为编码结果 [128, 128]
        x = self.encoder_fc1(x)
        x = self.encoder_bn1(x)
        x = torch.relu(x)
        encoded_result = self.encoder_do1(x)
        # encoded_result: [batch_size, encoding_dim]

        # decoder part
        y = self.decoder_fc1(encoded_result)
        y = self.decoder_bn1(y)
        y = torch.relu(y)
        y = self.decoder_do1(y)
        decode_result = self.decoder_fc2(y)
        # decode_result: [batch_size, input_dim]
        return decode_result

    # 编码函数 stage1
    def encode(self, input):
        # input: [batch_size, seq_len, windows_size, input_dim]
        batch_size, seq_len, windows_size, input_dim = input.shape
        input = input.view(batch_size*seq_len, windows_size, input_dim)
        x, _  = self.encoder_lstm1(input)   # [batch_size*seq_len, windows_size, 128]
        x = x[:, -1, :] # 取最后一个时间步的输出作为编码结果 [batch_size*seq_len, 128]
        x = self.encoder_fc1(x)     # [batch_size*seq_len, encoding_dim]
        x = self.encoder_bn1(x)     # [batch_size*seq_len, encoding_dim]
        x = torch.relu(x)
        x = self.encoder_do1(x)
        x = x.view(batch_size, seq_len, -1)
        # output: [batch_size, seq_len, encoding_dim]
        return x    
    
    # # 编码函数 stage2
    # def encode_stage2(self, input):
    #     # input: [batch_size, seq_len, input_dim]
    #     batch_size, seq_len, input_dim = input.shape
    #     input = input.view(batch_size*seq_len, 1, input_dim)
    #     x, _  = self.encoder_lstm1(input)
    #     x = x[:, -1, :] # 取最后一个时间步的输出作为编码结果 [batch_size, 128]
    #     x = self.encoder_fc1(x)     # [batch_size, encoding_dim]
    #     # x = self.encoder_bn1(x)     # 由于batch size是1 不可以使用batch normalization
    #     x = torch.relu(x)
    #     x = self.encoder_do1(x)
    #     x = x.view(batch_size, seq_len, -1)
    #     # output: [batch_size, seq_len, encoding_dim]
    #     return x

    # 解码函数
    def decode(self, encoded_result):
        # input: [batch_size, seq_len, encoding_dim]
        batch_size, seq_len, encoding_dim = encoded_result.shape
        encoded_result = encoded_result.view(batch_size*seq_len, encoding_dim)

        y = self.decoder_fc1(encoded_result)
        y = self.decoder_bn1(y)
        y = torch.relu(y)
        y = self.decoder_do1(y)
        y = self.decoder_fc2(y)

        y = y.view(batch_size, seq_len, -1)
        # output: [batch_size, seq_len, input_dim]
        return y
    
    def encode_test(self, input):
        x, _  = self.encoder_lstm1(input)
        x = x[:, -1, :] # 取最后一个时间步的输出作为编码结果 [128, 128]
        x = self.encoder_fc1(x)     # [128, encoding_dim]
        x = self.encoder_bn1(x)
        x = torch.relu(x)
        x = self.encoder_do1(x)
        return x
    
    def decode_test(self, encoded_result):

        y = self.decoder_fc1(encoded_result)
        y = self.decoder_bn1(y)
        y = torch.relu(y)
        y = self.decoder_do1(y)
        y = self.decoder_fc2(y)

        return y