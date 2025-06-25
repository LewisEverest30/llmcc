import torch
import torch.nn as nn

class throughput_regression_head(torch.nn.Module):  
    def __init__(
            self, 
            plm_embed_size, 
            cnn_kernel_size=5,
            lstm_hidden_size=64,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            ):  
        super(throughput_regression_head, self).__init__()
        self.device = device

        self.fc_head1 = nn.Sequential(
            nn.Linear(plm_embed_size, 128),  # [batchsize, seq_len, 128]
            nn.LeakyReLU(), 
        ).to(device)
        self.fc_head2 = nn.Sequential(
            nn.Linear(128, 64),  # [batchsize, seq_len, 64]
            nn.LeakyReLU(),
        ).to(device)
        self.fc_head3 = nn.Sequential(
            nn.Linear(64, 1),  # [batchsize, seq_len, 1]
            nn.LeakyReLU(),
        ).to(device)


        # self.fc_head1 = nn.Linear(151936, 256).to(self.device)  # 更改输出头，从输出的bitrate-level变成一个值
        # self.fc_head2 = nn.Linear(256, 64).to(self.device)
        # self.fc_head3 = nn.Linear(64, 10).to(self.device)
        # self.fc_head4 = nn.Linear(30, 1).to(self.device)



    def forward(self, plm_outputs):  
        plm_outputs = plm_outputs.to(self.device)   # [batchsize, seq_len, plm_embed_size]

        logits = self.fc_head1(plm_outputs)
        logits = self.fc_head2(logits)
        logits = self.fc_head3(logits)

        return logits
