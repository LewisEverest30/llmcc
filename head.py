import torch
import torch.nn as nn

class BitrateRegressionHead(torch.nn.Module):  
    def __init__(
            self, 
            plm_embed_size, 
            cnn_kernel_size=5,
            lstm_hidden_size=64,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            ):  
        super(BitrateRegressionHead, self).__init__()
        self.device = device

        self.fc_head1 = nn.Sequential(
            nn.Linear(plm_embed_size, 128),  # [batchsize, seq_len, 128]
            nn.LeakyReLU(), 
        ).to(device)
        self.fc_head2 = nn.Sequential(
            nn.Linear(128, 1),  # [batchsize, seq_len, 1]
            nn.LeakyReLU(),
        ).to(device)

    def forward(self, plm_outputs):  
        plm_outputs = plm_outputs.to(self.device)   # [batchsize, seq_len, plm_embed_size]
        logits = self.fc_head1(plm_outputs)
        logits = self.fc_head2(logits)

        return logits

class ActionClassificationHead(torch.nn.Module):  
    def __init__(
            self, 
            plm_embed_size, 
            cnn_kernel_size=5,
            lstm_hidden_size=64,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            ):  
        super(ActionClassificationHead, self).__init__()
        self.device = device

        self.fc_head1 = nn.Sequential(
            nn.Linear(plm_embed_size, 128),  # [batchsize, seq_len, 128]
            nn.LeakyReLU(), 
        ).to(device)
        self.fc_head2 = nn.Linear(128, 3)  # [batchsize, seq_len, 3]

    def forward(self, plm_outputs):  
        plm_outputs = plm_outputs.to(self.device)   # [batchsize, seq_len, plm_embed_size]

        logits = self.fc_head1(plm_outputs)
        logits = self.fc_head2(logits)
        return logits
