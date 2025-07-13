import torch
import torch.nn as nn

from config import cfg


class NetStateEncoder(torch.nn.Module):  
    def __init__(
            self, 
            state_encoded_dim=128, 
            cnn_kernel_size=5,
            lstm_hidden_size=64,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            ):  
        super(NetStateEncoder, self).__init__()
        self.device = device

        # self.fnn_encoder_rtt = nn.Sequential(
        #     nn.Linear(1, state_encoded_dim),  # [batchsize, 3*state_feature_dim]
        #     nn.LeakyReLU(),
        #     nn.Linear(state_encoded_dim, state_encoded_dim)
        # ).to(device)

        # self.fnn_encoder_loss = nn.Sequential(
        #     nn.Linear(1, state_encoded_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(state_encoded_dim, state_encoded_dim)
        # )

        # self.fnn_encoder_tp = nn.Sequential(
        #     nn.Linear(1, state_encoded_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(state_encoded_dim, state_encoded_dim)
        # )

        # self.fnn_concator = nn.Sequential(
        #     nn.Linear(3*state_encoded_dim, state_encoded_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(state_encoded_dim, state_encoded_dim)
        # )

        # ================================direct fnn encoder================================
        self.fnn_encoder_direct = nn.Sequential(
            nn.Linear(3, state_encoded_dim),
            nn.LeakyReLU(),
            nn.Linear(state_encoded_dim, state_encoded_dim)
        )

    def forward(self, input_states):  
        input_states = input_states.to(self.device)      # [batchsize, seq_len, 3]

        # ===============================fnn encoder================================
        # rtt_feature = self.fnn_encoder_rtt(input_states[:, :, 0].unsqueeze(2))
        # loss_feature = self.fnn_encoder_loss(input_states[:, :, 1].unsqueeze(2))
        # throughput_feature = self.fnn_encoder_tp(input_states[:, :, 2].unsqueeze(2))
        # encoded_net_states = self.fnn_concator(torch.cat([rtt_feature, loss_feature, throughput_feature], dim=2))

        # ===============================direct fnn encoder================================
        encoded_net_states = self.fnn_encoder_direct(input_states)

        return encoded_net_states

