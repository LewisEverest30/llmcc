import torch
import torch.nn as nn

class RewardLoss(nn.Module):
    '''
    奖励型loss
    '''
    def __init__(self, alpha=0.5, beta=0.5):
        super(RewardLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, reward_predict, network_states_real, available_resources: float, switch_threshold=0.5):

        # print("  reward_predict", reward_predict)
        # print("  network_states_real", network_states_real)
        # print("  available_resources", available_resources)

        reward_real = self.cal_reward(network_states_real, available_resources)
        # print("reward_real", reward_real)  
        two_reward_mse =  self.cal_mse(reward_predict, reward_real)
        # print("two_reward_mse", two_reward_mse)
        if two_reward_mse > switch_threshold:
            print("  noswitch loss: ", two_reward_mse)
            return two_reward_mse
        else:
            # predict_0_mse = self.cal_mse(reward_predict, torch.zeros_like(reward_predict))
            # print("  switch loss: ", predict_0_mse)
            return reward_predict

    # def forward(self, network_states_predict_last, network_states_real_new, available_resources_last, available_resources_new, switch_threshold=0.5):

    #     reward_predict = self.cal_reward(network_states_predict_last, available_resources_last)
    #     reward_real = self.cal_reward(network_states_real_new, available_resources_new)
    #     two_reward_mse =  self.cal_mse(reward_predict, reward_real)
        
    #     # print("reward_predict", reward_predict)
    #     # print("reward_real", reward_real)  
    #     # print("two_reward_mse", two_reward_mse)
    #     if two_reward_mse > switch_threshold:
    #         print("  noswitch loss: ", two_reward_mse)
    #         return two_reward_mse
    #     else:
    #         predict_0_mse = self.cal_mse(network_states_predict_last, torch.zeros_like(network_states_predict_last))
    #         print("  switch loss: ", predict_0_mse)
    #         return predict_0_mse

    def cal_reward(self, network_states, available_resources, alpha=8e-1, beta=1e0, gamma=1e0):    # 含义类似于强化学习中的奖励函数，但这里的reward越小越好
        rtt = network_states[:, 0]
        loss = network_states[:, 1]
        tp = network_states[:, 2]
        
        # print("tp", tp)
        # print("rtt", rtt)
        # print("loss", loss)
        # print("available_resources", available_resources)

        # 计算损失
        loss = alpha * (tp - available_resources) ** 2 + beta * rtt
        
        # print(alpha * (tp - available_resources) ** 2)
        # print(beta * rtt)
        # print(gamma * loss)
        # 对于批量数据，我们需要对所有样本的损失求平均或总和
        return torch.mean(loss)
    
    def cal_mse(self, value1: torch.Tensor, value2: torch.Tensor):
        return torch.mean((value1 - value2) ** 2)