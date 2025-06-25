import torch
from torch.utils.data import DataLoader, TensorDataset

from ae_model import Autoencoder
from dataset_ae import CcDatasetAeWithSeqOld, CcDatasetAeWithSeqPredict
import random

train_dataset_path = './data/gccgaobo.pkl'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 从checkpoint加载模型
model = Autoencoder(3, 896).to(device)
model.load_state_dict(torch.load('autoencoder/checkpoint/model_430.pth', weights_only=True))
model.eval()


train_dataset = CcDatasetAeWithSeqPredict(train_dataset_path)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

for i, (x, y) in enumerate(train_loader):
    with torch.no_grad():
        decoded_triplet = model(x.to(device))

    print("Decoded:", decoded_triplet)
    print("Label:", y)
    if i == 10:
        break






# # 从dataloader中随机取一对x和y
# random_index = random.randint(0, len(train_loader.dataset) - 1)
# x, y = train_loader.dataset[random_index]
# # print("x:", x[-1])
# print("y:", y)

# # 将x转为第一维度为batchsize的tensor
# x = x.unsqueeze(0)


# # 示例使用
# with torch.no_grad():
#     encoded_number = model.encode_test(x.to(device))
#     decoded_triplet = model.decode_test(encoded_number)

# # print("Encoded:", encoded_number)
# print("Decoded:", decoded_triplet)