import torch
import torch.nn as nn
from collections import deque

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)  # 输入维度为10，中间层维度为10
        self.fc2 = nn.Linear(10, 1)  # 输出维度为1

    def forward(self, x):
        a = self.fc(x)
        print("  a", a.grad_fn)
        b = self.fc2(a)
        print("  b", b.grad_fn)
        return b
model = SimpleModel()
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 使用SGD优化器

data_loader = [(torch.randn(1, 10), torch.randn(1, 1)) for _ in range(12)]

output_queue = deque(maxlen=2)
target_queue = deque(maxlen=2)
output_fn_queue = deque(maxlen=2)

# 训练循环
for epoch, (input_data, target) in enumerate(data_loader):  # 执行12次迭代\
    print("epoch", epoch)
    # optimizer.zero_grad()
    output = model(input_data)
    output_queue.append(output)
    target_queue.append(target)
    print("  output", output.grad_fn)
    output_fn_queue.append(output.grad_fn)
    
    if len(output_queue) == 2:  # 确保队列中有足够的元素来计算loss
        print("output_queue", output_queue)
        print("output_fn_queue", output_fn_queue)
        print("backward")
        optimizer.zero_grad()

        out = output_queue.popleft()
        print("  out", out.grad_fn)
        print("  out_2", output_queue[0].grad_fn)
        out.grad_fn = output_queue[0].grad_fn
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        