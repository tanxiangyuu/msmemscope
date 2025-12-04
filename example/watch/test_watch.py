import torch
import torch_npu
import torch.nn as nn
import torch.optim as optim
import mstx
import msmemscope


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 10)
        
    def forward(self, x):
        out = self.linear(x)
        return out    


def test():
    msmemscope.config(data_format='csv', watch='start,end,full-content') # 可以在这里设置监测的算子
    device = torch.device('npu:0')
    torch.npu.set_device(device)

    model = SimpleModel().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    inputs = torch.randn(32, 10).to(device)
    targets = torch.randn(32, 10).to(device)
    test_tensor = torch.randn(100, 100, dtype=torch.float32).to(device)

    msmemscope.start()
    for epoch in range(5):
        outputs = model(inputs)
        msmemscope.watcher.watch(test_tensor, name="leaksStWatch") # 添加监测的tensor
        loss = criterion(outputs, targets)
        msmemscope.watcher.remove(test_tensor) # 去除监测的tensor
        loss.requires_grad_(True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    msmemscope.stop()


def main():
    test()

if __name__ == "__main__":
    main()