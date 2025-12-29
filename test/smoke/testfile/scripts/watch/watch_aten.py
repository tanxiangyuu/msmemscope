import torch
import torch_npu
import torch.nn as nn
import torch.optim as optim
import mstx


import os
import sys
import msmemscope

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10,10)
    def forward(self, x):
        out = self.linear(x)
        return out    

def test():
    device = torch.device('npu:0')
    torch.npu.set_device(device)

    model = SimpleModel().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)


    inputs = torch.randn(32,10).to(device)
    targets = torch.randn(32,10).to(device)

    test_tensor = torch.randn(100, 100, dtype=torch.float32).to(device)

    for epoch in range(5):

        # step 开始
        id = mstx.range_start("step start")

        outputs = model(inputs)
        msmemscope.watcher.watch(test_tensor, name="memscopeStWatch")
        loss = criterion(outputs,targets)
        msmemscope.watcher.remove(test_tensor)
        loss.requires_grad_(True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # step 结束
        mstx.range_end(id)

def main():
    test()

if __name__ == "__main__":
    main()