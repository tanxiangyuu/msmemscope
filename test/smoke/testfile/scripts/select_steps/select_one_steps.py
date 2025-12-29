import torch
import torch_npu
import torch.nn as nn
import torch.optim as optim
import mstx

device = torch.device('npu:0')

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10,10)
    def forward(self, x):
        out = self.linear(x)
        return out    

for epoch in range(5):
    id = mstx.range_start("step start", None)
    model = SimpleModel().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    inputs = torch.randn(32,10).to(device)
    targets = torch.randn(32,10).to(device)

    outputs = model(inputs)
    loss = criterion(outputs,targets)
    loss.requires_grad_(True)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    memory_allocated = torch.npu.memory_allocated(device) / (1024 ** 2)
    max_memory_allocated = torch.npu.max_memory_allocated(device) / (1024 ** 2)
    print(f"Epoch {epoch} : Current Memory Allocated = {memory_allocated:.2f} MB,", 
    f"Max Memory Allocated = {max_memory_allocated:.2f} MB")
    mstx.range_end(id)
print("训练完成")

