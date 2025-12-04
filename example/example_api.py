# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import torch
import torch_npu
import torch.nn as nn
import torch.optim as optim
import msmemscope


def test():
    device = torch.device('npu:0') # 可以更改为想要的卡号
    torch.npu.set_device(device)

    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear = nn.Linear(10, 10)
        
        def forward(self, x):
            out = self.linear(x)
            return out    
    model = SimpleModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(32, 10).to(device)
    targets = torch.randn(32, 10).to(device)

    for epoch in range(6):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.requires_grad_(True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 2 == 0:
            memory_allocated = torch.npu.memory_allocated(device) / (1024 ** 2)
            max_memory_allocated = torch.npu.max_memory_allocated(device) / (1024 ** 2)
            print(f"Epoch {epoch} : Current Memory Allocated = {memory_allocated:.2f} MB,", 
            f"Max Memory Allocated = {max_memory_allocated:.2f} MB")
        
        torch.npu.empty_cache()


def main():
    test()
    print("Test finished.")

if __name__ == '__main__':
    msmemscope.config(
        events="alloc,free,access,launch",      # 采集申请、释放、访问还有下发事件
        level="kernel,op",                      # 采集op或者kernel级别的算子下发事件
        call_stack="c,python",                  # 采集调用栈，数据量较大
        analysis="leaks,inefficient,decompose", # 开启泄漏识别、低效显存识别、显存拆解功能
        output="./output",                      # 指定输出文件落盘路径
        data_format="csv"                       # 指定输出文件格式
    )                                           # 设置参数
    msmemscope.start()                          # 开始采集
    main()
    msmemscope.stop()                           # 结束采集