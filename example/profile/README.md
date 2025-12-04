# 内存采集案例

## 概述

本案例介绍了内存工具msmemscope的内存采集功能，为后续其他功能的基础。通过内存采集可以获取用户进程运行时的各种显存数据，帮助用户快速处理，高效定位问题。

## 前期准备

请参见[安装指南](docs/zh/install_guide.md)安装msmemscope工具。

此外在运行本样例中的python场景时，需要配置torch以及torch_npu，具体请参见[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html)文档。

## 执行样例

### 参数说明

以下仅提供本功能样例中使用到的参数解释。其他参数的详细说明请参见[API参考](docs/zh/api.md)。

| 参数 | 可选/必选 | 说明 |
|------|-----------|------|
| data_format | 可选 | 指定落盘文件的格式，默认为csv格式，可更改为db格式。 |
| output | 可选 | 指定落盘文件的位置，默认为当前目录下。 |

### Python场景接口方式采集内存

在python场景下，推荐使用接口方式采集内存信息。

1.配置阶段：使用接口方式需要提前配置环境变量LD_PRELOAD和LD_LIBRARY_PATH，需要用户根据内存工具安装目录进行更改。
```bash
TOOL_PATH='msmemscope_path'
export LD_PRELOAD=${TOOL_PATH}/lib64/libleaks_ascend_hal_hook.so:${TOOL_PATH}/lib64/libascend_mstx_hook.so:${TOOL_PATH}/lib64/libascend_kernel_hook.so
export LD_LIBRARY_PATH=${TOOL_PATH}/lib64/:${LD_LIBRARY_PATH}
```
在用户脚本中可以添加memscope.config()以及memscope.start()和memscope.stop()来控制采集项和采集范围，这里仅提供最基础的样例，参考test_profile.py。其他可配置的采集参数请参见[API参考](docs/zh/api.md)。

2.运行阶段：
```bash
cd ./example/profile/python
bash test_profile.sh
```
### 非Python场景命令行方式采集内存
在非python场景下，可以使用命令行方式采集内存信息。

1.编译阶段：编译前，请确保source了安装的cann包，执行如下命令完成用例编译：
```bash
cd ./example/profile/c
make
```
2.运行阶段
```bash
msmemscope ./test_kernel.fatbin
```
## 结果说明
内存采集的数据将落盘于用户指定路径或者默认路径下。其结构形如：

```bash
memscopeDumpResults
├── msmemscope_xxxxxx_2025xxxxxxxxxx_ascend
    ├── device_0
        └── dump
            └── memscope_dump_2025xxxxxxxxxx.csv
    ├── msmemscope_logs
        └── msmemscope_2025xxxxxxxxxx.log
    └── config.json
├── ...
└── msmemscope_xxxxxx_2025xxxxxxxxxx_ascend
```
不同卡上的不同进程会各自落盘，内存数据将会统一存于dump目录下。如果需要可视化，需要设置落盘数据格式为db，并采集python trace然后放入MindStudio Insight中进行可视化分析。