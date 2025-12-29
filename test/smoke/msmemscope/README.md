# output/bin/msmemsocpe

编译msmemscope，软连接msmemscope编译后的产物output文件夹到这里，或者自行复制bin和lib64文件夹到这里
全量冒烟一般包含api接口场景测试，因此还需要将python组件一并拖入，并且将lib64里面的_msmemscope.so拖入到python目录中

../../msmemscope/output的文件结构如下：

├── bin
│   ├── msmemscope
│   └── msmemscope.bin
├── lib64
│   ├── libascend_kernel_hook.so
│   ├── libascend_leaks.so
│   ├── libascend_mstx_hook.so
│   ├── libatb_abi_0_hook.so
│   ├── libatb_abi_1_hook.so
│   ├── libleaks_ascend_hal_hook.so
│   └── _msmemscope.so
└── python
    └── msmemscope
        ├── analyzer
        │   ├── base.py
        │   ├── inefficient.py
        │   ├── __init__.py
        │   ├── leaks.py
        │   └── utility_function.py
        ├── aten_collection.py
        ├── describe.py
        ├── __init__.py
        ├── _msmemscope.so
        └── optimizer_step_hook.py