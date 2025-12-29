# msmemscope_case

msmemscope冒烟工程


该工具使用：```python run_st.py [options]```(使用前需使能CANN环境变量)
 
针对该软件仓，整体目录设计思路如下：
```
|-- msmemscope // 存放msmemscope可执行文件  (msmemscope中的output目录以及python目录)
|-- scripts // 存放测试脚本
|-- src
   |-- test_suit // 测试套件与测试用例
   |-- utils // 工具模块

run_st.py // 执行冒烟脚本
```
