# memscope

基于AI全栈的事件监控能力，实现CANN的专用内存分析工具

该工具使用：```msmemscope <option(s)> prog-and-args```
 
针对该软件仓，整体目录设计思路如下：
```
memscope
|-- build
   |-- build.py
|-- csrc
   |-- framework // 命令行解析，完成和event_trace模块的交互，获取内存事件并发送给analysis模块进行处理
   |-- event_trace // 完成对内存事件的记录并提交框架模块
   |-- analysis // 内存事件处理和分析
   |-- main.cpp
|-- output
   |-- bin
      |-- msmemscope // 可执行文件
|-- test
```
 
说明：
1. 命名风格统一如下：
  + 文件夹与文件统一小写下划线风格。
  + 类和作用域统一大驼峰风格。
  + 类的成员变量统一风格，如: aasBbb_。
2. 对于头文件和实现文件放在一个目录中。