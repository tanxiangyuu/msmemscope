欢迎来到 MemScope 内存工具中文教程 ✨
=======================================

🌏 简介
-----------------------------
MemScope MemScope（内存工具）是基于昇腾硬件的内存检测工具，用于模型训练与推理过程中的内存问题定位，提供内存泄漏检测、内存对比、内存块监测、内存拆解和低效内存识别等功能，帮助用户完成问题定位与处理。

👉 推荐上手路径
-----------------------------
为了帮助你快速上手 MemScope 内存工具，我们推荐按照以下顺序进行学习：

* 对于想要使用 MemScope 内存工具的用户，建议先阅读 :doc:`安装指南 <install_guide>`，确保环境配置正确。
* 本教程提供的 :doc:`快速入门 <quick_start>` 将引导你完成基本的内存问题定位配置和运行。
* 详细功能部分将介绍 :doc:`内存采集 <memory_profile>` 、:doc:`内存分析 <memory_analysis>` 以及 :doc:`输出文件说明 <output_file_spec>` 等内容，帮助你更好地理解内存问题定位的场景。
* msMemScope工具提供API接口，便于快速分析内存情况，具体使用方法可以参考 :doc:`API使用 <api>`。
* 你可以参考 :doc:`开发者指南<development_guide/development_guide>` 部分，了解在 MemScope 内存工具的开发步骤。

.. toctree::
   :maxdepth: 2
   :caption: 🚀 开始你的第一步
   :hidden:

   install_guide
   quick_start

.. toctree::
   :maxdepth: 1
   :caption: 🧭 详细功能
   :hidden:

   memory_profile
   memory_analysis
   output_file_spec

.. toctree::
   :maxdepth: 2
   :caption: 🔬 API接口
   :hidden:

   api

.. toctree::
   :maxdepth: 2
   :caption: 💪 开发者指南
   :hidden:

   development_guide/development_guide
