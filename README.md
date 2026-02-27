<h1 align="center">MindStudio MemScope</h1>
<div align="center">
  <p>🚀 <b>昇腾 AI 内存调试调优利器</b></p>
  
  [![License](https://badgen.net/badge/License/MulanPSL-2.0/blue)](./License) [![Ascend](https://img.shields.io/badge/Hardware-Ascend-orange.svg)](https://www.hiascend.com/) 
  <br><br>
  [🌐项目主页](https://gitcode.com/Ascend/msmemscope) |
  [📖工具文档](https://msmemscope.readthedocs.io/zh-cn/latest/) |
  [🔥最新进展](#最新消息) |
  [🤔报告问题](https://gitcode.com/Ascend/msmemscope/issues/new)
  <br><br>
</div>


## 最新消息

- **[2026.2.02]** 🎉 **MindStudio Memscope 26.0.0-alpha.1版本上线！** 支持**Python API采集**方式使用、支持PyTorch框架下**采集内存快照**、支持**识别显存页表属性**并进行落盘、支持获取Driver新增的显存分配接口。

- [2025.12.30]：msMemScope项目首次上线。

## 简介

msMemScope（内存工具）是基于昇腾硬件的内存检测工具，用于模型训练与推理过程中的内存问题定位，提供内存泄漏检测、内存对比、内存块监测、内存拆解和低效内存识别等功能，帮助用户完成问题定位与处理。

## 目录结构

关键目录如下。

```shell
|-- build
   |-- make_run.sh       # 构建软件包脚本
   |-- build.py          # 构建脚本
|-- docs                 # 项目文档介绍 
|-- example              # 项目示例代码
|-- csrc                 # c++源码
   |-- framework         # 命令行解析，完成和event_trace模块的交互，获取内存事件并发送给analysis模块进行处理
   |-- event_trace       # 完成对内存事件的记录并提交框架模块
   |-- analysis          # 内存事件处理和分析
   |-- main.cpp
|-- output
   |-- bin
      |-- msmemscope     # 可执行文件
|-- test                 # UT测试、ST测试
```

## 版本说明
| 发布版本 | 发布时间        | 发布Tag       | 兼容性说明    |
  | ------- | --------------- | ------------- | ------------- |
  | 26.0.0-alpha.1 | 2026/02/04  | tag_MindStudio_26.0.0-alpha.1 | 兼容昇腾CANN 8.5.0及以前版本。请参考[CANN安装指南](https://www.hiascend.com/cann)获取CANN安装包。 |
### **v1.0（2025-12-10）**

**新增特性**

- 支持Python接口方式使用msMemScope。

## 兼容性信息

msMemScope工具当前支持CANN、Ascend Extension for PyTorch、MindSpore以及Aten算子的内存采集，具体版本支持情况如下表所示。

|产品|说明|
|--------|--------|
|CANN|CANN 8.2.RC1及之后版本的ATB算子（Ascend Transformers Boost）。|
|Ascend Extension for PyTorch|Ascend Extension for PyTorch 7.0.0及之后版本。|
|MindSpore|MindSpore 2.7.0及之后版本。|
|Aten算子|当采集Aten算子下发与访问事件时，需使用PyTorch 2.3.1或更高版本。|

## 环境部署

msMemScope工具支持通过软件包和源码两种方式进行安装，可根据实际需求选择合适的安装方式，具体安装步骤请参见[msMemScope安装指南](./docs/zh/install_guide.md)。

## 快速入门

快速入门旨在帮助用户快速熟悉msMemScope工具的使用方式。具体操作请参见[msMemScope快速入门](./docs/zh/quick_start.md)。

## 功能介绍

msMemScope工具提供内存采集、内存分析两大功能。

<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%;">
  <thead>
    <tr>
      <th>功能</th>
      <th>功能说明</th>
      <th>详细功能</th>
      <th>使用场景及说明</th>
    </tr>
  </thead>
  <tbody>
    <!-- 内存采集 -->
    <tr>
      <td rowspan="2">
        <a href="./docs/zh/memory_profile.md">内存采集</a>
      </td>
      <td rowspan="2">
        msMemScope工具提供内存事件的采集能力，允许自定义设置采集内存范围和采集项，为后续分析提供原始数据。
      </td>
      <td>Python接口方式采集</td>
      <td>支持通过Python接口采集信息，提供自定义设置采集内存范围和采集项，采集内存事件和Python Trace事件能力，实现精准采集、高效分析。</td>
    </tr>
    <tr>
      <td>命令行方式采集</td>
      <td>支持通过命令行方式采集信息，提供在非Python场景下采集内存事件与内存分析能力。</td>
    </tr>
    <!-- 内存分析 -->
    <tr>
      <td rowspan="5">
        <a href="./docs/zh/memory_analysis.md">内存分析</a>
      </td>
      <td rowspan="5">
        msMemScope工具基于采集的内存数据，提供泄漏、对比、监测、拆解、低效识别五项分析能力，帮助开发者快速诊断和优化内存问题。
      </td>
      <td>内存泄漏分析</td>
      <td>针对内存长时间未释放和内存泄漏等问题，需要进行内存分析时，msMemScope工具提供内存泄漏分析和kernelLaunch粒度的内存变化分析功能，进行告警定位与分析。</td>
    </tr>
    <tr>
      <td>内存对比分析</td>
      <td>当两个Step内存使用存在差异时，可能会导致内存使用过多，甚至出现OOM（Out of Memory，内存溢出）的问题，则需要使用msMemScope工具的内存对比分析功能来定位并分析问题。</td>
    </tr>
    <tr>
      <td>内存块监测</td>
      <td>在大模型场景中，当遇到内存踩踏定位困难时，msMemScope工具支持通过Python接口和命令行两种方式，在算子执行前后对指定的内存块进行监测。根据内存块数据的变化，快速确定算子间内存踩踏的范围或具体位置。</td>
    </tr>
    <tr>
      <td>内存拆解</td>
      <td>msMemScope工具提供内存拆解功能，支持对CANN层和Ascend Extension for PyTorch框架的内存使用情况进行拆解，输出模型权重、激活值、梯度，以及优化器等组件的内存占用情况。</td>
    </tr>
    <tr>
      <td>低效内存识别</td>
      <td>在训练推理模型过程中，可能会存在部分内存块申请后未立即使用，或使用完毕后未及时释放的低效情况。msMemScope工具可帮助识别这种低效内存的使用现象，从而优化训练推理模型。</td>
    </tr>
  </tbody>
</table>

## API参考

msMemScope工具提供API接口，便于快速分析内存情况，具体使用方法请参见[API参考](./docs/zh/api.md)。

## 安全声明

msMemScope工具安全相关信息，请参见[msMemScope安全声明](./docs/zh/security_statement.md)。

## 免责声明

### 致msMemScope使用者

1. 对于msMemScope测试用例以及示例文件中所涉及的数据，平台仅用于功能测试，华为不提供任何数据。

2. 如您在使用msMemScope工具过程中，发现任何问题（包括但不限于功能问题、合规问题），请在GitCode提交Issues，我们将及时审视并解决。

3. msMemScope工具依赖的第三方开源软件，均由第三方社区提供和维护，因第三方开源软件导致的问题需依赖相关社区的贡献和反馈进行修复。您应理解，msMemScope仓库不保证对第三方开源软件本身的问题进行修复，也不保证会测试、纠正所有第三方开源软件的漏洞和错误。

### 致数据所有者

如果您不希望您的数据集在msMemScope中被提及，或希望更新msMemScope中有关的描述，请在GitCode提交Issues，我们将根据您的Issues要求删除或更新您相关描述。衷心感谢您对msMemScope的理解和贡献。

## License

msMemScope工具的使用许可证，具体请参见[LICENSE](./License)。

msMemScope工具docs目录下的文档适用CC-BY 4.0许可证，具体请参见[LICENSE](./docs/LICENSE)。

## 贡献声明

1. **提交错误报告**：如果您在msMemScope中发现了一个不存在安全问题的漏洞，请在msMemScope仓库中的Issues中搜索，以防该漏洞被重复提交，如果找不到漏洞可以创建一个新的Issues。如果发现了一个安全问题请不要将其公开，请参阅安全问题处理方式。提交错误报告时应该包含完整信息。
2. **安全问题处理**：本项目中对安全问题处理的形式，请通过邮箱通知项目核心人员确认并编辑。
3. **解决现有问题**：通过查看仓库的Issues列表可以发现需要处理的问题信息，可以尝试解决其中的某个问题。
4. **如何提出新功能**：请使用Issues的Feature标签进行标记，我们会定期处理和确认开发。
5. **开始贡献**：
   1. Fork本项目的仓库。
   2. Clone到本地。
   3. 创建开发分支。
   4. 本地测试：提交前请通过所有单元测试，包括新增的测试用例。
   5. 提交代码。
   6. 新建Pull Request。
   7. 代码检视：您需要根据评审意见修改代码，并再次推送更新。此流程可能涉及多轮迭代。
   8. 当您的PR获得足够数量的检视者批准后，Committer会进行最终审核。
   9. 审核和测试通过后，CI会将您的PR合并入项目的主干分支。

   详细步骤请参见[开发者指南](./docs/zh/development_guide/development_guide.md)。
6. 贡献规范请参见[贡献指南](./docs/zh/development_guide/contributing_guide.md)。
7. 提交Issue相关操作请参见[Issue指南](./docs/zh/communication_guide/how_to_issue.md)。

## 建议与交流

华为MindStuido全流程开发工具链团队致力于提供端到端的昇腾AI应用开发解决方案，使能开发者高效完成训练开发、推理开发和算子开发。您可以通过以下渠道更深入了解华为MindStudio团队：
<div style="display: flex; align-items: center; gap: 10px;">
    <span>昇腾论坛：</span>
    <a href="https://www.hiascend.com/forum/" rel="nofollow">
        <img src="https://camo.githubusercontent.com/dd0b7ef70793ab93ce46688c049386e0755a18faab780e519df5d7f61153655e/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f576562736974652d2532333165333766663f7374796c653d666f722d7468652d6261646765266c6f676f3d6279746564616e6365266c6f676f436f6c6f723d7768697465" data-canonical-src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&amp;logo=bytedance&amp;logoColor=white" style="max-width: 100%;">
    </a>
    <span style="margin-left: 20px;">昇腾小助手：</span>
    <a href="https://gitcode.com/Ascend/msmemscope/blob/master/docs/zh/communication_guide/figures/ascend_assistant.jpg">
        <img src="https://camo.githubusercontent.com/22bbaa8aaa1bd0d664b5374d133c565213636ae50831af284ef901724e420f8f/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5765436861742d3037433136303f7374796c653d666f722d7468652d6261646765266c6f676f3d776563686174266c6f676f436f6c6f723d7768697465" data-canonical-src="./docs/zh/communication_guide/figures/ascend_assistant.jpg" style="max-width: 100%;">
    </a>
</div>


欢迎大家为社区做贡献。如果有任何疑问或建议，请参考[交流指南](./docs/zh/communication_guide/communication.md)获取详细的联系方式和支持渠道。
感谢您的支持。

## 致谢

感谢来自社区的每一个PR，欢迎贡献msMemScope。
