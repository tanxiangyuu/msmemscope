<div align="center">
  <br />
  <br />


  # **msMemScope**

  <!-- 用分隔线替代背景 -->
  ---

[![License](https://img.shields.io/badge/license-Apache--2.0-red?logo=apache)](https://www.apache.org/licenses/LICENSE-2.0)
    <br>
</div>

# 🔥 最新消息

- [2025.12.20]：msMemScope项目首次上线

# 🔭 简介

msMemScope工具是基于昇腾硬件的内存检测工具，用于模型训练与推理过程中的内存问题定位。提供内存泄漏检测、内存对比、内存块监测、内存拆解和低效内存识别等功能，帮助用户完成问题定位与处理。


# 📚目录结构

针对该软件仓，整体目录设计思路如下：

```shell
|-- build
   |-- make_run.sh       # 构建软件包脚本
   |-- build.py          # 构建脚本
|-- docs                 # 项目文档介绍 
|-- example              # 项目示例代码
|-- csrc                 # 源码
   |-- framework         # 命令行解析，完成和event_trace模块的交互，获取内存事件并发送给analysis模块进行处理
   |-- event_trace       # 完成对内存事件的记录并提交框架模块
   |-- analysis          # 内存事件处理和分析
   |-- main.cpp
|-- output
   |-- bin
      |-- msmemscope     # 可执行文件
|-- test                 # UT测试
```

# 🏷️ 版本说明

## **v1.0（2025-12-10）**

**✨ 新增特性**

- 支持Python接口方式使用msMemScope。

# ⚠️ 兼容性信息

当前msMemScope工具支持下列框架的内存检测：

- Ascend Extension for PyTorch 7.0.0及之后版本
- MindSpore 2.7.0及之后版本
- CANN 8.2.RC1及之后版本的ATB算子（Ascend Transformers Boost）

支持采集Aten算子下发与访问事件，PyTorch 2.3.1+。

# 🛠️ 安装说明

msMemScope工具的具体安装步骤请参见[安装指南](./docs/zh/install_guide.md)。

# 🚀 快速上手

快速上手旨在帮助用户快速熟悉msMemScope工具的使用方式。

具体快速上手请参见[快速上手](./docs/zh/get_started.md)。

# 📌 工具限制与注意事项

请在使用工具前，仔细阅读以下安全使用说明，以防范潜在风险。

- 权限约束
  - 出于安全性及权限最小化考虑，msMemScope工具不建议使用root等高权限用户安装使用，推荐使用普通用户权限。
  - 遵循最小权限原则（如禁止other用户可写，常见如禁止666、777）。
  - 请确保执行用户的umask值大于等于0027，否则会导致获取的性能数据所在目录和文件权限过高。
  - 请确保性能数据保存在当前用户目录下，且该目录不含软链接，以防止可能的安全问题。

- 安装使用约束  
  msMemScope为开发调测工具，不应在生产环境中使用。

- 文件校验约束  
  请对下载的文件（特别是模型权重等文件）使用SHA256等校验方法进行完整性校验，保证文件来源安全可信，从而有效避免潜在的安全风险。

- 兼容性约束  
  msMemScope工具在生成db格式文件时，需确保当前用户环境中已安装libsqlite3.so(SQLite库)等相关文件，并确保group和others用户组无修改权限。

# 📑 功能介绍

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

# 🔗 API参考

msMemScope工具的API请参考[API](./docs/zh/api.md)。

# 📍 免责声明

msMemScope工具依赖的第三方开源软件，均由第三方社区提供和维护，因第三方开源软件导致的问题的修复依赖相关社区的贡献和反馈。您应理解，msMemScope仓库不保证第三方开源软件本身的问题的修复，也不保证会测试或纠正所有第三方开源软件的漏洞和错误。

# ✒️ License

[License](LICENSE.md)

# 🗨️ 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交issues，我们会尽快回复。感谢您的支持。

# 🤝 致谢

msMemScope由华为公司的下列部门联合贡献：

- 计算产品线

感谢来自社区的每一个PR，欢迎贡献msMemScope。

# 📣 贡献声明

1. **提交错误报告**：如果您在msMemScope中发现了一个不存在安全问题的漏洞，请在msMemScope仓库中的 Issues 中搜索，以防该漏洞已被提交，如果找不到漏洞可以创建一个新的 Issues。如果发现了一个安全问题请不要将其公开，请参阅安全问题处理方式。提交错误报告时应该包含完整信息。
2. **安全问题处理**：本项目中对安全问题处理的形式，请通过邮箱通知项目核心人员确认编辑。 
3. **解决现有问题**：通过查看仓库的 Issues 列表可以发现需要处理的问题信息, 可以尝试解决其中的某个问题。
4. **如何提出新功能**：请使用 Issues 的 Feature 标签进行标记，我们会定期处理和确认开发。
5. **开始贡献：**
   1. Fork 本项目的仓库。
   2. Clone到本地。
   3. 创建开发分支。
   4. 本地自测，提交前请通过所有的已经单元测试，以及为您要解决的问题新增单元测试。
   5. 提交代码。
   6. 新建 Pull Request。
   7. 代码检视，您需要根据评审意见修改代码，并再次推送更新。此过程可能会有多轮。
   8. 当您的PR获得足够数量的检视者批准后，Committer会进行最终审核。
   9. 审核和测试通过后，CI会将您的PR合并入项目的主干分支。