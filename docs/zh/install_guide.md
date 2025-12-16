# **msMemScope安装指南**

## 安装说明

msMemScope工具是基于昇腾硬件的内存检测工具，用于模型训练与推理过程中的内存问题定位。工具提供内存泄漏检测、内存对比、内存块监测、内存拆解和低效内存识别等功能，帮助用户高效定位和快速处理内存问题。

msMemScope工具支持在Linux系统上使用，目前提供以下两种安装使用方式。

1. 稳定版本：提供安装软件包。
2. 最新版本：从源码安装，msMemScope提供编译打包功能，以便您可以快速安装使用或开发工具。


## 安装前准备

### 准备软件包	

**软件包下载**

点击[获取链接](https://gitcode.com/Ascend/msmemscope/releases)，下载msMemScope工具软件包。

软件包名称：`MindStudio_memscope_linux-<arch>.run`，<arch>表示CPU架构。

下载本软件即表示您同意[华为企业业务最终用户许可协议（EULA）](https://e.huawei.com/cn/about/eula)的条款和条件。

### 准备工具

使用msMemScope工具前，需要安装驱动固件和CANN软件包，并配置环境变量。

-   准备Atlas训练系列产品/Atlas推理产品的服务器，并安装对应的驱动和固件，具体安装过程请参见安装NPU驱动固件《[CANN 软件安装指南(CANN商用版)](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0005.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)》或《[CANN软件安装指南(CANN社区版)](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_0005.html?Mode=PmIns&OS=openEuler&Software=cannToolKit)》中的“安装NPU驱动固件”章节。
-   安装CANN软件包，请参考安装CANN《[CANN软件安装指南(CANN商用版)](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)》《[CANN 软件安装指南(CANN社区版)](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_0008.html?Mode=PmIns&OS=openEuler&Software=cannToolKit)》中选择“训练&推理&开发调试”场景安装CANN软件包。请根据系统选择aarch64或x86_64对应版本的CANN-toolkit、CANN-ops。

### 数字签名

为了防止软件包在传递过程或存储期间被恶意篡改，下载软件包时需下载对应的数字签名文件用于完整性验证。

请点击[PGP数字签名工具包](https://support.huawei.com/enterprise/zh/tool/pgp-verify-TL1000000054)获取工具包，将工具包解压后，请参考文件夹中的《OpenPGP签名验证指南》，对下载的软件包进行PGP数字签名校验。如果校验失败，请不要使用该软件包，访问支持与服务在论坛求助或提交技术工单。

## 安装步骤

### 安装依赖

安装前需确保Git、Python环境可用，若不满足可执行以下命令安装。

Debian系列：

```bash
sudo apt-get install -y python3 git
```

openEuler系列：

```bash
sudo yum install -y python3 git
```

### 安装msMemScope

1. 在终端执行以下git命令，克隆(clone)msMemScope源码。

   ```bash
   git clone https://gitcode.com/Ascend/msmemscope.git <remote-name>
   ```

   注：<remote-name>为远程仓库别名，需要指定。

2. 下载构建依赖以及编译。

   ```bash
   cd ./msmemscope/build
   python3 build.py local test
   ```

   其中参数说明如下：

   - `local`：代表是否本地构建，添加会下载gtest、json等依赖库用于本地构建，一般只有第一次需要，除非依赖库有更新。
   - `test`：代表是否要构建测试用例。

3. 在`./build`目录下执行以下命令，编译软件包。

   ```bash
   bash make_run.sh
   ```

   将工具的产物打包成一个run包，回显信息如下，表示打包成功，该包支持安装和升级的能力。

   ```bash
   [INFO] Run file created successfully: xx/MindStudio_memscope_linux-<arch>.run
   Usage instructions:
     Install: bash MindStudio_memscope_linux-<arch>.run --install --install-path=/path
     Upgrade: bash MindStudio_memscope_linux-<arch>.run --upgrade --install-path=/path
     Version: bash MindStudio_memscope_linux-<arch>.run --version
     Help:    bash MindStudio_memscope_linux-<arch>.run --help
   ```

   注： <arch>表示CPU架构。

4. 在`./build`目录下执行以下命令，安装软件包。

   ```bash
   bash MindStudio_memscope_linux-<arch>.run --install --install-path=<path>
   ```

   注：<path>为安装目录。

   将msMemScope安装在<path>目录下，安装成功后，打印以下信息。

   ```bash
   source <path>/msmemscope/set_env.sh
   [INFO] Installation completed successfully
   ```

### 安装后检查

请检查并确认安装目录：`<path>/msmemscope`下已生成`set_env.sh`文件。

## 安装后配置

在使用msMemScope工具前，需执行以下命令，配置PYTHONPATH和PATH环境变量。

```bash
source <path>/msmemscope/set_env.sh
```

环境变量配置成功后，打印以下信息。

```tex
Setting up msmemscope environment...
bash: local: can only be used in a function
✓ Added to PYTHONPATH (forced to front):<path>/msmemscope/python
bash: local: can only be used in a function
✓ Added to PATH (forced to front): <path>/msmemscope/bin
msmemscope environment setup completed
```

## 升级

msMemScope的软件包提供升级功能。

1. 点击[获取链接](https://gitcode.com/Ascend/msmemscope/releases)，选择更新版本的软件包下载。

2. 执行以下脚本升级软件。

   ```bash
   bash MindStudio_memscope_linux-<arch>.run --upgrade --install-path=<path>
   ```

   其中参数说明如下。

   - `--upgrade ` 参数升级产物。
   - `--install-path`指定目标目录，只升级选定的目录。

   升级完成后，若打印如下信息，则说明软件升级成功。

   ```bash
   [INFO] Upgrade completed successfully
   ```

## 卸载

**脚本卸载**

1. 进入安装msmemscope的路径。

   ```bash
   cd <path>/msmemscope
   ```

   注：<path>为软件包的安装路径，请根据实际情况替换。

2. 执行以下命令运行卸载脚本，完成卸载。

   ```bash
   ./uninstall.sh
   ```

   卸载程序会提示用户是否确定卸载，若确定则输入y，不卸载输入n。

   卸载完成后，若打印如下信息，则说明软件卸载成功。

   ```
   [INFO] Uninstallation completed successfully
   ```

## 附录A：参考信息

### 参数说明

本章节介绍了run格式（.run）软件包相关参数说明，run格式软件包支持通过命令行参数进行一键安装，各个命令之间可以配合使用，用户根据安装需要选择对应参数，所有参数都是可选参数。

安装命令格式：`./MindStudio_memscope_linux-<arch>.run [options]`

详细参数请参见[表1](#cli-args-table)。

>如果通过./MindStudio_memscope_linux-<arch>.run --help命令查询出的参数未解释在如下表格，则说明该参数为预留参数或适用于其他产品类型，用户无需关注。

**表 1**  参数说明

<a id="cli-args-table"></a>

<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%;">
<thead>
  <tr>
    <th>参数</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>--help</td>
    <td>查询帮助信息。</td>
  </tr>
  <tr>
    <td>--version</td>
    <td>查询版本信息。</td>
  </tr>
  <tr>
    <td>--install</td>
    <td>安装软件包。后面可以指定安装路径--install-path=&lt;path&gt;，也可以不指定安装路径，直接安装到默认路径下。</td>
  </tr>
  <tr>
    <td>--upgrade</td>
    <td>升级已安装的软件，支持在低版本升级至高版本情况下使用。 如果需要从高版本回退至低版本，需卸载高版本后重新安装所需版本。</td>
  </tr>
  <tr>
    <td>--install-path</td>
    <td>指定安装路径，需配合安装--install、升级--upgrade参数使用。</td>
  </tr>
</tbody>
</table>

