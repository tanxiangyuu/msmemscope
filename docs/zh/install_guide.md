# **msMemScope安装指南**

## 安装说明

本文档主要介绍msMemScope工具的安装方式，包括：

- **使用CANN包安装**：msMemScope工具完整功能已集成在CANN包中，请参考[CANN快速安装](https://www.hiascend.com/cann/download)安装昇腾NPU驱动和CANN软件（包括Toolkit和ops包）并配置环境变量。

- **[使用run包安装](#使用run包安装)**：msMemScope工具完整功能集成在CANN包中且依赖CANN软件，因此使用msMemScope工具需要**先完成CANN包安装**，若需要升级安装本工具代码仓中的最新功能，可以通过编译获取run包方式获取最新的msMemScope工具run包覆盖安装，具体操作见本安装方式中的[编译获取run包](#编译获取run包)获取最新版本run包。

## 使用run包安装

msMemScope工具支持在Linux系统上使用，目前提供以下两种方式获取run包。

1. releases页面下载run包。
2. 编译获取run包。

> [!NOTE]   
> run包需要在已安装CANN的环境中覆盖安装才能使用。

### releases页面下载run包

1. 请参考[msMemScope Release](https://gitcode.com/Ascend/msmemscope/releases)下载msMemScope的run包和对应的哈希校验文件（.sha256）。

   下载本软件即表示您同意[华为企业业务最终用户许可协议（EULA）](https://e.huawei.com/cn/about/eula)的条款和条件。

2. 验证run包的完整性。

    1. 在run包所在目录执行如下命令获取run包的sha256校验码。

        ```bash
        sha256sum mindstudio-memscope_<version>_linux-<arch>.run
        ```

        打印如下示例信息。

        ```ColdFusion
        {sha256} {name}.run
        ```

    2. 用记事本打开哈希校验文件查看sha256校验码。

    3. 比对两个文件的sha256校验码是否一致。

       若两个校验码一致，则表示下载了正确的软件包；若不一致，请不要使用该软件包，需要支持与服务请在论坛求助或提交技术工单。

### 编译获取run包

#### 安装依赖

安装前需确保Git、Python等环境可用，请满足[版本依赖](./development_guide/development_guide.md#1-开发环境配置)限制，若不满足可执行以下命令安装。

Debian系列：

```bash
sudo apt-get install -y python3 git build-essential cmake
```

openEuler系列：

```bash
sudo yum install -y python3 git gcc gcc-c++ make cmake
```

#### 编译构建run包

1. 在终端执行以下git命令，克隆(clone)msMemScope源码。

   ```bash
   git clone https://gitcode.com/Ascend/msmemscope.git <remote-name>
   ```

   注：其中`remote-name`为远程仓库别名，需要指定。

2. 执行以下命令下载Python三方依赖。注：`sqlite3`为离线功能使用依赖，可选安装。

   ```bash
   pip3 install -r ./requirements.txt
   ```

3. 下载构建依赖以及编译。

   ```bash
   cd ./<remote-name>/build
   python3 build.py local test
   ```

   其中参数说明如下：

   - `local`：代表是否本地构建，添加会下载gtest、json等依赖库用于本地构建，一般只有第一次需要，除非依赖库有更新。
   - `test`：代表是否要构建测试用例。

4. 在`./build`目录下执行以下命令，编译软件包。

   ```bash
   bash make_run.sh
   ```

   将工具的产物打包成一个run包，回显信息如下，表示打包成功，该包支持安装和升级的能力。

   ```bash
   [INFO] Run file created successfully: xx/mindstudio-memscope_<version>_linux-<arch>.run
   Usage instructions:
     Install: bash mindstudio-memscope_<version>_linux-<arch>.run --install --install-path=/path
     Upgrade: bash mindstudio-memscope_<version>_linux-<arch>.run --upgrade --install-path=/path
     Version: bash mindstudio-memscope_<version>_linux-<arch>.run --version
     Help:    bash mindstudio-memscope_<version>_linux-<arch>.run --help
   ```

   注：其中`arch`表示CPU架构。
   编译完成后，会在`./build`目录下生成软件包。

### 安装run包

1. 增加对run包的可执行权限。

    ```shell
    chmod +x mindstudio-memscope_<version>_linux-<arch>.run
    ```

2. 执行以下命令，安装软件包。

   ```bash
   bash mindstudio-memscope_<version>_linux-<arch>.run --install --install-path=<path>
   ```

   注：其中`path`为安装目录。

   将msMemScope安装在`path`目录下，安装成功后，打印以下信息。

   ```bash
   source <path>/msmemscope/set_env.sh
   [INFO] Installation completed successfully
   ```

#### 安装后检查

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

## 卸载

**脚本卸载**

1. 进入安装msmemscope的路径。

   ```bash
   cd <path>/msmemscope
   ```

   注：其中`path`为软件包的安装路径，请根据实际情况替换。

2. 执行以下命令运行卸载脚本，完成卸载。

   ```bash
   ./uninstall.sh
   ```

   卸载程序会提示用户是否确定卸载，若确定则输入y，不卸载输入n。

   卸载完成后，若打印如下信息，则说明软件卸载成功。

   ```tex
   [INFO] Uninstallation completed successfully
   ```

## 升级

msMemScope的软件包提供升级功能。

1. 点击[获取链接](https://www.openlibing.com/apps/obsDetails?bucketName=ascend-package)，选择更新版本的软件包下载。

2. 执行以下脚本升级软件。

   ```bash
   bash mindstudio-memscope_<version>_linux-<arch>.run --upgrade --install-path=<path>
   ```

   其中参数说明如下。

   - `--upgrade`指定升级操作。
   - `--install-path`指定目标目录，只升级选定的目录。

   升级完成后，若打印如下信息，则说明软件升级成功。

   ```bash
   [INFO] Upgrade completed successfully
   ```

## 附录A：参考信息

### 参数说明

本章节介绍了run格式（.run）软件包相关参数说明，run格式软件包支持通过命令行参数进行一键安装，各个命令之间可以配合使用，用户根据安装需要选择对应参数。

安装命令格式：`./mindstudio-memscope_<version>_linux-<arch>.run [options]`

详细参数请参见[表1](#cli-args-table)。

  > [!NOTE]   
  > 如果通过./mindstudio-memscope_\<version>_linux-{arch}.run --help命令查询出的参数没有在如下表格中解释，则说明该参数为预留参数或适用于其他产品类型，用户无需关注。

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
