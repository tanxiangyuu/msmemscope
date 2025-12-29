# contributing
## 构建与测试
在提交PR之前，建议您先在本地搭建开发环境，构建`msmemscope`并运行相关测试。
[搭建开发及测试环境](./development_guide.md)。

## PR标题与分类
只有特定类型的PR才会被审核。请在PR标题前添加合适的前缀，以明确PR类型。请使用以下分类之一。

- `[feature]`: 新增模块功能、基础功能等。
- `[bugfix]`: 修复项目bug。
- `[refactor]`: 对现有功能模块进行重构。
- `[docs]`: 有关说明文档的新增或修改。
- `[test]`: 对UT/ST测试用例的新增或修改。
- `[build]`: 构建系统变更（CMake、build.py 等）。

若您的PR当前只是草稿状态，不需要审核，请使用如下标签。
- `[WIP]`: 草稿，勿 review。

## Commits要求
1. 每一次commit的message需要明确对应代码的功能，无效信息不会通过，如“添加适配文件”、“First commit”等。
2. 对于多余无效的commits需要压缩，如连续的相同commit messages的commits、连续的codecheck修改等。

若存在多次commit，请按照两种方式其一，压缩为一条commit记录(尽管GitCode在合并PR时提供了`Squash 合并`的选项, 提前将PR整理为单个简洁的commit仍然被视为最佳实践，并且深受commiter们的欢迎)。

### 方式一：交互式变基（推荐）
1. 查看需要合并的最近几个commit（例如最近3个）。
``` shell
git log --online -n 3
```

2. 选中需要合并的几条commit记录的前一条commit ID，运行命令。
```shell
# commit_id 需要替换为实际值
git rebase -i commit_id
```

3. 将要压缩的commits ID对应的pick改为squash，需要至少保留一个pick。
4. 保存退出之后，跳入第二个编辑环境，需要调整commit messages。
5. 修改对应pick的commit的message。
6. 保存退出。
7. 强制推送更新后的分支 (仅限于您自己的特性分支)。
```shell
# branch_name 替换为实际分支名
git push -f origin branch_name
```

8. 最后对应的PR改变为我们需要的状态。

### 方式二：reset + 新建commit
```bash
# 获取最新的待合入的目标分支（如main）
git fetch origin main

# Soft-reset到主干分支--此操作会保存所有修改并回归到暂存区。
git reset --soft origin/main

# 将所有更改提交为一个新的commit
git commit -m "feat: concise description of your change"

# 强制推送以更新PR分支
git push --force-with-lease origin your-branch-name

```