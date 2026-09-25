
## 环境
目前已经安装isaac的环境，conda安装的，直接使用 Conda 环境中的 `isaaclab`，之后禁止使用uv

```bash
conda activate isaaclab_61
cd ~/IsaacLab_6.1
```

工作空间位于~/IsaacLab_6.1/Pick_Anything，只能修改Pick_Anything下的东西，禁止修改Pick_Anything外面的文件。只可以读取但是不能修改。IsaacLab_6.1下和Pick_Anything并列的文件也禁止修改。

## Git 暂存和提交规则

只有在用户明确要求“暂存”或“提交”或“推送”时，才执行相应 Git 操作。所有改动必须按需求目的拆分为独立、原子的提交：

- 一个功能只涉及一个文件时，只暂存并提交该文件。
- 同一个功能涉及两个或更多文件时，一次暂存该功能所需的全部相关文件。
- 不同功能不得混入同一个提交。
- 每次提交前必须检查暂存区，确认没有无关文件。
- 使用 `git commit -m "清晰说明功能目的的备注"`，提交说明应准确表达本次功能改动。
- Git 身份只能配置在 `/home/user123/IsaacLab_6.1`和~/IsaacLab_6.1/Pick_Anything 仓库局部，不修改全局配置。Author 固定为 `Kraziee-AI`。

```

