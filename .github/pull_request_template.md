<!--
PR 标题格式：<type>(<scope>): <摘要>
仓库使用 squash merge —— 这个标题就是最终进入 main 的提交信息。
规则见 AGENTS.md §2.2 / §3。
-->

## 1. 为什么

<!-- 关联的决策编号（D-xx）或待决事项（Q-xx）；引入新决策请先在 docs/design.md §10 增补 -->

Refs:

## 2. 改了什么

<!-- 按模块列出。对外契约（contracts.py / ports.py）的变化必须显式标注 -->

| 模块 | 改动 |
| --- | --- |
|  |  |

**对外契约变化**：无 / 有（说明：）

## 3. 怎么验证的

<!-- 贴真实命令与真实输出，不要写"应该没问题"。涉及真实模型/存储时给关键数字。 -->

```
$ python -m ruff check src tests
$ python -m pytest tests -q
$ python scripts/check_secrets.py
$ python -m ec_renew.rag_llama.calibrate
```

关键数字（点数 / 图计数 / 相似度 / 共识分）：

## 4. 已知限制 / 未验证项

<!-- 有意留白的部分写在这里。没跑过真机的路径必须标"未验证"，见 AGENTS.md §3.2 -->

-

---

## 检查单

- [ ] `ruff` / `pytest` / `check_secrets` 三条闸门本地全绿（上面贴了输出）
- [ ] 未提交 `.env`、`logs/`、`.cache/` 等运行产物
- [ ] 新增配置项**同时**更新了 `config.py` 与 `.env.example`
- [ ] 涉及设计取舍时，`docs/design.md` 决策记录已增补
- [ ] 新增/修改的行为有对应测试；修 bug 的 PR 含**先失败后通过**的用例
- [ ] 对外契约变化已在上方显式标注
- [ ] 降级路径是显式的（带原因 + 事件日志），没有静默回退
- [ ] 按 §3.5 的「变更类型 → 必须同步更新的文件」表核对过

## 评审

- [ ] 至少一次他人 approve（或按 AGENTS.md §3.4 留下自查记录）
- [ ] 所有评论已解决
- [ ] 合并方式为 **squash merge**，合并后删除源分支
