---
name: autoresearch-s1-research
description: AutoResearch S1 第三阶段：用云服务器上的外部编排启动、监督或恢复 Codex+GPT-5.6 Sol Max 和 Codex+Seed 2.1 Turbo High 两条独立长程研究，各至少 10h、目标 11h 有效工作，强制权限、逐轮轨迹和时长证据。不用于普通短实验。
---

# S1 3：外部编排与长程研究（完整工具包）

## 固定规则

这是专家侧 S1 工具包，不得整体放进研究 Agent 镜像或上下文。只使用用户第二版资料及最新要求；第一版任务规范不适用。读论文后实际制作、实验、连接、研究与打包均在云 GPU 服务器。本地编写/校验 Skill 不是正式实验。

四阶段通过服务器专家侧 `expert_evidence/阶段交接.md` 衔接。复用已核实工作；调用某阶段不自动启动后续阶段，用户已授权整套流程时可继续，无须重复确认。未获资源信息或访问凭据只询问必要缺口，不自行租机或无限追加费用。

本文优先保留关键决策；进入相应步骤时按下面路由读取参考。模板是空输入，不是已通过证据。脚本输出只代表其标明的检查范围，不代替真实研究、公平性复核或平台验收。

## 执行路径

1. 先读 [运行约束](references/workflow.md)、[外部控制与恢复](references/controller.md) 和 [适配合同](references/adapter-contract.md)。核实第二阶段双镜像、实际两模型接入及隔离证据，禁止把测试 adapter 用于生产。
2. 配置 [controller-config.json](assets/controller-config.json)：两模型分别 **Codex+GPT-5.6 Sol Max** 与 **Codex+Seed 2.1 Turbo High**，独立 candidate_root、同任务版本与资源协议；指定已审 adapter 的 argv 和哈希、真实隔离与模型试跑报告、总墙钟和尝试预算。所有控制文件由专家账户持有，研究用户不可读写。
3. 在目标 Linux 服务器受保护目录运行 `python3 scripts/supervisor.py CONFIG --state NEW_STATE_DIR`。按真实需要由服务器服务管理器持久托管，不能把当前聊天存活当作监督。该调度器以两轨迹轮流执行有限 slice；各自上下文通过 adapter 恢复，不能串轨。需要并行时先适配、测试资源公平及状态锁，不假称此脚本并行。
4. 脚本根据可信 adapter 的原始证据核验八字段、分数、方法快照和有效区间。合法提前返回后继续调度；伪分数、超出观察窗口的时长、零进展、未回收作业或异常退出会停在待外部处理状态，不把故障计入时长。
5. 监督原始进展和资源预算。每条最低 10h、目标 11h，不用 7h例外，不把挂机或排队计入。模型无权停止研究或容器；用户可随时通过外部 STOP 指令中止。到目标后仅标 READY_FOR_FINAL_REVIEW，仍需专家语义复核。
6. 执行 `python3 scripts/export_evidence.py STATE_DIR` 重新校验原始 receipts 并导出时间证据；对两条分别运行 `python3 scripts/trajectory_check.py TRAJECTORY --times TIME_JSON --root STATE_DIR --out REPORT`。详细事件不塞进八字段轨迹，保留专家附件。
7. 保存代码与必须产物、真实时长及日志，更新交接；由外部发出收尾决定进入 `$autoresearch-s1-deliver`。研究模型不能自行清理容器或宣称完成。

## 权限与故障边界

研究模型只做“假设→合法修改→Public/Dev→分析→保留/回退→下一轮”。它不能读取本 Skill/参考解/另一条轨迹，不能修改控制账本、协议或评分器。No Docker socket、宿主机管理凭据和容器控制工具；这些必须在第二阶段实测，不是 Python 脚本自动建立的沙箱。

预算耗尽、不可恢复故障和用户中止由外部执行并如实记录。正常可继续会话使用 `--resume` 验证配置/事件链；异常残留作业和 stale lock 必须先人工核实，不能为恢复盲目杀容器、删账本或反复付费重启。模拟永不算正式时间。

## 工具验证

执行 `python3 -m unittest discover -s tests -v`。测试模型是假适配器、不会访问网络；脚本校验的是证据结构/哈希/区间，真实研究有效性仍须查看外部原始日志。完整步骤见 [流程原文](references/flow-detail.md)。
