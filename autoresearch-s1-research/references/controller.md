# 控制器部署与恢复

## 能力范围

supervisor.py 是真实可运行的外部调度/验收驱动，使用 Python 标准库。production 限目标 Linux 服务器，每条固定10h最低、11h目标，按轨迹交替串行 slice 调度；不会自动购买服务器、调用任意模型API、创建隔离环境、删除容器或完成平台验收。实际 Harbor/Codex adapter 与权限隔离须第二阶段实现并验证。

## 文件与信任边界

config、controller state、adapter代码、原始证据和 STOP 文件由专家服务账号独占；两条候选目录与控制目录分离，研究用户不得读写控制状态。isolation_report（匹配task_id、status VERIFIED）和 model_probe_report 的哈希须真实；这些是外部验证结果，脚本校验内容与哈希但无法替代实际权限测试。

state 包含 events.jsonl（序号与SHA256链）、state.json、controller.lock、attempts、两条trajectory及时间导出。哈希链能发现常见意外篡改，不是签名，不防拥有编排权限的攻击者。不要将它作为独立真实性认证。

## 生命周期

`python3 scripts/supervisor.py CONFIG --state NEW_DIR` 创建唯一状态目录；使用前台方式先做受控smoke，然后由用户服务器已有服务管理器持续托管。总墙钟 max_wall_seconds 对本串行实现必须能容纳两条各11h及开销，建议由实测单轮时间和预算计算，不能只设置12h总预算。

合法 slice 返回后，外部校验身份、no_live_job、原始评分、方法快照和区间，记 accepted；模型提前返回不会直接结束研究。两条都达到11h目标且有真实完整轮次后标 READY_FOR_FINAL_REVIEW，10h最低线和最终质量仍需复核。若预算只允许低于目标但均超10h，当前脚本保守标 BUDGET_EXHAUSTED；由用户或授权专家审查实际证据并记录收尾决定，不能修改账本抹去未达目标事实。

## 停止与恢复

- 用户/外部管理者创建 STATE_DIR/STOP 可中止；研究 Agent 不具该权限。写 STOP 后核查 adapter 及实际平台作业，不假设终止本地进程自动停止服务器容器。
- `--resume` 要求同配置哈希、有效事件链、无未核实 attempt。不会计入中断间隙或重复 accepted 的时间。
- stale lock、未完成 attempt、异常作业、用户停止或预算结束都保守阻止自动重启。先核实已有进程与容器、恢复来源与用户授权，保留旧状态；需要新的恢复段时创建新状态目录并在阶段交接记录连续关系，由最终时间审查去重合并，不能伪装成第二条新独立轨迹。
- 模型不拥有容器管理、终止研究或改计时权限。用户始终拥有外部停止权，费用/墙钟/次数有上限；故障后不做无限重试。

## 验证与导出

测试 fixture 只支持 simulation，目标最多60秒，状态永远 SIMULATION_COMPLETE 而不是正式完成。`export_evidence.py` 重新读取 receipts 及原始哈希再导出；`trajectory_check.py` 对正式时长的结果仍是静态检查，不能证明模型真的研究了这些时段。
