---
name: autoresearch-s1-package
description: AutoResearch S1 第二阶段：在云 GPU 服务器构建 Agent/Verifier 双镜像和隔离，测试 Codex+GPT-5.6 Sol Max 与 Codex+Seed 2.1 Turbo High 的真实接入，完成小规模试跑并准备外部编排。不启动正式长程研究。
---

# S1 2：双镜像与模型连接（完整工具包）

## 固定规则

这是专家侧 S1 工具包，不得整体放进研究 Agent 镜像或上下文。只使用用户第二版资料及最新要求；第一版任务规范不适用。读论文后实际制作、实验、连接、研究与打包均在云 GPU 服务器。本地编写/校验 Skill 不是正式实验。

四阶段通过服务器专家侧 `expert_evidence/阶段交接.md` 衔接。复用已核实工作；调用某阶段不自动启动后续阶段，用户已授权整套流程时可继续，无须重复确认。未获资源信息或访问凭据只询问必要缺口，不自行租机或无限追加费用。

本文优先保留关键决策；进入相应步骤时按下面路由读取参考。模板是空输入，不是已通过证据。脚本输出只代表其标明的检查范围，不代替真实研究、公平性复核或平台验收。

## 执行路径

1. 读取阶段交接及 [封装流程](references/workflow.md)。先执行 `python3 scripts/preflight.py --root PROJECT --out HOST_REPORT` 获取有限服务器观察，再实际验证 GPU 容器、依赖及目标 Harbor 版本；主机探测不代表就绪。
2. Agent build context 固定 environment/，Verifier 固定 tests/。候选是 /workspace/solution；每轮 Public/Dev，结束后移交完整候选到独立 Verifier，运行 /tests/test.sh，可信进程原子写 /logs/verifier/reward.txt 或 reward.json。冻结面、私有标签及 Reference 对候选受保护。
3. 按 [目标平台适配合同](references/adapter-contract.md) 在真实服务器实现或复用可信 adapter；不能猜测模型 ID/档位参数。使用 [连接配置模板](assets/probe-config.json)，执行 `python3 scripts/probe_models.py CONFIG --out NEW_DIR`，分别核实 **GPT-5.6 Sol Max** 与 **Seed 2.1 Turbo High** 的真实请求、工具、公开评分及记录落盘。不得替换模型或降档。
4. 按 [隔离验收表](assets/isolation-checklist.md) 进行实际权限测试。准备第三阶段的受保护控制目录、adapter、计时与恢复，不只写 Prompt。记录 no_live_job 的含义，CLI 返回不表示远端容器已停止。
5. 在预先界定的短程预算内完成小规模端到端试跑，验证候选移交、独立 reward、异常和提前返回处理。按 [试跑模板](assets/pilot-report.md) 输出结论，更新交接，下一阶段 `$autoresearch-s1-research`。

正式两个模型各至少 10h、目标 11h有效研究；本阶段试跑不计入。不能把 source/example 的原生评分原样当作满足统一评分规则；按需只读 [PCA 教学附件](references/pca-teaching-example/README.md)，其中空轨迹和未运行证据不算完成。

## 测试与边界

`python3 -m unittest discover -s tests -v` 只使用会拒绝 production 的测试 adapter。工具包不包含凭据或已绑定某台服务器的 Harbor adapter；实际适配、两套构建、权限和两模型接入都通过后才可称启动就绪。完整细节见 [流程原文](references/flow-detail.md)。
