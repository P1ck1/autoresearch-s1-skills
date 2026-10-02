---
name: autoresearch-s1-deliver
description: AutoResearch S1 第四阶段：在双模型研究结束后复验各候选、选唯一最终方法、整理真实证据，按用户第二版 QA 做完整自检并打包六项交付。用于最终复验、质检修复和交付，不生成虚构轨迹或平台验收结论。
---

# S1 4：最终复验与六项交付（完整工具包）

## 固定规则

这是专家侧 S1 工具包，不得整体放进研究 Agent 镜像或上下文。只使用用户第二版资料及最新要求；第一版任务规范不适用。读论文后实际制作、实验、连接、研究与打包均在云 GPU 服务器。本地编写/校验 Skill 不是正式实验。

四阶段通过服务器专家侧 `expert_evidence/阶段交接.md` 衔接。复用已核实工作；调用某阶段不自动启动后续阶段，用户已授权整套流程时可继续，无须重复确认。未获资源信息或访问凭据只询问必要缺口，不自行租机或无限追加费用。

本文优先保留关键决策；进入相应步骤时按下面路由读取参考。模板是空输入，不是已通过证据。脚本输出只代表其标明的检查范围，不代替真实研究、公平性复核或平台验收。

## 执行路径

1. 核验前三阶段交接、两条实际有效时间和外部收尾依据。阅读 [复验与整理](references/workflow.md)，复测各轨迹最佳候选，恢复 /workspace/solution 后独立 Verifier 评分。按同 Public/Dev 口径选唯一 best_method 并在干净快照重跑，不用 Reference 代替。
2. 按 [六项文件与归档](references/delivery-contract.md) 整理 workspace、expert_evidence、optimization_evidence；保留全部成对 seed、原始失败日志、条件性模型和重载。best_method 保存最终代码，评分模型按任务合同交付。
3. 最终 QA 必须读取原样随包的 [任务 QA](references/supplied-qa/autoresearch-task-qa/SKILL.md) 和 [Baseline QA](references/supplied-qa/autoresearch-baseline-quality/SKILL.md)，按其 references 阅读规则及 review schema。核对 [适用口径说明](references/rule-alignment.md)，不得偷偷改原 QA 让任务通过。
4. 使用 `python3 scripts/qa_runner.py SOURCE --out QA_DIR` 仅收集；阅读全部实现和证据，按 QA 要求建立 review.json 后再用 `--review REVIEW_PATH` 生成终稿。报告放待检包外。首轮收集器退出码0不等于通过，必须检查报告结论、G01–G03、B01–B08、QA01–21、H01–06和复核完成状态。
5. 按 [checklist模板](assets/checklist.md) 填真实证据、失败/缺口和修复。QA静态阶段不执行提交或加载模型，动态修复/复验后再冻结新包重新审查。
6. 按已确认平台导入约定填写 [delivery-map.json](assets/delivery-map.json)，执行 `python3 scripts/bundle_delivery.py MAPPING --root SOURCE_ROOT --out NEW_DELIVERY_DIR`。脚本生成两ZIP和四独立文件，逐成员核验路径/哈希，内部 manifest 不是第七项必交。逐文件归档边界不能无依据声称是官方固定规范。
7. 提供六项真实链接和自验范围：evidence.zip、harbor_task.zip、小规模试跑结论、GPT-5.6 Sol Max轨迹、Seed 2.1 Turbo High轨迹、自检checklist。未完成不能包装成验收通过；平台正式 Hidden/防作弊与最终验收仍由平台负责。

## 便携来源与测试

本包包含用户提供第二版的原样 QA、[教程PDF](references/source-documents/tutorial-v2.pdf)、[最终清单](references/source-documents/deliverables.md)及哈希清单。文件内历史来源链接仅用于溯源，不能据此回退使用第一版规则。

运行新增工具测试 `python3 -m unittest discover -s tests -v`；QA 原测试在 `references/supplied-qa/autoresearch-task-qa/scripts` 下执行 `python3 -m unittest discover -p "test_*.py"`。完整阶段细节见 [流程原文](references/flow-detail.md)。
