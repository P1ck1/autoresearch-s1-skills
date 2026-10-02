# S1 第二阶段：双镜像、模型连接与试跑

接收已确定的方法、协议和实验标定，交付可启动正式研究的双镜像任务、两模型连接证据、小规模试跑结论及外部编排。实际构建、连接和验证均在云 GPU 服务器完成。

## 范围与输入

固定 S1，遵循用户第二版教程、example、QA skills、六项交付清单和最新补充要求。第一版规则不适用。详细流程见 第二部分（原文件身份见 sources.json；核心要求已收入本包）。这是专家侧 Skill，不直接提供给研究 Agent。

先读服务器 `expert_evidence/阶段交接.md`，核对任务/代码/数据/协议版本、Baseline/Reference 实测与质量门、Starter、公开评分和可信私有评分逻辑。定位目标 Harbor 版本、资源及模型认证方式。已有授权范围内直接实施；仅缺必要信息时询问，不因阶段切换重复确认。

本阶段包含有限、预先界定预算的两模型接入与工具闭环试跑，不包含两条正式 10h/11h 搜索。试跑从临时干净任务启动，输出独立保存，不计入正式研究时长。服务器或模型配置不明时先检查，不猜测服务参数、模型 ID 或平台命令。

## 服务器依赖与双镜像

1. 检查实际 Docker daemon/构建/容器权限、NVIDIA 驱动与容器 GPU、CPU/内存/共享内存/磁盘、网络、基础镜像缓存、目标 Harbor 和 adapter。受限容器云必须实测是否支持独立容器，不只看命令存在。
2. 列明已有可用、缺失和版本冲突，复用正确依赖，仅安装必要项；不自动购买或扩容服务器。宿主机依赖不能替代镜像内依赖。
3. 固定目标 Harbor 版本，阅读该版本配置模型/CLI/provider，验证 separate、artifacts 和 GPU 分配实际生效。不能把例题 schema 或版本号当作所有部署通用。
4. 以以下职责构建任务，内部评分模块可按题目组织：

   ```text
   workspace/harbor_task/
     instruction.md
     task.toml
     environment/                    # Agent build context
       Dockerfile
       requirements.txt
       starter/
       public_eval/
       public_assets/                # 按需放 train/dev/公开资源
     tests/                          # Verifier build context
       Dockerfile
       test.sh
       可信grader及约束模块
       授权私有资产或生成/安全注入机制
   workspace/reference/              # 专家侧，不给 Agent
   ```

   不用整个任务根作为 Agent 镜像上下文，不越界 COPY。Starter 初始化只读 `/workspace/starter` 和按题面可写的 `/workspace/solution`；公开评分放 `/workspace/public_eval`。只移交约定的 solution 代码、模型和配置，不传整个工作区、reward、Reference 或专家证据。
5. 两套镜像都实际构建，保存构建日志和不可变标识。镜像/依赖变化若影响第一阶段结果，复验受影响的 B/R 证据。

## 题面、评分与隔离

`instruction.md` 写实质内容并与 machine config 对应，八部分为 Goal、Task Setting、Objective and Metrics、Allowed Scope、Hard Boundaries、Submission Instructions、Workflow & Iteration、Completion Criteria。写清公开命令、提交合同、网络/工具策略和冻结边界，不泄露论文标题、仓库答案线索或参考方法；不要把研究总时长写成题目单次出分预算。

Agent 每轮只用 Public/Dev。结束后 Harbor 移交最终候选，独立 Verifier 执行 `bash /tests/test.sh`，写 `/logs/verifier/reward.txt` 或非空数值对象 `reward.json`，不依赖启动 cwd。stdout 分数或 pytest 成功不能替代实际评分。清理预写/残留 reward，由可信进程原子写入；冻结文件由可信副本恢复或校验。

候选在 Verifier 中仍不可信：隔离用户/进程/文件权限，只提供题目允许输入，不能读取私有标签、Reference 或写可信评分器和 reward。不得把不可信候选导入具有全部私有权限的 grader 进程后宣称隔离完成。Hard Gate、格式错、超时、资源超限、基础设施故障按协议分类；非作弊 Hard Gate 返回 -1，其他失败不一概伪装为 0 或 -1。

Reference、专家证据、正式私有评分及日志不进入 Agent 镜像、共享目录、缓存或 Git 历史。研究容器无宿主机管理凭据、Docker socket、容器管理工具或访问其他轨迹的权限。

## 测试两个指定模型

固定组合：**Codex + GPT-5.6 Sol Max**、**Codex + Seed 2.1 Turbo High**。测试真实部署的调用位置、认证、代理/端点、Codex/Harbor adapter 和模型/档位映射，不用显示名称猜 API 参数。

按成本递增验证：

1. 检查现有客户端、账户权限、网络与认证。凭据仅在可信运行侧管理，不注入候选可读环境或写入题面、镜像和交付日志。
2. 分别发起最小真实模型请求，确认有效返回，核查模型与档位配置依据，保留脱敏请求标识、耗时、结果或错误。HTTP 可达不等于模型可用。
3. 分别通过正式 Agent 接入链路运行有限工具闭环：读公开题面、合法修改/工具调用、Public/Dev 出分、八字段记录落盘。单次文本回复不算 Coding Agent 接入完成。
4. 若正式计划并行，做最小并发验证，检查限流、输出隔离和资源争用；顺序运行时记录安排和共同资源规则。
5. 各自形成通过/失败/未验证结论，记录实际版本、调用位置、模型和档位核验、原始评分输出、错误修复与复测。不可用时报告阻塞，不静默换模型、降档或把默认值视为生效。

## 外部编排与端到端试跑

在目标服务器实现或复用可检查的外部编排，正式启动命令准备好但本阶段不启动长程研究：

- 两条干净起点、独立输出和持久证据目录；任务及资源协议相同。
- 每条最低 10h 有效研究、目标 11h，不启用 7h 例外。墙钟预算覆盖开销和最终复测；12h 容器稳定性与有效研究时间分别记录。
- 编排在研究进程权限之外，负责启动、继续调度、恢复、计时、强制记录、收尾及容器生命周期；模型无权改时限、停研究或删除容器。
- 模型提前返回时不能直接算完成；外部记录事件并继续合法研究，排除无效空档。遇到不可恢复故障或已授权预算耗尽，外部记录中断，不伪称完成或无限追加费用。
- 每轮八字段及评分来源必检：round、policy_name、method_summary、status、score、failure_reason、retained_best、time；score 是评分器输出的 Public/Dev 原始指标，失败未出分为 null。外部保存有效/排除时间区间、原因和事件，模型不能篡改。

实际测试 Starter 出分、两模型工具闭环、移交必要产物、独立 Verifier 出分、冻结面保护和约束执行，以及提前返回/中断时的监督和记录机制。仅使用授权自验数据时明确正式 Hidden 尚待平台，不能冒充官方结果。生命周期检查在可丢弃的试跑对象上进行，避免影响已有研究或他人容器。

## 完成与交接

写 `expert_evidence/小规模试跑结论.md`，涵盖任务/镜像/模型版本、试跑预算、实际执行、结果、问题及修复、未验证部分和证据路径。更新 `expert_evidence/阶段交接.md`：实际构建命令与镜像标识、模型接入配置依据、外部编排位置、启动/恢复/外部停止方式、资源和日志目录、就绪与未完成项，不记录秘密。

只有两模型、必要评分链路、隔离和监督机制均有实际证据才称“可启动正式研究”；不能仅交 Dockerfile 或 Prompt 后声称就绪。原始日志和配置保持专家侧；本 Skill 及试跑候选不作为正式研究起点。

下一阶段：`$autoresearch-s1-research`。用户只要求本阶段时交付后结束；已授权全流程时按就绪条件进入下一阶段。
