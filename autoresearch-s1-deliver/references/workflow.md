# S1 第四阶段：最终复验、QA 与六项交付

在云 GPU 服务器把前三阶段的实现与真实记录整理为可复验的六项交付。本 Skill 是专家侧操作指南；不向研究 Agent 暴露 Reference、私有评分或其他轨迹。

## 读取输入并核验状态

固定 S1，使用用户第二版教程、example、QA skills、交付清单和最新补充要求。第一版规则不适用。详细流程见 第四部分（原文件身份见 sources.json；核心要求已收入本包）。

先读服务器 `expert_evidence/阶段交接.md`，核对：

- 论文方法来源、正式协议、Baseline/Reference 全部成对结果、训练模型与重载证据。
- 双镜像版本、实际构建/运行证据、模型连接与小规模试跑结论。
- Codex + GPT-5.6 Sol Max、Codex + Seed 2.1 Turbo High 两条独立轨迹，各至少 10h、目标 11h 的有效时间证据及外部收尾决定。
- 两条各自的最佳候选、代码/模型/配置与实验的对应关系。

先核查已有材料而不是重跑所有工作。缺证、时间不够或仍在研究时，不自行判完成、不删除运行现场；推进可独立进行的整理，说明具体缺口，按已有授权返回相应阶段补全。

## 最终方法与独立复验

1. 外部授权收尾后，对每条的最佳候选执行 Public/Dev 复测，将代码和任务合同要求的模型/配置恢复到 `/workspace/solution`。不能停在最后一次失败方案。
2. 每条结束后的候选通过已验证的 artifacts 链路移交独立 Verifier，执行真实 `/tests/test.sh` 并保存配置、候选标识、日志、reward 和 Trial 结果。已有完整对应运行证据可以复核使用。用授权自验数据时明确与正式 Hidden 区别，不把私有分数回传继续调优。
3. 在相同 Public/Dev 质量门和评分口径下比较两条有效方法，选归一化分数最好的唯一方案，并在干净任务快照独立重跑。固定协议有随机重复要求时遵守，不挑最好一次。
4. 不能重跑时修复合法打包/运行问题后复验，或选上一份可复现方法；不得用专家 Reference 代替 Agent 的 best_method。
5. 保存 `expert_evidence/best_method/` 的完整最终代码、入口和配置，不放训练 checkpoint。评分所需模型按提交合同随候选移交或采用平台认可的不可变资产入口；记录选择和复测依据。

## 组织材料

```text
workspace/
  harbor_task/                         # 两套环境、题面、配置及评分
  reference/                           # 专家可执行参考实现
expert_evidence/
  专家作业说明文档.md
  expert_annotation.json
  run_summary.json
  trajectory_codex.json
  trajectory_seed.json
  best_method/
  小规模试跑结论.md
  阶段交接.md及必要验证/计时附件
optimization_evidence/
  训练证据说明.md
  baseline_runs/seed_<真实seed>/
    result.json
    run.log
    model/                             # 仅训练型，模型+artifact.json+reload.log
  reference_runs/seed_<真实seed>/       # 同结构，全部正式 seed 配对
  comparison_summary.json
```

- 专家说明写问题、允许范围、两镜像 context/命令/标识、目标 Harbor 版本、运行/评分/隔离、B/R/U 和统计、两条研究过程、选优与限制。实际命令和结果不能靠配置反推。
- annotation 是精简任务标注，不混入日志或官方审核结论；run_summary 写专家自验、模型组合、实际有效时长和复测状态。必要额外事件记录作为附件，不随意改平台 schema。
- 两轨迹按八字段 rounds 结构整理：round、policy_name、method_summary、status、score、failure_reason、retained_best、time。未成功出分为 null；公开原始指标与归一化分数不混淆。
- 逐 seed 原始日志、失败、实际模型、哈希及干净进程重载必须可追溯；正式 B/R 独立实验不能拿 Agent 日志或同一模型重复评估替代。
- 非训练任务不创建空 model/；额外审计或消融不冒充最小必交项，不为了精简删除必要证据。

## 正式 QA：读取原规则，完成语义复核

最终 QA 以用户提供的原始 QA skills 为准，不能仅根据本文摘要打勾。默认位置：

- 任务 QA（原文件身份见 sources.json；核心要求已收入本包）
- Baseline QA（原文件身份见 sources.json；核心要求已收入本包）
- 第二版教程（原文件身份见 sources.json；核心要求已收入本包）
- 交付清单（原文件身份见 sources.json；核心要求已收入本包）

可在本地阅读规则，将同版本 QA 目录和必要规范上传到服务器专家侧运行质检，记录来源/哈希；不能假定 Windows 路径在 Linux 存在。材料移动时定位用户当前副本；不可访问时明确缺口，不宣称完成最终 QA。

按任务 QA 指定顺序读取 research-quality、implementation-policy、submission-format、docker-path-contract、harbor-harness、implementation-report 等参考及 Baseline 专项。核对适用版本：本项目构建 contexts 为 environment/ 与 tests/，轨迹提交八字段，各至少 10h、目标 11h且不启用 7h 例外。若工具保留不兼容路径假设，记录具体冲突、选择有证据支持的 profile/适配；不能偷偷修改检查器放行或回退采用第一版布局。

先介绍方法空间、Baseline/Reference 实际方法与全部成对跑分，再核 G01–G03、B01–B08、QA01–QA21 和 Harbor H01–H06。QA15 按规则跳过，但已有资源和公平预算要求仍需验证；QA07/08 的题面泄露检查不能冒充完整隔离认证。

从 QA Skill 根目录运行其可信收集器，报告目录放在待检材料外：

```text
python3 scripts/audit_task.py <完整材料目录或ZIP> --out-dir <独立QA目录>
# 阅读全部材料并按原 schema 建立 review.json 后：
python3 scripts/audit_task.py <同一材料> --out-dir <同一QA目录> --review <review.json路径>
```

命令中的路径用实际服务器位置替换。首次收集器输出不是终审。review 中 G01–G03、QA01–QA21、H01–H06 分别恰好各一次，填 overview、format/runtime_review、真实证据与修复/复验条件；不能手填 pass 覆盖脚本数值反证。最终回读 TXT/Markdown/JSON 及退回说明，避免无 review 的运行覆盖终稿。

QA 阶段只读，不执行待检代码、Docker 或反序列化模型。动态复验和修复属于前述已授权制作环节，修复后冻结新版再静态审查。缺失记未完成，明确失败记不通过；缺材料不是已证实造假。静态通过、专家实跑和平台正式验收分别表述。

## 自检、修复与六项打包

自检 checklist 写检查项、实际状态、证据位置、问题/修复和复验依据，覆盖质量门、主表选择、双镜像、模型档位、权限/外部停止控制、八字段、有效时长和六项文件完整性。只填写有证据的通过；每个未完成或失败项具体说明。

修复可确定问题并复验受影响部分，不无意义重跑无关实验。不为通过 QA 改锚点、挑 seed、补造时长、轨迹或平台结果。收集器报告不能代替语义复核。

最终六项必须单独可定位：

1. `evidence.zip`。
2. `harbor_task.zip`。
3. 小规模试跑结论。
4. Codex + GPT-5.6 Sol Max 轨迹。
5. Codex + Seed 2.1 Turbo High 轨迹。
6. 自检 checklist。

先查看平台实际导入要求和已有归档约定，再将逻辑三目录映射到两个 ZIP，记录每份 Reference、最终方法、模型和证据的归属。用户尚未指定 ZIP 内逐文件边界时，不声称某布局是官方强制；在兼容性可验证时作最小可逆归档选择并说明。缺确实影响导入的要求时，先完成其余材料，再询问该项。

任务包不能混入专家私有材料到 Agent 可见镜像。移除密钥但不改写实验事实；大模型按任务允许方式保留哈希及不可变入口。列出压缩包内容并在独立目录解压检查路径、引用、文件完整性，两份独立轨迹与证据包对应版本一致。QA 可审完整三目录 staging，最终还需检查分包后引用和可运行性；不因分包丢失必需证据。

更新阶段交接并向用户提供六项真实文件链接、QA 结论、复验范围与剩余限制。不得将未通过材料标成正式完成，也不声称平台验收通过。上传外部平台或发送他人仅在已有授权范围内执行。容器及临时资源清理由专家侧/编排在保全必要产物后按授权处理，研究模型无权清理。
