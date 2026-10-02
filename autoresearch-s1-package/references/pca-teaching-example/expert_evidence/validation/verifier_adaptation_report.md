# PCA 独立 Verifier 适配说明

本次只整理教学包的 `workspace/harbor_task/tests/` 与 `workspace/harbor_task/task.toml`；未调整 PCA 数学定义、评分公式、正式数据集、资源预算或候选安全执行机制。本报告不是完整 Hidden 跑分或平台验收结论。

## 来源与修改范围

源为采购包内层 `population-genetics-pca/environment/source/env/tasks/from_scratch_pca/`，不是最外层 Hyperfocal 包装。

原 `tests/` 共 25 个文件全部保留。只有 `task.toml`、`tests/Dockerfile`、`tests/test.sh` 共 3 个文件发生字节变化：将 `/app/submission` 统一替换为 `/workspace/solution`，包括注释、候选目录创建、WORKDIR、权限设置、日志描述与正式 grader 参数；另将 task.toml 的任务名称、task_id、ext_id 和展示标题改为教学变体标识（其余资源不变）；将 test.sh 最后的 reward 序列化改为有限 JSON 数值校验、同目录临时文件写入与 `os.replace` 原子替换。其余 23 个 tests 文件与来源逐字节一致。未运行原生评分器或完整 test.sh；只独立执行了修改后的 reward 输出片段做验证。未打印私有评测 key 内容。

## 双镜像与候选移交

- Agent 构建上下文为 `workspace/harbor_task/environment/`；Verifier 构建上下文为 `workspace/harbor_task/tests/`。
- `task.toml` 保留 `schema_version = "1.3"`、`[verifier] environment_mode = "separate"`、`user = "root"` 和独立 `[verifier.environment]`。
- 顶层 `artifacts = ["/workspace/solution"]`。Harbor 在 Agent 结束后移交候选；在新 Verifier 中恢复到同一路径，未导入 Agent 的 `/logs` 或 reward。
- `tests/Dockerfile` 的 `COPY . /tests` 在构建时装入评分入口、模块与私有材料。运行期不依赖 Harbor 再上传 tests，亦不依赖 Agent 的临时安装或未声明文件。
- `/tests/test.sh` 实际执行可信 grader、读取 `/workspace/solution`，输出 `/logs/verifier/reward_detail.json` 和 `reward.json`。候选 CLI 仍是目录中的 `pca`，签名为 `pca <vcf_path> <k> <out_path>`；本次未改成 `method.py`。
- 未增加每轮远程评分服务或第二个 Coding Agent；公开 Dev 迭代由 Agent 侧入口负责，不在本次 Verifier 适配范围内。

## 保留的资源与依赖

两侧均为 8 CPU、16,384 MB 内存、32,768 MB 存储、0 GPU；Agent timeout 86,400 秒；Verifier timeout 10,800 秒；Agent build timeout 1,800 秒，Verifier build timeout 3,600 秒。运行期网络仍为 `no-network`。在容器内运行模型客户端的平台，须由平台确认模型服务连接与其 Harness 实现兼容，不能把本地静态配置检查当作 Agent 已启动成功。

Verifier 保留 `python:3.12.8-slim-bookworm`、`numpy==2.2.6`、`scipy==1.15.3`，以及 `ca-certificates`、`curl`、`util-linux`。需要 Linux root、chroot、setpriv 等原生执行边界；正式环境不能用普通本机 Python 跑分代替。构建时需下载原样 checksum 固定的 1000 Genomes chr22 源文件（Dockerfile 校验字节数为 205,612,353）及 panel，并衍生私有真实数据；评分时另生成合成套件。本次未发起该下载、未构建镜像、未运行全量 Hidden。

## Hidden 与候选执行权限

原私有真实数据、标签及密钥的预置/生成方式保持不变。`tests/grader_pkg/reference/` 内保留 fast_pca.py、full_scan_pca.py、pca_core.py 等评分私有副本，供正确性与速度锚点使用；“Reference 不向 Agent 暴露”并不等于“Verifier 内禁止 Reference”。它们只能留在私有 tests 构建上下文中，不得复制到 Agent 环境。

`tests/grader_pkg/grader/private/release-v3.math-key` 为原包的私有评测材料，内容未修改或输出。含此文件的完整教学 ZIP 是专家/平台侧完整交付材料，不可作为 Agent 可见包整体挂载，也不适合无访问控制的公开发布。

可信 grader 从 root 所有的 `/tests` 启动，设置安全 Python 模块查找路径，避免候选目录劫持导入。候选被制成有界、禁止符号链接跟随的不可变快照；在只含固定 runtime 和本次输入输出的 chroot 内降 UID、组与 capabilities 执行。私有标签、Reference、评分器和最终 reward 不对候选开放；网络、时间、内存、进程、IPC/临时状态清理与输出校验保持原实现。

## 与教程评分要求的实际差异

这是“按教程目录整理、保留原生 PCA 评分的双镜像封装示例”，不是“已满足教程全部题目质量与证据门槛的合格新题”。

1. 原生单数据集得分为 `accuracy × (0.10 + 0.90 × systems_unlock(accuracy) × time_quality) × method_factors`，再按类别聚合。实现会将多个指标/因子限制到 `[0,1]`，总体也是有界分数；不是教程中的 `(x-B)/(U-B)` 或 `(B-x)/(B-U)`，不能据此声称超过预计上限后分数仍可大于 1。
2. full_scan 是正确性真值和运行时间对照，未映射为教程要求的 Baseline=0；fast_pca 是快速参考与可达速度锚点，设计上可达到接近/等于 1。不能把原生快参考分数认定为已符合教程 Reference 在 `[0.15,0.8]` 的要求，也不能把源码中历史跑分当成本次实测。
3. 原生接口/依赖等无效提交会产生带 failed 状态的零分结果；部分超时/未运行数据集按零分纳入聚合。这与教程 Hard Gate=-1、不同失败类别不能混成一个分数的规则不是完全相同。本次保留原行为，没有静默重写错误语义。
4. 原脚本直接写 `reward.json`，本次已对齐教程的原子输出要求：保留启动时清理旧 reward，将最后的标量输出改为检查 reward 是有限数值，以 `allow_nan=False` 写同目录临时文件，flush/fsync 后通过 `os.replace` 替换最终文件，并清理临时文件。最终结构仍为 `{"reward": 原生数值}`；没有重新缩放、裁剪或改动数学得分。
5. 未补造正式 Baseline/Reference 成对 seed 训练记录、专家模型轨迹或最终 Hidden 结果。要作为符合全部教程门槛的正式新题，需另行审定评分适配并按目标协议真实验证，不属于本次目录重排。

## 本次已完成的检查

- 逐文件比较：tests 完整保留 25 个文件；Dockerfile 仅替换候选路径，task.toml 同时更新候选路径和教学变体标识；test.sh 除路径替换外仅改最后的 reward 输出片段；23 个 tests 文件字节不变。
- 所有交付的 tests 文本/二进制内容及 task.toml 均未残留旧 `/app/submission` 候选路径。
- 21 个 Python 文件语法静态解析；Shell 中 4 段内嵌 Python 同样做 AST 解析。未导入或执行 grader、数据生成器、Reference 或候选。
- 独立检查修改后的 reward 输出片段：有限小数及零值保持原数值和 JSON 结构并替换既有文件；NaN、Infinity、布尔值被拒绝，未产生最终 reward；成功/失败用例均无遗留临时文件。检查使用临时目录中的合成 detail，不运行完整评分链路。
- `bash -n tests/test.sh` 通过。
- `tomllib` 解析 task.toml 通过；除 artifacts 与教学变体标识外，其余解析后配置与原文件相同。
- Verifier Dockerfile 的全部 COPY 源均存在且位于 tests 构建上下文以内。
- 私有 key 与评分核心内容字节保持一致；本次操作未写入任何 Agent 环境目录。

尚未完成：目标 Harbor 解析/调度实跑、两套镜像构建、全量 Hidden、Linux sandbox 可用性验证、12 小时稳定性验证；不能写“已构建成功”“评分通过”或虚构分数。系统默认 Python 缺少 TOML 解析模块，本次配置检查使用 Codex 捆绑 Python 的标准库 tomllib 完成，未为此安装包。
