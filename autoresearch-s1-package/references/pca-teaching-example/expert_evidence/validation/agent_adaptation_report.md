# PCA 教学包：Agent 环境适配记录

完成范围仅为 `pca_teaching_example/workspace/harbor_task/environment/`。
本记录不代表完整双镜像、正式 Hidden、G01/G02/G03 或长程轨迹验收通过。

## 源码核查与公开 Baseline

已先阅读源 `reference/full_scan_pca.py`（完整文件）和 `reference/pca_core.py`
（228 行），核查输入、FORMAT 解码、缺失值、HWE 标准化、Gram/SVD 路径及直接写输出的 CLI。
仅将这两个全扫描组件适配为公开 `starter/pca` 和 `starter/pca_core.py`：

- 保留全部 eligible marker 的精确计算；没有减样本、削弱精度或故意注入故障。
- 将包内相对导入改为提交目录内的 `pca_core` 导入；入口仍为 `pca <vcf> <k> <out>`。
- 公开文案去掉源参考模块路径、私有 fast 方法线索和论文/仓库说明。
- 不复制 `fast_pca.py`、`fast_submission`、原外层 `workspace/submission/pca`、
  `grader_pkg`、正式评分代码、原数据生成器、隐藏 seed 或 release key。
- 本变体新增了可运行的公开 full-scan Baseline；原内层任务并未自带本套 Public/Dev 教学材料。

全扫描 starter 的 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `starter/pca` | `635090a69f134c4971b31d36171b2024a1498dd36adcccbaefaac2676b0bf0c8` |
| `starter/pca_core.py` | `bb7a102c4408e56abe1fb0a933f8cbc71b9178bdd5fc507b8874e82b14c0c030` |

## Agent 运行方式

Docker 固定 `python:3.12.8-slim-bookworm`，NumPy `2.2.6`，SciPy `1.15.3`。
使用二进制 wheel，不安装编译工具。运行用户为非 root 的 `agent`。
构建时从 `starter/` 初始化 `/workspace/starter` 和 `/workspace/solution`；
Starter、公开评分器、公开资产和 README 由 root 拥有，Agent 无写权限；
`solution/` 为候选修改目录，`/expert_evidence` 是该次运行的空白笔记输出目录。

在教学包根目录构建：

```bash
docker build -t pca-teaching-agent:local \
  -f workspace/harbor_task/environment/Dockerfile \
  workspace/harbor_task/environment
```

容器内每轮反馈：

```bash
python /workspace/public_eval/grader.py --submission /workspace/solution
python /workspace/public_eval/grader.py --submission /workspace/solution --repeats 3
```

没有引入 `solve.sh`、每轮 Docker build 或每轮独立 Verifier 服务。
最终接口仍为 `solution/pca`；需要的 helper 位于同一提交目录。

## Public/Dev 代理协议

三个独立公开 synthetic fixture 由新写的 `public_eval/generate_dev.py` 生成，
不导入采购包的数据生成器或任何私有材料。固定 fixture 字节随包交付，
`public_assets/dev/manifest.json` 包含 seed、维度、大小与 SHA-256。

| case | 公开 seed | samples × generated markers | k | SHA-256 |
|---|---:|---:|---:|---|
| structured | 2026092901 | 64 × 768 | 3 | `b294f95de31c624814583a79e685a74b0bacef722a35d9485f5eb39d2371e629` |
| mixed_calls | 2026092902 | 80 × 960 | 4 | `3c7d6409bcc0eb5296dade6fab81883e82aa779d6893c54f139faa5361bc8e0e` |
| sample_heavy | 2026092903 | 96 × 40 | 3 | `193cc71782acfb2d3f088af5ec82524ffc240b80c225d9c869113c56cf721c28` |

每例另加 4 个 ineligible/monomorphic 记录；mixed_calls 包含 GT:DP:GQ、
haploid、缺失与 partial call。fixture 总计 724,106 字节。
生成时使用 Python 3.9.6 / NumPy 2.0.2；目标镜像无需重新生成。
跨 NumPy 版本重新生成可能改变字节，应以随包哈希为输入版本，不能默默覆盖。

评分器先用冻结的公开 Baseline 生成该 case 的精确结果，再运行候选。
对中心化、按列归一化的两组 score 取 SVD 正交基，计算
`similarity = ||Q_candidate.T @ Q_baseline||_F^2 / k`。
要求输出样本 ID/顺序、列数、有限性均正确且满列秩，所有 case 的每次运行
similarity 均至少 0.98。阈值只属于本教学 Public/Dev 协议。

成功时：

`score = mean_cases(log2(median(t_baseline) / median(t_candidate)))`。

该值是 maximize 方向、未裁剪的原始 **Dev 时间比代理值**，不是 B/U 归一化的
正式分数。默认每程序每 case 一次，支持 1–5 次并取时间中位数；两方重复次数相同。
计时含 Python 启动、读 VCF、计算和写文件，单次 30 s 超时。
失败返回 `score: null`，分类 status 和非零退出码；质量门不通过时保留逐例诊断。

固定执行顺序为 Baseline 然后 candidate，因此 cache/顺序/进程启动噪声可能很大。
即使候选就是 Baseline，一次计时的 score 也不保证等于 0。
这三个小样本只演示输入、解析、数值接口及迭代反馈，不能证明正式提升、
真实大数据性能、10h 优化空间或 G03 显著性；不能用公开分数替代 Hidden reward。
公开 evaluator 本身不是完整安全沙箱或正式 library/资源规则验收器。

## 已实际运行的轻量验证

本地 smoke 使用现成 bundled Python `3.12.14` / NumPy `2.3.5`，
路径 `/Users/bytedance/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`。
该 runtime 没有 SciPy；本公开 Baseline/Dev 仅用 NumPy，因此 smoke 未覆盖 SciPy 使用。
系统 Python 3.9.6 的 NumPy/SciPy 在 user site，第一次尝试被 grader 的
`PYTHONNOUSERSITE=1` 隔离而报 `evaluator_error`，随后改用上述 bundled runtime；
未下载或安装额外依赖，也未执行采购包原始全量 grader。

- 4 个 Python 文件 AST 静态解析：通过。
- 3 个冻结 fixture 的 SHA-256 核验：通过。
- 公开 starter 作为 candidate 通过 3 个 Dev case，子空间 similarity 均为 1.0。
- 该次 timing score 为 `0.38416920225556045`，仅为噪声敏感的 smoke 输出。
  三例 Baseline/candidate 秒数分别为 0.091240/0.047800、
  0.077174/0.068445、0.051890/0.050239；同代码得到正值恰好说明不能当作方法提升。
- 独立手工 6-sample 小 VCF，含缺失与应排除 indel，按手写 dosage 的 NumPy SVD
  独立核验 2 PCs：similarity `0.9999999999999031`。
- 错误 TSV header：返回 `invalid_output`、`score: null`、退出码 1。
- 全零/缺秩 PC：质量门拒绝；缺少 `pca`：`invalid_submission`。
- Agent context 扫描未发现 fast 参考模块、release key、私有 grader 包或原论文线索。
- 本地验证产生的 `__pycache__` 已清理，构建忽略 pyc 和缓存目录。

最终源码又以 `--repeats 3` 实跑一次（总耗时 1.371555 s），退出码 0、
全部质量门通过、Dev score `0.05065513417874165`。真实 stdout、stderr 和
带 UTC 时间、准确命令、版本及源码 SHA-256 的执行元数据保存在包内
`expert_evidence/validation/agent_smoke/final_smoke.*`。
先前失败首尝试、单次成功和小检查仅有工具 trace，没有伪装成原始日志；
其来源与摘要记录在同目录 `README.md`。

## 待验收

Docker daemon 当前未运行；本次未拉取/构建镜像，非 root 文件权限和目标 pinned
依赖组合仅静态检查，仍需实际构建及容器运行验证。未运行正式 Hidden、原全量
grader、大型真实 VCF、完整防作弊/资源门、12h 稳定性或长程 Agent 轨迹。
未制作或声称 G03 成对实验证据。Agent 最终候选移交与私有 Verifier 集成由
本轮其他适配部分及后续平台验收覆盖。
