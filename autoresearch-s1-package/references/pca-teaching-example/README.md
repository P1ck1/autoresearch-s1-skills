# PCA 教学对齐示例

版本：2026-09-29。依据用户提供的 population-genetics-pca.tar.gz 内层任务重排。

**用途：演示教程的三类交付目录、公开 Dev 迭代、结束后独立 Verifier 评分，以及参考解的隔离位置。本包是教学变体，不是全协议验收通过题。**

## 下载后先看

1. workspace/harbor_task/instruction.md：按教程八章节填写的真实题面。
2. workspace/harbor_task/environment/：公开起始方案、公开 Dev 评分、公开小数据和 Agent Dockerfile。
3. workspace/harbor_task/tests/：原 PCA 的私有评分器及独立 Dockerfile；提交路径改为 /workspace/solution。
4. workspace/reference/：原主参考解和可执行候选版本，仅供专家。
5. expert_evidence/专家作业说明文档.md：文件映射、操作命令、已做验证与未完成项。
6. optimization_evidence/训练证据说明.md：本题不训练模型；为何不能伪造训练/成对实验数据。

完整教学包包含评分源码和私有评测材料，只能交专家/平台，不可整体挂载或复制给 Agent。Agent 只使用 environment/ 构建出的公开镜像，公开题面由 Harness 提供。

## 已做的格式适配

- 使用 workspace/harbor_task、workspace/reference、expert_evidence、optimization_evidence 结构。
- 候选路径从原 /app/submission 改成 /workspace/solution，artifacts、Dockerfile、评分脚本一致。
- 保留 environment/ 与 tests/ 两个独立构建上下文和最终 separate Verifier。
- 原内层任务是空提交目录；这里新增可执行全扫描 Starter、独立公开 Dev 小样本与评分入口，让专家能按教程体验“修改→评分→继续迭代”。这会改变原任务的初始难度，不能把旧 README 的表现当成本变体结果。
- 原外层 workspace/submission/pca 实际等于快速参考解，没有用它初始化 Agent。
- 公开程序入口沿用 pca <vcf> <k> <out.tsv>，不强行改成 method.py；这符合教程“内部文件按优化面设置”的约定。
- 私有评分数学、数据生成和隔离执行实现保留；最终 reward.json 写入按教程改为原子替换。

## 评分差异必须保留

Public/Dev 使用包内明确说明的教学代理指标，供中间反馈；它不等价于正式 Hidden 分数，不能保证两者方法排名一致。公开小数据只能证明接口能运行，不能证明 10h 优化空间。

Hidden 保留原生 PCA native_reward：各数据集正确性与速度、方法因子结合，经源代码规定的聚合输出 [0,1] 分数。源评分包含截断和分段，完整扫描不是 B=0，快速参考是速度满分锚点，部分无效提交原生返回 0。因此本包没有假称已满足教程的“B=0、Reference∈[0.15,0.8]、连续不裁剪、Hard Gate=-1及其他失败分类”统一要求。

若要作为正式题提交，需先按统一评分合同改造并重新标定，重新运行 Baseline/Reference 全部成对协议、噪声统计与资源检查，再补两套 Agent 长程轨迹。不能只在原封顶分数外套一个线性映射就宣称消除了饱和。

## 最小公开演示

以下在本包根目录执行；需要已启动的 Docker。

```bash
docker build -t pca-teaching-agent:local \
  -f workspace/harbor_task/environment/Dockerfile \
  workspace/harbor_task/environment

docker run --rm --network none --cpus 8 --memory 16g \
  pca-teaching-agent:local \
  /opt/hyperfocal/pcabench/bin/python /workspace/public_eval/grader.py \
  --submission /workspace/solution
```

该命令演示一次 Starter 的公开评分，不会运行 Coding Agent，也不会访问 Hidden。正式自迭代使用平台提供的 Agent/Harness 启动方式，不在此虚构私有平台命令。

## 最终独立评分

```bash
docker build -t pca-teaching-verifier:local \
  -f workspace/harbor_task/tests/Dockerfile \
  workspace/harbor_task/tests
```

原 Verifier 构建需要 Linux、受支持架构和构建期网络，会下载经过校验的约 206 MB 压缩 chromosome 数据并派生私有资产；运行时无网络。不能把仅有 Dockerfile 当作镜像已构建完成。

平台按 task.toml 启动本题。Agent 每轮调用公开 Dev；结束前恢复 Dev 最佳候选到 /workspace/solution。Harbor 按 artifacts 传递该目录，在独立 Verifier 内运行 bash /tests/test.sh，结果为 /logs/verifier/reward.json。测试候选在评分时进入原有 chroot/降权/资源限制环境。

## 验证与证据状态

本轮真实检查记录放在 expert_evidence/validation/；本机 Docker daemon 不可用，未构建两套镜像、未执行完整 Harbor 往返和 Hidden。任何本地 smoke 的解释器/依赖版本都单列，不冒充固定镜像结果。

原包没有对应本教程的真实逐轮轨迹或完整成对运行记录。两份 trajectory 保留空 rounds；best_method/ 只有待补说明；正式分数、时长和统计量填写 null/NOT_RUN，而不把 Reference 冒充 Agent 最佳方法。无训练模型，因此不创建空 model/。
