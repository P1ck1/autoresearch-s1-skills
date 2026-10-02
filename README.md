# AutoResearch S1 Skills

AutoResearch S1 专家侧四阶段 Skill 工具包，依据用户提供的第二版教程、QA 与 PCA 双镜像示例整理。

| Skill | 阶段 |
| --- | --- |
| `autoresearch-s1-prepare` | 论文接收、Baseline/Reference 选择、成对实验、归一化与 3σ 标定 |
| `autoresearch-s1-package` | 云 GPU 双镜像、隔离、模型连接和小规模试跑 |
| `autoresearch-s1-research` | 外部编排的双模型独立长程研究，各至少 10h、目标 11h 有效工作 |
| `autoresearch-s1-deliver` | 最终候选复验、第二版 QA、自检和六项材料打包 |

每个 Skill 包含入口说明、UI 元数据、按需参考材料、可执行工具、模板和测试。四阶段通过目标服务器专家侧的 `expert_evidence/阶段交接.md` 传递状态，该文件属于内部衔接材料，不是第七项交付物。

## 使用

将所需 Skill 目录复制到 Codex 的 `skills` 目录，或直接从本仓库安装。按阶段调用：

```text
$autoresearch-s1-prepare
$autoresearch-s1-package
$autoresearch-s1-research
$autoresearch-s1-deliver
```

这是专家侧仓库。不要把整个 Skill、QA、Reference、私有评分资产或另一条轨迹暴露给研究 Agent。正式实验必须在云 GPU Linux 服务器完成；本地测试、模拟 adapter 和模板不构成正式研究证据。

## 验证

对每个 Skill 执行：

```bash
python3 -m unittest discover -s tests -v
```

使用 Codex `skill-creator` 自带的 `quick_validate.py` 校验 Skill 结构。用户第二版 QA 的完整测试应在正式 Linux 环境重跑。
