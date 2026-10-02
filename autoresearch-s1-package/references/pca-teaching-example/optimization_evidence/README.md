# 待补正式优化证据

这是非训练型 PCA 任务，不产生模型 checkpoint，model/ 不适用。
仍需在固定数据、资源、相同 seed/重复规则下独立运行 Baseline 与 Reference，证明原始指标和时延改善超过噪声。
本轮公开 smoke 只保存在 expert_evidence/validation/，没有混入 baseline_runs 或 reference_runs。
源包没有可迁移的逐 seed 结果和日志，因此不创建伪造的 seed_*/result.json，也不从 README 分数倒推统计量。
