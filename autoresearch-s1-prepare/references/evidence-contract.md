# 成对证据输入

`assets/paired-input.json` 是专家侧数值复算输入格式，不替换平台 result.json schema。formal_seeds 为已声明全集，paired_runs 每 seed 恰一对；direction 是 maximize/minimize；evaluation_mode 是 stochastic/deterministic。same_protocol/quality_valid 是待语义核实的声明，不能靠布尔值证明公平。

每对 baseline/reference 是原始指标，并分别提供 baseline_evidence/reference_evidence：path 相对 --root，sha256 是实际文件哈希，metric_path 是原始JSON内键/数组下标列表。脚本读取该文件、核哈希与数值，拒绝逃逸路径、重复seed、布尔/NaN等伪数值和未配齐集合。

upper_bound_basis 说明 U 的既定来源；deterministic 还需 determinism_basis。可选 absolute_min_improvement/absolute_threshold_basis 只适用于事前声明的有效容差，不能看结果再设阈值。

脚本输出 NUMERIC_CHECKS_PASS 只代表原始引用和算术一致；代表性、训练充分性、独立性、数据/预算公平、原始记录真实性与 U 合理性仍需人工语义核查。σ=0 仍需正改善，且不自动证明确定性；剩余 headroom 不足3σ只标复核信号，不擅增一刀切门槛。
