# PCA 专家侧参考解

本目录只交专家/平台，不进入 Agent 镜像。

- fast_pca.py：原作者的快速参考实现，未改算法。
- fast_submission/pca：其独立可提交版本。专家使用同一公开 Dev 入口、将 --submission 指向本目录中的 fast_submission 即可测试；不可作为 Agent 初始候选。
- full_scan_pca.py、pca_core.py：全扫描正确性/速度对照与数学模块。教学 Starter 从这套全扫描实现适配成独立程序；主快速参考解仍不公开。
- tests/grader_pkg/reference/：为了保留原生评分语义，Verifier 内保留源包的私有镜像副本，供计算真值和速度对照使用。候选在受限执行环境中不能读取它。

这不是 Agent 迭代生成的 best_method，也不是本次实跑的提升证据。原生评分会用快速参考实现的运行时间作锚点，与教程通用“Reference 不作固定满分锚点”要求存在差异，详见包根 README。
