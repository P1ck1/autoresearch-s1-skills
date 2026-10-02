# 可信适配器合同

工具包提供外部驱动和验证器，不猜测用户服务器的 Harbor CLI、账号、模型ID或推理参数。第二阶段需针对实际部署写一个专家持有的 adapter。适配器不是模型生成、候选目录中的程序，也不是测试夹具。

## 启动方式

`adapter_argv` 必须是参数数组，首项绝对可执行路径；驱动在末尾追加 `--request /protected/request.json --receipt /protected/receipt.json`。不经过 shell，不把秘密塞进 argv。adapter 以可信编排身份运行，负责调用目标 Harbor/Codex、把研究代码放在经过验证的独立权限容器内、恢复正确会话及收集原始证据。其自身执行权限不等于研究候选权限。

## request 与工作模式

- probe：operation、track（codex/seed）、model、effort、mode。
- research_slice：另含 task_id、task_version、attempt_id、next_round、candidate_root、previous_accepted_attempts。
- production 为真实云服务器调用；simulation 只用于测试，不能写正式完成结论。测试 adapter 明确拒绝 production。

模型展示名固定 GPT-5.6 Sol/Max 与 Seed 2.1 Turbo/High。adapter 需要保存实际 provider model ID、档位映射依据、版本、响应/工具/评分原始日志，不能只回显 request 就宣称模型一致。

## receipt

所有身份字段原样对应 request，`no_live_job: true` 表示此次 slice/probe 所有启动作业已完成或已受控停止并可从正确状态恢复；并不要求删除其容器。CLI 提前返回不代表作业结束。拿不到这一证据时必须返回 false 或失败，外部会停下要求核实。

probe 另含 text_response/tool_call/public_score/record_written 布尔值及非空 evidence 数组。每项证据包含相对 attempt 目录的 path 与 sha256；保存实际原始返回和工具行为。驱动验证哈希，不独立证明内容真实性。

research_slice 另含：

- rounds：完整八字段的零或多轮记录，连续接 next_round；跨 slice 的未完成轮不提前捏造分数。
- round_evidence：按 round 字符串索引，每项 path/sha256/metric_path 对应原始 Public/Dev JSON。metric_path 是 JSON 键/数组下标列表；null 对应真实未出分。
- candidate_evidence：完成轮次的方法快照及必需配置文件的 path/sha256。可信侧收集到本次 attempt 目录。
- active/excluded：时间区间列表，每项 start/end 为含时区 ISO8601，reason 非空，evidence 为对应原始记录。所有区间必须落在外部实际观测的 adapter 执行窗口内。

active 区间需要日志证明真实推理、方法实现、训练或评分；不能把 adapter 整体存活时间一概标为有效。excluded 记录排队、安装、故障、阻塞及空转。外部取 active 并集后减 excluded，不重复累计重叠时间。

## 故障与机密

adapter 应处理终止信号，终止/暂停仅属于本次任务的作业，保留候选和证据，再回报状态。外部超时会终止自己的 adapter 进程组，但它无法保证远端容器/作业已经停下；必须检查目标平台实际状态，禁止直接重复启动。不要删除用户数据、其他容器或已有研究。

端点认证由可信端管理，不进入候选可读环境。stdout/stderr 位于专家受保护目录，尽量在源头脱敏；不得把密钥作为正式证据。收到模型建议停止时仍由外部控制器决定继续，不能由 adapter 将高分翻译成正式完成。
