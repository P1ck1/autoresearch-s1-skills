# 六项交付与归档

最终六项不变：evidence.zip、harbor_task.zip、小规模试跑结论、Codex+GPT-5.6 Sol Max轨迹、Codex+Seed 2.1 Turbo High轨迹、自检checklist。

delivery-map.json 是专家侧显式映射，不是平台固定JSON。archives 恰为 evidence/harbor_task，每项 source 相对 --root，target 是ZIP内相对路径；files 恰为 pilot/codex/seed/checklist。qa_report 提供原始最终审查报告 path/sha256。模板故意留空，未填不能假装成功。

脚本要求 harbor_task.zip 的 task.toml/instruction.md 位于根，这是本工具采用的直接导入布局选择，不是从未给出的平台规则猜测的唯一格式。若目标平台要求外包装，应依目标要求适配脚本和测试并披露，不强行绕过平台。Reference、实验模型和专家材料的归属按实际文件映射留存，严禁加入Agent可见镜像。

脚本逐文件写ZIP并核对解读后哈希，拒绝逃逸/符号链接/重名、明显密钥路径和覆盖源目录。它不证明提交内容完整、不做通用密钥检测、不判QA通过；状态始终 PACKAGED_NOT_PLATFORM_ACCEPTED。除两ZIP和四独立文件外，delivery_manifest.json是内部核验附件，不是新增第七项交付。

打包前必须人工核查所有平台必需材料、最终QA结论与复核完成状态；打包后复核拆包引用是否可追溯、解压布局是否可供平台运行。对外发送或上传遵守已有授权，不自动发送给他人。
