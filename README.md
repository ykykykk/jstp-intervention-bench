# 联合状态转移预测　JSTP

**岳轲 · 理论、论文和可复验的交互研究。** 本仓库收录 JSTP 理论稿、五项交互研究、冻结权重和逐题执行记录。各项研究保留独立的实验协议，阳性和阴性结果均可查阅。

**第一次读 JSTP，从这里开始：** [《联合状态转移预测导论：从局部后果到少示范动作泛化》PDF](studies/joint_primer/paper/联合状态转移预测导论-岳轲.pdf) · [可编辑正文](studies/joint_primer/paper/manuscript.md) · [八种模型同场对照表（Excel）](studies/joint_primer/paper/tables/JSTP-八种模型同场对照.xlsx) · [逐题回放](studies/joint_primer/paper/evidence-room.html)

四轨装置先接受一次试探并复位，之后才公布目标。导论借此检验：固定物理经历、减少正确动作标注时，后果预期能否帮助同一网络选对动作？128 条示范时，JSTP、仅动作和一步世界模型的成对成功率为 **71.69%、49.79%、67.45%**。断开后果回输的匹配模型为 **66.75%**，在更大的两个示范预算上则略高。论文分开报告任务、功能性分组和隐状态干预。

两套官方适配模型在相同下游示范上重训，并在同一批 192 个布局上执行。128 条时，SmolVLA 局部微调为 **23.74%**，DINO-WM 任务适配为 **50.58%**。三档结果及配对区间见[论文表 1](studies/joint_primer/paper/联合状态转移预测导论-岳轲.pdf#page=4)；逐种子成绩与行为诊断见[论文表 2](studies/joint_primer/paper/联合状态转移预测导论-岳轲.pdf#page=5)。

| 找什么 | 入口 |
| --- | --- |
| 理论原意与适用边界 | [详细理论稿（2026-11-19）](theory/JSTP-详细理论稿-2026-11-19.md) · [版本索引](theory/README.md) |
| 第一项：同一画面，不同后果 | [功能性分组与新接线](studies/01-grouping/README.md) |
| 第二项：会预见就会据此行动吗？ | [自身运动与符号回输](studies/02-motor-symbol/README.md) |
| 第三项：试探之后，才给目标 | [晚到目标与对后果的隐式干预](studies/03-late-goals/README.md) |
| 第四项：同一套数据，不同预测架构 | [局部未来、世界模型、VLA 与预算桥接](studies/partial_futures/README.md) |
| 导论与第五项：同一批经历，需要多少动作示范？ | [八种模型同场比较及世界模型补充实验](studies/joint_primer/README.md) |

[全部研究目录](studies/README.md) · [导论引用](studies/joint_primer/CITATION.cff) · [研究软件引用](CITATION.cff) · [第三方来源](THIRD_PARTY.md) · [许可](LICENSE)。导论与第五项证据在 `studies/joint_primer/`，沿用第四项训练数据。各研究在同一库中保留各自的历史路径与证据。

**使用与引用。** 原创代码和生成的实验记录按 [MIT 许可](LICENSE) 发布；理论稿、论文、摘要和 PDF 保留作者版权，不随代码按 MIT 授权。第三方组件沿用各自的许可。
