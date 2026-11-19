# 联合状态转移预测　JSTP

**岳轲 · 理论、论文和可复验的交互研究。** 这是原来的私有研究库。各次工作在同一库中分开保存，阴性结果和旧权重均保留。

**第一次读 JSTP，从这里开始：** [《联合状态转移预测导论：从共同变化到对象、行动与语言》PDF](studies/joint_primer/paper/联合状态转移预测导论-岳轲.pdf)　·　[可编辑正文](studies/joint_primer/paper/manuscript.md)　·　[同数据训练与冻结证据](studies/joint_primer/README.md)　·　[可切换目标的逐题记录](studies/joint_primer/paper/evidence-room.html)

这篇导论从「杯子下落时，人可以只预测自己伸手」讲起，再用一台四轨装置检验一个具体命题：物理经历相同，正确动作示范很少时，后果能不能帮助行动？128 条示范下，共享转移网络、仅动作和一步世界模型的成对成功率分别为 **71.69%、49.79%、67.45%**。示范充足时，同样学过物理但断开后果回输的网络略胜。论文分开报告任务、功能性分组和隐状态干预，不把这些结果推成语言或意识结论。

| 找什么 | 入口 |
| --- | --- |
| 理论原意与适用边界 | [详细理论稿（2026-11-19）](theory/JSTP-详细理论稿-2026-11-19.md) · [版本索引](theory/README.md) |
| 第一项：同一画面，不同后果 | [功能性分组与新接线](studies/01-grouping/README.md) |
| 第二项：会预见就会据此行动吗？ | [自身运动与符号回输](studies/02-motor-symbol/README.md) |
| 第三项：试探之后，才给目标 | [晚到目标与对后果的隐式干预](studies/03-late-goals/README.md) |
| 第四项：同一套数据，不同预测架构 | [局部未来、世界模型、VLA 与预算桥接](studies/partial_futures/README.md) |
| 导论与第五项：同一批经历，需要多少动作示范？ | [导论、六组主试验与另一个考场的世界模型对照](studies/joint_primer/README.md) |

[全部研究目录](studies/README.md) · [导论引用](studies/joint_primer/CITATION.cff) · [研究软件引用](CITATION.cff) · [第三方来源](THIRD_PARTY.md) · [许可](LICENSE)。导论与第五项证据在 `studies/joint_primer/`，沿用第四项训练数据。各研究在同一库中保留各自的历史路径与证据。
