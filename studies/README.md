# 研究目录

第一次接触 JSTP，建议先读[理论导论（PDF）](joint_primer/paper/联合状态转移预测导论-岳轲.pdf)，再按要查的问题进入单项证据。

| 研究 | 问题 | 论文、权重和执行记录 |
| --- | --- | --- |
| [01 · 功能性分组](01-grouping/README.md) | 同一末帧，哪些局部会被把手一起带动？ | [第一篇论文](../paper/manuscript.md) · [证据](../evidence/paper/) |
| [02 · 自身运动与符号回输](02-motor-symbol/README.md) | 能预测后果，是否就会因后果不同而换手？ | [第二篇论文](../paper/mechanism_study/manuscript.md) · [证据](../evidence/mechanism/) |
| [03 · 晚到目标](03-late-goals/README.md) | 保留直接动作通路，只改物理预期会怎样？ | [方法与复现](../docs/LATE-GOALS.md) · [证据](../evidence/late-goals/) |
| [04 · 局部未来的同数据比较](partial_futures/README.md) | 当几种方法都能通关，怎样分清它们利用了什么？ | [新论文](partial_futures/paper/manuscript.md) · [逐题记录](partial_futures/evidence/) |
| [05 · 导论与示范稀缺](joint_primer/README.md) | 一套仍可直接行动的网络，能否用已见过的物理后果节省正确动作示范？ | [导论正文](joint_primer/paper/manuscript.md) · [六组主试验](joint_primer/evidence/analysis/summary.json) · [七组补考](joint_primer/evidence/challenge/results.json) |

01–03 保留原代码包、数据和论文的位置，以免破坏老实验的导入、校验与引用。新研究在自己的目录里放协议、代码、数据、证据和论文。原来的两个根目录历史入口仍可跳转。
