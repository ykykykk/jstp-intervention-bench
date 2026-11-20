# 联合状态转移预测　JSTP

**岳轲 · 理论、论文和可复验的交互研究。** 本仓库收录 JSTP 理论稿、五项交互研究、冻结权重和逐题执行记录。各项研究保留独立的实验协议，阳性和阴性结果均可查阅。

## 从这里开始 / Start here

| 要读什么 / What to read | English | 中文 |
| --- | --- | --- |
| **完整理论 / Full JSTP theory** — Ke Yue's proposals about prediction, grouping, action, language, and the self. 18 sections, two appendices. | **[Read the full theory](theory/JSTP-Theory.en.md)** · [PDF, 34 pp.](theory/JSTP-Theory-Ke-Yue.en.pdf) | **[中文原稿](theory/JSTP-详细理论稿-2026-11-19.md)** · [PDF，30 页](theory/JSTP-Theory-Ke-Yue.zh.pdf) |
| **实证论文 / Empirical paper** — *Goal Generalization with Sparse Action Labels*. | **[Revised PDF · v1.2](studies/joint_primer/paper/JSTP-Goal-Generalization-Ke-Yue.pdf)** · [Online text](studies/joint_primer/paper/manuscript-en.md) · [Cite v1.2 (DOI)](https://doi.org/10.5281/zenodo.24392467) | **[中文修订稿 PDF · v1.2](studies/joint_primer/paper/联合状态转移预测导论-岳轲.pdf)** · [在线正文](studies/joint_primer/paper/manuscript.md) |

The theory manuscript states the hypotheses; the empirical paper tests one claim about using a predicted consequence for action. Table 2 of [v1.2](https://doi.org/10.5281/zenodo.24392467) compares five external adaptations on the original layouts: three newly added and two re-evaluated. [Fixed comparison archive](https://github.com/ykykykk/jstp-intervention-bench/tree/5514eb4c851ec6670a17d8420184b1fc7a98ac2b/studies/fourtrack_foundations) · [Previous preprint (v1.1)](https://doi.org/10.5281/zenodo.24366742). [Zenodo version history](https://doi.org/10.5281/zenodo.24366435).

[完整 JSTP 与五种外部方法（结果与区间）](studies/fourtrack_foundations/outputs/fourtrack-20261120/四轨模型对比.md) · [下载这张表（Excel）](studies/fourtrack_foundations/outputs/fourtrack-20261120/完整JSTP与主流模型-四轨对照.xlsx) · [原八组对照表](studies/joint_primer/paper/tables/JSTP-八种模型同场对照.xlsx) · [原考场逐题回放](studies/joint_primer/paper/evidence-room.html)

**想先玩一局？** 在本地双击 [玩双舱救援.cmd](玩双舱救援.cmd)。按住甲板上的一根绞盘横拖，看电池跟不跟；再握住选好的绞盘，沿竖轨亲手把乘员送进青框，到线松手。也可以切到 JSTP 代驾，看它怎样选。[玩法与模型说明](docs/RESCUE-GAME.md)。这是 V7 双舱现场游戏，不是导论的四轨考场或论文录像。

四轨装置先试探、复位，再公布目标。两张相反的新目标卡给的是同一段经历，两次实际操作都达标才得分。128 条动作标注时，完整 JSTP 为 **71.69%**；保留两个训练损失、只断开后果回输后为 **66.75%**。这 4.94 个百分点衡量的是回输在同一架构里的作用，不是相对其他架构的普遍优势。

**加入 ACT 后，稀缺档的第一名变了。** 同样的 128 条标注下，ACT 为 **73.78%**，三个种子都高于 JSTP；其余四个外部适配低于 JSTP。1,024 条时，从头训练的 JSTP（18,304 个参数）为 **94.90%**，ACT 为 **94.58%**，配对区间跨零。表 2 报告三档完整结果；它只比较四轨任务，没有给这些外部方法测对象分割。[方法、冻结权重与执行记录](studies/fourtrack_foundations/README.md)

| 找什么 | 入口 |
| --- | --- |
| 理论原意与适用边界 | [详细理论稿（2026-11-19）](theory/JSTP-详细理论稿-2026-11-19.md) · [版本索引](theory/README.md) |
| 第一项：同一画面，不同后果 | [功能性分组与新接线](studies/01-grouping/README.md) |
| 第二项：会预见就会据此行动吗？ | [自身运动与符号回输](studies/02-motor-symbol/README.md) |
| 第三项：试探之后，才给目标 | [晚到目标与对后果的隐式干预](studies/03-late-goals/README.md) |
| 第四项：同一套数据，不同预测架构 | [局部未来、世界模型、VLA 与预算桥接](studies/partial_futures/README.md) |
| 导论与第五项：同一批经历，需要多少动作示范？ | [原机制对照、五种外部适配及独立布局补考](studies/joint_primer/README.md) |

[全部研究目录](studies/README.md) · [导论引用](studies/joint_primer/CITATION.cff) · [研究软件引用](CITATION.cff) · [第三方来源](THIRD_PARTY.md) · [许可](LICENSE)。导论与第五项证据在 `studies/joint_primer/`，沿用第四项训练数据。各研究在同一库中保留各自的历史路径与证据。

**使用与引用。** 原创代码和生成的实验记录按 [MIT 许可](LICENSE) 发布；理论稿、论文、摘要和 PDF 保留作者版权，不随代码按 MIT 授权。第三方组件沿用各自的许可。
