# 联合状态转移预测　JSTP

**岳轲 · 理论、论文和可复验的交互研究。** 本仓库收录 JSTP 理论稿、五项交互研究、冻结权重和逐题执行记录。各项研究保留独立的实验协议，阳性和阴性结果均可查阅。

## 从这里开始 / Start here

| 要读什么 / What to read | English | 中文 |
| --- | --- | --- |
| **完整理论 / Full JSTP theory** — Ke Yue's proposals about prediction, grouping, action, language, and the self. 18 sections, two appendices. | **[Read the full theory](theory/JSTP-Theory.en.md)** · [PDF, 34 pp.](theory/JSTP-Theory-Ke-Yue.en.pdf) | **[中文原稿](theory/JSTP-详细理论稿-2026-11-19.md)** · [PDF，30 页](theory/JSTP-Theory-Ke-Yue.zh.pdf) |
| **实证论文 / Empirical paper** — *Goal Generalization with Sparse Action Labels*. | **[Revised PDF · v1.2](studies/joint_primer/paper/JSTP-Goal-Generalization-Ke-Yue.pdf)** · [Online text](studies/joint_primer/paper/manuscript-en.md) · [Cite v1.2 (DOI)](https://doi.org/10.5281/zenodo.24392467) | **[中文修订稿 PDF · v1.2](studies/joint_primer/paper/联合状态转移预测导论-岳轲.pdf)** · [在线正文](studies/joint_primer/paper/manuscript.md) |

The theory manuscript states the hypotheses; the empirical paper tests one claim about using a predicted consequence for action. Table 2 of [v1.2](https://doi.org/10.5281/zenodo.24392467) compares five external adaptations on the original layouts: three newly added and two re-evaluated. [Fixed comparison archive](https://github.com/ykykykk/jstp-intervention-bench/tree/5514eb4c851ec6670a17d8420184b1fc7a98ac2b/studies/fourtrack_foundations). [Current preprint](https://doi.org/10.5281/zenodo.24366435).

[完整 JSTP 与五种外部方法（结果与区间）](studies/fourtrack_foundations/outputs/fourtrack-20261120/四轨模型对比.md) · [下载这张表（Excel）](studies/fourtrack_foundations/outputs/fourtrack-20261120/完整JSTP与主流模型-四轨对照.xlsx) · [原八组对照表](studies/joint_primer/paper/tables/JSTP-八种模型同场对照.xlsx) · [原考场逐题回放](studies/joint_primer/paper/evidence-room.html)

**想先玩一局？** 本地双击 **[潮汐港 · 四轨货运](玩四轨调度站.cmd)**。先把一艘小艇送进青框，再同时调度四种舱。自由港的 A、B 绞盘都能反复试，也能让冻结模型玩同一关。要玩论文 §5.1 的原题，选「☰ → 论文例题」；只有论文赛道限制试探次数并延迟揭晓结果。[交互与两种计分规则](docs/FOUR-TRACK-GAME.md)

**[直接看六个模型同场操作 / Watch the films](https://ykykykk.github.io/jstp-intervention-bench/)**：[128](https://ykykykk.github.io/jstp-intervention-bench/#128) · [1,024](https://ykykykk.github.io/jstp-intervention-bench/#1024) · [16,384](https://ykykykk.github.io/jstp-intervention-bench/#16384) 条动作示范，每段 32 秒。[实操日志、相机与下载](docs/harbor-films/README.md)。**接口不是纯视觉到鼠标：**模型读相机特征、公开仪表和目标卡，选 A/B，模拟器执行；测试不提供隐藏接线、物体真值坐标或正确动作。Human players use a mouse; the frozen controllers do not.

另一个游戏是 V7 的 **[双舱救援](玩双舱救援.cmd)**：横拖绞盘看电池跟不跟，再竖拖把乘员送进青框。它和导论四轨是两个不同的实验。[双舱玩法](docs/RESCUE-GAME.md)

四轨装置先横拉一个把手试探，四个滑块复位后才公布目标。正式竖拉 A 或 B 会让四轨都动；换把手，每轨方向都相反。一张考卡要求三轨各向上或向下，至少两轨符合即过关。两张相反的考卡从同一复位状态分别执行，两张都过才得一分。[论文 §5.1：一局四张卡怎样得分](studies/joint_primer/paper/联合状态转移预测导论-岳轲.pdf#page=4)。128 条动作标注时，完整 JSTP 为 **71.69%**；保留两个训练损失、只断开后果回输后为 **66.75%**。这 4.94 个百分点衡量的是回输在同一架构里的作用，不是相对其他架构的普遍优势。

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
