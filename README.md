# 联合状态转移预测　JSTP

**岳轲 · 理论、论文和可复验的交互研究。** 这是原来的私有研究库。各次工作在同一库中分开保存，阴性结果和旧权重均保留。

**最新研究：** [《不必预演整幅未来》PDF](studies/partial_futures/paper/JSTP-局部未来与同数据比较-岳轲.pdf)　·　[同数据比较、源码与复现](studies/partial_futures/README.md)　·　[可编辑论文](studies/partial_futures/paper/manuscript.md)　·　[训练曲线与冻结考场](studies/partial_futures/paper/evidence-room.html)

新模型只需预测这次涉及的变化，可以直接预测自己的动作，也可以让局部物理后果影响动作。相同训练预算下，在新布局上它和直接动作、一步世界模型都能把相反目标两题全做对；真正的区别要通过**读出什么、只改预期后动作会不会变**来测。官方 SmolVLA 微调、DINOv2 / DINO-WM 适配和下游经历逐条匹配的比较，另表报告，不以小分数推断全领域强弱。

| 找什么 | 入口 |
| --- | --- |
| 理论原意与适用边界 | [详细理论稿（2026-11-19）](theory/JSTP-详细理论稿-2026-11-19.md) · [版本索引](theory/README.md) |
| 第一项：同一画面，不同后果 | [功能性分组与新接线](studies/01-grouping/README.md) |
| 第二项：会预见就会据此行动吗？ | [自身运动与符号回输](studies/02-motor-symbol/README.md) |
| 第三项：试探之后，才给目标 | [晚到目标与对后果的隐式干预](studies/03-late-goals/README.md) |
| 第四项：同一套数据，不同预测架构 | [局部未来、世界模型、VLA 与预算桥接](studies/partial_futures/README.md) |

[全部研究目录](studies/README.md) · [引用信息](CITATION.cff) · [第三方来源](THIRD_PARTY.md) · [许可](LICENSE)。历史 `jstp/`、`mechanism/`、`lategoal/`、`paper/`、`evidence/` 的位置保持兼容。最新研究的协议、数据、代码、论文和证据集中在 `studies/partial_futures/`，不需新建仓库。
