# 第三方来源与授权边界

| 组件 | 固定来源 | 在发布包中 |
|---|---|---|
| DINOv2 ViT-S/14 编码器 | [facebookresearch/dinov2，7764ea0](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8)，[Apache-2.0](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE) | 不含其源码与预训练权重。完整重训脚本从上游下载。 |
| DINOv2 官方预训练权重 | [Meta 原始文件](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth)；SHA-256：`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` | 不再分发。发布的 DINO 适配检查点仅含本实验训练的动力学参数。 |
| DINO-WM ViTPredictor | [gaoyuezhou/dino_wm，0a9492f](https://github.com/gaoyuezhou/dino_wm/tree/0a9492fa12044b852ae9e001cc74604b79c8bb0c)，[MIT](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/LICENSE) | 运行时导入上游固定提交，未修改其预测器；本仓库只发布自己的适配代码。 |
| 排版字体 | [Google Fonts / Noto Serif SC，8b0a1d0](https://github.com/google/fonts/tree/8b0a1d0f5983c89bc2b93f1b5fb55f9e252744b5/ofl/notoserifsc) | 两项研究各附两个静态字重子集，改名 JSTP Study Serif；[研究二的字体许可](paper/mechanism_study/fonts/OFL.txt)，原许可证见 [SIL Open Font License 1.1](paper/fonts/OFL.txt)。可由 `scripts/prepare_fonts.py` 重建。 |

模拟器、主实验源码、早期模型推理代码和实验记录来自本 JSTP 项目；所有主实验画面由代码生成，不需要桌面截图、遥测或用户文件。文献的作品、商标和署名仍属各自作者；没有上游作者为本次适配评测背书的含义。

## 第四项研究：真实权重适配

- [SmolVLA 官方基座，5e8d12a](https://huggingface.co/lerobot/smolvla_base/tree/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6) 与 [LeRobot 0.7.0](https://github.com/huggingface/lerobot/tree/v0.7.0) 采用 Apache-2.0；[附许可证](studies/partial_futures/upstream/Apache-2.0.txt)。本库只分发本次微调后改变的层，不分发整个上游策略或处理器。SmolVLM2 处理器、基座及文件指纹固定在[迁移方案](studies/partial_futures/transfer-plan.json)。
- [DINOv2-S/14 官方转换权重，ed25f3a](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056) 不随库分发。不同于第一项研究的按需获取，本次将未修改的 [DINO-WM 作者预测器](studies/partial_futures/upstream/dino_wm/) 及 MIT 授权一起归档。
- 新论文另附 [OFL 中文字体子集及许可证](studies/partial_futures/paper/fonts/)。

## 理论导论与第五项研究

最初六组和隔离梯度的补考均从零训练。掩码轨迹对照是本地实现，相关方法与原论文见[导论文献](studies/joint_primer/paper/references.json)，不分发上游 MTM 代码或权重。导论另附 [Noto Serif SC 的 OFL 字体子集与许可证](studies/joint_primer/paper/fonts/)。

导论第 5.2 节说明[官方适配的三档训练](studies/joint_primer/official-plan.json)，表 1 和表 2 与六种从零模型同表报告。沿用上一节列出的固定 SmolVLA、处理器、DINOv2 版本和 DINO-WM 作者预测器源码；仓库只保存本次训练更新的参数，不再分发官方基座或处理器。DINO-WM 是针对该装置重训的适配，不是作者在原基准的完整策略检查点。
