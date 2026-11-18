# 第三方来源与授权边界

| 组件 | 固定来源 | 在发布包中 |
|---|---|---|
| DINOv2 ViT-S/14 编码器 | [facebookresearch/dinov2，7764ea0](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8)，[Apache-2.0](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE) | 不含其源码与预训练权重。完整重训脚本从上游下载。 |
| DINOv2 官方预训练权重 | [Meta 原始文件](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth)；SHA-256：`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` | 不再分发。发布的 DINO 适配检查点仅含本实验训练的动力学参数。 |
| DINO-WM ViTPredictor | [gaoyuezhou/dino_wm，0a9492f](https://github.com/gaoyuezhou/dino_wm/tree/0a9492fa12044b852ae9e001cc74604b79c8bb0c)，[MIT](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/LICENSE) | 运行时导入上游固定提交，未修改其预测器；本仓库只发布自己的适配代码。 |
| 排版字体 | [Google Fonts / Noto Serif SC，8b0a1d0](https://github.com/google/fonts/tree/8b0a1d0f5983c89bc2b93f1b5fb55f9e252744b5/ofl/notoserifsc) | 两个静态字重子集，改名 JSTP Study Serif；版权与再分发条款见 [SIL Open Font License 1.1](paper/fonts/OFL.txt)。可由 `scripts/prepare_fonts.py` 重建。 |

模拟器、主实验源码、早期模型推理代码和实验记录来自本 JSTP 项目；所有主实验画面由代码生成，不需要桌面截图、遥测或用户文件。文献的作品、商标和署名仍属各自作者；没有上游作者为本次适配评测背书的含义。
