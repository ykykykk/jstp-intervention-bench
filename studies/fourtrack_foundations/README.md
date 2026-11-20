# 四轨实操：完整 JSTP 与五种外部方法

[阅读实测对比表](outputs/fourtrack-20261120/四轨模型对比.md) · [下载 Excel（含逐种子分子、分母）](outputs/fourtrack-20261120/完整JSTP与主流模型-四轨对照.xlsx)

这项比较见导论 v1.2 的第 6.2 节、表 2：[中文正文](../joint_primer/paper/manuscript.md) · [English, Section 6.2](../joint_primer/paper/manuscript-en.md#62-external-comparison-act-changes-the-sparse-label-ranking)。论文将外部方法的任务排名与原同结构消融的机制估计分开报告；[源码包](../joint_primer/paper/JSTP-arXiv-source.tar.gz)内有可直接重算表 2 的逐布局计数。

同一段试探经历，给两张方向相反、训练未见的目标卡。两次下达的命令在模拟器里都达到目标，才记一对成功。比较的问题是：**给同样多的动作示范，哪种方法最能把已观察到的试探用于新目标？** 不测自主决定要不要试探，不单独验收分割，也不是通用机器人排行榜。

## 方法

| 模型 | 实际参与比较的部分 |
|---|---|
| 完整 JSTP | 不修改论文原版，九次从头训练。共同状态推演隐式后果，反馈到自身动作路径；直接动作路径也保留。 |
| [ACT](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human) | 官方 ALOHA 预训练，保留视觉、状态、编码器/解码器及全部 100 个动作查询。将试探前后编码成一张三通道观测；取首个动作槽，训练从官方参数初始化的动作投影与四轨读出。 |
| [Diffusion Policy](https://huggingface.co/lerobot/diffusion_pusht) | 官方 Push-T 权重。前后两帧、原生 U-Net、16 位轨迹，零初始噪声作 4 步 DDIM。冻结采样，取首个可执行动作、最后一步隐状态，在原生投影参数基础上训练终端投影和四轨读出，不是原场景的 100 步 DDPM。 |
| [V-JEPA 2.1](https://github.com/facebookresearch/vjepa2) | Meta 蒸馏 ViT-B/16 视频表征。将两个观察时刻各重复两帧，保留两组 2×4 空间区域；训练动作与光学读出。不是 V-JEPA-AC。 |
| [SmolVLA](https://huggingface.co/lerobot/smolvla_base) | 沿用同训练集、同抽样表下的九次原生流动作专家适配。本次重新加载冻结权重，重算全部命令并送入模拟器；旧封卷命令仅作核对。 |
| [DINO-WM](https://github.com/gaoyuezhou/dino_wm) | 沿用九次四轨适配：冻结 DINOv2-S/14，用原作者 ViTPredictor 预测本装置的光学变化，而非原任务的未来视觉特征。本次重新推理并实际执行，与论文中排除的自建一步世界模型不是同一个模型。 |

所有新外部适配层都得到完整的 18 个公开仪表值，不读取隐藏接线、私有状态、分割掩码或正确把手；预训练来自不同任务。ACT、Diffusion Policy 和 V-JEPA 的骨干不更新，训练的是下游接口，而非全部官方权重。冻结前向的特征按完全相同的公开输入去重，再恢复原来每一行的位置；不是把不同经历合成一个训练样本。

## 读表口径

- 训练始终有 **16,384 条物理经历**。128 / 1,024 / 16,384 指嵌套抽取的**不同动作标注**。三个种子 9301、9419、9547，沿用同一份示范、目标、顺序和 768×128 的更新安排。JSTP、ACT、Diffusion Policy、V-JEPA 2.1 和 DINO-WM 都用同批额外局部光学观测；SmolVLA 原训练只用动作目标。比较不是相同总监督量、预训练算力、模型容量或可更新参数量。
- 原论文的 192 个留出布局，每个布局两种接线、两种离合状态、两个指定试探方向，考 8 对相反目标。**每份权重 12,288 对**。三个种子重复同一批题，不是 576 个独立布局。只换目标时，新适配和 JSTP 使用同一个四维线性目标接口，天然倾向于对相反目标换手；换手本身不计分，两个目标都达到才算成功。
- 所有新训练保留最后一次更新。不看考场选模型或挑种子。先冻结全部权重，再读取公开考试经历并封存每条动作，最后用原装置的相机核对结果。原论文布局的历史成绩已知，不能称为新盲考。95% 区间按整布局配对重采样；128 档五项差值用 Bonferroni 校正。区间不覆盖重新选择预训练模型或调参的不确定性。
- [实验方案](plan.json)；[逐样本执行凭据](evidence/after.json)；[汇总](evidence/summary.json)；[逐种子](evidence/per-seed.json)。方案修订于新方法考试前：按作者要求取消大权重候选，保留已经训练的原版 JSTP；其余方法不按成绩增删。

## 复现

从仓库根目录运行。原版训练用项目 PyTorch；外部适配用本机实验环境（LeRobot 0.7.0、diffusers 0.39.0、timm 1.0.31）。大权重、中间特征放在忽略的 `work/` 中。ACT、Diffusion Policy 各自的官方配置和 `model.safetensors` 放在 `work/upstream-act`、`work/upstream-diffusion`；Meta 的官方 .pt 和对应提交源码分别放在 `work/upstream-vjepa21`、`work/upstream-vjepa21-code`。版本、来源与 SHA 锁定见方案和特征记录。原版数据和已有两项训练见 `studies/joint_primer`。

```powershell
python -m studies.fourtrack_foundations.common
foreach ($a in 'act','diffusion','vjepa21') {
    python -m studies.fourtrack_foundations.features $a train
    python -m studies.fourtrack_foundations.train $a
}
foreach ($a in 'act','diffusion','vjepa21') { python -m studies.fourtrack_foundations.features $a exam }
python -m studies.fourtrack_foundations.exam all
```

结果目录按只写一次设计；复现请使用干净的工作副本，不改写原论文证据。
