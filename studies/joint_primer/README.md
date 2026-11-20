# 联合状态转移预测导论：从局部后果到少示范动作泛化

**岳轲 · 预印本 v1.2 · 2026 年 11 月 20 日** · [Zenodo DOI：10.5281/zenodo.24392467](https://doi.org/10.5281/zenodo.24392467)

[中文论文 PDF](paper/联合状态转移预测导论-岳轲.pdf) · [中文正文](paper/manuscript.md) · [英文摘要](paper/abstract-en.md) · [表 2 外部对照与逐种子 Excel](../fourtrack_foundations/outputs/fourtrack-20261120/完整JSTP与主流模型-四轨对照.xlsx) · [原八组 Excel](paper/tables/JSTP-八种模型同场对照.xlsx) · [原考场逐题回放](paper/evidence-room.html)

**Full theoretical manuscript:** [Read in English](../../theory/JSTP-Theory.en.md) · [中文原稿](../../theory/JSTP-详细理论稿-2026-11-19.md) · [Bilingual PDF downloads](../../theory/README.md). This companion states the full hypothesis; the empirical paper below tests a specific prediction about action supervision.

**English empirical paper.** Ke Yue (2026), *Joint State Transition Prediction: Goal Generalization with Sparse Action Labels*. [Revised English PDF · v1.2](paper/JSTP-Goal-Generalization-Ke-Yue.pdf) · [Full text](paper/manuscript-en.md) · [LaTeX source, figure and per-layout data](paper/JSTP-arXiv-source.tar.gz). Section 5.1 explains the probe, reset, late goal, majority rule and paired score using archived layout 113670000, including a pair in which the model switched handles and lost both cards. Appendix A links the original study and the [fixed archive of the extended comparison](https://github.com/ykykykk/jstp-intervention-bench/tree/5514eb4c851ec6670a17d8420184b1fc7a98ac2b/studies/fourtrack_foundations) separately. To regenerate the English outputs from stored results, run `python studies/joint_primer/paper/build_en.py`.

本文以同一份物理经历检验 JSTP 的一个主张：动作本身可以直接预测，但在动作标注稀缺时，局部后果是否有助于网络为未见目标选择动作？论文将联合状态、功能性分组与语言层的理论命题，与本装置上可验证的结果分开。见[完整理论稿](../../theory/JSTP-详细理论稿-2026-11-19.md)。

## 修订稿新增：完整 JSTP 与五种外部方法（表 2）

直接翻到四轨计分实例：[中文，第 4 页](paper/联合状态转移预测导论-岳轲.pdf#page=4) · [English, p. 4](paper/JSTP-Goal-Generalization-Ke-Yue.pdf#page=4)。模型对照见表 2：[中文，第 6 页](paper/联合状态转移预测导论-岳轲.pdf#page=6) · [English, p. 7](paper/JSTP-Goal-Generalization-Ke-Yue.pdf#page=7)。

所有模型都考原来的 192 个留出布局。128 条动作标注下，ACT 在三个种子上都超过 JSTP：**73.78% 对 71.69%**，差 2.08 个百分点（校正后 99% 区间 [0.93, 3.28]）。1,024 条时 JSTP 为 **94.90%**、ACT 为 **94.58%**，差值区间跨零。全量标注时两者分别为 **99.67%** 和 **99.96%**。这限定了「少示范领先」的范围，不推翻与同结构断回输的机制比较。

[完整成绩、区间与适配细节](../fourtrack_foundations/outputs/fourtrack-20261120/四轨模型对比.md) · [方案与复现说明](../fourtrack_foundations/README.md) · [冻结命令](../fourtrack_foundations/evidence/before.json) · [实操记录](../fourtrack_foundations/evidence/after.json)。新增的 ACT、Diffusion Policy、V-JEPA 2.1 与原版 JSTP 共重新训练 36 份权重；SmolVLA 和 DINO-WM 的 18 份旧权重重新推理并执行。此表不含项目内部消融，也未测外部方法的对象分割。

## 原主考场：八种模型，三档示范（表 1）

16,384 条物理经历固定，目标—成功动作标注预算分别为 128、1,024、16,384。三个训练种子，每个模型冻结后在同一批 192 个训练未见布局中实际操作。一次试探之后分别给出两个相反的未见目标，两次执行都成功才计分。

| 成对成功率（%） | 128 条 | 1,024 条 | 16,384 条 |
| --- | ---: | ---: | ---: |
| JSTP 共享转移 | **71.69** | 94.90 | 99.67 |
| 一步世界模型 | 67.45 | 74.10 | 60.28 |
| 同结构，断开后果回输 | 66.75 | **96.46** | **99.84** |
| 掩码轨迹 Transformer | 54.01 | 74.54 | 84.27 |
| DINO-WM 任务适配 | 50.58 | 50.39 | 50.13 |
| 仅直接动作 | 49.79 | 62.74 | 90.17 |
| 条件流动作 | 39.04 | 60.90 | 78.14 |
| 官方 SmolVLA 局部微调 | 23.74 | 24.59 | 26.00 |

修订稿的表 1 保留原八组的差值与校正区间；表 3 保留其逐种子成绩、单题成功、换手与共动读出。原八组 Excel 使用修订前的编号，与历史档案一致。

- **最接近的机制对照。** 稀缺档中，JSTP 比仅动作高 21.90 个百分点，比保留同样两种训练损失、只断开后果回输的模型高 4.94 点（99% 布局区间 [3.97, 5.85]）。示范更充分时，后者反而略高。
- **两个官方适配。** 18 份适配权重从上游基座重新训练，封存动作后共实际执行 442,368 次。稀缺档相对 SmolVLA 和 DINO-WM 适配的差值为 47.95 点 [46.18, 49.69] 和 21.12 点 [18.79, 23.41]（各为 97.5% 布局区间）。SmolVLA 不使用光学损失；DINO 使用官方视觉编码器与作者预测器，在本任务学习位移。不能据此排名模型家族。
- **表征与行为。** 稀缺档严格共动读出：JSTP 72.35%，断回输 91.02%。固定真实经历和目标，仅替换与光学后果对齐的隐分量，JSTP 在应换手题上原本正确且定向换手的比例为 11.75%，等范数无关扰动为 5.79%。这既不是实例分割成绩，也不是通关率的提高。
- **另一批布局。** 表 4 单列另 192 个布局与隔离梯度的世界模型；未与主考场混算。五种外部适配都没有参加这一批次。

## 协议、权重与相机记录

| 阶段 | 入口 |
| --- | --- |
| 事先约定的对照、标注、指标与更新预算 | [主协议](protocol.json) · [原字节数据](../partial_futures/data/) · [实际训练排程](evidence/train/schedule.npz) |
| 架构、训练与最后权重 | [模型](models.py) · [训练](train.py) · [54 份权重、曲线和指纹](evidence/train/models/) |
| 先封存命令、再执行 | [考试程序](exam.py) · [封卷索引](evidence/before.json) · [相机执行索引](evidence/after.json) |
| 统计与不确定性 | [逐方法汇总与配对差](evidence/analysis/summary.json) · [逐种子](evidence/analysis/per-seed.json) · [可重算的逐布局计数](evidence/analysis/layout-counts.npz) |
| 主结果后追加的对手 | [补考协议](challenge-plan.json) · [九份追加权重](evidence/challenge/models/) · [七模型同场补考与逐种子成绩](evidence/challenge/results.json) |
| 官方适配回到主考场 | [训练前锁定的方案](official-plan.json) · [18 份新适配权重与曲线](evidence/official/models/) · [封卷](evidence/official/before.json) · [实操](evidence/official/after.json) · [逐种子](evidence/official/analysis/per-seed.json) · [配对比较](evidence/official/analysis/summary.json) |
| 五种外部适配（v1.2） | [逐种子](../fourtrack_foundations/evidence/per-seed.json) · [成对成功数、区间与差值](../fourtrack_foundations/evidence/summary.json) · [三种新适配和重算流程](../fourtrack_foundations/README.md)；论文源码包的 anc/ 另附逐布局计数 |

主协议与训练代码于本地 Git 提交 `7081973` 锁定，考试与统计于 `6965e1c` 锁定。看到主结果后才提出隔离梯度的比较，`9dace38` 在补考训练前锁定。没有以测试分数调参或挑选最好权重；已有前四项研究的结果在设计时是已知的。文件和哈希用于复核先后关系，不代表公开预注册。

## 本地复现

从仓库根目录依次执行；新结果写到尚不存在的 `work/primer-reproduction`，不会覆盖正式证据。下面六组从零模型和隔离梯度对照不需要外部权重；官方适配轨道另列。每次训练使用一台 RTX 3090，不假定不同方法的 FLOPs 相等。独立虚拟机构不会操作系统鼠标。

```powershell
$py = ".\.venv\Scripts\python.exe"
$out = "work/primer-reproduction"
& $py -m studies.joint_primer.train --out "$out/train" --device cuda
& $py -m studies.joint_primer.exam --out $out
& $py -m studies.joint_primer.analyze --out $out
& $py -m studies.joint_primer.challenge --out $out --device cuda
```

训练与考试环境沿用[仓库锁定依赖](../../requirements-windows-cu132.txt)。论文排版依赖见[清单](../../paper/requirements.txt)；发布时的字体子集与许可随文保存。从根目录运行 `python studies/joint_primer/paper/build.py` 可以用现存的数值和插图重排导论，不会重新训练。训练曲线与逐题可视化由 `python -m studies.joint_primer.paper.exhibit` 从正式记录生成，图像按存档指纹核对。

**只复现官方适配。** 使用研究四[锁定的 LeRobot / Transformers 环境](../partial_futures/requirements-transfer.txt)和原主试验封存的训练排程，不需要先重训 JSTP。`prepare_models` 从固定的官方版本下载基座和处理器；本机已有相同指纹的缓存就不用再下载。每一档、每一种子均从上游基座另起，不载入旧的下游权重。

```powershell
$native = ".\.venv\Scripts\python.exe"
$official = ".\work\vla-env\Scripts\python.exe"
$out = "work/primer-official-reproduction"
& $official -m studies.partial_futures.prepare_models  # 首次获取官方基座
& $official -m studies.joint_primer.official_train --out $out
& $native -m studies.joint_primer.official_exam --out $out --stage capture
& $official -m studies.joint_primer.official_exam --out $out --stage seal
& $native -m studies.joint_primer.official_exam --out $out --stage execute
& $native -m studies.joint_primer.official_exam --out $out --stage analyze
```

这里只把上游基座用于训练与模型推理；试探输入、实物推进、RGB 判分和配对统计都使用原主考场环境。原布局的初始、试探和最终相机指纹逐项核对，术语「实际执行」指独立模拟机构真实推进，并非现实机器人。
