# 04　不必预演整幅未来

[中文论文 PDF](paper/JSTP-局部未来与同数据比较-岳轲.pdf) · [可编辑论文](paper/manuscript.md) · [训练曲线与冻结考场（下载后本地打开）](paper/evidence-room.html) · [研究目录](../README.md)

**主要结论：满分不说明靠什么行动。** 在 192 个新布局上，JSTP-P、直接动作、一步世界模型相反目标成对成功率都是 100%。相同形式的校准读出分别有 100%、38.11%、100% 的严格四轨共动正确率。JSTP-P 在应换手题上，只移植物理相关隐状态后 17.75% 定向换手，等幅光学零空间对照 0.92%，不该换手的题错换 0.00%。这是局部运动关系可读、能影响部分动作的证据，不是证明发现了通用物体边界，也不是跑赢全部对手。

## 两张不同的问题清单

| 全预算，从零，三个种子 | 新布局成对成功 | 严格分组读出 |
| --- | ---: | ---: |
| JSTP-P 局部续写 | 100.00% | 100.00% |
| 光学监督保留、断回输 | 98.68% | 100.00% |
| 仅直接动作 | 100.00% | 38.11% |
| 两候选一步世界模型 | 100.00% | 100.00% |
| 动作词元 Transformer | 44.08% | 38.28% |
| 条件流动作 | 95.33% | 59.81% |

每种方法、每个种子 4,096 × 160 次采样；沿用相同批次。[旧、新面板和布局区间](evidence/analysis/metrics.csv) · [逐种子](evidence/analysis/per_seed.json)。小型词元和条件流分别代表机制，不是 OpenVLA 或 π 的原版权重。

| 仅一遍数据，单种子；五组逐行同序 | 新布局成对成功 |
| --- | ---: |
| 一步世界模型 | 86.87% |
| JSTP-P | 73.96% |
| 仅直接动作 | 69.86% |
| DINOv2 + DINO-WM 任务适配 | 49.71% |
| 官方 SmolVLA 局部微调 | 14.47% |

每组实际使用同一批 16,384 条交互和相同目标卡，512 × 32 次采样。前两表**不能当成一个同预算榜单**；第二表对齐的是下游经历与更新次数，不是外部预训练、模型规模或 FLOPs。前三个任务内模型的单轮训练为额外预算桥接，锁定在知道主结果和 DINO 结果之后。[单轮排程](evidence/transfer/schedule.npz) · [配对区间](evidence/transfer/paired-contrasts.json) · [适配方案](transfer-plan.json)。DINO 组实际使用官方 DINOv2 权重与作者的 ViTPredictor，在本装置重新训练预测器；SmolVLA 实际加载官方视觉语言干路与动作专家，不是代写的小模型。

## 顺着一条证据读

1. [协议](protocol.json)和[原字节数据](data/)：2,048 个训练布局、96 个验证布局；新布局只用于考试，没有加训练数据。
2. [模型](models.py)与[同序训练](train.py)：能只预测动作；不强制还原整幅画面。每次训练的曲线、实际耗时和冻结指纹在 [models/](evidence/models/)。
3. [校准、封卷、实操](exam.py) → [新布局逐题意向](evidence/fresh/sealed/) → [执行后的相机裁判](evidence/fresh/observed/) → [统计](analyze.py)。新旧面板先全部封卷再实际执行。
4. [官方模型的训练与执行](transfer.py)、[作者原版预测器与许可](upstream/dino_wm/)、[实际适配增量](evidence/transfer/)。基础权重由原作者分发，不装进本库。
5. [插图所用具体一题](paper/figures/case.json)：同一机构、原预期拉 A 成功，移植后拉 B 失败。配图是实际相机，不是假想图。若想操作另外的实时新布局，从仓库根目录运行 `python -m studies.partial_futures.window`，或双击“打开研究四试验台.cmd”。现场窗口支持六种小模型；官方大模型结果在迁移档案中，没有伪装成现场运行。

## 从仓库根目录复现

研究主环境沿用 [Windows / RTX 3090 锁定依赖](../../requirements-windows-cu132.txt)。以下命令将新训练结果写到 `work/`，不覆盖本次存档：

```powershell
.\.venv\Scripts\python.exe -m studies.partial_futures.run --out work/rebuild-partial
```

要重做官方权重适配，另建环境，勿改历史主环境。实测 CPython 3.13.5，LeRobot 0.7.0，Transformers 5.22.2，PyTorch 2.15.0+cu132，torchvision 0.29.2+cu132；其余版本见[锁定环境](requirements-transfer.txt)。首次下载需联网，此后用本地权重。基准只把冻结适配层保存进仓库；恢复时严格读官方基座，校验 SHA-256，然后覆盖本研究实际训练过的参数。

```powershell
uv venv --python 3.13 work/vla-env
uv pip install --python work/vla-env/Scripts/python.exe --torch-backend cu132 -r studies/partial_futures/requirements-transfer.txt
$py = ".\work\vla-env\Scripts\python.exe"
& $py -m studies.partial_futures.prepare_models
& $py -m studies.partial_futures.transfer smol --weights work/upstream-smolvla --out work/rebuild-partial/transfer
& $py -m studies.partial_futures.transfer dino --weights work/upstream-dino --out work/rebuild-partial/transfer
.\.venv\Scripts\python.exe -m studies.partial_futures.rebuild_matched --out work/rebuild-partial
```

`data/` 是可随仓库直接读取的训练资料，非外部占位链接。已有冻结结果不要求重新下载基础模型才能看论文、曲线或逐题回放。论文由 [build.py](paper/build.py) 从正式证据排版；含独立中文字体子集与授权。
