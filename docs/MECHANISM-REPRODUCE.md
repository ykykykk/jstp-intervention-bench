# 研究二 · 怎样独立核对一项主张

论文：[预期能否驱动行动？](../paper/mechanism_study/JSTP-运动与符号回输-岳轲.pdf)　作者：岳轲　版本：2.0.0

## 先区分三个时间点

| 在什么结果揭示前 | 锁定了什么 | 可追溯版本 |
|---|---|---|
| 训练前 | 两个主问题、划分、五个条件、三个种子、门槛及固定末步权重 | [342b9df · 主协议](https://github.com/ykykykk/jstp-intervention-bench/commit/342b9df) |
| 正式考试前，训练已启动 | 真实相机移植和未训练结构对照；不改变主判据 | [10d192f · 实现与诊断](https://github.com/ykykykk/jstp-intervention-bench/commit/10d192f) |
| 已看过首轮结果，尚未进行新的隐状态考试 | 另取 192 个布局标定两维物理方向，在另外两批各 160 个布局测试；不改变权重 | [d178154 · 后续协议](https://github.com/ykykykk/jstp-intervention-bench/commit/d178154) |

上表是这个私有 Git 仓库中的记录，不是第三方公开预注册。自学码的置换解释是描述性事后分析，不回写主门槛。

## 数据从哪里来

**搬运。** 每份经历都在独立虚拟装置中生成。训练共 1,536 个布局、76,635 条行为记录，其中包含 2,907 条误动后复位的纠正记录。验证为另外 96 个布局。每记录从给定目标、手部状态和先前试探出发，附模拟示范的驱动、真实载荷变化和本体反馈。训练还采样已实测的不同驱动，用于校准感觉后果；不能称为无示范或无先验学习。主动考试时不提供正确把手、候选评分、挂接真值或实例掩码。感觉输入是固定降采样的空间变化网格；任务也明示指定哪条带，不是从任意照片中发现全部对象。

**电路。** 确定种子生成三继电器初态、操作卡和中间物理状态。训练深度为 1–4；考试深度为 1、2、4、6、8，每层 384 对。孪生程序只反转第一张卡的触发条件，问题只询问真实终点不同的继电器。电路的三个可写位是唯一跨节拍通路。自学组三位本身没有词义标签，但有状态显示、整体码辅助目标和动作监督。独立的逐步标量执行器复核考试答案。

[生成器](../mechanism/data.py) · [输入、相机与驱动的数据流](../mechanism/net.py) · [训练损失](../mechanism/train.py) · [训练样本数量与哈希](../evidence/mechanism/experience.json)

## 从一题找回原始记录

以下示例只用 NumPy，取图 1 所用的首个熟悉布局，指定上台挂接 0、下台挂接 0、中带移动、主动搬运：

    from pathlib import Path
    import numpy as np
    e = Path("evidence/mechanism")
    s = e / "sealed/test_familiar"
    rows = np.load(s / "input.npz")["rows"]
    i = np.flatnonzero((rows[:, :6] == [0, 0, 0, 1, 1, 0]).all(axis=1)).item()
    z = np.load(s / "joint_symbol-1729.npz")
    o = np.load(e / "observed/test_familiar/joint_symbol-1729.npz")
    print("预演、六次提交、六次真实成败：", z["proposal"][i], z["u"][i], o["win"][i])
    print("同驱动的原世界与供体、上台网格：", z["prospect"][i,:96].reshape(8,12), z["donor"][i].reshape(8,12))

`u` 的六列依次是正常执行、原样重复、仅换相关台预期、仅换无关台预期、换相关台真实历史、抹平试探。数值 −1／0／1 是左驱动／静止／右驱动，不是候选分数。每个配置在每种接线分布下共有 7,680 行：3,840 主动，3,840 被动。每份电路封卷中 `d8_answer` 有 768 行、5 列，分别为自然、重算、供体、打乱、清空。

**顺序是预测 → 封存 → 真的执行 → 统计。** [封卷清单](../evidence/mechanism/sealed/before.json)记下权重、输入和每份提交的 SHA-256；[实测清单](../evidence/mechanism/observed/after.json)反向引用封卷指纹。执行器按布局、接线、外力和自身驱动索引真实相机：同驱动的多个决策复用一个确定性物理分支。不是把网络的预测当成真值，也不是每次提交都在不同的真实机器人上操作。

## 判据与失效条件

- **运动主判据。** 用所有主动题作分母；先要求正常运动正确，相关预期替换后改选供体世界的正确侧。减去无关替换产生的同类效应，要求差至少 10 个百分点、按布局配对区间下界高于零、至少两个种子同向；还要求自然任务区间下界高于 60%、严格两世界共动判别高于 50%。熟悉机构的后两项成立，定向换手差为零，**所以该主判据未通过**。这既不把成功率当作机制证据，也不把零效应写成一切内隐预测都不存在。
- **电路主判据。** 没有预置谓词语义的自学三位在 6 步、8 步均超过 75%，且至少有两个种子达到，即足以构成“这批题必须有指定符号通道”的反例。三个种子均达到；不能据此判定所有高级智能与语言无关。
- **区间。** 固定三个训练种子，按 160 个基础布局或 384 个孪生程序对做 6,000 次重采样；不把同一布局的所有动作当作独立样本。对全零观测的自举区间不外推为真概率恰好为零。[事前判据原文](../mechanism/protocol.json)

## 复现路径

1. **字节核对。** 在根目录运行 `python scripts/verify_release.py`。只需 Python 3.11+ 标准库。没有 Git 的 ZIP 也能核对。
2. **CPU 算分。** 安装 NumPy 2.5.5 后执行 `python -m mechanism.analyze --release evidence/mechanism --out work/recount-m2`。先核对承诺链，再输出 `metrics.csv`、`per_seed.csv`、`summary.json`。查看自学码可另运行 `python -m mechanism.codebook --release evidence/mechanism --out work/codebook-copy.json`。
3. **完全重训。** 按[首页](../README.md)建立独立环境，用 `python scripts/reproduce_mechanism.py --out work/fresh-m2`。从零生成训练资料，训练五种条件各三个种子，封卷、执行、计算两项主试验和后续诊断。再运行 `python scripts/compare_mechanism.py evidence/mechanism work/fresh-m2/release --require-identical`。精确模式发现有差别时返回非零；跨硬件比较可不加此选项，保留真实差异。研究二不需要下载第三方预训练模型。
4. **看在线过程。** 在训练使用的环境另开 `python -m mechanism.window --release work/fresh-m2/release`。窗口先显示验证曲线，首个主模型冻结后开始现场预测，正式结果封存后允许切换考试布局。用默认路径则可直接运行已随包发布的权重。
5. **重排论文。** 正常重排无需下载字体。在独立排版环境中安装 `python -m pip install -r paper/requirements.txt`，然后 `python paper/mechanism_study/build.py`。程序从封存的统计和相机指纹生成表、图、可编辑稿及 PDF。仅当更改字形范围时才需 `python scripts/prepare_fonts.py --paper-dir paper/mechanism_study`。

训练配置为 CPython 3.13.5、PyTorch 2.15.0+cu132、RTX 3090、NumPy 2.5.5；每份模型 159,443 参数，2,600 步。首轮 15 份模型的训练计时合计约 **11 分 39 秒**，不含生成数据、封卷、执行或补充诊断。同机独立环境重训的 **1,382,400 项运动提交和 288,000 项电路提交全部相同**；量化网格、离散消息、实测数组、主区间、补充试验和码表也一致。没有声称不同机器上浮点数必然逐位相同。[差异报告](../evidence/mechanism/independent-reproduction.json)

用来陈述作者原始立场的[理论原稿](../theory/JSTP-详细理论稿.md)、[第一项研究及其对照](../README-v1.md)、第二项研究的[逐题资料](../evidence/mechanism)各自保留；不要把早期游戏窗口的四次演示混入论文统计。
