# 从发布包复现

用 Git 克隆或解压整个源码包，进入含 README 的根目录；不依赖另一个私有项目。不要只复制某个 Python 文件或单独安装一个没有配置和证据的 wheel。

## A. 核对发布物和统计（CPU，离线数据）

1. 使用 CPython 3.11 或更高版本运行 `python scripts/verify_release.py`。它逐文件校验 `MANIFEST.sha256`，包括权重、逐题账本和 PDF；不连接网络。摘要用于发现文件错乱，不构成作者身份的数字签名。
2. 有 NumPy 后，执行 `python -m jstp.analysis --release evidence/paper --out work/recount`。只读冻结选择和真实的执行记录，不加载神经网络。发布版使用 NumPy 2.5.5。输出须为新目录。
3. 比较 `work/recount` 和 `evidence/paper/analysis` 的 `metrics.csv`、`per_seed.csv`、`paired_differences.csv`。`task`、`flat`、`exchange` 和两组区间存为 0–1 比例；`pixel_mae` 是像素。`physics_pair` 为空表示没有可比的 RGB 读出，不是零分。

想把虚拟执行器再走一遍而不是只重算统计？安装主环境的 PyTorch 与 NumPy 后运行：

    python -m jstp.exam score --release evidence/paper --out work/replayed-outcomes --device cpu

此命令先核对原始选择与权重的哈希，再重建每个世界、试探并执行已封存的选择。没有 DINO 权重也能用，因为它不重做推理。每个测试条件都会生成 `example.npz`、`outcomes.npz`；可比较其数组与原始 `observed` 目录对应内容，不比运行时间戳。要重新做**推理**和策略选择，请走下一条路线。

## B. 完整重训（已实做两次）

实测：Windows，Git，CPython 3.13.5，NVIDIA GeForce RTX 3090（24 GB），PyTorch 2.15.0+cu132。依赖见[锁定清单](../requirements-windows-cu132.txt)。上游 DINO-WM 预测器直接使用 CUDA，不把更换实现后的 CPU 运行当作同一基线。未做跨机等值保证。模型不下载个人文件或访问系统桌面。

    py -3.13 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements-windows-cu132.txt
    .\.venv\Scripts\python.exe scripts/reproduce.py
    .\.venv\Scripts\python.exe scripts/compare_runs.py evidence/paper work/reproduction-results

默认工作区 `work/reproduction`；默认输出 `work/reproduction-results`。另跑一份可传 `--work work/run-b --release work/result-b`，不覆盖原始证据。隔离检查使用新环境、只拷源码的空目录和重新下载的上游素材，未把发布的冻结权重用于训练。`compare_runs.py` 逐项比较两张试卷的输入指纹、真实分支指纹、选择、实际成败与局部位移差；差异以数字报告，不会改写发布结果。

**完整流程与数据边界**

1. `scripts/prepare_external.py` 获取两个固定 Git 提交和约 88 MB 的 Meta 官方编码器权重，检查 SHA-256。首次需网络，已就绪的相同缓存可复用。上游缓存及训练数组约 1.1 GB，另为 Python、CUDA 软件包及解压预留数 GB；第三方资产不进入本仓库。
2. `jstp.prepare data` 从不同布局种子生成 2,048 个直接拖动布局、2,048 个带四个真实分支的成对训练布局、96 个验证布局。目标的公开空间带是任务说明，不是物体分割标签。
3. `jstp.prepare features` 从相同 RGB/目标图缓存冻结 DINO 特征。训练只收到模型接口允许的观测；验证目标来自可见真实分支。`jstp.train` 为六种方法和三个种子各训练 300＋1,500 步，按验证集选检查点，不在最终考试上挑模型。
4. `jstp.exam seal` 写完全部方法和布局的代价/选择及适用的预测指纹，再写 `seal-before-action.json`。此阶段不执行正式竖拖。`jstp.exam score` 验证指纹后才正式竖拖，保存各方法的真实成败和真实分支指纹。
5. `jstp.analysis` 按每个条件的 128 个布局成簇做配对自举，10,000 次；同布局的目标、挂接、证据方式和种子不是独立场景。区间仅覆盖对已训练的三个模型进行布局抽样的不确定性。

可用 `python -m unittest discover -s tests` 单独核对：两种挂接复位后相同、每个目标的正确操作唯一、三位规则预测等于实际、以及一位规则不能偷看另两带。原始协议在 [configs/paper.json](../configs/paper.json)，哈希见各份 `freeze.json`。

## C. 怎样审计一道题？

两张试卷名为 `test_id`（未见布局、熟悉机构）与 `test_link`（未见布局、未见机构）。各自编号 0 对应协议范围的第一个种子。

| 文件（相对 evidence/paper） | 内容及维度 |
|---|---|
| `exam/<split>/observations.json` | 各布局的输入、过去命令、候选、目标等指纹 |
| `exam/<split>/<方法>-<种子>.npz` | `cost[布局,目标类型,证据干预,实际挂接,目标,候选]`，`choice` 去掉最后一维。选代价较小者（并列取先）。有 RGB 的方法还给 `dy[布局,挂接,候选,区域]`，以及第一布局两挂接两候选的完整预测图。其余布局保存摘要/指纹，可用冻结权重再推理。 |
| `observed/<split>/outcomes.npz` | 每个实例的成功数组与 `choice` 同维度；含两候选实际帧摘要 `actual_sha` 和真实局部位移 `dy`。 |
| `observed/<split>/example.npz` | 第一布局的复位帧、两种试探峰值、目标与区域、两挂接两候选的实际结果。 |

目标类型 0/1 是单区/双区；证据 0/1/2 是事实/抹平试探峰值/交换两世界的试探视觉经历。每条件 128 × 2 挂接 × 2 目标 × 2 类型 × 3 证据方式 × 20 个实例（18 组学习权重＋2 个规则）= 61,440 个选择。两张试卷共 122,880 个，含 40,960 个事实选择，不是 122,880 个独立场景。

## D. 历史证据不是主试验从零训练的结果

    python -m jstp.legacy v6 --device cuda
    python -m jstp.legacy v7 --device cuda

V6 归档所需推理权重和 300 条摘要，剔除优化器与私有目录信息，交付逐场成功与步数相同的**推理重放**。V7 有原始冻结权重、训练曲线、封存决定和执行记录；可重放 128 个布局的事实、删证据及换证据结果。不要把旧任务和新三载荷的分数相减。侧窗小游戏 4/4 未作为论文证据。

## E. 重新排论文和证据页

包内含可再分发的字体子集，生成 PDF 不需下载字体。为避免改动训练环境的 NumPy 版本，请另用排版环境：

    py -3.12 -m venv work/typeset
    .\work\typeset\Scripts\python.exe -m pip install -r paper/requirements.txt
    .\work\typeset\Scripts\python.exe paper/build.py
    .\work\typeset\Scripts\python.exe scripts/build_viewer.py

输入是 `paper/source.md`、`paper/references.json` 和 `evidence/paper`。输出 PDF、已填表 Markdown、两张数据图、BibTeX 及本地交互证据页。如果编辑后新增汉字不在字形子集中，先执行 `scripts/prepare_fonts.py`（需网络或原字体缓存）。协议或数据变化另起新版本，不改原封存结果。
