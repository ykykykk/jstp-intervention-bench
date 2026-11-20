# 完整 JSTP 理论 / The full theory

**岳轲 · Ke Yue**　[中文原稿](JSTP-详细理论稿-2026-11-19.md) · [中文 PDF](JSTP-Theory-Ke-Yue.zh.pdf)　|　**[Read the full English theory](JSTP-Theory.en.md)** · [English PDF](JSTP-Theory-Ke-Yue.en.pdf)

The two editions cover the same 18 sections, register of 13 core claims, and two appendices. Read **§§3–4** for why a joint state need not imply an exhaustive prediction, **§§8–11** for the author's stronger claims about language, action, and the self, and **§15.8** for the frozen-model observations and their limits.

**配套实证论文 / Companion empirical paper:** [Revised manuscript v1.2](../studies/joint_primer/paper/manuscript-en.md) · [中文修订稿](../studies/joint_primer/paper/manuscript.md) · [Cite the v1.2 preprint](https://doi.org/10.5281/zenodo.24392467) · [Archived v1.1](https://doi.org/10.5281/zenodo.24366742). Its external comparison tests action selection after a probe, not the full theory or the emergence of object segmentation.

## 稿本与来源

- [详细理论稿 · 2026-11-19 修订](JSTP-详细理论稿-2026-11-19.md)：当前研究采用的版本；同步自作者本地详细稿，仅调整仓库中的证据链接。
- [详细理论稿 · 原始研究快照](JSTP-详细理论稿.md)：保留原文，供追溯。
- [研究一之后的理论审阅](研究一之后的理论审阅.md)：当时的解释边界，不代替后来的澄清。

“预测自身动作”与“预测动作将引起的环境变化”都是 JSTP 容许的预测内容。实验要比较的是具体实现和路径，不是把直接动作定义成理论的反面。

**Rebuilding the two PDFs:** from the repository root, run `python theory/build.py` with Pandoc and the [paper dependencies](../paper/requirements.txt). The included font subsets use the [SIL Open Font License](fonts/OFL.txt); the manuscripts themselves retain the author's copyright. The Chinese PDF is typeset from the unaltered revised source above, and the English PDF from its full translation.

Only when the font subsets need to cover new characters, regenerate them with `python scripts/prepare_fonts.py --paper-dir theory --text-file JSTP-详细理论稿-2026-11-19.md --text-file JSTP-Theory.en.md --text-file build.py` before typesetting.
