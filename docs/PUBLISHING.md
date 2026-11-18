# 私有仓库与引用

作者：岳轲。仓库：[ykykykk/jstp-intervention-bench](https://github.com/ykykykk/jstp-intervention-bench)，按作者要求设为**私有**。未获授权的访问者看不到仓库或发行附件。建立私有仓库、存档 Git 时间和打标签都不是向期刊投稿，也不等于公开预注册。

- [研究一](../README-v1.md)保留原论文、权重、统计和实验；初版提交为 `4e75ec9`。
- [研究二](../README.md)同时保留训练前的协议、首轮成绩和主结果之后才注册的隐状态实验。[时间线](MECHANISM-REPRODUCE.md)区分三者。请引用具体论文标题、岳轲、年份、版本和所用提交；没有 DOI。
- 本仓库和发行包不包括系统桌面截图、用户文件路径、虚拟环境、凭据或外部预训练权重。软件与实验记录见 [LICENSE](../LICENSE)，论文、理论原稿和第三方材料的范围见 [THIRD_PARTY.md](../THIRD_PARTY.md)。

## 维护同一私有仓库

变更时保留原始封卷和证据；如更改实验协议或重训，写入新目录、新版本。在作者已认证的仓库中操作，不把 token 写进命令、远端 URL 或配置文件。暂存拟发布的文件后运行：

    python scripts/verify_release.py --write
    git add MANIFEST.sha256
    python scripts/verify_release.py

提交后以 Git 对象归档源码和证据。这个清单只涵盖所列发行文件；ZIP 本身不自我签名。附带的 GitHub Actions 仅在支持的环境中验证文件与重算 CPU 统计，不替代全量重训。没有自动推公共仓库、投稿或更改可见性的步骤。
