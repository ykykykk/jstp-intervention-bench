# 把这份工程发布到 GitHub

发行包不含系统桌面截图、用户目录、虚拟环境、GitHub 凭据或外部预训练权重。PDF 与冻结检查点在普通 Git 文件大小范围内，不需 LFS。远端名称、组织归属和可见性由作者控制。

1. 从作者的**独立仓库目录**发布，而不是它的父项目；若只有 ZIP，解压进入含 README 的目录，运行 `git init -b main`、`git add .`、`git commit -m "Release reproducible JSTP study"`。新克隆的姓名/邮箱按作者自己的 GitHub 隐私偏好设置。
2. 在 [GitHub 新建仓库](https://github.com/new)，例如 `jstp-intervention-bench`。不要预生成 README 或 License。也可在已登录的终端运行下列 GitHub CLI 命令。先把 `YOUR_ACCOUNT` 改为实际账号；只有想公开时才用 `--public`。

        gh repo create YOUR_ACCOUNT/jstp-intervention-bench --public
        git remote add origin https://github.com/YOUR_ACCOUNT/jstp-intervention-bench.git
        git push -u origin HEAD:main

3. 在 GitHub 上把 `main` 设为默认分支；如需固定首版，可创建 `v1.0.0` 标签和 Release，附论文 PDF 和源码包。附带的 GitHub Actions 仅核对发布清单与 CPU 统计，没有自动发布或云 GPU 训练步骤。
4. 有了实际远端地址再写入 `CITATION.cff` 或投稿系统。正式取得 DOI 后再填真实标识，不把临时仓库名、期刊接收或机构背书写成事实。

变更了发行文件时，先 `git add` 拟发布内容，再运行 `python scripts/verify_release.py --write`，审阅差异后 `git add MANIFEST.sha256` 并提交。不要因发版而更改原始试验，训练或协议变更应用新版本另存。
