# 仓库精简说明

目标：本仓库仅保留策略代码、回测（backtest）相关逻辑、说明文档，并且通过 GitHub Actions 定时执行回测、保存结果以及对新 issue 自动回复。

推荐目录结构：
- strategies/          # 策略代码（你的交易策略）
- backtests/           # 回测相关代码与配置（或把回测脚本放在 scripts/）
- scripts/             # 辅助脚本，例如 run_all_backtests.py
- docs/                # 文档（说明、使用方法）
- results/             # （由 CI 写入到 results 分支，用于存放历史回测产物）
- .github/workflows/   # workflow（定时回测、自动回复等）
- requirements.txt
- README.md

如何运行（本地）：
1. 安装依赖： pip install -r requirements.txt
2. 验证环境： python scripts/run_all_backtests.py --dry-run
3. 运行回测（示例）： python scripts/run_all_backtests.py

## 依赖管理
本仓库使用 requirements.txt 管理核心依赖：
- akshare==1.9.2
- backtrader==1.9.76.123  
- pandas==1.5.3
- numpy==1.23.5
- matplotlib==3.6.3

运行 `python scripts/run_all_backtests.py --dry-run` 可以验证所有依赖是否正确安装。

CI（自动化验证与定时回测）：
- 新增 .github/workflows/validate.yml，每次 push/PR 时自动安装依赖并验证环境完整性。
- 已添加 .github/workflows/scheduled-backtest.yml，会按计划运行回测并将结果推送到 `results` 分支，同时上传 artifact 供短期查看。

安全注意：
- 在将结果 commit 到仓库前，请确认结果文件不会包含大文件或敏感信息；对于很大的产物（数百 MB 以上），推荐上传到外部对象存储并在仓库保留 summary/metadata。
