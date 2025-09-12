# 在 Google Colab 中运行的版本
# ⚠️ DEPRECATION NOTICE / 弃用通知 ⚠️
# 本脚本使用了已弃用的 haili_backtest 内容，建议使用新的集成策略。
# This script uses deprecated haili_backtest content. Consider using the new integrated strategy.
# 
# 推荐使用 / Recommended usage:
# from haili_integrated_strategy import run_integrated_strategy, quick_run
#
print("⚠️ 注意：此脚本使用已弃用的回测逻辑 / Warning: This script uses deprecated backtest logic")
print("建议使用新的集成策略模块 / Consider using the new integrated strategy module")
print("更多信息请参考项目README / See project README for more information")
print()

!pip install tushare

import tushare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google.colab import files
import warnings
warnings.filterwarnings('ignore')

# ... (复制上面的 haili_backtest 函数内容)

# 运行回测
success = haili_backtest()

if success:
    # 在 Colab 中下载 CSV 文件
    files.download('daily_backtest_results.csv')
    files.download('strategy_summary.csv')
    files.download('monthly_performance.csv')
    files.download('analysis_data.csv')
    
    print("所有 CSV 文件已生成并可下载")
