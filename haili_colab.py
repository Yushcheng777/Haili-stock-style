# 在 Google Colab 中运行的版本
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
