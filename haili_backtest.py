import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def haili_backtest():
    """海利策略回测并生成CSV输出"""
    
    # 获取数据（示例：沪深300指数）
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
    
    try:
        # 获取沪深300指数数据
        df = ak.stock_zh_index_daily(symbol="sh000300")
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # 海利策略逻辑（示例）
        df['returns'] = df['close'].pct_change()
        df['signal'] = np.where(df['returns'].rolling(5).mean() > 0, 1, 0)
        df['strategy_returns'] = df['returns'] * df['signal'].shift(1)
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
        df['benchmark_returns'] = (1 + df['returns']).cumprod()
        
        # 计算关键指标
        total_return = df['cumulative_returns'].iloc[-1] - 1
        benchmark_return = df['benchmark_returns'].iloc[-1] - 1
        volatility = df['strategy_returns'].std() * np.sqrt(252)
        sharpe_ratio = (df['strategy_returns'].mean() * 252) / (df['strategy_returns'].std() * np.sqrt(252))
        max_drawdown = calculate_max_drawdown(df['cumulative_returns'])
        
        # 生成详细的CSV报告
        generate_csv_reports(df, {
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        return True
        
    except Exception as e:
        print(f"Error in backtest: {e}")
        return False

def calculate_max_drawdown(cumulative_returns):
    """计算最大回撤"""
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def generate_csv_reports(df, metrics):
    """生成多个CSV文件用于不同工具导入"""
    
    # 1. 每日交易记录 (详细数据)
    daily_report = df[['date', 'open', 'high', 'low', 'close', 'volume', 
                      'returns', 'signal', 'strategy_returns', 
                      'cumulative_returns', 'benchmark_returns']].copy()
    daily_report['date'] = daily_report['date'].dt.strftime('%Y-%m-%d')
    daily_report.to_csv('daily_backtest_results.csv', index=False, encoding='utf-8')
    
    # 2. 汇总指标报告 (适合Notion导入)
    summary_data = {
        'Run_Date': [metrics['run_date']],
        'Strategy': ['Haili Strategy'],
        'Period': [f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"],
        'Total_Return_Pct': [f"{metrics['total_return']:.2%}"],
        'Benchmark_Return_Pct': [f"{metrics['benchmark_return']:.2%}"],
        'Excess_Return_Pct': [f"{metrics['total_return'] - metrics['benchmark_return']:.2%}"],
        'Volatility_Pct': [f"{metrics['volatility']:.2%}"],
        'Sharpe_Ratio': [f"{metrics['sharpe_ratio']:.2f}"],
        'Max_Drawdown_Pct': [f"{metrics['max_drawdown']:.2%}"],
        'Trading_Days': [len(df)],
        'Winning_Days': [len(df[df['strategy_returns'] > 0])],
        'Win_Rate': [f"{len(df[df['strategy_returns'] > 0]) / len(df[df['strategy_returns'] != 0]):.2%}"]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('strategy_summary.csv', index=False, encoding='utf-8')
    
    # 3. 月度汇总 (适合追踪)
    monthly_df = df.copy()
    monthly_df['year_month'] = monthly_df['date'].dt.to_period('M')
    monthly_summary = monthly_df.groupby('year_month').agg({
        'strategy_returns': ['sum', 'std', 'count'],
        'returns': 'sum',
        'cumulative_returns': 'last'
    }).round(4)
    
    monthly_summary.columns = ['Monthly_Strategy_Return', 'Monthly_Volatility', 'Trading_Days',
                              'Monthly_Benchmark_Return', 'Cumulative_Value']
    monthly_summary.reset_index(inplace=True)
    monthly_summary['year_month'] = monthly_summary['year_month'].astype(str)
    monthly_summary.to_csv('monthly_performance.csv', index=False, encoding='utf-8')
    
    # 4. 适合Replit/数据分析的格式
    analysis_df = df[['date', 'close', 'cumulative_returns', 'benchmark_returns']].copy()
    analysis_df['date'] = analysis_df['date'].dt.strftime('%Y-%m-%d')
    analysis_df['strategy_value'] = analysis_df['cumulative_returns']
    analysis_df['benchmark_value'] = analysis_df['benchmark_returns']
    analysis_df.to_csv('analysis_data.csv', index=False, encoding='utf-8')
    
    print("CSV files generated:")
    print("- daily_backtest_results.csv (详细每日数据)")
    print("- strategy_summary.csv (策略汇总，适合Notion)")
    print("- monthly_performance.csv (月度表现)")
    print("- analysis_data.csv (分析数据，适合Replit)")

if __name__ == "__main__":
    success = haili_backtest()
    if success:
        print("回测完成，CSV文件已生成")
    else:
        print("回测失败")
