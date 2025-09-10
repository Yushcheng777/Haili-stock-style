import tushare as ak
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import numpy as np

# ==============================
# 参数设置
# ==============================
MIN_MARKET_CAP = 50    # 最小市值（亿）
MAX_MARKET_CAP = 100   # 最大市值（亿）
TOPIC_KEYWORDS = ["光刻机", "半导体", "芯片", "光刻胶", "封装"]
OUTPUT_CSV = "candidates_haili_style.csv"
BACKTEST_OUTPUT_DIR = "backtest_results"

# ==============================
# 选股模块 - 从原haili_strategy.py移植
# ==============================
def check_funds_inflow(stock_code):
    try:
        df_funds = ak.stock_individual_fund_flow(stock=stock_code)
        recent3 = df_funds.head(3)
        return recent3["主力净流入"].sum() > 0
    except:
        return False

def check_weekly_positive(df):
    df_w = df.resample('W-FRI', on='date').last().dropna()
    df_w['chg'] = df_w['close'].pct_change()
    return (df_w['chg'] > 0).tail(2).sum() >= 2

def check_no_consecutive_limit_down(df):
    df['pct_chg'] = df['close'].pct_change()
    recent5 = df.tail(6)
    flags = recent5['pct_chg'] <= -0.099
    consec = any(flags.iloc[i] and flags.iloc[i+1] 
                 for i in range(len(flags)-1))
    return not consec

def check_tech_conditions(df):
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = ema12 - ema26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    macd_ok = df['DIF'].iloc[-1] > df['DEA'].iloc[-1] and df['DIF'].iloc[-1] > 0

    # 均线多头排列
    ma5  = df['close'].rolling(5).mean().iloc[-1]
    ma10 = df['close'].rolling(10).mean().iloc[-1]
    ma20 = df['close'].rolling(20).mean().iloc[-1]
    ma60 = df['close'].rolling(60).mean().iloc[-1]
    ma_ok = ma5 > ma10 > ma20 > ma60

    # 放量突破
    vol_mean20 = df['vol'].rolling(20).mean().iloc[-1]
    vol_ok = df['vol'].iloc[-1] > 1.5 * vol_mean20

    return macd_ok and ma_ok and vol_ok

# ==============================
# 回测模块 - 从haili_backtest.py移植并优化
# ==============================
def compute_indicators(df, ticker=""):
    """计算技术指标并进行回测"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 计算技术指标
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = ema12 - ema26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2
    
    # 均线
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA60'] = df['close'].rolling(window=60).mean()
    
    # 布林带
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 生成交易信号
    df['signal'] = 0
    df.loc[(df['DIF'] > df['DEA']) & 
           (df['close'] > df['MA20']) & 
           (df['RSI'] < 70), 'signal'] = 1  # 买入信号
    df.loc[(df['DIF'] < df['DEA']) | 
           (df['RSI'] > 80), 'signal'] = -1  # 卖出信号
    
    # 计算回测收益
    df['position'] = df['signal'].shift(1).fillna(0)
    df['strategy_return'] = df['position'] * df['close'].pct_change()
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    df['benchmark_return'] = (1 + df['close'].pct_change()).cumprod()
    
    return df

def backtest_analysis(df, ticker=""):
    """回测分析统计"""
    if len(df) < 60:  # 数据不足
        return None
        
    # 基础统计
    total_return = df['cumulative_return'].iloc[-1] - 1
    benchmark_return = df['benchmark_return'].iloc[-1] - 1
    excess_return = total_return - benchmark_return
    
    # 风险指标
    returns = df['strategy_return'].dropna()
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    # 最大回撤
    cumret = df['cumulative_return']
    rolling_max = cumret.expanding().max()
    drawdown = (cumret - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # 交易统计
    signals = df[df['signal'] != 0]
    total_trades = len(signals)
    
    return {
        'ticker': ticker,
        'total_return': total_return,
        'benchmark_return': benchmark_return,
        'excess_return': excess_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'data_points': len(df)
    }

def plot_backtest_results(df, ticker="", save_path=None):
    """绘制回测结果图"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # 价格和均线
    ax1.plot(df['date'], df['close'], label='收盘价', linewidth=1)
    ax1.plot(df['date'], df['MA20'], label='MA20', alpha=0.7)
    ax1.plot(df['date'], df['BB_upper'], label='布林上轨', alpha=0.5, linestyle='--')
    ax1.plot(df['date'], df['BB_lower'], label='布林下轨', alpha=0.5, linestyle='--')
    
    # 标记买卖信号
    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]
    ax1.scatter(buy_signals['date'], buy_signals['close'], color='green', marker='^', s=50, label='买入')
    ax1.scatter(sell_signals['date'], sell_signals['close'], color='red', marker='v', s=50, label='卖出')
    
    ax1.set_title(f'{ticker} 价格走势与交易信号')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MACD
    ax2.plot(df['date'], df['DIF'], label='DIF', linewidth=1)
    ax2.plot(df['date'], df['DEA'], label='DEA', linewidth=1)
    ax2.bar(df['date'], df['MACD'], label='MACD柱', alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('MACD指标')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 累计收益对比
    ax3.plot(df['date'], (df['cumulative_return'] - 1) * 100, label='策略收益', linewidth=2)
    ax3.plot(df['date'], (df['benchmark_return'] - 1) * 100, label='基准收益', linewidth=2)
    ax3.set_title('累计收益对比 (%)')
    ax3.set_ylabel('收益率 (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ==============================
# 主集成函数
# ==============================
def run_integrated_strategy(current_positions=None, backtest_days=252):
    """
    集成策略：选股 + 回测
    
    Args:
        current_positions (dict): 当前持仓字典
        backtest_days (int): 回测天数，默认252（一年）
    
    Returns:
        tuple: (选股结果DataFrame, 回测结果列表)
    """
    print("=" * 50)
    print("开始执行海力风格集成策略")
    print("=" * 50)
    
    # 确保输出目录存在
    os.makedirs(BACKTEST_OUTPUT_DIR, exist_ok=True)
    
    # Step 1: 执行选股
    print("\n1. 执行选股逻辑...")
    selected_stocks = haili_style_selection_internal(current_positions)
    
    if selected_stocks.empty:
        print("未找到符合条件的股票")
        return selected_stocks, []
    
    print(f"选股完成，共选出 {len(selected_stocks)} 只股票")
    
    # Step 2: 对选出的股票执行回测
    print("\n2. 开始批量回测...")
    backtest_results = []
    
    for idx, row in selected_stocks.iterrows():
        code = row['代码']
        name = row['名称']
        
        try:
            print(f"正在回测: {code} {name}")
            
            # 获取历史数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=backtest_days + 100)).strftime('%Y%m%d')
            
            df_hist = ak.stock_zh_a_daily(symbol=code, start_date=start_date, end_date=end_date)
            df_hist = df_hist.reset_index()
            df_hist.rename(columns={"date": "date"}, inplace=True)
            df_hist['date'] = pd.to_datetime(df_hist['date'])
            
            if len(df_hist) < 60:  # 数据不足
                print(f"  {code} 数据不足，跳过")
                continue
            
            # 执行回测
            df_backtest = compute_indicators(df_hist, ticker=f"{code}_{name}")
            analysis = backtest_analysis(df_backtest, ticker=f"{code}_{name}")
            
            if analysis:
                backtest_results.append(analysis)
                
                # 绘制并保存回测图表
                chart_path = os.path.join(BACKTEST_OUTPUT_DIR, f"{code}_{name}_backtest.png")
                plot_backtest_results(df_backtest, ticker=f"{code} {name}", save_path=chart_path)
                
                print(f"  回测完成: 总收益 {analysis['total_return']:.2%}, 夏普比率 {analysis['sharpe_ratio']:.2f}")
            
        except Exception as e:
            print(f"  {code} 回测失败: {str(e)}")
            continue
    
    # Step 3: 生成汇总报告
    if backtest_results:
        print(f"\n3. 生成汇总报告...")
        df_backtest_summary = pd.DataFrame(backtest_results)
        
        # 排序：按夏普比率降序
        df_backtest_summary = df_backtest_summary.sort_values('sharpe_ratio', ascending=False)
        
        # 保存回测汇总
        summary_path = os.path.join(BACKTEST_OUTPUT_DIR, "backtest_summary.csv")
        df_backtest_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
        
        # 打印前5名
        print("\n回测结果TOP5 (按夏普比率排序):")
        print("-" * 80)
        for i, row in df_backtest_summary.head(5).iterrows():
            print(f"{row['ticker']}: 收益 {row['total_return']:.2%}, "
                  f"夏普 {row['sharpe_ratio']:.2f}, 最大回撤 {row['max_drawdown']:.2%}")
        
        print(f"\n详细回测结果已保存到: {BACKTEST_OUTPUT_DIR}/")
    
    print("\n" + "=" * 50)
    print("集成策略执行完成")
    print("=" * 50)
    
    return selected_stocks, backtest_results

def haili_style_selection_internal(current_positions=None):
    """
    内部选股函数，返回DataFrame而不保存文件
    """
    # 处理当前持仓数据
    if current_positions is None:
        current_positions = {}
        csv_path = "current_positions.csv"
        if os.path.exists(csv_path):
            try:
                df_pos = pd.read_csv(csv_path)
                if '代码' in df_pos.columns and '当前仓位(%)' in df_pos.columns:
                    for _, row in df_pos.iterrows():
                        current_positions[str(row['代码']).zfill(6)] = row['当前仓位(%)']
                elif 'code' in df_pos.columns and 'current_pos' in df_pos.columns:
                    for _, row in df_pos.iterrows():
                        current_positions[str(row['code']).zfill(6)] = row['current_pos']
            except Exception as e:
                print(f"读取持仓文件失败: {e}")
    
    stock_list = ak.stock_info_a_code_name()
    results = []

    for idx, row in stock_list.iterrows():
        code = row['code']
        name = row['name']
        try:
            # 基本面过滤：市值
            info = ak.stock_individual_info_em(code)
            market_cap = float(
                info.loc[info["item"]=="总市值", "value"].values[0]
            )
            if not (MIN_MARKET_CAP <= market_cap <= MAX_MARKET_CAP):
                continue

            # 题材匹配
            conc = ak.stock_board_concept_name_em()
            if not any(kw in " ".join(conc['板块名称']) 
                       for kw in TOPIC_KEYWORDS):
                continue

            # 获取日K数据
            df = ak.stock_zh_a_daily(symbol=code)
            df = df.reset_index()
            df.rename(columns={"date":"date"}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])

            # 各项检查
            if not check_no_consecutive_limit_down(df):
                continue
            if not check_tech_conditions(df):
                continue
            if not check_weekly_positive(df):
                continue
            if not check_funds_inflow(code):
                continue

            # 计算相关指标
            current_pos = current_positions.get(code, 0)
            target_pos = min(100, max(0, (100 - market_cap) * 0.5))
            
            action_score = 0
            if market_cap <= 80:
                action_score += 2
            if df['close'].iloc[-1] > df['close'].rolling(20).mean().iloc[-1]:
                action_score += 2
            if df['vol'].iloc[-1] > df['vol'].rolling(20).mean().iloc[-1]:
                action_score += 1
                
            pos_diff = target_pos - current_pos
            if abs(pos_diff) < 5:
                suggestion = "持有"
                direction = "无"
                adjust_pct = 0
            elif pos_diff > 0:
                suggestion = "买入"
                direction = "买入"
                adjust_pct = pos_diff
            else:
                suggestion = "卖出"
                direction = "卖出"  
                adjust_pct = abs(pos_diff)
            
            results.append({
                "代码": code,
                "名称": name,
                "总市值(亿)": market_cap,
                "action_score": action_score,
                "目标仓位(%)": target_pos,
                "当前仓位(%)": current_pos,
                "建议下单方向": direction,
                "建议调整(%)": adjust_pct,
            })

        except Exception:
            continue

    return pd.DataFrame(results)

# ==============================
# 快速使用函数
# ==============================
def quick_run(stock_codes=None, backtest_days=252):
    """
    快速运行：直接对指定股票代码进行回测
    
    Args:
        stock_codes (list): 股票代码列表，如['000001', '000002']
        backtest_days (int): 回测天数
    """
    if stock_codes is None:
        print("请提供股票代码列表")
        return
    
    os.makedirs(BACKTEST_OUTPUT_DIR, exist_ok=True)
    backtest_results = []
    
    for code in stock_codes:
        try:
            # 获取股票名称
            stock_info = ak.stock_individual_info_em(code)
            name = stock_info.loc[stock_info["item"]=="股票简称", "value"].values[0]
            
            print(f"正在回测: {code} {name}")
            
            # 获取历史数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=backtest_days + 100)).strftime('%Y%m%d')
            
            df_hist = ak.stock_zh_a_daily(symbol=code, start_date=start_date, end_date=end_date)
            df_hist = df_hist.reset_index()
            df_hist.rename(columns={"date": "date"}, inplace=True)
            df_hist['date'] = pd.to_datetime(df_hist['date'])
            
            # 执行回测
            df_backtest = compute_indicators(df_hist, ticker=f"{code}_{name}")
            analysis = backtest_analysis(df_backtest, ticker=f"{code}_{name}")
            
            if analysis:
                backtest_results.append(analysis)
                
                # 绘制并保存回测图表
                chart_path = os.path.join(BACKTEST_OUTPUT_DIR, f"{code}_{name}_backtest.png")
                plot_backtest_results(df_backtest, ticker=f"{code} {name}", save_path=chart_path)
                
                print(f"  回测完成: 总收益 {analysis['total_return']:.2%}")
            
        except Exception as e:
            print(f"  {code} 回测失败: {str(e)}")
    
    # 保存汇总结果
    if backtest_results:
        df_summary = pd.DataFrame(backtest_results)
        summary_path = os.path.join(BACKTEST_OUTPUT_DIR, "quick_backtest_summary.csv")
        df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n回测汇总已保存到: {summary_path}")
    
    return backtest_results

# ==============================
# 主入口
# ==============================
if __name__ == "__main__":
    # 方式1: 完整的选股+回测流程
    selected_stocks, backtest_results = run_integrated_strategy()
    
    # 方式2: 快速回测指定股票（取消注释使用）
    # quick_run(['000001', '000002', '600036'])