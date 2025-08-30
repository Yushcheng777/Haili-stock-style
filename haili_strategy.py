import tushare as ak
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# ==============================
# 参数设置
# ==============================
MIN_MARKET_CAP = 50    # 最小市值（亿）
MAX_MARKET_CAP = 100   # 最大市值（亿）
TOPIC_KEYWORDS = ["光刻机", "半导体", "芯片", "光刻胶", "封装"]
OUTPUT_CSV = "candidates_haili_style.csv"

# ==============================
# 检查连续3日资金净流入
# ==============================
def check_funds_inflow(stock_code):
    try:
        df_funds = ak.stock_individual_fund_flow(stock=stock_code)
        recent3 = df_funds.head(3)
        return recent3["主力净流入"].sum() > 0
    except:
        return False

# ==============================
# 检查周线连阳 ≥ 2
# ==============================
def check_weekly_positive(df):
    df_w = df.resample('W-FRI', on='date').last().dropna()
    df_w['chg'] = df_w['close'].pct_change()
    return (df_w['chg'] > 0).tail(2).sum() >= 2

# ==============================
# 检查5日内无连续两个跌停
# ==============================
def check_no_consecutive_limit_down(df):
    # 计算日涨跌幅
    df['pct_chg'] = df['close'].pct_change()
    recent5 = df.tail(6)  # 包含今天往前5个间隔共6条记录
    # 判断跌停：pct_chg <= -0.099 （约-9.9%）
    flags = recent5['pct_chg'] <= -0.099
    # 找到任何连续两天均为True的情况
    consec = any(flags.iloc[i] and flags.iloc[i+1] 
                 for i in range(len(flags)-1))
    return not consec

# ==============================
# 计算MACD、均线多头排列、放量突破
# ==============================
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

    # 放量突破（当日成交量 > 1.5 * 20日均量）
    vol_mean20 = df['vol'].rolling(20).mean().iloc[-1]
    vol_ok = df['vol'].iloc[-1] > 1.5 * vol_mean20

    return macd_ok and ma_ok and vol_ok

# ==============================
# 绘制日线+周线叠加决策图
# ==============================
def plot_decision_chart(stock_code, stock_name, df):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['date'], df['close'], label="日线收盘价", color='blue')
    df_w = df.resample('W-FRI', on='date').last().dropna()
    ax.plot(df_w.index, df_w['close'], label="周线收盘价",
            color='orange', linestyle='--')
    ax.legend()
    ax.set_title(f"{stock_name} ({stock_code}) 决策图")
    ax.set_xlabel("日期")
    ax.set_ylabel("价格")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{stock_code}_{stock_name}_decision.png")
    plt.close()

# ==============================
# 主选股逻辑
# ==============================
def haili_style_selection(current_positions=None):
    """
    海力风格选股函数
    
    Args:
        current_positions (dict, optional): 当前持仓字典，格式为 {股票代码: 持仓百分比}
                                          如果为None，会尝试读取 current_positions.csv 文件
    """
    # 处理当前持仓数据
    if current_positions is None:
        current_positions = {}
        csv_path = "current_positions.csv"
        if os.path.exists(csv_path):
            try:
                df_pos = pd.read_csv(csv_path)
                # 支持中文和英文列名
                if '代码' in df_pos.columns and '当前仓位(%)' in df_pos.columns:
                    for _, row in df_pos.iterrows():
                        current_positions[str(row['代码']).zfill(6)] = row['当前仓位(%)']
                elif 'code' in df_pos.columns and 'current_pos' in df_pos.columns:
                    for _, row in df_pos.iterrows():
                        current_positions[str(row['code']).zfill(6)] = row['current_pos']
                print(f"成功读取当前持仓文件，共 {len(current_positions)} 只股票")
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

            # 5日内无连续两个跌停
            if not check_no_consecutive_limit_down(df):
                continue

            # 技术面条件
            if not check_tech_conditions(df):
                continue

            # 周线连阳检测
            if not check_weekly_positive(df):
                continue

            # 连续3日资金净流入
            if not check_funds_inflow(code):
                continue

            # 计算相关指标
            current_pos = current_positions.get(code, 0)
            
            # 简化的目标仓位计算 (基于市值和技术指标)
            target_pos = min(100, max(0, (100 - market_cap) * 0.5))  # 简化算法
            
            # 计算动作分数
            action_score = 0
            if market_cap <= 80:
                action_score += 2
            if df['close'].iloc[-1] > df['close'].rolling(20).mean().iloc[-1]:
                action_score += 2
            if df['vol'].iloc[-1] > df['vol'].rolling(20).mean().iloc[-1]:
                action_score += 1
                
            # 计算建议
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
            
            # 交易指令
            if current_pos == 0:
                trade_empty = f"买入 {target_pos:.1f}%"
                trade_with_pos = "无持仓"
                trade_specific = "无持仓"
            else:
                trade_empty = "已有持仓"
                trade_with_pos = suggestion
                trade_specific = f"{direction} {adjust_pct:.1f}%"
            
            # 触发理由
            reasons = []
            if market_cap <= 80:
                reasons.append("市值适中")
            if action_score >= 4:
                reasons.append("技术面强势")
            trigger_reason = ", ".join(reasons) if reasons else "符合筛选条件"

            # 绘图
            plot_decision_chart(code, name, df)
            results.append({
                "代码": code,
                "名称": name,
                "总市值(亿)": market_cap,
                "action_score": action_score,
                "目标仓位(%)": target_pos,
                "当前仓位(%)": current_pos,
                "交易指令_空仓": trade_empty,
                "交易指令_有仓": trade_with_pos,
                "交易指令_有仓_具体": trade_specific,
                "建议下单方向": direction,
                "建议调整(%)": adjust_pct,
                "触发理由": trigger_reason
            })

        except Exception:
            continue

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"筛选完成，共 {len(results)} 只股票，已输出到 {OUTPUT_CSV}")

if __name__ == "__main__":
    haili_style_selection()