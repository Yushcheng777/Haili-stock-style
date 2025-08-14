import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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
def haili_style_selection():
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
            conc = ak.stock_concept_detail_ths()
            if not any(kw in " ".join(conc['概念名称']) 
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

            # 绘图
            plot_decision_chart(code, name, df)
            results.append({
                "代码": code,
                "���称": name,
                "总市值(亿)": market_cap
            })

        except Exception:
            continue

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"筛选完成，共 {len(results)} 只股票，已输出到 {OUTPUT_CSV}")

if __name__ == "__main__":
    haili_style_selection()