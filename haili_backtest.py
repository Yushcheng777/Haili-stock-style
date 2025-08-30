#!/usr/bin/env python3
"""
haili_backtest.py

生成符合 data/templates/haili_detailed_header.csv 的详细技术指标 CSV 文件。

用法示例：
  python haili_backtest.py --tickers 000001.SZ 600000.SS
  python haili_backtest.py --input-csv data/prices/000001.SZ.csv --ticker 000001.SZ

依赖：
  pip install pandas numpy tushare

说明：
  - 脚本尝试从本地 CSV (if provided) 读取历史行情，若未提供则尝试用 tushare 下载（A 股示例）。
  - 输出文件名： {TICKER}_haili_detailed_{YYYYmmdd_HHMMSS}.csv，放在当前工作目录。
  - 计算的一些信号为简化版本，供回测/记录使用，可根据需要扩展。
"""

import os
import sys
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

try:
    import tushare as ak
except Exception:
    ak = None

HEADER = [
"Ticker","Date","Open","High","Low","Close","AdjClose","Volume","Turnover",
"MA20","MA50","MA200","EMA20","EMA50","EMA200",
"BB_Middle","BB_Upper","BB_Lower","BB_Width","BB_PctB",
"Ichimoku_Tenkan","Ichimoku_Kijun","Ichimoku_SenkouA","Ichimoku_SenkouB","Ichimoku_Chikou",
"MACD_Line","MACD_Signal","MACD_Hist",
"RSI14","Stoch_K","Stoch_D","ROC_20",
"ADX","DI_Plus","DI_Minus",
"ATR14","HV20","Gap_Pct",
"VWAP","OBV","CMF_20",
"AvgVol20","VolumeRatio20",
"Ret1D","Ret5D","Ret20D","RelToSector20D","RelToBenchmark20D",
"GoldenCross_ShortOverLong","DeathCross_ShortUnderLong",
"BB_Breakout_Upper","BB_Breakdown_Lower",
"MACD_AboveZero","MACD_BullCross","MACD_BearCross",
"RSI_BullRange","RSI_BearRange","RSI_Divergence_Flag",
"ATR_Breakout_Flag","Vol_Expansion_Flag",
"Radar_Pass","Tech_Confirm","Buy_Signal","Signal_Light","Radar_Score"
]

def read_price_from_csv(path):
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    # normalize column names
    for c in ['AdjClose','Adj Close','Adj_Close']:
        if c in df.columns and 'AdjClose' not in df.columns:
            df.rename(columns={c:'AdjClose'}, inplace=True)
    return df

def fetch_tushare_a_stock(ticker, start=None, end=None):
    # ticker like '000001.SZ' => tushare requires 'sh000001' or specific function
    # This is a simplified attempt; user may adjust for correct tushare function & symbol
    if ak is None:
        raise RuntimeError("tushare not installed; provide local CSV instead.")
    # Example: use ak.stock_zh_a_hist for A shares (ticker like 000001)
    sym = ticker
    if '.' in ticker:
        sym = ticker.split('.')[0]
    df = ak.stock_zh_a_hist(symbol=sym, period="daily", adjust="qfq")
    df = df.rename(columns={'日期':'Date','开盘':'Open','最高':'High','最低':'Low','收盘':'Close','成交量':'Volume','成交额':'Turnover','收盘复权价':'AdjClose'})
    df['Date'] = pd.to_datetime(df['Date'])
    # keep needed cols
    needed = ['Date','Open','High','Low','Close','AdjClose','Volume','Turnover']
    df = df[needed].sort_values('Date').reset_index(drop=True)
    return df

def sma(series, n):
    return series.rolling(n, min_periods=1).mean()

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def compute_indicators(df, ticker=None, benchmark=None, sector=None):
    df = df.copy()
    # ensure numeric
    for c in ['Open','High','Low','Close','AdjClose','Volume','Turnover']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df['MA20'] = sma(df['Close'], 20)
    df['MA50'] = sma(df['Close'], 50)
    df['MA200'] = sma(df['Close'], 200)
    df['EMA20'] = ema(df['Close'], 20)
    df['EMA50'] = ema(df['Close'], 50)
    df['EMA200'] = ema(df['Close'], 200)

    # Bollinger
    df['BB_Middle'] = sma(df['Close'], 20)
    df['BB_STD'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_STD']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_STD']
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_PctB'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    # Ichimoku (simplified)
    high = df['High']; low = df['Low']; close = df['Close']
    df['Ichimoku_Tenkan'] = (high.rolling(9).max() + low.rolling(9).min()) / 2
    df['Ichimoku_Kijun'] = (high.rolling(26).max() + low.rolling(26).min()) / 2
    df['Ichimoku_SenkouA'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(26)
    df['Ichimoku_SenkouB'] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    df['Ichimoku_Chikou'] = close.shift(-26)

    # MACD
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    df['MACD_Line'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']

    # RSI14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df['RSI14'] = 100 - (100 / (1 + rs))

    # Stochastic %K %D
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df['Stoch_K'] = 100 * (close - low14) / (high14 - low14)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # ROC20
    df['ROC_20'] = close.pct_change(20)

    # ATR14
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14).mean()

    # ADX, DI+ DI- (simplified Wilder smoothing)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * (pd.Series(plus_dm).rolling(14).sum() / tr14)
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).sum() / tr14)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX'] = dx.rolling(14).mean()
    df['DI_Plus'] = plus_di
    df['DI_Minus'] = minus_di

    # HV20 (20-day std dev of daily returns, annualized)
    ret = close.pct_change().replace(0, np.nan)
    df['HV20'] = ret.rolling(20).std() * np.sqrt(252)

    # Gap_Pct
    df['Gap_Pct'] = (df['Open'] - prev_close) / prev_close

    # VWAP (for daily bars: typical price as per-day VWAP)
    typical = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (typical * df['Volume']).cumsum() / df['Volume'].cumsum()  # running VWAP

    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df.at[i, 'Close'] > df.at[i-1, 'Close']:
            obv.append(obv[-1] + df.at[i, 'Volume'])
        elif df.at[i, 'Close'] < df.at[i-1, 'Close']:
            obv.append(obv[-1] - df.at[i, 'Volume'])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    # CMF_20 (Chaikin Money Flow)
    mf = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mf = mf.replace([np.inf, -np.inf], 0).fillna(0)
    df['CMF_20'] = ( (mf * df['Volume']).rolling(20).sum() ) / (df['Volume'].rolling(20).sum())

    df['AvgVol20'] = df['Volume'].rolling(20).mean()
    df['VolumeRatio20'] = df['Volume'] / df['AvgVol20']

    # Returns
    df['Ret1D'] = close.pct_change(1)
    df['Ret5D'] = close.pct_change(5)
    df['Ret20D'] = close.pct_change(20)

    # RelToSector20D / RelToBenchmark20D - requires additional series; compute if provided
    if sector is not None:
        sclose = sector['Close'].reindex(df.index).ffill()
        df['RelToSector20D'] = df['Ret20D'] - sclose.pct_change(20)
    else:
        df['RelToSector20D'] = np.nan
    if benchmark is not None:
        bclose = benchmark['Close'].reindex(df.index).ffill()
        df['RelToBenchmark20D'] = df['Ret20D'] - bclose.pct_change(20)
    else:
        df['RelToBenchmark20D'] = np.nan

    # Cross / breakout flags
    df['GoldenCross_ShortOverLong'] = (df['MA50'] > df['MA200']).astype(int)
    df['DeathCross_ShortUnderLong'] = (df['MA50'] < df['MA200']).astype(int)
    df['BB_Breakout_Upper'] = (df['Close'] > df['BB_Upper']).astype(int)
    df['BB_Breakdown_Lower'] = (df['Close'] < df['BB_Lower']).astype(int)
    df['MACD_AboveZero'] = (df['MACD_Line'] > 0).astype(int)
    df['MACD_BullCross'] = ((df['MACD_Line'] > df['MACD_Signal']) & (df['MACD_Line'].shift(1) <= df['MACD_Signal'].shift(1))).astype(int)
    df['MACD_BearCross'] = ((df['MACD_Line'] < df['MACD_Signal']) & (df['MACD_Line'].shift(1) >= df['MACD_Signal'].shift(1))).astype(int)
    df['RSI_BullRange'] = (df['RSI14'] > 60).astype(int)
    df['RSI_BearRange'] = (df['RSI14'] < 40).astype(int)
    df['RSI_Divergence_Flag'] = 0  # Placeholder: divergence detection requires pattern analysis

    df['ATR_Breakout_Flag'] = (df['Close'] > (df['Close'].shift(1) + df['ATR14'] * 1.5)).astype(int)
    df['Vol_Expansion_Flag'] = (df['Volume'] > df['AvgVol20'] * 2).astype(int)

    # Radar scoring - simple sum of positive signals
    df['Radar_Score'] = (
        df['GoldenCross_ShortOverLong'].fillna(0) * 2 +
        df['MACD_BullCross'].fillna(0) * 2 +
        df['BB_Breakout_Upper'].fillna(0) * 1 +
        df['RSI_BullRange'].fillna(0) * 1 +
        df['Vol_Expansion_Flag'].fillna(0) * 1
    )
    df['Radar_Pass'] = (df['Radar_Score'] >= 3).astype(int)
    df['Tech_Confirm'] = ((df['Radar_Score'] >= 2) & (df['MACD_AboveZero'] == 1)).astype(int)
    df['Buy_Signal'] = ((df['Radar_Pass'] == 1) & (df['ATR_Breakout_Flag'] == 1)).astype(int)

    def signal_light(score):
        if np.isnan(score): return ""
        if score >= 5: return "Green"
        if score >= 3: return "Yellow"
        return "Red"
    df['Signal_Light'] = df['Radar_Score'].apply(signal_light)

    # Keep only header columns in order
    out = pd.DataFrame(index=df.index)
    for col in HEADER:
        if col == 'Ticker':
            out[col] = ticker
        elif col == 'Date':
            out[col] = df['Date'].dt.strftime('%Y-%m-%d')
        else:
            out[col] = df.get(col, np.nan)
    return out

def save_csv(df_out, ticker):
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    fname = f"{ticker}_haili_detailed_{ts}.csv"
    df_out.to_csv(fname, index=False)
    print(f"Saved: {fname}")
    return fname

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', nargs='*', help='tickers to process (space separated)')
    parser.add_argument('--input-csv', help='local CSV file for a single ticker (overrides remote fetch)')
    parser.add_argument('--ticker', help='ticker name when using --input-csv')
    args = parser.parse_args()

    tickers = args.tickers or []
    if args.input_csv:
        if not args.ticker:
            print("When using --input-csv you must also pass --ticker")
            sys.exit(1)
        df = read_price_from_csv(args.input_csv)
        df_out = compute_indicators(df, ticker=args.ticker)
        save_csv(df_out, args.ticker)
        return

    if not tickers:
        print("No tickers provided. Use --tickers or --input-csv")
        sys.exit(1)

    for t in tickers:
        try:
            # prefer local file at data/prices/{t}.csv if exists
            local_path = f"data/prices/{t}.csv"
            if os.path.exists(local_path):
                df = read_price_from_csv(local_path)
            else:
                df = fetch_tushare_a_stock(t)
            df_out = compute_indicators(df, ticker=t)
            save_csv(df_out, t)
        except Exception as e:
            print(f"Error processing {t}: {e}")

if __name__ == "__main__":
    main()