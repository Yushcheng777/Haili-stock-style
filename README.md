# Haili Integrated Strategy (Haili Stock Style)

> 量化选股 + 回测 + 指标评估一体化策略框架  
> A unified pipeline for multi-indicator stock selection, signal generation, backtesting, and performance evaluation.

---

## 1. 概述 (Overview)
`haili_integrated_strategy.py` 提供了一站式量化研究与验证流程：从多因子/多技术指标筛选股票，到生成买卖信号、执行回测、计算风险收益指标、输出可视化与结构化结果，旨在帮助快速验证策略有效性并支持迭代优化。

核心目标：
- 降低“零散脚本 + 手工拼接”成本
- 使策略逻辑透明、可扩展、可比较
- 为后续实时或半实时执行奠定结构基础

---

## 2. 核心特性 (Key Features)
- 多技术指标融合：MACD、RSI、布林带、均线体系（可扩展）
- 条件过滤选股：可设定指标阈值 / 组合打分
- 统一流水线：选股 → 回测 → 绩效汇总 → 输出
- 买卖信号：基于趋势 + 动量 + 波动过滤（可自定义）
- 回测评估：年化收益、最大回撤、胜率、夏普、波动率、交易次数
- 批量回测：支持多标的迭代处理并排名
- 文件归档：图表 / JSON 报告 / CSV 指标
- 内存衔接：尽量减少中间冗余 I/O
- 模块化：便于新增指标 / 风控 / 资金管理
- 可迁移：后续可接入实时行情或交易执行模块

---

## 3. 快速开始 (Quick Start)

### 3.1 获取与安装
```bash
git clone https://github.com/Yushcheng777/Haili-stock-style.git
cd Haili-stock-style
pip install -r requirements.txt
```

### 3.2 推荐：使用统一CLI（新版本）
```bash
# 完整策略流程（推荐）
python scripts/integrated_backtest_cli.py

# 快速回测指定股票
python scripts/integrated_backtest_cli.py --codes 000001 000002 600036

# 自定义输出目录和回测天数
python scripts/integrated_backtest_cli.py --out-dir my_results --backtest-days 120

# 查看所有选项
python scripts/integrated_backtest_cli.py --help
```

### 3.3 备选：直接调用集成策略模块
```bash
python haili_integrated_strategy.py
```
运行后将：
1. 载入或获取数据
2. 计算技术指标
3. 筛选候选股票
4. 逐一回测
5. 生成信号、绩效与图表
6. 输出汇总文件

### 3.4 在代码中调用（示例）
```python
from haili_integrated_strategy import run_integrated_strategy, quick_run

# 完整集成流程
selected, perf_df = run_integrated_strategy()

# 对指定股票快速回测（跳过选股阶段）
quick_run(['000001', '000002', '600036'])
```

### 3.5 ⚠️ 弃用通知 (Deprecation Notice)
**haili_backtest.py 已被弃用** - `haili_backtest.py` 脚本已被弃用，请改用新的统一CLI：`scripts/integrated_backtest_cli.py`。虽然旧脚本暂时仍可使用（会自动重定向），但建议迁移到新的CLI以获得更好的功能和维护支持。

**haili_backtest.py is deprecated** - The `haili_backtest.py` script has been deprecated. Please use the new unified CLI: `scripts/integrated_backtest_cli.py`. While the old script still works temporarily (with automatic redirect), we recommend migrating to the new CLI for better functionality and maintenance support.

## 4. CLI 用法 (CLI Usage)

### 4.1 统一回测CLI - scripts/integrated_backtest_cli.py
推荐的回测入口，提供用户友好的命令行界面：

```bash
# 基本用法：运行完整策略流程
python scripts/integrated_backtest_cli.py

# 快速回测指定股票代码
python scripts/integrated_backtest_cli.py --codes 000001 000002 600036

# 自定义输出目录和回测参数
python scripts/integrated_backtest_cli.py \
  --out-dir custom_results \
  --backtest-days 120 \
  --verbose

# 查看完整选项
python scripts/integrated_backtest_cli.py --help
```

**CLI选项说明：**
- `--codes`: 可选股票代码列表，用于快速回测指定股票
- `--out-dir`: 输出目录，默认为 `backtest_results`
- `--backtest-days`: 回测天数，默认252天（一年）
- `--verbose`: 启用详细日志输出

**默认输出位置：** `backtest_results/` 目录（可通过 `--out-dir` 自定义）

---

## 5. 主要函数 (API 概览)
| 函数 | 功能 | 返回 |
|------|------|------|
| `run_integrated_strategy()` | 执行选股 + 回测全流程 | (selected_stocks, performance_df) |
| `quick_run(code_list)` | 直接回测给定股票列表 | None（生成输出文件） |
| `calculate_indicators(df)` | 计算指标（MACD/RSI/均线/布林等） | 增强后的 DataFrame |
| `generate_signals(df)` | 根据指标组合生成信号 | 含 `signal` 列的 DataFrame |
| `backtest_single(df)` | 单标的回测与绩效计算 | dict（绩效 & 交易记录） |

> 说明：函数名称需与当前实际代码保持一致；若不一致请调整脚本或更新本文档。

---

## 6. 输出内容 (Outputs)

运行后默认输出目录（示例）：
```
backtest_results/
  summary_metrics.csv        # 汇总绩效排名
  selected_stocks.csv        # 通过筛选的标的
  charts/
    000001_price_signals.png
    000001_indicator_panel.png
    000001_equity_curve.png
  reports/
    000001_report.json       # 可选：结构化单股回测结果
  logs/                      # 可选：运行日志
```

### 6.1 汇总指标字段 (summary_metrics.csv)
| 字段 | 含义 |
|------|------|
| code | 股票代码 |
| name | 股票名称（若可获取） |
| annual_return | 年化收益 |
| sharpe | 夏普比率 |
| max_drawdown | 最大回撤 |
| win_rate | 胜率 |
| volatility | 年化波动率 |
| trades | 交易回合数 |

### 5.2 示例信号逻辑（可自定义）
- 买入：`MACD 金叉 AND RSI > 50 AND 收盘价上穿布林中轨`
- 卖出：`MACD 死叉 OR RSI < 40 OR 收盘价跌破短期均线`
> 建议将规则抽象配置化（如 JSON / YAML），便于快速切换策略版本。

---

## 6. 绩效指标说明 (Metrics)
| 指标 | 说明 |
|------|------|
| Annual Return | 回测总收益折算年化 |
| Sharpe Ratio | (超额收益均值 / 收益标准差) × 年化因子 |
| Max Drawdown | 资金曲线最大峰谷跌幅 |
| Win Rate | 盈利交易数 / 总交易数 |
| Volatility | 收益序列年化标准差 |
| Trade Count | 完成的交易回合 |
| Equity Curve | 随时间的资金净值轨迹 |

---

## 7. 配置文件 (config.json 示例)
若脚本支持配置加载，可使用：
```json
{
  "start_date": "2023-01-01",
  "end_date": "2024-08-30",
  "universe": ["000001", "000002", "600036"],
  "initial_capital": 1000000,
  "commission": 0.0005,
  "slippage": 0.001,
  "position_mode": "equal_weight",
  "risk_control": {
    "max_position_per_stock": 0.2,
    "max_total_positions": 10,
    "stop_loss_pct": 0.08
  }
}
```
> 尚未实现的字段可逐步补齐，并在实现后更新文档。

---

## 8. 策略执行流程 (Workflow)
```
数据获取 → 指标计算 → 条件筛选/打分 → 候选池生成
        → 逐标的信号生成 → 回测撮合模拟
        → 资金曲线 & 风险收益指标 → 汇总排名输出
```

---

## 9. 扩展方向 (Extensibility)
| 方向 | 建议 |
|------|------|
| 新指标 | 在 `calculate_indicators()` 增加列 |
| 信号框架 | 引入策略类 / 规则表驱动 |
| 多因子 | 建立权重 + 标准化/打分模型 |
| 风控 | 加入动态仓位、止损/止盈、风险预算 |
| 执行层 | 接入实时行情 / 模拟撮合 / 滑点模型 |
| 可视化 | 增加交互式图（Plotly / Bokeh） |
| 报告 | 生成 HTML / Markdown / PDF |

---

## 10. 常见问题 (FAQ)
Q: 为什么部分股票无图表？  
A: 可能未通过筛选或数据缺失；检查 `selected_stocks.csv`。

Q: 夏普比率为 NaN？  
A: 收益方差为 0 或交易次数过少；扩大时间窗口或放宽信号条件。

Q: 如何只测试自选股票？  
A: 使用 `quick_run(['代码1','代码2', ...])`。

Q: 如何快速对比不同指标组合？  
A: 抽象策略参数，循环多组配置写入不同结果目录（如 `experiments/exp_001`）。

---

## 11. Roadmap
- [ ] 指标/信号参数配置化
- [ ] 多线程 / 异步数据与回测调度
- [ ] 多策略并行对比框架
- [ ] 更完善风险控制模块
- [ ] 实时行情适配层
- [ ] 自动化 HTML / 可交互报告
- [ ] 因子暴露 & 归因分析
- [ ] 回测速度优化（向量化 / numba）

---

## 12. Contributing
欢迎提交 PR / Issue：
1. Fork & 创建特性分支  
2. 保持代码风格与注释规范  
3. 若引入新依赖请更新 `requirements.txt`  
4. 附带最小复现实例或测试用例  

---

## 13. License
MIT License（详见 [LICENSE](LICENSE)）

---

## 14. 免责声明 (Disclaimer)
本项目仅用于教育与研究，不构成投资建议。实盘使用请结合自身风险承受能力。  
This project is for educational purposes only and does not constitute investment advice.

---

欢迎提出改进建议！如果需要英文独立精简版，可新增 `README_EN.md`.