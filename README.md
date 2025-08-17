# Haili-stock-style

This repository contains `haili_strategy.py`, a stock selection script using akshare with technical and funds filters.

## New API: programmatic usage

`haili_style_selection(current_positions=None)`

- The function accepts an optional dictionary `current_positions` mapping stock code (string, keep leading zeros) to current position percentage (float, 0-100).
- If `current_positions` is provided, the function uses it directly and will provide specific suggestions (建议下单方向, 建议调整(%), 交易指令_有仓_具体) for each stock based on the current holding.
- If `current_positions` is `None`, the function falls back to trying to read `current_positions.csv` in the repository root (if present) or operates without exact current holdings and only outputs generic suggestions.

### CSV format supported

The script supports two CSV column formats for `current_positions.csv`:

- Chinese headers (recommended):

```
代码,当前仓位(%)
000001,20
600000,0
```

- English headers:

```
code,current_pos
000001,20
600000,0
```

Make sure stock codes are strings with leading zeros preserved.

### Usage examples

- Programmatic (library) usage:

```python
from haili_strategy import haili_style_selection
# pass current positions as a dict
haili_style_selection(current_positions={"000001": 20, "600000": 0})
```

- CLI / script usage:

If `current_positions.csv` exists in the project root, running the script will read it and produce concrete suggestions:

```bash
python haili_strategy.py
```

Output: `candidates_haili_style.csv` (UTF-8-sig) with columns including:
- 代码, 名称, 总市值(亿), action_score, 目标仓位(%), 当前仓位(%), 交易指令_空仓, 交易指令_有仓, 交易指令_有仓_具体, 建议下单方向, 建议调整(%), 触发理由

## Tests

A simple pytest test is provided at `tests/test_haili_strategy.py`. The test monkeypatches `akshare` calls used by the script to provide deterministic data, so the test can run offline.

Run tests with:

```bash
pip install -r requirements.txt   # ensure pytest and pandas installed
pytest -q
```

## Example script

An example script `examples/run_with_positions.py` demonstrates how to call the function from other code and pass a `current_positions` dictionary.
