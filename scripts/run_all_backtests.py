#!/usr/bin/env python3
"""
轻量验证脚本：
- 检查关键依赖能否 import
- 打印版本信息
- 支持 --dry-run（仅做检查，不 actually run 回测）
此脚本用于 CI 验证（确保仓库依赖齐全）
"""
import sys
import argparse

REQUIRED = [
    ("tushare", "tushare"),
    ("backtrader", "backtrader"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
]

def check_imports():
    missing = []
    for name, mod in REQUIRED:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", None)
            print(f"OK: {name} imported, version={ver}")
        except Exception as e:
            missing.append((name, str(e)))
    return missing

def main():
    parser = argparse.ArgumentParser(description="Validate environment for Haili-stock-style")
    parser.add_argument("--dry-run", action="store_true", help="Only check imports / versions")
    args = parser.parse_args()

    missing = check_imports()
    if missing:
        print("\nERROR: Some required packages failed to import:")
        for name, err in missing:
            print(f" - {name}: {err}")
        print("\nPlease run: pip install -r requirements.txt")
        sys.exit(2)

    print("\nAll required packages imported successfully.")
    if args.dry_run:
        print("Dry run complete.")
        sys.exit(0)

    # 如果需要可以在这里调用真正的回测入口 (保留为注释或安全导入)
    # from backtests import run_all
    # run_all()
    print("Note: --dry-run 未指定，仓库中没有自动运行回测的默认行为。")

if __name__ == "__main__":
    main()