#!/usr/bin/env python3
"""
integrated_backtest_cli.py

统一回测CLI入口 / Unified Backtest CLI Entry Point

提供一个维护的单一入口来运行集成策略并在backtest_results/中输出结果。
Provides a single, maintained entry to run the integrated strategy end-to-end 
and emit outputs in backtest_results/.

用法示例 / Usage Examples:
  python scripts/integrated_backtest_cli.py
  python scripts/integrated_backtest_cli.py --codes 000001 000002
  python scripts/integrated_backtest_cli.py --out-dir custom_results --verbose
  python scripts/integrated_backtest_cli.py --backtest-days 120
"""

import os
import sys
import argparse
from datetime import datetime

# Add the project root to Python path to import haili_integrated_strategy
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def cli_print(message, prefix="[cli]"):
    """Print message with CLI prefix for user-friendly logging"""
    print(f"{prefix} {message}")

def main():
    """主CLI入口函数 / Main CLI entry function"""
    parser = argparse.ArgumentParser(
        description="海力风格统一回测CLI / Haili Style Unified Backtest CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 / Examples:
  python scripts/integrated_backtest_cli.py                           # 完整策略流程
  python scripts/integrated_backtest_cli.py --codes 000001 000002     # 快速回测指定股票
  python scripts/integrated_backtest_cli.py --out-dir my_results      # 自定义输出目录
  python scripts/integrated_backtest_cli.py --backtest-days 120       # 自定义回测天数
"""
    )
    
    parser.add_argument(
        "--codes", 
        nargs="+", 
        help="可选的股票代码列表，用于快速回测 / Optional list of stock codes for quick run"
    )
    parser.add_argument(
        "--out-dir", 
        default="backtest_results",
        help="输出目录，默认backtest_results / Output directory, default: backtest_results"
    )
    parser.add_argument(
        "--backtest-days", 
        type=int, 
        default=252,
        help="回测天数，默认252天 / Backtest days, default: 252"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="启用详细日志 / Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Welcome message
    cli_print("海力风格统一回测CLI启动 / Haili Style Unified Backtest CLI Starting")
    cli_print(f"输出目录 / Output directory: {args.out_dir}")
    
    try:
        # Import the integrated strategy module
        try:
            import haili_integrated_strategy
        except ImportError as e:
            cli_print(f"错误：无法导入haili_integrated_strategy模块 / Error: Cannot import haili_integrated_strategy module: {e}", "[错误/Error]")
            cli_print("请确保已安装所需依赖：pip install -r requirements.txt", "[错误/Error]")
            cli_print("Please ensure dependencies are installed: pip install -r requirements.txt", "[错误/Error]")
            return 1
        
        # Set the output directory in the integrated strategy module
        original_output_dir = getattr(haili_integrated_strategy, 'BACKTEST_OUTPUT_DIR', 'backtest_results')
        haili_integrated_strategy.BACKTEST_OUTPUT_DIR = args.out_dir
        
        # Create output directory
        os.makedirs(args.out_dir, exist_ok=True)
        cli_print(f"输出目录已创建 / Output directory created: {args.out_dir}")
        
        if args.verbose:
            cli_print("详细模式已启用 / Verbose mode enabled")
        
        # Choose execution path
        if args.codes:
            # Quick run path for specific codes
            cli_print(f"快速回测模式，股票代码 / Quick run mode for codes: {', '.join(args.codes)}")
            cli_print("调用quick_run函数 / Calling quick_run function")
            
            result = haili_integrated_strategy.quick_run(
                stock_codes=args.codes, 
                backtest_days=args.backtest_days
            )
            
            if result:
                cli_print(f"快速回测完成，处理了 {len(result)} 只股票 / Quick run completed, processed {len(result)} stocks")
            else:
                cli_print("快速回测完成 / Quick run completed")
                
        else:
            # Full integrated strategy
            cli_print("完整集成策略模式 / Full integrated strategy mode")
            cli_print("调用run_integrated_strategy函数 / Calling run_integrated_strategy function")
            
            selected_stocks, backtest_results = haili_integrated_strategy.run_integrated_strategy(
                current_positions=None,
                backtest_days=args.backtest_days
            )
            
            if not selected_stocks.empty:
                cli_print(f"策略执行完成，选出 {len(selected_stocks)} 只股票 / Strategy completed, selected {len(selected_stocks)} stocks")
            
            if backtest_results:
                cli_print(f"回测完成，成功回测 {len(backtest_results)} 只股票 / Backtest completed, successfully tested {len(backtest_results)} stocks")
            else:
                cli_print("未生成回测结果 / No backtest results generated")
        
        # Restore original output directory
        haili_integrated_strategy.BACKTEST_OUTPUT_DIR = original_output_dir
        
        cli_print(f"所有结果已保存到 / All results saved to: {args.out_dir}")
        cli_print("回测CLI执行成功完成 / Backtest CLI execution completed successfully")
        
        return 0
        
    except Exception as e:
        cli_print(f"执行过程中发生错误 / Error during execution: {str(e)}", "[错误/Error]")
        if args.verbose:
            import traceback
            cli_print("详细错误信息 / Detailed error:", "[错误/Error]")
            traceback.print_exc()
        
        cli_print("程序将退出，但这不会影响其他处理 / Program will exit, but this won't affect other processing", "[错误/Error]")
        return 0  # Exit 0 even on errors as per requirements - be resilient

if __name__ == "__main__":
    sys.exit(main())