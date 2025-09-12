#!/usr/bin/env python3
"""
generate_backtest_report.py

生成回测报告，包括性能指标、滚动夏普比率图表等。
Generate backtest report with performance metrics and rolling Sharpe ratio chart.

用法示例 / Usage:
    python generate_backtest_report.py --input backtest_results/equity_curve.csv

依赖 / Dependencies:
    pip install pandas numpy matplotlib jinja2

说明 / Description:
    - 读取股票回测的权益曲线数据
    - 计算基本性能指标：年化收益率、夏普比率、最大回撤等
    - 生成滚动夏普比率图表
    - 使用Jinja2模板生成Markdown报告
    - Read equity curve data from backtest
    - Calculate basic performance metrics: annualized return, Sharpe ratio, max drawdown, etc.
    - Generate rolling Sharpe ratio chart
    - Use Jinja2 template to generate Markdown report
"""

import os
import sys
import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from jinja2 import Template, FileSystemLoader, Environment

def load_equity_curve(csv_path):
    """
    加载权益曲线数据 / Load equity curve data
    Expected columns: Date, PortfolioValue (or similar)
    """
    print(f"[report] Loading equity curve from {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Equity curve file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Normalize column names - look for common variations
    if 'Date' not in df.columns:
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            df.rename(columns={date_cols[0]: 'Date'}, inplace=True)
    
    # Look for portfolio value column
    value_cols = ['PortfolioValue', 'Portfolio_Value', 'Value', 'Total_Value', 'Total']
    portfolio_col = None
    for col in value_cols:
        if col in df.columns:
            portfolio_col = col
            break
    
    if portfolio_col is None:
        # Use the second column as portfolio value if not found
        if len(df.columns) >= 2:
            portfolio_col = df.columns[1]
            df.rename(columns={portfolio_col: 'PortfolioValue'}, inplace=True)
        else:
            raise ValueError("Could not find portfolio value column")
    elif portfolio_col != 'PortfolioValue':
        df.rename(columns={portfolio_col: 'PortfolioValue'}, inplace=True)
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"[report] Loaded {len(df)} data points from {df['Date'].min()} to {df['Date'].max()}")
    return df

def calculate_metrics(df):
    """
    计算基本性能指标 / Calculate basic performance metrics
    """
    print(f"[report] Calculating performance metrics")
    
    # Calculate returns
    df['Daily_Return'] = df['PortfolioValue'].pct_change()
    daily_returns = df['Daily_Return'].dropna()
    
    if len(daily_returns) == 0:
        print(f"[report] Warning: No valid returns found")
        return {}
    
    # Basic metrics
    total_return = (df['PortfolioValue'].iloc[-1] / df['PortfolioValue'].iloc[0]) - 1
    
    # Annualized return
    trading_days = len(df)
    if trading_days > 1:
        years = trading_days / 252  # Assume 252 trading days per year
        annualized_return = (1 + total_return) ** (1/years) - 1
    else:
        annualized_return = 0
    
    # Volatility (annualized)
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        volatility_annualized = daily_returns.std() * np.sqrt(252)
    else:
        volatility_annualized = None
    print(f"[report] Computing annualized volatility: {volatility_annualized}")
    
    # Sharpe ratio (assuming risk-free rate = 0)
    if daily_returns.std() > 0:
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = None
    
    # Maximum drawdown
    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Calmar ratio
    if max_drawdown is not None and abs(max_drawdown) > 0:
        calmar_ratio = annualized_return / abs(max_drawdown)
    else:
        calmar_ratio = None
    print(f"[report] Computing Calmar ratio: {calmar_ratio}")
    
    # Sortino ratio (downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino_ratio = daily_returns.mean() / downside_returns.std() * np.sqrt(252)
    else:
        sortino_ratio = None
    
    # Win rate
    win_rate = (daily_returns > 0).mean() if len(daily_returns) > 0 else 0
    
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility_annualized': volatility_annualized,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar': calmar_ratio,
        'win_rate': win_rate,
        'total_trades': len(daily_returns),
        'start_date': df['Date'].iloc[0].strftime('%Y-%m-%d'),
        'end_date': df['Date'].iloc[-1].strftime('%Y-%m-%d'),
        'trading_days': trading_days
    }
    
    return metrics

def generate_rolling_sharpe_chart(df, output_dir='backtest_results'):
    """
    生成滚动夏普比率图表 / Generate rolling Sharpe ratio chart
    """
    print(f"[report] Generating rolling Sharpe ratio chart")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    charts_dir = os.path.join(output_dir, 'charts')
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir)
    
    # Calculate rolling Sharpe ratio (30-day window)
    df['Daily_Return'] = df['PortfolioValue'].pct_change()
    rolling_mean = df['Daily_Return'].rolling(window=30, min_periods=20).mean()
    rolling_std = df['Daily_Return'].rolling(window=30, min_periods=20).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], rolling_sharpe, linewidth=1.5, color='steelblue')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=1, color='green', linestyle='--', alpha=0.7)
    
    plt.title('滚动夏普比率 (30天) / Rolling Sharpe Ratio (30-day)', fontsize=14, fontweight='bold')
    plt.xlabel('日期 / Date')
    plt.ylabel('夏普比率 / Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    chart_path = os.path.join(charts_dir, 'rolling_sharpe.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[report] Rolling Sharpe chart saved to {chart_path}")
    return chart_path

def save_metrics_json(metrics, output_dir='backtest_results'):
    """
    保存指标到JSON文件 / Save metrics to JSON file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Structure the metrics data
    metrics_data = {
        'performance': {
            'total_return': metrics.get('total_return'),
            'annualized_return': metrics.get('annualized_return'),
            'volatility_annualized': metrics.get('volatility_annualized'),
            'sharpe_ratio': metrics.get('sharpe_ratio'),
            'sortino_ratio': metrics.get('sortino_ratio'),
            'max_drawdown': metrics.get('max_drawdown'),
            'calmar': metrics.get('calmar'),
            'win_rate': metrics.get('win_rate')
        },
        'summary': {
            'start_date': metrics.get('start_date'),
            'end_date': metrics.get('end_date'),
            'trading_days': metrics.get('trading_days'),
            'total_trades': metrics.get('total_trades')
        },
        'generated_at': datetime.now().isoformat()
    }
    
    json_path = os.path.join(output_dir, 'metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    
    print(f"[report] Metrics saved to {json_path}")
    return json_path

def generate_markdown_report(metrics, chart_path, output_dir='backtest_results', html_path=None, pdf_path=None):
    """
    使用Jinja2模板生成Markdown报告 / Generate Markdown report using Jinja2 template
    """
    print(f"[report] Generating Markdown report")
    
    # Look for template
    template_paths = [
        'templates/report_template.md.j2',
        'data/templates/report_template.md.j2',
        'report_template.md.j2'
    ]
    
    template_path = None
    for path in template_paths:
        if os.path.exists(path):
            template_path = path
            break
    
    if template_path is None:
        # Create a basic template
        template_content = """# 回测报告 / Backtest Report

## 基础信息 / Basic Information
- **开始日期 / Start Date**: {{ summary.start_date }}
- **结束日期 / End Date**: {{ summary.end_date }}
- **交易天数 / Trading Days**: {{ summary.trading_days }}

## 核心指标 / Core Metrics

| 指标 / Metric | 数值 / Value |
|---------------|--------------|
| 总收益率 / Total Return | {{ "%.2f%%" | format(performance.total_return * 100) if performance.total_return is not none else "N/A" }} |
| 年化收益率 / Annualized Return | {{ "%.2f%%" | format(performance.annualized_return * 100) if performance.annualized_return is not none else "N/A" }} |
| 年化波动率 / Annualized Volatility | {{ "%.2f%%" | format(performance.volatility_annualized * 100) if performance.volatility_annualized is not none else "N/A" }} |
| 夏普比率 / Sharpe Ratio | {{ "%.3f" | format(performance.sharpe_ratio) if performance.sharpe_ratio is not none else "N/A" }} |
| Sortino 比率 / Sortino Ratio | {{ "%.3f" | format(performance.sortino_ratio) if performance.sortino_ratio is not none else "N/A" }} |
| 最大回撤 / Max Drawdown | {{ "%.2f%%" | format(performance.max_drawdown * 100) if performance.max_drawdown is not none else "N/A" }} |
| Calmar 比率 / Calmar Ratio | {{ "%.3f" | format(performance.calmar) if performance.calmar is not none else "N/A" }} |
| 胜率 / Win Rate | {{ "%.2f%%" | format(performance.win_rate * 100) if performance.win_rate is not none else "N/A" }} |

## 图表 / Charts

### 滚动夏普比率 / Rolling Sharpe Ratio
![Rolling Sharpe]({{ chart_path }})

{% if exports %}
## 导出 / Exports
{% for export in exports %}
- **{{ export.type }}**: {{ export.path }}
{% endfor %}
{% endif %}

---
*报告生成时间 / Report Generated*: {{ generated_at }}
"""
        
        # Create templates directory
        template_dir = 'templates'
        if not os.path.exists(template_dir):
            os.makedirs(template_dir)
        
        template_path = os.path.join(template_dir, 'report_template.md.j2')
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        print(f"[report] Created default template at {template_path}")
    
    # Load template
    template_dir = os.path.dirname(template_path)
    template_name = os.path.basename(template_path)
    
    env = Environment(loader=FileSystemLoader(template_dir if template_dir else '.'))
    template = env.get_template(template_name)
    
    # Prepare template context
    context = {
        'performance': metrics,
        'summary': {
            'start_date': metrics.get('start_date'),
            'end_date': metrics.get('end_date'),
            'trading_days': metrics.get('trading_days'),
            'total_trades': metrics.get('total_trades')
        },
        'chart_path': os.path.relpath(chart_path, output_dir) if chart_path else None,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Check for export files and add to context
    exports = []
    
    # Use provided paths if available, otherwise check if files exist
    html_exists = html_path is not None or os.path.exists(os.path.join(output_dir, 'report.html'))
    pdf_exists = pdf_path is not None or os.path.exists(os.path.join(output_dir, 'report.pdf'))
    
    if html_exists:
        exports.append({'type': 'HTML', 'path': 'report.html'})
    if pdf_exists:
        exports.append({'type': 'PDF', 'path': 'report.pdf'})
    
    if exports:
        context['exports'] = exports
    
    # Render template
    markdown_content = template.render(**context)
    
    # Save Markdown report
    report_path = os.path.join(output_dir, 'report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"[report] Markdown report saved to {report_path}")
    return report_path

def generate_html_report(markdown_path, output_dir='backtest_results'):
    """
    生成HTML报告 / Generate HTML report
    """
    if os.environ.get('EXPORT_HTML', '1') == '0':
        print("[report] HTML export disabled by EXPORT_HTML=0")
        return None
    
    try:
        import markdown
        print("[report] Generating HTML report using markdown library")
        
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        html_content = markdown.markdown(markdown_content, extensions=['tables'])
        
        # Add basic CSS styling
        html_with_style = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>回测报告 / Backtest Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        img {{ max-width: 100%; height: auto; }}
        h1, h2, h3 {{ color: #333; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
        
        html_path = os.path.join(output_dir, 'report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_with_style)
        
        print(f"[report] HTML report saved to {html_path}")
        return html_path
        
    except ImportError:
        print("[report] Markdown library not available, skipping HTML generation")
        return None

def generate_pdf_report(html_path=None, metrics=None, output_dir='backtest_results'):
    """
    生成PDF报告 / Generate PDF report
    """
    if os.environ.get('EXPORT_PDF', '1') == '0':
        print("[report] PDF export disabled by EXPORT_PDF=0")
        return None
    
    pdf_path = os.path.join(output_dir, 'report.pdf')
    
    # Try weasyprint first
    try:
        import weasyprint
        if html_path and os.path.exists(html_path):
            print("[report] Generating PDF using weasyprint")
            weasyprint.HTML(filename=html_path).write_pdf(pdf_path)
            print(f"[report] PDF report saved to {pdf_path}")
            return pdf_path
    except ImportError:
        print("[report] weasyprint not available, trying reportlab")
    except Exception as e:
        print(f"[report] weasyprint failed: {e}, trying reportlab")
    
    # Try reportlab as fallback
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        
        print("[report] Generating PDF using reportlab")
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.darkblue,
            spaceAfter=20
        )
        story.append(Paragraph("回测报告 / Backtest Report", title_style))
        story.append(Spacer(1, 12))
        
        if metrics:
            # Performance metrics table
            story.append(Paragraph("核心指标 / Core Metrics", styles['Heading2']))
            
            table_data = [
                ['指标 / Metric', '数值 / Value'],
                ['总收益率 / Total Return', f"{metrics.get('total_return', 0)*100:.2f}%" if metrics.get('total_return') is not None else "N/A"],
                ['年化收益率 / Annualized Return', f"{metrics.get('annualized_return', 0)*100:.2f}%" if metrics.get('annualized_return') is not None else "N/A"],
                ['年化波动率 / Annualized Volatility', f"{metrics.get('volatility_annualized', 0)*100:.2f}%" if metrics.get('volatility_annualized') is not None else "N/A"],
                ['夏普比率 / Sharpe Ratio', f"{metrics.get('sharpe_ratio'):.3f}" if metrics.get('sharpe_ratio') is not None else "N/A"],
                ['最大回撤 / Max Drawdown', f"{metrics.get('max_drawdown', 0)*100:.2f}%" if metrics.get('max_drawdown') is not None else "N/A"],
                ['Calmar 比率 / Calmar Ratio', f"{metrics.get('calmar'):.3f}" if metrics.get('calmar') is not None else "N/A"],
                ['胜率 / Win Rate', f"{metrics.get('win_rate', 0)*100:.2f}%" if metrics.get('win_rate') is not None else "N/A"]
            ]
            
            table = Table(table_data, colWidths=[3*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
        
        doc.build(story)
        print(f"[report] PDF report saved to {pdf_path}")
        return pdf_path
        
    except ImportError:
        print("[report] reportlab not available, skipping PDF generation")
        return None
    except Exception as e:
        print(f"[report] PDF generation failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate backtest report with performance metrics')
    parser.add_argument('--input', required=True, help='Path to equity curve CSV file')
    parser.add_argument('--output-dir', default='backtest_results', help='Output directory for reports')
    args = parser.parse_args()
    
    try:
        # Load data and calculate metrics
        df = load_equity_curve(args.input)
        metrics = calculate_metrics(df)
        
        # Create output directory
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        # Generate chart
        chart_path = generate_rolling_sharpe_chart(df, args.output_dir)
        
        # Save metrics JSON
        save_metrics_json(metrics, args.output_dir)
        
        # Generate Markdown report first (placeholder)
        markdown_path = generate_markdown_report(metrics, chart_path, args.output_dir)
        
        # Generate HTML from the markdown
        html_path = generate_html_report(markdown_path, args.output_dir)
        
        # Generate PDF
        pdf_path = generate_pdf_report(html_path, metrics, args.output_dir)
        
        # Regenerate Markdown with export info if HTML/PDF were generated
        if html_path or pdf_path:
            markdown_path = generate_markdown_report(metrics, chart_path, args.output_dir, html_path, pdf_path)
        
        print(f"[report] Report generation completed successfully")
        print(f"[report] Markdown: {markdown_path}")
        if html_path:
            print(f"[report] HTML: {html_path}")
        if pdf_path:
            print(f"[report] PDF: {pdf_path}")
            
    except Exception as e:
        print(f"Error generating report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()