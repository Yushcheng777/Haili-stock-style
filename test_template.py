#!/usr/bin/env python3
"""
Simple template test for the Jinja template
"""

import json
import os
from datetime import datetime

def simple_template_render(template_path, context):
    """Simple template rendering without Jinja2."""
    
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    # Very basic template rendering for testing
    # This is just for verification, real Jinja2 would be used in production
    
    # Replace basic variables
    replacements = {
        '{{ "%.2f%%" | format((performance.annualized_return or 0) * 100) }}': f"{(context.get('performance', {}).get('annualized_return', 0) * 100):.2f}%",
        '{{ "%.3f" | format(performance.sharpe or 0) }}': f"{context.get('performance', {}).get('sharpe', 0):.3f}",
        '{{ "%.3f" | format(performance.sortino) if performance.sortino is not none else "N/A" }}': f"{context.get('performance', {}).get('sortino', 0):.3f}" if context.get('performance', {}).get('sortino') is not None else "N/A",
        '{{ "%.2f%%" | format((performance.max_drawdown or 0) * 100) }}': f"{(context.get('performance', {}).get('max_drawdown', 0) * 100):.2f}%",
        '{{ "%.2f%%" | format((performance.volatility or 0) * 100) }}': f"{(context.get('performance', {}).get('volatility', 0) * 100):.2f}%",
        '{{ "%.2f%%" | format((performance.win_rate or 0) * 100) }}': f"{(context.get('performance', {}).get('win_rate', 0) * 100):.2f}%",
        '{{ performance.total_trades or 0 }}': str(context.get('performance', {}).get('total_trades', 0)),
        '{% if charts.rolling_sharpe_png -%}': '' if context.get('charts', {}).get('rolling_sharpe_png') else '<!--',
        '{% else -%}': '-->' if context.get('charts', {}).get('rolling_sharpe_png') else '',
        '{% endif %}': '',
        '{{ charts.rolling_sharpe_png }}': context.get('charts', {}).get('rolling_sharpe_png', ''),
        '{{ rolling_window_days | default(60) }}': str(context.get('rolling_window_days', 60)),
        '{{ risk_free_rate | default(0) }}': str(context.get('risk_free_rate', 0)),
    }
    
    for placeholder, value in replacements.items():
        template_content = template_content.replace(placeholder, str(value))
    
    return template_content


def test_template():
    """Test the template rendering with sample data."""
    
    # Load test metrics
    with open('/tmp/test_backtest_results/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    # Create context for template
    context = {
        'performance': metrics['performance'],
        'charts': metrics['charts'],
        'rolling_window_days': 60,
        'risk_free_rate': 0.0,
        'backtest_start_date': '2023-01-01',
        'backtest_end_date': '2023-06-30',
        'data_points': 130,
        'report_generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Render template
    template_path = '/home/runner/work/Haili-stock-style/Haili-stock-style/templates/report_template.md.j2'
    report_content = simple_template_render(template_path, context)
    
    # Save rendered report
    output_path = '/tmp/test_backtest_results/test_report.md'
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    print(f"Template test completed. Report saved to: {output_path}")
    
    # Show first few lines
    lines = report_content.split('\n')
    print("\nFirst 20 lines of rendered report:")
    for i, line in enumerate(lines[:20]):
        print(f"{i+1:2d}: {line}")


if __name__ == "__main__":
    test_template()