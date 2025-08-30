import pandas as pd
import os
import pytest
import haili_strategy

# Helper to create a simple daily k-line DataFrame
def make_daily_df():
    dates = pd.date_range(end=pd.Timestamp.today(), periods=70, freq='B')
    close = (pd.Series(range(len(dates))) * 0.1 + 10).values
    vol = (pd.Series([1000 + i for i in range(len(dates))])).values
    df = pd.DataFrame({'date': dates, 'close': close, 'vol': vol})
    return df

# Helper for fund flow DataFrame
def make_fund_flow():
    df = pd.DataFrame({'主力净流入': [100, 50, 10, -20]})
    return df

def test_haili_style_selection_with_mock(monkeypatch, tmp_path):
    # Mock tushare functions used in haili_strategy
    monkeypatch.setattr(haili_strategy.ak, 'stock_info_a_code_name', lambda: pd.DataFrame([{'code': '000001', 'name': 'TestCo'}]))

    def mock_individual_info(code):
        return pd.DataFrame({'item': ['总市值'], 'value': ['60']})
    monkeypatch.setattr(haili_strategy.ak, 'stock_individual_info_em', lambda code: mock_individual_info(code))

    monkeypatch.setattr(haili_strategy.ak, 'stock_board_concept_name_em', lambda: pd.DataFrame({'板块名称': ['半导体']}))

    monkeypatch.setattr(haili_strategy.ak, 'stock_zh_a_daily', lambda symbol: make_daily_df())
    monkeypatch.setattr(haili_strategy.ak, 'stock_individual_fund_flow', lambda stock: make_fund_flow())

    # Override all the check functions to return True to simplify testing
    monkeypatch.setattr(haili_strategy, 'check_no_consecutive_limit_down', lambda df: True)
    monkeypatch.setattr(haili_strategy, 'check_tech_conditions', lambda df: True)
    monkeypatch.setattr(haili_strategy, 'check_weekly_positive', lambda df: True)
    monkeypatch.setattr(haili_strategy, 'check_funds_inflow', lambda code: True)
    
    # Mock the plotting function to do nothing
    monkeypatch.setattr(haili_strategy, 'plot_decision_chart', lambda code, name, df: None)

    # Override output path
    out = tmp_path / 'out.csv'
    haili_strategy.OUTPUT_CSV = str(out)

    # Call with provided current_positions dict
    haili_strategy.haili_style_selection(current_positions={'000001': 20})

    assert out.exists(), 'Output CSV was not created'
    df = pd.read_csv(out)
    assert len(df) > 0, 'No stocks were found in the output'
    
    # Check expected columns exist
    for col in ['代码', '名称', '目标仓位(%)', '当前仓位(%)', '建议下单方向', '建议调整(%)']:
        assert col in df.columns, f'Column {col} not found in output'
    # Ensure current position was recorded
    assert df.loc[0, '当前仓位(%)'] == 20

def test_csv_reading_functionality(monkeypatch, tmp_path):
    """Test CSV reading functionality with Chinese headers"""
    # Create a test CSV file
    csv_content = """代码,当前仓位(%)
000001,20
600000,0"""
    csv_file = tmp_path / 'current_positions.csv'
    csv_file.write_text(csv_content, encoding='utf-8')
    
    # Change to the temp directory
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Mock tushare functions
        monkeypatch.setattr(haili_strategy.ak, 'stock_info_a_code_name', lambda: pd.DataFrame([{'code': '000001', 'name': 'TestCo'}]))
        def mock_individual_info(code):
            return pd.DataFrame({'item': ['总市值'], 'value': ['60']})
        monkeypatch.setattr(haili_strategy.ak, 'stock_individual_info_em', lambda code: mock_individual_info(code))
        monkeypatch.setattr(haili_strategy.ak, 'stock_board_concept_name_em', lambda: pd.DataFrame({'板块名称': ['半导体']}))
        monkeypatch.setattr(haili_strategy.ak, 'stock_zh_a_daily', lambda symbol: make_daily_df())
        monkeypatch.setattr(haili_strategy.ak, 'stock_individual_fund_flow', lambda stock: make_fund_flow())
        
        # Override all the check functions to return True
        monkeypatch.setattr(haili_strategy, 'check_no_consecutive_limit_down', lambda df: True)
        monkeypatch.setattr(haili_strategy, 'check_tech_conditions', lambda df: True)
        monkeypatch.setattr(haili_strategy, 'check_weekly_positive', lambda df: True)
        monkeypatch.setattr(haili_strategy, 'check_funds_inflow', lambda code: True)
        
        # Mock the plotting function
        monkeypatch.setattr(haili_strategy, 'plot_decision_chart', lambda code, name, df: None)
        
        # Override output path
        out = tmp_path / 'out.csv'
        haili_strategy.OUTPUT_CSV = str(out)
        
        # Call without current_positions - should read from CSV
        haili_strategy.haili_style_selection(current_positions=None)
        
        assert out.exists(), 'Output CSV was not created'
        df = pd.read_csv(out)
        assert len(df) > 0, 'No stocks were found in the output'
        assert df.loc[0, '当前仓位(%)'] == 20, 'CSV position not read correctly'
        
    finally:
        os.chdir(original_cwd)