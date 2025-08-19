from haili_strategy import haili_style_selection

if __name__ == '__main__':
    # 示例：以字典方式传入当前持仓（代码必须为字符串，保留前导零）
    positions = {
        '000001': 20.0,
        '600000': 0.0
    }
    haili_style_selection(current_positions=positions)
    print('运行完成，结果写入 candidates_haili_style.csv')