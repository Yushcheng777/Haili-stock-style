import os
import re

def replace_in_file(filepath, patterns):
    """Replace all patterns in a single file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = content
    for pat, repl in patterns:
        new_content = re.sub(pat, repl, new_content, flags=re.IGNORECASE)

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Replaced in: {filepath}")

def should_replace(filename):
    """Only replace in .py files."""
    return filename.endswith('.py')

def main(root='.'):
    # 1. akshare 替换为 tushare（import、调用等，大小写不敏感）
    patterns = [
        (r'\bakshare\b', 'tushare'),    # import akshare as ak
        (r'\bAKSHARE\b', 'tushare'),    # AKSHARE as tushare
        # 你可以根据需要添加更多模式，如 import akshare as ak -> import tushare as ak
    ]

    for dirpath, dirs, files in os.walk(root):
        for filename in files:
            if should_replace(filename):
                filepath = os.path.join(dirpath, filename)
                replace_in_file(filepath, patterns)

if __name__ == '__main__':
    main()
