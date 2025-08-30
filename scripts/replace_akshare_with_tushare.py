import os
import re
 auto-replace-akshare-1756546566
AKSHARE_IMPORT_PATTERN = re.compile(r'^\s*import\s+tushare\s+as\s+ak', re.MULTILINE)
AKSHARE_USAGE_PATTERN = re.compile(r'\bak\.')
def replace_in_file(filepath, patterns):
    """Replace all patterns in a single file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
 main

    new_content = content
    for pat, repl in patterns:
        new_content = re.sub(pat, repl, new_content, flags=re.IGNORECASE)
 auto-replace-akshare-1756546566
def replace_in_file(file_path, dry_run=False):
    with open(file_path, encoding="utf-8") as f:
        orig = f.read()
    if "tushare" not in orig:
        return False
    new_content = transform_content(orig)
    if not dry_run:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
    print(f"Replaced tushare with tushare in: {file_path}")
    return True

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Replaced in: {filepath}")
 main

def should_replace(filename):
    """Only replace in .py files."""
    return filename.endswith('.py')

 auto-replace-akshare-1756546566
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Replace tushare with tushare in all .py files")
    parser.add_argument("--root", default=".", help="Root directory to scan")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would change")
    args = parser.parse_args()
    walk_and_replace(args.root, dry_run=args.dry_run)

def main(root='.'):
    # 1. tushare 替换为 tushare（import、调用等，大小写不敏感）
    patterns = [
        (r'\btushare\b', 'tushare'),    # import tushare as ak
        (r'\bAKSHARE\b', 'tushare'),    # AKSHARE as tushare
        # 你可以根据需要添加更多模式，如 import tushare as ak -> import tushare as ak
    ]

    for dirpath, dirs, files in os.walk(root):
        for filename in files:
            if should_replace(filename):
                filepath = os.path.join(dirpath, filename)
                replace_in_file(filepath, patterns)

if __name__ == '__main__':
    main()
 main
