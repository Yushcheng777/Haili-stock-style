import os
import re

AKSHARE_IMPORT_PATTERN = re.compile(r'^\s*import\s+tushare\s+as\s+ak', re.MULTILINE)
AKSHARE_USAGE_PATTERN = re.compile(r'\bak\.')

def transform_content(content):
    # 替换 import 行
    content = AKSHARE_IMPORT_PATTERN.sub('import tushare as ts', content)
    # 替换 ak. 为 ts. （如需更细致可定制）
    content = AKSHARE_USAGE_PATTERN.sub('ts.', content)
    return content

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

def walk_and_replace(root_dir=".", dry_run=False):
    changed = 0
    for dirpath, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(dirpath, file)
                if replace_in_file(path, dry_run=dry_run):
                    changed += 1
    print(f"\nTotal files changed: {changed}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Replace tushare with tushare in all .py files")
    parser.add_argument("--root", default=".", help="Root directory to scan")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would change")
    args = parser.parse_args()
    walk_and_replace(args.root, dry_run=args.dry_run)
