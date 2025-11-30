#!/usr/bin/env python3
"""
Enhanced Log Generator v3.0 - CLI Entry Point

コマンドラインインターフェースのエントリーポイント

使用方法:
    python cli.py generate --events 10000
    python cli.py validate output.jsonl
    python cli.py info
    
または:
    python -m log_generator_v3.cli generate --events 10000
"""

import sys

# cliパッケージからmain関数をインポート
from cli import main


if __name__ == "__main__":
    sys.exit(main())