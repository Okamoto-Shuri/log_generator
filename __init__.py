"""
Enhanced Log Generator v3.0

機械学習用の高品質な合成ログデータセット生成ツール

使用例:
    # コマンドラインから
    python cli.py generate --events 10000
    
    # Pythonコードから
    from log_generator.core import GeneratorConfig
    from log_generator.main_generator import EnhancedLogGenerator
    
    config = GeneratorConfig(total_events=1000)
    generator = EnhancedLogGenerator(config)
    generator.run()
"""

# バージョン情報
from core import VERSION, BUILD_DATE

__version__ = VERSION
__build_date__ = BUILD_DATE

# 主要なクラスをルートレベルでインポート可能に
from core import (
    GeneratorConfig,
    ScenarioMetadata,
    SCENARIO_META
)

from main_generator import EnhancedLogGenerator

__all__ = [
    '__version__',
    '__build_date__',
    'GeneratorConfig',
    'ScenarioMetadata',
    'SCENARIO_META',
    'EnhancedLogGenerator',
]