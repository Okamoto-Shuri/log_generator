"""
Scenario Base Module - Part 3-1 Integration (基底クラス)

シナリオ生成の基底クラス
"""

from abc import ABC, abstractmethod
from typing import Dict, List

# coreパッケージからのインポート
from core import SCENARIO_META, LogRecord


# ==================== シナリオ基底クラス ====================

class ScenarioGenerator(ABC):
    """シナリオ生成の基底クラス"""
    
    def __init__(
        self,
        record_factory: 'LogRecordFactory',
        formatter: 'LogFormatter',
        scenario_code: str
    ):
        """
        Args:
            record_factory: レコード生成ファクトリー
            formatter: ログフォーマッター
            scenario_code: シナリオコード（A-U）
        """
        self.factory = record_factory
        self.fmt = formatter
        self.code = scenario_code
        self.meta = SCENARIO_META[scenario_code]
    
    def get_label(self) -> Dict:
        """シナリオのラベル情報を取得"""
        return {
            "scenario": self.code,
            "root_cause": self.meta.cause,
            "category": self.meta.category,
            "severity": self.meta.severity,
            "impact": self.meta.impact
        }
    
    @abstractmethod
    def generate(self, base_time) -> List[LogRecord]:
        """
        シナリオに応じたログレコードを生成
        
        Args:
            base_time: 基準時刻
            
        Returns:
            ログレコードのリスト
        """
        pass