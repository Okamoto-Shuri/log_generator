"""
Core Statistics Module - Part 3-3 Integration (統計部分)

統計収集とデータセット検証
"""

import json
from typing import Dict, Any, Set
from collections import defaultdict
from datetime import datetime

# core.configからのインポート
from .config import logger, GeneratorConfig, SCENARIO_META


# ==================== 統計管理 ====================

class StatisticsCollector:
    """生成統計を収集・管理するクラス"""
    
    def __init__(self):
        self.scenario_counts: Dict[str, int] = defaultdict(int)
        self.category_counts: Dict[str, int] = defaultdict(int)
        self.severity_counts: Dict[str, int] = defaultdict(int)
        self.total_logs: int = 0
        self.total_events: int = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    def record_scenario(self, scenario_code: str, log_count: int) -> None:
        """シナリオ生成を記録"""
        self.scenario_counts[scenario_code] += 1
        self.total_logs += log_count
        self.total_events += 1
        
        # メタデータから情報を取得
        if scenario_code != "normal":
            meta = SCENARIO_META.get(scenario_code)
            if meta:
                self.category_counts[meta.category] += 1
                self.severity_counts[meta.severity] += 1
    
    def start_timing(self) -> None:
        """計測開始"""
        self.start_time = datetime.now()
    
    def end_timing(self) -> None:
        """計測終了"""
        self.end_time = datetime.now()
    
    def get_elapsed_time(self) -> float:
        """経過時間を秒で取得"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def print_summary(self, config: GeneratorConfig) -> None:
        """統計サマリーを出力"""
        print("\n" + "=" * 70)
        print("=" * 25 + " GENERATION SUMMARY " + "=" * 25)
        print("=" * 70)
        
        # 基本情報
        print(f"\n📊 Basic Statistics:")
        print(f"  Total Events Generated:  {self.total_events:,}")
        print(f"  Total Log Records:       {self.total_logs:,}")
        print(f"  Average Logs per Event:  {self.total_logs / max(self.total_events, 1):.2f}")
        print(f"  Output File:             {config.output_file}")
        
        # パフォーマンス情報
        elapsed = self.get_elapsed_time()
        if elapsed > 0:
            print(f"\n⏱️  Performance:")
            print(f"  Elapsed Time:            {elapsed:.2f} seconds")
            print(f"  Events per Second:       {self.total_events / elapsed:.2f}")
            print(f"  Logs per Second:         {self.total_logs / elapsed:.2f}")
        
        # 正常/異常の比率
        normal_count = self.scenario_counts.get("normal", 0)
        abnormal_count = self.total_events - normal_count
        
        print(f"\n📈 Event Distribution:")
        print(f"  Normal Events:           {normal_count:>6,} ({normal_count/max(self.total_events,1)*100:>5.1f}%)")
        print(f"  Abnormal Events:         {abnormal_count:>6,} ({abnormal_count/max(self.total_events,1)*100:>5.1f}%)")
        
        # カテゴリ別統計
        if self.category_counts:
            print(f"\n🏷️  Abnormal Events by Category:")
            sorted_categories = sorted(
                self.category_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for category, count in sorted_categories:
                pct = count / max(abnormal_count, 1) * 100
                print(f"  {category:<20} {count:>6,} ({pct:>5.1f}% of abnormal)")
        
        # 重大度別統計
        if self.severity_counts:
            print(f"\n⚠️  Abnormal Events by Severity:")
            severity_order = ["critical", "fatal", "error", "warning", "info"]
            for severity in severity_order:
                if severity in self.severity_counts:
                    count = self.severity_counts[severity]
                    pct = count / max(abnormal_count, 1) * 100
                    print(f"  {severity.upper():<20} {count:>6,} ({pct:>5.1f}% of abnormal)")
        
        # シナリオ別統計（上位10件）
        print(f"\n📋 Top 10 Scenario Frequencies:")
        sorted_scenarios = sorted(
            self.scenario_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for idx, (code, count) in enumerate(sorted_scenarios[:10], 1):
            if code == "normal":
                cause = "Normal Operation"
                category = "normal"
            else:
                meta = SCENARIO_META.get(code)
                cause = meta.cause if meta else "unknown"
                category = meta.category if meta else "unknown"
            
            pct = count / max(self.total_events, 1) * 100
            print(f"  {idx:2}. [{code}] {cause:<35} {count:>5,} ({pct:>5.1f}%)")
            print(f"      Category: {category}")
        
        print("\n" + "=" * 70)


# ==================== 検証ユーティリティ ====================

class DatasetValidator:
    """生成されたデータセットを検証するクラス"""
    
    @staticmethod
    def validate_file(file_path: str) -> Dict[str, Any]:
        """
        JSONLファイルを検証
        
        Args:
            file_path: 検証するファイルパス
            
        Returns:
            検証結果の辞書
        """
        logger.info(f"Validating dataset: {file_path}")
        
        results = {
            "valid": True,
            "total_lines": 0,
            "invalid_lines": [],
            "missing_fields": defaultdict(int),
            "unique_traces": set(),
            "unique_correlations": set(),
            "timestamp_errors": 0
        }
        
        required_fields = [
            "timestamp", "service", "host", "level",
            "message", "metrics", "label", "message_vector"
        ]
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                prev_timestamp = None
                
                for line_num, line in enumerate(f, 1):
                    results["total_lines"] += 1
                    
                    try:
                        record = json.loads(line.strip())
                        
                        # 必須フィールドのチェック
                        for field in required_fields:
                            if field not in record:
                                results["missing_fields"][field] += 1
                                results["valid"] = False
                        
                        # trace_idとcorrelation_idの収集
                        if record.get("trace_id"):
                            results["unique_traces"].add(record["trace_id"])
                        if record.get("correlation_id"):
                            results["unique_correlations"].add(record["correlation_id"])
                        
                        # タイムスタンプの順序チェック
                        current_timestamp = record.get("timestamp")
                        if prev_timestamp and current_timestamp:
                            if current_timestamp < prev_timestamp:
                                results["timestamp_errors"] += 1
                        prev_timestamp = current_timestamp
                    
                    except json.JSONDecodeError:
                        results["invalid_lines"].append(line_num)
                        results["valid"] = False
            
            # 統計変換
            results["unique_traces"] = len(results["unique_traces"])
            results["unique_correlations"] = len(results["unique_correlations"])
            results["missing_fields"] = dict(results["missing_fields"])
            
            logger.info(
                f"Validation completed: "
                f"{'PASS' if results['valid'] else 'FAIL'}"
            )
            
            return results
        
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return {"valid": False, "error": "File not found"}
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {"valid": False, "error": str(e)}
    
    @staticmethod
    def print_validation_results(results: Dict) -> None:
        """検証結果を出力"""
        print("\n" + "=" * 70)
        print("=" * 25 + " VALIDATION RESULTS " + "=" * 26)
        print("=" * 70)
        
        if "error" in results:
            print(f"\n❌ Validation Error: {results['error']}")
            return
        
        status = "✅ PASS" if results["valid"] else "❌ FAIL"
        print(f"\nStatus: {status}")
        print(f"Total Lines: {results['total_lines']:,}")
        
        if results["invalid_lines"]:
            print(f"\n⚠️  Invalid JSON Lines: {len(results['invalid_lines'])}")
            print(f"   Line numbers: {results['invalid_lines'][:10]}")
            if len(results['invalid_lines']) > 10:
                print(f"   ... and {len(results['invalid_lines']) - 10} more")
        
        if results["missing_fields"]:
            print(f"\n⚠️  Missing Fields:")
            for field, count in results["missing_fields"].items():
                print(f"   {field}: {count} occurrences")
        
        print(f"\n📊 Statistics:")
        print(f"   Unique Trace IDs:       {results['unique_traces']:,}")
        print(f"   Unique Correlation IDs: {results['unique_correlations']:,}")
        
        if results["timestamp_errors"] > 0:
            print(f"\n⚠️  Timestamp Ordering Errors: {results['timestamp_errors']}")
        
        print("\n" + "=" * 70)