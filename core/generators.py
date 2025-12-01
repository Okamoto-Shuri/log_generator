"""
Core Generators Module - Part 2 Integration

ログ生成の中核機能
"""

import json
import random
import uuid
import math
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# core.configからのインポート
from .config import (
    logger,
    GeneratorConfig,
    CATEGORY_VECTOR_OFFSETS,
    USER_AGENTS,
    TRAFFIC_PATTERNS,
    get_config
)

# core.protocolsからのインポート（抽象層のみに依存）
from .protocols import HostStateManagerProtocol


# ==================== セマンティックベクトル生成 ====================

class SemanticVectorGenerator:
    """改善版: より明確なカテゴリ分離を持つベクトル生成"""
    
    def __init__(self, embedding_dim: int = 384, noise_std: Optional[float] = None):
        """
        Args:
            embedding_dim: ベクトルの次元数
            noise_std: ノイズの標準偏差（小さいほど明確な分離）
                      Noneの場合は設定ファイルから読み込む
        """
        config = get_config()
        sv_config = config.get("semantic_vector", {})
        
        self.dim = embedding_dim
        self.noise_std = noise_std if noise_std is not None else sv_config.get("noise_std", 0.05)
        
        # 重大度ごとのスケール係数を設定から読み込む
        self.SEVERITY_SCALES = sv_config.get("severity_scales", {
            "info": 0.3,
            "warning": 0.6,
            "error": 1.0,
            "critical": 1.5,
            "fatal": 2.0
        })
    
    def generate(self, message: str, label: Dict) -> List[float]:
        """
        メッセージとラベルから決定的なセマンティックベクトルを生成
        
        Args:
            message: ログメッセージ
            label: ラベル情報（root_cause, category, severity等）
            
        Returns:
            セマンティックベクトル
        """
        # メッセージとroot_causeから決定的なシード生成
        seed_str = message + label.get("root_cause", "unknown")
        seed = hash(seed_str) % (2**32)
        rng = random.Random(seed)
        
        # 1. ベースノイズ（小さめ）
        base_vector = [rng.gauss(0, self.noise_std) for _ in range(self.dim)]
        
        # 2. カテゴリ方向の信号
        category = label.get("category", "normal")
        category_offset = CATEGORY_VECTOR_OFFSETS.get(
            category, 
            [0.0] * self.dim
        )
        
        # 3. 重大度によるスケーリング
        severity = label.get("severity", "info")
        scale = self.SEVERITY_SCALES.get(severity, 1.0)
        
        # 4. root_causeごとの微細な差異（同一カテゴリ内の識別用）
        root_cause = label.get("root_cause", "unknown")
        cause_seed = hash(root_cause) % 1000
        cause_offset = [
            math.sin((i + cause_seed) * 0.01) * 0.1 
            for i in range(self.dim)
        ]
        
        # 5. 合成（scale * (base + category) + cause）
        vector = [
            scale * (base + cat) + cause 
            for base, cat, cause in zip(base_vector, category_offset, cause_offset)
        ]
        
        return vector
    
    def compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """コサイン類似度を計算（デバッグ用）"""
        if len(vec1) != len(vec2):
            raise ValueError("Vector dimensions must match")
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


# ==================== メトリクス生成 ====================

class MetricsGenerator:
    """時系列相関を持つメトリクス生成"""
    
    def __init__(self, host_state_manager: Optional[HostStateManagerProtocol] = None):
        """
        Args:
            host_state_manager: ホスト状態管理（Protocolに依存、Noneの場合は状態管理なし）
        """
        config = get_config()
        metrics_config = config.get("metrics", {})
        
        # カテゴリごとのメトリクス目標値を設定から読み込む
        targets = metrics_config.get("targets", {})
        
        # リストをタプルに変換
        self.METRIC_TARGETS = {}
        for category, metrics in targets.items():
            self.METRIC_TARGETS[category] = {
                k: tuple(v) if isinstance(v, list) else v
                for k, v in metrics.items()
            }
        
        # デフォルト値が設定ファイルにない場合のフォールバック
        if not self.METRIC_TARGETS:
            self.METRIC_TARGETS = {
                "normal": {
                    "cpu_usage": (10, 60),
                    "memory_usage": (20, 70),
                    "response_time_ms": (5, 50),
                    "disk_usage": (30, 70),
                    "network_latency_ms": (5, 20)
                }
            }
        
        self.hsm = host_state_manager
    
    def generate(
        self, 
        label: Dict, 
        host: str,
        force_immediate: bool = False
    ) -> Dict[str, float]:
        """
        ラベルとホストに応じたメトリクスを生成
        
        Args:
            label: ラベル情報
            host: ホスト名
            force_immediate: Trueの場合、状態を即座に変更
            
        Returns:
            メトリクス辞書
        """
        # カテゴリとroot_causeから適切なターゲットを選択
        category = label.get("category", "normal")
        root_cause = label.get("root_cause", "")
        
        # root_causeに応じた細かい分類
        if category == "resource":
            if "memory" in root_cause or "oom" in root_cause:
                target_key = "resource_memory"
            elif "disk" in root_cause:
                target_key = "resource_disk"
            elif "cpu" in root_cause:
                target_key = "resource_cpu"
            else:
                target_key = "normal"
        else:
            target_key = category if category in self.METRIC_TARGETS else "normal"
        
        # 目標値の範囲を取得
        targets = self.METRIC_TARGETS.get(target_key, self.METRIC_TARGETS["normal"])
        
        # ランダムな目標値を生成
        target_state = {}
        for metric, (min_val, max_val) in targets.items():
            target_state[metric] = random.uniform(min_val, max_val)
        
        # ホスト状態管理が有効な場合は滑らかに遷移
        if self.hsm:
            metrics = self.hsm.update_state(host, target_state, immediate=force_immediate)
        else:
            metrics = target_state
        
        # 値を丸める
        return {k: round(v, 1) for k, v in metrics.items()}


# ==================== ログフォーマッター ====================

class LogFormatter:
    """各種ログ形式のフォーマッター"""
    
    @staticmethod
    def format_syslog(
        proc: str, 
        msg: str, 
        timestamp: datetime
    ) -> str:
        """
        Syslog形式のログを生成
        
        Args:
            proc: プロセス名
            msg: メッセージ
            timestamp: タイムスタンプ
            
        Returns:
            フォーマット済みログ
        """
        ts = timestamp.strftime("%b %d %H:%M:%S")
        return f"{ts} {proc}: {msg}"
    
    @staticmethod
    def format_nginx(
        method: str,
        path: str,
        status: int,
        latency_s: float,
        timestamp: datetime,
        extra: str = ""
    ) -> str:
        """
        Nginx Access Log形式のログを生成
        
        Args:
            method: HTTPメソッド
            path: リクエストパス
            status: ステータスコード
            latency_s: レイテンシ（秒）
            timestamp: タイムスタンプ
            extra: 追加情報
            
        Returns:
            フォーマット済みログ
        """
        ts = timestamp.strftime("%d/%b/%Y:%H:%M:%S +0000")
        ip = str(ipaddress.IPv4Address(random.randint(0, 2**32 - 1)))
        user_agent = random.choice(USER_AGENTS)
        size = random.randint(512, 4096)
        
        log = (
            f'{ip} - - [{ts}] '
            f'"{method} {path} HTTP/1.1" '
            f'{status} {size} "-" "{user_agent}" {latency_s:.3f}'
        )
        
        if extra:
            log += f' {extra}'
        
        return log
    
    @staticmethod
    def format_app_json(
        level: str,
        msg: str,
        extra: Optional[Dict] = None
    ) -> str:
        """
        アプリケーションJSON形式のログを生成
        
        Args:
            level: ログレベル
            msg: メッセージ
            extra: 追加フィールド
            
        Returns:
            JSON文字列
        """
        log_dict = {"lvl": level, "msg": msg}
        
        if extra:
            log_dict.update(extra)
        
        return json.dumps(log_dict, ensure_ascii=False)


# ==================== レコード生成 ====================

@dataclass
class LogRecord:
    """ログレコードのデータクラス"""
    timestamp: str
    service: str
    host: str
    level: str
    trace_id: Optional[str]
    correlation_id: str
    message: str
    metrics: Dict[str, float]
    label: Dict
    message_vector: List[float]
    
    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return asdict(self)


class LogRecordFactory:
    """ログレコード生成のファクトリークラス"""
    
    def __init__(
        self,
        config: GeneratorConfig,
        vector_generator: SemanticVectorGenerator,
        metrics_generator: MetricsGenerator
    ):
        """
        Args:
            config: ジェネレータ設定
            vector_generator: ベクトル生成器
            metrics_generator: メトリクス生成器
        """
        self.config = config
        self.vector_gen = vector_generator
        self.metrics_gen = metrics_generator
    
    def create_record(
        self,
        base_time: datetime,
        offset_ms: int,
        service: str,
        level: str,
        message: str,
        label: Dict,
        trace_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        force_immediate_metrics: bool = False
    ) -> LogRecord:
        """
        ログレコードを生成
        
        Args:
            base_time: 基準時刻
            offset_ms: オフセット（ミリ秒）
            service: サービス名
            level: ログレベル
            message: メッセージ
            label: ラベル情報
            trace_id: トレースID（Noneの場合は生成しない）
            correlation_id: 相関ID（Noneの場合は自動生成）
            metrics: メトリクス（Noneの場合は自動生成）
            force_immediate_metrics: メトリクスを即座に目標値に変更
            
        Returns:
            ログレコード
        """
        # タイムスタンプ計算
        timestamp = base_time + timedelta(milliseconds=offset_ms)
        
        # ホスト選択
        host = self._get_host(service)
        
        # correlation_idの決定（優先順位: 引数 > trace_id > 新規生成）
        final_correlation_id = correlation_id or trace_id or str(uuid.uuid4())
        
        # メトリクス生成（指定がなければ自動生成）
        if metrics is None:
            metrics = self.metrics_gen.generate(
                label, 
                host, 
                force_immediate=force_immediate_metrics
            )
        
        # セマンティックベクトル生成
        vector = self.vector_gen.generate(message, label)
        
        return LogRecord(
            timestamp=timestamp.isoformat() + "Z",
            service=service,
            host=host,
            level=level,
            trace_id=trace_id,
            correlation_id=final_correlation_id,
            message=message,
            metrics=metrics,
            label=label,
            message_vector=vector
        )
    
    def _get_host(self, service: str) -> str:
        """サービスからホストを選択"""
        candidates = self.config.service_topology.get(service, ["unknown-host"])
        return random.choice(candidates)


# ==================== 時間管理 ====================

class TimeManager:
    """時刻とトラフィックパターンを管理"""
    
    def __init__(self, start_time: datetime):
        """
        Args:
            start_time: 開始時刻
        """
        self.current_time = start_time
    
    def advance(self) -> None:
        """時刻を進める（トラフィックパターンに応じた間隔）"""
        interval_ms = self._get_interval()
        self.current_time += timedelta(milliseconds=interval_ms)
    
    def _get_interval(self) -> int:
        """現在の時間帯に応じた次のイベント間隔を取得（ミリ秒）"""
        hour = self.current_time.hour
        day_of_week = self.current_time.weekday()  # 0=月曜, 6=日曜
        
        # 週末判定
        if day_of_week >= 5:
            pattern = "weekend"
        # 平日のパターン
        elif 9 <= hour < 12 or 14 <= hour < 17:
            pattern = "peak"
        elif 12 <= hour < 14:
            pattern = "lunch"
        elif 0 <= hour < 6:
            pattern = "late_night"
        else:
            pattern = "normal"
        
        # 設定から動的に取得（設定変更に対応）
        config = get_config()
        patterns_config = config.get("traffic_patterns", {
            "peak": [10, 100],
            "lunch": [100, 400],
            "normal": [200, 800],
            "late_night": [2000, 5000],
            "weekend": [1000, 5000]
        })
        
        pattern_value = patterns_config.get(pattern, [200, 800])
        if isinstance(pattern_value, list):
            min_interval, max_interval = tuple(pattern_value)
        else:
            min_interval, max_interval = pattern_value
        
        return random.randint(min_interval, max_interval)
    
    def get_current_time(self) -> datetime:
        """現在時刻を取得"""
        return self.current_time