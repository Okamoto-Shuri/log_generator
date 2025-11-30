"""
Core Configuration Module - Part 1 Integration

設定クラス、メタデータ、ユーティリティクラス
"""

import logging
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

# ==================== ロギング設定 ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ==================== バージョン情報 ====================

VERSION = "3.0.0"
BUILD_DATE = "2025-11-28"


# ==================== 列挙型 ====================

class Severity(Enum):
    """ログの重大度レベル"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class Category(Enum):
    """異常のカテゴリ"""
    NORMAL = "normal"
    RESOURCE = "resource"
    NETWORK = "network"
    SECURITY = "security"
    APPLICATION = "application"
    DEPENDENCY = "dependency"
    CONFIGURATION = "configuration"
    MIDDLEWARE = "middleware"
    INFRASTRUCTURE = "infrastructure"


# ==================== データクラス ====================

@dataclass
class ScenarioMetadata:
    """シナリオのメタデータ"""
    weight: float
    cause: str
    category: str
    severity: str
    impact: str
    
    def __post_init__(self):
        """検証"""
        if self.weight < 0:
            raise ValueError(f"Weight must be non-negative: {self.weight}")
        if self.severity not in [s.value for s in Severity]:
            raise ValueError(f"Invalid severity: {self.severity}")


@dataclass
class GeneratorConfig:
    """ジェネレータの設定"""
    output_file: str = "training_dataset_v3.jsonl"
    total_events: int = 2000
    start_time_days_ago: int = 1
    embedding_dim: int = 384
    abnormal_ratio: float = 0.2
    batch_size: int = 1000
    random_seed: Optional[int] = None
    enable_time_correlation: bool = True
    enable_host_state: bool = True
    service_topology: Dict[str, List[str]] = field(default_factory=dict)
    
    def __post_init__(self):
        """設定の検証"""
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive: {self.embedding_dim}")
        if not (0 <= self.abnormal_ratio <= 1):
            raise ValueError(f"abnormal_ratio must be in [0,1]: {self.abnormal_ratio}")
        if self.total_events <= 0:
            raise ValueError(f"total_events must be positive: {self.total_events}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive: {self.batch_size}")
        
        # デフォルトのトポロジー設定
        if not self.service_topology:
            self.service_topology = {
                "nginx": ["web-01", "web-02"],
                "order-api": ["app-01", "app-02"],
                "inventory-api": ["app-01", "app-02"],
                "payment-api": ["app-01", "app-02"],
                "worker": ["app-01", "app-02"],
                "kernel": ["web-01", "web-02", "app-01", "app-02", "db-01", "db-02"],
                "postgresql": ["db-01", "db-02"]
            }


# ==================== シナリオ定義（修正版：合計1.0） ====================

SCENARIO_META: Dict[str, ScenarioMetadata] = {
    "A": ScenarioMetadata(0.02, "resource_memory_oom", "resource", "critical", "service_down"),
    "B": ScenarioMetadata(0.10, "network_db_latency", "network", "warning", "performance_degradation"),
    "C": ScenarioMetadata(0.01, "security_ddos", "security", "critical", "service_degradation"),
    "D": ScenarioMetadata(0.03, "resource_disk_full", "resource", "critical", "data_loss_risk"),
    "E": ScenarioMetadata(0.05, "network_external_api_down", "dependency", "error", "feature_unavailable"),
    "F": ScenarioMetadata(0.15, "app_logic_bug", "application", "error", "feature_unavailable"),
    "G": ScenarioMetadata(0.01, "security_sql_injection", "security", "critical", "data_breach_risk"),
    "H": ScenarioMetadata(0.05, "app_async_fail", "application", "error", "data_inconsistency"),
    "I": ScenarioMetadata(0.01, "config_ssl_expired", "configuration", "critical", "service_down"),
    "J": ScenarioMetadata(0.05, "resource_memory_leak", "resource", "warning", "performance_degradation"),
    "K": ScenarioMetadata(0.05, "config_auth_mismatch", "configuration", "fatal", "service_down"),
    "L": ScenarioMetadata(0.03, "network_dns_failure", "network", "error", "service_unavailable"),
    "M": ScenarioMetadata(0.03, "app_db_deadlock", "application", "error", "transaction_fail"),
    "N": ScenarioMetadata(0.03, "middleware_pool_exhausted", "middleware", "error", "service_unavailable"),
    "O": ScenarioMetadata(0.01, "security_payload_limit", "security", "warning", "request_rejected"),
    "P": ScenarioMetadata(0.05, "app_data_integrity", "application", "error", "transaction_fail"),
    "Q": ScenarioMetadata(0.01, "config_clock_skew", "configuration", "error", "auth_failure"),
    "R": ScenarioMetadata(0.02, "config_permission_denied", "configuration", "error", "feature_unavailable"),
    "S": ScenarioMetadata(0.05, "infrastructure_io_wait", "infrastructure", "warning", "performance_degradation"),
    "T": ScenarioMetadata(0.01, "infrastructure_split_brain", "infrastructure", "critical", "data_corruption_risk"),
    "U": ScenarioMetadata(0.23, "app_timeout", "application", "error", "request_timeout"),
}


# カテゴリごとのベクトル方向オフセット（グローバル変数として定義）
CATEGORY_VECTOR_OFFSETS: Dict[str, List[float]] = {}


def _initialize_category_vectors(dim: int = 384) -> None:
    """カテゴリベクトルを初期化（遅延初期化）"""
    global CATEGORY_VECTOR_OFFSETS
    
    # 各カテゴリに100次元を割り当て（より明確な分離）
    segment_size = 100
    base_strength = 0.8  # 信号強度を上げる
    
    categories = [
        "normal", "resource", "network", "security", 
        "application", "dependency", "configuration", 
        "middleware", "infrastructure"
    ]
    
    for idx, cat in enumerate(categories):
        vector = [0.0] * dim
        start_idx = (idx * segment_size) % dim
        end_idx = min(start_idx + segment_size, dim)
        
        # より強い信号を設定
        for i in range(start_idx, end_idx):
            vector[i] = base_strength
        
        CATEGORY_VECTOR_OFFSETS[cat] = vector
    
    logger.info(f"Initialized category vectors with dimension={dim}, strength={base_strength}")


# ==================== ユーティリティクラス ====================

class WeightNormalizer:
    """重みの正規化と検証を行うクラス"""
    
    @staticmethod
    def normalize_weights(scenarios: Dict[str, ScenarioMetadata]) -> Dict[str, float]:
        """
        シナリオの重みを正規化（合計を1.0に）
        
        Args:
            scenarios: シナリオメタデータの辞書
            
        Returns:
            正規化された重みの辞書
        """
        total_weight = sum(meta.weight for meta in scenarios.values())
        
        if total_weight == 0:
            raise ValueError("Total weight is zero, cannot normalize")
        
        normalized = {
            code: meta.weight / total_weight 
            for code, meta in scenarios.items()
        }
        
        # 検証
        final_sum = sum(normalized.values())
        if not (0.9999 <= final_sum <= 1.0001):  # 浮動小数点誤差を考慮
            logger.warning(f"Normalized weights sum to {final_sum:.10f}, expected 1.0")
        
        logger.info(f"Normalized {len(scenarios)} scenario weights: {total_weight:.4f} -> 1.0000")
        
        return normalized
    
    @staticmethod
    def validate_distribution(normalized_weights: Dict[str, float], tolerance: float = 1e-6) -> bool:
        """正規化された重みの分布を検証"""
        total = sum(normalized_weights.values())
        is_valid = abs(total - 1.0) < tolerance
        
        if not is_valid:
            logger.error(f"Weight distribution invalid: sum={total:.10f}")
        
        return is_valid


class HostStateManager:
    """ホストの状態を管理するクラス（時系列相関用）"""
    
    def __init__(self, smoothing_factor: float = 0.3):
        """
        Args:
            smoothing_factor: 状態遷移の平滑化係数 (0=変化なし, 1=即座に変化)
        """
        self.states: Dict[str, Dict[str, float]] = {}
        self.alpha = smoothing_factor
        self._default_state = {
            "cpu_usage": 30.0,
            "memory_usage": 40.0,
            "response_time_ms": 20.0,
            "disk_usage": 50.0,
            "network_latency_ms": 10.0
        }
    
    def get_state(self, host: str) -> Dict[str, float]:
        """ホストの現在状態を取得"""
        if host not in self.states:
            self.states[host] = self._default_state.copy()
        return self.states[host].copy()
    
    def update_state(
        self, 
        host: str, 
        target_state: Dict[str, float],
        immediate: bool = False
    ) -> Dict[str, float]:
        """
        ホストの状態を更新（指数移動平均）
        
        Args:
            host: ホスト名
            target_state: 目標状態
            immediate: Trueの場合、即座に目標状態に変更
            
        Returns:
            更新後の状態
        """
        current = self.get_state(host)
        
        if immediate:
            new_state = target_state.copy()
        else:
            # 指数移動平均で滑らかに遷移
            new_state = {}
            for key in current.keys():
                if key in target_state:
                    new_state[key] = (
                        current[key] * (1 - self.alpha) + 
                        target_state[key] * self.alpha
                    )
                else:
                    new_state[key] = current[key]
        
        self.states[host] = new_state
        return new_state.copy()
    
    def reset(self) -> None:
        """全ホストの状態をリセット"""
        self.states.clear()
        logger.debug("Host states reset")


# ==================== 定数 ====================

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15",
    "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36",
]

# 時間帯別のトラフィックパターン（ミリ秒）
TRAFFIC_PATTERNS = {
    "peak": (10, 100),          # 平日ピーク時
    "lunch": (100, 400),        # ランチタイム
    "normal": (200, 800),       # 通常時間帯
    "late_night": (2000, 5000), # 深夜
    "weekend": (1000, 5000),    # 週末
}


# ==================== 初期化 ====================

def initialize_generator(config: GeneratorConfig) -> None:
    """ジェネレータの初期化処理"""
    import random
    
    # ランダムシード設定
    if config.random_seed is not None:
        random.seed(config.random_seed)
        logger.info(f"Random seed set to: {config.random_seed}")
    
    # カテゴリベクトルの初期化
    _initialize_category_vectors(config.embedding_dim)
    
    # 重みの検証
    normalizer = WeightNormalizer()
    normalized = normalizer.normalize_weights(SCENARIO_META)
    
    if not normalizer.validate_distribution(normalized):
        raise ValueError("Scenario weight distribution is invalid")
    
    logger.info("Generator initialization completed")