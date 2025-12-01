"""
Core Configuration Module - Part 1 Integration

設定クラス、メタデータ、ユーティリティクラス
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Protocol
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

# ==================== YAMLライブラリのインポート ====================

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML is not installed. Config file loading will be disabled. Install with: pip install PyYAML")


# ==================== バージョン情報 ====================

VERSION = "1.0.0"
BUILD_DATE = "2025-11-28"


# ==================== 設定ファイル読み込み ====================

class ConfigLoader:
    """設定ファイルと環境変数から設定を読み込むクラス"""
    
    _config_cache: Optional[Dict[str, Any]] = None
    
    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        設定ファイルと環境変数から設定を読み込む
        
        Args:
            config_path: 設定ファイルのパス（Noneの場合はデフォルトパスを検索）
            
        Returns:
            設定辞書
        """
        if cls._config_cache is not None:
            return cls._config_cache
        
        # デフォルトの設定
        default_config = cls._get_default_config()
        
        # 設定ファイルから読み込み
        config_file_path = cls._find_config_file(config_path)
        if config_file_path and config_file_path.exists():
            if not YAML_AVAILABLE:
                logger.warning(f"Config file {config_file_path} found but PyYAML is not installed. Using defaults.")
            else:
                try:
                    with open(config_file_path, 'r', encoding='utf-8') as f:
                        file_config = yaml.safe_load(f) or {}
                    # ファイル設定でデフォルトを上書き
                    cls._merge_config(default_config, file_config)
                    logger.info(f"Loaded configuration from {config_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to load config file {config_file_path}: {e}. Using defaults.")
        else:
            logger.info("No config file found. Using default configuration.")
        
        # 環境変数で上書き
        cls._apply_env_overrides(default_config)
        
        cls._config_cache = default_config
        return default_config
    
    @classmethod
    def _get_default_config(cls) -> Dict[str, Any]:
        """デフォルト設定を返す"""
        return {
            "semantic_vector": {
                "severity_scales": {
                    "info": 0.3,
                    "warning": 0.6,
                    "error": 1.0,
                    "critical": 1.5,
                    "fatal": 2.0
                },
                "noise_std": 0.05,
                "category_vector": {
                    "segment_size": 100,
                    "base_strength": 0.8
                }
            },
            "metrics": {
                "targets": {
                    "normal": {
                        "cpu_usage": [10, 60],
                        "memory_usage": [20, 70],
                        "response_time_ms": [5, 50],
                        "disk_usage": [30, 70],
                        "network_latency_ms": [5, 20]
                    },
                    "resource_memory": {
                        "cpu_usage": [40, 80],
                        "memory_usage": [85, 100],
                        "response_time_ms": [100, 500],
                        "disk_usage": [30, 70],
                        "network_latency_ms": [10, 30]
                    },
                    "resource_disk": {
                        "cpu_usage": [10, 40],
                        "memory_usage": [30, 70],
                        "response_time_ms": [200, 1000],
                        "disk_usage": [95, 100],
                        "network_latency_ms": [10, 30]
                    },
                    "resource_cpu": {
                        "cpu_usage": [85, 100],
                        "memory_usage": [40, 80],
                        "response_time_ms": [50, 200],
                        "disk_usage": [30, 70],
                        "network_latency_ms": [10, 30]
                    },
                    "network": {
                        "cpu_usage": [5, 30],
                        "memory_usage": [20, 60],
                        "response_time_ms": [1000, 5000],
                        "disk_usage": [30, 70],
                        "network_latency_ms": [500, 3000]
                    },
                    "security": {
                        "cpu_usage": [80, 100],
                        "memory_usage": [60, 90],
                        "response_time_ms": [10, 100],
                        "disk_usage": [30, 70],
                        "network_latency_ms": [5, 50]
                    },
                    "infrastructure": {
                        "cpu_usage": [10, 40],
                        "memory_usage": [30, 70],
                        "response_time_ms": [500, 3000],
                        "disk_usage": [30, 70],
                        "network_latency_ms": [100, 500]
                    },
                    "application": {
                        "cpu_usage": [30, 70],
                        "memory_usage": [40, 80],
                        "response_time_ms": [100, 500],
                        "disk_usage": [30, 70],
                        "network_latency_ms": [10, 50]
                    },
                    "dependency": {
                        "cpu_usage": [10, 40],
                        "memory_usage": [30, 60],
                        "response_time_ms": [2000, 6000],
                        "disk_usage": [30, 70],
                        "network_latency_ms": [1000, 4000]
                    },
                    "configuration": {
                        "cpu_usage": [20, 60],
                        "memory_usage": [30, 70],
                        "response_time_ms": [50, 200],
                        "disk_usage": [30, 70],
                        "network_latency_ms": [10, 50]
                    },
                    "middleware": {
                        "cpu_usage": [30, 70],
                        "memory_usage": [50, 90],
                        "response_time_ms": [2000, 6000],
                        "disk_usage": [30, 70],
                        "network_latency_ms": [20, 100]
                    }
                }
            },
            "host_state": {
                "smoothing_factor": 0.3,
                "default_state": {
                    "cpu_usage": 30.0,
                    "memory_usage": 40.0,
                    "response_time_ms": 20.0,
                    "disk_usage": 50.0,
                    "network_latency_ms": 10.0
                }
            },
            "traffic_patterns": {
                "peak": [10, 100],
                "lunch": [100, 400],
                "normal": [200, 800],
                "late_night": [2000, 5000],
                "weekend": [1000, 5000]
            },
            "generator": {
                "output_file": "training_dataset.jsonl",
                "total_events": 2000,
                "start_time_days_ago": 1,
                "embedding_dim": 384,
                "abnormal_ratio": 0.2,
                "batch_size": 1000,
                "enable_time_correlation": True,
                "enable_host_state": True
            }
        }
    
    @classmethod
    def _find_config_file(cls, config_path: Optional[str] = None) -> Optional[Path]:
        """設定ファイルのパスを検索"""
        if config_path:
            return Path(config_path)
        
        # 現在のディレクトリから順に検索
        current = Path.cwd()
        for path in [current, current.parent]:
            config_file = path / "config.yaml"
            if config_file.exists():
                return config_file
        
        # プロジェクトルートを検索
        project_root = Path(__file__).parent.parent
        config_file = project_root / "config.yaml"
        if config_file.exists():
            return config_file
        
        return None
    
    @classmethod
    def _merge_config(cls, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """設定を再帰的にマージ"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                cls._merge_config(base[key], value)
            else:
                base[key] = value
    
    @classmethod
    def _apply_env_overrides(cls, config: Dict[str, Any]) -> None:
        """環境変数で設定を上書き"""
        # セマンティックベクトル設定
        if os.getenv("LOG_GEN_SEVERITY_INFO"):
            config["semantic_vector"]["severity_scales"]["info"] = float(os.getenv("LOG_GEN_SEVERITY_INFO"))
        if os.getenv("LOG_GEN_SEVERITY_WARNING"):
            config["semantic_vector"]["severity_scales"]["warning"] = float(os.getenv("LOG_GEN_SEVERITY_WARNING"))
        if os.getenv("LOG_GEN_SEVERITY_ERROR"):
            config["semantic_vector"]["severity_scales"]["error"] = float(os.getenv("LOG_GEN_SEVERITY_ERROR"))
        if os.getenv("LOG_GEN_SEVERITY_CRITICAL"):
            config["semantic_vector"]["severity_scales"]["critical"] = float(os.getenv("LOG_GEN_SEVERITY_CRITICAL"))
        if os.getenv("LOG_GEN_SEVERITY_FATAL"):
            config["semantic_vector"]["severity_scales"]["fatal"] = float(os.getenv("LOG_GEN_SEVERITY_FATAL"))
        if os.getenv("LOG_GEN_NOISE_STD"):
            config["semantic_vector"]["noise_std"] = float(os.getenv("LOG_GEN_NOISE_STD"))
        
        # ホスト状態設定
        if os.getenv("LOG_GEN_SMOOTHING_FACTOR"):
            config["host_state"]["smoothing_factor"] = float(os.getenv("LOG_GEN_SMOOTHING_FACTOR"))
        
        # ジェネレータ設定
        if os.getenv("LOG_GEN_OUTPUT_FILE"):
            config["generator"]["output_file"] = os.getenv("LOG_GEN_OUTPUT_FILE")
        if os.getenv("LOG_GEN_TOTAL_EVENTS"):
            config["generator"]["total_events"] = int(os.getenv("LOG_GEN_TOTAL_EVENTS"))
        if os.getenv("LOG_GEN_EMBEDDING_DIM"):
            config["generator"]["embedding_dim"] = int(os.getenv("LOG_GEN_EMBEDDING_DIM"))
        if os.getenv("LOG_GEN_ABNORMAL_RATIO"):
            config["generator"]["abnormal_ratio"] = float(os.getenv("LOG_GEN_ABNORMAL_RATIO"))
    
    @classmethod
    def reset_cache(cls) -> None:
        """設定キャッシュをリセット（テスト用）"""
        cls._config_cache = None


# グローバル設定インスタンス
_app_config = ConfigLoader.load_config()


def get_config() -> Dict[str, Any]:
    """設定を取得"""
    return _app_config


def reload_config(config_path: Optional[str] = None) -> None:
    """設定を再読み込み"""
    ConfigLoader.reset_cache()
    global _app_config
    _app_config = ConfigLoader.load_config(config_path)


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
    output_file: str = "training_dataset.jsonl"
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
    
    config = get_config()
    cv_config = config.get("semantic_vector", {}).get("category_vector", {})
    
    # 設定から値を取得（デフォルト値あり）
    segment_size = cv_config.get("segment_size", 100)
    base_strength = cv_config.get("base_strength", 0.8)
    
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


# ==================== 依存性の逆転: Protocol定義 ====================

class HostStateManagerProtocol(Protocol):
    """ホスト状態管理のプロトコル（依存性の逆転）"""
    
    def get_state(self, host: str) -> Dict[str, float]:
        """ホストの現在状態を取得"""
        ...
    
    def update_state(
        self, 
        host: str, 
        target_state: Dict[str, float],
        immediate: bool = False
    ) -> Dict[str, float]:
        """ホストの状態を更新"""
        ...


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
    
    def __init__(self, smoothing_factor: Optional[float] = None):
        """
        Args:
            smoothing_factor: 状態遷移の平滑化係数 (0=変化なし, 1=即座に変化)
                            Noneの場合は設定ファイルから読み込む
        """
        config = get_config()
        hs_config = config.get("host_state", {})
        
        self.states: Dict[str, Dict[str, float]] = {}
        self.alpha = smoothing_factor if smoothing_factor is not None else hs_config.get("smoothing_factor", 0.3)
        self._default_state = hs_config.get("default_state", {
            "cpu_usage": 30.0,
            "memory_usage": 40.0,
            "response_time_ms": 20.0,
            "disk_usage": 50.0,
            "network_latency_ms": 10.0
        }).copy()
    
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
def _get_traffic_patterns() -> Dict[str, tuple]:
    """トラフィックパターンを設定から取得"""
    config = get_config()
    patterns = config.get("traffic_patterns", {
        "peak": [10, 100],
        "lunch": [100, 400],
        "normal": [200, 800],
        "late_night": [2000, 5000],
        "weekend": [1000, 5000]
    })
    
    # リストをタプルに変換
    return {k: tuple(v) if isinstance(v, list) else v for k, v in patterns.items()}


TRAFFIC_PATTERNS = _get_traffic_patterns()


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