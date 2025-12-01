"""
Core Package Initialization

coreパッケージの公開API
"""

# config.pyからのエクスポート
from .config import (
    logger,
    VERSION,
    BUILD_DATE,
    Severity,
    Category,
    ScenarioMetadata,
    GeneratorConfig,
    SCENARIO_META,
    CATEGORY_VECTOR_OFFSETS,
    WeightNormalizer,
    USER_AGENTS,
    TRAFFIC_PATTERNS,
    initialize_generator,
    _initialize_category_vectors,
    get_config,
    reload_config
)

# protocols.pyからのエクスポート（抽象層）
from .protocols import (
    HostStateManagerProtocol
)

# host_state.pyからのエクスポート（具象実装）
from .host_state import (
    HostStateManager
)

# generators.pyからのエクスポート
from .generators import (
    SemanticVectorGenerator,
    MetricsGenerator,
    LogFormatter,
    LogRecord,
    LogRecordFactory,
    TimeManager
)

# statistics.pyからのエクスポート
from .statistics import (
    StatisticsCollector,
    DatasetValidator
)

__all__ = [
    # config
    'logger',
    'VERSION',
    'BUILD_DATE',
    'Severity',
    'Category',
    'ScenarioMetadata',
    'GeneratorConfig',
    'SCENARIO_META',
    'CATEGORY_VECTOR_OFFSETS',
    'WeightNormalizer',
    'USER_AGENTS',
    'TRAFFIC_PATTERNS',
    'initialize_generator',
    '_initialize_category_vectors',
    'get_config',
    'reload_config',
    
    # protocols (抽象層)
    'HostStateManagerProtocol',
    
    # host_state (具象実装)
    'HostStateManager',
    
    # generators
    'SemanticVectorGenerator',
    'MetricsGenerator',
    'LogFormatter',
    'LogRecord',
    'LogRecordFactory',
    'TimeManager',
    
    # statistics
    'StatisticsCollector',
    'DatasetValidator',
]