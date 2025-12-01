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
    HostStateManager,
    USER_AGENTS,
    TRAFFIC_PATTERNS,
    initialize_generator,
    _initialize_category_vectors,
    get_config,
    reload_config
)

# generators.pyからのエクスポート
from .generators import (
    HostStateManagerProtocol,
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
    'HostStateManager',
    'USER_AGENTS',
    'TRAFFIC_PATTERNS',
    'initialize_generator',
    '_initialize_category_vectors',
    'get_config',
    'reload_config',
    
    # generators
    'HostStateManagerProtocol',
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