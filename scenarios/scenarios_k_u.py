"""
Scenarios K-U Module - Part 3-2 Integration

シナリオK〜Uの実装と完全なファクトリー
"""

import uuid
import random
from typing import List
from datetime import timedelta

# coreパッケージからのインポート
from core import LogRecord, LogRecordFactory, LogFormatter

# scenariosパッケージからのインポート
from .base import ScenarioGenerator
from .scenarios_a_j import (
    ScenarioA, ScenarioB, ScenarioC, ScenarioD, ScenarioE,
    ScenarioF, ScenarioG, ScenarioH, ScenarioI, ScenarioJ
)


# ==================== 異常シナリオ K-U ====================

class ScenarioK(ScenarioGenerator):
    """Config Mismatch (Auth)"""
    
    def generate(self, base_time) -> List[LogRecord]:
        trace_id = str(uuid.uuid4())
        logs = []
        label = self.get_label()
        
        # 1. App log (Auth Failed)
        db_hosts = ["db-01", "db-02", "db-primary"]
        msg_app = self.fmt.format_app_json(
            "FATAL",
            'AuthenticationFailed: password authentication failed for user "app"',
            {"db_host": random.choice(db_hosts), "auth_method": "md5"}
        )
        logs.append(self.factory.create_record(
            base_time, 0, "order-api", "FATAL", msg_app, label,
            trace_id=trace_id,
            force_immediate_metrics=True
        ))
        
        # 2. Web log (500 Internal Server Error)
        msg_web = self.fmt.format_nginx(
            "GET", "/profile", 500, 0.050, base_time
        )
        logs.append(self.factory.create_record(
            base_time, 50, "nginx", "ERROR", msg_web, label,
            trace_id=trace_id
        ))
        
        return logs


class ScenarioL(ScenarioGenerator):
    """DNS Resolution Failure (Network)"""
    
    def generate(self, base_time) -> List[LogRecord]:
        trace_id = str(uuid.uuid4())
        logs = []
        label = self.get_label()
        
        # 1. App log (Unknown Host)
        hostnames = ["db-primary-01", "cache-cluster.internal", "api-backend"]
        resolver_ips = ["10.0.0.2", "10.0.0.53", "8.8.8.8"]
        
        msg_app = self.fmt.format_app_json(
            "ERROR",
            f"UnknownHostException: {random.choice(hostnames)}",
            {"resolver": random.choice(resolver_ips)}
        )
        logs.append(self.factory.create_record(
            base_time, 20, "order-api", "ERROR", msg_app, label,
            trace_id=trace_id
        ))
        
        # 2. Web log (503 Service Unavailable)
        msg_web = self.fmt.format_nginx(
            "GET", "/data", 503, 0.030, base_time
        )
        logs.append(self.factory.create_record(
            base_time, 30, "nginx", "ERROR", msg_web, label,
            trace_id=trace_id
        ))
        
        return logs


class ScenarioM(ScenarioGenerator):
    """DB Deadlock (Application)"""
    
    def generate(self, base_time) -> List[LogRecord]:
        trace_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        logs = []
        label = self.get_label()
        
        # 1. DB log (Deadlock Detected)
        pid1 = random.randint(100, 999)
        pid2 = random.randint(100, 999)
        txn1 = random.randint(1000, 9999)
        txn2 = random.randint(1000, 9999)
        
        msg_db = (
            f"Process {pid1} detected deadlock while waiting for ShareLock on transaction {txn1}; "
            f"DETAIL: Process {pid1} waits for ShareLock on transaction {txn1}; "
            f"process {pid2} waits for ShareLock on transaction {txn2}."
        )
        logs.append(self.factory.create_record(
            base_time, 0, "postgresql", "ERROR", msg_db, label,
            trace_id=None,
            correlation_id=correlation_id
        ))
        
        # 2. App log (Transaction Rollback)
        msg_app = self.fmt.format_app_json(
            "ERROR",
            "TransactionRolledbackException: Deadlock found when trying to get lock",
            {"table": "orders", "operation": "UPDATE"}
        )
        logs.append(self.factory.create_record(
            base_time, 10, "order-api", "ERROR", msg_app, label,
            trace_id=trace_id,
            correlation_id=correlation_id
        ))
        
        # 3. Web log (500 Internal Server Error)
        msg_web = self.fmt.format_nginx(
            "POST", "/update", 500, 0.150, base_time
        )
        logs.append(self.factory.create_record(
            base_time, 20, "nginx", "ERROR", msg_web, label,
            trace_id=trace_id,
            correlation_id=correlation_id
        ))
        
        return logs


class ScenarioN(ScenarioGenerator):
    """Connection Pool Exhausted (Middleware)"""
    
    def generate(self, base_time) -> List[LogRecord]:
        trace_id = str(uuid.uuid4())
        logs = []
        label = self.get_label()
        
        # 1. App log (Pool Exhausted)
        active_conn = random.randint(45, 50)
        max_conn = 50
        wait_time = random.randint(4500, 5500)
        
        msg_app = self.fmt.format_app_json(
            "ERROR",
            "PoolExhaustedException: Timeout waiting for idle object",
            {
                "active": active_conn,
                "idle": 0,
                "max": max_conn,
                "max_wait": wait_time
            }
        )
        logs.append(self.factory.create_record(
            base_time, wait_time, "order-api", "ERROR", msg_app, label,
            trace_id=trace_id,
            metrics={
                "response_time_ms": float(wait_time),
                "cpu_usage": 25.0,
                "memory_usage": 60.0
            },
            force_immediate_metrics=True
        ))
        
        # 2. Web log (503 Service Unavailable)
        msg_web = self.fmt.format_nginx(
            "GET", "/db-heavy", 503, wait_time / 1000, base_time
        )
        logs.append(self.factory.create_record(
            base_time, wait_time + 10, "nginx", "ERROR", msg_web, label,
            trace_id=trace_id
        ))
        
        return logs


class ScenarioO(ScenarioGenerator):
    """Payload Too Large (Security)"""
    
    def generate(self, base_time) -> List[LogRecord]:
        trace_id = str(uuid.uuid4())
        logs = []
        label = self.get_label()
        
        # 1. Web log (413 Payload Too Large)
        msg_web = self.fmt.format_nginx(
            "POST", "/upload", 413, 0.010, base_time
        )
        logs.append(self.factory.create_record(
            base_time, 10, "nginx", "ERROR", msg_web, label,
            trace_id=trace_id
        ))
        
        # 2. App log (Size Limit Exceeded)
        actual_size = random.randint(15, 50)
        max_size = 10
        
        msg_app = self.fmt.format_app_json(
            "WARN",
            f"MaxUploadSizeExceededException: {actual_size}MB > {max_size}MB",
            {"max_size_mb": max_size, "actual_size_mb": actual_size}
        )
        logs.append(self.factory.create_record(
            base_time, 15, "order-api", "WARN", msg_app, label,
            trace_id=trace_id
        ))
        
        return logs


class ScenarioP(ScenarioGenerator):
    """Poison Message (Data Integrity)"""
    
    def generate(self, base_time) -> List[LogRecord]:
        trace_id = str(uuid.uuid4())
        logs = []
        label = self.get_label()
        
        # 1. App log (Number Format Exception)
        invalid_values = ['"null"', '"undefined"', '"NaN"', '""']
        fields = ["price", "quantity", "user_id", "product_id"]
        
        msg_app = self.fmt.format_app_json(
            "ERROR",
            f'NumberFormatException: For input string: {random.choice(invalid_values)}',
            {"field": random.choice(fields), "input_data": "corrupted"}
        )
        logs.append(self.factory.create_record(
            base_time, 20, "order-api", "ERROR", msg_app, label,
            trace_id=trace_id
        ))
        
        # 2. Web log (500 Internal Server Error)
        msg_web = self.fmt.format_nginx(
            "POST", "/import", 500, 0.030, base_time
        )
        logs.append(self.factory.create_record(
            base_time, 30, "nginx", "ERROR", msg_web, label,
            trace_id=trace_id
        ))
        
        return logs


class ScenarioQ(ScenarioGenerator):
    """Clock Skew (Configuration)"""
    
    def generate(self, base_time) -> List[LogRecord]:
        trace_id = str(uuid.uuid4())
        logs = []
        label = self.get_label()
        
        # 1. App log (Token Not Yet Valid)
        # 未来の時刻を生成（5-10分先）
        future_time = base_time + timedelta(minutes=random.randint(5, 10))
        
        msg_app = self.fmt.format_app_json(
            "ERROR",
            "TokenNotYetValid (nbf claim is in future)",
            {
                "nbf": future_time.isoformat() + "Z",
                "current_time": base_time.isoformat() + "Z",
                "clock_skew_seconds": random.randint(300, 600)
            }
        )
        logs.append(self.factory.create_record(
            base_time, 20, "order-api", "ERROR", msg_app, label,
            trace_id=trace_id
        ))
        
        # 2. Web log (401 Unauthorized)
        msg_web = self.fmt.format_nginx(
            "POST", "/auth", 401, 0.030, base_time
        )
        logs.append(self.factory.create_record(
            base_time, 30, "nginx", "WARN", msg_web, label,
            trace_id=trace_id
        ))
        
        return logs


class ScenarioR(ScenarioGenerator):
    """File Permission (Configuration)"""
    
    def generate(self, base_time) -> List[LogRecord]:
        trace_id = str(uuid.uuid4())
        logs = []
        label = self.get_label()
        
        # 1. App log (Access Denied)
        paths = [
            "/var/log/audit.log",
            "/etc/app/config.yml",
            "/opt/app/data/cache",
            "/tmp/upload"
        ]
        
        msg_app = self.fmt.format_app_json(
            "ERROR",
            f"AccessDeniedException: {random.choice(paths)} (Permission denied)",
            {"required_permission": "rw-r--r--", "current_user": "app"}
        )
        logs.append(self.factory.create_record(
            base_time, 10, "order-api", "ERROR", msg_app, label,
            trace_id=trace_id
        ))
        
        # 2. Web log (500 Internal Server Error)
        msg_web = self.fmt.format_nginx(
            "POST", "/log", 500, 0.020, base_time
        )
        logs.append(self.factory.create_record(
            base_time, 20, "nginx", "ERROR", msg_web, label,
            trace_id=trace_id
        ))
        
        return logs


class ScenarioS(ScenarioGenerator):
    """Noisy Neighbor / I/O Wait (Infrastructure)"""
    
    def generate(self, base_time) -> List[LogRecord]:
        trace_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        logs = []
        label = self.get_label()
        
        # 1. Kernel log (Task Blocked)
        process_name = random.choice(["postgres", "mysqld", "java", "python"])
        pid = random.randint(1000, 9999)
        blocked_seconds = random.randint(120, 300)
        
        msg_kernel = self.fmt.format_syslog(
            "kernel",
            f"task {process_name}:{pid} blocked for more than {blocked_seconds} seconds",
            base_time
        )
        logs.append(self.factory.create_record(
            base_time, 0, "kernel", "WARN", msg_kernel, label,
            trace_id=None,
            correlation_id=correlation_id,
            metrics={
                "cpu_usage": 10.0,
                "memory_usage": 40.0,
                "response_time_ms": 2500.0
            },
            force_immediate_metrics=True
        ))
        
        # 2. Web log (Slow Response - 200 OK but slow)
        msg_web = self.fmt.format_nginx(
            "GET", "/items", 200, 2.500, base_time
        )
        logs.append(self.factory.create_record(
            base_time, 2500, "nginx", "WARN", msg_web, label,
            trace_id=trace_id,
            correlation_id=correlation_id
        ))
        
        return logs


class ScenarioT(ScenarioGenerator):
    """Split Brain (Infrastructure)"""
    
    def generate(self, base_time) -> List[LogRecord]:
        trace_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        logs = []
        label = self.get_label()
        
        # 1. Kernel log (Corosync Network Down)
        msg_kernel = self.fmt.format_syslog(
            "corosync",
            "TOTEM: The network interface [0.0.0.0] is down",
            base_time
        )
        logs.append(self.factory.create_record(
            base_time, 0, "kernel", "CRITICAL", msg_kernel, label,
            trace_id=None,
            correlation_id=correlation_id,
            force_immediate_metrics=True
        ))
        
        # 2. App log (Split Brain Detected)
        msg_app = self.fmt.format_app_json(
            "ERROR",
            "QuorumLossException: Split Brain Detected - fencing required",
            {
                "cluster_nodes": 2,
                "quorum_required": 2,
                "quorum_present": 1,
                "action": "stonith"
            }
        )
        logs.append(self.factory.create_record(
            base_time, 50, "order-api", "ERROR", msg_app, label,
            trace_id=trace_id,
            correlation_id=correlation_id
        ))
        
        # 3. Web log (503 Service Unavailable)
        msg_web = self.fmt.format_nginx(
            "POST", "/transaction", 503, 0.060, base_time
        )
        logs.append(self.factory.create_record(
            base_time, 60, "nginx", "ERROR", msg_web, label,
            trace_id=trace_id,
            correlation_id=correlation_id
        ))
        
        return logs


class ScenarioU(ScenarioGenerator):
    """Application Timeout (Application)"""
    
    def generate(self, base_time) -> List[LogRecord]:
        trace_id = str(uuid.uuid4())
        logs = []
        label = self.get_label()
        
        # 1. App log (Request Timeout)
        timeout_ms = random.randint(8000, 12000)
        
        msg_app = self.fmt.format_app_json(
            "ERROR",
            f"RequestTimeoutException: Request processing exceeded {timeout_ms}ms",
            {
                "timeout_ms": timeout_ms,
                "endpoint": random.choice(["/api/search", "/api/report", "/api/export"]),
                "method": "GET"
            }
        )
        logs.append(self.factory.create_record(
            base_time, timeout_ms, "order-api", "ERROR", msg_app, label,
            trace_id=trace_id,
            metrics={
                "response_time_ms": float(timeout_ms),
                "cpu_usage": 45.0,
                "memory_usage": 65.0
            },
            force_immediate_metrics=True
        ))
        
        # 2. Web log (504 Gateway Timeout)
        msg_web = self.fmt.format_nginx(
            "GET", "/api/search", 504, timeout_ms / 1000, base_time
        )
        logs.append(self.factory.create_record(
            base_time, timeout_ms + 10, "nginx", "ERROR", msg_web, label,
            trace_id=trace_id
        ))
        
        return logs


# ==================== 完全なシナリオファクトリー ====================

class CompleteScenarioFactory:
    """全シナリオ（A-U）を管理するファクトリークラス"""
    
    def __init__(
        self,
        record_factory: LogRecordFactory,
        formatter: LogFormatter
    ):
        self.record_factory = record_factory
        self.formatter = formatter
        
        # 全シナリオマッピング（A-U）
        self.scenarios = {
            'A': ScenarioA,
            'B': ScenarioB,
            'C': ScenarioC,
            'D': ScenarioD,
            'E': ScenarioE,
            'F': ScenarioF,
            'G': ScenarioG,
            'H': ScenarioH,
            'I': ScenarioI,
            'J': ScenarioJ,
            'K': ScenarioK,
            'L': ScenarioL,
            'M': ScenarioM,
            'N': ScenarioN,
            'O': ScenarioO,
            'P': ScenarioP,
            'Q': ScenarioQ,
            'R': ScenarioR,
            'S': ScenarioS,
            'T': ScenarioT,
            'U': ScenarioU,
        }
    
    def create(self, scenario_code: str) -> ScenarioGenerator:
        """
        シナリオコードに応じた生成器を作成
        
        Args:
            scenario_code: シナリオコード（A-U）
            
        Returns:
            シナリオ生成器
            
        Raises:
            ValueError: 未知のシナリオコード
        """
        scenario_class = self.scenarios.get(scenario_code)
        
        if scenario_class is None:
            raise ValueError(f"Unknown scenario code: {scenario_code}")
        
        return scenario_class(
            self.record_factory,
            self.formatter,
            scenario_code
        )
    
    def get_all_scenario_codes(self) -> List[str]:
        """全シナリオコードのリストを取得"""
        return list(self.scenarios.keys())