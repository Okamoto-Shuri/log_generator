"""
Scenarios A-J Module - Part 3-1 Integration

シナリオA〜Jと正常系の実装
"""

import uuid
import random
from typing import List
from datetime import datetime

# coreパッケージからのインポート
from core import LogRecord, LogRecordFactory, LogFormatter

# scenariosパッケージからのインポート
from .base import ScenarioGenerator


# ==================== 正常系シナリオ ====================

class NormalScenarioGenerator:
    """正常系トランザクション生成"""
    
    def __init__(
        self,
        record_factory: LogRecordFactory,
        formatter: LogFormatter
    ):
        self.factory = record_factory
        self.fmt = formatter
    
    def generate(self, base_time: datetime) -> List[LogRecord]:
        """正常系のログを生成（複数パターン）"""
        trace_id = str(uuid.uuid4())
        logs = []
        pattern = random.choice(['static', 'api_simple', 'api_db'])
        
        label = {
            "scenario": "normal",
            "root_cause": "none",
            "category": "normal",
            "severity": "info",
            "impact": "none"
        }
        
        if pattern == 'static':
            # 静的コンテンツ配信
            msg = self.fmt.format_nginx(
                "GET", "/assets/logo.png", 200, 0.005, base_time
            )
            logs.append(self.factory.create_record(
                base_time, 0, "nginx", "INFO", msg, label, trace_id
            ))
        
        elif pattern == 'api_simple':
            # Web -> App (計算のみ)
            msg_web = self.fmt.format_nginx(
                "GET", "/api/calc", 200, 0.020, base_time
            )
            logs.append(self.factory.create_record(
                base_time, 0, "nginx", "INFO", msg_web, label, trace_id
            ))
            
            msg_app = self.fmt.format_app_json(
                "INFO", "Calculation finished", {"duration": 15}
            )
            logs.append(self.factory.create_record(
                base_time, 5, "order-api", "INFO", msg_app, label, trace_id
            ))
        
        elif pattern == 'api_db':
            # Web -> App -> DB
            msg_web = self.fmt.format_nginx(
                "GET", "/api/orders", 200, 0.100, base_time
            )
            logs.append(self.factory.create_record(
                base_time, 0, "nginx", "INFO", msg_web, label, trace_id
            ))
            
            msg_app = self.fmt.format_app_json(
                "INFO", "Fetching orders", {"user_id": 123}
            )
            logs.append(self.factory.create_record(
                base_time, 10, "order-api", "INFO", msg_app, label, trace_id
            ))
        
        return logs


# ==================== 異常シナリオ A-J ====================

class ScenarioA(ScenarioGenerator):
    """OOM Killer (Resource)"""
    
    def generate(self, base_time: datetime) -> List[LogRecord]:
        trace_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        logs = []
        label = self.get_label()
        
        # 1. Kernel log (OOM Killer)
        msg_kernel = self.fmt.format_syslog(
            "kernel",
            "Out of memory: Kill process 2345 (python) score 850 or sacrifice child",
            base_time
        )
        logs.append(self.factory.create_record(
            base_time, 0, "kernel", "CRITICAL", msg_kernel, label,
            trace_id=None,
            correlation_id=correlation_id,
            metrics={"memory_usage": 99.8, "cpu_usage": 45.0, "response_time_ms": 0.0},
            force_immediate_metrics=True
        ))
        
        # 2. App log (Broken Pipe)
        msg_app = self.fmt.format_app_json(
            "ERROR",
            "Connection to worker failed: BrokenPipeError"
        )
        logs.append(self.factory.create_record(
            base_time, 10, "order-api", "ERROR", msg_app, label,
            trace_id=trace_id,
            correlation_id=correlation_id
        ))
        
        # 3. Web log (502 Bad Gateway)
        msg_web = self.fmt.format_nginx(
            "POST", "/checkout", 502, 0.020, base_time
        )
        logs.append(self.factory.create_record(
            base_time, 20, "nginx", "ERROR", msg_web, label,
            trace_id=trace_id,
            correlation_id=correlation_id
        ))
        
        return logs


class ScenarioB(ScenarioGenerator):
    """DB Latency (Network)"""
    
    def generate(self, base_time: datetime) -> List[LogRecord]:
        trace_id = str(uuid.uuid4())
        logs = []
        label = self.get_label()
        
        # 1. App log (SQL Timeout)
        msg_app = self.fmt.format_app_json(
            "ERROR",
            "SQLTimeoutException: query exceeded 3000ms",
            {"sql_query_id": f"q-{random.randint(1000, 9999)}"}
        )
        logs.append(self.factory.create_record(
            base_time, 3000, "order-api", "ERROR", msg_app, label,
            trace_id=trace_id,
            metrics={"response_time_ms": 3005.0, "cpu_usage": 15.0, "memory_usage": 45.0},
            force_immediate_metrics=True
        ))
        
        # 2. Web log (504 Gateway Timeout)
        msg_web = self.fmt.format_nginx(
            "GET", "/history", 504, 3.005, base_time
        )
        logs.append(self.factory.create_record(
            base_time, 3005, "nginx", "ERROR", msg_web, label,
            trace_id=trace_id
        ))
        
        return logs


class ScenarioC(ScenarioGenerator):
    """DDoS (Security)"""
    
    def generate(self, base_time: datetime) -> List[LogRecord]:
        trace_id = str(uuid.uuid4())
        logs = []
        label = self.get_label()
        
        # 短期間に大量のエラー（5回）
        for i in range(5):
            offset = i * 2
            msg = self.fmt.format_nginx(
                "GET", "/", 503, 0.001, base_time,
                extra='error="Too many open files"'
            )
            logs.append(self.factory.create_record(
                base_time, offset, "nginx", "ERROR", msg, label,
                trace_id=trace_id,
                metrics={"cpu_usage": 95.0, "memory_usage": 75.0, "response_time_ms": 1.0},
                force_immediate_metrics=(i == 0)
            ))
        
        return logs


class ScenarioD(ScenarioGenerator):
    """Disk Full (Resource)"""
    
    def generate(self, base_time: datetime) -> List[LogRecord]:
        trace_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        logs = []
        label = self.get_label()
        
        # 1. Kernel log (I/O Error)
        msg_kernel = self.fmt.format_syslog(
            "kernel",
            f"blk_update_request: I/O error, dev sda, sector {random.randint(1000, 9999)}",
            base_time
        )
        logs.append(self.factory.create_record(
            base_time, 0, "kernel", "CRITICAL", msg_kernel, label,
            trace_id=None,
            correlation_id=correlation_id,
            metrics={"disk_usage": 100.0, "cpu_usage": 20.0, "memory_usage": 50.0},
            force_immediate_metrics=True
        ))
        
        # 2. App log (IOException)
        msg_app = self.fmt.format_app_json(
            "FATAL",
            "java.io.IOException: No space left on device",
            {"path": "/var/log/app.log"}
        )
        logs.append(self.factory.create_record(
            base_time, 10, "inventory-api", "FATAL", msg_app, label,
            trace_id=trace_id,
            correlation_id=correlation_id
        ))
        
        # 3. Web log (500 Internal Server Error)
        msg_web = self.fmt.format_nginx(
            "POST", "/upload", 500, 0.010, base_time
        )
        logs.append(self.factory.create_record(
            base_time, 20, "nginx", "ERROR", msg_web, label,
            trace_id=trace_id,
            correlation_id=correlation_id
        ))
        
        return logs


class ScenarioE(ScenarioGenerator):
    """External API Down (Dependency)"""
    
    def generate(self, base_time: datetime) -> List[LogRecord]:
        trace_id = str(uuid.uuid4())
        logs = []
        label = self.get_label()
        
        # 1. App log (Connection Timeout)
        msg_app = self.fmt.format_app_json(
            "ERROR",
            "TimeoutException: failed to connect to payment-gateway.com:443",
            {"timeout_ms": 5000}
        )
        logs.append(self.factory.create_record(
            base_time, 5000, "payment-api", "ERROR", msg_app, label,
            trace_id=trace_id,
            metrics={"response_time_ms": 5000.0, "cpu_usage": 12.0, "memory_usage": 40.0},
            force_immediate_metrics=True
        ))
        
        # 2. Web log (502 Bad Gateway)
        msg_web = self.fmt.format_nginx(
            "POST", "/pay", 502, 5.010, base_time
        )
        logs.append(self.factory.create_record(
            base_time, 5010, "nginx", "ERROR", msg_web, label,
            trace_id=trace_id
        ))
        
        return logs


class ScenarioF(ScenarioGenerator):
    """Logic Bug (Application)"""
    
    def generate(self, base_time: datetime) -> List[LogRecord]:
        trace_id = str(uuid.uuid4())
        logs = []
        label = self.get_label()
        
        # 1. App log (NullPointerException)
        msg_app = self.fmt.format_app_json(
            "ERROR",
            f"NullPointerException at Controller.Campaign (line {random.randint(30, 100)})",
            {"stack": "java.lang.NullPointerException at com.example..."}
        )
        logs.append(self.factory.create_record(
            base_time, 30, "order-api", "ERROR", msg_app, label,
            trace_id=trace_id
        ))
        
        # 2. Web log (500 Internal Server Error)
        msg_web = self.fmt.format_nginx(
            "GET", "/campaign", 500, 0.035, base_time
        )
        logs.append(self.factory.create_record(
            base_time, 35, "nginx", "ERROR", msg_web, label,
            trace_id=trace_id
        ))
        
        return logs


class ScenarioG(ScenarioGenerator):
    """SQL Injection (Security)"""
    
    def generate(self, base_time: datetime) -> List[LogRecord]:
        trace_id = str(uuid.uuid4())
        logs = []
        label = self.get_label()
        
        malicious_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users--",
            "' UNION SELECT * FROM passwords--"
        ]
        payload = random.choice(malicious_payloads)
        
        # 1. Web log (攻撃リクエスト - 200 OK)
        msg_web = self.fmt.format_nginx(
            "POST", f"/login?user={payload}", 200, 0.050, base_time
        )
        logs.append(self.factory.create_record(
            base_time, 0, "nginx", "INFO", msg_web, label,
            trace_id=trace_id
        ))
        
        # 2. App log (SQL Syntax Error)
        msg_app = self.fmt.format_app_json(
            "ERROR",
            f"SQLException: Syntax error near '{payload}'",
            {"query": "SELECT * FROM users WHERE username=..."}
        )
        logs.append(self.factory.create_record(
            base_time, 50, "order-api", "ERROR", msg_app, label,
            trace_id=trace_id
        ))
        
        return logs


class ScenarioH(ScenarioGenerator):
    """Async Worker Fail (Application)"""
    
    def generate(self, base_time: datetime) -> List[LogRecord]:
        trace_id = str(uuid.uuid4())
        logs = []
        label = self.get_label()
        
        # 1. Web log (200 OK - フロントは成功)
        msg_web = self.fmt.format_nginx(
            "POST", "/register", 200, 0.050, base_time
        )
        logs.append(self.factory.create_record(
            base_time, 0, "nginx", "INFO", msg_web, label,
            trace_id=trace_id
        ))
        
        # 2. Worker log (非同期ジョブ失敗 - 2秒後)
        jobs = ["SendWelcomeEmail", "CreateUserProfile", "NotifyAdmin"]
        msg_worker = self.fmt.format_app_json(
            "ERROR",
            f"Job '{random.choice(jobs)}' failed: MailServerRefused",
            {"job_id": str(uuid.uuid4()), "retry_count": 3}
        )
        logs.append(self.factory.create_record(
            base_time, 2000, "worker", "ERROR", msg_worker, label,
            trace_id=trace_id
        ))
        
        return logs


class ScenarioI(ScenarioGenerator):
    """SSL Expired (Configuration)"""
    
    def generate(self, base_time: datetime) -> List[LogRecord]:
        trace_id = str(uuid.uuid4())
        logs = []
        label = self.get_label()
        
        # 1. App log (SSL Handshake Error)
        msg_app = self.fmt.format_app_json(
            "ERROR",
            "SSLHandshakeException: Certificate expired",
            {"target": "internal-api.example.com", "expired_date": "2025-11-01"}
        )
        logs.append(self.factory.create_record(
            base_time, 20, "order-api", "ERROR", msg_app, label,
            trace_id=trace_id
        ))
        
        # 2. Web log (502 Bad Gateway)
        msg_web = self.fmt.format_nginx(
            "POST", "/internal-api", 502, 0.030, base_time
        )
        logs.append(self.factory.create_record(
            base_time, 30, "nginx", "ERROR", msg_web, label,
            trace_id=trace_id
        ))
        
        return logs


class ScenarioJ(ScenarioGenerator):
    """Memory Leak / Slow GC (Resource)"""
    
    def generate(self, base_time: datetime) -> List[LogRecord]:
        trace_id = str(uuid.uuid4())
        logs = []
        label = self.get_label()
        
        # 1. App log (GC Warning)
        heap_before = random.randint(1400, 1500)
        heap_after = random.randint(1350, 1450)
        heap_total = random.randint(1500, 1600)
        
        msg_app = self.fmt.format_app_json(
            "WARN",
            f"[GC (Allocation Failure) {heap_before}K->{heap_after}K({heap_total}K), 1.5 secs]",
            {"gc_type": "Full GC", "duration_ms": 1500}
        )
        logs.append(self.factory.create_record(
            base_time, 1500, "order-api", "WARN", msg_app, label,
            trace_id=trace_id,
            metrics={"memory_usage": 92.0, "response_time_ms": 1500.0, "cpu_usage": 35.0},
            force_immediate_metrics=True
        ))
        
        # 2. Web log (504 Gateway Timeout)
        msg_web = self.fmt.format_nginx(
            "GET", "/search", 504, 3.001, base_time
        )
        logs.append(self.factory.create_record(
            base_time, 3000, "nginx", "ERROR", msg_web, label,
            trace_id=trace_id
        ))
        
        return logs


# ==================== シナリオファクトリー（A-J用） ====================

class ScenarioFactoryAJ:
    """シナリオA-J生成のファクトリークラス"""
    
    def __init__(
        self,
        record_factory: LogRecordFactory,
        formatter: LogFormatter
    ):
        self.record_factory = record_factory
        self.formatter = formatter
        
        # シナリオマッピング（A-J）
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
        }
    
    def create(self, scenario_code: str) -> ScenarioGenerator:
        """
        シナリオコードに応じた生成器を作成
        
        Args:
            scenario_code: シナリオコード（A-J）
            
        Returns:
            シナリオ生成器
        """
        scenario_class = self.scenarios.get(scenario_code)
        
        if scenario_class is None:
            raise ValueError(f"Unknown scenario code: {scenario_code}")
        
        return scenario_class(
            self.record_factory,
            self.formatter,
            scenario_code
        )