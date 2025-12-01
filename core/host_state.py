"""
Core Host State Module

ホスト状態管理の具象実装
Protocolを実装する具象クラスを提供する
"""

from typing import Dict

from .protocols import HostStateManagerProtocol
from .config import get_config, logger


class HostStateManager:
    """
    ホストの状態を管理するクラス（時系列相関用）
    
    HostStateManagerProtocolを実装する具象クラス。
    設定ファイルから設定を読み込み、ホストの状態を管理する。
    """
    
    def __init__(self, smoothing_factor: float = None):
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
        """
        ホストの現在状態を取得
        
        Args:
            host: ホスト名
            
        Returns:
            ホストの現在状態（コピー）
        """
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
            更新後の状態（コピー）
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

