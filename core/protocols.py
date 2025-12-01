"""
Core Protocols Module

依存性の逆転のためのProtocol定義（抽象層）
Protocolは抽象層として独立したモジュールに配置し、
具象実装とは完全に分離する
"""

from typing import Dict, Protocol


class HostStateManagerProtocol(Protocol):
    """
    ホスト状態管理のプロトコル（依存性の逆転）
    
    このProtocolは抽象層として定義され、具象実装に依存しない。
    新しい実装を追加する際は、このProtocolを実装するだけでよい。
    """
    
    def get_state(self, host: str) -> Dict[str, float]:
        """
        ホストの現在状態を取得
        
        Args:
            host: ホスト名
            
        Returns:
            ホストの現在状態（メトリクスの辞書）
        """
        ...
    
    def update_state(
        self, 
        host: str, 
        target_state: Dict[str, float],
        immediate: bool = False
    ) -> Dict[str, float]:
        """
        ホストの状態を更新
        
        Args:
            host: ホスト名
            target_state: 目標状態
            immediate: Trueの場合、即座に目標状態に変更
            
        Returns:
            更新後の状態
        """
        ...

