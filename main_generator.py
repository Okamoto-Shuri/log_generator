"""
Main Generator Module - Part 3-3 Integration

メインのログジェネレータクラス
"""

import json
import random
import sys
from typing import List
from datetime import datetime, timedelta

# coreパッケージからのインポート
from core import (
    logger,
    GeneratorConfig,
    WeightNormalizer,
    HostStateManager,
    SemanticVectorGenerator,
    MetricsGenerator,
    LogFormatter,
    LogRecordFactory,
    LogRecord,
    TimeManager,
    StatisticsCollector,
    SCENARIO_META,
    initialize_generator
)

# scenariosパッケージからのインポート
from scenarios import (
    NormalScenarioGenerator,
    CompleteScenarioFactory
)


# ==================== メインジェネレータ ====================

class EnhancedLogGenerator:
    """改善版ログジェネレータ（v3.0）"""
    
    def __init__(self, config: GeneratorConfig):
        """
        Args:
            config: ジェネレータ設定
        """
        self.config = config
        self.stats = StatisticsCollector()
        
        # 初期化処理
        logger.info("Initializing Enhanced Log Generator v3.0...")
        initialize_generator(config)
        
        # コンポーネントの初期化
        self.host_state_manager = HostStateManager() if config.enable_host_state else None
        self.vector_generator = SemanticVectorGenerator(config.embedding_dim)
        self.metrics_generator = MetricsGenerator(self.host_state_manager)
        self.formatter = LogFormatter()
        self.record_factory = LogRecordFactory(
            config,
            self.vector_generator,
            self.metrics_generator
        )
        self.time_manager = TimeManager(
            datetime.now() - timedelta(days=config.start_time_days_ago)
        )
        
        # シナリオジェネレータ
        self.normal_generator = NormalScenarioGenerator(
            self.record_factory,
            self.formatter
        )
        self.scenario_factory = CompleteScenarioFactory(
            self.record_factory,
            self.formatter
        )
        
        # 重みの正規化
        normalizer = WeightNormalizer()
        self.normalized_weights = normalizer.normalize_weights(SCENARIO_META)
        
        logger.info("Initialization completed")
    
    def _prepare_event_schedule(self) -> List[str]:
        """
        イベントスケジュールを事前に作成
        
        Returns:
            イベントタイプのリスト（"normal" または シナリオコード）
        """
        logger.info("Preparing event schedule...")
        
        # 異常・正常の件数決定
        abnormal_count = int(self.config.total_events * self.config.abnormal_ratio)
        normal_count = self.config.total_events - abnormal_count
        
        # 異常シナリオの割り当て（重み付き）
        scenarios = list(self.normalized_weights.keys())
        weights = list(self.normalized_weights.values())
        abnormal_events = random.choices(scenarios, weights=weights, k=abnormal_count)
        
        # 全イベントリストを作成してシャッフル
        all_events = abnormal_events + ["normal"] * normal_count
        random.shuffle(all_events)
        
        logger.info(
            f"Schedule prepared: {normal_count} normal, "
            f"{abnormal_count} abnormal events"
        )
        
        return all_events
    
    def _generate_event(self, event_type: str) -> List[LogRecord]:
        """
        単一のイベントを生成
        
        Args:
            event_type: "normal" または シナリオコード
            
        Returns:
            ログレコードのリスト
        """
        base_time = self.time_manager.get_current_time()
        
        try:
            if event_type == "normal":
                logs = self.normal_generator.generate(base_time)
            else:
                scenario = self.scenario_factory.create(event_type)
                logs = scenario.generate(base_time)
            
            # 統計記録
            self.stats.record_scenario(event_type, len(logs))
            
            return logs
        
        except Exception as e:
            logger.error(f"Failed to generate event {event_type}: {e}")
            # フォールバック: 正常系を生成
            logs = self.normal_generator.generate(base_time)
            self.stats.record_scenario("normal", len(logs))
            return logs
    
    def _write_batch(
        self,
        file_handle,
        batch: List[LogRecord]
    ) -> None:
        """
        バッチをファイルに書き込み
        
        Args:
            file_handle: ファイルハンドル
            batch: ログレコードのバッチ
        """
        if not batch:
            return
        
        # タイムスタンプでソート
        batch.sort(key=lambda x: x.timestamp)
        
        # JSON Lines形式で書き込み
        for record in batch:
            json_line = json.dumps(record.to_dict(), ensure_ascii=False)
            file_handle.write(json_line + "\n")
        
        logger.debug(f"Wrote batch of {len(batch)} records")
    
    def _save_partial_results(
        self,
        logs: List[LogRecord],
        reason: str = "interrupted"
    ) -> None:
        """
        部分的な結果を保存
        
        Args:
            logs: 保存するログレコード
            reason: 保存理由
        """
        if not logs:
            logger.warning("No logs to save")
            return
        
        partial_file = self.config.output_file.replace(
            ".jsonl",
            f"_partial_{reason}.jsonl"
        )
        
        try:
            with open(partial_file, "w", encoding="utf-8") as f:
                self._write_batch(f, logs)
            
            logger.info(
                f"Partial results ({len(logs)} logs) saved to {partial_file}"
            )
            print(f"\n⚠️  Partial results saved to: {partial_file}")
        
        except Exception as e:
            logger.error(f"Failed to save partial results: {e}")
    
    def run(self) -> None:
        """メイン実行ロジック"""
        logger.info("Starting log generation...")
        self.stats.start_timing()
        
        print("\n" + "=" * 70)
        print(f"🚀 Enhanced Log Generator v3.0")
        print("=" * 70)
        print(f"Total Events:     {self.config.total_events:,}")
        print(f"Abnormal Ratio:   {self.config.abnormal_ratio:.1%}")
        print(f"Batch Size:       {self.config.batch_size:,}")
        print(f"Output File:      {self.config.output_file}")
        print("=" * 70 + "\n")
        
        # イベントスケジュール作成
        event_schedule = self._prepare_event_schedule()
        
        # バッチバッファ
        batch_buffer: List[LogRecord] = []
        
        try:
            # プログレスバー（tqdmが利用可能な場合）
            try:
                from tqdm import tqdm
                event_iterator = tqdm(
                    event_schedule,
                    desc="Generating logs",
                    unit="event"
                )
            except ImportError:
                event_iterator = event_schedule
                logger.info("Install tqdm for progress bar: pip install tqdm")
                print("⏳ Generating logs (install tqdm for progress bar)...\n")
            
            # ファイルを開いてバッチ処理
            with open(self.config.output_file, "w", encoding="utf-8") as f:
                for event_type in event_iterator:
                    # 時刻を進める
                    self.time_manager.advance()
                    
                    # イベント生成
                    logs = self._generate_event(event_type)
                    batch_buffer.extend(logs)
                    
                    # バッチサイズに達したら書き込み
                    if len(batch_buffer) >= self.config.batch_size:
                        self._write_batch(f, batch_buffer)
                        batch_buffer.clear()
                
                # 残りのバッファを書き込み
                if batch_buffer:
                    self._write_batch(f, batch_buffer)
                    batch_buffer.clear()
            
            self.stats.end_timing()
            logger.info("Log generation completed successfully")
            
            # 統計出力
            self.stats.print_summary(self.config)
            
            print("\n✅ Generation completed successfully!")
            print(f"📁 Output: {self.config.output_file}")
        
        except KeyboardInterrupt:
            logger.warning("Generation interrupted by user")
            self.stats.end_timing()
            
            # 部分的な結果を保存
            if batch_buffer:
                self._save_partial_results(batch_buffer, "interrupted")
            
            print("\n\n⚠️  Generation interrupted by user")
            self.stats.print_summary(self.config)
            
            sys.exit(130)
        
        except IOError as e:
            logger.error(f"File I/O error: {e}")
            
            # 部分的な結果を保存
            if batch_buffer:
                self._save_partial_results(batch_buffer, "io_error")
            
            print(f"\n❌ File I/O error: {e}")
            sys.exit(1)
        
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            
            # 部分的な結果を保存
            if batch_buffer:
                self._save_partial_results(batch_buffer, "error")
            
            print(f"\n❌ Unexpected error: {e}")
            print("Check logs for details")
            sys.exit(1)