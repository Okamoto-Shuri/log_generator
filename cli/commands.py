"""
CLI Commands Module - Part 3-4a Integration

コマンドラインインターフェースの実装
"""

import argparse
import sys
import json
from pathlib import Path
from collections import defaultdict

# coreパッケージからのインポート
from core import (
    logger,
    VERSION,
    BUILD_DATE,
    GeneratorConfig,
    SCENARIO_META,
    WeightNormalizer,
    DatasetValidator
)

# main_generatorからのインポート
from main_generator import EnhancedLogGenerator


# ==================== CLIヘルパー ====================

class CLIHelper:
    """CLIユーティリティ関数"""
    
    @staticmethod
    def print_banner():
        """バナーを表示"""
        print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║          Enhanced Log Generator                                    ║
║          Synthetic Log Dataset Generator for ML Training         ║
║                                                                  ║
║          Version: 1.0.0                                          ║
║          Build: 2025-11-28                                       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
        """)
    
    @staticmethod
    def print_config_summary(config: GeneratorConfig):
        """設定サマリーを表示"""
        print("\n📋 Configuration Summary:")
        print("─" * 60)
        print(f"  Output File:         {config.output_file}")
        print(f"  Total Events:        {config.total_events:,}")
        print(f"  Abnormal Ratio:      {config.abnormal_ratio:.1%}")
        print(f"  Embedding Dim:       {config.embedding_dim}")
        print(f"  Batch Size:          {config.batch_size:,}")
        print(f"  Start Time:          {config.start_time_days_ago} days ago")
        print(f"  Random Seed:         {config.random_seed or 'None (random)'}")
        print(f"  Time Correlation:    {'Enabled' if config.enable_time_correlation else 'Disabled'}")
        print(f"  Host State Mgmt:     {'Enabled' if config.enable_host_state else 'Disabled'}")
        print("─" * 60)
    
    @staticmethod
    def confirm_action(message: str, default: bool = True) -> bool:
        """
        ユーザーに確認を求める
        
        Args:
            message: 確認メッセージ
            default: デフォルト値
            
        Returns:
            ユーザーの選択
        """
        suffix = " [Y/n]: " if default else " [y/N]: "
        
        while True:
            response = input(message + suffix).strip().lower()
            
            if response == "":
                return default
            elif response in ["y", "yes"]:
                return True
            elif response in ["n", "no"]:
                return False
            else:
                print("Please enter 'y' or 'n'")


# ==================== サブコマンド: generate ====================

def cmd_generate(args: argparse.Namespace) -> int:
    """
    ログ生成コマンド
    
    Args:
        args: コマンドライン引数
        
    Returns:
        終了コード
    """
    # 設定の構築
    config = GeneratorConfig(
        output_file=args.output,
        total_events=args.events,
        start_time_days_ago=args.start_days_ago,
        embedding_dim=args.embedding_dim,
        abnormal_ratio=args.abnormal_ratio,
        batch_size=args.batch_size,
        random_seed=args.seed,
        enable_time_correlation=args.enable_time_correlation,
        enable_host_state=args.enable_host_state
    )
    
    # バナー表示
    if not args.quiet:
        CLIHelper.print_banner()
        CLIHelper.print_config_summary(config)
    
    # 出力ファイルの存在チェック
    output_path = Path(config.output_file)
    if output_path.exists() and not args.force:
        print(f"\n⚠️  Output file already exists: {config.output_file}")
        
        if not args.yes and not CLIHelper.confirm_action("Overwrite?", default=False):
            print("Operation cancelled.")
            return 1
    
    # ジェネレータの実行
    try:
        generator = EnhancedLogGenerator(config)
        generator.run()
        
        # 自動検証
        if args.validate:
            print("\n🔍 Running automatic validation...")
            validator = DatasetValidator()
            results = validator.validate_file(config.output_file)
            validator.print_validation_results(results)
            
            if not results.get("valid", False):
                print("\n⚠️  Validation found issues (see above)")
                return 2
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation cancelled by user")
        return 130
    
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        logger.exception("Generation failed")
        return 1


# ==================== サブコマンド: validate ====================

def cmd_validate(args: argparse.Namespace) -> int:
    """
    データセット検証コマンド
    
    Args:
        args: コマンドライン引数
        
    Returns:
        終了コード
    """
    input_path = Path(args.input)
    
    # ファイルの存在チェック
    if not input_path.exists():
        print(f"❌ File not found: {args.input}")
        return 1
    
    print(f"\n🔍 Validating dataset: {args.input}")
    print("─" * 60)
    
    # 検証実行
    validator = DatasetValidator()
    results = validator.validate_file(args.input)
    
    # 結果表示
    validator.print_validation_results(results)
    
    # JSON出力
    if args.json_output:
        json_path = args.json_output
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n📄 Results saved to: {json_path}")
        except Exception as e:
            print(f"\n⚠️  Failed to save JSON: {e}")
    
    # 終了コード
    return 0 if results.get("valid", False) else 2


# ==================== サブコマンド: info ====================

def cmd_info(args: argparse.Namespace) -> int:
    """
    シナリオ情報表示コマンド
    
    Args:
        args: コマンドライン引数
        
    Returns:
        終了コード
    """
    print("\n" + "=" * 70)
    print("=" * 22 + " SCENARIO INFORMATION " + "=" * 27)
    print("=" * 70)
    
    # 全シナリオ情報の表示
    print(f"\n📚 Total Scenarios: {len(SCENARIO_META)}")
    print("\n📋 Scenario Details:")
    print("─" * 70)
    
    # カテゴリでグループ化
    by_category = defaultdict(list)
    
    for code, meta in sorted(SCENARIO_META.items()):
        by_category[meta.category].append((code, meta))
    
    # カテゴリごとに表示
    for category in sorted(by_category.keys()):
        scenarios = by_category[category]
        print(f"\n🏷️  Category: {category.upper()} ({len(scenarios)} scenarios)")
        
        for code, meta in scenarios:
            print(f"\n  [{code}] {meta.cause}")
            print(f"      Severity:   {meta.severity}")
            print(f"      Impact:     {meta.impact}")
            print(f"      Weight:     {meta.weight:.4f}")
    
    # 重みの検証
    print("\n" + "─" * 70)
    print("\n⚖️  Weight Distribution:")
    
    normalizer = WeightNormalizer()
    normalized = normalizer.normalize_weights(SCENARIO_META)
    
    total_raw = sum(meta.weight for meta in SCENARIO_META.values())
    total_normalized = sum(normalized.values())
    
    print(f"  Raw Total:        {total_raw:.6f}")
    print(f"  Normalized Total: {total_normalized:.10f}")
    print(f"  Status:           {'✅ Valid' if abs(total_normalized - 1.0) < 1e-6 else '⚠️  Invalid'}")
    
    # 統計
    print("\n" + "─" * 70)
    print("\n📊 Statistics:")
    
    severity_counts = defaultdict(int)
    category_counts = defaultdict(int)
    
    for meta in SCENARIO_META.values():
        severity_counts[meta.severity] += 1
        category_counts[meta.category] += 1
    
    print("\n  By Severity:")
    for severity in ["critical", "fatal", "error", "warning", "info"]:
        if severity in severity_counts:
            print(f"    {severity:<10} {severity_counts[severity]:>3}")
    
    print("\n  By Category:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {category:<15} {count:>3}")
    
    print("\n" + "=" * 70)
    
    return 0


# ==================== メインCLI ====================

def create_parser() -> argparse.ArgumentParser:
    """
    ArgumentParserを作成
    
    Returns:
        設定済みのArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Enhanced Log Generator - Synthetic log dataset generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10,000 events with 20%% abnormal ratio
  %(prog)s generate --events 10000 --abnormal-ratio 0.2

  # Generate with specific random seed for reproducibility
  %(prog)s generate --events 5000 --seed 42

  # Validate a generated dataset
  %(prog)s validate training_dataset.jsonl

  # Show scenario information
  %(prog)s info

For more information, see the documentation.
        """
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION} (build {BUILD_DATE})"
    )
    
    # サブコマンド
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # generate コマンド
    parser_gen = subparsers.add_parser(
        "generate",
        help="Generate synthetic log dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser_gen.add_argument(
        "--events",
        type=int,
        default=2000,
        help="Total number of events (transactions) to generate"
    )
    
    parser_gen.add_argument(
        "--abnormal-ratio",
        type=float,
        default=0.2,
        help="Ratio of abnormal events (0.0 to 1.0)"
    )
    
    parser_gen.add_argument(
        "--output", "-o",
        type=str,
        default="training_dataset.jsonl",
        help="Output JSONL file path"
    )
    
    parser_gen.add_argument(
        "--embedding-dim",
        type=int,
        default=384,
        help="Dimension of embedding vectors"
    )
    
    parser_gen.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for file writing"
    )
    
    parser_gen.add_argument(
        "--start-days-ago",
        type=int,
        default=1,
        help="Start generating logs from N days ago"
    )
    
    parser_gen.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser_gen.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite output file without confirmation"
    )
    
    parser_gen.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Answer yes to all prompts"
    )
    
    parser_gen.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress banner and summary output"
    )
    
    parser_gen.add_argument(
        "--validate",
        action="store_true",
        help="Automatically validate after generation"
    )
    
    parser_gen.add_argument(
        "--disable-time-correlation",
        dest="enable_time_correlation",
        action="store_false",
        default=True,
        help="Disable time correlation features"
    )
    
    parser_gen.add_argument(
        "--disable-host-state",
        dest="enable_host_state",
        action="store_false",
        default=True,
        help="Disable host state management"
    )
    
    # validate コマンド
    parser_val = subparsers.add_parser(
        "validate",
        help="Validate generated dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser_val.add_argument(
        "input",
        type=str,
        help="Input JSONL file to validate"
    )
    
    parser_val.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Save validation results as JSON"
    )
    
    # info コマンド
    parser_info = subparsers.add_parser(
        "info",
        help="Show scenario information"
    )
    
    return parser


def main() -> int:
    """
    メインエントリーポイント
    
    Returns:
        終了コード
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # コマンドが指定されていない場合
    if args.command is None:
        parser.print_help()
        return 1
    
    # コマンドの実行
    try:
        if args.command == "generate":
            return cmd_generate(args)
        elif args.command == "validate":
            return cmd_validate(args)
        elif args.command == "info":
            return cmd_info(args)
        else:
            parser.print_help()
            return 1
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.exception("Fatal error in main")
        return 1