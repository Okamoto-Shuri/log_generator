# Enhanced Log Generator 

機械学習用の高品質な合成ログデータセット生成ツール

## 🎯 プロジェクト構成

```
log_generator/
├── __init__.py                    # ルートパッケージ初期化
├── core/
│   ├── __init__.py               # coreパッケージ公開API
│   ├── config.py                 # 設定クラス、メタデータ（Part 1）
│   ├── generators.py             # ベクトル/メトリクス生成（Part 2）
│   └── statistics.py             # 統計収集・検証（Part 3-3）
├── scenarios/
│   ├── __init__.py               # scenariosパッケージ公開API
│   ├── base.py                   # シナリオ基底クラス（Part 3-1）
│   ├── scenarios_a_j.py          # シナリオA〜J（Part 3-1）
│   └── scenarios_k_u.py          # シナリオK〜U（Part 3-2）
├── cli/
│   ├── __init__.py               # cliパッケージ公開API
│   └── commands.py               # CLIコマンド実装（Part 3-4a）
├── main_generator.py             # メインジェネレータ（Part 3-3）
├── cli.py                        # エントリーポイント
└── README.md                     # このファイル
```

---

## 🚀 インストールと起動

### 必要な環境

- Python 3.8以上
- （オプション）tqdm（プログレスバー用）

### インストール

```bash
# tqdmのインストール（推奨）
pip install tqdm

# プロジェクトディレクトリに移動
cd log_generator/
```

### 基本的な使用方法

```bash
# 1. 基本的な生成（2000イベント、異常率20%）
python cli.py generate

# 2. イベント数指定
python cli.py generate --events 10000

# 3. 異常率30%で生成
python cli.py generate --events 5000 --abnormal-ratio 0.3

# 4. 再現性のある生成（ランダムシード指定）
python cli.py generate --events 2000 --seed 42

# 5. 出力ファイル指定
python cli.py generate --events 1000 --output my_dataset.jsonl

# 6. 生成後に自動検証
python cli.py generate --events 1000 --validate

# 7. データセットの検証のみ
python cli.py validate training_dataset_v3.jsonl

# 8. シナリオ情報の表示
python cli.py info
```

---

## 📖 コマンドリファレンス

### `generate` コマンド

ログデータセットを生成します。

```bash
python cli.py generate [OPTIONS]
```

**オプション:**

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--events N` | 2000 | 生成するイベント数 |
| `--abnormal-ratio R` | 0.2 | 異常イベントの比率（0.0〜1.0） |
| `--output FILE` | training_dataset_v3.jsonl | 出力ファイルパス |
| `--embedding-dim N` | 384 | ベクトルの次元数 |
| `--batch-size N` | 1000 | バッチ書き込みサイズ |
| `--start-days-ago N` | 1 | N日前から開始 |
| `--seed N` | None | ランダムシード（再現性） |
| `--force, -f` | - | 確認なしで上書き |
| `--yes, -y` | - | 全プロンプトにYes |
| `--quiet, -q` | - | バナー非表示 |
| `--validate` | - | 生成後に自動検証 |
| `--disable-time-correlation` | - | 時系列相関を無効化 |
| `--disable-host-state` | - | ホスト状態管理を無効化 |

### `validate` コマンド

生成されたデータセットを検証します。

```bash
python cli.py validate <input_file> [OPTIONS]
```

**オプション:**

| オプション | 説明 |
|-----------|------|
| `--json-output FILE` | 検証結果をJSONで保存 |

### `info` コマンド

シナリオ情報を表示します。

```bash
python cli.py info
```

---

## 💻 Pythonコードから使用

### 基本的な使用

```python
from log_generator import GeneratorConfig, EnhancedLogGenerator

# 設定
config = GeneratorConfig(
    output_file="my_dataset.jsonl",
    total_events=10000,
    abnormal_ratio=0.3,
    random_seed=42
)

# 生成
generator = EnhancedLogGenerator(config)
generator.run()
```

### カスタム設定

```python
from log_generator.core import GeneratorConfig
from log_generator.main_generator import EnhancedLogGenerator

config = GeneratorConfig(
    output_file="custom_dataset.jsonl",
    total_events=5000,
    abnormal_ratio=0.25,
    embedding_dim=512,  # ベクトル次元を変更
    batch_size=500,
    random_seed=12345,
    enable_time_correlation=True,
    enable_host_state=True,
    service_topology={
        "nginx": ["web-01", "web-02", "web-03"],
        "api": ["app-01", "app-02"]
    }
)

generator = EnhancedLogGenerator(config)
generator.run()
```

### データセット検証

```python
from log_generator.core import DatasetValidator

validator = DatasetValidator()
results = validator.validate_file("training_dataset_v3.jsonl")

if results["valid"]:
    print("✅ Dataset is valid!")
    print(f"Total logs: {results['total_lines']}")
    print(f"Unique traces: {results['unique_traces']}")
else:
    print("❌ Dataset has issues")
    print(f"Invalid lines: {len(results['invalid_lines'])}")
```

---

## 📋 生成されるデータフォーマット

### JSONL形式（各行が1つのログレコード）

```json
{
  "timestamp": "2025-11-27T15:30:45.123Z",
  "service": "order-api",
  "host": "app-01",
  "level": "ERROR",
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "correlation_id": "650e8400-e29b-41d4-a716-446655440001",
  "message": "{\"lvl\":\"ERROR\",\"msg\":\"Connection timeout\"}",
  "metrics": {
    "cpu_usage": 45.2,
    "memory_usage": 78.5,
    "response_time_ms": 3005.0,
    "disk_usage": 55.0,
    "network_latency_ms": 150.0
  },
  "label": {
    "scenario": "B",
    "root_cause": "network_db_latency",
    "category": "network",
    "severity": "warning",
    "impact": "performance_degradation"
  },
  "message_vector": [0.123, -0.456, 0.789, ...]
}
```

### フィールド説明

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `timestamp` | string | ISO 8601形式のタイムスタンプ |
| `service` | string | サービス名（nginx, order-api等） |
| `host` | string | ホスト名 |
| `level` | string | ログレベル（INFO, WARN, ERROR等） |
| `trace_id` | string/null | 分散トレーシングID |
| `correlation_id` | string | ログ間の相関ID |
| `message` | string | ログメッセージ本文 |
| `metrics` | object | システムメトリクス |
| `label` | object | 異常検知用ラベル |
| `message_vector` | array | 384次元のセマンティックベクトル |

---

## 🎓 シナリオ一覧（21種類）

| コード | 原因 | カテゴリ | 重要度 | 重み |
|--------|------|----------|--------|------|
| A | OOM Killer | resource | critical | 2% |
| B | DB Latency | network | warning | 10% |
| C | DDoS | security | critical | 1% |
| D | Disk Full | resource | critical | 3% |
| E | External API Down | dependency | error | 5% |
| F | Logic Bug | application | error | 15% |
| G | SQL Injection | security | critical | 1% |
| H | Async Worker Fail | application | error | 5% |
| I | SSL Expired | configuration | critical | 1% |
| J | Memory Leak | resource | warning | 5% |
| K | Auth Mismatch | configuration | fatal | 5% |
| L | DNS Failure | network | error | 3% |
| M | DB Deadlock | application | error | 3% |
| N | Pool Exhausted | middleware | error | 3% |
| O | Payload Limit | security | warning | 1% |
| P | Data Integrity | application | error | 5% |
| Q | Clock Skew | configuration | error | 1% |
| R | Permission Denied | configuration | error | 2% |
| S | I/O Wait | infrastructure | warning | 5% |
| T | Split Brain | infrastructure | critical | 1% |
| U | App Timeout | application | error | 23% |

**合計: 100%（正確に1.0）**

---

## 🔬 主要な改善点（元コードから）

### ✅ Critical Issues（完全解決）

1. **重みの正規化**: 0.77 → 1.0000（+29.9%）
2. **correlation_id一貫性**: 明示的な優先順位実装
3. **メトリクス時系列相関**: ホスト状態管理（指数移動平均）
4. **ベクトル品質**: ノイズ削減（0.1→0.05）、信号強化（0.5→0.8）

### ✅ Major Issues（完全解決）

5. **エラーハンドリング**: Ctrl+C対応、部分保存機能
6. **メモリ効率**: バッチ処理（1000件単位）
7. **統計出力**: カテゴリ別、重大度別、パフォーマンス情報

---

## 📈 パフォーマンス

### ベンチマーク（参考値）

| イベント数 | 生成時間 | スループット | ファイルサイズ |
|-----------|---------|-------------|--------------|
| 1,000 | 2-3秒 | 400 events/s | 2.3 MB |
| 10,000 | 15-20秒 | 550 events/s | 23 MB |
| 100,000 | 2-3分 | 600 events/s | 230 MB |

---

## 🧪 データセット活用例

### PyTorchでの使用

```python
import json
import torch
from torch.utils.data import Dataset, DataLoader

class LogDataset(Dataset):
    def __init__(self, jsonl_path):
        with open(jsonl_path) as f:
            self.data = [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        log = self.data[idx]
        vector = torch.tensor(log["message_vector"], dtype=torch.float32)
        label = 0 if log["label"]["scenario"] == "normal" else 1
        return vector, label

# 使用例
dataset = LogDataset("training_dataset_v3.jsonl")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for vectors, labels in loader:
    # モデル学習
    pass
```

---

## 🔍 トラブルシューティング

### Q: プログレスバーが表示されない

```bash
pip install tqdm
```

### Q: メモリ不足エラー

バッチサイズを小さくしてください：

```bash
python cli.py generate --events 100000 --batch-size 500
```

### Q: 生成が遅い

- `--seed`を指定すると若干高速化します
- SSDを使用してください

### Q: ModuleNotFoundError

プロジェクトのルートディレクトリで実行してください：

```bash
cd log_generator/
python cli.py generate
```

---

## 📝 ライセンス

MIT License

