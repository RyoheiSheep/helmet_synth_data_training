
# ヘルメットあごひもゆるみ検出 VLM 自己改善パイプライン 設計書

**バージョン:** 0.3  
**作成日:** 2026-04-04  
**ステータス:** 設計中（実装前）  
**変更履歴:** v0.2 → v0.3 レビュー反映（KISS原則適用、Step B簡素化、Step F延期、評価設計強化）

---

## 1. 概要と目的

### 問題設定
- ヘルメットのあごひものゆるみ（tight／loose）を画像から自動判定する軽量VLMを構築する
- 実データのアノテーションコストを最小化するため、**合成データ中心の自己改善ループ**を採用する
- 対象言語：英語のみ

### 基本仮説
1. 画像生成AIによる「編集」ならば、シードのあごひも状態をある程度忠実に引き継いだ多様な画像が生成できる
2. 一定割合（8〜9割）で画像とラベルが一致していれば、VLMのfine-tuningは進む
3. ループを重ねるごとに「より難しい画像」でも識別できるようになる（自己改善）

### [v0.3変更] 設計原則
- **KISS原則:** 各ステップは最小限の構成要素で始め、効果が検証されてから複雑化する
- **検証駆動:** 仮説を明示し、各ループで仮説の成否を数値で判断する
- **段階的拡張:** v0.3ではまず1ループを完走させることを最優先とする。自己改善ループの拡張は1ループ完走後に判断する

---

## 2. 全体アーキテクチャ

### [v0.3変更] 簡素化されたパイプライン（4ステップ）

```
[シード画像 ≤100枚, ラベル: tight/loose]
         │
         ▼
  [Step A] Image Generation
    画像生成AIで編集。ループNに応じて
    背景・人物・アングル変化量を段階的に拡大
         │
         ▼ (generated images + inherited seed label)
  [Step B] Teacher Screening & Labeling
    Teacher VLM 単独で：
      1. ラベル検証（seed_label と一致するか）→ keep/reject
      2. keep画像に対して rationale 生成（オプション）
    → JSONL出力
         │
         ▼ (image, label, [rationale])
  [Step C] Dataset Build
    固定質問プロンプト + 構造化出力スキーマを付与
    JSONL形式（HuggingFace datasets互換）
         │
         ▼
  [Step D] Fine-tuning
    軽量VLMをLoRAでfine-tune
         │
         ▼ (fine-tuned model loop N)
  [Evaluation Gate]
    テストセットで評価 → 終了条件を満たすか判定
         │
    [次のループへ → Step A]
```

**v0.2からの変更点:**
- 旧Step B（3モデル投票）→ Teacher単独判定に簡素化
- 旧Step C（Rationale生成）→ Step B内に統合（ラベル検証と同時にTeacherが実行。1回の推論で完了）
- 旧Step D（Dataset Build）→ Step Cに繰り上げ
- 旧Step E（Fine-tuning）→ Step Dに繰り上げ
- 旧Step F（Self-Screening）→ v0.3では実装しない（後述の拡張オプション参照）

### ループ戦略

| ループ | `angle_deg` | `chinstrap_delta` | `background` | `person` | 目的 |
|---|---|---|---|---|---|
| Loop 1 | ≤5° | 0 (keep) | low | low | ラベル一致率を高く保つ |
| Loop 2 | ≤15° | 1 (slight) | medium | medium | 多様性を増やす |
| Loop 3〜 | ≤30° | 2 (moderate) | high | high | 難易度を増す |

[v0.3変更] 各パラメータの数値範囲を明記。再現性を確保する。

---

## 3. 各ステップの詳細設計

### Step A: 画像生成（Image Generation）

**目的:** シードのあごひも状態を保ちつつ、多様な合成画像を生成する

**入力:**
```
seeds/
  ├── images/       # シード画像（JPG/PNG）
  └── labels.csv    # image_id, label(tight/loose)
```

**処理:**
- Image Edit API（Flux-Kontext, DALL·E Edit, InstructPix2Pix等）を利用
- プロンプトテンプレートで `background`, `person`, `clothing`, `camera_angle` を制御
- ループ番号で変化量パラメータを段階的に増やす

**プロンプトテンプレート（ループ1）:**
```
Edit this image: change the background to [BACKGROUND] and
clothing to [CLOTHING]. Keep the helmet chinstrap condition
exactly as-is. Camera angle: [ANGLE_DESC].
```

**出力:**
```
generated/loop_{N}/
  ├── images/       # 生成画像
  └── meta.csv      # image_id, seed_id, seed_label, prompt, loop_num
```

**設定（config/step_a.yaml）:**
```yaml
loop: 1
variations_per_seed: 10
diversity:
  person: low          # low / medium / high
  background: low
  angle_deg: 5         # seed cameraからの最大偏差
  chinstrap_delta: 0   # 0=keep, 1=slight, 2=moderate
api:
  provider: "replicate"
  model: "black-forest-labs/flux-kontext"
```

---

### [v0.3変更] Step B: Teacher スクリーニング＆ラベリング

**目的:** Teacher VLM 単独で (1) seed_label の整合性検証 (2) 学習用データの生成 を1パスで行う

**v0.2との差分:**
- 3モデル投票を廃止。Teacher VLM 単独判定に変更。
- Rationale生成をこのステップに統合。Teacherへの推論は1回で完了。
- Base VLM・Fine-Tuned VLMの推論が不要になり、GPU使用量・実装複雑度が大幅に低下。

**ラベリングロジック:**
```
keep 条件: Teacher の予測 == seed_label
reject:    Teacher の予測 ≠ seed_label
```

**Teacher へのプロンプト（固定）:**
```
You are a safety inspection assistant.
Look at the image and determine the helmet chinstrap status.

Respond ONLY in the following JSON format:
{
  "label": "tight" | "loose",
  "rationale": "<one sentence explaining visible evidence>"
}
```

**[v0.3変更] 構造化出力の強制:**
- vLLM の `guided_decoding`（JSON schema mode）を使用し、出力フォーマットを強制する
- これによりJSONパースエラーのリトライが不要になる

```yaml
# vLLM推論時の設定
guided_decoding:
  backend: "outlines"
  json_schema:
    type: object
    required: ["label", "rationale"]
    properties:
      label:
        type: string
        enum: ["tight", "loose"]
      rationale:
        type: string
    additionalProperties: false
```

**入力:**
```
generated/loop_{N}/images/
generated/loop_{N}/meta.csv      # seed_label を含む
```

**出力:**
```
screened/loop_{N}/
  ├── screening.csv
  │   # image_id, seed_label, pred_teacher, keep
  └── labeled.jsonl
      # keep=True のもののみ
      # {"image_id": "...", "label": "tight", "rationale": "..."}
```

**設定（config/step_b.yaml）:**
```yaml
teacher_model: "Qwen/Qwen2.5-VL-72B-Instruct"
teacher_backend: "vllm"
guided_decoding:
  backend: "outlines"
batch_size: 20
rationale: true            # false にすると label-only で高速化
```

---

### Step C: データセット構築（Dataset Build）

**目的:** fine-tuning用のJSONLを作る。質問プロンプトは固定・単一。

**固定質問プロンプト（学習・推論で共通）:**
```
Is the helmet chinstrap properly fastened?
Answer with JSON: {"label": "tight"|"loose", "rationale": "<reason>"}
```

**[v0.3変更] label-onlyモード:** rationale を使わない場合の質問プロンプト:
```
Is the helmet chinstrap properly fastened?
Answer with JSON: {"label": "tight"|"loose"}
```

**入力:**
```
screened/loop_{N}/labeled.jsonl
generated/loop_{N}/images/
```

**処理:**
- `labeled.jsonl` のサンプルを train / (オプション: val) に分割
- 構造化出力スキーマを answer フォーマットとして埋め込む

**出力:**
```
dataset/loop_{N}/
  ├── train.jsonl
  │   # {"image_path": "...", "question": "<固定>",
  │   #  "answer": {"label": "tight", "rationale": "..."}}
  └── stats.json
      # total, tight_count, loose_count, keep_rate
```

---

### Step D: Fine-tuning

**目的:** 学習データでVLMをLoRA fine-tuneする

**使用モデル候補（軽量VLM）:**

| モデル | パラメータ | 備考 |
|---|---|---|
| LLaVA-1.5-7B | 7B | 実績あり、扱いやすい |
| PaliGemma2-3B | 3B | 軽量、Google製 |
| Qwen2-VL-7B | 7B | 精度高め |
| moondream2 | 1.9B | 最軽量、エッジ向け |

**Fine-tuning方式:** LoRA（量子化なし、Full precision）

**入力:**
```
dataset/loop_{N}/train.jsonl
models/base/         # ベースモデル（Loop 1）
models/loop_{N-1}/   # 前ループモデル（Loop 2〜）
```

**出力:**
```
models/loop_{N}/
  ├── lora_weights/     # LoRAアダプタ
  └── training_log.json
```

**設定（config/step_d.yaml）:**
```yaml
base_model: "llava-hf/llava-1.5-7b-hf"
lora:
  r: 16
  alpha: 32
  target_modules: ["q_proj", "v_proj"]
  quantization: none     # Full precision LoRA
epochs: 3
batch_size: 4
learning_rate: 2e-4
output_format: structured_json   # label + rationale
```

---

## 4. 評価設計

### [v0.3変更] テスト・バリデーションセット（規模拡大）

```
eval/
  ├── test_real/         # 実データ（人手ラベル）≥50枚（必須）
  │                      # シードから確保 + 追加収集
  │                      # 学習には絶対に使わない
  ├── val_generated/     # 各ループで人手アノテーション（10〜20枚）
  └── edge_cases/        # エッジケース専用（人手ラベル）≥20枚
                         # 半端な締め方、顎を引いた状態、逆光等
```

**[v0.3変更] 統計的検証の追加:**
- 各ループの test_real Accuracy を記録し、**ループ間の差分をMcNemar検定（p < 0.05）で検証**する
- テストセット50枚は2クラス分類における5%の精度差（80%→85%）を検出するのに概ね十分な検出力を持つ

### 評価指標

| 指標 | 説明 |
|---|---|
| Test Accuracy（実データ） | test_real に対する正解率（**最重要指標**） |
| Generated Accuracy | val_generated に対する正解率 |
| Edge Case Accuracy | edge_cases のみの正解率 |
| Keep Rate | Step B での keep 率（ループごと） |
| [v0.3] McNemar p-value | ループ N vs N-1 の差が有意か |

### ループ終了条件

[v0.3変更] 複数条件の明確化:

- **成功終了:** test_real Accuracy ≥ 目標値（例: 85%）に達し、かつ前ループ比で McNemar p < 0.05
- **停滞終了:** 2ループ連続でtest_real Accuracyの改善が +2%未満
- **品質劣化終了:** Step B の keep率が 50% を下回った
- **上限終了:** 最大ループ数（5回）に達した

---

## 5. 固定スキーマ（構造化出力）

VLMへの入出力は全ステップで以下のスキーマに統一する。

**質問（固定・単一）:**
```
Is the helmet chinstrap properly fastened?
Answer with JSON: {"label": "tight"|"loose", "rationale": "<one sentence>"}
```

**期待する出力（JSON）:**
```json
{
  "label": "tight",
  "rationale": "The chinstrap is visibly pulled snug under the chin with no visible slack."
}
```

**バリデーション:**
- `label` は `"tight"` または `"loose"` のみ許可
- `rationale` は1文、50〜200文字程度
- [v0.3変更] vLLMのguided_decoding（JSON schema制約）で出力を強制。パースエラーは原則発生しない

---

## 6. コンテナ構成

### [v0.3変更] 3コンテナに簡素化

v0.2の6コンテナから3コンテナに削減。Step CはPythonスクリプト単体で実行（コンテナ化不要）。

```
docker/
  ├── step_a_imagegen/
  │   ├── Dockerfile          # CPU, Python 3.11
  │   ├── generate.py
  │   └── requirements.txt    # httpx, Pillow, pyyaml
  ├── step_b_screening/
  │   ├── Dockerfile          # GPU必須（Teacher vLLMをローカル実行）
  │   ├── screen_and_label.py # スクリーニング＋ラベリングを統合
  │   └── requirements.txt    # vllm, pyyaml
  └── step_d_finetune/
      ├── Dockerfile          # GPU必須（CUDA 12.x）
      ├── finetune.py
      └── requirements.txt    # transformers, peft, torch, accelerate

scripts/
  ├── build_dataset.py        # Step C（コンテナ不要、ローカル実行可）
  └── evaluate.py             # 評価スクリプト
```

### docker-compose.yml（概要）

```yaml
services:
  step_a:
    build: ./docker/step_a_imagegen
    volumes:
      - ./seeds:/data/seeds:ro
      - ./generated:/data/generated
      - ./config:/data/config:ro
    env_file: .env

  step_b:
    build: ./docker/step_b_screening
    runtime: nvidia
    volumes:
      - ./generated:/data/generated:ro
      - ./screened:/data/screened
      - ./config:/data/config:ro
      - ${TEACHER_MODEL_PATH}:/data/teacher_model:ro
    env_file: .env

  step_d:
    build: ./docker/step_d_finetune
    runtime: nvidia
    volumes:
      - ./dataset:/data/dataset:ro
      - ./models:/data/models
      - ./config:/data/config:ro
    env_file: .env
```

### 環境変数（.env.example）
```
REPLICATE_API_TOKEN=xxx   # 画像生成API
HF_TOKEN=xxx              # HuggingFaceモデル取得
TEACHER_MODEL_PATH=/models/Qwen2.5-VL-72B-Instruct  # ローカルモデルパス
LOOP_NUM=1
```

---

## 7. テスト設計

各ステップに**必ずユニットテストを含める**。入出力スキーマの検証を中心とする。

| ステップ | テスト内容 |
|---|---|
| Step A | 出力CSVスキーマ確認、画像ファイル存在確認 |
| Step B | screening.csvのスキーマ確認、labeled.jsonlのJSONパース確認、labelがtight/looseのみ、keep率が0〜1の範囲内 |
| Step C | train.jsonlのスキーマ確認、stats.jsonの整合性 |
| Step D | lora_weights/の存在確認、training_log.jsonのloss推移 |

実装の際は、各dockerコンテナの入出力を模したスクリプトを作成し、そこで作成した入出力でテストする。
理由は、ローカル環境ではVLMを動かすことができないから。
そののち、そのテストの形式の入出力ができるdockerコンテナを作成する。


---

## 8. ディレクトリ構造（全体）

```
project_root/
├── seeds/                  # シード画像・ラベル（≤100枚）
├── generated/              # 生成画像（loop_N/ごと）
├── screened/               # Step B出力（loop_N/ごと）
├── dataset/                # Step C出力（loop_N/ごと）
├── models/                 # fine-tunedモデル（loop_N/ごと）
├── eval/                   # 評価データ（人手ラベル）≥50枚
├── config/                 # ステップごとのYAML設定
├── docker/                 # 各ステップのDockerfile（3つ）
├── scripts/                # build_dataset.py, evaluate.py
├── tests/                  # 各ステップのpytestテスト
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 9. 未解決の問題・リスク

| 項目 | 内容 | 対策案 |
|---|---|---|
| 画像生成のラベル保持率 | 編集後にあごひも状態が変化する可能性 | Step BのTeacher検証で除去。Loop 1の keep率を確認して判断 |
| Teacher VLMの信頼性上限 | Teacher自身の誤判定がラベルノイズになる | eval/test_real で Teacher単体の精度も計測し、上限を把握 |
| Fine-tuning 過学習 | 合成データ偏りによる過学習 | eval/test_real（実データ）での評価を必ず実施 |
| 合成データのドメインギャップ | 生成画像と実環境の差異 | eval/test_real を**実環境で撮影した画像**で構成する |

---

## 10. [v0.3変更] 将来の拡張オプション（v0.4以降で検討）

以下はv0.3では実装しないが、v0.4以降で検討する項目：

| 拡張 | 概要 | 発動条件 |
|---|---|---|
| Self-Screening（旧Step F） | Fine-Tuned VLMで学習データを再検証し、誤ラベルを除去 | Loop 2以降、keep率が安定している場合 |
| Multi-Model Voting（旧Step B拡張） | Fine-Tuned VLMをTeacherの補助投票者として追加 | Teacher単体のkeep率が低い（<70%）場合 |
| Rationale ablation | rationale有無による精度差を検証 | Loop 1完了後 |
| Hard negative mining | Fine-Tuned VLMが間違えた画像を重点生成 | Loop 3以降 |

---

## 11. 実装優先順位（AIコーディング準備）

[v0.3変更] 4ステップに簡素化されたため、優先順位も更新：

1. **Step C（Dataset Build）** — `scripts/build_dataset.py` として実装。外部依存なし、テスト容易
2. **Step B（Teacher Screening & Labeling）** — vLLM + guided_decoding の最小構成
3. **Step A（Image Generation）** — 画像生成API選定が必要
4. **Step D（Fine-tuning）** — GPU環境が必要、最後に検証
5. **evaluate.py** — テストセットでの評価スクリプト

---