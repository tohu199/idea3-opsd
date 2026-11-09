# idea3-opsd — 特権 leakage 重み付け自己蒸留（Idea 3）実験パッケージ

OPSD（On-Policy Self-Distillation）に対する提案法「特権 leakage 重み付け」の
実験一式を、別の計算機でそのまま再実行できる形にまとめたリポジトリ。

- **提案法**: 参照解（特権情報）を見た教師の分布と、特権を見ていない凍結した
  初期モデルの分布の全語彙 KL を位置ごとの leakage
  `ℓ_t = D_KL(p_T(·|prefix, z) ‖ p_ref(·|prefix))` と定義し、蒸留損失の項全体に
  `w_t = exp(−ℓ_t/τ)` を掛けて、特権依存の大きい位置の学習を弱める。
  手法の詳細は [docs/method_summary.md](docs/method_summary.md)。
- **ベース**: OPSD 公式コード（[siyan-zhao/OPSD](https://github.com/siyan-zhao/OPSD)、
  commit `ae7d251`）＋ Qwen3-1.7B + LoRA (r64)。
- **状態（2026-11-09）**: ゲート1（40ステップ・1シード）で A1 15.00% vs
  A0 11.67%（held-out 300問）。シャッフル対照 A1S 11.33% により「位置情報が
  生存」の事前登録分岐を確認済み。複数シード拡大が次の段階（§6）。
  判定の詳細は [docs/gate1_verdict.md](docs/gate1_verdict.md) /
  [docs/shuffle_verdict.md](docs/shuffle_verdict.md)。

## 0. 必要なもの

- NVIDIA GPU（VRAM 8GB で動作実績あり: RTX 3070。ピーク 8,005 MiB）
- Linux（Ubuntu 22.04 / WSL2 で実績）。ホスト RAM は空き 13.5GB 以上を確保
- Python 3.10、git、[uv](https://docs.astral.sh/uv/)（環境構築に使用）
- ディスク: モデル＋データ＋チェックポイントで約 15GB
- 所要時間の目安（RTX 3070）: τ較正 約8分 / 学習1条件 約1時間45分 / 評価 約22分

## 1. 環境構築

```bash
git clone https://github.com/tohu199/idea3-opsd.git
cd idea3-opsd

# (1) Python 環境（.venv を作る。バージョンはゲート1実行時の実測スタックに固定）
bash setup/install_env.sh

# (2) OPSD 公式コードを固定コミットで取得
git clone https://github.com/siyan-zhao/OPSD opsd
git -C opsd checkout ae7d251

# (3) ハーネスの重ね合わせ＋パッチ適用（順序固定。opsd_trainer.py と
#     gate1_train.py がゲート1実行時とバイト単位で一致することを検証済み）
bash apply_patches.sh
```

パッチの中身: `patch_opsd.py`（leakage 重み付け・leak_stats 記録・CLI 引数、
`gate1_train.py` の生成）、`patch_opsd2.py`、`patch_opsd3.py`（重みの
max-shift 安定化）、`patch_opsd4–6.py`（vLLM 経路内のみ・HF 実行では不活性、
バイト一致のため適用）、`patch_opsd_shuffle.py`（A1S 用 `fullkl_shuffle`）、
`patch_opsd_shuffle_fix.py`（シャッフルの最小修正）。
アンカーが1つでも合わなければパッチは abort する。

## 2. データセット準備

```bash
.venv/bin/python setup/dl_assets.py   # モデルも同時に取得する（§3）
.venv/bin/python check_data.py        # 任意: データの健全性チェック
```

- データセット: Hugging Face `siyanzhao/Openthoughts_math_30k_opsd`
  （OPSD 公式の数学データ 29,434 問、`problem` / `solution` フィールド）
- 実験での固定分割（インデックス）:
  - 学習: **[0, 960)**（40ステップ × 実効バッチ8）
  - held-out 評価: **[960, 1260)** の300問
  - AVSD 診断: [960, 1060) の100問
- ダウンロード先は HF キャッシュ（`HF_HOME` 未指定時は `./hf-cache` を
  `run_arm.sh` 等が設定する）

## 3. モデル準備

- モデル: Hugging Face `Qwen/Qwen3-1.7B`（§2 の `dl_assets.py` で取得済み。
  学習・評価ともこの名前で参照する）
- 教師は `fixed_teacher`（初期モデルを凍結して教師にする OPSD 公式設定）。
  提案法の参照モデル p_ref も同じ初期モデルの凍結コピーで、追加のモデル
  ダウンロードは不要。

## 4. 実行方法

### 4.1 τ の較正（学習前に1回）

```bash
cd opsd
../.venv/bin/python calibrate_tau.py   # 初期モデルで32問ロールアウトし ℓ_t の中央値を τ に固定
cd ..
# -> gate1_out/tau_calibration.json （ゲート1実測: τ = 5.306877e-04、約505秒）
```

τ は事前登録ルール（32問ロールアウトの ℓ_t 中央値）で決め、以後動かさない。
複数シード拡大でも τ はこの値を全シード共通で使う（A2 の公平化を除く。§6）。

### 4.2 スモーク（任意、3ステップ）

```bash
bash scripts/run_smoke.sh   # 上流の smoke_train.py で学習が回ることだけ確認する
```

### 4.3 アーム（条件）の実行

```bash
# 使い方: run_arm.sh ARM MODE TAU MAXSTEPS BS ACCUM SAVE_STRATEGY LOGGING_STEPS
#   BS=2, ACCUM=4 で実効バッチ8。save_steps は 20 固定（途中チェックポイント用）
bash scripts/run_arm.sh A0  none          0            40 2 4 steps 1   # 素の OPSD
bash scripts/run_arm.sh A1  fullkl        0.0005306877 40 2 4 steps 1   # 提案法
bash scripts/run_arm.sh A2  sampled       0.0005306877 40 2 4 steps 1   # 対照（Δ_t 信号）
bash scripts/run_arm.sh A1S fullkl_shuffle 0.0005306877 40 2 4 steps 1  # シャッフル対照
```

- ログは `logs/<ARM>_<日時>.log`、leak 統計は
  `gate1_out/<ARM>/leak_stats_<MODE>.jsonl`、チェックポイントは
  `gate1_out/` 配下（checkpoint-20 / checkpoint-40）に出る。
- 進捗確認: `bash scripts/check_arm.sh A1`
- **シード変更**: 既定は 42（ゲート1のシード）。変える場合は環境変数で
  `SEED=43 bash scripts/run_arm.sh A1 fullkl 0.0005306877 40 2 4 steps 1`
- **A1S のプローブ（必須の関門）**: 本番前に `MAXSTEPS=1` で1ステップだけ
  走らせ（ARM 名は probeA1S 等）、`leak_stats_fullkl_shuffle.jsonl` の
  最初のマイクロバッチの ESS が A1 の `leak_stats_fullkl.jsonl` 最初の行と
  一致することを確認する。一致しなければシャッフル実装か RNG 分離の誤り
  なので本番に進まない（設計: docs/shuffle_control_design.md）。
- 注意: `--resume_from_checkpoint` はこの Trainer 実装では trainer 状態が
  復元されない「疑似再開」になる（ゲート1で確認済み）。途中落下時は
  チェックポイントからの再開ではなく、最初から回し直すこと。
- 注意: A2 はゲート1で2回とも生成中に CUDA assert で落下した（重みが
  単一トークンへ集中する定義上の帰結）。§6 の公平化なしでは完走しない
  可能性がある。

### 4.4 評価（held-out 300問、greedy）

```bash
cd opsd
# 学習前（初期モデル）: --adapter なし
../.venv/bin/python eval_harness.py --out ../gate1_out/eval_step0.json
# 学習後: そのアームの checkpoint-40 アダプタを指定
../.venv/bin/python eval_harness.py --adapter <checkpoint-40へのパス> --out ../gate1_out/eval_A1.json
cd ..
```

出力 JSON に正答率（`accuracy`）と平均エントロピーが入る。評価集合
[960, 1260)・生成長 512・\boxed 正規化一致が既定値。

### 4.5 AVSD 死角の診断（任意、学習と独立）

```bash
cd opsd && ../.venv/bin/python avsd_diag.py && cd ..
# -> gate1_out/avsd_diag.json （ゲート1実測: 高ℓ位置の見逃し率 2.9%。主張は不採用）
```

## 5. 事前登録の判定基準（結果を見て動かさない）

- ゲート1: **A1 ≥ A0 + 2pt かつ A1 ≥ A2**（held-out 正答率）で通過。
  A1 ≤ A0 ならアイデア3は終了。
- シャッフル対照: A1S ≥ 14.0% → 位置特異性は死（実質終了）。
  A1S ≤ 12.7% かつ A1 − A1S ≥ 2pt → 位置情報が生存。
  12.7% ＜ A1S ＜ 14.0% → 曖昧、主張しない。
- ゲート1実測（1シード）: A0 11.67% / A1 15.00% / A2 12.0%（20ステップ
  代替評価）/ A1S 11.33%。n=300 の二項標準誤差は約 ±1.9pt。

## 6. 次の段階: 複数シード拡大（事前登録済みの計画）

A1S が「位置情報が生存」の分岐に入ったため、次は複数シードで
A1 − A0 の平均差を確認する。計画（docs/gate1_verdict.md §手順2）:

- A0 / A1 を追加2シード（例: 43, 44）で各40ステップ＋評価
  → `SEED` 環境変数で切り替える。τ は §4.1 の値を共通使用
- 公正な A2 を1シード: τ を Δ_t 自身の分布で再較正してから実行する
  （ゲート1の A2 は ℓ 用の τ を流用したため重みが崩壊した）。手順は
  `harness/calibrate_tau_delta.py`（約8分）で、ルールは
  [docs/fair_a2_calibration.md](docs/fair_a2_calibration.md) に事前登録済み:
  プールした Δ に対する `exp(−Δ/τ)` の ESS 比率が A1 の Gate 1 平均
  0.479 になる τ を推奨 τ とする。出力
  `gate1_out/tau_calibration_delta.json` の `recommended_tau` を使って
  `SEED=43 bash scripts/run_arm.sh A2 sampled <recommended_tau> 40 2 4 steps 1`
- 所要: 1条件 約2時間09分（学習 6,389秒＋評価 1,362秒、A1S 実測）×
  5条件 ≈ **約10時間45分** ＋ 較正・プローブ分。API 費用なし
- 読み取り: シード平均の A1 − A0。A2 が再び完走できない場合は
  A0/A1 の比較だけでも成立させる

## 7. 移植にあたっての変更点（bt-gaming 上の原版との差）

実験の数値条件は一切変えていない。変えたのは配置だけ:

- パッチ・スクリプト内の絶対パス（`~/muse/idea3-opsd`、
  `~/muse/gpu-env`）を、リポジトリ相対＋環境変数
  （`IDEA3_HOME`、`SEED`、`HF_HOME`）に置換
- `gate1_train.py` はコピーではなく `patch_opsd.py` が
  `opsd_train.py` から生成する（原版でも生成物）
- vLLM 試行分（不採用）は `patches/archive_vllm/run_arm_vllm.sh` に
  参考として分離。経緯は docs/gate1_results.md

## 8. ドキュメント

- [docs/method_summary.md](docs/method_summary.md) — 提案法のまとめ
- [docs/gate1_design.md](docs/gate1_design.md) — ゲート1の事前登録設計
- [docs/shuffle_control_design.md](docs/shuffle_control_design.md) — A1S の事前登録設計
- [docs/impl_notes.md](docs/impl_notes.md) — 実装メモ
- [docs/gate1_results.md](docs/gate1_results.md) — ゲート1全実測（逸脱記録を含む）
- [docs/gate1_verdict.md](docs/gate1_verdict.md) / [docs/shuffle_verdict.md](docs/shuffle_verdict.md) — 判定
- [docs/falsification.md](docs/falsification.md) — 新規性の反証記録
- [docs/fair_a2_calibration.md](docs/fair_a2_calibration.md) — 公正な A2 の τ 再較正ルール（事前登録）

## 9. 出典

- OPSD: https://github.com/siyan-zhao/OPSD (commit ae7d251)
- モデル: Qwen/Qwen3-1.7B（Hugging Face）
- データ: siyanzhao/Openthoughts_math_30k_opsd（Hugging Face）
- 本リポジトリのハーネス・パッチ・ドキュメントは上記 OPSD に対する
  実験用追加物であり、OPSD 本体は含まない（セットアップ時に取得する）
