# ゲート1 実験結果 — アイデア3 Leakage 重み付け蒸留

- 設計・判定基準: [gate1_design.md](gate1_design.md)（事前登録。ステップ数の 120 → 40 変更のみユーザー承認の上で変更履歴に記録済み。判定基準・τ・評価集合は不変）
- 環境: PC bt-gaming / WSL `~/muse/idea3-opsd/`、OPSD 公式コード commit ae7d251（+ gate1 用パッチ、下記）、Qwen3-1.7B + LoRA r64、RTX 3070 8GB
- ログ: PC `~/muse/idea3-opsd/logs/`（各アームの学習ログ、VRAM/RAM サンプリング、leak_stats_*.jsonl）

## 実行条件（全アーム共通、確定値）

- 学習率 5e-6（公式 1.7B スクリプト）、β=0、pointwise clip 0.05、fixed_teacher、temperature 1.1、top_p 0.95、top_k 20、シード 42
- 実効バッチ 8（per_device 2 × 勾配蓄積 4）、生成長 512、max_length 2048、**40 ステップ**（120 → 40 の規模変更は gate1_design.md 変更履歴 2026-11-08 23:41、ユーザー承認。学習データ範囲 [0, 960) は不変で、各アームは同一順序の先頭 320 問を消費した）
- 学習データ: 公式データ indices [0, 960)。評価: held-out [960, 1260) の 300 問、greedy 生成 512 トークン、\boxed 抽出＋正規化一致
- パッチ内容（公式コードへの変更、すべて記録済み）:
  - `patch_opsd.py`: A1/A2 の leakage 重み付け（p_ref の3本目 forward、全語彙 KL ℓ_t / サンプルトークン Δ_t、重みを損失項全体へ適用）、leak_stats の JSONL 記録、学習データ範囲の制限、CLI 引数追加
  - `patch_opsd2.py`: 教師・参照 forward で `logits_to_keep` を使い補完位置の logits のみ計算（lm_head は位置独立のため値は同一；検証プローブで step 1 損失がパッチ前と完全一致 −0.0036）。p_ref の log-probs は fp32 で計算し bf16 で保持
  - `patch_opsd3.py`: 重みの数値安定化（指数の最大値シフト。重み付き平均は共通定数に不変なので定義は不変）と ESS 統計の記録
  - `expandable_segments:True` は WSL2 で CUDA マッピング失敗を起こしたため不使用（probeA1 で発生、対策後に完走）

## τ キャリブレーション（完了）

- τ = **5.306877e-04**（初期モデルの32問ロールアウト、全位置の ℓ_t の中央値。事前登録ルール通り、以後の変更なし）
- 対象トークン数 15,863。ℓ_t 分布: 平均 0.2492、p25 1.24e-06、p50 = τ、p75 0.06944、p90 0.3756、p99 3.2017、最大 45.245。ℓ_t > 0.01 の位置は 37.6%、> 0.1 は 21.9%、ちょうど 0 の位置は 0%
- 読み: 分布は強く右裾。τ が中央値のため、A1 の重み w=exp(−ℓ/τ) は ℓ ≲ 0.001 の位置（全体の半分強）に実質集中し、ℓ ≥ 0.01 の位置はほぼゼロになる。これは事前登録した τ ルールの帰結としてそのまま記録する
- 所要 505 秒。記録: PC `gate1_out/tau_calibration.json`、ログ `logs/calibrate_tau.log`

## 学習結果（40ステップ/アーム、2026-11-08〜09 再開担当が実施）

- **A0（素の OPSD）**: 完走（exit 0）。wall 5,832 秒（約146秒/step）、VRAM ピーク 8,005 MiB、ホスト RAM 最小空き約13.6GB。損失は step 1 の 0.0128 から末尾 −0.0034 まで（bf16 数値誤差域）。checkpoint-20 / checkpoint-40 保存済み。ログ `logs/A0_20261108_234304.log`。
- **A1（提案法、fullkl、τ=5.306877e-04）**: 完走（exit 0）。wall 6,197 秒、VRAM ピーク 8,004 MiB。重み付き損失は最初から約 0.0001 の桁で推移（重みが低 ℓ 位置＝発散ほぼ 0 の位置に集中するため、定義の帰結）。leak 統計（160 マイクロバッチ）: ESS 比率 平均 0.479（最小 0.136、最大 0.716）、top_weight_share 平均 0.0052（最大 0.0819）— 重みは健全に分散した。ログ `logs/A1_20261109_014336.log`、統計 `gate1_out/A1/leak_stats_fullkl.jsonl`。
- **A2（対照、sampled Δ、τ 同値）**: **40ステップを完走できなかった（2回とも同一症状で落下）**。
  - run 1（ログ `logs/A2_20261109_034945.log`）: step 23 まで完了、step 24 の生成中に `torch.AcceleratorError: CUDA error: device-side assert triggered`（`generate` → `_has_unfinished_sequences`）で exit 1。wall 3,581 秒。
  - run 2（ログ `logs/A2_20261109_050455.log`、run 1 と同一条件の新規実行）: step 26 まで完了、step 27 の生成中に同一の device-side assert で exit 1。wall 4,035 秒。
  - 両 run とも落下前に損失のスパイクが拡大した（run 1 の step 21〜23: 0.0497 / 0.0526 / 0.2563、run 2 の step 21〜23: 0.0201 / 0.2169 / 0.327）。leak 統計は両 run とも ESS 比率 平均 約 0.0011（≒1/1024）、top_weight_share 平均 1.0000 で、事前登録時に予測された通り重みが実効的に単一トークンへ完全集中していた。この不安定化の先で生成が assert に至る、というのが実測の事実関係である（assert の直接原因の特定まではしていない）。
  - したがって A2 の評価は **run 2 の checkpoint-20（20ステップ時点）** で実施した（下記評価節と逸脱記録を参照）。

## vLLM 高速化の試行（不採用、2026-11-08 23:09〜23:29）

- バージョン選定: 最新 0.33.0 は transformers 5.x 必須で不適合。0.22.0 は TRL 0.26.0 の import 自体が失敗（trl が参照する `vllm.transformers_utils.tokenizer` が消滅）。TRL 0.26.0 の対応版は 0.10.2/0.11.0/0.11.1/0.11.2 のみ → **vLLM 0.11.2**（torch==2.9.0、transformers ≥4.56 <5）を採用し、学習スタック（transformers 4.57.1 / trl 0.26.0 / peft 0.17.1 / accelerate 1.11.0 / datasets 3.6.0）を維持した複製 venv `~/muse/gate1-env` を構築（gpu-env 本体は無変更）。旧試行環境は `~/muse/gate1-env.vllm022-broken` として保持。手順ログ `logs/vllm_env_install*.log`
- 方式選定: server 方式（別プロセス常駐）は学習ピーク約7.5GB + サーバー約4GB で 8GB を超過するため不採用。**colocate + sleep mode** を検証した。
- エンジン単体スモーク（`vllm_smoke.py` / `vllm_smoke2.py`、ログあり）: 初期化 17.1 秒、128トークン生成 **1.4 秒（≈91 tok/s）**（HF generate は batch 1 で 13〜17 tok/s、batch 2 で 24〜31 tok/s）。sleep(level=2) → wake(weights) → load_weights → wake(kv_cache) → 生成の順序では出力は健全（wake だけでは重み内容が復元されず文字化けすることも確認 — Trainer の同期手順が正しいことの裏付け）。
- Trainer 内プローブ（5ステップ、A0 と同一設定、util は試行値）で順に衝突した壁:
  1. util 0.55: 起動時空きメモリ 2.96 GiB < 要求 4.4 GiB で初期化失敗（学習モデルが先に GPU 常駐）→ **patch_opsd4**（エンジン初期化中だけ学習モデルを CPU に駐車）で解消
  2. util 0.50: KV キャッシュ不足（必要 0.22 GiB > 利用可能 0.11 GiB、max_model_len=2048 の1リクエスト分）→ util 0.55 で解消
  3. util 0.55: 初回 training_step の `wake_up(["kv_cache"])` で CUDA invalid argument（level-2 sleep で破棄された重みメモリへ buffer を復元しようとする公式コードの順序バグ。重み同期コールバックは step 終了時にしか走らない）→ **patch_opsd5**（初回生成前に限り `_move_model_to_vllm()` で同期）で解消
  4. 次の壁: 重みプールの再マップ時に **CuMemAllocator の create_and_map が "device not ready"** で失敗（VRAM ピーク 7,931 MiB まで到達）。**patch_opsd6**（enforce_eager で CUDA グラフプールを排除）でも同一箇所で失敗（ピーク 7,576 MiB）。WSL2 の cuMem マッピングの脆弱性（本実験序盤に expandable_segments でも同種の "Failed to create GPU mapping" を観測）とみられ、許容範囲の変更では解消できなかった。
- 結論: **vLLM はこの 8GB / WSL2 環境では本 Trainer と共存できず不採用**。エンジン自体の速度（約3〜7倍）には問題がなく、ボトルネックは共存メモリと WSL2 の cuMem である。
- 復帰確認: パッチ 4〜6 は全て `--use_vllm` 経路内のみで HF 経路では不活性。gpu-env + 従来スクリプトで HF プローブを再実行し正常終了（損失 0.0035、exit 0、ログ `logs/probeHF_20261108_232830.log`）。

## 評価結果（held-out 300問）

- step 0（初期モデル、学習前）: **正答率 0.1267（38/300）**、平均エントロピー 0.2018 nats。所要約20分（生成 batch 8 + エントロピー測定）。記録: PC `gate1_out/eval_step0.json`、ログ `logs/eval_step0.log`
- **A0（40ステップ）: 正答率 0.1167（35/300）**、平均エントロピー 0.3488 nats。step 0 比 **−1.00 pt**、エントロピーは +0.147 と大幅増（entropy overshoot 方向）。評価 wall 1,344 秒。記録: `gate1_out/eval_A0.json`
- **A1（40ステップ）: 正答率 0.1500（45/300）**、平均エントロピー 0.2146 nats。step 0 比 **+2.33 pt**、A0 比 **+3.33 pt**。エントロピーは step 0（0.2018）とほぼ同水準で、A0 のような増大は見られなかった。評価 wall 1,339 秒。記録: `gate1_out/eval_A1.json`
- **A2（run 2 の checkpoint-20、20ステップ時点）: 正答率 0.1200（36/300）**、平均エントロピー 0.2058 nats。step 0 比 −0.67 pt。**40ステップ完走が不可能だったための代替評価であり、A0/A1 とステップ数が揃っていない**。評価 wall 1,355 秒。記録: `gate1_out/eval_A2_ckpt20.json`
- **A1S（シャッフル対照、40ステップ、2026-11-09 追加実施）: 正答率 0.1133（34/300）**、平均エントロピー 0.3311 nats。step 0 比 −1.33 pt、A1 比 −3.67 pt。評価 wall 1,362 秒。記録: `gate1_out/eval_A1S.json`。詳細は本ファイル末尾の A1S 節と `reports/2026-11-09_idea3-shuffle-verdict.md`

## AVSD 死角の診断（実施済み、2026-11-09）

設計の定義通り実施（held-out [960, 1060) の100問、初期モデルのロールアウト、単一参照解から full / partial / answer の3ビューを構成、J_t = D_KL(q^A ‖ q̃^G)、全て温度 1.1 の分布）。wall 1,672 秒。記録: `gate1_out/avsd_diag.json` / `avsd_diag.npz`、ログ `logs/avsd_diag.log`。

- 対象位置数: 49,722
- **「ℓ 上位25% かつ J が中央値以下」の割合: 0.0288（2.88%）**
- 参考統計: ℓ の p75 閾値 0.0619、J の中央値 2.83e-06（J は大半の位置でほぼ 0）、ℓ 平均 0.2306、J 平均 0.00429、ℓ と J の Pearson 相関 0.233
- 事前に固定した読み（割合 ≥ 50% なら「AVSD のゲートは高 leakage 位置の過半を見逃す」）に対して、実測 0.0288 は **その条件を満たさなかった**。すなわち、この operationalization の下では、高 leakage 位置の 97% 以上が J 中央値超（ビュー間不一致あり）側にあり、AVSD 型ゲートが原理的に見逃す割合は小さいという測定結果である。

## 判定基準への当てはめ（数値のみ。最終判定の宣言はメイン側）

事前登録基準: **A1 ≥ A0 + 2pt かつ A1 ≥ A2**（held-out 正答率）。

- A1 − A0 = 0.1500 − 0.1167 = **+3.33 pt**（基準 +2 pt に対して数値上は上回る。正味 +10 問）
- A1 − A2 = 0.1500 − 0.1200 = **+3.00 pt**（A1 ≥ A2 を数値上は満たす。**ただし A2 は 20ステップのチェックポイント評価で、ステップ数が揃っていない**）
- 統計的な留保（算術的事実）: n=300・1シードのため、各正答率の二項標準誤差は約 ±1.9〜2.0 pt。上記の差はいずれもこの誤差範囲と同程度の大きさである。
- 検出力の留保（設計変更履歴の通り）: 40ステップは公式の言うピーク（100ステップ以内）より手前であり、A0 自体が step 0 を下回った（−1.00 pt）段階での比較である。

## 逸脱・事故の記録

- A0 初回ラン（HF 方式）は速度不足のため方針変更で停止（step 3、チェックポイントなし）。設計・判定基準の変更ではない。
- vLLM 試行に伴う公式コードへの追加パッチは patch_opsd4/5/6（いずれも `--use_vllm` 経路内のみ、HF 経路では不活性）。導入先・バージョン・手順は `logs/vllm_env_install*.log` と本ファイルに記録。
- A2 の数値的性質（再掲）: τ=5.3e-04 と符号つき Δ の組合せでは重みが実効的に最大の1トークンへ集中する（プローブ τ=0.5 時点で ESS=1/1024、top_weight_share=1.0）。事前登録した定義の帰結として、学習時も ESS を記録して報告する。
- **A2 の落下と再開の経緯（2026-11-09、再開担当）**:
  - run 1 が step 24 の生成で CUDA device-side assert により落下（exit 1）。checkpoint-20 から `--resume_from_checkpoint` で再開を試みたが、この Trainer 実装では trainer 状態（ステップ数・データ順序）が復元されず、重みだけ引き継いでステップ 0 から数え直す「疑似再開」になることを leak_stats の global_step リセットで確認したため、その run は step 約5で意図的に停止し、成果物は無効として切り離した（ログ `logs/A2_20261109_045011.log` は保持。run 1 の成果物一式は `gate1_out/A2_run1_crashed/` に退避。なお同ディレクトリの `leak_stats_sampled.jsonl` には run 1 分（micro 0〜92、global_step 0〜22）の後に疑似再開分（micro リセット後、global_step 0〜5）が混在している）。
  - 同一条件の新規 run 2 も step 27 の生成で同一 assert により落下したため、A2 は「事前登録した定義のままでは 40ステップ完走不能」と記録し、評価は run 2 の checkpoint-20 で代替した（ステップ数不揃いは判定への当てはめ節にも明記）。
- run_arm.sh の `--save_steps` を 40 → 20 に変更した（step 20 のチェックポイントを残すため。チェックポイント保存間隔は学習の数値条件ではない）。

## シャッフル対照 A1S（2026-11-09、ゲート1判定後の決定的確認）

設計・判定基準: [shuffle_control_design.md](shuffle_control_design.md)（事前登録）。判定: `reports/2026-11-09_idea3-shuffle-verdict.md`。A1 と完全同一条件（τ=5.306877e-04、40ステップ、シード42、データ [0,960)、LR 5e-6、β=0、clip 0.05、実効バッチ8、生成長512、logging_steps 1）で、マイクロバッチごとに重み w_t を行内・マスク位置内でのみ無作為に並べ替える（独立 Generator、シード 20261109）。

- **パッチと最小修正**: `patch_opsd_shuffle.py` を適用（`grep fullkl_shuffle opsd_trainer.py` = 3ヒット）。ただし初回プローブが `NameError: name 'self' is not defined` で落下（`generalized_jsd_loss` は `@staticmethod` で、挿入ブロックが `self._shuffle_gen` を参照していた。ログ `logs/probeA1S_20261109_181938.log`、49 秒）。`patch_opsd_shuffle_fix.py` で引数渡し（`shuffle_gen` パラメータ追加、呼び出し側から `self._shuffle_gen` を渡す）へ最小修正。条件・意味論の変更なし。
- **プローブ（ハードゲート）**: 修正後プローブ（`probeA1S`、ログ `logs/probeA1S_20261109_182144.log`、wall 173 秒、exit 0）。最初のマイクロバッチ: ess_frac 0.5555515289306641 vs A1 0.5555514693260193（絶対差 5.96e-08、相対 1.07e-07)、n_tokens / signal_mean / signal_median / signal_p90 / top_weight_share は全桁一致。signal がビット同一＝生成・ℓ計算は A1 と同一であり、ESS の差は並べ替えに伴う浮動小数点の加算順序差と判断し合格。
- **学習**: 完走（exit 0）。wall 6,389 秒（train_runtime 6,376 秒）、VRAM ピーク 8,001 MiB、ホスト RAM 最小空き約13.5GB。checkpoint-20 / checkpoint-40 保存済み。ログ `logs/A1S_20261109_182459.log`、統計 `gate1_out/A1S/leak_stats_fullkl_shuffle.jsonl`（160 マイクロバッチ、ess_frac 平均 0.4656［0.1193〜0.6996]、top_weight_share 平均 0.0060。A1 は平均 0.4793、平均 0.0052 で、重みの分散は同水準）。
- **評価**: checkpoint-40（root 保存アダプタと md5 一致を確認済み）で実施。**正答率 0.1133（34/300）**、平均エントロピー 0.3311 nats。step 0 比 −1.33 pt、A1 比 −3.67 pt、A0 比 −0.33 pt（1問差）。評価 wall 1,362 秒。記録: `gate1_out/eval_A1S.json`、ログ `logs/eval_A1S.log`。
- **当てはめ**: 事前登録の第2分岐（A1S ≤ 12.7% かつ A1 − A1S ≥ 2pt）に該当 → 位置情報が生存。副次読み: エントロピー 0.3311 は A0（0.3488）並みで、overshoot 抑制自体が位置特異的だった。留保: n=300・1シードの二項標準誤差は約 ±1.9 pt で、差はノイズと同程度の大きさを含む。
- **運用事故**: 本番学習を保持していた監視側 SSH セッションが約89分時点で切断されたが、PC 側プロセスは生存し正常完走した（exit 0、成果物は完全）。結果への影響なし。
