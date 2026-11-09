# Idea 3 (leakage 重み付け蒸留) 実装計画メモ

作成: 2026-11-08 / 環境構築＋スモーク担当
作業場所 (PC bt-gaming, WSL): `~/muse/idea3-opsd/` — clone 済み OPSD は `~/muse/idea3-opsd/opsd/` (commit ae7d251, 2026-09-04)
本メモの数値は、断りのない限り実測。時間見積もりは「見積もり」と明記した箇所のみ。

---

## 1. 公式コード調査 (タスク1)

リポジトリ: https://github.com/siyan-zhao/OPSD (Self-Distilled Reasoner, arXiv:2601.18734)

### (a) 学習の入口と設定方法
- 入口は `opsd_train.py`。公式の起動は `scripts/run_opsd_1b.sh` 等から `accelerate launch --config_file accelerate.yaml --num_processes 4 opsd_train.py ...`。
- 設定は全て CLI 引数 (TRL の TrlParser: `CustomScriptArguments` + `GOLDConfig` + `ModelConfig`)。YAML 設定ファイルは accelerate 用のみ。
- 単一 GPU では `accelerate launch` なしで `python opsd_train.py ...` で直接実行できる (スモークで確認済み)。
- 注意: `opsd_train.py` は main プロセスで無条件に `wandb.init` を呼ぶ。ログインなし環境では `WANDB_MODE=disabled` が必要。

### (b) 蒸留損失と pointwise clipping の場所
- ファイル: `opsd_trainer.py`、クラス `OPSDTrainer` (TRL experimental GOLD trainer 継承)。
- 損失本体: `generalized_jsd_loss()` (382行目〜)。β 一般化 JSD を `F.kl_div(..., reduction="none", log_target=True)` で語彙要素ごとに計算。公式 1.7B スクリプトは `--beta 0` (forward KL 方向)。
- **pointwise clipping**: 同関数内 463-464行目、`jsd = jsd.clamp(max=token_clip)`。語彙要素ごとの発散寄与を上限で切る (「per-token」ではなく分布要素ごとの pointwise)。引数は `--jsd_token_clip` (デフォルト 0.05、1.7B thinking 設定も 0.05)。
- 呼び出し側: `compute_loss()` (626行目〜) が student forward → teacher forward (no_grad) → `generalized_jsd_loss(..., labels=shifted_labels, token_clip=self.jsd_token_clip)` の順。
- 別経路として `--use_tinker_loss` (Thinking Machines 式) があり、サンプルトークンの log 比 (advantage = log π_T − log π_S) を使う。A2 の実装で流用できる gather パターンがここにある。

### (c) 教師条件とロールアウト
- 教師条件は `data_collator.py` の `SelfDistillationDataCollator` が構築:
  - 生徒プロンプト: 問題文のみ + "Please reason step by step, and put your final answer within \boxed{}"。
  - 教師プロンプト: 問題文 + "Here is a reference solution to this problem:" + **参照解の全文** + transition prompt (「自分の言葉で独立に同じ答えを導け」) + 同じ指示文。chat template 適用、教師は thinking 有効がデフォルト (`--teacher_thinking True`)。
  - `--reason_first` 時は、教師がまず参照解の分析文を生成し、それを教師プロンプトに連結してから教える2段構え (生成は `generate_teacher_reasoning`)。
- ロールアウト: 生徒が問題のみから生成 (`training_step` → `generate_on_policy_outputs` の HF generate 経路、または `--use_vllm` 時 `_generate_on_policy_outputs_vllm` の vLLM colocate/server 経路)。生成トークン列を生徒系列・教師系列の両方に連結し、教師は特権文脈で生徒トークンをスコアリングする。
- 教師の重み: デフォルトは現在の生徒と同一 (dynamic)。`--fixed_teacher` (要 `--use_peft`) では LoRA アダプタ無効化 = 初期ポリシーで固定。公式 1.7B スクリプトは fixed_teacher を使用。EMA 教師 (`--use_ema_teacher`) も選択肢として実装済み。

### (d) データセット
- `opsd_train.py` 266行目で `load_dataset("siyanzhao/Openthoughts_math_30k_opsd")` を直接ロード (train 29,434 件)。列は `problem` / `solution` を collator が使用。別途の入手手順は不要で、HF から自動取得される。実在確認済み (HF API + 実ダウンロード済み)。

### (e) 想定モデルと 8GB での現実性
- 公式想定: Qwen3-1.7B / 4B / 8B。1.7B の公式スクリプト自体が **LoRA (r=64, α=128, 全線形層)** + 勾配チェックポイントで、4×H100・global batch 32・生成長 1024・「~15分、100ステップ以内にピーク」と記載。
- 8GB での所見: **Qwen3-1.7B + LoRA は現実的 (スモークで実証、後述)**。ただし公式のバッチ/生成長設定はそのままでは入らない。per_device batch と生成長を絞る必要がある。3B 級以上はこの方式 (全語彙 logits を複数保持) では 8GB に収まらない見込み (未実測の所見)。

---

## 2. スモーク結果 (タスク3、実測)

環境: RTX 3070 8GB、venv `~/muse/gpu-env` (torch 2.14.1+cu130) に transformers==4.57.1 / trl==0.26.0 / peft==0.17.1 / accelerate==1.11.0 / datasets==3.6.0 / wandb==0.22.3 等を追加 (environment.yml のピンに合わせた。vllm / deepspeed / flash-attn / bitsandbytes は未導入)。
素材: モデル `Qwen/Qwen3-1.7B` (HF API で実在確認、apache-2.0) と公式データセットを `~/muse/hf-cache` (HF_HOME) に取得済み (計 4.7GB)。

### 公式スクリプトからの変更点 (全部)
1. `opsd_train.py` を `smoke_train.py` にコピーし、データセットを `shuffle(seed=0).select(range(32))` に制限 (この1行のみ追加)。原典ファイルは無変更。
2. vLLM なし (HF generate 経路)、attention は `sdpa` (flash-attn 未導入のため)。
3. 単一プロセス直接実行 (accelerate launch なし、num_processes=1)。
4. per_device_train_batch_size 4→**1**、gradient_accumulation_steps 2→**1**、max_completion_length 1024→**128**、max_length 20000→**2048**、num_train_epochs 30→1 + `--max_steps 3`、`--save_strategy no`、`--report_to none`、`WANDB_MODE=disabled`。
5. 維持した公式値: lr 5e-6、β=0、temperature 1.1、top_p 0.95、top_k 20、lmbda 1、LoRA r64/α128/全線形層、fixed_teacher、jsd_token_clip 0.05、勾配チェックポイント。
6. 再試行時の追加対策: `TOKENIZERS_PARALLELISM=false` (後述の OOM 切り分け用)。
スクリプト実体: PC 上 `~/muse/idea3-opsd/run_smoke.sh`、ログ `~/muse/idea3-opsd/smoke_out/smoke.log`。

### 実測値
- **完走**: 3ステップ全てで損失が記録された。loss = −0.0015 / −0.0057 / −0.0034、grad_norm 0.53 / 0.37 / 0.38。
  - 補足: 負の小さな値は公式実装がこの設定で実際に出力した数値。初期は fixed teacher = 生徒の初期ポリシーで真の発散はほぼ 0 のため、bf16 の数値誤差範囲の値と解釈される (解釈であり、値そのものは実測)。
- **生成時間**: 128 トークン・batch 1 で 9.2s (初回、ウォームアップ含む) → 6.6s / 6.7s (約 19 tok/s)。
- **train_runtime**: 3ステップ 24.0s (生成込み、約 8s/ステップ)。起動〜学習開始までのオーバーヘッド (import+モデルロード+データ処理) は約 16s。全体の wall は 40s。
- **VRAM ピーク: 6,461 MiB / 8,192 MiB** (500ms 間隔サンプリング)。batch 1・生成 128・max_length 2048 での値。
- **ホスト RAM**: 実行中の MemAvailable 最低値は約 13.1GB (開始時 約15.2GB 空き) で、この実行自体は逼迫しなかった。

### 詰まった点 (正直な記録)
- **1回目の実行は exit 137 (OOM-kill) で死亡**。カーネルログで自プロセス (python) が kill されたことを確認。ただしその時点では VS Code server の検索系 (rg) 等の他プロセスも多数走っており、VRAM ピークは 5,737 MiB と余裕があったため、主因は自プロセス単独ではなくホスト RAM の一時的な圧迫とみられる (切り分けはここまで。断定はしない)。対策後に同一条件で再試行し完走した。
  - → 本実験 (長時間実行) では、同一 PC 上の他作業と RAM を奪い合うリスクがある。save_steps によるチェックポイント分割と再開可能性を前提に設計すべき。

---

## 3. 本実験 (約4k問) の所要時間見積もり ※ここは見積もり

前提の置き方で大きく変わるため、実測 (batch 1・128 トークン生成 6.6s、約19 tok/s、教師系列込みの fwd/bwd が残り約1.4s/ステップ) から外挿する。

- 公式相当 (生成長 1024、global batch 32) を batch 1 のまま回すと生成だけで 1 系列 ~54s となり非現実的。
- 現実的な縮小設定の見積もり: per_device 2 / accum 4 (実効 batch 8)、生成長 512、約4k問 = 500 オプティマイザステップ。バッチ生成の並列効果を見込んでも、生成 + 教師・生徒・(A1では参照) の各 forward で **1ステップ数分級**、総計 **数十時間級** になる見積もり。フォアグラウンド SSH 前提では単独では回し切れない。
- 縮小案 (推奨): 公式が「100ステップ以内にピーク」と言う通り、**ステップ数を 100〜150 に区切り**、データはその分 (実効 batch 8 なら 800〜1,200 問分) だけ使う設計なら **数時間級** の見積もり。A0/A1/A2 の3条件を回す場合はその3倍を見込む。
- 高速化の選択肢 (未検証): vLLM colocate は生成を大幅に速くするが、公式値の vllm_gpu_memory_utilization 0.6 は学習側と合わせ 8GB に収まらない。割合を下げた colocate か、生成だけ vLLM サーバに分ける構成は別途検証が必要。

---

## 4. A1 / A2 の差し込み計画 (タスク4)

### 共通の追加要素: p_ref の forward
- p_ref = **凍結した初期モデル (LoRA 無効) が、特権文脈なしで** 同じロールアウトをスコアリングした分布。
- 決定的に好都合な点: p_ref の入力系列は生徒の入力系列 (`inputs["student_input_ids"]`) そのもの。新しい collator は不要。
- `compute_loss()` の教師 forward の直後に、`with torch.no_grad(), model.disable_adapter():` で3本目の forward を追加し、`logits[:, student_prompt_len-1:-1]` を切り出して log_softmax (温度は損失と同じ `self.temperature`) を取る。既存コードの「切り出して即 del + empty_cache()」パターンに倣い、log-probs のみ保持する。
- 前提: `--use_peft` 必須 (disable_adapter が前提)。fixed_teacher との併用時は教師=初期+特権文脈、p_ref=初期+通常文脈となり、ℓ_t は「重みの差」ではなく「文脈 (特権情報) の差」だけを測る形になる。この組合せが定義 (同一モデル、文脈のみ差) に最も忠実。

### A1: leakage 重み (全語彙 KL)
- ℓ_t の計算: 教師 log-probs `log p_T(·|prefix, z)` と参照 log-probs `log p_ref(·|prefix)` は両方とも全語彙で揃うので、
  `ell_t = Σ_v p_T(v) · (log p_T(v) − log p_ref(v))` を token ごとに計算 (stop-grad)。教師 logits は現在 JSD 用に保持されているので追加メモリは log-probs 変換分が主。ℓ_t は (batch, seq) の小テンソル。
- 重みの適用先: `generalized_jsd_loss` は現在、語彙要素ごとの発散 → clamp → mask → reduction を内部で完結している。A1 では **token ごとの発散 d_t = Σ_v (clamp 後の要素発散)** を先に作り、`loss = Σ_t w_t · d_t · mask_t / Σ_t w_t · mask_t` (重み付き平均) に変える必要がある。具体案:
  - `generalized_jsd_loss` に任意引数 `token_weights=None` を追加。reduction 前に token 軸で d_t を集約し、weights があれば重み付き batchmean を返す分岐を足す (既存経路は weights=None で完全に従来通り)。
  - `w_t = exp(−ell_t / τ)` (stop-grad)。τ は CLI 引数 `--leakage_tau` で必須指定 (事前登録した値をスクリプトに明記する)。
  - Trainer 側は `OPSDTrainer.__init__` に `leakage_mode` ("none"/"fullkl"/"sampled") と `leakage_tau` を追加し、`opsd_train.py` の `CustomScriptArguments` に対応する CLI を足す。
- メモリ上の注意: 本実験スケール (batch 2・生成 512) では logits 級テンソルが student/teacher/ref の3系統 + JSD 中間量となり、ピークはスモークの 6.5GB を超える見込み。対策候補: ℓ_t と d_t を系列方向にチャンク計算する、または `--top_k_loss` の併用 (ただし top-k 化すると ℓ_t の「全語彙」性が崩れるので A1 の定義と矛盾する。チャンク計算が本命)。

### A2: 信号をサンプルトークン Δ_t に替えた版
- Δ_t = log p_T(y_t|prefix, z) − log p_ref(y_t|prefix) (stop-grad)。y_t は `sampled_token_ids` (compute_loss 内に既存)。
- 実装は A1 と同経路で、ℓ_t の代わりに Δ_t を使う分岐のみ。log-probs からの gather は Thinking Machines 経路 (`student_log_probs_sampled` の gather) と同じ書き方でできる。全語彙 KL を計算しないため A1 より軽い。
- 重みは A1 と同じ形 `w_t = exp(−Δ_t / τ)` を使い、信号の差だけを分離する (Δ_t は負にもなるため、その場合は w_t > 1 になる点に注意。クリップの要否は設計判断として親側で決める事項として明記する)。

### AVSD 診断の実装可能性
- 目的: 3ビュー (z₁ 付き教師 / z₂ 付き教師 / 特権なし) で、J_t ≈ 0 なのに ℓ_t が大きいトークンの割合を測る。
- 必要な追加 forward: z₂ 文脈の教師が1本増えるだけ (計4系統の log-probs)。いずれも no_grad の推論なので、学習なしの独立スクリプトとして実装可能。対象は凍結した初期モデル + 数十〜百問のロールアウトで十分なはず (診断は分布の割合の推定であり、学習を回す必要がない)。
- J_t の定義案: `J_t = D_KL(p_T^{z1} ‖ p_T^{z2})` (または対称化した JS)。ℓ_t は A1 の定義そのまま流用できる。
- **データ側の制約 (要判断)**: 公式データ `Openthoughts_math_30k_opsd` は1問に1つの `solution` しか持たない。z₂ (別の正解解法) の供給方法が別途必要:
  - 案1: 同一問題の別解を初期モデルに参照答え付きで生成させ、math-verify (environment.yml に含まれるが未導入) で正解検証を通ったものだけを z₂ とする。
  - 案2: `--reason_first` で教師が生成する「参照解の分析文」を z₂ の近似とする (独立した別解ではない、という留保つき)。
  - 案1が定義に忠実。生成+検証は推論のみで済み、診断スクリプトの一部として実装可能と判断する。
- 集計: 「J_t が小さい (例: 下位側の閾値 ε_J) かつ ℓ_t が大きい (例: τ スケール超)」の token 割合、J_t と ℓ_t の相関、を出力する。閾値 ε_J / ε_ℓ の置き方は事前登録すべき設計判断として残す。

---

## 5. 次に進む前の確認事項 (親への申し送り)
1. 本実験の規模設定 (ステップ数 100〜150 への区切り / 実効バッチ / 生成長) の確定。これで所要時間が数時間級か数十時間級かが決まる。
2. 長時間実行の運用: フォアグラウンド SSH が前提のため、save_steps を細かく刻み、セッション断・RAM 圧迫時にも再開できる構成にすること。
3. A2 の Δ_t 重みで w_t > 1 を許すか (クリップするか) の設計判断。
4. 診断の z₂ 供給は案1 (生成+math-verify 検証) を推奨。math-verify==0.8.0 の追加インストールが必要。
