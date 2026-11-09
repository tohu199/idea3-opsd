# アイデア3「特権 leakage 重み付け蒸留」反証探索

- 実施日: 2026-11-08
- 担当: 反証探索サブエージェント（一次情報のみ。arXiv abs/HTML 全文を直接精読）
- 目的: アイデア3が先行研究に既に潰されていないかを確かめる。推測・二次情報で断定しない。確認できなかった事項は「未確認」と明記する。

## 検証対象の主張（アイデア3）

特権蒸留（教師が正解・結果・特権情報 z を条件に含み、生徒は含まない）において、各トークン位置 t の leakage を

- ℓ_t = D_KL(p_T(·|prefix, z) ‖ p_0(·|prefix))（p_0 = 特権なし同一モデルの分布）

と定義し、蒸留損失を

- L = Σ_t w(ℓ_t) · D_KL(p_T ‖ p_S)（w は減少関数）

で重み付けする。新規性の主張は次の3点。

- (a) leakage のトークン単位の定量化そのもの
- (b) それによる連続的な損失重み付け
- (c) OPSD / Taste-Bench蒸留 / SDPO が w≡1 の特殊ケースという統一視点

## 本ファイルの分析上の注意（数学的事実）

RLSD（§6）の Δ_t = sg(log P_T(y_t) − log P_S(y_t)) に対し、アイデア3の ℓ_t は

- ℓ_t = Σ_v p_T(v) [log p_T(v) − log p_0(v)] = E_{v∼p_T}[Δ_t(v)]

すなわち **ℓ_t は RLSD の Δ を教師分布で期待値を取ったもの**（p_0 = P_S の場合）。Δ_t の定義は一次情報で確認済みであり、この期待値への変形は自明な計算である。したがって (a) の量は「サンプルトークン版」と「全語彙期待値版」の差しか残らない。

もう一点: 自己蒸留設定で p_0 を「現在の生徒 p_S そのもの」と同一視すると、損失項 D_KL(p_T‖p_S) と ℓ_t が一致し、w(ℓ_t) は「損失の大きさ自体による自己参照的な重み付け」（ロバスト損失／クリッピングの連続版）に退化する。アイデア3に独立した内容があるのは、p_0 を凍結した参照モデル（初期モデル等、訓練中の生徒とは別）とする場合に限られる。実験設計ではこの区別が必須。

---

## 1. E2-OPSD

- **実在確認**: arXiv:2610.05048（v2 全文 HTML 精読）。著者: Yifei Liu, Minghao Fang, Xinyu Gu, Chengkai Yao, Mengdi Liu, Tengfei Ma, Jiangbin Zheng, Chang Yu, Zhangyang Gao。題名 "E2-OPSD: Taming Entropy Overshoot in On-Policy Self-Distillation"。コード https://github.com/orionfei/E2-OPSD（論文・サイクル1検証で確認）。
- **entropy overshoot の定義**: 訓練中、生徒のトークンエントロピーが教師のそれを超えて上昇し、高止まりする現象。原因は二つ: (i) 教師側 — 参照解条件付き教師は自らの解答方向パスでは自信を持つが、その自信は生徒生成 prefix には転移しにくく、監督が「再利用可能な推論パターンではなく解答固有の手がかり」に過度に結びつく。(ii) 生徒側 — forward KL が生徒分布を拡散し続ける。
- **exemplar-guided teaching (EGT) の仕組み**: 教師条件から現在の問題の参照解 y* を**取り除き**、検索した近傍の解済み問題 (x′_g, y′*_g) に置換する（埋め込み cosine 類似度で K 近傍を事前計算、ロールアウトごとに1つを一様抽出）。教師は (x, x′_g, y′*_g, ŷ_<t)、生徒は (x, ŷ_<t) のみを見る。論文自身の表現は "turning answer leakage into strategy transfer"。
- **特権依存を「測って」いるか**: **測っていない。置換で回避している。** 教師の特権依存度を定量化する量は EGT には存在しない。
- **ただし重み付けはある**: Entropy-Aware Distillation (EAD) は (i) Bidirectional Entropy Routing — エントロピーギャップ Δ_t = H(p_t) − H(q_t)（生徒−教師）の符号で KL 方向を切替、(ii) Adaptive Gap Damping — 連続重み c_t = Δ_t² / (Δ_t² + max(σ², ε²)) ∈ [0,1) を損失に乗算（式7: L = (1/|T|) Σ sg(c_{i,t})·D^BER_{i,t}）。つまり**蒸留損失への連続的なトークン重み付けという形式 (b) は実装済み**。ただし信号は生徒–教師のエントロピーギャップであり、特権有無の分布間発散ではない。また重みはギャップ大で**増加**する（アイデア3の減少方向と逆）。
- **関連研究の分類（本ファイルにとって重要）**: E2 は先行対処法を verifier-directed / signal-purification（「追加の教師パスで特権バイアスを打ち消す」Pan et al. 2026; Shen et al. 2026）/ context-curation に分類し、教師の特権信号が「reproducible guidance と privilege-dependent behavior を混ぜる」こと、特権教師が「irreducible objective gap (Yang et al., 2026)」を持つことを既知の問題として記述している。
- **重なり度: 部分的に近い**
- **潰れる主張**: (b) の「形式」部分（連続重み付け自体は新規でない）。(a) は潰れない（leakage を測っていない）。(c) には無関係。

## 2. OPSD 原論文（Self-Distilled Reasoner）

- **実在確認**: arXiv:2601.18734（v3 全文 HTML 精読）。著者: Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, Aditya Grover。題名 "Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models"。コード https://github.com/siyan-zhao/OPSD（abstract 内に記載）。
- **損失**: 式(1) L_OPSD(θ) = E[ Σ_n D(p_T(·|x, y*, ŷ_<n) ‖ p_S(·|x, ŷ_<n)) ]。トークン重みは一様（w≡1）。唯一の非一様要素は **per-token pointwise KL clipping**（語彙要素ごとの発散寄与を τ でキャップ）で、これは「発散値の大きさによる飽和型の重み付け」の粗い前例ではあるが、leakage 量による重み付けではない（クリッピングの動機は stylistic token の支配防止）。
- **"privilege illusion" の語**: **v3 全文中に現れない**（browser.find で "illusion" 0件を本調査で再確認）。OPSD は特権情報の存在は述べるが、illusion の定義も対処も提示していない。この語を自称で導入したのは DOPD（§7、「we dub privilege illusion」）。DAPD は典拠の一つに Zhao et al. 2026a を挙げるが、OPSD 本文中での使用は確認できない。
- **重なり度: 主張 (c) の根拠そのもの**（「潰す」のではなく、アイデア3の特殊ケース記述が原論文については正確であることを裏付ける）
- **潰れる主張**: なし。むしろ (c) は OPSD については成立。ただし pointwise clipping の存在により、「OPSD は完全に一様」という記述は厳密には要修正（w≡1 + 要素ごとキャップ）。

## 3. DAPD: Dual-Anchored Policy Distillation

- **実在確認**: arXiv:2608.01735（v2 全文 HTML 精読）。著者: Jianyu Wu, Yizhou Wang, Encheng Su, Chen Tang, Shixiang Tang。コード https://github.com/uanu2002/DAPD。alphaxiv 経由の言及どおり実在。
- **診断**: privilege illusion の根本原因は特権教師と推論時生徒の間の **information asymmetry**。元の OPSD 目的関数は reproducible guidance と privilege-dependent behavior を混ぜて転送する（"Entangled Distillation"）。
- **対処は重み付けか**: **否。情報マッチング（アンカリング）。** 自己条件付き bridge 分布 p_Self（モデルに完全な completion を条件付け）を導入し、3分布 None（特権なし）/ Cross（特権あり教師）/ Self の間に3つの有向目的関数を設定: L_ent（元の OPSD 項）、L_infer（Self→None）、L_priv（Self→Cross）。Dual-Path Anchoring が情報条件の揃った2経路で整合させ、Dual-Source Anchoring が reference→rollout と rollout→reference の両方向を λ でバランスする（式7）。係数は方向レベルのスカラーであり、**トークンごとの leakage 重みは存在しない**。
- **重要な傍証**: DAPD の関連研究は既存の対処を (1) 特権情報の直接利用、(2) **"selective transfer from privileged supervision, which filters or reweights the teacher signal to suppress privilege-dependent supervision (Nguyen et al. 2026; …)"** に分類し、その限界は構造的（信号をいじるのではなく情報条件を揃えるべき）だと論じている。すなわち「特権依存監督を reweight する」という**研究カテゴリ自体は DAPD 時点で既存のものとして整理済み**。
- **診断の定量化**: wrong claims（根拠なく答えを断言する生成）の頻度を behavioral probe として数える。leakage の分布量ではない。
- **結果**: Qwen3-4B で OPSD 比 +2.00（6タスク平均）、4B で +2.69、32B で +2.78。
- **重なり度: 部分的に近い**（問題設定と3分布の形式化は共通。手法は別系統）
- **潰れる主張**: (a)(b) は潰れない。ただし (b) の「カテゴリとしての新規性」は DAPD の分類により弱まる。None/Cross の対比という形式化自体も既出である点は明記すべき。

## 4. OCSD: Agentic RL with Observation-Calibrated Self-Distillation

- **実在確認**: arXiv:2608.04788（v1 全文 HTML 精読）。著者: Yi Yang, Cong Qin, Xiaodan Liu, Chishui Chen, Qing Dong, Yan Zhang, Cao Liu, Zhao Yang, Lu Pan, Jiaye Lin, Yi Feng（Meituan LongCat / 南京大学ほか）。コード https://github.com/yiy1x/OCSD。export.arxiv.org 経由の言及どおり実在。
- **confounding の分離法**: 特権 replay view による token support δ_{i,k,t}(E) = log π_T(y_{i,k,t}|h, E) − log π_S(y_{i,k,t}|h)（式2）は、(i) 特権情報そのものと (ii) replay scaffold が誘発するスコア変化を混ぜている。OCSD は構造を揃えた2つの view — Full（実際の未来観測あり）と Observation-Ablated（観測だけを "Observation: not provided" に置換、scaffold は同一）— を対比し、**observation residual** e_{i,k,t} = δ^F − δ^A（式6）を取る。生徒 log-prob は差で相殺されるため、e は式7のとおり **log π_F(y|h, E^F) − log π_A(y|h, E^A)**、すなわち「特権あり view」と「特権を除いた matched view」のサンプルトークン log 確率差そのものになる。
- **使い方**: q_{i,k,t} = tanh(e/2) ∈ [−1,1]（式8）とし、高不確実ステップ（NLL 上位 ρ=0.2）でのみ GRPO advantage を Â^OCSD = Â [1 + β·sgn(Â)·q]（式11、β=0.5）と変調する。方向（符号）は GRPO が保持。
- **leakage の定量化と重なるか**: **概念的にかなり重なる。** 「特権あり／なしの matched な2条件の差分で、特権に帰属するトークン単位の信号を分離する」という (a) の核は、サンプルトークンの log-prob 差分の形で実装済み。ただし (i) 全語彙 KL ではなくサンプル済みトークンの log 比、(ii) 対比は Full vs Ablated（どちらも replay scaffold 付き）で「教師 vs 特権なし同一モデル」そのものではない、(iii) 用途は蒸留損失の重みではなく GRPO advantage の変調。
- **診断値（一次情報）**: step 平均した信号の valid/invalid action 判別 AUROC は Ablated 0.464 / Full 0.654 / residual 0.707。
- **重なり度: 部分的に近い**
- **潰れる主張**: (a) の「発想そのものの新規性」は大きく弱まる（差分による特権帰属の分離は既出）。(b) は蒸留損失への適用としては潰れないが、「特権由来信号による連続変調」一般は既出。

## 5. SDPO（Hübotter et al.）

- **実在確認**: arXiv:2601.20802（v2、2026-02-16 改訂）。著者: Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, Andreas Krause。題名 "Reinforcement Learning via Self-Distillation"。コード https://github.com/lasgroup/SDPO（サイクル1検証）。
- **特権の扱い**: 現在のモデルをフィードバック f（環境フィードバック、成功した sibling rollout 等）で条件付けたものを self-teacher とし、その次トークン予測をポリシーへ蒸留する。損失はトークンごとの発散（Harne et al. §8 の統一式(1)、RLSD §6 の記述と整合）を**一様重み**で課すもので、leakage の測定もトークン重み付けも原論文にはない。後続の SC-SDPO（arXiv:2605.27765）は質問単位の pass-rate 重みを導入したが、トークン単位でも特権由来でもない。
- **重なり度: 主張 (c) の根拠そのもの**（SDPO 原論文については w≡1 の記述が成立）
- **潰れる主張**: なし。ただし (c) を「2026年の後続文献全体」への主張に広げることはできない（§6–§11）。

## 6. RLSD: Self-Distilled RLVR（★ 横断検索で発見した最重要の先行研究）

- **実在確認**: arXiv:2604.03128（v2 全文 HTML 精読）。著者: Chenxu Yang, Chuanyu Qin, Qingyi Si, Minghui Chen, Naibin Gu, Dingyu Yao, Zheng Lin, Weiping Wang, Jiaqi Wang, Nan Duan。題名 "Self-Distilled RLVR"。
- **理論**: 情報非対称の下での分布マッチングは不可能であることを証明。OPSD 目的関数には生徒が決して消せない **irreducible mutual information gap I(Y_t; R | X, Y_<t) > 0** が含まれ（Theorem 1）、サンプルごとの勾配はこの MI に比例する分散の r 固有偏差を持つ。すなわち **leakage の情報理論的な定量化は既出**。
- **トークン単位の定量化**: **privileged information gain** Δ_t = sg(log P_T(y_t) − log P_S(y_t))（式13）。同一モデルの特権あり／なし両条件で各トークンの log 確率を計算し、その差を「特権情報がそのトークンへの belief をどれだけ改訂したか」と定義する。これは (a) のサンプルトークン版そのものであり、本ファイル冒頭のとおり ℓ_t はその教師分布期待値である。
- **重み付け**: **direction-aware evidence reweighting** — w_t = exp(sign(A)·Δ_t) = (P_T(y_t)/P_S(y_t))^{sign(A)}（式14）を clip して GRPO の advantage に乗算（式15–16）。**特権由来のトークン量による連続的な重み付けは実装済み**。ただし対象は蒸留損失ではなく RL の advantage であり、方向も「正解軌跡で特権に支持されるトークンの credit を増やす」＝証拠重み付けであって、leakage の大きいトークンを抑制する発想ではない。RLSD の設計思想はむしろ逆で、分布マッチング（蒸留損失）自体を捨て、方向は環境報酬に委ねる。
- **統一視点について**: §4.3 で GRPO / on-policy self-distillation / RLSD を単一の policy gradient テンプレート（Â_t の定義だけが違う）として統一する議論を既に提示している。
- **重なり度: 部分的に近い（ただし (a) に対してはほぼ完全一致に近い）**
- **潰れる主張**: **(a) は実質的に潰れる**（定量化の対象・定義・「同一モデルの特権有無の差」という構成がすべて既出。残る差は全語彙 KL かサンプル log 比かという集約方法のみ）。**(b) は「特権量による連続重み付け」一般としては潰れる**（適用先が蒸留損失でない点、方向が逆である点のみが差分）。(c) の「統一視点」も、統一という行為自体は RLSD が先行している。

## 7. DOPD: Dual On-policy Distillation

- **実在確認**: arXiv:2606.30626（v1 全文 HTML 精読）。著者: Xinlei Yu, Gen Li, Qingyi Si, Guibin Zhang, Yuqi Xu, Congcong Wang, Shuai Dong, Kaiwen Tuo, Xiangyu Zeng, Kaituo Feng, Qunzhong Wang, Yang Shi, Xiaobin Hu, Xiangyu Yue, Jiaqi Wang, Shuicheng Yan。題名 "DOPD: Dual On-policy Distillation"。
- **用語**: abstract で "a failure mode **we dub privilege illusion**" と自称しており、この語の導入者は DOPD である可能性が高い（OPSD 原論文には現れない。初出の厳密な同定は未確認）。
- **手法**: **privilege advantage gap** A = |log Π_T(y_n|x, p, y_<n) − log Π_S(y_n|x, p, y_<n)|（式1）。重要: 両側とも特権 p を条件に含む。解釈は「両者が特権を見る条件では、大きい gap = 能力差、小さい gap = 教師の優位は特権由来」。この A と相対確率 q_S, q_T をバッチ平均と比較し、トークンを4レジームに振り分け: LH（軽い教師蒸留 β_l·Top-K reverse KL）、LL（弱い自己正則化 β_w）、HT（重み1の full-vocab JS）、HS（軽い特権生徒蒸留 β_l）。総和は indicator マスクの和（式10）。実装値は β_w=0.3, β_l=0.6, 単位重み=1.0。
- **アイデア3との差**: 信号 A は「特権あり同士」の gap であり ℓ_t（特権あり vs 特権なし）とは別の対比。重み付けは連続関数ではなく4値の離散ルーティング＋固定係数。
- **傍証（重要）**: 高 A トークンを除く ablation で性能向上が約半分（VLM では約20%）に落ちる一方、低 A トークンの除去はほぼ無影響（§3.1.2）。これは「教師–生徒 gap の小さいトークンは捨てても痛くない」という方向の証拠であり、アイデア3の「ℓ_t 大のトークンを弱める」とは別軸だが、トークン選択の実証的知見として実験設計時に参照すべき一次情報。
- **重なり度: 部分的に近い**
- **潰れる主張**: (b) の「トークンごとに監督の強さ・目的を変える」という設計思想は既出（ただし離散）。(a) は潰れない（A は leakage ではない）が、「gap で能力成分と情報成分を分ける」という発想自体は既出になる。

## 8. Harne et al.: Privileged, but Biased

- **実在確認**: arXiv:2608.04794（v1 全文 HTML 精読）。著者: Sarthak Harne, Chinmay Karkar, Yash Pandya, Ahmed Awadallah, Akshay Nambi（Microsoft Research）。題名 "Privileged, but Biased: How PI-Conditioned Teachers Break Self-Distillation"。
- **PI Bias Score の定義（式2）**: 対象系列 w に対し PS_t(w) = s^T_t(w) − s^S_t(w) = (1/K) Σ_k log [p_T(w_k|x, y*, y_<t) / p_S(w_k|x, y_<t)]。教師−生徒の log 確率比（平均）であり、forward pass のみで計算できる。実測: in-context 解で 0.52、**同じ問題の別の正解解では ≤0.02**（無関係な問題の解とほぼ同水準）→ 教師は「正しさ」ではなく「見せられた特定の軌跡」を符号化している。
- **位置づけ**: これは (a) と同じ族の量（特権条件付き教師と特権なし生徒の log 比）を**診断目的**で定量化したもの。ただし (i) 対象系列に対する平均でありトークン位置ごとの全語彙 KL ではない、(ii) 損失の重みには使われていない。また同論文は RLSD の I(Y_t; R|X, Y_<t) を「その理論版」と位置づけている。
- **その他の実測（実験設計に有用）**: 損失の大半は低情報トークン（stopword・句読点・不確実性マーカー等）に落ちる（代表 step で 55.38%／要旨では 55.4%）。正解 rollout 内でも off-path トークンの KL は ≈0.31 vs on-path ≈0.08。SD 単独目的は困難タスクで検証精度を悪化させる（agentic で平均 −3.51、個別ベンチで −7.0）。
- **重なり度: 部分的に近い**
- **潰れる主張**: (a) の「特権依存を log 比で測る」という部分は既出（診断用途）。(b) は潰れない。

## 9. Purified OPSD

- **実在確認**: arXiv:2607.02234（abstract 精読、2026-07-02 投稿）。著者: Zhanming Shen, Jintao Tong, Shaotian Yan, Chen Shen, Hao Chen, Wentao Ye, Xiaomeng Hu, Rui Miao, Haobo Wang, Junbo Zhao, Gang Chen, Jieping Ye（浙江大学 / Tongyi Lab）。題名 "Purified OPSD: On-Policy Self-Distillation Without Losing How to Think"。E2-OPSD が分類した "signal purification" 系の Shen et al. 2026 は本論文。
- **手法**: 教師の監督信号を、(i) reference-induced component（質問なしでも存在し、参照固有の近道を暗記させる、非転移成分）と (ii) question-conditioned で推論転移可能な成分に**分解**。参照のみを条件付けた reference-only teacher を作って非転移成分を分離し、差し引いた残差を PMI で well-formed な PMI target distribution に変換して生徒が蒸留する。
- **アイデア3との関係**: 重み付けではなく**ターゲットの変換・差し引き**。しかし「教師信号の特権由来成分をトークン単位で分離・定量化して取り除く」という (a) の周辺思想はここにも実装されている。
- **重なり度: 部分的に近い**
- **潰れる主張**: (a) の周辺（信号の特権成分の分離という発想は既出）。(b) は潰れない。実験では必須の比較対象（重み付け vs ターゲット差し引き）。

## 10. USD: Matching Supervision to the Student's Learning Capacity

- **実在確認**: arXiv:2608.08176（abstract 精読、2026-08-08 投稿）。著者: Yongkang Yang, Zhezheng Hao, Hong Zhang, Yi Liu, Xiankun Lin, Wence Ji, Fanjunduo Wei, Jiarui Yu, Qiang Lin, Xiaoyun Liang, Hande Dong。題名 "Matching Supervision to the Student's Learning Capacity: A Unified Framework for On-Policy Self-Distillation"。コードは abstract 内リンク（github.com/lauvlalala/USD と検索結果に表示）。
- **内容（abstract の範囲）**: OPSD の改良は「どのトークンから学ぶか（token weighting）」と「教師にどれだけ特権を見せるか（PI control）」の2系統があり、両者は生徒の学習容量を通じて結合している（特権情報がトークンごとの発散を決め、重み付けがその吸収先を選ぶ）。両者を「総発散の最大化、ただし吸収可能な総学習困難度に予算制約」という統一最適化として定式化し、単一の双対変数がトークン選択閾値と特権調整方向の両方を決める USD を提案。
- **重なり度: 部分的に近い**
- **潰れる主張**: (c) の「統一視点」一般は既出（ただし USD の統一は容量予算によるもので、「w≡1 の特殊ケース」という形ではない）。(a)(b) の具体形は潰れない。手法詳細（重みの具体形）は abstract の範囲を超えるため**未確認**。

## 11. SERL（What and When to Distill）

- **実在確認**: arXiv:2605.19447（abstract 精読、2026-05-19 投稿）。著者: Xiaozhe Li, Tianyi Lyu, Yang Li, Yichuan Ma, Peiji Li, Linyang Li, Qipeng Guo, Dahua Lin, Kai Chen。題名 "What and When to Distill: Selective Hindsight Distillation for Multi-Turn Agents"。コード https://github.com/OliverLeeXZ/SERL（検索結果の本文スニペットに記載）。
- **内容**: 環境条件付き教師が、生徒自身の action token を hindsight feedback あり／なし両方でスコアし、その **log-probability gap を bounded で sign-aware な GRPO advantage の reweight に変換**する。蒸留は実行可能な action span に限定し、教師信号は訓練後期の privileged-information leakage を防ぐため減衰させる（以上は検索で取得した本文スニペットに基づく。abstract は「task reward が方向を決め、environment feedback が配置と大きさを調整する」と整合）。数式の正確な形は**未確認**（全文未精読）。
- **重なり度: 部分的に近い**
- **潰れる主張**: (a)(b) の RL 版パターンの追加の実例（特権あり／なしの gap → 連続 reweight）。蒸留損失への適用としては潰れない。

## 12. Taste-Bench 蒸留（Pan et al.）— 主張 (c) の対象確認

- **実在確認**: arXiv:2609.25804 "The Tasteful Agent: Measuring and Improving Taste in Long-Horizon Tasks"（Pan et al.、Microsoft / CityU HK）。サイクル1で全文精読済み（reports/2026-11-08_cycle-01.md）。データ: https://huggingface.co/datasets/wenbopan/taste-bench（アクセス申請制）。
- **蒸留の形**: 支持された候補を見せられた privileged teacher（同一の Qwen3.6-27B）から、生徒 LoRA へ SDPO 式の forward KL（teacher サンプル継続に対する）で蒸留。held-out で 30.0% → 47.9%（+17.9pt）。トークン重みは一様であり、leakage の測定・重み付けはない。
- **判定**: (c) の記述は Taste-Bench 蒸留についても原論文の範囲では成立。
- **注記**: E2-OPSD が "signal purification (Pan et al., 2026)" として引く Pan et al. 論文が本論文と同一かは**未確認**（E2 の文脈では別の purification 論文の可能性が残る）。

## 13. その他の横断検索ヒット（存在のみ／詳細未確認のものを含む）

- **AVSD (Nguyen et al., 2026)**: OG-OPSD の関連研究記述によれば複数の特権教師 view を組み合わせる手法。DAPD の分類では「selective transfer（filter/reweight）」側に Nguyen et al. 2026 が挙がる。原論文の同定・詳細は**未確認**。アイデア3の (b) に最も近い可能性のある未確認項目であり、本格着手前に一次確認が必要。
- **CaOPD** "The Illusion of Certainty: Decoupling Capability and Calibration in On-Policy Distillation"（arXiv:2604.16830、abstract を検索結果で確認）: 特権文脈が entropy collapse と optimism bias を生むことを情報非対称として定式化し、capability と calibration を分離する target 置換で対処。重み付けではない。
- **EOPD**（arXiv:2603.07079、サイクル1検証済み）: 教師エントロピーの高いトークンで forward KL を追加するエントロピー駆動の調整。特権発散による重み付けではない。
- **DASD**（arXiv:2605.22263）: エントロピーで方向を変える自己蒸留（高エントロピーは教師から遠ざける）。信号はエントロピー。
- **GUI-SD**（arXiv:2605.00642）/ **Visual-SDPO**（arXiv:2606.10334）: トークン重み付けは存在するが、駆動信号は digit significance・教師信頼度・欠陥責任スコア等で、特権発散ではない。
- **SC-SDPO**（arXiv:2605.27765）: 質問単位の pass-rate 重み。トークン単位でも特権由来でもない。
- **ST-OPSD**（arXiv:2610.34005）/ **DualOPSD**（arXiv:2608.26019）/ **OPD²**（arXiv:2607.15161）/ **Kaur et al.**（ICML 2026 FoGen workshop）: いずれも特権条件付けの害への対処だが、活性化ステアリング／交互更新／教師–ベースモデル間 delta／診断、であり leakage 重み付けではない（存在と概要は検索・サイクル1の範囲。詳細は未確認のものを含む）。

---

## 総括

### 主張ごとの判定

| 主張 | 判定 | 根拠 |
|---|---|---|
| (a) leakage のトークン単位の定量化そのもの | **潰れる** | RLSD の Δ_t（同一モデルの特権有無 log 比、式13）とその理論版 I(Y_t; R\|X, Y_<t)（Theorem 1）が既出。ℓ_t は Δ_t の教師分布期待値に等しい。Harne の PI Bias Score、OCSD の δ／residual も同族の定量化。残る差は「全語彙 KL として集約する」という表現のみ |
| (b) それによる連続的な損失重み付け | **蒸留損失への適用に限れば生存、ただし強く圧迫されている** | 連続重み付けの形式は E2-OPSD（エントロピーギャップ信号）で蒸留損失に実装済み。特権由来量による連続 reweight は RLSD / OCSD / SERL が RL advantage 側で実装済み。「蒸留 KL 損失 × 特権発散信号 × 減少方向」の三点セットは、今回精読した範囲では未確認 |
| (c) OPSD / Taste / SDPO は w≡1 の特殊ケースという統一視点 | **原論文3件については事実として成立するが、新規性としては弱い** | 3件とも一様重みを確認（OPSD は pointwise clipping の留保付き）。ただし「統一視点」自体は RLSD §4.3（token-level advantage テンプレート）と USD（容量予算による統一）が先行。2026年後続文献全体に広げると w≡1 でない手法が多数あり、主張の射程は原論文3件に限定される |

### 生き残る差分（ここまで狭めるべき）

1. **重みの信号**: 全語彙 D_KL(p_T(·|prefix, z) ‖ p_0(·|prefix))。ただし p_0 は訓練中の生徒ではなく**凍結した特権なし参照モデル**でなければならない（そうでないと損失自身による自己参照的重み付け＝クリッピングの連続版に退化し、OPSD の pointwise clipping との差が消える）。
2. **方向**: leakage が大きいトークンほど蒸留損失を**弱める**（減少関数）。これは RLSD（証拠のあるトークンを増幅）とも E2-OPSD（ミスマッチの大きいトークンを増幅）とも逆の符号選択であり、この符号で勝てるかは未検証の実証問題。
3. **適用先**: 報酬なしの純粋な蒸留損失の中での重み付け（RLSD/OCSD/SERL はすべて RL 側に逃がしており、蒸留損失単独の枠組みでは未実装）。

### 実験で示すべき最小の主張

「特権バイアスが実測できる設定（例: Harne et al. の困難タスク設定、または Taste-Bench engineering fork）において、凍結参照モデルで測った全語彙 ℓ_t による減少型重み付け蒸留が、以下すべてを上回る」

1. w≡1（vanilla OPSD / Taste 式蒸留）
2. **OPSD の pointwise clipping**（「発散値による重み付け」の既存の粗い形。これに勝てなければ貢献は消える）
3. エントロピーギャップ重み付け（E2-OPSD の AGD 方式）
4. サンプルトークン Δ_t を信号に使った重み付け（RLSD 信号の蒸留版アブレーション。全語彙 KL で測ること自体の価値の検証）

併せて、改善が「高 ℓ_t トークン由来の振る舞い変化」（wrong claims の減少、PI Bias Score 的な測定でのバイアス低下）と結びついていることを示すこと。精度だけの改善では、重み付け一般の効果と区別がつかない。

### リスク・注意

- Harne et al. の実測では、損失は正しさにほぼ盲目であり、高発散トークンの抑制は「信号の大部分を捨てる」ことと紙一重になりうる。DOPD の ablation（高 gap トークン除去で向上が半減）も、gap の大きいトークンに実質的な学習内容が含まれる可能性を示唆する（ただし DOPD の gap は別定義）。
- DAPD の主張する構造的限界（信号をいじる限り情報非対称は残る）は、重み付け系すべてへの批判として実験報告で正面から扱う必要がある。
- 本格着手前の残タスク: **AVSD (Nguyen et al., 2026) の一次確認**（DAPD の分類で reweight 系に挙げられており、(b) を直接潰す可能性がある未確認項目）。および E2 の言う "Pan et al. 2026" purification 論文の同定。

### 未確認事項の一覧

- "privilege illusion" の厳密な初出（DOPD が自称するが、他論文での先行使用の有無は未確認）
- AVSD (Nguyen et al., 2026) の実在詳細・手法
- E2-OPSD が引く "Pan et al., 2026"（signal purification）の同定
- USD の重みの具体形（abstract の範囲を超える詳細）
- SERL の reweight の数式の正確な形（本文スニペット＋abstract で整合は確認、全文未精読）
- OPSD のコード（github.com/siyan-zhao/OPSD）の実装詳細（論文式(1)＋clipping の記述で判断）
