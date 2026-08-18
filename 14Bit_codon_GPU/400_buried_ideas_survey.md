# 過去の埋もれた最適化案 発掘調査 (2026-07-23)

対象: Py_tar.gz 全413ファイル(rev 1〜329)のヘッダーコメント + README.md 全146エントリ。
docstring内のアイデア語彙(案/候補/検討/保留/不採用/撤回/次の一手/未着手)を持つ122ファイルを走査し、
「僅差で不採用」「提案されたが未実施」「構造的に無効化済み」の3系統に分類した。

---

## カテゴリC: 提案されたが一度も実施されていない案 (最有力)

### C-1. 動的ワークスティーリング / persistent kernel (rev 292「案C-1」、未着手のままブロック)

292のヘッダーに完全な形で埋まっていた。当時のncu実測(279系)で
`Avg Active Threads Per Warp = 4.88/32`(SIMT効率15%)、`Achieved Occupancy = 11.19%`
が判明し、「静的な並べ替えではこの粒度のばらつきは消せない。動的ワークスティーリングは
実装コストは大きいが、狙える上振れ幅は静的リオーダー系とは桁が違う可能性がある」と
明記されている。

- 当時の判断: Codonの`@gpu.kernel`がatomic演算をサポートしているか不明
  (exaloop/codon#588のDiscussionが未回答のまま)。「もしなければこの方向は
  現状のCodonでは詰み」とし、**「お手元でcodon stdlib gpu.codonのソースにatomicらしき
  関数がないか確認いただけますか」という確認依頼が出されたまま、記録上、一度も
  実施されていない。** 代わりにatomic不要の近似案C-2(K-batching)が292で実装され
  -4.4%を獲得、そのまま本線がK sweep(304-308)へ進んだ。
- **329の新知見との接続**: 329のSourceCounters再解析で、stall_branch_resolvingの65.5%が
  集中する2箇所のPCは`Divergent Branches=0`かつ`Avg Threads Executed=2〜3/32`、すなわち
  occupancy collapse / tail effectだと特定された。これは292当時の4.88/32の発見と同一の
  現象であり、**292案C-1がまさにこの問題の唯一の原理的対策として提案されていた**。
  5戦5敗のkernel分解とはアーキテクチャ的に別物である点も292時点で認識済み。
- **次の一手(コスト: bashコマンド1本、リスク: ゼロ)**:
  `grep -in "atomic" $(codonのインストール先)/stdlib/gpu.codon`
  atomicが存在すれば、persistent kernel + グローバルタスクキューが330番台の本命候補になる。
  存在しなければC-2へ。

### C-2. CUDA C ランナー (rev 84の提案、未着手)

84のヘッダー: 「私なら次の一手は、84 streamの.binを読むCUDA C runner」。
コンステレーション生成・並べ替え・binキャッシュはCodonのまま維持し、
**カーネルとランチャーだけをCUDA C(.cu)で書いて同じbinを読む**構成。一度も試されていない。

これが解禁するもの(全てCodonツールチェーンで「到達不能」と結論済みの項目):
1. **warpレベルintrinsics** (`__shfl_sync`/`__ballot_sync`/`__activemask`)
   → warp内レーン圧縮・warp協調DFS。**tail effect(2〜3/32)への唯一の直接対策**で、
   atomicの有無に依存しない(warp内はatomic不要)。
2. **atomic演算** → C-1のpersistent kernelがCodonのサポート状況と無関係に実装可能。
3. **`-lineinfo`付きnvccビルド** → per-line SASS帰属。318-321で「Codon+ncuでは
   恒久的に到達不能」と結論した制約が丸ごと消える。
4. `__launch_bounds__`、shared memory、`#pragma unroll`等の細粒度制御。

コストは大きい(カーネル移植 + 正当性再検証一式)が、リスクの質はkernel分解とは異なる
(アルゴリズムは不変、言語だけ変更)。カーネルは現在1個(maxd14が実質全て)なので
移植対象は限定的。

### C-3. w8_j7 (funcid_reorder window_mult/phase_jump) のN21再スイープ

現行の`funcid_reorder_v2_params: w8_j7`のreason文字列は今も「**N22** measured best
baseline w8_j7」。94-99時代(2026年6月上旬)にN22で決めた値が、その後の
chunkshape148(128-148)、nibble schedule(168)、root-preroll(196-232)、
K-batching(292-308)、variant2(311)という**下流の全面改変を経て一度も再検証されていない**。
95のスイープはw10/12/16/32×j5/7/11のみでw8_j7自体は当時のbaseline由来。
ホスト側パラメータのみでカーネル無変更、既存のsweep手順が流用可能な低リスク案件。

---

## カテゴリA: 僅差(ノイズレベル)で不採用になった案

正直な評価を先に: これらは「悪化したから」ではなく「改善しなかったから」棄却されており、
現在の文脈で再試行しても期待値はノイズレベル。ただし記録として:

| rev | 内容 | 当時の差 | 当時の判断 |
|---|---|---|---|
| 173 | frame metadataのentry時一括decode | +0.008% | 複雑化に見合わず不採用 |
| 188 | futuremask guard | +0.139% | 「差は小さいが184を上回らず」不採用 |
| 191 | 配置演算のOR→XOR置換 | +0.002% | 同等、不採用 |
| 284 | (root系probe) | +0.023% | 不採用 |
| 293 | dual lane | +0.065% | 不採用 |
| 299 | K=64 on 296 | +0.064% | K sweepでK=48採用 |

この中で唯一、文脈変化により意味が変わりうるのは173(decode一括化)だが、
168のnibble scheduleと296のstack_ptr化が同じレジスタ圧範囲を既に最適化しており、
優先度は低い。

## カテゴリB: 保留のまま格上げされ忘れた案

- **rev 45: funcptn sort「採用候補に格上げでよいです」** → その後100-148の
  broadmarktail/chunkshapeパイプラインに全面的に置き換えられ、事実上消滅。
  再発掘の価値なし(下記Dに実質統合)。
- **rev 45: max_blocks=640 / sort_mode=2 保留** → 同上。ただしmax_blocks系は
  K-batchingでstride=grid×blockとして再結合されており、304-308のK sweepが
  実質的にこの軸を再探索済み。クローズ扱いで妥当。

## カテゴリD: 発掘不要(再試行してはいけない)と確認できたもの

1. **block size再スイープ**: 292のヘッダー自身が「README 1717行目: block size
   32/64/128スイープ→32が最良、占有率を上げても速くならないことは実測済み」と
   車輪の再発明を自己撤回している。しかも「占有率を上げても効かない」という
   この結果自体が、tail effect(warp内早期終了)が真因であることの傍証として
   292→329で一貫している。**再試行不要、ただしこの否定的結果はC-1/C-2の
   動機づけとして価値が高い。**
2. **kernel/ループ分解**: 240/266-269/273/326の5戦5敗+326(-13.7%)。確定クローズ。
3. **bucket/expand系(25-31)、warmup/repeat(68-71)**: 現行パイプラインで構造的に無効。
4. **BROADMARK variant 1/4**: 309/310で-36〜37%の大差却下。確定。

---

## 提案: 330番の設計

**330(即日・リスクゼロ)**: Codon atomicサポート調査。
cudacodon上で `find / -name "gpu.codon" 2>/dev/null` → 該当ファイルを
`grep -in "atomic"`。結果をREADMEに記録するだけの調査リビジョン
(316のenv調査と同じ位置づけ)。同時に、C-3のw/jスイープ用スクリプトを準備。

**331以降の分岐**:
- atomicあり → persistent kernel probe(292案C-1の実装)を設計。
- atomicなし → C-3(w/jスイープ、ホスト側・低リスク)を先に消化しつつ、
  C-2(CUDA Cランナー)のスパイク(maxd14カーネル1個の移植+N=18での正当性確認のみの
  最小構成)を別トラックで計画。

期待値の序列は C-1/C-2 ≫ C-3 > カテゴリA再試行。tail effect(branch-resolving stallの
65.5%、全stallの22.6%)は現在確認されている最大の単一ボトルネックであり、
カテゴリCの2案だけがこれに正面から届く。
