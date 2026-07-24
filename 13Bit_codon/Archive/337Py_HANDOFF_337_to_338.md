# 引継ぎメモ: rev337時点 → rev338開始に向けて

作成日: 2026-07-24(今回セッション終了時点)

## 現在の状態

- **最新確定リビジョン: 337**(bin-format-reader-design)。クローズ済み。
- メインソルバのカーネル・ディスパッチャは**328以来無変更**(w3_j7、K=48、SoA w_lo_arr/w_hi_arrのまま)。
- 添付予定の `337Py_bin_format_reader_design.py` に、この経緯全体がOpen ObjectivesとVERSION_TAGとして埋め込まれています。`README.md` にも同内容の年代順ログがあります。**このメモは要点だけの補助であり、詳細はソース/READMEが一次情報です。**

## 今日(334→337)の要点

課題3(tail effect、stall_branch_resolvingの65.5%集中)への唯一の残存対策経路として、rev84提案・数か月未着手だった**CUDA Cランナー(発掘案C-2)**のスパイク設計を開始し、以下を実機で確定・実証した:

| rev | 内容 | 結果 |
|---|---|---|
| 334 | nvcc実機確認 | `/usr/local/cuda/bin/nvcc`、CUDA 13.0(V13.0.88) |
| 335 | cuobjdump再探索・compute_cap実測 | cuobjdumpはこのCUDA 13.0ツールキットに**同梱されていない**ことを確定(SASS逆アセンブルは今後`sudo ncu --section SourceCounters --page source`に一本化)。A10G compute_cap=**8.6**(`-arch=sm_86`確定) |
| 336 | 最小CUDA Cスモークテスト(`336_cudac_smoke_test.cu`) | ビルド・GPU実行ともに成功、`match=True`。**cudacodon実機でのnvccビルド+GPU実行round-tripを初実証** |
| 337 | bin形式の仕様確定+C側リーダー(`337_bin_format_reader.cu`) | ソース精読で「ヘッダーなし・16バイト固定長レコード(ld/rd/col/startijkl、各u32 LE)のフラット配列」と確定。実ファイル`constellations_N21_6.bin`に対し検証: `records_read=2025282`(=`EXPECTED_TASKS`と一致)、`checksum_u64=13342728758502` |

いずれもメインソルバのカーネル・ディスパッチャには一切触れていません(N=21フルvalidateは334-337の全リビジョンで正当性314666222712・elapsed 450秒台前半のノイズ内再現を継続確認済み)。

## 338で着手する予定の内容

**方針: まずコードを書かない設計リビジョンにする**(324→325と同じパターン。いきなり実装せず、先に仕様を固める)。

`kernel_dfs_iter_gpu_maxd14`のN=18限定移植に向けて、337までで片付いていない残課題:

1. **SoA配列の導出ロジックをC側で再現する設計**
   bin上のld/rd/col/startijklから、`symmetry`関数によるw_arr(→w_lo_arr/w_hi_arr分割)、mark1/mark2/funcid/ijkl各配列がどう導出されるか(現在はCodon側のホストコードが担っている)を仕様化する。
2. **K-batchingのメモリレイアウト仕様化**
   K=48、`stack_bytes_per_thread=208`のスタック構造(save_sp/next_depth/cur_depth/stack_ptr含む)を、CUDA C側の構造体・配列としてどう表現するかを設計する。
3. **N=18限定の正当性確認スコープの具体化**
   rev84の原提案どおり、N=18で移植版と既存Codon版の出力を突き合わせる最小構成の設計(N=21フルはこの後)。

338はこの3点の設計文書化のみを行い、実装(`.cu`コード)は339以降に回す想定です。

## 来週の開始手順(いつもどおり)

1. `337Py_bin_format_reader_design.py` を添付いただく(このメモも一緒に共有いただければ経緯の再確認が省けます)。
2. 338の設計案を提示します。
3. 承認いただければ338の3ファイル(`.py`/`.sh`/README更新)を作成します。338は設計のみのためGPU実行を伴わない可能性が高いですが、通常どおりN=21フルvalidateでの無変更確認は行います。

以上です。今日もお疲れさまでした。良い週末を。
