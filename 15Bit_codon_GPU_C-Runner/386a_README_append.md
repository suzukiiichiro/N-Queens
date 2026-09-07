## 386a: bench_mode==28/29(broadmarktail単独実行)の削除

**位置づけ**: 381a〜381eの流儀にならい、386の棚卸し表のうち削除確認が
取れた項目を1つずつ小さく刻んで進める最初のステップ。

**確認事項の整理(このセッションでの確認)**:

| 項目 | 結論 |
|---|---|
| bench_mode=28/29(broadmarktail単独実行) | **削除してよい**(直近使っていない) |
| worker_id/worker_count(マルチワーカー分割) | **残す**(訂正: A10G 4枚構成インスタンスがあり、将来マルチGPU対応で使う) |
| broadmark_tail_variantのCLI上書き経路 | **削除してよい**(variant=2以外を試す予定なし)→386bで対応 |

**386aで削除したもの**:

- `bench_mode==28`(broadmarktail-reorder-sim-only)と`bench_mode==29`
  (broadmarktail-reorder-gpu)の標準入口2つ。いずれも`build_broad_
  markdist_tail_reordered_bin()`を単独デバッグ用に叩くだけのモードで、
  330で`BROADMARK_VARIANT`軸がクローズして以降、単独で使う場面が無く
  なっていた。
- CLIGATE(bench_modeホワイトリスト)・PRESETGATEからの28/29除去。
- argv解析: `bench_mode==29`が`worker_id`/`worker_count`/
  `broadmark_tail_variant`/`chunkshape148_*`を読んでいた箇所は
  `bench_mode==31`専用に絞り込み(31はこのままworker分割対応を維持、
  上表の通り残す方針のため)。
- `gpu_log_level>=1`時の28/29用ログ行。
- `main()`Nループ内の28/29それぞれの実行分岐(計約40行)。

**触れていないもの**:

- `build_broad_markdist_tail_reordered_bin()`自体は**1行も変更していない**。
  `bench_mode==31`(chunkshape148フルパイプライン)が引き続き内部で
  同じ関数を呼ぶため。
- `worker_id`/`worker_count`まわりの配線は完全に維持。

**新たに見つかった孤立関数(未対応)**: `exec_solutions_gpu_bin_stream_
funcid_reorder()`は、唯一の呼び出し元だった`bench_mode==29`が消えた
ことで、定義はあるがどこからも呼ばれない関数になった。**今回は削除
していない**(鈴木さんが確認したのは「bench_mode=28/29を使うか」で
あり、この関数個別の要否ではないため)。次の棚卸し候補として記録のみ。

**事前予測**: `bench_mode==28`を指定すると、既存の`[warning] bench_
mode=... was removed`の仕組み(276/361/365/366/368/369で使われてきた
のと同じ経路)でクリーンに弾かれる、と予測する。`bench_mode==31`
(N=21)は削除前と完全に同じ`total=314666222712`を返す、と予測する
(共有関数は無変更のため)。**この`bench_mode==31`経路は385/386で使った
`bench_mode==37`のCバイナリ経路より遅い**(Codon側のsplit145/
chunkshape148パイプラインを丸ごと通るため)。過去のログでは同条件で
7分台の実行例があるので、385/386の`bench_mode==37`(約3分23秒)より
時間がかかる想定で問題ない。

**結果**: 2026-09-07、cudacodon実機で`bash 386a_validate.sh`実行、
`OK=10 FAIL=0`で`386a PASSED`。

- `bench_mode==28`指定時、既存の`[warning] bench_mode=28 was removed in
  276 restore274/coretrim; using 0`機構でクリーンに弾かれることを確認
  (N=5のCPU計算にフォールバックし、クラッシュも誤動作もなし)。
- `bench_mode==31`(N=21): reorder bin・chunkshape148 binとも初回ビルド
  から通しで実行し(`[split291-base-build]`→`[chunkshape148-build]`→
  3チャンクの`[split145-gpu-chunk-end]`)、`total=314666222712`
  (オラクル完全一致)、`0:05:15.254`で完走。共有関数`build_broad_
  markdist_tail_reordered_bin()`が無変更のまま正しく動作することを
  確認した(予測通り、385/386の`bench_mode==37`約3分23秒より遅い)。

**確定**: bench_mode=28/29の削除は実機で無害と確認された。

**ドキュメントのみの訂正**: 386の段階で、ヘッダーdocstring内の項目7
見出し行(`7. [解決・351で採用・352で軸をクローズ] 上位半分は恒等的に
ゼロ`)が編集中に脱落していたことに気づき、386a・出力済み386Pyの両方で
復元した。実行コードのdocstring外領域には影響しない(コード自体は
385/386で実機確認済みの通り無変更)が、ドキュメントの整合性としてここに
記録する。
