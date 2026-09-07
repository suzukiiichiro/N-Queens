## 385: maxd-gated CRunner os.systemディスパッチ

**位置づけ**: 384で実機確認した「ローカルmultiファイルimport」「os.system+
ログファイル往復」の2つの仕組みを、実際に本番の`364_kernel_maxd14`バイナリへ
向けて配線した最初のrevision。383は欠番(384がfeasibility spikeとして
その枠を消費)。

**変更点(すべて加算のみ、既存コードは無変更)**:

- `CRunnerEntry`クラス: `maxd_max` / `binary_path` / `env_prefix` /
  `done_prefix` / `correctness_prefix`を保持する対応表の1エントリ。
- `crunner_dispatch_table()`: 現時点では`364_kernel_maxd14`
  (`maxd_max=14`)の1件のみを登録。CLI引数順(`<N> <in_soa7_bin>
  <out_results_bin> [expected_total]`)と出力マーカー
  (`[gpu-run-done]` / `[gpu-run-correctness]`)は、364_kernel_maxd14.cu
  の実ソース(729〜738行)から直接確認した値であり、推測ではない。
- `crunner_select_entry_index()`: `required_maxd`に対して最初に
  条件を満たすエントリのインデックスを返す(未対応なら-1)。
- `crunner_parse_result()`: ログファイルから`total_sum=` /
  `kernel_ms=` / `MATCH`・`MISMATCH`をパース。
- `crunner_run()`: `os.system()`でバイナリを起動しログへリダイレクト、
  上記パーサで結果を回収。オラクル一致(`MATCH`行)が確認できない限り
  `ok=False`を返す(oracle-first: CRunner側が確認していない値は
  正としない)。
- 新設`bench_mode=37`: `main()`のNループに追加。`check_required_maxd_
  for_N()`(366、無変更)でrequired_maxdを求め、対応表になければ
  `[crunner-unsupported]`と明示して`continue`(誤答を出さない)。

**364_kernel_maxd14.cuは1行も変更していない。** 382の`.cu`
(`382_kernel_dfs_hybrid.cu`)がすでに364と同一のフィールド名
(`total_sum=`/`kernel_ms=`)を使っているため、今週追加予定の
maxd16/18/20/21バイナリがこの形式(タグは`[gpu-hybrid-run-*]`のままで
可)を踏襲すれば、`crunner_dispatch_table()`へ1行追加するだけで
組み込める設計にした。

**事前予測(実行前に記録)**:

- N=21で`bench_mode=37`を実行すると、`[crunner-dispatch-summary]`の
  `total`が`314666222712`(オラクル)と一致し、テーブル行のステータスが
  `ok`になる、と予測する。
- `kernel_ms`は374本番の実測アンカー(`201232.422`)に近い値になる、と
  予測する。`os.system`起動やログファイルI/Oのオーバーヘッドは
  CUDA eventで計測される`kernel_ms`区間の外側(ホスト側)なので、この
  値自体を動かすとは考えていない。動いた場合は、パース側の取り違え
  (別セッションの古いログを読んでいる等)を先に疑う。

**結果(r1)**: 2026-09-07、cudacodon実機で`bash 385_validate.sh`実行。
静的チェック・`codon build -release`とも一旦OKに見えたが、型注釈ミス
(`time_elapsed:datetime=...`、正しくは`timedelta`)でビルドが実際には
失敗しており、修正して再ビルドしたところ`-g 21 21 ... 37`の実行が
**40分以上経過しても完了しない**という報告を受けた。

**r1の根本原因(2つ、いずれもオラクル比較には未到達)**:

1. 型注釈ミス。`codon build -release`自体が型エラーで検出、実行にすら
   至っていない。
2. **より重大**: `crunner_run()`に渡すファイルが間違っていた。364/382の
   `.cu`が読むのは`dump_soa_reference_c_port()`(361)の10フィールド/
   40バイト参照ダンプを、さらに外部`363_filter_maxd14_only.py`で
   7フィールド/28バイトのmaxd14専用形式へ絞り込んだファイルであり、
   `ensure_constellations_bin_stream()`が返す生のstream_fnameとは
   2段階違う。誤った形式のファイルを読ませ続けていたため、
   `364_kernel_maxd14`が長時間終わらなかったと考えられる。
   **設計時にこの2段階のパイプラインを見落としていたのが根本原因**。

**誤答は出ていない**: `crunner_parse_result()`は`[gpu-run-done]`行が
一度も現れない限り`found_done=0`のままなので、この間に間違った
`total`がオラクル一致として報告されることはなかった。

**r2の修正**: `crunner_input_fname()`(期待される絞り込み済みファイル名を
`stream_fname`から導出)と`crunner_input_valid()`(存在確認+28バイト
境界チェックのみ、`file_exists()`など既存の実証済み関数を再利用)を
追加。ファイルが無ければ`[crunner-input-missing]`で即座に失敗する形
にし、**外部フィルタスクリプトの自動実行はこの修正パッチには含めない**
(実機未検証の新規経路を緊急修正に重ねて追加するリスクを避けるため)。
`385_validate.sh`側にも、実行前にこのファイルの存在を確認するチェックを
追加(数分かかるビルド+実行より先に、数秒で判定できるように)。

**結果(r2)**: (鈴木さんの実機実行後、ここに追記)

**この後**: 385がPASSしたら、386(検証関数群の別ファイル分離、
`rev386_...`接頭辞)へ進む。今週のCRunner maxd拡張で新しいバイナリが
できるたびに、`crunner_dispatch_table()`へのエントリ追加だけで387
(`-g`単独の連続N出力)の対応範囲が伸びていく想定。
