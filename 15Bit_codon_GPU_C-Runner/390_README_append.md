## 390: maxd16カーネルの新規実装

**位置づけ**: `390_maxd16_kernel_port_spec.md`(設計文書、日本語版)に
基づく実装。設計文書のレビューをいただいた後、着手。

**やったこと**:

- ファイル内に現存していた`kernel_dfs_iter_gpu_maxd16`(maxd14作成時に
  同一内容を別名でコピーしただけの、以後一度も更新されていない古い
  世代の実装)を削除し、**現行の最適化済みmaxd14カーネルから機械的に
  導出した新しい実装**に置き換えた。
- **スクリプトで確認済み**: 新しいmaxd16カーネルの本体は、関数名と
  定数1つ(`MAXD14_ANCESTOR`13→`MAXD16_ANCESTOR`15)以外、maxd14の
  現行コードと**byte-identical**。設計文書の予測通り、
  `schedule_lo`/`schedule_hi`のニブル詰め・`child_jmark_mask`・
  `terminal_parent_depth`/`terminal_is_base14`・11個のマスク/オペコード
  定数は無改造で済んだ。
- `MAXD16_ANCESTOR:Static[int]=15`を新設。
- `packed_schedule_words_for_maxd(16)`(4→0)・`packed_stack_bytes_per_
  thread(16)`(272→240)を設計文書通り訂正(表示専用と確認済みなので
  無リスク)。
- **実装中に見つけた見落とし**: `launch_kernel_dfs_iter_gpu_static_
  maxd()`の`selected_maxd==16`呼び出し箇所が`kbatch_stride`引数を
  渡していなかった(旧カーネルにはgrid-strideループが無くこの引数を
  取らなかったため)。新カーネルの新しいシグネチャに合わせて修正。
  これはビルド失敗で気づいたのではなく、実装後にディスパッチャを
  読み返して発見したもの——このリビジョンはまだ一度も実機Codonビルドを
  通していないため、コンパイルエラーとして検出される保証はなかった。

**まだ実機未確認**: このリビジョンのコードは、静的な分析とdiffによる
証明のみで、**一度もビルド・実行していない**。

**事前予測**: `bench_mode=33`(`exec_solutions_gpu_single_shot()`、既に
maxd汎用設計、新規配線不要)をN=23の実データに対して実行すると、
`[single-shot-maxd-dispatch]`で`selected_maxd=16`と表示され、
`total=24233937684440`(N=23の公表オラクル、`expected[]`配列index23)
になる、と予測する。もし合計値が違えば、設計文書2節の「depth非依存」
という核心的主張の再検証が必要になる(単純な打ち間違いと決めつけない)。

**結果**: (鈴木さんの実機実行後、ここに追記)
