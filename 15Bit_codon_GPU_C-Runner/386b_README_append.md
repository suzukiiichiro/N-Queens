## 386b: broadmark_tail_variantのCLI上書き経路を削除

**位置づけ**: 386aに続く2つ目の小さな削除ステップ。

**削除したもの**:

- `bench_mode==31`のargv解析内、旧`argv[15]`で`broadmark_tail_variant`
  を上書きしていた1行(`if argc>=16: broadmark_tail_variant=int(sys.
  argv[15])`)。
- `bench_mode==30`のargv解析内、旧`argv[16]`で同様に上書きしていた
  1行。

**触れていないもの**:

- `broadmark_tail_variant`変数自体、モジュールレベルのデフォルト
  (`BROAD_MARKDIST_TAIL_VARIANT=2`)、bare `-g`時のデフォルト
  (`A10G_FINAL_DEFAULT_BROADMARK_VARIANT=2`)——すべて無変更。variantは
  今後、常にこの採用済みの2に固定される。
- **後続の位置引数の並び**。`bench_mode==31`の`chunkshape148_bucket_
  run`(`argv[16]`)・`chunkshape148_iter_sort`(`argv[17]`)は詰め直して
  いない。空いた`argv[15]`は単に読まれなくなるだけなので、この2つを
  使う既存のコマンド列があってもそのまま動く設計にした。

**事前予測**: `argv[15]`にどんな値(例えば0)を渡しても、ログの
`broadmarktail_params: ... variant=2 ...`行が変わらず`variant=2`の
ままになる、と予測する(渡した値が読まれなくなったことの直接証拠)。
`chunkshape148_params: ... bucket_run=2048`は影響を受けず、`bench_
mode==31`のN=21実行は`total=314666222712`を変わらず返す、と予測する。

**結果**: 2026-09-07、cudacodon実機で`bash 386b_validate.sh`実行、
`OK=9 FAIL=0`で`386b PASSED`。

- `argv[15]=0`を渡しても`broadmarktail_params: ... variant=2 ...`の
  ままであることを確認(CLI上書き経路が本当に消えたことの直接証拠)。
- `chunkshape148_params: ... bucket_run=2048`が`argv[16]`から正しく
  届いていることを確認(位置引数のズレなし)。
- N=21で`total=314666222712`(オラクル完全一致)、`0:05:02.910`
  (reorder bin・chunkshape148 binとも386aでのビルド結果を再利用、
  `[split291-base-reuse]`/`[chunkshape148-reuse]`でキャッシュヒット)。

**確定**: broadmark_tail_variantのCLI上書き経路削除は実機で無害と
確認された。386の棚卸し表のうち削除確認済みだった2項目(bench_mode=
28/29、broadmark_tail_variant CLI上書き)は386a/386bで完了。

**ドキュメントのみの追記**: 386aの.pyヘッダーdocstringに386a自身の
per-session narrativeパラグラフを追加し忘れていたことに気づき
(README側には書いていたが.py側は「最終更新」行とVERSION_TAGのみ更新
していた)、386a・386bの両方にまとめて追記した。実行コードには影響
しない。
