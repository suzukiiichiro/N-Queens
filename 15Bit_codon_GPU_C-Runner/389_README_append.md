## 389: CRunner入力ファイルの自動構築 + `.cu`のリネーム継続

**位置づけ**: N=23のmaxd確認に進む前に、鈴木さんから2点いただいた
ご指摘への対応。

1. 「N21/N22のbinがない場合の処理を作っておいたほうが良いのでは」
2. (直前のやり取りから継続)「.cuファイルは...明示的に同一Rev番号を
   振ってRev毎に完結した管理にしたい」

**1点目: `ensure_crunner_input_bin()`の新設**

385がr2で意図的に手動のままにしていた2段パイプラインを、実際に実行
するようにした:

1. `dump_soa_reference_c_port()`(361・Codonネイティブ、既存)で
   10フィールド/40バイトの参照ダンプを構築(既にあればスキップ)
2. 外部`363_filter_maxd14_only.py`(os.system、384で実証済みの仕組み)
   で7フィールド/28バイトのmaxd14専用形式へ絞り込み

`bench_mode==37`のディスパッチは、まずこの自動構築を試み、それでも
失敗した場合のみ(=外部フィルタスクリプト自体が見つからない等)
`[crunner-input-missing]`を報告する。**これでbare `-g`は自動的に
N=22まで到達するようになるはず**(N=22の生SoAダンプは既に366等で
生成済みのはずなので、今回は主に絞り込み段階の初回実行になる)。

**2点目: `.cu`ファイルのリネーム継続**

`388_kernel_maxd14.cu`を`389_kernel_maxd14.cu`へ(内容は無変更の
純粋なリネーム)、`crunner_dispatch_table()`も`./389_kernel_maxd14`
へ更新。コード領域はdiffでbyte-identical確認済み(sha256は364/388
以来一貫して同一)。**これは今後、鈴木さんの標準方針として毎リビジョン
続ける想定**(388だけの一回限りの対応ではなく)。

**まだ実機未確認**: `389_kernel_maxd14.cu`のこのファイル名での
nvccビルド自体、および`ensure_crunner_input_bin()`のN=22での
実地動作(初回の大規模ファイル生成を含む)。

**事前予測**: N=22を`bench_mode=37`で実行すると、`[crunner-input-
build]`のログ(soa_ref構築→フィルタ実行)が出た後、
`total=2691008701644`(オラクル一致)・`ok`になる、と予測する。
初回のみ、通常のN=21実行(約3〜4分)より長くかかる(N=22は約
2,870万レコード)、と予測する。実行完了後は`constellations_N22_7.bin.
soa_ref_361.bin.maxd14only_363.bin`が生成され、以後のbare `-g`実行
ではこの段階がスキップされ、N=22も高速に完走するはず。

**結果**: 2026-09-07、cudacodon実機で`bash 389_validate.sh`実行、
`OK=15 FAIL=0`で`389 PASSED`。

- **N=21再構築の証明**: 既存ファイルを退避→`[crunner-input-build]`
  でsoa_ref構築→フィルタ実行→`total=314666222712`(オラクル一致)、
  `0:03:47.644`。**再構築後のファイルは退避しておいた元ファイルと
  チェックサムが完全一致**(`n21_rebuilt_file_byte_identical_to_
  original`)——`ensure_crunner_input_bin()`がN=22専用ではなく本当に
  汎用であることが実機でも確認できた。
- **N=22の初回自動構築**(本命): `[filter-done] records_in=28719035
  records_kept(depth<=14)=28719035 records_dropped=0`——2,871万9,035
  レコード全件がmaxd=14に収まることも同時に確認。
  `total=2691008701644`(オラクル完全一致)、`0:34:27.132`
  (見積もり通り約30分)。
- 完了後、`constellations_N22_7.bin.soa_ref_361.bin.maxd14only_
  363.bin`が生成され永続化。以後のbare `-g`実行ではこの段階が
  スキップされ、N=22も高速に完走するはず。

**確定**: N21/N22とも、CRunner入力ファイルの自動構築が実機で
end-to-endに機能することを確認した。

---

## 参考: N=23のmaxd確認結果(次のステップへの引き継ぎ)

389完了後、`./389Py_kernel_maxd14_final -g 23 23 32 484 1 0 7 34`
(`bench_mode=34`、読み取り専用診断)を実行、以下の結果を得た:

```
[stream-build-summary] N=23 preset_queens=7 sc=18410 records=44271796 bin=constellations_N23_7.bin
[maxd-check] N=23 records=44271796 required_maxd=15 selected_maxd=16 schedule_words=4 stack_bytes_per_thread=272 supported=yes has_c_port=no(codon-only)
```

**N=23は`required_maxd=15`(`selected_maxd=16`)——maxd14では収まらない
ことが確定した。** `has_c_port=no(codon-only)`の通り、Codon側の
`kernel_dfs_iter_gpu_maxd16`は既に存在するが、Cポート(364/388/389の
`maxd14`版に相当するもの)はまだ無い。N=23対応には、maxd16のCUDA C
ポートが必要になる——これがまさに今週の本題そのものである。
