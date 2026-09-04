## 375 — `-lineinfo`/ncuによるper-line SourceCounters計測(計測専用、コード変更ゼロ)

362→363、376の設計→実装パターンと同様、375自体は計測ハーネス
(`375_lineinfo_ncu_probe_N18.sh`)の新設のみ。`374Py_kernel_maxd14.cu`
の実行コード領域はsha256照合(`cu_file_identical_to_374`)により
一切変更していないことを毎回確認してから計測を行う構成とした。

### 当初計画の誤りと訂正(r1→r2)

当初は355/357の「N=18トリック」を踏襲し、374Pyの`bench_mode=32`で
N=18のSoA参照ダンプを生成する計画だった。しかし実機実行の結果、
`bench_mode=32`は370Pyのディスパッチで**`N>=21`のガード付き**である
ことが判明した(`if use_gpu and N>=21 and bench_mode==32:`)。この
ストリーミングbinパイプライン(`ensure_constellations_bin_stream`
以下)はN=21実データ専用に作られたものであり、N=18は
`gen_constellations()`経由の別の実行パスに落ち、通常のソルバー実行
(`18: 666090624 ... ok`)が走るだけでダンプは生成されなかった。

r2で方針変更: 374で実機確認済みの
`constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin`から、
**先頭15,488レコード(=1グリッド分、grid-strideループが1周もしない
最小構成)をddで切り出す**方式に変更した。Nは21で固定(データの出自が
N=21のため、`board_mask`/`n3`/`n4`を21で計算する必要がある)。
Codonディスパッチを一切経由しないため、この切り出し自体に正当性
リスクはない。352/353の「レジスタ数・占有率はコンパイル時属性で
N非依存」という知見に基づけば、"N=18トリック"の本質は「小さい
レコード数で高速に回すこと」であり、N=18という数字そのものではない
ため、この代替でも計測目的は損なわれない。

### 実行環境の追加整備

- g5.2xlarge移行時にNsight Compute(`ncu`)本体のインストールが漏れて
  いたことが判明(`cuda-toolkit-13-0`の最小構成インストールがncuを
  含まなかった)。`sudo dnf install -y cuda-nsight-compute-13-0`で
  追加インストール。
- `sudo ncu`が`secure_path`の制約で`ncu`を見つけられない問題が発生。
  `.sh`側を`command -v ncu`でフルパスを解決してから`sudo`にそのまま
  渡す方式に修正(`NCU="${NCU:-$(command -v ncu 2>/dev/null)}"`)。
  Suzukiさん側でも`/usr/local/cuda/bin/ncu`を`/usr/bin/ncu`に
  シンボリックリンクする対処を並行して実施。

### 結果(確定)

`sudo ncu --section SourceCounters --page source`が実機で成功、
`.ncu-rep`を生成(`OK=9 FAIL=0`)。コンソールログ(494行)は列崩れが
激しく人間が直接読むのは困難だったため、Pythonで各SASS命令行の
`Warp Stall Sampling (All Samples)`列を抽出・降順ソートして解析した。

**上位ホットスポット(全体3,670,004サンプル中)**:

| 順位 | 命令 | 割合 |
|---|---|---|
| 1 | `BRA` (push分岐の再収束直後) | 8.6% |
| 2 | `LOP3.LUT`(`cur_avail=saved_avail&bm`、pop直後のマスク演算) | 6.9% |
| 3 | `BRA`(pop側の再収束直後、ループ先頭へ) | 4.4% |
| 4 | `BSYNC B5`(push分岐の再収束点) | 3.6% |

上位4命令だけで全体の23.5%を占める。

`cuobjdump -sass`(注: `-lineinfo`はcuobjdumpのオプションではなく
nvccのビルドオプション。誤って`cuobjdump -lineinfo -sass`を実行し
`Unknown option`で失敗する一幕があった)で得た静的SASSダンプと、
ncuレポートの実行時アドレスとの間に単純なオフセット対応
(`offset = ncu_addr - 0x7f9433288300`、先頭命令のアドレス一致で
確認)があることを利用し、上位命令をソースの制御構造にマッピング
した。結果、**ホットスポットは`process_one_task`内の明示スタックの
push/pop分岐そのものではなく、push/pop分岐が終わった直後にwarp全体
32レーンが足並みを揃えるための`BSYNC`/`BRA`再収束点に集中**して
いることが判明した。

### 意義

grid-strideが1回も回らない(=タスク割り当て方式が一切関与しない)
条件下でもこの偏りが観測されたことから、**355/357のCodon側診断
(Divergent Branches=0、Avg Threads Executed 2〜3/32、warp occupancy
collapse/tail effect)がCUDA C版でも同一箇所で再現することが実測で
確定**した。また、この事実は376の設計判断([タスク単位のatomic
動的スケジューリングでは解決しない])を直接に裏付ける根拠となった。

### 次のステップ

376(フレーム単位persistent kernel + atomicワークキュー設計)、
377a(キュー機構のみ導入、ロジック無変更)へ。
