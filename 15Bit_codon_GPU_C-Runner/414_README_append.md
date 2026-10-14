## 414 — N=24 に向けた入力構築の下ごしらえ：生の bin の全件 Dict 読みをチャンク走査に（`.py` のみ、GPU ゼロ）

413 の結果と N=23 本番値は `413_results_README_append.md`、README 冒頭は `413_README_head_20261009.md`。
本番は 413（N=21 97,946／N=22 849,096／**N=23 7,717,284 ms**）のまま。このリビジョンは値を主張しない。

### 位置づけ

suzuki の決定（2026-10-09）で次の軸は (a) N=24 → (b) N=23 の時間短縮。N=24 の入力 `constellations_N24_*` は手元にない
（N=23 の `constellations_N23_7.bin` は 9/7 に 389 の `-g 23 23 32 484 1 0 7 34`（bench_mode=34）が作ったもの。
366 期の「OOM の前歴」は旧世代の生成器と一時ディスクの話で、今の生成器では起きていない）。N=24 の入力構築に入る前に、
構築経路で**メモリが件数に比例して膨らむ箇所**を潰しておく。

### 生成経路の読み（GPU ゼロ、`413Py` のソースから）

| 工程 | 関数 | メモリの形 | N=24 |
|---|---|---|---|
| 生成 | `gen_constellations_stream_to_bin` | 開始配置（sc）1 個ずつ生成して追記。常駐は 1 sc 分の `visited`／署名集合 | 安全。preset_queens は 22〜24 で 7（`select_dynamic_preset_queens`） |
| 361 ダンプ | `dump_soa_reference_c_port` | **全件を `List[Dict]` に読み**＋TaskSoA 11 配列×m | **危険**。N=23 の 44M 件は 30 GB で通った（1.5 分）が Dict 1 件は数百 B |
| required_maxd | `check_required_maxd_for_N`（bench_mode=34、`-g` 経路の配車前） | 同じ全件 Dict 読み | 同上 |
| フィルタ | `413_filter_maxd16_only.py` | `array` 1 本（40 B×m）＋メモ化辞書 | 安全。深さヒストグラムが `required_maxd` を与える |
| sched | `394f_permute_soa7.py` | 28 B×m のソート | N=23 で数 GB。N=24 の RSS は N24d で測る |

`read_constellations_bin_range(fname, off, max)` は 365 以降オフセット付きで存在するので、チャンク化は 2 関数の中だけで済む。
365 の single-shot と 368 の maxd 診断にも全件読みが残るが、N=24 の経路（bench_mode 32／34／37）ではないので触らない。

### 変更（1 点）

`dump_soa_reference_c_port()` と `check_required_maxd_for_N()` が生の bin を **`STREAM_SCAN_CHUNK=2,097,152` 件ずつ**読み、
チャンクごとに同じ `build_soa_for_range()` を自分の `TaskSoA` に掛けて順に書く／深さの最大を取る。出力バイト列・u64 チェックサム・
`required_maxd` は 361／366 と同一（ゲート D21〜D23・M21）。`gpu_log_level>=1` で `[soa-ref-dump-progress]` を出す（新規行）。
`.py` の実行行は 413Py 比 **removed=34 added=54**（コメント行を除く）。ほかは VERSION_TAG／REV_TAG／テーブル行のパスの 3 行。
`414_kernel_maxd16.cu` は 413 のリネーム（コード領域 sha `06eaa8f0` 一致、ヘッダに rev414 ノート）。

### ゲート（`414_validate.sh`、GPU ゼロ）

| | 内容 |
|---|---|
| S0 | 静的 OK=16：py 指紋 34/54、REV_TAG、VERSION_TAG、ノート、テーブル行、`STREAM_SCAN_CHUNK`、2 関数に全件読みが残っていない、`.cu` コード領域＝413、ヘッダノート |
| S1 | `nvcc -O3 -arch=sm_86` でビルド、`cuobjdump -sass`（アドレス・エンコーディング除去）が既存 `./413_kernel_maxd16` と**完全一致**（ハード、403-r2b C0：nvcc は決定的）。できたものが `./414_kernel_maxd16` |
| B | `codon build -release` で `414Py_kernel_maxd16_final`（なければ） |
| D21／D22／D23 | `./414Py -g N N 32 960 1 0 <preset> 32` で `*.soa_ref_361.bin` を作り直し、**sha256 が既存ファイルと一致**（ハード。既存は先に退避、不一致なら戻す）。各段の時間と最大 RSS（`/usr/bin/time -v`）を INFO |
| R23 | 413Py（チャンク化前）で同じ N=23 ダンプ（1.5 分、30 GB で通った実績あり）。RSS の比較用（情報、`SKIP_R23=1` で省略） |
| M21 | `-g 21 21 32 960 1 0 6 34` → `[maxd-check] required_maxd=14`（ハード） |
| N24a | **N=24 生成＋ダンプ**（`-g 24 24 32 960 1 0 7 32`）。`systemd-run --user --scope -p MemoryMax=26G` の下（使えなければ上限なしで INFO）。行ごとに時刻を付けて生成終了（`stream-build-summary`）とダンプ終了を分ける。rc=0・`.done`＝件数（ハード） |
| N24b | 同じコマンドをもう 1 回（生成はキャッシュヒット）＝**ダンプ単独の RSS**。sha が N24a と一致（決定性、ハード） |
| N24c | `MAXDEPTH=21` で 413 フィルタ → `*.all_414.bin`（全件保持がハード）、**`max_depth`＝N=24 の `required_maxd`**、ヒストグラム |
| N24d | `394f_permute_soa7.py sched` → `*.all_414.bin.sched394f.bin`、件数一致（ハード）、時間・RSS |

N=24 の段は既定で走る（`SKIP_N24=1` で省略）。ディスク空き 15 GB をゲート（見込み：bin 1.0〜1.8 GB、ダンプ 2.6〜4.4 GB、
SoA7×2 で 3.6〜6 GB）。`ulimit -v` は使わない（413 の教訓 26：CUDA 初期化で Segfault）。Codon バイナリは CUDA を初期化するので
cgroup（RSS 側）で守る。

### 事前登録（実測前に固定、方向を表明）

| | 表明 | 外れたとき |
|---|---|---|
| P1 | D23 の最大 RSS ≤ R23 の 1/4 | チャンク化が効いていない（Codon の Dict 解放の問題）。`STREAM_SCAN_CHUNK` を小さくして再測 |
| P2 | N=24 件数 65〜110M（N=22→23 の ×1.54 を基準） | 情報。ディスクと時間の見積もりを直す |
| P3 | N24b（ダンプ単独）の最大 RSS ≤ 8 GB | 同 P1 |
| P4 | N=24 の `required_maxd`＝16（21／22 が 14、23 が 15） | **17 以上なら N=24 カーネルは祖先 16 段以上**（16 B フレームで 256 B）。設計を改める |
| — | 生成時間は未知。記録のみ | — |

採用則：ハード全通過で 414 が本番 `.py`（値の主張なし、本番値は 413 のまま）。D21〜D23 のどれかが不一致なら不採用、413Py のまま。

### N=24 に向けた静的読み（このリビジョンでは測らない）

- 12 B 配置：col 22＋avail 22＋ld 23＋rd 30＝**97 ＞ 96**。rd の上限 30 の導出は N 依存（`bit 2N−3−j−start ≤ 30`）で N=24 では
  再計算が要り、31 に悪化する可能性もある。死にビットを 1 本見つけるか、16 B フレーム（`A = ld | col<<32`、`B = rd | avail<<32`、
  各 32 ビット余裕）で祖先 15 段 240 B か。413 の教訓 28（L1 をはみ出しても 12 ブロック/SM）が 240 B×12＝92 KB で成り立つかは
  分からないので MB は 640／800／960 を掃引する。
- CPU 等価の参照は 392（maxd16、391 で Codon GPU と一致済み）。`required_maxd=16` なら 392 の範囲内。
- 時間：解の数の比 227.5T／24.23T＝9.39。N=22→23 は解の比 9.01 に対し時間比 9.09 だったので、同じ単価なら 7,717 s×9.4≈72,500 s
  ＝**20 時間**、240 B フレームの費用が乗って 20〜25 時間の見込み（夜間 2 晩またぎ、`timeout` 36 h、電力上限で 1680〜1695 MHz）。

### 所要と実行

```
STATIC_ONLY=1 bash 414_validate.sh      # OK=16、GPU ゼロ、数秒
SKIP_N24=1    bash 414_validate.sh      # ＋ビルド・D21〜D23・R23・M21、約 10 分
              bash 414_validate.sh 2>&1 | tee 414_validate_stdout.log   # ＋N=24 構築、生成時間未知（30〜90 分の見込み）
```

**事前条件**：`413Py_kernel_maxd16_final.py`／`413Py_kernel_maxd16_final`／`413_kernel_maxd16.cu`／`413_kernel_maxd16`、
`413_filter_maxd16_only.py`、`394f_permute_soa7.py`、`rev386_validation_helpers.py`、`414_strip_c_comments.py`（同梱）、
`constellations_N21_6.bin`／`N22_7`／`N23_7`（＋`.done`）とそれぞれの `*.soa_ref_361.bin`（N=21／22 のダンプが残っていなければ
その N は作成のみで一致ゲートは出ない）、nvcc／cuobjdump／codon、ディスク空き 15 GB 以上、`systemd-run --user` が使えること（任意）。

---
