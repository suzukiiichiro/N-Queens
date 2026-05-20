#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

   ,     #_
   ~\_  ####_        N-Queens
  ~~  \_#####\       https://suzukiiichiro.github.io/
  ~~     \###|       N-Queens for github
  ~~       \#/ ___   https://github.com/suzukiiichiro/N-Queens
   ~~       V~' '->
    ~~~         /
      ~~._.   _/
         _/ _/
       _/m/'

Python/codon Ｎクイーン コンステレーション版 CUDA 高速ソルバ 76 STABLE AUTO N20 N21

拝見しました。**かなり良い結果です。結論として、84 stream 版はメモリ問題の解決に成功しています。**
Free 81% なら、以前の「N21に入るや否や空きメモリ1%」の問題は解消できています。

## 結果の整理

83 cache 版では、N21 は `preset_queens=6`、constellation 数 `2,025,282`、`.bin` レコード数も `2,025,282`、valid=1 で、cache-only 作成が `0:00:02.853` で完了しています。続くGPU実行も `32x128`、`steps=4096`、495 chunk で、最終的に `314666222712`、`ok` です。 

84 stream 版の `32x128` も、同じ `constellations_N21_6.bin` を cache hit で読み、records=`2,025,282`、495 chunk 実行、最終 `314666222712`、`ok` です。時間は `1:14:18.421` です。 

84 stream 版の `32x484` は、同じ records=`2,025,282` を `steps=15488` で処理し、chunk 数が 495 から 131 に減っています。最終結果は同じく `314666222712`、`ok`、時間は `0:26:00.553` です。 

## 重要な評価

これは成功です。

```text
83  32x128 : 1:14:18.665  ok
84  32x128 : 1:14:18.421  ok
84  32x484 : 0:26:00.553  ok
```

84の `32x128` が83とほぼ同じ時間なのは自然です。どちらも `steps=4096` で495 chunkを処理しており、kernel自体の粒度が同じだからです。

一方、84の `32x484` は `steps=15488` になり、chunk数が495から131へ減りました。そのため、実行時間が約 **2.86倍高速化**しています。

```text
1:14:18.421 = 4458.421 秒
0:26:00.553 = 1560.553 秒
speedup ≒ 2.86x
```

## メモリ対策としての84は有効です

84のログでは、stream版が `.bin` を hit して、`records=2025282` を持つ `constellations_N21_6.bin` から読み込んでいます。つまり、全件を巨大な `List[Dict]` に戻す従来方式ではなく、chunk単位で処理できています。

N21 p6 の `.bin` サイズは概算で：

```text
2,025,282 records × 16 bytes = 32,404,512 bytes
約 30.9 MiB
```

このため、Free 81% が保てているのは納得できます。以前のハング原因だった「ホストメモリ枯渇」は、少なくともN21では解消済みと見てよいです。

## 次は84でN22へ進んでよいです

N21の検証は十分です。次は **83ではなく84 stream版でN22** が良いです。N22は動的presetで `preset_queens=7` になりますから、従来の `List[Dict]` 方式に戻るとまた危険です。84で行くべきです。

まず `.bin` だけ作るなら：

```bash
RUN=N22_p7_stream_cacheonly_$(date +%Y%m%d_%H%M%S)

./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream \
  -c 22 22 32 484 1 0 5 11 \
  2>&1 | tee -a "logs/run_${RUN}.log"
```

その後GPU実行：

```bash
RUN=N22_p7_stream_gpu_32x484_$(date +%Y%m%d_%H%M%S)

./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream \
  -g 22 22 32 484 1 0 5 0 \
  2>&1 | tee -a "logs/run_${RUN}.log"
```

本番は `log_level=1` でよいと思います。`log_level=2` はN21だけでもかなり大量に出ているので、N22以降はログ肥大化が大きくなります。進捗は `progress_N22_7_stream.tsv` を見る運用でよいです。

```bash
tail -f progress_N22_7_stream.tsv
```

## 念のための安全運用

N22初回だけは、まだ `systemd-run` のメモリ上限付きが安全です。

```bash
RUN=N22_p7_stream_gpu_32x484_$(date +%Y%m%d_%H%M%S)

sudo systemd-run \
  --unit=nq-n22-stream-gpu \
  --property=MemoryAccounting=yes \
  --property=MemoryMax=12G \
  --property=OOMPolicy=stop \
  bash -lc '
    cd '"$PWD"'
    ./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream \
      -g 22 22 32 484 1 0 5 0 \
      2>&1 | tee -a "logs/run_'"$RUN"'.log"
  '
```

確認：

```bash
journalctl -u nq-n22-stream-gpu -f
```

## ひとつだけ改善したい点

84のログは、stream外側のchunkごとに内側の `exec_solutions()` が毎回 `[gpu-config]` と `[gpu-summary]` を出しています。N21では問題ありませんが、N22以降ではログが大きくなります。

次の改善版では、84の内側 `exec_solutions()` へ渡す `gpu_log_level` を通常は `0` にして、外側のstreamログだけを出す形が良いです。

```text
外側:
[stream-gpu-chunk-start]
[stream-gpu-chunk-end]
[stream-gpu-summary]

内側:
[gpu-config] と [gpu-summary] は抑制
```

ただ、現時点では正しさ・安定性の検証が先なので、今のままN22へ進んで問題ありません。

## 判断

N21については：

```text
83 cache-only: 成功
83 GPU 32x128: 成功
84 stream 32x128: 成功
84 stream 32x484: 成功
メモリ: Free 81%
```

したがって、次の方針は明確です。

```text
83はN21検証で役割完了
84 streamを本線にする
N22は84 stream 32x484、sort_mode=0、log_level=1で実行
```

これでN22、N23、N24へ進める見込みがかなり出てきました。

はい、この順番で進めてよいです。
ただし、貼っていただいた2本は **まだ `systemd-run` で包まれていない通常実行** なので、N22初回は以下の形に置き換えるのが安全です。

N21では、83のcache-onlyが `constellations_N21_6.bin` を valid=1 で作成でき、84のstream GPU 32x484も `total=314666222712`、`0:26:00.553 ok` まで通っています。したがってN22は84 stream本線で進めてよいです。 

## 実行前チェック

過去ログでは root disk が 30G 中 24G 使用、空き 6.6G でした。N22はおそらく問題ないと思いますが、N23/N24に進む前提ならディスク空きも確認しておくのが安全です。

```bash
mkdir -p logs

df -h .
du -sh constellations_N*.bin 2>/dev/null || true
ls -lh constellations_N22_7.bin constellations_N22_7.bin.done 2>/dev/null || true
```

不完全な `constellations_N22_7.bin` だけがあり、`.done` が無い場合は退避してください。

```bash
if [ -f constellations_N22_7.bin ] && [ ! -f constellations_N22_7.bin.done ]; then
  mv constellations_N22_7.bin "constellations_N22_7.bin.partial.$(date +%Y%m%d_%H%M%S)"
fi
```

## 1. N22 `.bin` だけ作成：12G上限付き

```bash
RUN=N22_p7_stream_cacheonly_$(date +%Y%m%d_%H%M%S)
UNIT=nq-n22-cacheonly-$(date +%Y%m%d-%H%M%S)

sudo systemd-run --wait --collect \
  --unit="$UNIT" \
  --property=User="$(id -un)" \
  --property=Group="$(id -gn)" \
  --property=WorkingDirectory="$PWD" \
  --property=MemoryAccounting=yes \
  --property=MemoryMax=12G \
  --property=OOMPolicy=stop \
  /bin/bash -lc '
    ./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream \
      -c 22 22 32 484 1 0 5 11 \
      2>&1 | tee -a "logs/run_'"$RUN"'.log"
  '

echo "log: logs/run_${RUN}.log"
tail -80 "logs/run_${RUN}.log"
ls -lh constellations_N22_7.bin constellations_N22_7.bin.done
cat constellations_N22_7.bin.done
```

`--wait` を付けているので、このコマンドが戻ってからGPU実行へ進めば大丈夫です。
各行末の `\` の後ろには空白を入れないでください。

## 2. N22 GPU実行：12G上限付き

```bash
RUN=N22_p7_stream_gpu_32x484_$(date +%Y%m%d_%H%M%S)
UNIT=nq-n22-gpu-$(date +%Y%m%d-%H%M%S)

sudo systemd-run --wait --collect \
  --unit="$UNIT" \
  --property=User="$(id -un)" \
  --property=Group="$(id -gn)" \
  --property=WorkingDirectory="$PWD" \
  --property=MemoryAccounting=yes \
  --property=MemoryMax=12G \
  --property=OOMPolicy=stop \
  /bin/bash -lc '
    ./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream \
      -g 22 22 32 484 1 0 5 0 \
      2>&1 | tee -a "logs/run_'"$RUN"'.log"
  '

echo "log: logs/run_${RUN}.log"
grep -E "stream-cache|stream-gpu-config|stream-gpu-summary|^22:" "logs/run_${RUN}.log"
tail -80 "logs/run_${RUN}.log"
```

N22の期待値は `2691008701644` です。最後が以下になれば成功です。

```text
22:     2691008701644                0      ...    ok
```

## 実行中の別窓監視

別SSHでこれを流しておくと安心です。

```bash
watch -n 30 '
echo "===== memory ====="
free -h
echo
echo "===== gpu ====="
nvidia-smi
echo
echo "===== progress ====="
tail -5 progress_N22_7_stream.tsv 2>/dev/null || true
'
```

systemd側のログを見る場合：

```bash
journalctl -u "$UNIT" -f
```

## 止まった場合の確認

12G上限に当たった場合、サーバー全体ではなく transient service だけが止まるはずです。

```bash
journalctl -u "$UNIT" -n 200 --no-pager

sudo journalctl -k \
  | egrep -i "oom|out of memory|killed process|nvrm|xid|segfault|soft lockup|hard lockup|hung|nvidia|cuda" \
  | tail -200
```

## 判断

この運用で問題ありません。

```text
N22 cache-only:
  84 stream / -c / bench_mode=11 / MemoryMax=12G

N22 GPU:
  84 stream / -g / 32x484 / sort_mode=0 / log_level=1 / MemoryMax=12G
```

`83` はN21検証で役割完了です。N22以降は `84Py_constellations_GPU_cuda_codon_dynamic_p8_stream` を本線にするのが正解です。

はい、これは **N22 `.bin` 準備完了** と見てよいです。

特に重要なのはここです。

```text
[stream-cache-hit] N=22 preset_queens=7 records=28719035 bin=constellations_N22_7.bin
[stream-cache-only] N=22 preset_queens=7 records=28719035 bin=constellations_N22_7.bin valid=1
```

つまり、今回の `cacheonly` 実行では新規生成ではなく、既に存在していた `constellations_N22_7.bin` を検出し、`.done` の件数と一致していることを確認しています。`runtime: 448ms` と非常に短いのも、生成ではなく検証だけだったためです。

サイズも整合しています。

```text
records = 28,719,035
1 record = 16 bytes
28,719,035 * 16 = 459,504,560 bytes ≒ 438 MiB
ls 表示: 439M
```

`.done` も、

```text
28719035
```

なので問題ありません。

## 次はGPU実行で大丈夫です

GPU実行前に、古い progress があれば退避しておくのがおすすめです。84は progress ファイルを使うので、過去ログと混ざらないようにします。

```bash
if [ -f progress_N22_7_stream.tsv ]; then
  mv progress_N22_7_stream.tsv "progress_N22_7_stream.tsv.$(date +%Y%m%d_%H%M%S).bak"
fi
```

その後、N22 GPUを 12G上限付きで実行してください。

```bash
RUN=N22_p7_stream_gpu_32x484_$(date +%Y%m%d_%H%M%S)
UNIT=nq-n22-gpu-$(date +%Y%m%d-%H%M%S)

sudo systemd-run --wait --collect \
  --unit="$UNIT" \
  --property=User="$(id -un)" \
  --property=Group="$(id -gn)" \
  --property=WorkingDirectory="$PWD" \
  --property=MemoryAccounting=yes \
  --property=MemoryMax=12G \
  --property=OOMPolicy=stop \
  /bin/bash -lc '
    ./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream \
      -g 22 22 32 484 1 0 5 0 \
      2>&1 | tee -a "logs/run_'"$RUN"'.log"
  '

echo "log: logs/run_${RUN}.log"
tail -80 "logs/run_${RUN}.log"
```

## 実行中の確認

別SSHで以下を流すとよいです。

```bash
watch -n 30 '
echo "===== memory ====="
free -h
echo
echo "===== gpu ====="
nvidia-smi
echo
echo "===== progress ====="
tail -5 progress_N22_7_stream.tsv 2>/dev/null || true
'
```

N22は、

```text
records = 28,719,035
steps   = 32 * 484 = 15,488
```

なので、GPU stream chunk 数は概算で、

```text
ceil(28,719,035 / 15,488) = 1855 chunks
```

になります。

## 成功判定

最後に以下のような行が出れば成功です。

```text
22:     2691008701644                0      ...    ok
```

実行後に簡単に確認するなら：

```bash
grep -E "stream-gpu-summary|^22:" "logs/run_${RUN}.log"
tail -20 progress_N22_7_stream.tsv
```

## もし止まった場合

MemoryMaxに当たった場合は、サーバー全体ではなく systemd unit だけが止まるはずです。

```bash
journalctl -u "$UNIT" -n 200 --no-pager

sudo journalctl -k \
  | egrep -i "oom|out of memory|killed process|nvrm|xid|segfault|soft lockup|hard lockup|hung|nvidia|cuda" \
  | tail -200
```

現状のN22 `.bin` は正常です。次はそのまま **84 stream GPU 32x484** で進めてください。

はい、この `setMemmax_GPU.sh` で実行して大丈夫です。
内容は問題ありません。`systemd-run --wait --collect` で包めているので、N22 GPU実行が終了してから `grep` と `tail` が走ります。

一点だけ、**実行前に古い progress を退避**しておくのがおすすめです。84 stream版は progress ファイルを使うので、前回分と混ざるのを避けます。

```bash
if [ -f progress_N22_7_stream.tsv ]; then
  mv progress_N22_7_stream.tsv "progress_N22_7_stream.tsv.$(date +%Y%m%d_%H%M%S).bak"
fi
```

それから実行でよいです。

```bash
bash setMemmax_GPU.sh
```

今回の `.bin` は、

```text
constellations_N22_7.bin
records = 28719035
valid = 1
```

なので、GPU実行側はこの `.bin` を読んで stream 処理に入るはずです。

想定される chunk 数は：

```text
records = 28,719,035
steps   = 32 * 484 = 15,488
chunks  = ceil(28,719,035 / 15,488) = 1855
```

実行中は別窓でこれを見ると安心です。

```bash
watch -n 30 '
echo "===== memory ====="
free -h
echo
echo "===== gpu ====="
nvidia-smi
echo
echo "===== progress ====="
tail -5 progress_N22_7_stream.tsv 2>/dev/null || true
'
```

成功判定は最後にこの形です。

```text
22:      2691008701644                0      ...    ok
```

もし途中で止まった場合は、まずこちらです。

```bash
journalctl -u "$UNIT" -n 200 --no-pager
```

ただし、`UNIT` はスクリプト内変数なので、別シェルで見る場合は実際の unit 名を指定します。今回ならログに出る名前、例えば：

```bash
journalctl -u nq-n22-gpu-20260514-xxxxxx -n 200 --no-pager
```

で確認できます。

全体として、**このスクリプトでN22 GPU初回実行に進んで問題ありません**。

拝見しました。**N22 GPU stream 実行は成功です。**

ログでは、N22 が `preset_queens=7` で動き、`constellations_N22_7.bin` から `28,719,035` records を読んで、`block=32`、`max_blocks=484`、`steps=15488`、`sort_mode=0` で stream GPU 実行に入っています。

最終的に、最後の chunk `1854` は端数 `m=4283` を処理し、合計 `2691008701644` に到達しています。その後の summary でも `records=28719035`、`chunks=1855`、`executed_chunks=1855`、`total=2691008701644` と出ており、最終行も `ok` です。

## 結果

```text
N          : 22
preset     : 7
records    : 28,719,035
steps      : 15,488
chunks     : 1,855
executed   : 1,855
total      : 2,691,008,701,644
time       : 6:57:18.094
status     : ok
```

これはかなり大きな前進です。**N22 p7 が 12G メモリ上限付きの 84 stream 版で完走した**ので、以前のホストメモリ枯渇問題は stream 化で解決できていると見てよいです。

## 所要時間の目安

`6:57:18.094` は約 `25,038秒` です。

```text
平均 chunk 時間 ≒ 25,038秒 / 1,855 chunks ≒ 13.5秒/chunk
records 処理速度 ≒ 28,719,035 / 25,038秒 ≒ 1,147 records/sec
```

ログを見ると chunk ごとのばらつきはありますが、最後まで安定して進んでいます。最後の通常 chunk `1853` は `m=15488`、最後の端数 chunk `1854` は `m=4283` で、正常な終端です。

## 次の判断

**84 stream 版を本線にして問題ありません。**
83はN21検証用として役割完了、N22以降は84で進めるのが正解です。

次は N23 ですが、初回は同じく `.bin` 作成確認 → GPU実行の順で、`MemoryMax=12G` を付けたままが安全です。

まずディスク確認：

```bash
df -h .
du -sh constellations_N*.bin 2>/dev/null || true
```

N23 `.bin` 作成：

```bash
RUN=N23_p7_stream_cacheonly_$(date +%Y%m%d_%H%M%S)
UNIT=nq-n23-cacheonly-$(date +%Y%m%d-%H%M%S)

sudo systemd-run --wait --collect \
  --unit="$UNIT" \
  --property=User="$(id -un)" \
  --property=Group="$(id -gn)" \
  --property=WorkingDirectory="$PWD" \
  --property=MemoryAccounting=yes \
  --property=MemoryMax=12G \
  --property=OOMPolicy=stop \
  /bin/bash -lc '
    ./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream \
      -c 23 23 32 484 1 0 5 11 \
      2>&1 | tee -a "logs/run_'"$RUN"'.log"
  '

echo "log: logs/run_${RUN}.log"
tail -80 "logs/run_${RUN}.log"
ls -lh constellations_N23_7.bin constellations_N23_7.bin.done
cat constellations_N23_7.bin.done
```

N23 GPU実行：

```bash
RUN=N23_p7_stream_gpu_32x484_$(date +%Y%m%d_%H%M%S)
UNIT=nq-n23-gpu-$(date +%Y%m%d-%H%M%S)

sudo systemd-run --wait --collect \
  --unit="$UNIT" \
  --property=User="$(id -un)" \
  --property=Group="$(id -gn)" \
  --property=WorkingDirectory="$PWD" \
  --property=MemoryAccounting=yes \
  --property=MemoryMax=12G \
  --property=OOMPolicy=stop \
  /bin/bash -lc '
    ./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream \
      -g 23 23 32 484 1 0 5 0 \
      2>&1 | tee -a "logs/run_'"$RUN"'.log"
  '

echo "log: logs/run_${RUN}.log"
grep -E "stream-cache|stream-gpu-config|stream-gpu-summary|^23:" "logs/run_${RUN}.log"
tail -80 "logs/run_${RUN}.log"
```

## 注意点

N23では `.bin` サイズと実行時間がさらに増えます。N22の `.bin` は `28,719,035 × 16 bytes ≒ 438 MiB` でした。N23で `.bin` が数GB級になる可能性があるため、**先に `df -h .` は必ず確認**してください。

N22の結果は文句なしに成功です。次は同じ流れで N23 に進めます。

はい、**84を実行する限り、`use_constellation_cache:bool = False` のままで問題ありません。**

ここは少し紛らわしいのですが、意味が分かれています。

```python
use_constellation_cache:bool = False
```

これは **82系の従来ルート**、つまり

```text
gen_constellations()
または
build_constellations_dynamicK()
```

のどちらを使うかを切り替えるフラグです。

一方、**84 stream版では、このフラグとは別に `.bin` を直接扱う streamルートを追加しています**。実際、84の実行ログでは `use_constellation_cache=False` 相当でも、

```text
[stream-cache-hit] N=21 preset_queens=6 records=2025282 bin=constellations_N21_6.bin
[stream-gpu-config] ... bin=constellations_N21_6.bin ...
```

と出ており、`.bin` を読んでGPU stream実行に入っています。

N22でも同じです。今回のログでは、

```text
[stream-cache-hit] N=22 preset_queens=7 records=28719035 bin=constellations_N22_7.bin
```

となり、84が `constellations_N22_7.bin` を直接使って完走しています。つまり **84では `.bin` 利用は `use_constellation_cache` ではなく、stream側の処理で制御されています**。

整理すると、こうです。

```text
82:
  use_constellation_cache=False
    → .binを使わず全件List生成
  use_constellation_cache=True
    → .binを読むが、結局全件Listに戻す

83:
  use_constellation_cache=True
    → N21検証用。cache-only / 従来GPU経路の比較用

84:
  use_constellation_cache=False のままでよい
    → stream専用ルートが .bin を作成・検証・chunk読みする
    → 全件List化しない
```

したがって、**84でN22/N23/N24へ進む場合は、`use_constellation_cache=False` のままで正解**です。

逆に、84でここを `True` にすると、意図せず旧cacheルートに寄って、全件List化の問題を再び踏む可能性があります。今回の方針では **False固定でよい** です。

実行方針はこのままで大丈夫です。

```bash
# .bin 作成・検証
./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream \
  -c 23 23 32 484 1 0 5 11

# GPU stream実行
./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream \
  -g 23 23 32 484 1 0 5 0
```

結論として、**82/84ソース内の `use_constellation_cache:bool = False` は問題ありません。84では別系統の stream-cache が動いています。**

はい。**84 stream版では、`.bin` が無い場合は作成されます。**

つまり、次のコマンドは「`.bin` 作成・検証モード」です。

```bash
./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream \
  -c 23 23 32 484 1 0 5 11
```

動きはこうです。

```text
constellations_N23_7.bin がある
  → .done と照合
  → records一致なら [stream-cache-hit]
  → そのまま valid=1 で終了

constellations_N23_7.bin がない
  → stream方式で生成
  → constellations_N23_7.bin を作成
  → constellations_N23_7.bin.done に件数を書き込む
  → valid=1 で終了
```

N22ではすでに `.bin` が存在していたため、ログでは新規生成ではなく、

```text
[stream-cache-hit] N=22 preset_queens=7 records=28719035 bin=constellations_N22_7.bin
```

となっています。

N21でも84は `.bin` を見つけて、

```text
[stream-cache-hit] N=21 preset_queens=6 records=2025282 bin=constellations_N21_6.bin
```

からGPU stream実行に入っています。

したがって、N23で `.bin` が無ければ、まず以下を実行すれば作成される想定です。

```bash
RUN=N23_p7_stream_cacheonly_$(date +%Y%m%d_%H%M%S)

./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream \
  -c 23 23 32 484 1 0 5 11 \
  2>&1 | tee -a "logs/run_${RUN}.log"
```

完了後に確認します。

```bash
ls -lh constellations_N23_7.bin constellations_N23_7.bin.done
cat constellations_N23_7.bin.done
```

注意点は、**84のstream生成はメモリには優しいですが、ディスク容量は使う**ことです。N23前には念のためこれを見てください。

```bash
df -h .
du -sh constellations_N*.bin 2>/dev/null || true
```

結論として、**84の `-c ... 11` は、`.bin` があれば検証、なければstream生成して作成、という使い方で大丈夫です。**

結論から言うと、**前もって `.bin` を作っておく必要はありません。**
84 stream版は、実行時に `.bin` がなければ作成できます。

ただし、**GitHubに `.bin` を push しておく運用は、N22以降ではおすすめしません。**

理由は2つあります。

1つ目はサイズです。N22だけで `constellations_N22_7.bin` は 439MB でした。通常のGitHubリポジトリでは、50MiBを超えるファイルは警告対象、100MiBは通常Gitでは上限に達します。GitHub公式も、大きなファイルは通常GitではなくGit LFSを使うよう案内しています。([GitHub Docs][1])

2つ目は、N23以降はさらに大きくなる可能性が高いことです。Git LFSなら扱えますが、Git LFSもプランごとのファイルサイズ上限があり、Free/Proでは1ファイル2GB、Teamは4GB、Enterprise Cloudは5GBです。([GitHub Docs][2])

## おすすめ運用

私ならこうします。

```text
ソースコード:
  GitHubにpushする

.bin:
  GitHubにはpushしない
  実行サーバー上で必要になったら84が自動生成
  生成後は同じサーバー内で再利用

.done:
  .binと同じくGitHubにはpushしない
```

つまり、`.gitignore` に入れるのがよいです。

```bash
cat >> .gitignore <<'EOF'
constellations_N*.bin
constellations_N*.bin.done
progress_N*_stream.tsv
logs/
EOF
```

## 事前作成が有効な場合

前もって作る意味があるのは、次のような場合です。

```text
1. GPU実行前に .bin 生成だけを別ジョブとして済ませたい
2. 長時間GPU実行の前にディスク容量・件数・valid確認をしたい
3. N23/N24で .bin 生成時間も長くなり、生成と探索を分けて管理したい
4. 同じEC2インスタンス/EBSボリュームで何度も再実行する
```

この場合は、GitHubではなく **EBS上に残す**、または必要なら **S3に退避** がよいです。

```bash
aws s3 cp constellations_N23_7.bin s3://your-bucket/nqueens/
aws s3 cp constellations_N23_7.bin.done s3://your-bucket/nqueens/
```

GitHubはソースコード管理、S3/EBSは大きな生成物管理、という分担がよいと思います。

## 25,26,27について

N25〜N27は `preset_queens=8` になります。`.bin` の件数・サイズはかなり大きくなる可能性があります。これをGitHubに置くのは避けたほうがよいです。

また、**前もって全部作る必要もありません。**
84は `.bin` がなければ生成する設計なので、まずは N23、次に N24、という順に、必要なものだけ作るのが安全です。

推奨はこの流れです。

```bash
# N23 .bin 作成・検証
./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream \
  -c 23 23 32 484 1 0 5 11

# N23 GPU実行
./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream \
  -g 23 23 32 484 1 0 5 0
```

N23が通ってからN24、N24が通ってからN25、でよいです。

## 判断

```text
GitHubにpush:
  84Py_constellations_GPU_cuda_codon_dynamic_p8_stream.py
  実行シェル
  README
  .gitignore

GitHubにpushしない:
  constellations_N*.bin
  constellations_N*.bin.done
  progress_N*_stream.tsv
  logs/
```

したがって、**`.bin` は事前にgit pushする必要はありません。実行時に無ければ84が作るので、生成物はGitではなくEBS/S3で扱うのがよいです。**

[1]: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github?utm_source=chatgpt.com "About large files on GitHub"
[2]: https://docs.github.com/repositories/working-with-files/managing-large-files/about-git-large-file-storage?utm_source=chatgpt.com "About Git Large File Storage"

はい、ここまでの結果は非常に良いと思います。
**84 stream版は、N23まで正しく完走しています。**

出力を見る限り、

```text
21:      314666222712                0          0:25:38.459    ok
22:     2691008701644                0          3:18:42.963    ok
23:    24233937684440                0   1 day, 6:08:25.451    ok
```

ですので、N21〜N23までの正解値一致は確認済みです。特にN23が `ok` で通ったのは大きいです。

`.bin` / `.done` を Git 管理しない方針も正解です。`.gitignore` はこの形でよいと思います。

```bash
cat >> .gitignore <<'EOF'
constellations_N*.bin
constellations_N*.bin.done
progress_N*_stream.tsv
logs/
EOF
```

現在の `./84... -g` は引数なしなので、おそらくデフォルトの N5〜N24 通し実行に入り、今はN24を処理中という理解でよいです。N23まで `ok` が出ているので、次にN24の最終行が出れば今回の主目的に到達します。

実行中は、別SSHで以下だけ見ておけば十分です。

```bash
tail -f progress_N24_7_stream.tsv
```

または軽く監視するなら：

```bash
watch -n 60 '
date
free -h
df -h .
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv
tail -5 progress_N24_7_stream.tsv 2>/dev/null || true
'
```

N24の期待値は以下です。

```text
227514171973736
```

最終的に、

```text
24:   227514171973736                0   ...    ok
```

となれば成功です。

一点だけ注意するなら、N24は長期実行なので、`progress_N24_7_stream.tsv` とログ、`.bin/.done` はGitではなく、必要に応じてEBSまたはS3へ退避するのが安全です。

```bash
ls -lh constellations_N24_7.bin constellations_N24_7.bin.done progress_N24_7_stream.tsv 2>/dev/null
```

ここまでの流れを見る限り、**メモリ枯渇問題は84 stream化で解消し、N24の結果待ちの段階に入った**と考えてよいです。

はい、**GPU/CUDA/C に移植した方が速度が出る可能性はあります。**
ただし、率直に言うと、**同じ設計のまま Codon → CUDA C に置き換えるだけでは、劇的には速くならない可能性が高い**と思います。

今回の歯がゆさの本質は、言語というよりも、GPU向けの仕事の割り振りにあります。

## 現状の見立て

いまの84 stream版は、メモリ問題は非常にうまく解決しています。N21〜N23まで `ok` で通り、N24も実行中という点では大きな前進です。

ただ、速度面では、GPUらしい大きな優位がまだ出ていません。過去ログを見ると、以前のC/CUDA実装では N21 が `18:46.52`、N22 が `3:00:22.54`、N23 が `1日6:03:49.29` という記録があります。
一方、今回の84 stream版では、N21が約25分、N22が約3時間18分、N23が約1日6時間8分ですから、**現行Codon GPUは過去のCUDA C実装と同等か、やや遅い程度**に見えます。

これは逆に言えば、Codonが全く駄目というより、**現行GPUカーネルの探索構造がGPUの得意形になり切っていない**ということだと思います。

## CUDA Cにすると速くなる部分

CUDA Cにすれば、次の部分は改善しやすいです。

```text
1. レジスタ使用量の制御
2. shared / constant memory の明示利用
3. __device__ / __forceinline__ による関数展開
4. warp単位の最適化
5. launch bounds / occupancy 調整
6. プロファイラ Nsight Compute による詳細解析
7. メモリアクセス・分岐・レジスタスピルの把握
```

特に、今のカーネルは1 constellation を1 thread が深さ優先探索する形です。この構造はシンプルですが、GPUでは以下が起きやすいです。

```text
・threadごとの探索量が大きく違う
・warp内で分岐がばらける
・長く走るthreadが全体を引っ張る
・探索の深さが不揃い
・スタック配列でレジスタ/ローカルメモリを多く使う
```

このあたりは、CUDA Cのほうが観測・調整しやすいです。

## ただし、単純移植だけでは限界があります

現行カーネルの考え方は、

```text
1 constellation = 1 GPU thread
各threadが自分のDFSを最後まで実行
```

です。

この方式だと、CPUでは非常に自然ですが、GPUでは負荷の偏りが出ます。重い constellation を担当したthreadだけが長く残り、同じwarp内の他threadは早く終わって待つような状態になります。

そのため、CUDA Cへ移植しても、同じ構造なら改善幅はおそらく、

```text
1.2倍〜2倍程度
```

に収まる可能性があります。
もちろん、Codon生成コードでレジスタスピルや無駄な型変換が多ければ、もっと改善する余地はあります。しかし、**10倍速くなるような種類の改善は、言語変更だけでは難しい**と思います。

## 本当に速くするなら「CUDA C化」より「GPU向け再設計」

GPUでCPUを明確にしのぐには、CUDA Cへの移植と同時に、探索の粒度をGPU向けに変える必要があります。

具体的には、次のような方向です。

### 1. constellationをさらに細かく分割する

現在は p6/p7/p8 の constellation をGPU threadに渡していますが、1タスクがまだ重すぎます。

GPU向けには、

```text
preset_queens を深くする
または
GPU投入前に数段だけCPUで展開して、より細かいtaskにする
```

方がよいです。

ただし、深くしすぎると `.bin` が巨大化します。なので、今回作った stream方式はかなり重要です。`.bin`巨大化を許容できる構造になったからです。

### 2. 重さ別にタスクを並べ替える

N21/N22のログを見ると、chunkごとの kernel 時間にかなり差があります。84 streamのN21 32x484でも、chunkごとに10秒台から14秒台程度までばらついています。

これは、同じ15488件でも中身の重さが違うということです。

次は、

```text
軽いtask
中くらいtask
重いtask
```

を推定して並べ替える、または重いものだけ別kernelにする価値があります。

### 3. function id / 境界分類ごとにkernelを分ける

現在のkernelは、`funcid` を見ながら内部で分岐しています。GPUでは分岐が大きなコストになります。

CUDA C化するなら、

```text
SQB系
SQd0系
SQd1系
SQd2系
```

のように境界分類別にkernelを分けると、分岐が減り、warp内の処理が揃いやすくなります。

これはGPUらしい改善です。

### 4. persistent kernel / work queue 方式

本格的には、1 thread 1 constellation 固定ではなく、

```text
GPU上のworkerがキューから次のtaskを取る
早く終わったworkerは次のtaskへ進む
```

方式が考えられます。

ただし、これは実装難度がかなり上がります。atomicやqueue管理が必要になり、以前Codonでatomic周りに苦労した流れを考えると、CUDA Cの方が向いています。

## CUDA C化するなら段階的が良いです

いきなり全部をC/CUDAに書き直すより、次の順番が良いと思います。

```text
第1段階:
  84 streamの .bin 形式をそのまま使う
  GPU kernel部分だけ CUDA C で実装
  Python/Codon側は .bin生成と検証に残す

第2段階:
  CUDA C版で .bin chunk読み → GPU実行
  結果が84と一致するかN21/N22で検証

第3段階:
  function id別kernel、重さ別sort、境界分類別kernelを試す

第4段階:
  persistent kernel / work queue を検討
```

これが一番安全です。

つまり、**84を捨てるのではなく、84をCUDA C版の入力生成器・検証器として使う**のが良いです。

## 私のおすすめ判断

現時点での判断はこうです。

```text
Codon 84 stream:
  正しさ確認・N24検証・.bin生成基盤として継続

CUDA C:
  速度改善用の別ブランチとして開始

いきなり全面移植:
  おすすめしない

まず kernel runner だけCUDA C化:
  おすすめ
```

特に、今回のstream化で `.bin` 形式ができたのは大きいです。
`constellations_N{N}_{preset}.bin` を共通入力にすれば、

```text
Codon版GPU
CUDA C版GPU
CPU検証版
```

を同じ入力で比較できます。

## 期待できる速度感

現実的には、単純なCUDA C移植なら、

```text
1.2倍〜2倍
```

本格的にGPU向け再設計まで進めれば、

```text
2倍〜5倍以上
```

は狙える可能性があります。

ただし、N-QueensのDFSは分岐・偏り・探索深さの不均一が強いので、行列計算のようなGPU向け問題とは違います。GPUに載せれば自然に速くなるタイプではありません。

## 結論

**CUDA Cに変換した方が速度が出る可能性はあります。**
しかし、最大のポイントは「CodonかCUDA Cか」ではなく、**1 constellation = 1 thread の不均一DFSを、GPUが処理しやすい粒度・分類・キュー構造に変えること**です。

私なら次の一手は、N24の結果を待ちつつ、

```text
84 streamの .bin を読む CUDA C runner
```

を別ブランチで作ります。まずはkernelロジックを極力同じにして、N21/N22で84と完全一致させる。その後、CUDA Cならではの最適化に進むのが一番堅いです。

はい、**方向性としてはかなり有望**です。
ただし、表現を少し正確にすると、**「カーネルを複数にする」だけで不均一が解消するというより、`funcid`/境界ケースごとにタスクを分け、同種のタスクだけを同じGPU投入単位にまとめると、不均一と分岐発散をかなり減らせる可能性がある**、という見立てです。

15Pyの設計はその意味で重要なヒントになります。

## 15Pyの良い点

15Pyでは、`exec_solutions()` が `startijkl` から `j,k,l` を復元し、境界条件に応じて `SQBkBlBjrB`、`SQBklBjrB`、`SQd2B` など多数の `SQ...` 再帰関数へ振り分けています。つまり、探索本体が「巨大な汎用DFS」ではなく、ケース別の専用DFSになっています。

コードコメントでも、`exec_solutions() → SQ... 群` がコーナー、境界、特殊行 `j,k,l` に応じて最適な再帰ソルバを選ぶ構造だと説明されています。

さらに、15Pyのレビューコメントでは、`SQ` 系について「次行先読みを `_extra_block_for_row()` / `_should_go_plus1()` の2段で関数化し、多数のSQバリアントにまたがる可読性と再利用性を維持」「`row==mark1` や `<<2 / >>2` の2行スキップなど固定行の強制通過最適化も良い」と整理されています。

この思想はGPUにも効く可能性があります。

## 現在の84 GPU kernelとの違い

現在の84系GPUカーネルは、1つの `kernel_dfs_iter_gpu()` の中で `funcid` を見て、メタテーブルとビットマスクで状態遷移を行う方式です。

これは実装上は非常にうまくまとまっています。
ただしGPUでは、1つのwarp内で異なる `funcid` のthreadが混ざると、

```text
あるthreadは SQd2B 系
あるthreadは SQBjlBkBlBjrB 系
あるthreadは SQd0BkB 系
```

のようになり、内部分岐や探索の深さがばらけます。

結果として、

```text
・warp divergence が増える
・短いタスクのthreadが先に終わる
・重いタスクのthreadだけが残る
・chunk全体の終了が重いタスクに引っ張られる
```

という状態になります。

15Pyのようにケース別に分けると、この混在を減らせます。

## ただし「複数カーネル化」には2種類あります

ここが大事です。

### A. カーネルを関数別に分けるだけ

例えば、

```text
kernel_SQd0B
kernel_SQd1B
kernel_SQd2B
kernel_SQBjrB
...
```

を作るだけです。

これは分岐発散を減らす効果はあります。
しかし、**同じkernel内でも重いconstellationと軽いconstellationが混ざれば、不均一DFSは残ります。**

つまり効果はありますが、これだけでは不十分です。

### B. `.bin` を funcid ごとに分けて、同種タスクだけ同じkernelへ流す

こちらが本命です。

```text
constellations_N24_7_fid00.bin
constellations_N24_7_fid01.bin
...
constellations_N24_7_fid25.bin
```

または1ファイルのままでもよいですが、stream読み込み時に `funcid` ごとのbucketへ振り分けます。

そのうえで、

```text
fid=14 のタスクだけ kernel_fid14 へ
fid=12 のタスクだけ kernel_fid12 へ
fid=0  のタスクだけ kernel_fid0 へ
```

と流します。

これなら、

```text
・warp内の分岐が揃う
・1 kernel 内の処理パターンが近くなる
・重いfidだけ別扱いできる
・軽いfidを巨大chunkで流し、重いfidを小chunkで流せる
```

という改善が期待できます。

## 不均一解消に効くのは「カーネル分割」より「タスク分類」

実際には、次の順番がよいと思います。

```text
第1段階:
  84 streamのまま、各constellationの funcid を集計する

第2段階:
  fid別に件数・total・chunk時間を出す

第3段階:
  fid別に .bin を分割する、またはstream時にbucket化する

第4段階:
  まずは同じ kernel_dfs_iter_gpu をfid別bucketごとに呼ぶ

第5段階:
  効果があれば、重いfidだけ専用kernel化する
```

いきなり26個の専用GPU kernelを作るより、まず **同じカーネルのままfid別投入** がよいです。

なぜなら、これだけで、

```text
warp内funcid混在
```

をかなり減らせるからです。

それで効果が出るなら、次に専用kernel化へ進む価値があります。

## 15Py方式をGPUへ戻すなら、まずは「fid別bucket」がよい

15PyはCPU/Codonでは `@par for constellation in constellations` で各星座を並列処理し、その中でケースごとの `SQ...` 関数を呼び分けています。

GPU版でこれをそのまま再帰関数群として復元するのは難度が高いです。
Codon GPU kernel内で多数の再帰関数、または擬似再帰関数を分けるより、現在の非再帰スタック型kernelを活かしながら、入力側を分類する方が安全です。

つまり、まずはこうです。

```text
84 stream
  ↓
read bin chunk
  ↓
build_soa_for_range() で funcid を計算
  ↓
funcid別に task index をbucket化
  ↓
bucketごとに kernel_dfs_iter_gpu を呼ぶ
```

最初はkernel自体は1つのままで構いません。
この時点で性能が上がるなら、「分岐混在」がかなり悪さをしていた証拠になります。

## 次にやるべき具体策

私は次の86番をおすすめします。

```text
86Py_constellations_GPU_cuda_codon_stream_fid_bucket.py
```

機能は以下です。

```text
1. 84 streamをベースにする
2. .binからSTEPS件読む
3. build_soa_for_range()でfuncidを計算
4. funcid_arrを見て 0..25 のbucketへ分ける
5. 各bucketを小さいTaskSoAへ詰め直す
6. bucketごとに kernel_dfs_iter_gpu を呼ぶ
7. fid別の件数・時間・totalをログ出力
```

この段階では、まだ専用kernelは作りません。

ログはこういう形が良いです。

```text
[fid-bucket] N=22 fid=12 count=xxxx kernel=... total=...
[fid-bucket] N=22 fid=14 count=xxxx kernel=... total=...
[fid-bucket-summary] N=22 fid=12 count=... time=... total=...
```

これで、どの `fid` が重いかが分かります。

## その次に専用kernel化する

fid別bucketで効果が確認できたら、次に重い `fid` だけ専用kernelにします。

例えばN24境界診断でよく問題になっていた `SQd2` 系であれば、

```text
kernel_fid12_SQd2BkBlB
kernel_fid16_SQd2BlBkB
kernel_fid17_SQd2BlkB
```

のように、重いものから専用化します。

全funcidを一気に専用kernel化する必要はありません。

まずは、

```text
重いfid上位3つ
```

だけでよいです。

## 注意点

カーネルを複数にしても、GPU kernel launch 回数は増えます。
N21やN22では問題ないと思いますが、小さいNではlaunch overheadが勝つ可能性があります。

したがって対象は、

```text
N22以上
```

でよいと思います。

また、fid別にbucket化すると、元の順序は変わりますが、合計値だけ取るなら問題ありません。結果は加算なので順序非依存です。

## 判断

ご質問への答えは、**はい、15Pyのような細分化思想はGPUの不均一DFS改善に有効だと思います。**

ただし、いきなり「SQ関数ごとの複数GPU kernel」にするより、まずは次がよいです。

```text
1. 84 streamを維持
2. funcid別にbucket化
3. 同じkernelをbucketごとに呼ぶ
4. fid別の時間と件数を測る
5. 重いfidだけ専用kernel化
```

この順序なら、正しさを崩しにくく、N21/N22で84との比較もできます。

結論として、**15Pyの細分化は非常に良いヒントです。ただし最初の一手は「複数kernel」ではなく「fid別bucket投入」です。そこから重いfidだけ専用kernel化するのが最も堅いです。**





# N23 .bin 作成・検証
./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream \
  -c 23 23 32 484 1 0 5 11

# N23 GPU実行
./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream \
  -g 23 23 32 484 1 0 5 0
  
workspace#suzuki$ date
2026年  5月 15日 金曜日 20:50:42 JST
workspace#suzuki$ uname -a
Linux ip-172-31-14-193.us-west-2.compute.internal 6.1.115-126.197.amzn2023.x86_64 #1 SMP PREEMPT_DYNAMIC Tue Nov  5 17:36:57 UTC 2024 x86_64 x86_64 x86_64 GNU/Linux
workspace#suzuki$ codon build -release 84Py_constellations_GPU_cuda_codon_dynamic_p8_stream.py
workspace#suzuki$ ./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream -c
CPU mode selected
 N:             Total           Unique         hh:mm:ss.ms
 5:                10                0          0:00:00.000
 6:                 4                0          0:00:00.088    ok
 7:                40                0          0:00:00.024    ok
 8:                92                0          0:00:00.001    ok
 9:               352                0          0:00:00.005    ok
10:               724                0          0:00:00.002    ok
11:              2680                0          0:00:00.010    ok
12:             14200                0          0:00:00.020    ok
13:             73712                0          0:00:00.041    ok
14:            365596                0          0:00:00.091    ok
15:           2279184                0          0:00:00.171    ok
16:          14772512                0          0:00:00.275    ok
17:          95815104                0          0:00:00.409    ok
18:         666090624                0          0:00:04.455    ok
19:        4968057848                0          0:00:17.064    ok
20:       39029188884                0          0:02:10.450    ok
21:      314666222712                0          0:18:05.956    ok
22:     2691008701644                0          2:38:08.664    ok
23:    24233937684440                0   1 day, 0:43:10.509    ok


suzuki@cudacodon$ date
2026年  5月 15日 金曜日 09:34:47 UTC
suzuki@cudacodon$ codon build -release 84Py_constellations_GPU_cuda_codon_dynamic_p8_stream.py
suzuki@cudacodon$ ./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream -g     GPU mode selected
version        : 84 stream bin GPU runner from 82 dynamic preset P8
cross_stripe_safe: 0
 N:             Total           Unique         hh:mm:ss.ms
 5:                10                0          0:00:00.000
 6:                 4                0          0:00:00.004    ok
 7:                40                0          0:00:00.003    ok
 8:                92                0          0:00:00.002    ok
 9:               352                0          0:00:00.002    ok
10:               724                0          0:00:00.003    ok
11:              2680                0          0:00:00.004    ok
12:             14200                0          0:00:00.007    ok
13:             73712                0          0:00:00.011    ok
14:            365596                0          0:00:00.018    ok
15:           2279184                0          0:00:00.037    ok
16:          14772512                0          0:00:00.107    ok
17:          95815104                0          0:00:00.466    ok
18:         666090624                0          0:00:03.505    ok
19:        4968057848                0          0:00:22.592    ok
20:       39029188884                0          0:02:24.917    ok
21:      314666222712                0          0:25:38.459    ok
22:     2691008701644                0          3:18:42.963    ok
23:    24233937684440                0   1 day, 6:08:25.451    ok

g5.16xlarge は NVIDIA A10G GPU を搭載しており、CUDA 13.0 対応のドライバが入っています。
g5.xlarge は NVIDIA A10G GPU を搭載しており、CUDA 13.0 対応のドライバが入っています。

速度が上がらない理由
-----------------------
g5.xlarge  → A10G 1枚
g5.16xlarge → A10G 1枚
------------------------

2023/11/22 これまでの最高速実装（CUDA GPU 使用/C）
C/CUDA NVIDIA(GPU)
$ nvcc -O3 -arch=sm_61 -m64 -ptx -prec-div=false 04CUDA_Symmetry_BitBoard.cu && POCL_DEBUG=all ./a.out -n ;
対称解除法 GPUビットボード
18:         666090624        83263591    000:00:00:01.65
19:        4968057848       621012754    000:00:00:13.80
20:       39029188884      4878666808    000:00:02:02.52
21:      314666222712     39333324973    000:00:18:46.52
22:     2691008701644    336376244042    000:03:00:22.54
23:    24233937684440   3029242658210    001:06:03:49.29
24:   227514171973736  28439272956934    012:23:38:21.02
25:  2207893435808352 275986683743434    140:07:39:29.96

"""

import gpu
import sys
from typing import List,Set,Dict,Tuple
from datetime import datetime

MAXD:Static[int]=32

VERSION_TAG:str="84 stream bin GPU runner from 82 dynamic preset P8"
CROSS_STRIPE_SAFE_DEFAULT:bool=False

# bench_mode=10 の診断用。通常は False のまま。
# True にすると preset_queens<=5 の constellation_signatures 重複排除を無効化し、
# N24 境界分類で signature prune が潰しすぎていないかを調べる。
DISABLE_CONSTELLATION_SIGNATURE_PRUNE:bool=False


"""  構造体配列 (SoA) タスク管理クラス """
class TaskSoA:
  """ コンストラクタ """
  def __init__(self,m:int):
    self.ld_arr:List[int]=[0]*m
    self.rd_arr:List[int]=[0]*m
    self.col_arr:List[int]=[0]*m
    self.row_arr:List[int]=[0]*m
    self.free_arr:List[int]=[0]*m
    self.jmark_arr:List[int]=[0]*m
    self.end_arr:List[int]=[0]*m
    self.mark1_arr:List[int]=[0]*m
    self.mark2_arr:List[int]=[0]*m
    self.funcid_arr:List[int]=[0]*m
    self.ijkl_arr:List[int]=[0]*m

""" CUDA GPU 用 DFS カーネル関数  """
@gpu.kernel
def kernel_dfs_iter_gpu(
    ld_arr,rd_arr,col_arr,row_arr,free_arr,
    jmark_arr,end_arr,mark1_arr,mark2_arr,
    funcid_arr,w_arr,
    meta_next:Ptr[u8],
    results,
    m:int,board_mask:int,
    n3:int,n4:int,
):
    """
    機能:
      GPU 上で「1 constellation = 1 thread」の DFS を非再帰で実行し、
      この constellation が担当する部分探索の解数を数えて results[i] に格納します。
      最終的に results[i] には（解数 * 対称性重み）を保存します。

    引数（抜粋）:
      ld_arr/rd_arr/col_arr/row_arr/free_arr:
        constellation ごとの開始状態（ビットボード）。
      funcid_arr:
        分岐モードID（functionid）。
      w_arr:
        対称性の重み（2/4/8）。results へ書く直前に掛けます。
      meta_next:
        functionid -> next functionid の遷移表（u8 配列）。
      board_mask:
        (1<<N)-1。ビットボードを常にこの範囲へ正規化します。
      m:
        タスク数（i >= m は処理しない）。

    前提/不変条件:
      - ld/rd/col/free は board_mask 内に収まる（念のため kernel 側でも &mask します）。
      - スタック深さ sp は 0..MAXD-1。超えた場合は安全弁で早期 return します。

    ホットパス（ソース引用）:
      bit = a & -a
      avail[sp] = a ^ bit
    """
    # NOTE: GPU では list/tuple 参照が遅くなりがちなので、
    #       分岐テーブルを Static[int] のビットマスクとして焼き込み、
    #       (MASK >> f) & 1 の O(1) 判定に寄せる。
    META_AVAIL_MASK:Static[int]=69226252
    IS_BASE_MASK:Static[int]=69222408
    IS_JMARK_MASK:Static[int]=4
    IS_MARK_MASK:Static[int]=199209203
    IS_P5_MASK:Static[int]=(1<<8)|(1<<9)|(1<<10)|(1<<11)
    SEL2_MASK:Static[int]=(1<<1)|(1<<6)|(1<<13)|(1<<17)|(1<<20)|(1<<25)
    STP3_MASK:Static[int]=(1<<4)|(1<<7)|(1<<15)|(1<<18)|(1<<22)|(1<<24)
    MASK_K_N3:Static[int]=185471169
    MASK_K_N4:Static[int]=4227088
    MASK_L_1:Static[int]=12689458
    MASK_L_2:Static[int]=17039488

    # mixed32 ビットボード系。
    # rd は盤面外の高位ビットが後続の >> で盤面内へ入る可能性があるため int のまま保持する。
    # ld は高位ビットが再び盤面へ戻らないが、rd と揃えて int のまま。
    # col/avail/ctrl は盤面幅内または小さい制御値なので u32 化してローカルメモリ圧を下げる。
    # 未初期化 ctrl: bit19=0, bits0..4=current fid, bits5..9=current row
    # 初期化済 ctrl: bit19=1
    #   bits 0..4   : child/next fid
    #   bits 5..9   : child row after this frame transition
    #   bits 10..11 : step (1/2/3), block時のみデコード
    #   bit  12     : add1, block時のみデコード
    #   bit  13     : use_blocks
    #   bit  14     : future check enabled
    #   bits 15..16 : blockL encoded value
    #   bits 17..18 : blockK type: 0=0, 1=n3, 2=n4

    # child ctrl は低10bitそのものなので、通常pushでは
    #   ctrl[child] = ctrl[parent] & 1023
    # とでき、hotpathの int(ctrl)>>14 デコードを避ける。
    INIT_MASK:Static[int]=524288  # 1<<19
    ld=__array__[int](MAXD)
    rd=__array__[int](MAXD)
    col=__array__[u32](MAXD)
    avail=__array__[u32](MAXD)
    ctrl=__array__[u32](MAXD)
    bm:u32=u32(board_mask)
    bK=0
    bL:u32=u32(0)

    i=(gpu.block.x*gpu.block.dim.x)+gpu.thread.x
    if i>=m:return

    jmark=jmark_arr[i]
    endm=end_arr[i]
    mark1=mark1_arr[i]
    mark2=mark2_arr[i]
    sp=0
    ctrl[0]=u32(funcid_arr[i] | (row_arr[i]<<5))
    ld[0]=ld_arr[i]
    rd[0]=rd_arr[i]
    col[0]=u32(col_arr[i])

    free0:u32=u32(free_arr[i])&bm
    if free0==u32(0):
      results[i]=u64(0)
      return
    avail[0]=free0
    total:u64=u64(0)
    while sp>=0:
      a=avail[sp]
      if a==u32(0):
        sp-=1
        continue
      cv0=ctrl[sp]
      if (cv0&u32(INIT_MASK))==u32(0):
        cv0i:int=int(cv0)
        f:int=cv0i&31
        rowv:int=(cv0i>>5)&31
        nfid=meta_next[f]

        #######################################
        # P5 same-row transition
        #
        # fid=8..11:
        #   8  SQBjlBkBlBjrB -> 0 SQBkBlBjrB
        #   9  SQBjlBklBjrB  -> 4 SQBklBjrB
        #   10 SQBjlBlBkBjrB -> 5 SQBlBkBjrB
        #   11 SQBjlBlkBjrB  -> 7 SQBlkBjrB
        #
        # mark1 到達時、盤面/row/free を変えずに next fid へ遷移する。
        # その後、同じ row で next fid 側の step=2/3 + block が発火する。
        #######################################
        if ((IS_P5_MASK>>f)&1)==1:
          if rowv==mark1:
            f=int(nfid)
            nfid=meta_next[f]

        #######################################
        # 基底 is_base
        isb=(IS_BASE_MASK>>f)&1
        #######################################
        if isb==1 and rowv==endm:
          if f==14:# SQd2B 特例
            total+=u64(1) if ((a&~u32(1))!=u32(0)) else u64(0)
          else:
            total+=u64(1)
          sp-=1
          continue
        #######################################
        # 通常状態設定
        aflag=(META_AVAIL_MASK>>f)&1
        #######################################
        stepv=1
        addv=0
        use_blocks=0
        use_future=1 if (aflag==1) else 0
        nextfv=f
        #######################################
        # is_mark step=2/3 + block
        ism=(IS_MARK_MASK>>f)&1
        #######################################
        if ism==1:
          at_mark=0
          ###################
          # sel
          sel=2 if ((SEL2_MASK>>f)&1) else 1
          ###################
          if sel==1:
            if rowv==mark1:
              at_mark=1
          if sel==2:
            if rowv==mark2:
              at_mark=1
          ###################
          # mark
          ###################
          if at_mark==1 and a!=u32(0):
            use_blocks=1
            use_future=0
            ###################
            # step
            stepv=3 if ((STP3_MASK>>f)&1) else 2
            ###################
            # add
            addv=1 if f==20 else 0
            ###################
            nextfv=int(nfid)
        #######################################
        # is_jmark
        isj=(IS_JMARK_MASK>>f)&1
        #######################################
        if isj==1:
          if rowv==jmark:
            a&=~u32(1)
            avail[sp]=a
            if a==u32(0):
              sp-=1
              continue
            ld[sp]|=1
            nextfv=int(nfid)
        
        fcv=0
        if use_future==1 and (rowv+stepv)<endm:
          fcv=1
        bLv=0
        ktype=0
        if use_blocks==1:
          bLv=((MASK_L_1>>f)&1)|(((MASK_L_2>>f)&1)<<1)
          if ((MASK_K_N3>>f)&1)==1:
            ktype=1
          if ((MASK_K_N4>>f)&1)==1:
            ktype=2
        child_row:int=rowv+stepv
        ctrl[sp]=u32(524288 | nextfv | (child_row<<5) | (stepv<<10) | (addv<<12) | (use_blocks<<13) | (fcv<<14) | (bLv<<15) | (ktype<<17))
      #----------------
      # 1bit 展開
      #----------------
      a=avail[sp]
      bit:u32=a&(u32(0)-a)
      avail[sp]=a^bit
      #----------------
      # 次状態計算（2値選択はそのまま）
      #----------------
      cv=ctrl[sp]
      bit_i:int=int(bit)
      if (cv&u32(8192))!=u32(0):  # use_blocks bit13
        cvi:int=int(cv)
        stepv:int=(cvi>>10)&3
        addv:int=(cvi>>12)&1
        bLi:int=(cvi>>15)&3
        kt:int=(cvi>>17)&3
        bK=0
        if kt==1:
          bK=n3
        elif kt==2:
          bK=n4
        nld=((ld[sp]|bit_i)<<stepv)|addv|bLi
        nrd=((rd[sp]|bit_i)>>stepv)|bK
      else:
        nld=(ld[sp]|bit_i)<<1
        nrd=(rd[sp]|bit_i)>>1
      ncol:u32=col[sp]|bit
      nf:u32=bm&~(u32(nld)|u32(nrd)|ncol)
      if nf==u32(0):
        continue
      if (cv&u32(16384))!=u32(0):  # future bit14
        if (bm&~(u32(nld<<1)|u32(nrd>>1)|ncol))==u32(0):
          continue
      #----------------
      # push
      #----------------
      sp+=1
      if sp>=MAXD:
        results[i]=total*w_arr[i]
        return
      ctrl[sp]=cv&u32(1023)  # child fid + child row
      ld[sp]=nld
      rd[sp]=nrd
      col[sp]=ncol
      avail[sp]=nf

    results[i]=total*w_arr[i]

"""dfs()の非再帰版"""
def dfs_iter(
  meta,blockK,blockL,board_mask,
  functionid:int,ld:int,rd:int,col:int,row:int,free:int,
  jmark:int,endmark:int,mark1:int,mark2:int
)->u64:
  """
  CPU 上で DFS を非再帰で実行する。

  78 FIX:
    funcptn==4 / P5 / fid=8..11 を追加。

    fid=8..11:
      8  SQBjlBkBlBjrB  -> 0 SQBkBlBjrB
      9  SQBjlBklBjrB   -> 4 SQBklBjrB
      10 SQBjlBlBkBjrB  -> 5 SQBlBkBjrB
      11 SQBjlBlkBjrB   -> 7 SQBlkBjrB

    P5 は mark1 到達時に queen を置かず、row も進めず、
    same-row のまま next_funcid へ遷移する。
  """

  total:u64=u64(0)

  # スタック要素:
  #   functionid, ld, rd, col, row, free
  stack:List[Tuple[int,int,int,int,int,int]]=[(functionid,ld,rd,col,row,free)]

  while stack:
    functionid,ld,rd,col,row,free=stack.pop()

    if not free:
      continue

    next_funcid,funcptn,avail_flag=meta[functionid]
    avail:int=free

    # ------------------------------------------------------------
    # 基底
    # ------------------------------------------------------------
    if funcptn==5 and row==endmark:
      # fid=14 SQd2B 特例
      if functionid==14:
        total+=u64(1) if (avail>>1) else u64(0)
      else:
        total+=u64(1)
      continue

    # ------------------------------------------------------------
    # 既定値
    # ------------------------------------------------------------
    step:int=1
    add1:int=0
    row_step:int=row+1

    use_blocks:bool=False
    use_future:bool=(avail_flag==1)

    local_next_funcid:int=functionid

    _blockK:int=0
    _blockL:int=0

    # ------------------------------------------------------------
    # P1/P2/P3: mark 行で step=2/3 + block
    # ------------------------------------------------------------
    if funcptn in (0,1,2):
      at_mark:bool=(row==mark1) if funcptn in (0,2) else (row==mark2)

      if at_mark and avail:
        step=2 if funcptn in (0,1) else 3
        add1=1 if (funcptn==1 and functionid==20) else 0
        row_step=row+step

        _blockK=blockK[functionid]
        _blockL=blockL[functionid]

        use_blocks=True
        use_future=False
        local_next_funcid=next_funcid

    # ------------------------------------------------------------
    # P4: jmark 特殊
    # ------------------------------------------------------------
    elif funcptn==3 and row==jmark:
      # 列0禁止
      avail&=~1

      # ld LSB を立てる
      ld|=1

      local_next_funcid=next_funcid

      if not avail:
        continue

    # ------------------------------------------------------------
    # P5: SQBjl*jrB 系
    #
    # fid=8..11 は mark1 に到達するまでは future 付き通常探索。
    # mark1 に到達したら、盤面を変えず、row も進めず、same-row で
    # next_funcid へ遷移する。
    #
    # 例:
    #   fid=11 SQBjlBlkBjrB
    #       -> fid=7 SQBlkBjrB
    #
    # fid=7 側が同じ row==mark1 で step=3 + block を処理する。
    # ------------------------------------------------------------
    elif funcptn==4 and row==mark1:
      stack.append((next_funcid,ld,rd,col,row,avail))
      continue

    # ============================================================
    # ループ1: step=2/3 + block
    # ============================================================
    if use_blocks:
      while avail:
        bit:int=avail&-avail
        avail&=avail-1

        nld:int=((ld|bit)<<step)|add1|_blockL
        nrd:int=((rd|bit)>>step)|_blockK
        ncol:int=col|bit

        nf:int=board_mask&~(nld|nrd|ncol)

        if nf:
          stack.append((local_next_funcid,nld,nrd,ncol,row_step,nf))

      continue

    # ============================================================
    # ループ2: 通常 +1、先読みなし
    # ============================================================
    if not use_future:
      while avail:
        bit:int=avail&-avail
        avail&=avail-1

        nld:int=(ld|bit)<<1
        nrd:int=(rd|bit)>>1
        ncol:int=col|bit

        nf:int=board_mask&~(nld|nrd|ncol)

        if nf:
          stack.append((local_next_funcid,nld,nrd,ncol,row_step,nf))

      continue

    # ============================================================
    # ループ3: 通常 +1、終端付近は先読みなし
    # ============================================================
    if row_step>=endmark:
      while avail:
        bit:int=avail&-avail
        avail&=avail-1

        nld:int=(ld|bit)<<1
        nrd:int=(rd|bit)>>1
        ncol:int=col|bit

        nf:int=board_mask&~(nld|nrd|ncol)

        if nf:
          stack.append((local_next_funcid,nld,nrd,ncol,row_step,nf))

      continue

    # ============================================================
    # ループ3B: 通常 +1、先読みあり
    # ============================================================
    while avail:
      bit:int=avail&-avail
      avail&=avail-1

      nld:int=(ld|bit)<<1
      nrd:int=(rd|bit)>>1
      ncol:int=col|bit

      nf:int=board_mask&~(nld|nrd|ncol)

      if not nf:
        continue

      # 次の次が 0 なら枝刈り
      if board_mask&~((nld<<1)|(nrd>>1)|ncol):
        stack.append((local_next_funcid,nld,nrd,ncol,row_step,nf))

  return total

"""汎用 DFS カーネル。古い SQ???? 関数群を 1 本化し、func_meta の記述に従って切り替える。"""
def dfs(
    meta:List[Tuple[int,int,int]],
    blockK_by_funcid:List[int],blockL_by_funcid:List[int],
    board_mask:int,
    functionid:int,
    ld:int,rd:int,col:int,row:int,free:int,
    jmark:int,endmark:int,mark1:int,mark2:int)->u64:
  """
  78 FIX:
    funcptn==4 / P5 / fid=8..11 を追加。

    P5 は mark1 到達時に、盤面を変えず、row も進めず、
    same-row のまま next_funcid へ遷移する。
  """

  next_funcid,funcptn,avail_flag=meta[functionid]

  avail:int=free
  if not avail:
    return u64(0)

  total:u64=u64(0)

  # ------------------------------------------------------------
  # 基底
  # ------------------------------------------------------------
  if funcptn==5 and row==endmark:
    if functionid==14:
      return u64(1) if (avail>>1) else u64(0)
    return u64(1)

  # ------------------------------------------------------------
  # 既定値
  # ------------------------------------------------------------
  step:int=1
  add1:int=0
  row_step:int=row+1

  use_blocks:bool=False
  use_future:bool=(avail_flag==1)

  local_next_funcid:int=functionid

  bK:int=0
  bL:int=0

  # ------------------------------------------------------------
  # P1/P2/P3: mark 行で step=2/3 + block
  # ------------------------------------------------------------
  if funcptn in (0,1,2):
    at_mark:bool=(row==mark1) if funcptn in (0,2) else (row==mark2)

    if at_mark and avail:
      step=2 if funcptn in (0,1) else 3
      add1=1 if (funcptn==1 and functionid==20) else 0
      row_step=row+step

      bK=blockK_by_funcid[functionid]
      bL=blockL_by_funcid[functionid]

      use_blocks=True
      use_future=False
      local_next_funcid=next_funcid

  # ------------------------------------------------------------
  # P4: jmark 特殊
  # ------------------------------------------------------------
  elif funcptn==3 and row==jmark:
    avail&=~1
    ld|=1
    local_next_funcid=next_funcid

    if not avail:
      return u64(0)

  # ------------------------------------------------------------
  # P5: SQBjl*jrB 系
  #
  # fid=8..11:
  #   8  -> 0
  #   9  -> 4
  #   10 -> 5
  #   11 -> 7
  #
  # mark1 に到達したら、queen を置かず、row も進めず、
  # next_funcid 側へ同じ状態を渡す。
  # ------------------------------------------------------------
  elif funcptn==4 and row==mark1:
    return dfs(
      meta,
      blockK_by_funcid,
      blockL_by_funcid,
      board_mask,
      next_funcid,
      ld,rd,col,row,avail,
      jmark,endmark,mark1,mark2
    )

  # ============================================================
  # ループ1: step=2/3 + block
  # ============================================================
  if use_blocks:
    while avail:
      bit:int=avail&-avail
      avail&=avail-1

      nld:int=((ld|bit)<<step)|add1|bL
      nrd:int=((rd|bit)>>step)|bK
      ncol:int=col|bit

      nf:int=board_mask&~(nld|nrd|ncol)

      if nf:
        total+=dfs(
          meta,
          blockK_by_funcid,
          blockL_by_funcid,
          board_mask,
          local_next_funcid,
          nld,nrd,ncol,row_step,nf,
          jmark,endmark,mark1,mark2
        )

    return total

  # ============================================================
  # ループ2: 通常 +1、先読みなし
  # ============================================================
  if not use_future:
    while avail:
      bit:int=avail&-avail
      avail&=avail-1

      nld:int=(ld|bit)<<1
      nrd:int=(rd|bit)>>1
      ncol:int=col|bit

      nf:int=board_mask&~(nld|nrd|ncol)

      if nf:
        total+=dfs(
          meta,
          blockK_by_funcid,
          blockL_by_funcid,
          board_mask,
          local_next_funcid,
          nld,nrd,ncol,row_step,nf,
          jmark,endmark,mark1,mark2
        )

    return total

  # ============================================================
  # ループ3: 通常 +1、終端付近は先読みなし
  # ============================================================
  if row_step>=endmark:
    while avail:
      bit:int=avail&-avail
      avail&=avail-1

      nld:int=(ld|bit)<<1
      nrd:int=(rd|bit)>>1
      ncol:int=col|bit

      nf:int=board_mask&~(nld|nrd|ncol)

      if nf:
        total+=dfs(
          meta,
          blockK_by_funcid,
          blockL_by_funcid,
          board_mask,
          local_next_funcid,
          nld,nrd,ncol,row_step,nf,
          jmark,endmark,mark1,mark2
        )

    return total

  # ============================================================
  # ループ3B: 通常 +1、先読みあり
  # ============================================================
  while avail:
    bit:int=avail&-avail
    avail&=avail-1

    nld:int=(ld|bit)<<1
    nrd:int=(rd|bit)>>1
    ncol:int=col|bit

    nf:int=board_mask&~(nld|nrd|ncol)

    if not nf:
      continue

    if board_mask&~(((nld<<1)|(nrd>>1)|ncol)):
      total+=dfs(
        meta,
        blockK_by_funcid,
        blockL_by_funcid,
        board_mask,
        local_next_funcid,
        nld,nrd,ncol,row_step,nf,
        jmark,endmark,mark1,mark2
      )

  return total

""" constellations の一部を TaskSoA 形式に変換して返すユーティリティ """
def build_soa_for_range(
    N,
    constellations:List[Dict[str,int]],
    off:int,
    m:int,
    soa:TaskSoA,
    w_arr:List[u64]
)->Tuple[TaskSoA,List[u64]]:
    """
    機能:
      constellations[off:off+m] を SoA（Structure of Arrays）へ展開し、
      DFS（CPU/GPU）の入力として必要な配列群を “同一 index” に揃えて埋める。
      さらに、対称性の重み（2/4/8）を w_arr に計算して格納する。

    目的（なぜ SoA か）:
      - dict 参照（ハッシュ）を探索ループから追い出し、前処理で配列へ変換する。
      - CPU(@par) / GPU(kernel) どちらでも「t 番のタスク状態」を連続配列から取り出せる。
      - GPU では AoS より SoA の方がメモリアクセス効率が良くなりやすい。

    引数:
      N:
        盤サイズ。
      constellations:
        タスク dict の配列。少なくとも "ld","rd","col","startijkl" を持つ。
      off, m:
        対象レンジ。t=0..m-1 に constellations[off+t] を詰める。
      soa:
        出力先の SoA（ld_arr/rd_arr/col_arr/... の配列群を保持）。
      w_arr:
        出力先の重み配列。w_arr[t] = symmetry(soa.ijkl_arr[t], N)。

    返り値:
      (soa, w_arr)

    前提/不変条件:
      - constellation["ld"], ["rd"], ["col"] はビットボード（board_mask 内が望ましい）。
      - constellation["startijkl"] は
          start = start_ijkl >> 20   （開始 row）
          ijkl  = start_ijkl & ((1<<20)-1) （開始星座 pack）
        という構造でパックされていること。
      - exec_solutions() 側の meta / blockK / blockL と、ここで選ぶ target(functionid) は整合必須。

    実装上のコツ（この関数の要点）:
      - startijkl から start(row) と ijkl(i,j,k,l pack) を復元し、
        そこから「探索開始時点の ld/rd/col/free」を再構築する。
      - その状態の特徴（j,k,l,start など）から、最適な分岐 target(functionid) と
        mark/jmark/endmark を決め、SoA へ格納する。
    """

    # ----------------------------------------
    # ビットマスク類（盤面幅の正規化に使う）
    # ----------------------------------------
    board_mask:int=(1<<N)-1

    # small_mask は「N-2 幅」のマスク（N が小さいときは 0 幅を許容）
    # col を組み立てる際に ~small_mask を混ぜる設計（既存実装の意図を保持）
    small_mask:int=(1<<max(0,N-2))-1

    # よく使う定数
    N1:int=N-1
    N2:int=N-2

    # 出力（soa は外から渡される前提。必要なら TaskSoA(m) を呼び出し側で確保）
    # soa = TaskSoA(m)
    # ----------------------------------------
    # レンジ分のタスクを SoA に詰める
    # ----------------------------------------
    for t in range(m):
        constellation=constellations[off+t]

        # 特殊行（後段 DFS で使う）
        #   - jmark: funcptn==3 のときに "row==jmark" で特別処理に入る
        #   - mark1/mark2: funcptn in (0,1,2) のときに "row==mark1/mark2" で mark段(step=2/3)に入る
        jmark=0
        mark1=0
        mark2=0

        # startijkl: 上位に start(row)、下位20bitに ijkl pack を持つ
        #   start = start_ijkl >> 20  （探索開始行）
        #   ijkl  = start_ijkl & ((1<<20)-1) （開始星座(i,j,k,l)パック）
        start_ijkl=constellation["startijkl"]
        start=start_ijkl>>20
        ijkl=start_ijkl&((1<<20)-1)

        # ijkl から j,k,l を取り出し（i はここでは不要なので取っていない）
        j,k,l=getj(ijkl),getk(ijkl),getl(ijkl)

        # ----------------------------------------
        # 開始状態（ld/rd/col）の再構築
        #   - constellation 側の ld/rd/col は “ある基準”で作られているので、
        #     ここで start(row) に合わせて正規化・補正して探索入口に合わせる。
        # ----------------------------------------

        # ld/rd/col は 1bit シフトして “内部表現”を合わせている（既存設計）
        #   ※ dfs 側は「次段生成で <<1/>>1」するので、入口の位置合わせが重要
        ld=constellation["ld"]>>1
        rd=constellation["rd"]>>1

        # col: (col>>1) に ~small_mask を混ぜ、board_mask で正規化して盤面外ビットを落とす
        col=(constellation["col"]>>1)|(~small_mask)

        # col を盤面幅へ正規化（上位ゴミビット除去）
        col&=board_mask

        # LD: j と l の列ビット（MSB側基準）を作る
        # 例: (1 << (N-1-j)) は列 j に相当
        LD=(1<<(N1-j))|(1<<(N1-l))

        # ld は start 行に合わせて LD を右にずらして混ぜる（既存式のまま）
        ld|=LD>>(N-start)

        # rd 側の補正（start と k の関係で入れるビットが変わる）
        if start>k:
            rd|=(1<<(N1-(start-k+1)))

        # j がゲート条件を満たすとき rd へ追加補正
        if j>=2*N-33-start:
            rd|=(1<<(N1-j))<<(N2-start)

        # ----------------------------------------
        # free: 現在行(start)で置ける候補列
        # ----------------------------------------
        free=board_mask&~(ld|rd|col)

        # ----------------------------------------
        # 分岐（現行の exec_solutions と同一）
        #   target(functionid) と mark/jmark/endmark を決める
        #
        # target (=functionid) は FID/SQラベルと 1:1 対応
        #   func_meta[functionid] = (next_funcid, funcptn, availptn)
        #     - funcptn: 段パターン
        #         0/1/2: mark系（row==mark1/mark2 で step=2/3 + block）
        #         3    : jmark系（row==jmark で列0禁止 + ld LSB）
        #         5    : base系（row==endmark で解カウント）
        #         4    : “通常”扱いに落ちる（dfs の else 経路に入る）
        #     - availptn: 1なら先読み枝刈りを有効化（dfs の use_future）
        # ----------------------------------------
        endmark=0
        target=0

        # 条件を事前に bool 化（枝の可読性/分岐コスト低減）
        j_lt_N3=(j<N-3)
        j_eq_N3=(j==N-3)
        j_eq_N2=(j==N-2)

        k_lt_l=(k<l)
        start_lt_k=(start<k)
        start_lt_l=(start<l)

        l_eq_kp1=(l==k+1)
        k_eq_lp1=(k==l+1)

        # j_gate: ある境界より j が大きいと “ゲートON” 扱い（既存設計）
        j_gate=(j>2*N-34-start)

        # --------------------------
        # case 1) j < N-3
        #   - “一般ケース”の大半
        #   - jmark = j+1, endmark = N-2
        #   - gate ON/OFF でターゲット（=functionid）を切り替える
        # --------------------------
        if j_lt_N3:
            # jmark: j+1 行で jmark 特殊を入れる設計
            jmark=j+1

            # endmark: ここでは N-2 を終端とする
            endmark=N2

            if j_gate:
                # ---- ゲートON 側（より特殊な分岐）----
                if k_lt_l:
                    # mark 行は (k-1, l-1)（k<l のとき）
                    mark1,mark2=k-1,l-1

                    if start_lt_l:
                        if start_lt_k:
                            # l==k+1 の特例で target を変える
                            target=0 if (not l_eq_kp1) else 4
                            #  0: SQBkBlBjrB  meta=(1,0,0) -> P1, future=off, next=1
                            #  4: SQBklBjrB   meta=(2,2,0) -> P3, future=off, next=2
                        else:
                            target=1
                            #  1: SQBlBjrB    meta=(2,1,0) -> P2, future=off, next=2
                    else:
                        target=2
                        #  2: SQBjrB      meta=(3,3,1) -> P4(jmark系), future=on, next=3
                else:
                    # k>=l のときは mark を入れ替える
                    mark1,mark2=l-1,k-1

                    if start_lt_k:
                        if start_lt_l:
                            # k==l+1 の特例で target を変える
                            target=5 if (not k_eq_lp1) else 7
                            #  5: SQBlBkBjrB  meta=(6,0,0) -> P1, future=off, next=6
                            #  7: SQBlkBjrB   meta=(2,2,0) -> P3, future=off, next=2
                        else:
                            target=6
                            #  6: SQBkBjrB    meta=(2,1,0) -> P2, future=off, next=2
                    else:
                        target=2
                        #  2: SQBjrB      meta=(3,3,1) -> P4(jmark系), future=on, next=3
            else:
                # ---- ゲートOFF 側（比較的単純な分岐）----
                if k_lt_l:
                    mark1,mark2=k-1,l-1
                    target=8 if (not l_eq_kp1) else 9
                    #  8: SQBjlBkBlBjrB meta=(0,4,1) -> P5, future=on, next=0
                    #  9: SQBjlBklBjrB  meta=(4,4,1) -> P5, future=on, next=4
                else:
                    mark1,mark2=l-1,k-1
                    target=10 if (not k_eq_lp1) else 11
                    # 10: SQBjlBlBkBjrB meta=(5,4,1) -> P5, future=on, next=5
                    # 11: SQBjlBlkBjrB  meta=(7,4,1) -> P5, future=on, next=7

        # --------------------------
        # case 2) j == N-3
        #   - 境界ケース（N-3 列を含む開始星座）
        #   - endmark = N-2
        # --------------------------
        elif j_eq_N3:
            endmark=N2

            if k_lt_l:
                mark1,mark2=k-1,l-1

                if start_lt_l:
                    if start_lt_k:
                        target=12 if (not l_eq_kp1) else 15
                        # 12: SQd2BkBlB  meta=(13,0,0) -> P1, future=off, next=13
                        # 15: SQd2BklB   meta=(14,2,0) -> P3, future=off, next=14
                    else:
                        # ここでは mark2 のみを設定（意図: 特殊パターン）
                        mark2=l-1
                        target=13
                        # 13: SQd2BlB    meta=(14,1,0) -> P2, future=off, next=14
                else:
                    target=14
                    # 14: SQd2B      meta=(14,5,1) -> P6(base系), future=on, next=14
                    #     ※dfs_iter: functionid==14 の特例（SQd2B は endmark 到達時の数え方が違う）
            else:
                mark1,mark2=l-1,k-1

                if start_lt_k:
                    if start_lt_l:
                        target=16 if (not k_eq_lp1) else 18
                        # 16: SQd2BlBkB  meta=(17,0,0) -> P1, future=off, next=17
                        # 18: SQd2BlkB   meta=(14,2,0) -> P3, future=off, next=14
                    else:
                        mark2=k-1
                        target=17
                        # 17: SQd2BkB    meta=(14,1,0) -> P2, future=off, next=14
                else:
                    target=14
                    # 14: SQd2B      meta=(14,5,1) -> P6(base系), future=on, next=14（dfs 特例あり）

        # --------------------------
        # case 3) j == N-2
        #   - さらに境界（N-2 列）
        # --------------------------
        elif j_eq_N2:
            if k_lt_l:
                endmark=N2
                if start_lt_l:
                    if start_lt_k:
                        mark1=k-1
                        if not l_eq_kp1:
                            mark2=l-1
                            target=19
                            # 19: SQd1BkBlB  meta=(20,0,0) -> P1, future=off, next=20
                        else:
                            target=22
                            # 22: SQd1BklB   meta=(21,2,0) -> P3, future=off, next=21
                    else:
                        mark2=l-1
                        target=20
                        # 20: SQd1BlB    meta=(21,1,0) -> P2, future=off, next=21
                        #     ※dfs_iter: functionid==20 のとき add1=1 特例（コメントの通り）
                else:
                    target=21
                    # 21: SQd1B      meta=(21,5,1) -> P6(base系), future=on, next=21
            else:
                if start_lt_k:
                    if start_lt_l:
                        if k<N2:
                            mark1,endmark=l-1,N2
                            if not k_eq_lp1:
                                mark2=k-1
                                target=23
                                # 23: SQd1BlBkB  meta=(25,0,0) -> P1, future=off, next=25
                            else:
                                target=24
                                # 24: SQd1BlkB   meta=(21,2,0) -> P3, future=off, next=21
                        else:
                            if l!=(N-3):
                                mark2,endmark=l-1,N-3
                                target=20
                                # 20: SQd1BlB    meta=(21,1,0) -> P2, future=off, next=21（add1 特例）
                            else:
                                endmark=N-4
                                target=21
                                # 21: SQd1B      meta=(21,5,1) -> P6(base系), future=on, next=21
                    else:
                        if k!=N2:
                            mark2,endmark=k-1,N2
                            target=25
                            # 25: SQd1BkB    meta=(21,1,0) -> P2, future=off, next=21
                        else:
                            endmark=N-3
                            target=21
                            # 21: SQd1B      meta=(21,5,1) -> P6(base系), future=on, next=21
                else:
                    endmark=N2
                    target=21
                    # 21: SQd1B      meta=(21,5,1) -> P6(base系), future=on, next=21

        # --------------------------
        # case 4) それ以外（j がさらに大きい等）
        #   - SQd0 系へ落ちる
        # --------------------------
        else:
            endmark=N2
            if start>k:
                target=26
                # 26: SQd0B     meta=(26,5,1) -> P6(base系), future=on, next=26
            else:
                mark1=k-1
                target=27
                # 27: SQd0BkB   meta=(26,0,0) -> P1, future=off, next=26

        # ----------------------------------------
        # SoA へ格納（t番目）
        #   row_arr[t] は start（探索開始行）
        #   ijkl_arr[t] は “開始星座 pack（下位20bit）”
        # ----------------------------------------
        soa.ld_arr[t]=ld
        soa.rd_arr[t]=rd
        soa.col_arr[t]=col
        soa.row_arr[t]=start
        soa.free_arr[t]=free
        soa.jmark_arr[t]=jmark
        soa.end_arr[t]=endmark
        soa.mark1_arr[t]=mark1
        soa.mark2_arr[t]=mark2
        soa.funcid_arr[t]=target
        soa.ijkl_arr[t]=ijkl

    # ----------------------------------------
    # w_arr（対称性重み 2/4/8）
    #   - この重みは「ユニーク解数 → トータル解数」への復元係数
    #   - 後段で results[t] *= w_arr[t] の形で使う
    # ----------------------------------------
    @par
    for t in range(m):
        w_arr[t]=symmetry(soa.ijkl_arr[t],N)

    return soa,w_arr

####################################################
#
# boundary classification diagnostics
#
####################################################

"""N24 境界分類診断: j の境界から大分類 ID を返す。"""
def bc_id(N:int,j:int)->int:
  if j<N-3:
    return 0   # B / normal
  if j==N-3:
    return 1   # SQd2
  if j==N-2:
    return 2   # SQd1
  return 3     # SQd0

"""N24 境界分類診断: 大分類名。"""
def bc_name(cid:int,N:int)->str:
  if cid==0:
    return f"B(j<{N-3})"
  if cid==1:
    return f"SQd2(j={N-3})"
  if cid==2:
    return f"SQd1(j={N-2})"
  return f"SQd0(j>{N-2})"

"""functionid 名。build_soa_for_range() の target と対応。"""
def fid_name(fid:int)->str:
  names:List[str]=[
    "SQBkBlBjrB","SQBlBjrB","SQBjrB","SQB",
    "SQBklBjrB","SQBlBkBjrB","SQBkBjrB","SQBlkBjrB",
    "SQBjlBkBlBjrB","SQBjlBklBjrB","SQBjlBlBkBjrB","SQBjlBlkBjrB",
    "SQd2BkBlB","SQd2BlB","SQd2B","SQd2BklB","SQd2BlBkB",
    "SQd2BkB","SQd2BlkB","SQd1BkBlB","SQd1BlB","SQd1B",
    "SQd1BklB","SQd1BlBkB","SQd1BlkB","SQd1BkB","SQd0B","SQd0BkB"
  ]
  if fid>=0 and fid<28:
    return names[fid]
  return "UNKNOWN"

"""大分類と functionid 範囲が一致しているか。"""
def bc_expected_fid(cid:int,fid:int)->bool:
  if cid==0:
    return fid>=0 and fid<=11
  if cid==1:
    return fid>=12 and fid<=18
  if cid==2:
    return fid>=19 and fid<=25
  if cid==3:
    return fid==26 or fid==27
  return False

"""大分類と endmark が概ね一致しているか。SQd1 は N-3/N-4 の境界終端を許す。"""
def bc_expected_endmark(N:int,cid:int,endmark:int)->bool:
  if cid==0:
    return endmark==N-2
  if cid==1:
    return endmark==N-2
  if cid==2:
    return endmark==N-2 or endmark==N-3 or endmark==N-4
  if cid==3:
    return endmark==N-2
  return False

"""DFS を走らせず、build_soa_for_range() の境界分類だけを集計する。"""
def diagnose_boundary_classification(N:int,constellations:List[Dict[str,int]])->None:
  m:int=len(constellations)
  if m==0:
    print(f"[bc-summary] N={N} constellations=0")
    return

  soa:TaskSoA=TaskSoA(m)
  w_arr:List[u64]=[u64(0)]*m
  soa,w_arr=build_soa_for_range(N,constellations,0,m,soa,w_arr)

  case_cnt:List[int]=[0]*4
  case_free0:List[int]=[0]*4
  case_w2:List[int]=[0]*4
  case_w4:List[int]=[0]*4
  case_w8:List[int]=[0]*4
  case_bad_fid:List[int]=[0]*4
  case_bad_end:List[int]=[0]*4
  fid_cnt:List[int]=[0]*28
  case_fid_cnt:List[int]=[0]*(4*28)
  case_start_cnt:List[int]=[0]*(4*(N+1))
  case_end_cnt:List[int]=[0]*(4*(N+1))

  bad_printed:int=0

  for idx in range(m):
    ijkl:int=soa.ijkl_arr[idx]
    j:int=getj(ijkl)
    cid:int=bc_id(N,j)
    fid:int=soa.funcid_arr[idx]
    endmark:int=soa.end_arr[idx]
    start:int=soa.row_arr[idx]

    case_cnt[cid]+=1

    if fid>=0 and fid<28:
      fid_cnt[fid]+=1
      case_fid_cnt[cid*28+fid]+=1
    else:
      case_bad_fid[cid]+=1

    if not bc_expected_fid(cid,fid):
      case_bad_fid[cid]+=1
      if bad_printed<20:
        print(f"[bc-error-fid] idx={idx} case={bc_name(cid,N)} i={geti(ijkl)} j={j} k={getk(ijkl)} l={getl(ijkl)} fid={fid} {fid_name(fid)} start={start} end={endmark}")
        bad_printed+=1

    if not bc_expected_endmark(N,cid,endmark):
      case_bad_end[cid]+=1
      if bad_printed<20:
        print(f"[bc-error-end] idx={idx} case={bc_name(cid,N)} i={geti(ijkl)} j={j} k={getk(ijkl)} l={getl(ijkl)} fid={fid} {fid_name(fid)} start={start} end={endmark}")
        bad_printed+=1

    if soa.free_arr[idx]==0:
      case_free0[cid]+=1

    w:int=int(w_arr[idx])
    if w==2:
      case_w2[cid]+=1
    elif w==4:
      case_w4[cid]+=1
    elif w==8:
      case_w8[cid]+=1

    if start>=0 and start<=N:
      case_start_cnt[cid*(N+1)+start]+=1
    if endmark>=0 and endmark<=N:
      case_end_cnt[cid*(N+1)+endmark]+=1

  print(f"[bc-summary] N={N} constellations={m} N-3={N-3} N-2={N-2} signature_prune_disabled={1 if DISABLE_CONSTELLATION_SIGNATURE_PRUNE else 0}")

  for cid in range(4):
    print(f"[bc-case] {bc_name(cid,N)} count={case_cnt[cid]} free0={case_free0[cid]} w2={case_w2[cid]} w4={case_w4[cid]} w8={case_w8[cid]} bad_fid={case_bad_fid[cid]} bad_end={case_bad_end[cid]}")

    line:str="[bc-start] " + bc_name(cid,N)
    for r in range(N+1):
      v:int=case_start_cnt[cid*(N+1)+r]
      if v>0:
        line += f" r{r}={v}"
    print(line)

    line="[bc-end]   " + bc_name(cid,N)
    for r in range(N+1):
      v:int=case_end_cnt[cid*(N+1)+r]
      if v>0:
        line += f" e{r}={v}"
    print(line)

    for fid in range(28):
      c:int=case_fid_cnt[cid*28+fid]
      if c>0:
        print(f"[bc-fid] {bc_name(cid,N)} fid={fid} {fid_name(fid)} count={c}")

  print("[bc-fid-total]")
  for fid in range(28):
    if fid_cnt[fid]>0:
      print(f"[bc-fid-total] fid={fid} {fid_name(fid)} count={fid_cnt[fid]}")

"""exec_solutions() 後に、境界分類別 / functionid 別の solutions 合計を出す。CPU向け。"""
def diagnose_solution_by_boundary(N:int,constellations:List[Dict[str,int]])->None:
  m:int=len(constellations)
  if m==0:
    print(f"[bc-sol-summary] N={N} constellations=0")
    return

  soa:TaskSoA=TaskSoA(m)
  w_arr:List[u64]=[u64(0)]*m
  soa,w_arr=build_soa_for_range(N,constellations,0,m,soa,w_arr)

  case_cnt:List[int]=[0]*4
  case_total:List[int]=[0]*4
  case_fid_cnt:List[int]=[0]*(4*28)
  case_fid_total:List[int]=[0]*(4*28)
  all_total:int=0

  for idx in range(m):
    ijkl:int=soa.ijkl_arr[idx]
    j:int=getj(ijkl)
    cid:int=bc_id(N,j)
    fid:int=soa.funcid_arr[idx]
    sol:int=constellations[idx]["solutions"]

    case_cnt[cid]+=1
    case_total[cid]+=sol
    all_total+=sol

    if fid>=0 and fid<28:
      case_fid_cnt[cid*28+fid]+=1
      case_fid_total[cid*28+fid]+=sol

  print(f"[bc-sol-summary] N={N} constellations={m} total={all_total}")
  for cid in range(4):
    print(f"[bc-sol-case] {bc_name(cid,N)} count={case_cnt[cid]} total={case_total[cid]}")
    for fid in range(28):
      c:int=case_fid_cnt[cid*28+fid]
      t:int=case_fid_total[cid*28+fid]
      if c>0 or t>0:
        print(f"[bc-sol-fid] {bc_name(cid,N)} fid={fid} {fid_name(fid)} count={c} total={t}")

"""76 の auto sort 方針。N=20/N=21 は sort_mode=9、それ以外は安全側で sort_mode=0。"""
def auto_sort_mode(N:int)->int:
  if N==20 or N==21:
    return 9
  return 0

"""cross_stripe_safe 用の chunk/range 検証。kernel ロジックには影響させない。"""
def validate_chunk_range(label:str,start:int,end:int,total:int)->bool:
  ok:bool=True
  if start<0:
    print(f"[cross-stripe-safe][error] {label}: start < 0 start={start} total={total}")
    ok=False
  if end<start:
    print(f"[cross-stripe-safe][error] {label}: end < start start={start} end={end} total={total}")
    ok=False
  if end>total:
    print(f"[cross-stripe-safe][error] {label}: end > total start={start} end={end} total={total}")
    ok=False
  if ok and start==end:
    print(f"[cross-stripe-safe][warn] {label}: empty range start={start} end={end} total={total}")
  return ok

"""reorder 件数の軽量検証。これは常時実行しても重くない。"""
def validate_reordered_count(label:str,expected:int,actual:int)->bool:
  if expected!=actual:
    print(f"[stripe-reorder][error] {label}: reordered count mismatch expected={expected} actual={actual}")
    return False
  return True

"""cross_stripe_safe/reorder-only 用の index permutation 検証。重複投入・欠落投入を検出する。"""
def validate_reordered_indices(label:str,expected:int,idxs:List[int])->bool:
  if not validate_reordered_count(label,expected,len(idxs)):
    return False
  seen:List[int]=[0]*expected
  for v in idxs:
    if v<0 or v>=expected:
      print(f"[cross-stripe-safe][error] {label}: index out of range idx={v} expected={expected}")
      return False
    if seen[v]!=0:
      print(f"[cross-stripe-safe][error] {label}: duplicated index idx={v}")
      return False
    seen[v]=1
  missing:int=0
  first_missing:int=-1
  for i in range(expected):
    if seen[i]==0:
      missing+=1
      if first_missing<0:
        first_missing=i
  if missing!=0:
    print(f"[cross-stripe-safe][error] {label}: missing count={missing} first_missing={first_missing}")
    return False
  return True

""" sort_mode=3 用の軽量 popcount。Codon / CPython 両方で動くよう int だけで実装。"""
def popcount_int(x:int)->int:
  c:int=0
  while x:
    x&=x-1
    c+=1
  return c

"""各 Constellation（部分盤面）ごとに最適分岐（functionid）を選び、`dfs()` で解数を取得。 結果は `solutions` に書き込み、最後に `symmetry()` の重みで補正する。前段で SoA 展開し 並列化区間のループ体を軽量化。"""
def exec_solutions(N:int,constellations:List[Dict[str,int]],use_gpu:bool,gpu_block:int=32,gpu_max_blocks:int=484,gpu_log_level:int=0,gpu_sort_mode:int=-1,cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT,reorder_only:bool=False,chunk_only:bool=False,debug_chunk_start:int=0,debug_chunk_count:int=1)->None:
  """
  機能:
    すべての constellation について DFS を実行し、各 constellation["solutions"] に
    「その constellation が代表する解数（対称性重み込み）」を格納します。

  処理の流れ:
    1) functionid のカテゴリ分けや meta テーブルを構築（分岐モードの定義）
    2) SoA を構築して（CPU なら @par、GPU なら kernel）で解数を列挙
    3) results / out を constellation 側へ書き戻す

  引数:
    N: 盤サイズ
    constellations: タスク（dict）のリスト
    use_gpu: True なら GPU 実行、False なら CPU 実行

  注意:
    - GPU は STEPS 件ずつ処理するため、投入回数と転送コストのトレードオフがあります。
    - CPU はホットパスを dfs_iter() に集約し、並列は @par に寄せています。
  """
  N1:int=N-1
  N2:int=N-2
  board_mask:int=(1<<N)-1

  # sort_mode auto:
  #   76 STABLE policy:
  #     N=20/N=21 は sort_mode=9（cross stripe only）を採用。
  #     N>=22 は従来どおり sort_mode=0。
  #   N=22 以降へ sort_mode=9 を自動展開せず、reorder-only/chunk-only で検証する。
  if gpu_sort_mode < 0:
    gpu_sort_mode = auto_sort_mode(N)

  FUNC_CATEGORY={
    # N-3
    "SQBkBlBjrB":3,"SQBlkBjrB":3,"SQBkBjrB":3,
    "SQd2BkBlB":3,"SQd2BkB":3,"SQd2BlkB":3,
    "SQd1BkBlB":3,"SQd1BlkB":3,"SQd1BkB":3,"SQd0BkB":3,
    # N-4
    "SQBklBjrB":4,"SQd2BklB":4,"SQd1BklB":4,
    # 0（上記以外）
    "SQBlBjrB":0,"SQBjrB":0,"SQB":0,"SQBlBkBjrB":0,
    "SQBjlBkBlBjrB":0,"SQBjlBklBjrB":0,"SQBjlBlBkBjrB":0,"SQBjlBlkBjrB":0,
    "SQd2BlB":0,"SQd2B":0,"SQd2BlBkB":0,
    "SQd1BlB":0,"SQd1B":0,"SQd1BlBkB":0,"SQd0B":0
  }
  FID={
    "SQBkBlBjrB":0,"SQBlBjrB":1,"SQBjrB":2,"SQB":3,
    "SQBklBjrB":4,"SQBlBkBjrB":5,"SQBkBjrB":6,"SQBlkBjrB":7,
    "SQBjlBkBlBjrB":8,"SQBjlBklBjrB":9,"SQBjlBlBkBjrB":10,"SQBjlBlkBjrB":11,
    "SQd2BkBlB":12,"SQd2BlB":13,"SQd2B":14,"SQd2BklB":15,"SQd2BlBkB":16,
    "SQd2BkB":17,"SQd2BlkB":18,"SQd1BkBlB":19,"SQd1BlB":20,"SQd1B":21,
    "SQd1BklB":22,"SQd1BlBkB":23,"SQd1BlkB":24,"SQd1BkB":25,"SQd0B":26,"SQd0BkB":27
  }

  # next_funcid, funcptn, availptn の3つだけ持つ
  func_meta=[
    (1,0,0),#  0 SQBkBlBjrB   -> P1, 先読みなし
    (2,1,0),#  1 SQBlBjrB     -> P2, 先読みなし
    (3,3,1),#  2 SQBjrB       -> P4, 先読みあり
    (3,5,1),#  3 SQB          -> P6, 先読みあり
    (2,2,0),#  4 SQBklBjrB    -> P3, 先読みなし
    (6,0,0),#  5 SQBlBkBjrB   -> P1, 先読みなし
    (2,1,0),#  6 SQBkBjrB     -> P2, 先読みなし
    (2,2,0),#  7 SQBlkBjrB    -> P3, 先読みなし
    (0,4,1),#  8 SQBjlBkBlBjrB-> P5, 先読みあり
    (4,4,1),#  9 SQBjlBklBjrB -> P5, 先読みあり
    (5,4,1),# 10 SQBjlBlBkBjrB-> P5, 先読みあり
    (7,4,1),# 11 SQBjlBlkBjrB -> P5, 先読みあり
    (13,0,0),# 12 SQd2BkBlB    -> P1, 先読みなし
    (14,1,0),# 13 SQd2BlB      -> P2, 先読みなし
    (14,5,1),# 14 SQd2B        -> P6, 先読みあり（avail 特例）
    (14,2,0),# 15 SQd2BklB     -> P3, 先読みなし
    (17,0,0),# 16 SQd2BlBkB    -> P1, 先読みなし
    (14,1,0),# 17 SQd2BkB      -> P2, 先読みなし
    (14,2,0),# 18 SQd2BlkB     -> P3, 先読みなし
    (20,0,0),# 19 SQd1BkBlB    -> P1, 先読みなし
    (21,1,0),# 20 SQd1BlB      -> P2, 先読みなし（add1=1 は dfs 内で特別扱い）
    (21,5,1),# 21 SQd1B        -> P6, 先読みあり
    (21,2,0),# 22 SQd1BklB     -> P3, 先読みなし
    (25,0,0),# 23 SQd1BlBkB    -> P1, 先読みなし
    (21,2,0),# 24 SQd1BlkB     -> P3, 先読みなし
    (21,1,0),# 25 SQd1BkB      -> P2, 先読みなし
    (26,5,1),# 26 SQd0B        -> P6, 先読みあり
    (26,0,0),# 27 SQd0BkB      -> P1, 先読みなし
  ]
  F=len(func_meta)
  funcptn_by_fid:List[int]=[0]*F
  for f,(nxt,ptn,aflag) in enumerate(func_meta):
      funcptn_by_fid[f]=ptn
  is_base=[0]*F   # ptn==5
  is_jmark=[0]*F   # ptn==3
  is_mark=[0]*F   # ptn in {0,1,2}

  mark_sel=[0]*F  # 0:none 1:mark1 2:mark2
  mark_step=[1]*F  # 1 or 2 or 3
  mark_add1=[0]*F  # 0/1
  for f,(nxt,ptn,aflag) in enumerate(func_meta):
      if ptn==5:
          is_base[f]=1
      elif ptn==3:
          is_jmark[f]=1
      elif ptn==0 or ptn==1 or ptn==2:
          is_mark[f]=1
          if ptn==1:
              mark_sel[f]=2
              mark_step[f]=2
              if f==20:
                  mark_add1[f]=1
          else:
              mark_sel[f]=1
              mark_step[f]=2 if ptn==0 else 3

  n3=1<<max(0,N-3)   # 念のため負シフト防止
  n4=1<<max(0,N-4)   # N3,N4とは違います
  # ===== 前処理ステージ（単一スレッド） =====
  m=len(constellations)
  # ===== GPU分割設定 =====
  # ===== GPU投入サイズを実行時に調整 =====
  # 19オリジナルの構造は維持し、block と chunk の大きさだけを変える診断版。
  # STEPS = block * max_blocks が 1回にGPUへ投げる constellation 数。
  BLOCK=gpu_block
  MAX_BLOCKS=gpu_max_blocks
  # 72 STABLE FINAL BENCH:
  #   54/55 の実測で、N=20 は 32x484 が最良。
  #   max_blocks<=0 は 54 実験版の 1 chunk 指定で大幅に遅くなるため、公開版では安全側に丸める。
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS=BLOCK*MAX_BLOCKS
  # STEPS = 24576 if use_gpu else m_all
  # STEPS=24576
  m_all=len(constellations)

  # w_pre: List[u64] = [u64(0)] * m_all
  # for i in range(m_all):
  #     w_pre[i] = u64(symmetry(constellations[i]["startijkl"], N))




  ##########
  # GPU
  ##########
  if use_gpu:
    # 72 STABLE FINAL BENCH:
    #   sort_mode=6 は元chunk 0..last を within 方向にストライプ化して大きく改善した。
    #   ただしログでは chunk7 付近に 19秒台の山が残った。
    #   原因候補は「元chunk間」だけでなく「within方向の重い帯」。
    #
    #   sort_mode=5: 旧 stripe + sort4（比較用。不採用寄り）
    #   sort_mode=6: 旧 stripe only（62 stable 候補）
    #   sort_mode=7: balanced stripe only。各出力chunkが within の residue をずらす。
    #   sort_mode=8: balanced stripe + sort4（比較用）
    #   sort_mode=9: cross stripe only。src_ch と within residue を直交させる。
    #   sort_mode=10: cross stripe + sort4（比較用）
    #
    #   解数は加算なので、direct_total では元 index への scatter は不要。
    stripe_chunks:bool=(gpu_sort_mode==5 or gpu_sort_mode==6 or gpu_sort_mode==7 or gpu_sort_mode==8 or gpu_sort_mode==9 or gpu_sort_mode==10)
    balanced_stripe:bool=(gpu_sort_mode==7 or gpu_sort_mode==8)
    cross_stripe:bool=(gpu_sort_mode==9 or gpu_sort_mode==10)
    local_sort_mode:int=gpu_sort_mode
    work_constellations:List[Dict[str,int]]=constellations
    if reorder_only and not stripe_chunks:
      print(f"[reorder-only] N={N} sort_mode={gpu_sort_mode} stripe=0 original={m_all} steps={STEPS} ok")
      if m_all>0:
        constellations[0]["solutions"]=0
      return
    if stripe_chunks:
      n_chunks_est:int=(m_all + STEPS - 1)//STEPS
      reordered:List[Dict[str,int]]=[]
      validate_reorder:bool=(cross_stripe_safe or reorder_only)
      reordered_idx:List[int]=[]
      if cross_stripe:
        # 74 SAFE CROSS STRIPE FIX:
        #   73 の式は slot から src_ch/base を同時に作っていたため、
        #   STEPS % n_chunks_est != 0 の場合に一部 src_ch の末尾 within が欠落した。
        #   N=21/32x484 では STEPS=15488, n_chunks_est=13, STEPS%13=5 となり、
        #   full chunk の末尾 5 件が複数 chunk で落ち、reordered が 35 件不足した。
        #
        #   この版では out_ch/base/src_ch を独立に回し、
        #   任意の idx=(src_ch, within) が out_ch=(within%n_chunks_est-src_ch)%n_chunks_est
        #   でちょうど一度だけ出るようにする。kernel ロジックは変更しない。
        out_ch:int=0
        while out_ch<n_chunks_est:
          base:int=0
          while base*n_chunks_est<STEPS:
            src_ch:int=0
            while src_ch<n_chunks_est:
              rem:int=(src_ch + out_ch) % n_chunks_est
              within:int=base*n_chunks_est + rem
              if within<STEPS:
                idx:int=src_ch*STEPS+within
                if idx<m_all:
                  reordered.append(constellations[idx])
                  if validate_reorder:
                    reordered_idx.append(idx)
              src_ch+=1
            base+=1
          out_ch+=1
      elif balanced_stripe:
        # 出力chunkごとに、within を 0..STEPS-1 全域から拾う。
        # 旧 stripe は出力chunkごとに within の帯が残り、chunk7 が重くなりやすかった。
        out_ch:int=0
        while out_ch<n_chunks_est:
          slot:int=0
          while slot<STEPS:
            src_ch:int=slot % n_chunks_est
            base:int=slot // n_chunks_est
            within:int=(out_ch + base*n_chunks_est) % STEPS
            idx:int=src_ch*STEPS+within
            if idx<m_all:
              reordered.append(constellations[idx])
              if validate_reorder:
                reordered_idx.append(idx)
            slot+=1
          out_ch+=1
      else:
        within:int=0
        while within<STEPS:
          ch:int=0
          while ch<n_chunks_est:
            idx:int=ch*STEPS+within
            if idx<m_all:
              reordered.append(constellations[idx])
              if validate_reorder:
                reordered_idx.append(idx)
            ch+=1
          within+=1
      work_constellations=reordered
      if not validate_reordered_count("stripe_reorder",m_all,len(work_constellations)):
        return
      if validate_reorder:
        if not validate_reordered_indices("stripe_reorder",m_all,reordered_idx):
          return
      if reorder_only:
        print(f"[reorder-only] N={N} sort_mode={gpu_sort_mode} original={m_all} reordered={len(work_constellations)} chunks={n_chunks_est} steps={STEPS} ok")
        if m_all>0:
          constellations[0]["solutions"]=0
        return
      if gpu_sort_mode==5 or gpu_sort_mode==8 or gpu_sort_mode==10:
        local_sort_mode=4
      else:
        local_sort_mode=0
    if gpu_log_level>=1:
      print(f"[gpu-config] N={N} original=1 mixed32=1 hotpath=1 trunk75={BLOCK}x{MAX_BLOCKS} sort_mode={gpu_sort_mode} local_sort={local_sort_mode} stripe={1 if stripe_chunks else 0} balanced={1 if balanced_stripe else 0} cross={1 if cross_stripe else 0} cross_safe={1 if cross_stripe_safe else 0} reorder_only={1 if reorder_only else 0} chunk_only={1 if chunk_only else 0} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} log_level={gpu_log_level}")
    soa:TaskSoA=TaskSoA(STEPS)
    results:List[u64]=[u64(0)]*STEPS
    # 60 DIRECT TOTAL: GPU結果は constellation ごとに書き戻さず、chunkごとに合計する。
    # main() の sum(c["solutions"]) 互換のため、最後に constellations[0]["solutions"] だけへ合計値を入れる。
    gpu_total:int=0
    w_arr:List[u64]=[u64(0)]*STEPS

    # sort_mode > 0 は「kernelを増やさず、chunk内だけ並び替える」診断。
    # 0: そのまま
    # 1: functionid順
    # 2: funcptn順
    # 3: funcptn + work bucket順（free popcount と depth を粗く見る）
    # 29のbucket化は複数kernel化で遅くなったため、単一chunk内の順序だけを変える。
    sort_soa:TaskSoA=TaskSoA(STEPS)
    sort_w_arr:List[u64]=[u64(0)]*STEPS
    # direct total では元indexへのscatterが不要。sort後の順序でも合計値は不変。
    order:List[int]=[0]*STEPS

    meta_next: List[u8] = [ u8(1),u8(2),u8(3),u8(3),u8(2),u8(6),u8(2),u8(2), u8(0),u8(4),u8(5),u8(7),u8(13),u8(14),u8(14),u8(14), u8(17),u8(14),u8(14),u8(20),u8(21),u8(21),u8(21),u8(25), u8(21),u8(21),u8(26),u8(26) ]
    # ===== STEPS件ずつ処理 =====
    off = 0
    # u8 の 28要素デバイス配列を用意
    # meta_next = ( [1,2,3,3,2,6,2,2,0,4,5,7,13,14,14,14,17,14,14,20,21,21,21,25,21,21,26,26])
    n3 = 1 << (N - 3)
    n4 = 1 << (N - 4)
    chunks:int=0
    executed_chunks:int=0
    if chunk_only:
      if debug_chunk_start<0:
        debug_chunk_start=0
      if debug_chunk_count<=0:
        debug_chunk_count=1
      if gpu_log_level>=1:
        print(f"[chunk-only] mode=7 executes selected chunks only; start={debug_chunk_start} count={debug_chunk_count}")
    while off < m_all:
      m = min(STEPS, m_all - off)
      chunk_index:int=chunks
      chunks+=1
      if chunk_only:
        run_this_chunk:bool=(chunk_index>=debug_chunk_start and chunk_index<debug_chunk_start+debug_chunk_count)
        if not run_this_chunk:
          if gpu_log_level>=2:
            print(f"[gpu-chunk-skip] N={N} chunk={chunk_index} off={off} m={m}")
          off += m
          continue
      executed_chunks+=1
      if cross_stripe_safe:
        if not validate_chunk_range("gpu_chunk",off,off+m,m_all):
          return
      if gpu_log_level>=2:
        t0=datetime.now()
      # 戻り値を使わないので破棄
      build_soa_for_range(N,work_constellations, off, m,soa,w_arr)
      if gpu_log_level>=2:
        t1=datetime.now()
      # sort_mode は chunk 内だけを stable bucket sort する。
      # kernel数は増やさないので、29のような bucket 複数起動のオーバーヘッドを避ける。
      use_sorted:bool=(local_sort_mode==1 or local_sort_mode==2 or local_sort_mode==3 or local_sort_mode==4)
      if gpu_log_level>=2:
        ts0=datetime.now()
      if use_sorted:
        nb:int=28
        if local_sort_mode==2:
          nb=6
        if local_sort_mode==3:
          nb=24  # funcptn 6種 x work bucket 4種
        if local_sort_mode==4:
          nb=48  # funcptn 6種 x work bucket 8種
        counts:List[int]=[0]*64
        pos:List[int]=[0]*64
        cur:List[int]=[0]*64
        for i in range(m):
          fid0:int=soa.funcid_arr[i]
          key:int=fid0
          if local_sort_mode==2:
            key=funcptn_by_fid[fid0]
          if local_sort_mode==3:
            ptn:int=funcptn_by_fid[fid0]
            depth:int=soa.end_arr[i]-soa.row_arr[i]
            if depth<0:
              depth=0
            pc:int=popcount_int(soa.free_arr[i])
            wb:int=0
            if pc>=3:
              wb+=1
            if depth>=12:
              wb+=2
            key=ptn*4+wb
          if local_sort_mode==4:
            ptn:int=funcptn_by_fid[fid0]
            depth:int=soa.end_arr[i]-soa.row_arr[i]
            if depth<0:
              depth=0
            pc:int=popcount_int(soa.free_arr[i])
            wb:int=0
            if pc>=2:
              wb+=1
            if pc>=4:
              wb+=2
            if depth>=10:
              wb+=4
            key=ptn*8+wb
          counts[key]+=1
        run:int=0
        for b in range(nb):
          pos[b]=run
          cur[b]=run
          run+=counts[b]
        for i in range(m):
          fid0:int=soa.funcid_arr[i]
          key:int=fid0
          if local_sort_mode==2:
            key=funcptn_by_fid[fid0]
          if local_sort_mode==3:
            ptn:int=funcptn_by_fid[fid0]
            depth:int=soa.end_arr[i]-soa.row_arr[i]
            if depth<0:
              depth=0
            pc:int=popcount_int(soa.free_arr[i])
            wb:int=0
            if pc>=3:
              wb+=1
            if depth>=12:
              wb+=2
            key=ptn*4+wb
          if local_sort_mode==4:
            ptn:int=funcptn_by_fid[fid0]
            depth:int=soa.end_arr[i]-soa.row_arr[i]
            if depth<0:
              depth=0
            pc:int=popcount_int(soa.free_arr[i])
            wb:int=0
            if pc>=2:
              wb+=1
            if pc>=4:
              wb+=2
            if depth>=10:
              wb+=4
            key=ptn*8+wb
          p:int=cur[key]
          order[p]=i
          cur[key]+=1
        for p in range(m):
          q:int=order[p]
          sort_soa.ld_arr[p]=soa.ld_arr[q]
          sort_soa.rd_arr[p]=soa.rd_arr[q]
          sort_soa.col_arr[p]=soa.col_arr[q]
          sort_soa.row_arr[p]=soa.row_arr[q]
          sort_soa.free_arr[p]=soa.free_arr[q]
          sort_soa.jmark_arr[p]=soa.jmark_arr[q]
          sort_soa.end_arr[p]=soa.end_arr[q]
          sort_soa.mark1_arr[p]=soa.mark1_arr[q]
          sort_soa.mark2_arr[p]=soa.mark2_arr[q]
          sort_soa.funcid_arr[p]=soa.funcid_arr[q]
          sort_soa.ijkl_arr[p]=soa.ijkl_arr[q]
          sort_w_arr[p]=w_arr[q]
      if gpu_log_level>=2:
        ts1=datetime.now()
      GRID = (m + BLOCK - 1) // BLOCK

      # 81 GPU RESTORE:
      #   80 では kernel_dfs_iter_gpu() 呼び出しがコメントアウトされたままになっていたため、
      #   results[] が初期値 0 のまま合計され、GPU total が常に 0 になっていた。
      #   ここで sort 有無に応じて実際に GPU kernel を起動する。
      if use_sorted:
        kernel_dfs_iter_gpu(
          gpu.raw(sort_soa.ld_arr), gpu.raw(sort_soa.rd_arr), gpu.raw(sort_soa.col_arr),
          gpu.raw(sort_soa.row_arr), gpu.raw(sort_soa.free_arr),
          gpu.raw(sort_soa.jmark_arr), gpu.raw(sort_soa.end_arr),
          gpu.raw(sort_soa.mark1_arr), gpu.raw(sort_soa.mark2_arr),
          gpu.raw(sort_soa.funcid_arr), gpu.raw(sort_w_arr),
          gpu.raw(meta_next),
          gpu.raw(results),
          m, board_mask,
          n3, n4,
          grid=GRID, block=BLOCK
        )
      else:
        kernel_dfs_iter_gpu(
          gpu.raw(soa.ld_arr), gpu.raw(soa.rd_arr), gpu.raw(soa.col_arr),
          gpu.raw(soa.row_arr), gpu.raw(soa.free_arr),
          gpu.raw(soa.jmark_arr), gpu.raw(soa.end_arr),
          gpu.raw(soa.mark1_arr), gpu.raw(soa.mark2_arr),
          gpu.raw(soa.funcid_arr), gpu.raw(w_arr),
          gpu.raw(meta_next),
          gpu.raw(results),
          m, board_mask,
          n3, n4,
          grid=GRID, block=BLOCK
        )
      if gpu_log_level>=2:
        t2=datetime.now()
      # 60 DIRECT TOTAL:
      # 56/58/59 は results_all へ scatter し、最後に全 constellation へ書き戻してから
      # main() で sum() していた。ベンチ用途では個別 solutions は不要なので、
      # GPU結果をここで直接合計し、scatter/copy-back/final write を省く。
      chunk_total:int=0
      for i in range(m):
        chunk_total += int(results[i])
      gpu_total += chunk_total
      if gpu_log_level>=2:
        t3=datetime.now()
        print(f"[gpu-chunk] N={N} chunk={chunk_index} off={off} m={m} grid={GRID} sort={gpu_sort_mode}/{local_sort_mode} build={str(t1-t0)[:-3]} sort_time={str(ts1-ts0)[:-3]} kernel={str(t2-ts1)[:-3]} sum={str(t3-t2)[:-3]} partial_total={chunk_total}")
      off += m

    if m_all>0:
      constellations[0]["solutions"] = gpu_total
    if gpu_log_level>=1:
      print(f"[gpu-summary] N={N} constellations={m_all} chunks={chunks} executed_chunks={executed_chunks} direct_total=1 stripe={1 if stripe_chunks else 0} balanced={1 if balanced_stripe else 0} cross={1 if cross_stripe else 0} cross_safe={1 if cross_stripe_safe else 0} chunk_only={1 if chunk_only else 0}")
    return
  ##########
  # CPU 
  ##########
  else:
    soa:TaskSoA = TaskSoA(m_all)
    results: List[u64] = [u64(0)] * m_all
    results_all: List[u64] = [u64(0)] * m_all
    w_arr: List[u64] = [u64(0)] * m_all

    size=max(FID.values())+1
    blockK_by_funcid=[0]*size
    blockL_by_funcid=[0,1,0,0,1,1,0,2,0,0,0,0,0,1,0,1,1,0,2,0,0,0,1,1,2,0,0,0]
    for fn,cat in FUNC_CATEGORY.items():# FUNC_CATEGORY: {関数名: 3 or 4 or 0}
      fid=FID[fn]
      blockK_by_funcid[fid]=n3 if cat==3 else (n4 if cat==4 else 0)

    m_all = len(constellations) # CPUは全件を1回で SoA + w_arr を作る（これがないと壊れる）
    if m_all == 0:
      return
    soa, w_arr = build_soa_for_range(N,constellations, 0, m_all, soa, w_arr)
    results:List[u64] = [u64(0)] * m_all
    @par
    for i in range(m_all):
      """ CPU版は dfs_iter を使う（dfs_iter は再帰なしで、functionid ごとの分岐も内包する形で実装） """
      use_itter = True
      # 2024-06-08 時点では dfs_iter の方が速い（理由は不明。dfs_iter は再帰なしで分岐も内包しているので、CPUの分岐予測や関数呼び出しコストが効いている可能性がある）。 dfs_iter を使うと全体で約3秒程度速くなっている。  
      # use_itter = True
      # 18:         666090624                0          0:00:31.516    ok
      # use_itter = False
      # 18:         666090624                0          0:00:35.136    ok
      if use_itter:
        cnt:u64 = dfs_iter(
            func_meta,
            blockK_by_funcid,blockL_by_funcid,
            board_mask,
            soa.funcid_arr[i],
            soa.ld_arr[i], soa.rd_arr[i], soa.col_arr[i], 
            soa.row_arr[i],soa.free_arr[i], 
            soa.jmark_arr[i], soa.end_arr[i],
            soa.mark1_arr[i], soa.mark2_arr[i])
      else:
        cnt:u64 = dfs(
            func_meta,
            blockK_by_funcid,blockL_by_funcid,
            board_mask,
            soa.funcid_arr[i],
            soa.ld_arr[i], soa.rd_arr[i], soa.col_arr[i],
            soa.row_arr[i],soa.free_arr[i], 
            soa.jmark_arr[i], soa.end_arr[i],
            soa.mark1_arr[i], soa.mark2_arr[i])
      results[i]=cnt*w_arr[i]
  ##########
  # 集計（CPUのみ。GPUは direct_total で上で return 済み）
  ##########
  out = results
  for i, constellation in enumerate(constellations):
    constellation["solutions"] = int(out[i])

####################################################
#
# utility
#
####################################################

""" splitmix64 ミキサ最終段 """
def mix64(x:u64)->u64:
  x=(x^(x>>u64(30)))*u64(0xBF58476D1CE4E5B9)
  x=(x^(x>>u64(27)))*u64(0x94D049BB133111EB)
  x^=(x>>u64(31))
  return x

""" Zobrist テーブル用乱数リスト生成 """
def gen_list(cnt:int,seed:u64)->List[u64]:
  out:List[u64]=[]
  s:u64=seed
  # _mix64=self.mix64
  for _ in range(cnt):
    s=s+u64(0x9E3779B97F4A7C15)
    out.append(mix64(s))
  return out

""" Zobrist テーブル初期化 """
# def init_zobrist(N:int)->None:
def init_zobrist(N:int,zobrist_hash_tables: Dict[int, Dict[str, List[u64]]])->Dict[str,List[u64]]:
  if N in zobrist_hash_tables:
    return zobrist_hash_tables[N]
  base_seed:u64=(u64(0xC0D0_0000_0000_0000)^(u64(N)<<u64(32)))
  tbl:Dict[str,List[u64]]={
    'ld':gen_list(N,base_seed^u64(0x01)),
    'rd':gen_list(N,base_seed^u64(0x02)),
    'col':gen_list(N,base_seed^u64(0x03)),
    'LD':gen_list(N,base_seed^u64(0x04)),
    'RD':gen_list(N,base_seed^u64(0x05)),
    'row':gen_list(N,base_seed^u64(0x06)),
    'queens':gen_list(N,base_seed^u64(0x07)),
    'k':gen_list(N,base_seed^u64(0x08)),
    'l':gen_list(N,base_seed^u64(0x09)),
  }
  """ キャッシュ保存 """
  zobrist_hash_tables[N]=tbl
  return tbl

""" Zobrist Hash を用いた盤面の 64bit ハッシュ値生成  """
def zobrist_hash(N:int, ld: int, rd: int, col: int, row: int, queens: int, k: int, l: int, LD: int, RD: int,zobrist_hash_tables:Dict[int, Dict[str, List[u64]]]) -> u64:
  tbl: Dict[str, List[u64]] = init_zobrist(N,zobrist_hash_tables)

  # ここでテーブルが u64 で作られている前提（init_zobrist側も u64 に）
  ld_tbl  = tbl["ld"]    # List[u64]
  rd_tbl  = tbl["rd"]    # List[u64]
  col_tbl = tbl["col"]   # List[u64]
  LD_tbl  = tbl["LD"]    # List[u64]
  RD_tbl  = tbl["RD"]    # List[u64]
  row_tbl = tbl["row"]   # List[u64]
  q_tbl   = tbl["queens"]# List[u64]
  k_tbl   = tbl["k"]     # List[u64]
  l_tbl   = tbl["l"]     # List[u64]

  # mask は u64 で作る（1<<N が int のままだと型が揺れやすい）
  mask: u64 = (u64(1) << u64(N)) - u64(1)

  # 入力ビット集合を u64 に揃えてマスク
  ld64: u64  = u64(ld)  & mask
  rd64: u64  = u64(rd)  & mask
  col64: u64 = u64(col) & mask
  LD64: u64  = u64(LD)  & mask
  RD64: u64  = u64(RD)  & mask

  h: u64 = u64(0)

  m: u64 = ld64
  i: int = 0
  while i < N:
    if (m & u64(1)) != u64(0):
      h ^= u64(ld_tbl[i])
    m >>= u64(1)
    i += 1

  m = rd64; i = 0
  while i < N:
    if (m & u64(1)) != u64(0):
      h ^= u64(rd_tbl[i])
    m >>= u64(1)
    i += 1

  m = col64; i = 0
  while i < N:
    if (m & u64(1)) != u64(0):
      h ^= u64(col_tbl[i])
    m >>= u64(1)
    i += 1

  m = LD64; i = 0
  while i < N:
    if (m & u64(1)) != u64(0):
      h ^= u64(LD_tbl[i])
    m >>= u64(1)
    i += 1

  m = RD64; i = 0
  while i < N:
    if (m & u64(1)) != u64(0):
      h ^= u64(RD_tbl[i])
    m >>= u64(1)
    i += 1

  if 0 <= row < N:
    h ^= u64(row_tbl[row])
  if 0 <= queens < N:
    h ^= u64(q_tbl[queens])
  if 0 <= k < N:
    h ^= u64(k_tbl[k])
  if 0 <= l < N:
    h ^= u64(l_tbl[l])

  return h

"""(i,j,k,l) を 5bit×4=20bit にパック/アンパックするユーティリティ。 mirvert は上下ミラー（行: N-1-?）＋ (k,l) の入れ替えで左右ミラー相当を実現。"""
def to_ijkl(i:int,j:int,k:int,l:int)->int:return (i<<15)+(j<<10)+(k<<5)+l
def mirvert(ijkl:int,N:int)->int:return to_ijkl(N-1-geti(ijkl),N-1-getj(ijkl),getl(ijkl),getk(ijkl))
def ffmin(a:int,b:int)->int:return min(a,b)
def geti(ijkl:int)->int:return (ijkl>>15)&0x1F
def getj(ijkl:int)->int:return (ijkl>>10)&0x1F
def getk(ijkl:int)->int:return (ijkl>>5)&0x1F
def getl(ijkl:int)->int:return ijkl&0x1F

"""(i,j,k,l) パック値に対して盤面 90°/180° 回転を適用した新しいパック値を返す。 回転の定義: (r,c) -> (c, N-1-r)。対称性チェック・正規化に利用。"""
def rot90(ijkl:int,N:int)->int:return ((N-1-getk(ijkl))<<15)+((N-1-getl(ijkl))<<10)+(getj(ijkl)<<5)+geti(ijkl)
def rot180(ijkl:int,N:int)->int:return ((N-1-getj(ijkl))<<15)+((N-1-geti(ijkl))<<10)+((N-1-getl(ijkl))<<5)+(N-1-getk(ijkl))
def symmetry(ijkl:int,N:int)->u64:return u64(2) if symmetry90(ijkl,N) else u64(4) if geti(ijkl)==N-1-getj(ijkl) and getk(ijkl)==N-1-getl(ijkl) else u64(8)
def symmetry90(ijkl:int,N:int)->bool:return ((geti(ijkl)<<15)+(getj(ijkl)<<10)+(getk(ijkl)<<5)+getl(ijkl))==(((N-1-getk(ijkl))<<15)+((N-1-getl(ijkl))<<10)+(getj(ijkl)<<5)+geti(ijkl))

"""与えた (i,j,k,l) の 90/180/270° 回転形が既出集合 ijkl_list に含まれるかを判定する。"""
def check_rotations(ijkl_list:Set[int],i:int,j:int,k:int,l:int,N:int)->bool:
  return any(rot in ijkl_list for rot in [((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i,((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+(N-1-k),(l<<15)+(k<<10)+((N-1-i)<<5)+(N-1-j)])

""" キャッシュ付き Jasmin 正規化ラッパー """
jasmin_cache_global:Dict[Tuple[int,int],int]={}

def get_jasmin(N:int,c:int)->int:
  """ Jasmin 正規化のキャッシュ付ラッパ。盤面パック値 c を回転/ミラーで規約化した代表値を返す。
  59 PREBUILD LIGHT:
    旧版は関数内で jasmin_cache を毎回作っていたため、実質キャッシュになっていなかった。
    グローバル辞書へ逃がし、同一 N / 同一 ijkl の再計算を避ける。
  """
  key=(c,N)
  if key in jasmin_cache_global:
    return jasmin_cache_global[key]
  result=jasmin(c,N)
  jasmin_cache_global[key]=result
  return result

""" Jasmin 正規化。盤面パック値 ijkl を回転/ミラーで規約化した代表値を返す。"""
def jasmin(ijkl:int,N:int)->int:
  # 最初の最小値と引数を設定
  arg=0
  min_val=ffmin(getj(ijkl),N-1-getj(ijkl))
  # i: 最初の行（上端） 90度回転2回
  if ffmin(geti(ijkl),N-1-geti(ijkl))<min_val:
    arg=2
    min_val=ffmin(geti(ijkl),N-1-geti(ijkl))
  # k: 最初の列（左端） 90度回転3回
  if ffmin(getk(ijkl),N-1-getk(ijkl))<min_val:
    arg=3
    min_val=ffmin(getk(ijkl),N-1-getk(ijkl))
  # l: 最後の列（右端） 90度回転1回
  if ffmin(getl(ijkl),N-1-getl(ijkl))<min_val:
    arg=1
    min_val=ffmin(getl(ijkl),N-1-getl(ijkl))
  # 90度回転を arg 回繰り返す
  _rot90=rot90
  for _ in range(arg):
    # ijkl=rot90(ijkl,N)
    ijkl=_rot90(ijkl,N)
  # 必要に応じて垂直方向のミラーリングを実行
  if getj(ijkl)<N-1-getj(ijkl):
    ijkl=mirvert(ijkl,N)
  return ijkl

####################################################
#
# cache
#
####################################################

"""サブコンステレーション生成のキャッシュ付ラッパ。StateKey で一意化し、 同一状態での重複再帰を回避して生成量を抑制する。"""
def set_pre_queens_cached(
  N:int,
  ijkl_list:Set[int],
  subconst_cache:set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],
  ld:int, rd:int, col:int,
  k:int, l:int,
  row:int, queens:int,
  LD:int, RD:int,
  counter:List[int],
  constellations:List[Dict[str,int]],
  preset_queens:int,
  visited:Set[int],
  constellation_signatures:Set[Tuple[int,int,int,int,int,int]],
  zobrist_hash_tables: Dict[int, Dict[str, List[u64]]]
)->Tuple[Set[int], Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]], List[Dict[str,int]], int]:
  """
  機能:
    `set_pre_queens()` の“入口”にキャッシュを付け、
    同じ (ld,rd,col,k,l,row,queens,LD,RD,N,preset_queens) の状態からの
    サブコンステレーション生成を重複実行しないようにする。

  引数:
    N / preset_queens:
      キャッシュキーに含める（同じ ld/rd/col でも N や preset が違えば別問題）。
    ijkl_list:
      生成過程で参照・更新される開始星座集合（必要なら追加されうる）。
    subconst_cache:
      “この実行内”での重複抑止集合。key が存在する場合は何もせず戻る。
    ld,rd,col,k,l,row,queens,LD,RD:
      set_pre_queens に渡す状態。
    counter/constellations:
      set_pre_queens が constellation を append するための出力先。
    visited/constellation_signatures/zobrist_hash_tables:
      set_pre_queens 内部の枝刈り・重複排除用。

  返り値:
    (ijkl_list, subconst_cache, constellations, preset_queens)
    ※現行の上位呼び出し側の受けを崩さないためにこの形に揃える。

  実装上のコツ:
    - “キャッシュ登録してから本体呼び出し”にすることで、
      並行再入（同一状態からの重複突入）も抑止できる設計。
  """

  # ------------------------------------------------------------
  # 80 FIX: preset>=7 multiplicity preservation
  #
  # subconst_cache is useful as a recursion de-duplication guard for
  # preset<=6, but with preset=7 distinct pre-queen histories can reach
  # the same (ld,rd,col,k,l,row,queens,LD,RD,N,preset) state.
  # Those histories must still be counted with multiplicity.
  #
  # If we suppress the later hit, the emitted constellation task is lost.
  # In N=18 / preset=7 this appears as the SQd0 residual -21,024.
  # Therefore preset>=7 bypasses this cache and lets identical terminal
  # tasks be appended multiple times.
  # ------------------------------------------------------------
  if preset_queens>=7:
    ijkl_list, subconst_cache, constellations, preset_queens = set_pre_queens(
      N, ijkl_list, subconst_cache,
      ld, rd, col,
      k, l,
      row, queens,
      LD, RD,
      counter, constellations, preset_queens,
      visited, constellation_signatures,
      zobrist_hash_tables
    )
    return ijkl_list, subconst_cache, constellations, preset_queens

  # ---- キャッシュキー（状態を丸ごと）----
  # NOTE: queens や row も含めるので「途中段の重複」も止められる。
  key:Tuple[int,int,int,int,int,int,int,int,int,int,int] = (
    ld, rd, col, k, l, row, queens, LD, RD, N, preset_queens
  )

  # ---- 既にこの状態から展開済みなら何もしない ----
  if key in subconst_cache:
    return ijkl_list, subconst_cache, constellations, preset_queens

  # ---- 先に登録（再入・並列時の二重実行も抑止）----
  subconst_cache.add(key)

  # ---- 新規実行：本体へ ----
  ijkl_list, subconst_cache, constellations, preset_queens = set_pre_queens(
    N, ijkl_list, subconst_cache,
    ld, rd, col,
    k, l,
    row, queens,
    LD, RD,
    counter, constellations, preset_queens,
    visited, constellation_signatures,
    zobrist_hash_tables
  )
  return ijkl_list, subconst_cache, constellations, preset_queens

""" zorbist hash を使った visited pruning を有効にするか（constellations 内の状態をハッシュして重複排除）。効果はケースバイケースで、キャッシュの方が安定して速い可能性もある。"""
use_visited_prune = False  
"""事前に置く行 (k,l) を強制しつつ、queens==preset_queens に到達するまで再帰列挙。 `visited` には軽量な `state_hash` を入れて枝刈り。到達時は {ld,rd,col,startijkl} を constellation に追加。"""
def set_pre_queens(N:int,ijkl_list:Set[int],subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],ld:int,rd:int,col:int,k:int,l:int,row:int,queens:int,LD:int,RD:int,counter:list,constellations:List[Dict[str,int]],preset_queens:int,visited:Set[int],constellation_signatures:Set[Tuple[int,int,int,int,int,int]],zobrist_hash_tables: Dict[int, Dict[str, List[u64]]])->Tuple[Set[int], Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]], List[Dict[str,int]], int]:
  # mask = nq_get(N)._board_mask
  board_mask= (1<<N)-1
  # ---------------------------------------------------------------------
  # 状態ハッシュによる探索枝の枝刈り バックトラック系の冒頭に追加　やりすぎると解が合わない
  #
  # <>zobrist_hash
  # 各ビットを見てテーブルから XOR するため O(N)（ld/rd/col/LD/RDそれぞれで最大 N 回）。
  # とはいえ N≤17 なのでコストは小さめ。衝突耐性は高い。
  # マスク漏れや負数の扱いを誤ると不一致が起きる点に注意（先ほどの & ((1<<N)-1) 修正で解決）。
  # zobrist_tables: Dict[int, Dict[str, List[int]]] = {}
  # 59 PREBUILD LIGHT:
  # use_visited_prune=False の通常運用では Zobrist hash は使わない。
  # 旧版は False でも毎回 O(N) の zobrist_hash() を計算していたため、
  # constellation 生成の前処理で無駄が出ていた。
  if use_visited_prune:
    h: int = int(zobrist_hash(N,ld & board_mask, rd & board_mask, col & board_mask, row, queens, k, l, LD & board_mask, RD & board_mask,zobrist_hash_tables))
    if h in visited:
      return ijkl_list, subconst_cache, constellations, preset_queens
    visited.add(h)

  #
  # ---------------------------------------------------------------------
  # k行とl行はスキップ
  if row==k or row==l:
    ijkl_list, subconst_cache, constellations, preset_queens = set_pre_queens_cached(N,ijkl_list,subconst_cache,ld<<1,rd>>1,col,k,l,row+1,queens,LD,RD,counter,constellations,preset_queens,visited,constellation_signatures,zobrist_hash_tables)
    return ijkl_list, subconst_cache, constellations, preset_queens
  # クイーンの数がpreset_queensに達した場合、現在の状態を保存
  if queens == preset_queens:
    if (not DISABLE_CONSTELLATION_SIGNATURE_PRUNE) and preset_queens <= 5:
      sig = (ld, rd, col, k, l, row)    # これが signature (tuple)
      if sig in constellation_signatures:
        return ijkl_list, subconst_cache, constellations, preset_queens
      constellation_signatures.add(sig)
    constellation={"ld":ld,"rd":rd,"col":col,"startijkl":row<<20,"solutions":0}
    constellations.append(constellation) #星座データ追加
    counter[0]+=1
    return ijkl_list, subconst_cache, constellations, preset_queens
  # 現在の行にクイーンを配置できる位置を計算
  free=~(ld|rd|col|(LD>>(N-1-row))|(RD<<(N-1-row)))&board_mask
  # _set_pre_queens_cached=self.set_pre_queens_cached
  while free:
    bit:int=free&-free
    free&=free-1
    ijkl_list, subconst_cache, constellations, preset_queens = set_pre_queens_cached(N,ijkl_list,subconst_cache,(ld|bit)<<1,(rd|bit)>>1,col|bit,k,l,row+1,queens+1,LD,RD,counter,constellations,preset_queens,visited,constellation_signatures,zobrist_hash_tables)

  return ijkl_list, subconst_cache, constellations, preset_queens

####################################################
#
# constellation / solution cached
#
####################################################

"""開始コンステレーション（代表部分盤面）の列挙。中央列（奇数 N）特例、回転重複排除 （`check_rotations`）、Jasmin 正規化（`get_jasmin`）を経て、各 sc から `set_pre_queens_cached` でサブ構成を作る。"""
def gen_constellations(
  N:int,
  ijkl_list:Set[int],
  subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],
  constellations:List[Dict[str,int]],
  preset_queens:int
)->Tuple[
  Set[int],
  Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],
  List[Dict[str,int]],
  int
]:
  """
  機能:
    N-Queens の探索を分割するための「開始コンステレーション（部分盤面）」を列挙し、
    各開始コンステレーションから `set_pre_queens_cached()` を使って
    preset_queens 行までの“サブコンステレーション”を生成して `constellations` に追加する。

  引数:
    N:
      盤サイズ。
    ijkl_list:
      開始コンステレーション候補のパック値集合（to_ijkl の結果）。
      - 本関数内で update / Jasmin 変換を行い更新される。
    subconst_cache:
      サブコンステレーション生成の重複防止キャッシュ（key は (ld,rd,col,k,l,row,queens,LD,RD,N,preset_queens)）。
      - 実行ごとに clear() して「今回実行内」の重複排除に限定する（安全側）。
    constellations:
      出力のタスク配列。各要素は dict で、少なくとも "ld","rd","col","startijkl" を持つ。
      - `set_pre_queens_cached()` が append する。
    preset_queens:
      事前に置く行数（“星座の深さ”のようなもの）。
      - この値に到達した時点の状態を constellation タスクとして採用する。

  返り値:
    (ijkl_list, subconst_cache, constellations, 追加した constellation 数)

  前提/不変条件:
    - to_ijkl / geti/getj/getk/getl / get_jasmin / check_rotations が定義済み。
    - set_pre_queens_cached() が constellation を append する実装になっている。

  設計のポイント（ソース内の意図）:
    - 開始星座（i,j,k,l）は回転重複を check_rotations() で排除。
    - その後 Jasmin 変換で正規形へ寄せる（同型の統一）。
    - 各開始星座 sc から (ld,rd,col,LD,RD,…) を作り、preset_queens まで展開してタスク化。

  注意:
    - 本関数は「開始星座の列挙」と「サブ星座生成の入口」を担当。
      実際にどの状態を constellation として採用するかは set_pre_queens 系の方針に依存する。
  """

  # ---- 定数・補助値 ----
  halfN = (N + 1) // 2        # N の半分（切り上げ）。開始星座生成の範囲を絞るために使う
  N1:int = N - 1              # 最終列 index
  N2:int = N - 2

  # ---- 実行ごとにメモ化（重複抑止）をリセット ----
  # N や preset_queens が変わると key も変わるが、
  # “長寿命プロセス”で繰り返し呼ばれる可能性を考えると毎回クリアが安全。
  subconst_cache.clear()

  # 79 FIX:
  #   subconst_cache は set_pre_queens_cached() の再帰内重複抑止用。
  #   これを全 sc 共通にすると、preset_queens>=6 で別の開始星座 sc が
  #   同じ (ld,rd,col,k,l,row,queens,LD,RD,N,preset) 状態へ合流したとき、
  #   後続 sc 側の constellation 生成が丸ごと抑止される。
  #   preset=5 では影響が出にくいが、preset=6/7 で SQd0 側の不足が出る。
  #   よって subconst_cache は各 sc ごとに clear する。
  #
  # 80 FIX:
  #   preset=7 では同一 sc 内でも同じ状態へ複数経路で合流し、
  #   その multiplicity 自体が必要になる。set_pre_queens_cached() 側で
  #   preset_queens>=7 のときは subconst_cache を bypass する。

  # constellation_signatures は「同一開始 sc 内」での重複排除（サブ生成の内部で使う想定）
  constellation_signatures: Set[Tuple[int,int,int,int,int,int]] = set()

  # ---- 奇数 N の中央列特例（center を固定した開始星座を追加）----
  if N % 2 == 1:
    center = N // 2
    # center を k に固定した開始星座を列挙
    ijkl_list.update(
      to_ijkl(i, j, center, l)
      for l in range(center + 1, N1)
      for i in range(center + 1, N1)
      if i != (N1) - l
      for j in range(N - center - 2, 0, -1)
      if j != i and j != l
      # 回転重複の排除（既に登録済みなら skip）
      if not check_rotations(ijkl_list, i, j, center, l, N)
    )

  # ---- (A) コーナーにクイーンがない開始星座 ----
  # ここが一番大きい候補生成。回転重複排除 check_rotations が効く前提。
  ijkl_list.update(
    to_ijkl(i, j, k, l)
    for k in range(1, halfN)
    for l in range(k + 1, N1)
    for i in range(k + 1, N1)
    if i != (N1) - l
    for j in range(N - k - 2, 0, -1)
    if j != i and j != l
    if not check_rotations(ijkl_list, i, j, k, l, N)
  )

  # ---- (B) コーナーにクイーンがある開始星座 ----
  # (0,j,0,l) 型を追加（“角あり”のクラス）
  ijkl_list.update({to_ijkl(0, j, 0, l) for j in range(1, N2) for l in range(j + 1, N1)})

  # ---- Jasmin 変換：開始星座を正規形に寄せる ----
  ijkl_list = {get_jasmin(N, c) for c in ijkl_list}

  # 左端列のビット（MSB 側）を 1 にするための基準
  # ※この実装では「左端 = 1<<(N-1)」としている
  L = 1 << (N1)

  # 追加した constellation 数を返すために counter を使う（set_pre_queens 側が増やす）
  # （List にして参照渡し＝ミュータブルにしている）
  # ※既存実装の方針に合わせる
  # counter[0] が “今回 sc から追加した constellation 数” になる
  for sc in ijkl_list:
    # 79 FIX:
    #   subconst_cache を sc ごとに初期化する。
    #   全 sc 共通キャッシュにすると、後続 sc の正当な constellation が
    #   cache hit で生成されず、preset=6/7 で不足する。
    subconst_cache.clear()

    # sc ごとに重複抑止セットを初期化（＝この sc の内部だけで重複排除）
    constellation_signatures = set()

    # sc から (i,j,k,l) を復元
    i, j, k, l = geti(sc), getj(sc), getk(sc), getl(sc)

    # i/j/l の列ビット（L を右シフトして作る）
    Lj = L >> j
    Li = L >> i
    Ll = L >> l

    # ---- 開始状態（ld, rd, col, …）の構築 ----
    # ld/rd は「斜め攻撃線」、col は「縦列占有」。
    # ここは開始星座の“型”に依存する初期化で、探索の入口を作る。
    ld = (((L >> (i - 1)) if i > 0 else 0) | (1 << (N - k)))
    rd = ((L >> (i + 1)) | (1 << (l - 1)))
    col = (1 | L | Li | Lj)

    # mark 行などで使う補助ブロック（実装の意図に沿って保持）
    LD = (Lj | Ll)
    RD = (Lj | (1 << k))

    # ---- サブコンステレーション生成準備 ----
    counter: List[int] = [0]     # set_pre_queens 側が増やす
    visited: Set[int] = set()    # 枝刈り用 visited（hash を入れる設計）

    # Opt-04: preset_queens 行を事前に置く
    # Zobrist テーブルは “必要になった時に初期化” する設計（既存実装に合わせる）
    zobrist_hash_tables: Dict[int, Dict[str, List[u64]]] = {}

    # ---- サブ生成（キャッシュ付き）----
    # row=1、queens は (j==N1) かどうかで 3/4 を切り替えている（既存ロジック）
    ijkl_list, subconst_cache, constellations, preset_queens = set_pre_queens_cached(
      N, ijkl_list, subconst_cache,
      ld, rd, col,
      k, l,
      1,
      3 if j == N1 else 4,
      LD, RD,
      counter, constellations, preset_queens,
      visited, constellation_signatures,
      zobrist_hash_tables
    )

    # ---- startijkl に “開始星座 base” を追記 ----
    # set_pre_queens 側で作った constellation["startijkl"] は「途中状態の pack」なので、
    # ここで base=(i,j,k,l) を OR して “起点” を埋める。
    base = to_ijkl(i, j, k, l)

    # 直近に追加された counter[0] 件へ OR をかける（末尾から辿る）
    for a in range(counter[0]):
      constellations[-1 - a]["startijkl"] |= base

  # 返す 4 つ目は “最後に作った sc の counter” ではなく、
  # 元実装どおり「最後の counter[0]」を返す（上位で使っている想定）
  # 49 FIX: return preset_queens itself.
  # The previous counter[0] was the number of constellations added by the last sc,
  # which caused misleading logs such as preset_queens=13 or preset_queens=128.
  return ijkl_list, subconst_cache, constellations, preset_queens

""" コンステレーションリストの妥当性確認ヘルパ。各要素に 'ld','rd','col','startijkl' キーが存在するかをチェック。"""
def validate_constellation_list(constellations:List[Dict[str,int]])->bool: 
  return all(all(k in c for k in ("ld","rd","col","startijkl")) for c in constellations)

"""32bit little-endian の相互変換ヘルパ。Codon/CPython の差異に注意。"""
def read_uint32_le(b:str)->int: 
  return (ord(b[0])&0xFF)|((ord(b[1])&0xFF)<<8)|((ord(b[2])&0xFF)<<16)|((ord(b[3])&0xFF)<<24)

"""32bit little-endian バイト列への変換ヘルパ。"""
def int_to_le_bytes(x:int)->List[int]: 
  return [(x>>(8*i))&0xFF for i in range(4)]

"""ファイル存在チェック（読み取り open の可否で判定）。"""
def file_exists(fname:str)->bool:
  try:
    with open(fname,"rb"):
      return True
  except:
    return False

"""bin キャッシュのサイズ妥当性確認（1 レコード 16 バイトの整数倍か）。"""
def validate_bin_file(fname:str)->bool:
  try:
    with open(fname,"rb") as f:
      f.seek(0,2)  # ファイル末尾に移動
      size=f.tell()
    return size%16==0
  except:
    return False

"""バイナリ形式での解exec_solutions()のキャッシュ入出力""" 
def u64_to_le_bytes(x: u64) -> List[int]:
  v:int = int(x)
  return [(v >> (8*i)) & 0xFF for i in range(8)]

""" バイト列を little-endian u64 に変換 """
def read_uint64_le( raw: str) -> u64:
  v:int = 0
  for i in range(8):
    v |= (ord(raw[i]) & 0xFF) << (8*i)
  return u64(v)

""" テキスト形式での解exec_solutions()のキャッシュ保存"""
def save_solutions_txt(fname:str,constellations:List[Dict[str,int]]) -> None:
  f = open(fname, "w")
  f.write("startijkl,solutions\n")
  for d in constellations:
    f.write(str(d["startijkl"]))
    f.write(",")
    f.write(str(int(d["solutions"])))
    f.write("\n")
  f.close()

"""バイナリ形式での解exec_solutions()のキャッシュ保存v2"""
def save_solutions_bin_v2(fname:str,constellations:List[Dict[str,int]]) -> None:
  b8 = u64_to_le_bytes
  f = open(fname, "wb")
  for d in constellations:
    # u64 で揃える（40 bytes/record）
    for x in (
      u64(d["startijkl"]),
      u64(d["ld"]),
      u64(d["rd"]),
      u64(d["col"]),
      u64(d["solutions"]),
    ):
      bb = b8(x)
      f.write("".join(chr(c) for c in bb))
  f.close()

"""テキスト形式での解exec_solutions()のキャッシュ入出力"""
def load_solutions_txt_into(fname:str,constellations:List[Dict[str,int]]) -> bool:
  try:
    f = open(fname, "r")
  except:
    return False
  text = f.read()
  f.close()
  if text is None:
    return False
  lines = text.split("\n")
  if len(lines) < 2:
    return False
  if lines[0].strip() != "startijkl,solutions":
    return False

  # startijkl -> solutions
  mp: Dict[int, int] = {}
  for idx in range(1, len(lines)):
    line = lines[idx].strip()
    if line == "":
      continue
    parts = line.split(",")
    if len(parts) != 2:
      return False
    k = int(parts[0])
    v = int(parts[1])
    mp[k] = v
  # 全 constellations に埋める（欠けがあれば失敗）
  for d in constellations:
    s = d["startijkl"]
    if s not in mp:
      # print("[cache miss] startijkl=", int(s[0])," ld=", int(s[1]), " rd=", int(s[2]), " col=", int(s[3]))
      return False
    d["solutions"] = mp[s]

  return True

""" バイナリ形式での解exec_solutions()のキャッシュ読み込みv2"""
def load_solutions_bin_into_v2(fname:str,constellations:List[Dict[str,int]])->bool:
  try:
    f = open(fname, "rb")
  except:
    return False
  data = f.read()
  f.close()
  if data is None:
    return False
  rec:int = 40
  n:int = len(data)
  if n == 0 or (n % rec) != 0:
    return False
  nrec:int = n // rec
  r8 = read_uint64_le
  mp: Dict[Tuple[u64,u64,u64,u64], u64] = {}
  p:int = 0
  for _ in range(nrec):
    s  = r8(data[p:p+8]);   p += 8
    ld = r8(data[p:p+8]);   p += 8
    rd = r8(data[p:p+8]);   p += 8
    col= r8(data[p:p+8]);   p += 8
    sol= r8(data[p:p+8]);   p += 8
    mp[(s, ld, rd, col)] = sol
  for d in constellations:
    key = (u64(d["startijkl"]), u64(d["ld"]), u64(d["rd"]), u64(d["col"]))
    if key not in mp:
      print("[cache miss] startijkl=", int(key[0])," ld=", int(key[1]), " rd=", int(key[2]), " col=", int(key[3]))
      return False
    d["solutions"] = int(mp[key])

  return True

"""テキスト形式での解exec_solutions()のキャッシュ入出力ラッパー"""
def load_or_build_solutions_txt(N:int,constellations:List[Dict[str,int]],preset_queens:int,use_gpu:bool,cache_tag:str = "") -> None:

  tag = "_" + cache_tag if cache_tag != "" else ""
  fname = "solutions_N" + str(N) + "_" + str(preset_queens) + tag + ".txt"

  if file_exists(fname):
    if load_solutions_txt_into(fname, constellations):
      return
    else:
      print("[警告] solutions txt キャッシュ不一致: " + fname + " を再生成します")

  # なければ計算して保存
  exec_solutions(N,constellations,use_gpu)
  save_solutions_txt(fname, constellations)

"""バイナリ形式での解exec_solutions()のキャッシュ入出力ラッパー"""
def load_or_build_solutions_bin(N:int,constellations:List[Dict[str,int]],preset_queens:int,use_gpu:bool,cache_tag:str = "") -> None:

  tag = f"_{cache_tag}" if cache_tag != "" else ""
  fname = f"solutions_N{N}_{preset_queens}{tag}.bin"

  if file_exists(fname):
    if load_solutions_bin_into_v2(fname, constellations):
      return
    else:
      print(f"[警告] solutions キャッシュ不一致/破損: {fname} を再生成します")

  # なければ計算して保存
  exec_solutions(N,constellations, use_gpu)
  save_solutions_bin_v2(fname, constellations)

"""テキスト形式で constellations を保存/復元する（1 行 5 数値: ld rd col startijkl solutions）。"""
def save_constellations_txt(path:str,constellations:List[Dict[str,int]])->None:
  with open(path,"w") as f:
    for c in constellations:
      ld=c["ld"]
      rd=c["rd"]
      col=c["col"]
      startijkl=c["startijkl"]
      solutions=c.get("solutions",0)
      f.write(f"{ld} {rd} {col} {startijkl} {solutions}\n")

"""テキスト形式で constellations を保存/復元する（1 行 5 数値: ld rd col startijkl solutions）。"""
def load_constellations_txt(path:str,constellations:List[Dict[str,int]])->List[Dict[str,int]]:
  # out:List[Dict[str,int]]=[]
  with open(path,"r") as f:
    for line in f:
      parts=line.strip().split()
      if len(parts)!=5:
        continue
      ld=int(parts[0]);rd=int(parts[1]);col=int(parts[2])
      startijkl=int(parts[3]);solutions=int(parts[4])
      # out.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions": solutions})
      constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions": solutions})
  # return out
  return constellations

"""テキストキャッシュを読み込み。壊れていれば `gen_constellations()` で再生成して保存する。"""
def load_or_build_constellations_txt(N:int,ijkl_list:Set[int],subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],constellations:List[Dict[str,int]],preset_queens:int)->Tuple[Set[int],Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],List[Dict[str,int]],int]:

  # N と preset_queens に基づいて一意のファイル名を構成
  fname=f"constellations_N{N}_{preset_queens}.txt"
  # ファイルが存在すれば読み込むが、破損チェックも行う
  if file_exists(fname):
    try:
      constellations=load_constellations_txt(fname,constellations)
      if validate_constellation_list(constellations):
        return ijkl_list,subconst_cache,constellations,preset_queens
      else:
        print(f"[警告] 不正なキャッシュ形式: {fname} を再生成します")
    except Exception as e:
      print(f"[警告] キャッシュ読み込み失敗: {fname}, 理由: {e}")
  # ファイルがなければ生成・保存
  # constellations:List[Dict[str,int]]=[]
  ijkl_list,subconst_cache,constellations,preset_queens=gen_constellations(N,ijkl_list,subconst_cache,constellations,preset_queens)
  save_constellations_txt(fname,constellations)
  return ijkl_list,subconst_cache,constellations,preset_queens

"""bin 形式で constellations を保存/復元。Codon では str をバイト列として扱う前提のため、CPython では bytes で書き込むよう分岐/注意が必要。"""
def save_constellations_bin(N:int,fname:str,constellations:List[Dict[str,int]])->None:
  # _int_to_le_bytes=int_to_le_bytes
  with open(fname,"wb") as f:
    for d in constellations:
      for key in ["ld","rd","col","startijkl"]:
        b=int_to_le_bytes(d[key])
        # int_to_le_bytes(d[key])
        f.write("".join(chr(c) for c in b))  # Codonでは str がバイト文字列扱い

"""bin 形式で constellations を保存/復元。Codon では str をバイト列として扱う前提のため、CPython では bytes で書き込むよう分岐/注意が必要。"""
def load_constellations_bin(N:int,fname:str,constellations:List[Dict[str,int]],)->List[Dict[str,int]]:
  # constellations:List[Dict[str,int]]=[]
  _read_uint32_le=read_uint32_le
  with open(fname,"rb") as f:
    while True:
      raw=f.read(16)
      if len(raw)<16:
        break
      ld=read_uint32_le(raw[0:4])
      rd=read_uint32_le(raw[4:8])
      col=read_uint32_le(raw[8:12])
      startijkl=_read_uint32_le(raw[12:16])
      constellations.append({ "ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0 })
  return constellations

"""bin キャッシュを読み込み。検証に失敗した場合は再生成して保存し、その結果を返す。"""
def load_or_build_constellations_bin(N:int,ijkl_list:Set[int],subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],constellations:List[Dict[str,int]],preset_queens:int)->Tuple[Set[int],Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],List[Dict[str,int]],int]:

  # N と preset_queens に基づいて一意のファイル名を構成
  fname=f"constellations_N{N}_{preset_queens}.bin"
  if file_exists(fname):
    # ファイルが存在すれば読み込むが、破損チェックも行う
    try:
      constellations=load_constellations_bin(N,fname,constellations)
      if validate_bin_file(fname) and validate_constellation_list(constellations):
        return ijkl_list,subconst_cache,constellations,preset_queens
      else:
        print(f"[警告] 不正なキャッシュ形式: {fname} を再生成します")
    except Exception as e:
      print(f"[警告] キャッシュ読み込み失敗: {fname}, 理由: {e}")
  # ファイルがなければ生成・保存
  # constellations:List[Dict[str,int]]=[]
  ijkl_list,subconst_cache,constellations,preset_queens=gen_constellations(N,ijkl_list,subconst_cache,constellations,preset_queens)
  save_constellations_bin(N,fname,constellations)
  return ijkl_list,subconst_cache,constellations,preset_queens



####################################################
#
# 84 stream constellation bin generation / GPU chunk runner
#
####################################################

"""bin キャッシュのレコード数を返す（1 record = 16 bytes）。破損時は 0。"""
def count_constellations_bin_records(fname:str)->int:
  try:
    with open(fname,"rb") as f:
      f.seek(0,2)
      size:int=f.tell()
    if size%16!=0:
      return 0
    return size//16
  except:
    return 0

"""stream 完了マーカーを読む。存在しない/不正なら -1。"""
def read_stream_done_count(fname:str)->int:
  try:
    with open(fname,"r") as f:
      text:str=f.read().strip()
    if text=="":
      return -1
    return int(text)
  except:
    return -1

"""stream 完了マーカーを書く。"""
def write_stream_done_count(fname:str,count:int)->None:
  with open(fname,"w") as f:
    f.write(str(count))
    f.write("\n")

"""stream 生成のために既存 bin を空にする。"""
def truncate_constellations_bin(fname:str)->None:
  with open(fname,"wb") as f:
    pass
  write_stream_done_count(fname+".done",0)

"""constellation chunk を bin へ追記する。"""
def append_constellations_bin(fname:str,constellations:List[Dict[str,int]])->None:
  with open(fname,"ab") as f:
    for d in constellations:
      for key in ["ld","rd","col","startijkl"]:
        b=int_to_le_bytes(d[key])
        f.write("".join(chr(c) for c in b))

"""N-Queens constellation を全件 List に持たず、sc 単位で .bin へ直接書き出す。"""
def gen_constellations_stream_to_bin(
  N:int,
  ijkl_list:Set[int],
  subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],
  fname:str,
  preset_queens:int,
  gpu_log_level:int=0
)->Tuple[Set[int],Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],int,int]:

  halfN=(N+1)//2
  N1:int=N-1
  N2:int=N-2
  subconst_cache.clear()

  constellation_signatures:Set[Tuple[int,int,int,int,int,int]]=set()

  if N%2==1:
    center=N//2
    ijkl_list.update(
      to_ijkl(i,j,center,l)
      for l in range(center+1,N1)
      for i in range(center+1,N1)
      if i!=(N1)-l
      for j in range(N-center-2,0,-1)
      if j!=i and j!=l
      if not check_rotations(ijkl_list,i,j,center,l,N)
    )

  ijkl_list.update(
    to_ijkl(i,j,k,l)
    for k in range(1,halfN)
    for l in range(k+1,N1)
    for i in range(k+1,N1)
    if i!=(N1)-l
    for j in range(N-k-2,0,-1)
    if j!=i and j!=l
    if not check_rotations(ijkl_list,i,j,k,l,N)
  )

  ijkl_list.update({to_ijkl(0,j,0,l) for j in range(1,N2) for l in range(j+1,N1)})
  ijkl_list={get_jasmin(N,c) for c in ijkl_list}

  L=1<<(N1)
  total_count:int=0
  sc_index:int=0
  truncate_constellations_bin(fname)

  for sc in ijkl_list:
    subconst_cache.clear()
    constellation_signatures=set()

    i,j,k,l=geti(sc),getj(sc),getk(sc),getl(sc)
    Lj=L>>j
    Li=L>>i
    Ll=L>>l

    ld=(((L>>(i-1)) if i>0 else 0)|(1<<(N-k)))
    rd=((L>>(i+1))|(1<<(l-1)))
    col=(1|L|Li|Lj)
    LD=(Lj|Ll)
    RD=(Lj|(1<<k))

    counter:List[int]=[0]
    visited:Set[int]=set()
    zobrist_hash_tables:Dict[int,Dict[str,List[u64]]]={}
    sc_constellations:List[Dict[str,int]]=[]

    ijkl_list,subconst_cache,sc_constellations,preset_queens=set_pre_queens_cached(
      N,ijkl_list,subconst_cache,
      ld,rd,col,
      k,l,
      1,
      3 if j==N1 else 4,
      LD,RD,
      counter,sc_constellations,preset_queens,
      visited,constellation_signatures,
      zobrist_hash_tables
    )

    base=to_ijkl(i,j,k,l)
    for a in range(counter[0]):
      sc_constellations[-1-a]["startijkl"]|=base

    if counter[0]>0:
      append_constellations_bin(fname,sc_constellations)
      total_count+=counter[0]

    if gpu_log_level>=2:
      print(f"[stream-build-sc] N={N} sc_index={sc_index} added={counter[0]} total={total_count}")

    sc_index+=1

  write_stream_done_count(fname+".done",total_count)
  if gpu_log_level>=1:
    print(f"[stream-build-summary] N={N} preset_queens={preset_queens} sc={sc_index} records={total_count} bin={fname}")

  return ijkl_list,subconst_cache,total_count,preset_queens

"""stream 版: .bin が完了済みなら使い、なければ sc 単位で生成し、レコード数を返す。"""
def ensure_constellations_bin_stream(N:int,ijkl_list:Set[int],subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],preset_queens:int,gpu_log_level:int=0)->Tuple[Set[int],Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],int,int,str]:
  fname:str=f"constellations_N{N}_{preset_queens}.bin"
  records:int=count_constellations_bin_records(fname)
  done_count:int=read_stream_done_count(fname+".done")
  if records>0 and done_count==records:
    if gpu_log_level>=1:
      print(f"[stream-cache-hit] N={N} preset_queens={preset_queens} records={records} bin={fname}")
    return ijkl_list,subconst_cache,records,preset_queens,fname

  if file_exists(fname):
    print(f"[stream-cache-warning] invalid/incomplete bin cache: {fname}; records={records} done={done_count}; rebuilding")
  else:
    if gpu_log_level>=1:
      print(f"[stream-cache-miss] N={N} preset_queens={preset_queens} bin={fname}; building")

  ijkl_list,subconst_cache,records,preset_queens=gen_constellations_stream_to_bin(N,ijkl_list,subconst_cache,fname,preset_queens,gpu_log_level)
  return ijkl_list,subconst_cache,records,preset_queens,fname

"""stream progress を TSV に追記する。"""
def append_stream_progress(progress_fname:str,N:int,preset_queens:int,chunk_index:int,off:int,m:int,chunk_total:int,gpu_total:int)->None:
  with open(progress_fname,"a") as f:
    f.write(f"{N}\t{preset_queens}\t{chunk_index}\t{off}\t{m}\t{chunk_total}\t{gpu_total}\n")

"""bin を STEPS 件ずつ読み、各 chunk を既存 GPU kernel へ投入する低メモリ GPU 実行。"""
def exec_solutions_gpu_bin_stream(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1,
  cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT,
  chunk_only:bool=False,
  debug_chunk_start:int=0,
  debug_chunk_count:int=1
)->int:

  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  total_records:int=count_constellations_bin_records(fname)
  progress_fname:str=f"progress_N{N}_{preset_queens}_stream.tsv"
  with open(progress_fname,"w") as pf:
    pf.write("N\tpreset\tchunk\toff\tm\tchunk_total\tgpu_total\n")

  if chunk_only:
    if debug_chunk_start<0:
      debug_chunk_start=0
    if debug_chunk_count<=0:
      debug_chunk_count=1

  if gpu_log_level>=1:
    print(f"[stream-gpu-config] N={N} records={total_records} bin={fname} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} sort_mode={gpu_sort_mode} chunk_only={1 if chunk_only else 0} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} progress={progress_fname}")

  gpu_total:int=0
  off:int=0
  chunk_index:int=0
  executed_chunks:int=0
  _read_uint32_le=read_uint32_le

  with open(fname,"rb") as f:
    while True:
      chunk_constellations:List[Dict[str,int]]=[]
      i:int=0
      while i<STEPS:
        raw=f.read(16)
        if len(raw)<16:
          break
        ld=_read_uint32_le(raw[0:4])
        rd=_read_uint32_le(raw[4:8])
        col=_read_uint32_le(raw[8:12])
        startijkl=_read_uint32_le(raw[12:16])
        chunk_constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
        i+=1

      m:int=len(chunk_constellations)
      if m==0:
        break

      if chunk_only:
        run_this_chunk:bool=(chunk_index>=debug_chunk_start and chunk_index<debug_chunk_start+debug_chunk_count)
        if not run_this_chunk:
          if gpu_log_level>=2:
            print(f"[stream-gpu-chunk-skip] N={N} chunk={chunk_index} off={off} m={m}")
          off+=m
          chunk_index+=1
          continue

      t0=datetime.now()
      if gpu_log_level>=1:
        print(f"[stream-gpu-chunk-start] N={N} chunk={chunk_index} off={off} m={m}")

      exec_solutions(N,chunk_constellations,True,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe)

      chunk_total:int=0
      if m>0:
        chunk_total=chunk_constellations[0]["solutions"]
      gpu_total+=chunk_total
      executed_chunks+=1
      t1=datetime.now()
      append_stream_progress(progress_fname,N,preset_queens,chunk_index,off,m,chunk_total,gpu_total)
      if gpu_log_level>=1:
        print(f"[stream-gpu-chunk-end] N={N} chunk={chunk_index} off={off} m={m} elapsed={str(t1-t0)[:-3]} chunk_total={chunk_total} gpu_total={gpu_total}")

      off+=m
      chunk_index+=1

  if gpu_log_level>=1:
    print(f"[stream-gpu-summary] N={N} records={total_records} chunks={chunk_index} executed_chunks={executed_chunks} total={gpu_total} progress={progress_fname}")

  return gpu_total

"""N に応じて preset_queens を動的に選択する。"""
def select_dynamic_preset_queens(N:int,preset_queens:int)->int:
  if N>=5 and N<=17:
    return 5
  elif N>=18 and N<=21:
    return 6
  elif N>=22 and N<=24:
    return 7
  elif N>=25 and N<=27:
    return 8
  return preset_queens


"""プリセットクイーン数を調整 preset_queensとconstellationsを返却"""
def build_constellations_dynamicK(N: int, ijkl_list:Set[int],subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],constellations:List[Dict[str,int]],use_gpu: bool,preset_queens:int)->Tuple[Set[int],Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],List[Dict[str,int]],int]:

  preset_queens=select_dynamic_preset_queens(N,preset_queens)
  use_bin=True
  if use_bin:
    # bin
    ijkl_list,subconst_cache,constellations,preset_queens=load_or_build_constellations_bin(N,ijkl_list,subconst_cache, constellations, preset_queens)
    #
  else:
    # txt
    ijkl_list,subconst_cache,constellations,preset_queens=load_or_build_constellations_txt(N,ijkl_list,subconst_cache, constellations, preset_queens)

  return  ijkl_list,subconst_cache,constellations,preset_queens

"""小さな N 用の素朴な全列挙（対称重みなし）。ビットボードで列/斜線の占有を管理して再帰的に合計を返す。検算/フォールバック用。"""
def _bit_total(N:int)->int:
  mask:int=(1<<N)-1
  """ 小さなNは正攻法で数える（対称重みなし・全列挙） """
  def bt(row:int,left:int,down:int,right:int):
    if row==N:
      return 1
    total:int=0
    bitmap:int=mask&~(left|down|right)
    while bitmap:
      bit:int=-bitmap&bitmap
      bitmap^=bit
      total+=bt(row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
    return total
  return bt(0,0,0,0)

"""N=5..17 の合計解を計測。N<=5 は `_bit_total()` のフォールバック、それ以外は星座キャッシュ（.bin/.txt）→ `exec_solutions()` → 合計→既知解 `expected` と照合。"""
def main()->None:
  global DISABLE_CONSTELLATION_SIGNATURE_PRUNE
  
  expected:List[int]=[0,0,0,0,0,10,4,40,92,352,724,2680,14200,73712,365596,2279184,14772512,95815104,666090624,4968057848,39029188884,314666222712,2691008701644,24233937684440,227514171973736,2207893435808352,22317699616364044,234907967154122528]     
  nmin:int=5
  nmax:int=28
  use_gpu:bool=False
  gpu_block:int=32
  gpu_max_blocks:int=484
  gpu_log_level:int=0
  gpu_sort_mode:int=-1
  cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT
  debug_chunk_start:int=0
  debug_chunk_count:int=1
  bench_mode:int=0  # 0:normal, 1:N20 warmup repeat, 2:N19 preheat, 3:N18+N19 preheat, 4:N20 repeat3 sweep, 5:N20 repeat2 benchmark, 6:reorder-only debug, 7:chunk-only debug, 8:boundary-classification-only, 9:boundary-solution-summary, 10:boundary-classification-only + signature prune disabled, 11:stream-bin-build-only
  # 通常運用では preset_queens は 5 固定。診断用 bench_mode>=8 のときだけ引数の preset を許可する。
  preset_queens_arg:int=5
  requested_preset_arg:int=5
  argc:int=len(sys.argv)

  if argc == 1:
    print("CPU mode selected")
    pass
  elif argc >= 2:
    arg = sys.argv[1]
    if arg == "-c":
      use_gpu = False
      print("CPU mode selected")
    elif arg == "-g":
      use_gpu = True
      print("GPU mode selected")
    else:
      print(f"Unknown option: {arg}")
      print("Usage: nqueens [-c | -g] [nmin nmax] [gpu_block gpu_max_blocks log_level sort_mode] [preset_queens] [bench_mode] [cross_stripe_safe] [debug_chunk_start] [debug_chunk_count]")
      return

    # nmax は指定時だけ inclusive として扱う。
    # 例: ./30... -g 18 18 256 32 1
    if argc >= 4:
      nmin=int(sys.argv[2])
      nmax=int(sys.argv[3])+1
    if argc >= 5:
      gpu_block=int(sys.argv[4])
    if argc >= 6:
      gpu_max_blocks=int(sys.argv[5])
    if argc >= 7:
      gpu_log_level=int(sys.argv[6])
    if argc >= 8:
      gpu_sort_mode=int(sys.argv[7])
    if argc >= 9:
      requested_preset_arg=int(sys.argv[8])
    if argc >= 10:
      bench_mode=int(sys.argv[9])
      if bench_mode<0 or bench_mode>11:
        print(f"[warning] unknown bench_mode={bench_mode}; using 0")
        bench_mode=0
    if bench_mode>=8:
      preset_queens_arg=requested_preset_arg
    else:
      if requested_preset_arg!=5:
        print(f"[warning] preset_queens={requested_preset_arg} is disabled in 77 normal modes; using 5")
      preset_queens_arg=5
    if argc >= 11:
      cross_stripe_safe=(int(sys.argv[10])!=0)
    if argc >= 12:
      debug_chunk_start=int(sys.argv[11])
    if argc >= 13:
      debug_chunk_count=int(sys.argv[12])
    if argc > 13:
      print("Too many arguments")
      print("Usage: nqueens [-c | -g] [nmin nmax] [gpu_block gpu_max_blocks log_level sort_mode] [preset_queens] [bench_mode] [cross_stripe_safe] [debug_chunk_start] [debug_chunk_count]")
      return
  else:
    print("Usage: nqueens [-c | -g] [nmin nmax] [gpu_block gpu_max_blocks log_level sort_mode] [preset_queens] [bench_mode] [cross_stripe_safe] [debug_chunk_start] [debug_chunk_count]")
    return

  if bench_mode==10:
    DISABLE_CONSTELLATION_SIGNATURE_PRUNE=True
  else:
    DISABLE_CONSTELLATION_SIGNATURE_PRUNE=False

  if use_gpu:
    print(f"version        : {VERSION_TAG}")
    print(f"cross_stripe_safe: {1 if cross_stripe_safe else 0}")
    if bench_mode==7:
      print(f"chunk_only    : start={debug_chunk_start} count={debug_chunk_count}")
    if bench_mode==8 or bench_mode==9 or bench_mode==10:
      print(f"boundary_debug: mode={bench_mode} preset={preset_queens_arg} signature_prune_disabled={1 if DISABLE_CONSTELLATION_SIGNATURE_PRUNE else 0}")
    if bench_mode==11:
      print(f"stream_bin_only: mode={bench_mode} preset={preset_queens_arg}")
  print(" N:             Total           Unique         hh:mm:ss.ms")
  for N in range(nmin,nmax):
    override_elapsed_text:str=""
    start_time=datetime.now()
    if N<=5:

      """ 小さなNは正攻法で数える（対称重みなし・全列挙） """
      total=_bit_total(N)

      dt=datetime.now()-start_time
      text=str(dt)[:-3]
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}")
      continue

    ijkl_list:Set[int]=set()
    constellations:List[Dict[str,int]]=[]
    subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]]=set()

    """ constellasions()でキャッシュを使う """
    use_constellation_cache:bool = False

    preset_queens:int = preset_queens_arg # preset_queens CPUが担当する深さ
    preset_queens=select_dynamic_preset_queens(N,preset_queens)

    if gpu_log_level>=1:
      print(f"[dynamic-preset] N={N} preset_queens={preset_queens}")

    # 84 STREAM:
    #   bench_mode=11 は CPU/GPU どちらでも .bin を stream 生成して終了。
    #   GPU の N>=21 通常実行は全件 List[Dict] を作らず、.bin から STEPS 件ずつ読み込む。
    if bench_mode==11:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"[stream-cache-only] N={N} preset_queens={preset_queens} records={stream_records} bin={stream_fname} valid={1 if validate_bin_file(stream_fname) else 0}")
      print(f"{N:2d}:{0:18d}{0:17d}{text:>21s}    stream-cache-only")
      continue

    if use_gpu and N>=21 and not (bench_mode==8 or bench_mode==9 or bench_mode==10):
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      stream_chunk_only:bool=(bench_mode==7)
      total:int=exec_solutions_gpu_bin_stream(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,stream_chunk_only,debug_chunk_start,debug_chunk_count)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      status:str="ok" if expected[N]==total else f"ng({total}!={expected[N]})"
      if stream_chunk_only:
        status="stream-chunk-only"
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    {status}")
      continue

    if use_constellation_cache:
      ijkl_list,subconst_cache,constellations,preset_queens= build_constellations_dynamicK(N,ijkl_list,subconst_cache,constellations, use_gpu,preset_queens)
    else:
      ijkl_list,subconst_cache,constellations,preset_queens=gen_constellations(N,ijkl_list,subconst_cache,constellations,preset_queens)

    if bench_mode==8 or bench_mode==10:
      print(f"[constellation-config] N={N} preset_queens={preset_queens} constellations={len(constellations)} signature_prune_disabled={1 if DISABLE_CONSTELLATION_SIGNATURE_PRUNE else 0}")
      diagnose_boundary_classification(N,constellations)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      status:str="boundary-only"
      if bench_mode==10:
        status="boundary-only-nosig"
      print(f"{N:2d}:{0:18d}{0:17d}{text:>21s}    {status}")
      continue


    """ solutions()でキャッシュを使って実行 """
    use_solution_cache = False
    if use_solution_cache:
        #
        # text
        # load_or_build_solutions_txt(N,constellations, preset_queens, use_gpu, cache_tag="v2")
        # 
        # bin
        load_or_build_solutions_bin(N,constellations, preset_queens, use_gpu, cache_tag="v2")
        # 
    else:
        # 72 STABLE FINAL BENCH:
        #   kernel/探索ロジックは変更せず、N=20 単体が通し実行より遅い現象を切り分ける。
        #   bench_mode=1: N=20 を同一プロセス内で 1回 warmup し、2回目を測定
        #   bench_mode=2: N=19 を非表示 preheat してから N=20 を測定
        #   bench_mode=3: N=18 と N=19 を非表示 preheat してから N=20 を測定
        if bench_mode==6 and use_gpu:
          if gpu_log_level>=1:
            print(f"[constellation-config] N={N} preset_queens={preset_queens} constellations={len(constellations)}")
            print(f"[reorder-only] mode=6 validates launch-order permutation only; GPU kernel is not executed")
          exec_solutions(N,constellations,use_gpu,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,True,True)
          override_elapsed_text="reorder-only"
        elif bench_mode==7 and use_gpu:
          if gpu_log_level>=1:
            print(f"[constellation-config] N={N} preset_queens={preset_queens} constellations={len(constellations)}")
            print(f"[chunk-only] mode=7 executes selected chunks only; GPU kernel runs only for the requested range")
          exec_solutions(N,constellations,use_gpu,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,False,True,debug_chunk_start,debug_chunk_count)
          override_elapsed_text="chunk-only"
        elif bench_mode==1 and use_gpu and N==20:
          if gpu_log_level>=1:
            print(f"[constellation-config] N={N} preset_queens={preset_queens} constellations={len(constellations)}")
            print(f"[bench-warmup] mode=1 first run is warmup; second run is measured")
          warm_start=datetime.now()
          exec_solutions(N,constellations,use_gpu,gpu_block,gpu_max_blocks,0,gpu_sort_mode,cross_stripe_safe)
          warm_total:int=sum(c['solutions'] for c in constellations if c['solutions']>0)
          warm_elapsed=datetime.now()-warm_start
          warm_text=str(warm_elapsed)[:-3]
          warm_status:str="ok" if expected[N]==warm_total else f"ng({warm_total}!={expected[N]})"
          print(f"[warmup] N={N} total={warm_total} elapsed={warm_text} {warm_status}")
          for c in constellations:
            c["solutions"]=0
          start_time=datetime.now()
          if gpu_log_level>=1:
            print(f"[constellation-config] N={N} preset_queens={preset_queens} constellations={len(constellations)}")
          exec_solutions(N,constellations,use_gpu,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe)
        elif bench_mode==5 and use_gpu and N==20:
          # 72 STABLE FINAL BENCH: run1 warmup, run2 measured.
          if gpu_log_level>=1:
            print(f"[constellation-config] N={N} preset_queens={preset_queens} constellations={len(constellations)}")
            print(f"[bench-repeat2] mode=5 run N=20 twice in the same process; run 2 is measured")
          for run_no in range(1,3):
            for c in constellations:
              c["solutions"]=0
            run_t0=datetime.now()
            run_log:int=gpu_log_level if run_no==2 else 0
            exec_solutions(N,constellations,use_gpu,gpu_block,gpu_max_blocks,run_log,gpu_sort_mode,cross_stripe_safe)
            run_total:int=sum(c['solutions'] for c in constellations if c['solutions']>0)
            run_elapsed=datetime.now()-run_t0
            run_text=str(run_elapsed)[:-3]
            if run_no==2:
              override_elapsed_text=run_text
            run_status:str="ok" if expected[N]==run_total else f"ng({run_total}!={expected[N]})"
            print(f"[repeat2] N={N} run={run_no} total={run_total} elapsed={run_text} {run_status}")
            if run_no<2:
              for c in constellations:
                c["solutions"]=0
        elif bench_mode==4 and use_gpu and N==20:
          if gpu_log_level>=1:
            print(f"[constellation-config] N={N} preset_queens={preset_queens} constellations={len(constellations)}")
            print(f"[bench-repeat] mode=4 run N=20 three times in the same process; run 3 is measured")
          for run_no in range(1,4):
            for c in constellations:
              c["solutions"]=0
            run_t0=datetime.now()
            run_log:int=gpu_log_level if run_no==3 else 0
            exec_solutions(N,constellations,use_gpu,gpu_block,gpu_max_blocks,run_log,gpu_sort_mode,cross_stripe_safe)
            run_total:int=sum(c['solutions'] for c in constellations if c['solutions']>0)
            run_elapsed=datetime.now()-run_t0
            run_text=str(run_elapsed)[:-3]
            if run_no==3:
              override_elapsed_text=run_text
            run_status:str="ok" if expected[N]==run_total else f"ng({run_total}!={expected[N]})"
            print(f"[repeat] N={N} run={run_no} total={run_total} elapsed={run_text} {run_status}")
            if run_no<3:
              for c in constellations:
                c["solutions"]=0
        elif (bench_mode==2 or bench_mode==3) and use_gpu and N==20:
          if gpu_log_level>=1:
            print(f"[bench-preheat] mode={bench_mode} preheat before measured N=20")
          pre_start_N:int=19
          if bench_mode==3:
            pre_start_N=18
          for PN in range(pre_start_N,20):
            pre_ijkl_list:Set[int]=set()
            pre_constellations:List[Dict[str,int]]=[]
            pre_subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]]=set()
            pre_preset_queens:int=5
            pre_t0=datetime.now()
            pre_ijkl_list,pre_subconst_cache,pre_constellations,pre_preset_queens=gen_constellations(PN,pre_ijkl_list,pre_subconst_cache,pre_constellations,pre_preset_queens)
            exec_solutions(PN,pre_constellations,use_gpu,gpu_block,gpu_max_blocks,0,gpu_sort_mode,cross_stripe_safe)
            pre_total:int=sum(c['solutions'] for c in pre_constellations if c['solutions']>0)
            pre_elapsed=datetime.now()-pre_t0
            pre_text=str(pre_elapsed)[:-3]
            pre_status:str="ok" if expected[PN]==pre_total else f"ng({pre_total}!={expected[PN]})"
            print(f"[preheat] N={PN} total={pre_total} elapsed={pre_text} {pre_status}")
          start_time=datetime.now()
          if gpu_log_level>=1:
            print(f"[constellation-config] N={N} preset_queens={preset_queens} constellations={len(constellations)}")
          exec_solutions(N,constellations,use_gpu,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe)
        else:
          if gpu_log_level>=1:
            print(f"[constellation-config] N={N} preset_queens={preset_queens} constellations={len(constellations)}")
          exec_solutions(N,constellations,use_gpu,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe)

    if bench_mode==9:
      if use_gpu:
        print("[bc-sol-warning] bench_mode=9 is intended for CPU. GPU direct_total stores only constellations[0].")
      else:
        diagnose_solution_by_boundary(N,constellations)
    
    """ 合計 """
    total:int=sum(c['solutions'] for c in constellations if c['solutions']>0)
    time_elapsed=datetime.now()-start_time
    text=str(time_elapsed)[:-3]
    if override_elapsed_text != "":
      text=override_elapsed_text
    status:str="ok" if expected[N]==total else f"ng({total}!={expected[N]})"
    if bench_mode==6 and use_gpu:
      status="reorder-only"
    if bench_mode==7 and use_gpu:
      status="chunk-only"
    print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    {status}")

""" エントリポイント """
if __name__=="__main__":
  main()
