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

Python/codon Ｎクイーン コンステレーション版 CUDA 高速ソルバ

はい、これは **GPU/Codonの計算エラーではなく、完全にディスク容量不足**です。

原因は、95の `mode16` が

```text
window_mult = 8,16,32
phase_jump  = 5,7,11
```

の **9条件ぶんの reordered .bin** を作ろうとしたことです。N22では1本あたり約439MBなので、9本だけで約4GB増えます。さらに一時bucketや既存のN22/N23/N24生成物があるため、30GBの `/` を使い切りました。

`tee: ... デバイスに空き領域がありません` と `std.internal.file.File._errcheck` は同じ原因です。まず95の途中生成物だけ消してください。

## 1. まず95の途中生成物を削除

**消してよいもの**だけを指定しています。
`constellations_N22_7.bin` と、94で成功確認済みの `constellations_N22_7_funcid_reorder_v2.bin` は消しません。

```bash
# 念のため、95がまだ動いていないか確認
pgrep -af '95Py|funcid_reorder|codon' || true

# 95 sweep の param 別 .bin。削除OK
rm -f constellations_N22_7_funcid_reorder_v2_w*_j*.bin

# 95 safe版用 temporary bin があれば削除OK
rm -f constellations_N22_7_funcid_reorder_v2_sweep_tmp.bin

# 95/94 reordering 用の一時 bucket bin。必要なら再生成されるので削除OK
rm -f constellations_N22_7_funcid_reorder_v2_A.bin
rm -f constellations_N22_7_funcid_reorder_v2_B.bin
rm -f constellations_N22_7_funcid_reorder_v2_C.bin
rm -f constellations_N22_7_funcid_reorder_v2_G.bin
rm -f constellations_N22_7_funcid_reorder_v2_O.bin

# 途中で壊れた可能性がある95 mode16ログ
rm -f run95_N22_mode16_sweep_sim.log

df -h .
```

その後、何が大きいか確認するならこれです。

```bash
du -ahx . 2>/dev/null | sort -h | tail -40
```

## 2. 残してよい/残すべきもの

これは残してください。

```text
constellations_N22_7.bin
  → 元のN22 constellation入力。必要。

constellations_N22_7.bin.done
  → 件数確認用。必要。

constellations_N22_7_funcid_reorder_v2.bin
  → 94のw16/j7相当で、N22 recovered ok に使ったreordered bin。残してよい。

progress_N22_7_stream_funcid_reorder_v2.tsv
  → N22成功復元の証拠。必ず残す。
```

## 3. 95は修正版を使ってください

今回の95初版は、`mode16` で9本の `.bin` を残す設計だったので、30GBルートでは危険でした。
修正版では、`mode16` の sweep simulation 時だけ **1本の temporary bin を上書き再利用**します。progress TSV は条件別に残ります。

修正版はこちらです。

[95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe.py](sandbox:/mnt/data/95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe.py)

差分はこちらです。

[95_sweep_safe.patch](sandbox:/mnt/data/95_sweep_safe.patch)

ビルドはこれです。

```bash
codon build -release 95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe.py
```

## 4. safe版でmode16を再実行

cleanup後に、まず空きが最低2GB以上あることを確認してください。

```bash
df -h .
```

その後に実行します。

```bash
./95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe \
  -g 22 22 32 484 1 0 5 16 \
  2>&1 | tee run95_N22_mode16_sweep_sim_safe.log
```

safe版の `mode16` は、ログ内でこう出るはずです。

```text
temporary_bin=1
bin=constellations_N22_7_funcid_reorder_v2_sweep_tmp.bin
```

完了後、temporary bin とbucket binは削除して大丈夫です。

```bash
rm -f constellations_N22_7_funcid_reorder_v2_sweep_tmp.bin
rm -f constellations_N22_7_funcid_reorder_v2_A.bin
rm -f constellations_N22_7_funcid_reorder_v2_B.bin
rm -f constellations_N22_7_funcid_reorder_v2_C.bin
rm -f constellations_N22_7_funcid_reorder_v2_G.bin
rm -f constellations_N22_7_funcid_reorder_v2_O.bin

df -h .
```

## 5. ただし、今後はmode16より「1条件ずつ」が安全です

N22でGPU実測する場合は、9条件を一気に走らせるより、まず候補を絞って1本ずつ実行する方が安全です。

例えば `w8/j5` なら、

```bash
./95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe \
  -g 22 22 32 484 1 0 5 15 8 5 \
  2>&1 | tee run95_N22_w8_j5_mode15_gpu.log
```

確認は、

```bash
grep -E "funcid-reorder-v2-gpu-summary|funcid-reorder-v2-gpu-done|^22:" \
  run95_N22_w8_j5_mode15_gpu.log
```

成功したら、

```bash
22:     2691008701644                0      ...    ok
```

になります。

実測が終わった候補の `.bin` は、証拠ログとprogressを残してから削除してよいです。

```bash
rm -f constellations_N22_7_funcid_reorder_v2_w8_j5.bin
rm -f constellations_N22_7_funcid_reorder_v2_A.bin
rm -f constellations_N22_7_funcid_reorder_v2_B.bin
rm -f constellations_N22_7_funcid_reorder_v2_C.bin
rm -f constellations_N22_7_funcid_reorder_v2_G.bin
rm -f constellations_N22_7_funcid_reorder_v2_O.bin
```

## 判断

今回の状態はこうです。

```text
N22 94結果:
  影響なし。成功扱いのままでOK。

95 mode16初版:
  ディスク満杯で失敗。
  計算結果としては採用不可。
  途中生成 .bin は削除。

次:
  1. 95途中生成物を削除
  2. df -h . で空き回復確認
  3. 95 safe版をビルド
  4. mode16 safe、または1条件ずつmode15
```

まずは cleanup だけで大丈夫です。N22の成功記録は壊れていません。

2026-05-25 04:48作成の貼り付けログを見ると、**cleanup は成功しています**。`/` は 100% から **残り 2.9G / 使用率91%** まで戻っています。`cconstellations...` 以降の `コマンドが見つかりません` は、私の説明文をそのままシェルに貼ってしまっただけなので、**ファイル破壊などの実害はありません**。

ただし、次に進む前に重要な点があります。

## 現状判断

`ls -la` には、

```text
95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep
95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep.py
```

だけがあり、**safe版の `95..._sweep_safe.py` はまだサーバー上にありません**。
したがって、このまま既存の `95..._sweep` を再実行すると、また9条件ぶんの `.bin` を作ろうとしてディスクを圧迫する可能性があります。

また、`w8_j5`、`w8_j7`、`w8_j11`、`w16_j5`、`w16_j7` の `.done` だけが残っています。これは本体 `.bin` を削除した後の **残骸マーカー** です。さらに `w16_j11_sim.tsv` は存在しますが、他より小さめで、ディスクフルで途中停止した可能性が高いです。

まずは stale な95 sweep残骸を片付けて、safe版に切り替えるのがよいです。

## 1. 95残骸を整理

N22成功復元に使う本線ファイルは消さず、95 sweep系だけ整理します。

```bash
mkdir -p old95_failed_$(date +%Y%m%d_%H%M%S)
OLD95=$(ls -td old95_failed_* | head -1)

# 失敗/途中の95 sweep simulation TSVを退避
mv progress_N22_7_stream_funcid_reorder_v2_w*_j*_sim.tsv "$OLD95"/ 2>/dev/null || true

# .bin本体が消えた後に残ったdone markerは削除
rm -f constellations_N22_7_funcid_reorder_v2_w*_j*.bin.done

# 念のため、param別binとtemporary/bucketは削除
rm -f constellations_N22_7_funcid_reorder_v2_w*_j*.bin
rm -f constellations_N22_7_funcid_reorder_v2_sweep_tmp.bin
rm -f constellations_N22_7_funcid_reorder_v2_A.bin
rm -f constellations_N22_7_funcid_reorder_v2_B.bin
rm -f constellations_N22_7_funcid_reorder_v2_C.bin
rm -f constellations_N22_7_funcid_reorder_v2_G.bin
rm -f constellations_N22_7_funcid_reorder_v2_O.bin

df -h .
```

残すべきものはこの4つです。

```text
constellations_N22_7.bin
constellations_N22_7.bin.done
constellations_N22_7_funcid_reorder_v2.bin
progress_N22_7_stream_funcid_reorder_v2.tsv
```

## 2. safe版ソースを配置

こちらがsafe版です。

[95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe.py](sandbox:/mnt/data/95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe.py)

サーバーに配置した後、存在確認してください。

```bash
ls -lh 95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe.py
```

その後ビルドです。

```bash
codon build -release 95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe.py
```

## 3. safe版だけを実行

既存の `95..._sweep` ではなく、必ず **`95..._sweep_safe`** を実行してください。

```bash
./95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe \
  -g 22 22 32 484 1 0 5 16 \
  2>&1 | tee run95_N22_mode16_sweep_sim_safe.log
```

ログ内に以下が出れば、safe版で動いています。

```text
temporary_bin=1
bin=constellations_N22_7_funcid_reorder_v2_sweep_tmp.bin
```

完了後に一時ファイルは削除できます。

```bash
rm -f constellations_N22_7_funcid_reorder_v2_sweep_tmp.bin
rm -f constellations_N22_7_funcid_reorder_v2_A.bin
rm -f constellations_N22_7_funcid_reorder_v2_B.bin
rm -f constellations_N22_7_funcid_reorder_v2_C.bin
rm -f constellations_N22_7_funcid_reorder_v2_G.bin
rm -f constellations_N22_7_funcid_reorder_v2_O.bin

df -h .
```

## 4. 2.9Gで足りるか

今回のsafe版なら、N22では概算で、

```text
temporary reordered bin : 約439M
bucket bin合計          : 約439M
progress TSV            : 数MB未満
ログ                     : 小さい
```

なので、**残り2.9Gならsafe版mode16は実行可能な範囲**です。
ただし、N23/N24の `.bin` も同じディスクにあるため、N23以降の新しいreorder sweepはまだ危険です。

## 結論

```text
cleanup:
  成功。残り2.9Gまで回復。

誤貼り付け:
  無害。command not foundだけ。

現状の95:
  まだ unsafe 初版しかサーバーにない。
  safe版を配置するまで再実行しない。

次:
  1. 95 sweep残骸を退避/削除
  2. safe版ソースをサーバーへ配置
  3. safe版をビルド
  4. mode16 safeを再実行
```

この状態なら、**N22の成功記録は守られています**。次はsafe版への差し替えだけです。

貼っていただいた結果では、**cleanup は成功**です。`/` の空きが **3.0G** まで戻っています。

```text
/dev/nvme0n1p1    30G   27G  3.0G   91% /
```

ただし、次に進む前の注意点は変わりません。直前の `ls -la` では、サーバー上にある95はまだ **unsafe初版**の

```text
95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep
95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep.py
```

だけで、`..._sweep_safe.py` は出ていませんでした。
なので、**このまま `./95..._sweep` を再実行しないでください**。

次はこの順番で進めてください。

## 1. 現在の残骸確認

最新cleanup後に、念のため確認します。

```bash
echo "===== disk ====="
df -h .

echo "===== unsafe/safe 95 files ====="
ls -lh 95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep* 2>/dev/null || true

echo "===== remaining 95 generated files ====="
ls -lh constellations_N22_7_funcid_reorder_v2_w*_j*.bin* 2>/dev/null || echo "no param bin/done"
ls -lh progress_N22_7_stream_funcid_reorder_v2_w*_j*_sim.tsv 2>/dev/null || echo "no param sim tsv in current dir"
```

期待はこうです。

```text
constellations_N22_7_funcid_reorder_v2_w*_j*.bin*
  → no param bin/done

progress_N22_7_stream_funcid_reorder_v2_w*_j*_sim.tsv
  → no param sim tsv in current dir
```

もし `progress...w*_j*_sim.tsv` がまだ出る場合は、もう一度だけ退避します。

```bash
mkdir -p old95_failed_$(date +%Y%m%d_%H%M%S)
OLD95=$(ls -td old95_failed_* | head -1)
mv progress_N22_7_stream_funcid_reorder_v2_w*_j*_sim.tsv "$OLD95"/ 2>/dev/null || true
```

## 2. safe版をサーバーに置く

こちらのファイルをサーバー側の同じディレクトリへ置いてください。

[95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe.py](sandbox:/mnt/data/95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe.py)

配置後に確認します。

```bash
ls -lh 95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe.py
```

## 3. safe版をビルド

```bash
codon build -release 95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe.py

ls -lh 95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe
```

ここで **`..._sweep_safe`** という実行ファイルができていればOKです。

## 4. safe版mode16を実行

3.0G空いていれば、N22のsafe版mode16は実行可能な範囲です。まずログ用ディレクトリを作ってから実行します。

```bash
mkdir -p logs

set -o pipefail

./95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe \
  -g 22 22 32 484 1 0 5 16 \
  2>&1 | tee logs/run95_N22_mode16_sweep_sim_safe.log
```

ログ内にこれが出れば、safe版で動いています。

```text
temporary_bin=1
bin=constellations_N22_7_funcid_reorder_v2_sweep_tmp.bin
```

終了後に確認します。

```bash
grep -E "sweep|temporary_bin|build-summary|sim-only|w[0-9]+_j[0-9]+" \
  logs/run95_N22_mode16_sweep_sim_safe.log | tail -80

df -h .
```

## 5. 完了後の一時ファイル削除

safe版mode16が終わったら、temporary bin と bucket bin は消してよいです。

```bash
rm -f constellations_N22_7_funcid_reorder_v2_sweep_tmp.bin
rm -f constellations_N22_7_funcid_reorder_v2_A.bin
rm -f constellations_N22_7_funcid_reorder_v2_B.bin
rm -f constellations_N22_7_funcid_reorder_v2_C.bin
rm -f constellations_N22_7_funcid_reorder_v2_G.bin
rm -f constellations_N22_7_funcid_reorder_v2_O.bin

df -h .
```

## 現時点の判断

```text
cleanup:
  成功。空き 3.0G。

N22成功記録:
  維持。壊れていない。

次に実行してよいもの:
  95 safe版のみ。

まだ実行しないもの:
  ./95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep
  → unsafe初版なので再びディスクを圧迫する可能性あり。
```

まずは `..._sweep_safe.py` を配置して、`ls -lh ..._sweep_safe.py` が見える状態にするところからで大丈夫です。

はい、**現在の本線が94/95の `funcid_reorder_v2` なら、非v2の `funcid_reorder` 系は不要**です。

ただし、注意点があります。
**`v2以外` を全部削除、ではありません。**
元データの

```text
constellations_N21_6.bin
constellations_N21_6.bin.done
```

は非v2ですが、これは再生成元なので残す候補です。

## 削除してよいもの

### 1. N21の非v2 reorder、つまり93系

これは今の94/95 v2本線では不要です。

```bash
rm -f constellations_N21_6_funcid_reorder.bin
rm -f constellations_N21_6_funcid_reorder.bin.done
rm -f constellations_N21_6_funcid_reorder_A.bin
rm -f constellations_N21_6_funcid_reorder_B.bin
rm -f constellations_N21_6_funcid_reorder_C.bin
rm -f constellations_N21_6_funcid_reorder_G.bin
rm -f constellations_N21_6_funcid_reorder_O.bin
```

### 2. N21のv2 bucket一時ファイル

`v2_A/B/C/G/O` は、v2本体を作る途中のbucketです。
`constellations_N21_6_funcid_reorder_v2.bin` ができていて `.done` もあるので、bucketは不要です。

```bash
rm -f constellations_N21_6_funcid_reorder_v2_A.bin
rm -f constellations_N21_6_funcid_reorder_v2_B.bin
rm -f constellations_N21_6_funcid_reorder_v2_C.bin
rm -f constellations_N21_6_funcid_reorder_v2_G.bin
rm -f constellations_N21_6_funcid_reorder_v2_O.bin
```

### 3. N22の95失敗sweep残骸

この `.done` は、本体 `.bin` が削除済みなら意味がありません。削除でよいです。

```bash
rm -f constellations_N22_7_funcid_reorder_v2_w*_j*.bin.done
```

95 safe版で再実行する前提なら、途中のsimulation TSVも削除してよいです。

```bash
rm -f progress_N22_7_stream_funcid_reorder_v2_w*_j*_sim.tsv
```

## 残した方がよいもの

これは残してください。

```text
constellations_N21_6.bin
constellations_N21_6.bin.done
```

N21回帰テスト用の元データです。小さいので残してよいです。

```text
constellations_N21_6_funcid_reorder_v2.bin
constellations_N21_6_funcid_reorder_v2.bin.done
```

N21 v2回帰用に残してよいです。ただし、不要なら再生成可能です。

```text
constellations_N22_7.bin
constellations_N22_7.bin.done
constellations_N22_7_funcid_reorder_v2.bin
constellations_N22_7_funcid_reorder_v2.bin.done
progress_N22_7_stream_funcid_reorder_v2.tsv
```

これはN22成功確認・復元に関係するので、今は残してください。

## 重複バックアップの扱い

これは重複の可能性が高いです。

```text
constellations_N21_6_funcid_reorder_v2_94_ok_20260521.bin
```

まず一致確認してください。

```bash
sha256sum \
  constellations_N21_6_funcid_reorder_v2.bin \
  constellations_N21_6_funcid_reorder_v2_94_ok_20260521.bin
```

ハッシュが同じなら、片方だけで十分です。ログやprogressを残しているなら、バックアップ側は削除してよいです。

```bash
cmp -s \
  constellations_N21_6_funcid_reorder_v2.bin \
  constellations_N21_6_funcid_reorder_v2_94_ok_20260521.bin \
  && rm -f constellations_N21_6_funcid_reorder_v2_94_ok_20260521.bin
```

## まとめて実行するなら

```bash
echo "===== before ====="
df -h .

# N21 非v2 reorder 93系。削除OK
rm -f constellations_N21_6_funcid_reorder.bin
rm -f constellations_N21_6_funcid_reorder.bin.done
rm -f constellations_N21_6_funcid_reorder_A.bin
rm -f constellations_N21_6_funcid_reorder_B.bin
rm -f constellations_N21_6_funcid_reorder_C.bin
rm -f constellations_N21_6_funcid_reorder_G.bin
rm -f constellations_N21_6_funcid_reorder_O.bin

# N21 v2 bucket一時ファイル。削除OK
rm -f constellations_N21_6_funcid_reorder_v2_A.bin
rm -f constellations_N21_6_funcid_reorder_v2_B.bin
rm -f constellations_N21_6_funcid_reorder_v2_C.bin
rm -f constellations_N21_6_funcid_reorder_v2_G.bin
rm -f constellations_N21_6_funcid_reorder_v2_O.bin

# N22 95失敗sweepの残骸。削除OK
rm -f constellations_N22_7_funcid_reorder_v2_w*_j*.bin.done
rm -f progress_N22_7_stream_funcid_reorder_v2_w*_j*_sim.tsv

# N21 v2 backupが同一なら削除
if [ -f constellations_N21_6_funcid_reorder_v2.bin ] && \
   [ -f constellations_N21_6_funcid_reorder_v2_94_ok_20260521.bin ]; then
  if cmp -s constellations_N21_6_funcid_reorder_v2.bin \
            constellations_N21_6_funcid_reorder_v2_94_ok_20260521.bin; then
    rm -f constellations_N21_6_funcid_reorder_v2_94_ok_20260521.bin
  else
    echo "N21 v2 backup differs; kept both"
  fi
fi

echo "===== after ====="
df -h .
```

## 削除効果

N21まわりを掃除しても、増える空きはだいたい **100〜130MB程度**です。
安全には良いですが、3GB空き問題を大きく改善するほどではありません。

大きいのは以下です。

```text
constellations_N22_7.bin                 439M
constellations_N22_7_funcid_reorder_v2.bin 439M
constellations_N23_7.bin                 676M
constellations_N24_7.bin                 1.0G
```

ただし、これらはN22/N23/N24で使う元データなので、今すぐ消すより、必要ならS3や別ディスクへ退避する方が安全です。

結論です。

```text
非v2 funcid_reorder:
  削除OK。

v2_A/B/C/G/O:
  一時bucketなので削除OK。

元の constellations_N21_6.bin:
  非v2だが残す候補。

N21 v2本体:
  回帰テスト用に残してよい。

N22本線ファイル:
  今は残す。
```

はい。まだ消せます。優先順位はこうです。

## まず確実に消してよいもの

現在の一覧では、次の **N22 v2 bucket 一時ファイル** がまだ残っています。

```text
constellations_N22_7_funcid_reorder_v2_A.bin   86.8M
constellations_N22_7_funcid_reorder_v2_B.bin   61.4M
constellations_N22_7_funcid_reorder_v2_C.bin   27.1M
constellations_N22_7_funcid_reorder_v2_G.bin  215.3M
constellations_N22_7_funcid_reorder_v2_O.bin   68.9M
```

合計で約 **459MB** です。これは reorder `.bin` を作る途中の bucket なので、削除して大丈夫です。

さらに、

```text
constellations_N22_7_funcid_reorder_v2_sweep_tmp.bin       293M
constellations_N22_7_funcid_reorder_v2_sweep_tmp.bin.done
```

も、95 safe sweep の temporary 出力なので、**現在95が動いていなければ削除OK**です。

まずこれを実行してください。

```bash
echo "===== before ====="
df -h .

echo "===== running check ====="
pgrep -af '95Py|94Py|84Py|funcid_reorder|codon' || true
```

ここで **95/94/84 が動いていない**なら、以下を削除します。

```bash
rm -f constellations_N22_7_funcid_reorder_v2_A.bin
rm -f constellations_N22_7_funcid_reorder_v2_B.bin
rm -f constellations_N22_7_funcid_reorder_v2_C.bin
rm -f constellations_N22_7_funcid_reorder_v2_G.bin
rm -f constellations_N22_7_funcid_reorder_v2_O.bin

rm -f constellations_N22_7_funcid_reorder_v2_sweep_tmp.bin
rm -f constellations_N22_7_funcid_reorder_v2_sweep_tmp.bin.done

rm -f constellations_N22_7_funcid_reorder_v2_w*_j*.bin.done

rm -f bc_sol_dynamic_GPU_N24_p7.log

echo "===== after ====="
df -h .
```

これで **約750〜760MB** 空くはずです。

---

## 次に消せるもの：N21の重複

これは重複候補です。

```text
constellations_N21_6_funcid_reorder_v2.bin
constellations_N21_6_funcid_reorder_v2_94_ok_20260521.bin
```

まず一致確認してください。

```bash
sha256sum \
  constellations_N21_6_funcid_reorder_v2.bin \
  constellations_N21_6_funcid_reorder_v2_94_ok_20260521.bin
```

同じならバックアップ側を削除できます。

```bash
cmp -s \
  constellations_N21_6_funcid_reorder_v2.bin \
  constellations_N21_6_funcid_reorder_v2_94_ok_20260521.bin \
  && rm -f constellations_N21_6_funcid_reorder_v2_94_ok_20260521.bin
```

これは約 **32MB** だけですが、消して問題ありません。

N21をもう回帰確認に使わないなら、N21の v2 本体も消せます。

```bash
rm -f constellations_N21_6_funcid_reorder_v2.bin
rm -f constellations_N21_6_funcid_reorder_v2.bin.done
```

ただし、N21回帰テストをすぐ使うなら残してよいです。小さいので優先度は低いです。

---

## 大きく空けるなら：N23/N24 `.bin`

現在の大物はこれです。

```text
constellations_N23_7.bin   708M
constellations_N24_7.bin   1.0G
```

N22の95 sweepには **N23/N24の `.bin` は不要**です。
したがって、今の作業を **N22の95検証に集中する**なら、N23/N24は退避または削除できます。

### 退避ディレクトリに移す場合

同じディスク内の移動なので空き容量は増えませんが、誤削除防止にはなります。

```bash
mkdir -p hold_N23_N24_bins

mv constellations_N23_7.bin constellations_N23_7.bin.done hold_N23_N24_bins/ 2>/dev/null || true
mv constellations_N24_7.bin constellations_N24_7.bin.done hold_N23_N24_bins/ 2>/dev/null || true
```

### 本当に空きを増やす場合

N23/N24の `.bin` は84 streamで再生成可能です。
ただし再生成に時間はかかるので、消す前に「今はN22だけ」と割り切る場合だけにしてください。

```bash
rm -f constellations_N23_7.bin constellations_N23_7.bin.done
rm -f constellations_N24_7.bin constellations_N24_7.bin.done

df -h .
```

これで約 **1.7GB** 空きます。

---

## N22で残すべきもの

これは今は残してください。

```text
constellations_N22_7.bin
constellations_N22_7.bin.done
progress_N22_7_stream_funcid_reorder_v2.tsv
run94_N22_mode15_gpu.log
```

理由は、`constellations_N22_7.bin` はN22の元データ、`progress_N22_7_stream_funcid_reorder_v2.tsv` はN22 recovered ok の証拠だからです。

`constellations_N22_7_funcid_reorder_v2.bin` は判断が分かれます。

```text
残す場合:
  94 w16/j7 を再実行しやすい

消す場合:
  約459MB空く
  必要なら mode14 で再生成できる
```

95 safe版で sweepするだけなら、これは必須ではありません。空きが厳しければ削除候補にできます。

```bash
rm -f constellations_N22_7_funcid_reorder_v2.bin
rm -f constellations_N22_7_funcid_reorder_v2.bin.done
```

ただし、N22の94結果を再現しやすく残したいなら、今は残しておく方が安全です。

---

## 追加で大物を探す

まだ空きが少ない場合は、これで大きいものを確認してください。

```bash
du -ahx . 2>/dev/null | sort -h | tail -50
```

特に見るべきディレクトリはこれです。

```bash
du -sh Archive procMonitor old95_failed_* logs '~' 2>/dev/null
```

`old95_failed_*` や `logs` が大きければ、次の候補になります。

---

## 私ならこの順番で消します

まずこれです。

```bash
pgrep -af '95Py|94Py|84Py|funcid_reorder|codon' || true

rm -f constellations_N22_7_funcid_reorder_v2_A.bin
rm -f constellations_N22_7_funcid_reorder_v2_B.bin
rm -f constellations_N22_7_funcid_reorder_v2_C.bin
rm -f constellations_N22_7_funcid_reorder_v2_G.bin
rm -f constellations_N22_7_funcid_reorder_v2_O.bin
rm -f constellations_N22_7_funcid_reorder_v2_sweep_tmp.bin
rm -f constellations_N22_7_funcid_reorder_v2_sweep_tmp.bin.done
rm -f constellations_N22_7_funcid_reorder_v2_w*_j*.bin.done
rm -f bc_sol_dynamic_GPU_N24_p7.log

df -h .
```

これでまだ不足なら、次にN21重複。

```bash
cmp -s \
  constellations_N21_6_funcid_reorder_v2.bin \
  constellations_N21_6_funcid_reorder_v2_94_ok_20260521.bin \
  && rm -f constellations_N21_6_funcid_reorder_v2_94_ok_20260521.bin

df -h .
```

それでも足りなければ、N23/N24を一時的に削除します。

```bash
rm -f constellations_N23_7.bin constellations_N23_7.bin.done
rm -f constellations_N24_7.bin constellations_N24_7.bin.done

df -h .
```

結論として、**今すぐ安全に消せる最大候補は N22 bucket + sweep_tmp で約760MB**です。
さらに本気で空けるなら、**N23/N24 `.bin` を消して約1.7GB追加**です。

確認しました。今回の結果はかなり良いです。

## 1. 95 safe版の mode16 は成功です

`95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe.py` が配置され、`codon build -release` も通っています。続く `mode16` では、9条件すべてが `valid=1` で完了しています。

実行された9条件はこれです。

```text
w8_j5
w8_j7
w8_j11
w16_j5
w16_j7
w16_j11
w32_j5
w32_j7
w32_j11
```

最後も、

```text
[funcid-reorder-v2-sweep-summary] N=22 preset_queens=7 records=28719035 cases=9 elapsed=0:19:02.589
22:                 0                0          0:19:02.589    funcid-reorder-v2-sweep-sim
```

なので、**95 safe の simulation sweep は完走**です。
ここで `22: 0` なのは正常です。`mode16` はGPU探索をしない simulation-only なので、解数は計算していません。

## 2. cleanup も成功しています

safe sweep 実行後、不要な bucket / temporary bin を削除して、空きは最終的に **3.5G** まで回復しています。

```text
/dev/nvme0n1p1    30G   27G  3.5G   89% /
```

この状態なら、**N22の95 mode15を1条件ずつ実測するには足ります**。

## 3. ただし N23/N24 はまだディスク上に残っている可能性が高いです

ここは重要です。

実行ログでは、

```bash
mkdir -p hold_N23_N24_bins

mv constellations_N23_7.bin constellations_N23_7.bin.done hold_N23_N24_bins/
mv constellations_N24_7.bin constellations_N24_7.bin.done hold_N23_N24_bins/
```

を実行しています。

これは **同じ `/` ファイルシステム内での移動**なので、空き容量は増えません。
その後に、

```bash
rm -f constellations_N23_7.bin constellations_N23_7.bin.done
rm -f constellations_N24_7.bin constellations_N24_7.bin.done
```

を実行していますが、ファイルはすでに `hold_N23_N24_bins/` に移動済みなので、この `rm` は実質的に何も消していません。

確認してください。

```bash
du -sh hold_N23_N24_bins 2>/dev/null
ls -lh hold_N23_N24_bins 2>/dev/null
```

もし今はN22の95検証だけに集中するなら、これでさらに約1.7GB空けられます。

```bash
rm -f hold_N23_N24_bins/constellations_N23_7.bin
rm -f hold_N23_N24_bins/constellations_N23_7.bin.done
rm -f hold_N23_N24_bins/constellations_N24_7.bin
rm -f hold_N23_N24_bins/constellations_N24_7.bin.done

df -h .
```

N23/N24を残したい場合は、同じディスク内の `hold_N23_N24_bins` ではなく、S3や別ディスクへ退避しないと空き容量は増えません。

## 4. 今は simulation TSV を消さない方がよいです

以下の9個の simulation TSV は、95 sweep の成果物です。

```text
progress_N22_7_stream_funcid_reorder_v2_w8_j5_sim.tsv
progress_N22_7_stream_funcid_reorder_v2_w8_j7_sim.tsv
progress_N22_7_stream_funcid_reorder_v2_w8_j11_sim.tsv
progress_N22_7_stream_funcid_reorder_v2_w16_j5_sim.tsv
progress_N22_7_stream_funcid_reorder_v2_w16_j7_sim.tsv
progress_N22_7_stream_funcid_reorder_v2_w16_j11_sim.tsv
progress_N22_7_stream_funcid_reorder_v2_w32_j5_sim.tsv
progress_N22_7_stream_funcid_reorder_v2_w32_j7_sim.tsv
progress_N22_7_stream_funcid_reorder_v2_w32_j11_sim.tsv
```

サイズは各600KB前後なので、全部残しても数MBです。
これは候補選定・後日比較に使えるので、今は残してください。

## 5. 次は mode15 GPU実測です

`mode16` では正しさ・生成可能性は確認できました。
次に必要なのは **N22 mode15で実際のGPU時間を見ること**です。

まずは1条件だけ走らせるのが安全です。私は最初に **w8_j5** を推します。理由は、baselineの `w16_j7` はすでに94で recovered ok があり、まず小さい window の効果を見るのが有益だからです。

```bash
mkdir -p logs

W=8
J=5
PARAM="w${W}_j${J}"
LOG="logs/run95_N22_${PARAM}_mode15_gpu_$(date +%Y%m%d_%H%M%S).log"
PROG="progress_N22_7_stream_funcid_reorder_v2_${PARAM}.tsv"

# 古い同名progressがあれば退避
if [ -f "$PROG" ]; then
  mv "$PROG" "${PROG}.$(date +%Y%m%d_%H%M%S).bak"
fi

set -o pipefail

{
  echo "PARAM=$PARAM"
  date
  hostname
  pwd
  df -h .
  nvidia-smi
} | tee "$LOG"

./95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe \
  -g 22 22 32 484 1 0 5 15 "$W" "$J" \
  2>&1 | tee -a "$LOG"

STATUS=${PIPESTATUS[0]}

{
  echo "EXIT=$STATUS"
  date
  echo "===== final lines ====="
  grep -E "funcid-reorder-v2-gpu-summary|funcid-reorder-v2-gpu-done|^22:" "$LOG" || true
  echo "===== progress recovery check ====="

  if [ -f "$PROG" ]; then
    awk -F'\t' '
    NR==1 {
      for (i=1; i<=NF; i++) c[$i]=i
      next
    }
    {
      rows++
      last_chunk=$(c["chunk"])
      last_m=$(c["m"])
      last_gpu_total=$(c["gpu_total"])
      last_done=$(c["done_records"])
      last_total=$(c["total_records"])
      last_remaining=$(c["remaining_records"])
      sum_ms += $(c["elapsed_ms"])
    }
    END {
      print "rows=" rows
      print "last_chunk=" last_chunk
      print "last_m=" last_m
      print "gpu_total=" last_gpu_total
      print "done_records=" last_done
      print "total_records=" last_total
      print "remaining_records=" last_remaining
      printf "sum_chunk_elapsed=%.3f sec\n", sum_ms/1000.0

      if (last_chunk==1854 && last_m==4283 && last_done==28719035 && last_total==28719035 && last_remaining==0 && last_gpu_total==2691008701644) {
        print "RECOVERED_N22_OK"
      } else {
        print "NOT_RECOVERED_OR_INCOMPLETE"
      }
    }
    ' "$PROG"
  else
    echo "NO_PROGRESS_FILE: $PROG"
  fi
} | tee -a "$LOG"

exit "$STATUS"
```

成功したら、次のどちらかが出ます。

```text
22:     2691008701644                0      ...    ok
```

または、ログ終端が欠けても progress 側で、

```text
RECOVERED_N22_OK
```

が出れば成功扱いできます。

## 6. 1条件ごとに掃除してください

mode15を1条件実行すると、param別の `.bin` が作られるはずです。
次の条件へ進む前に、ログとprogressを残して `.bin` と bucket を削除してください。

```bash
PARAM=w8_j5

rm -f constellations_N22_7_funcid_reorder_v2_${PARAM}.bin
rm -f constellations_N22_7_funcid_reorder_v2_${PARAM}.bin.done

rm -f constellations_N22_7_funcid_reorder_v2_A.bin
rm -f constellations_N22_7_funcid_reorder_v2_B.bin
rm -f constellations_N22_7_funcid_reorder_v2_C.bin
rm -f constellations_N22_7_funcid_reorder_v2_G.bin
rm -f constellations_N22_7_funcid_reorder_v2_O.bin

df -h .
```

## 7. 実測候補の順番

全部9条件をN22で回すとかなり重いので、まずはこの順番がよいです。

```text
1. w8_j5
   小さいwindow + 小さいphase。baselineから一番挙動が変わる候補。

2. w32_j7
   baselineと同じphase_jump=7で、windowだけ大きくした反対側の候補。

3. w16_j5
   baseline window=16を維持して、phaseだけ変える候補。

baseline:
  w16_j7 は94のN22 recovered okが既にある。
```

判断基準はこれです。

```text
baseline 94 w16_j7:
  sum_chunk_elapsed = 13290.452 sec
  total = 2691008701644
  status = recovered ok

95候補:
  totalが同じ
  remaining_records=0
  sum_chunk_elapsed が baseline より小さい
  できれば正式ログの wall-clock も小さい
```

## 現時点のまとめ

```text
95 safe source:
  配置OK、build OK。

95 mode16:
  9条件すべて valid=1。
  simulation sweep 成功。

cleanup:
  空き 3.5G。
  N22 mode15を1条件ずつ回すには足りる。

注意:
  hold_N23_N24_bins は同じディスク上なので、まだ容量を食っている可能性が高い。

次:
  w8_j5 の N22 mode15 GPU実測。
```

結論として、**95の準備段階は成功**です。次は `w8_j5` の実測に進めて大丈夫です。


はい、**4.5G空きなら次へ進めて大丈夫**です。

```text
/dev/nvme0n1p1    30G   26G  4.5G   85% /
```

N22の95 `mode15` を **1条件ずつ実測**するには十分です。95 safe版の `mode16` はすでに9条件すべて `valid=1` で通っているので、次はGPU実測に入れます。

## 次にやること

まずは **w8_j5** を実測してください。

```bash
mkdir -p logs

W=8
J=5
PARAM="w${W}_j${J}"
LOG="logs/run95_N22_${PARAM}_mode15_gpu_$(date +%Y%m%d_%H%M%S).log"
PROG="progress_N22_7_stream_funcid_reorder_v2_${PARAM}.tsv"

# 古い同名progressがあれば退避
if [ -f "$PROG" ]; then
  mv "$PROG" "${PROG}.$(date +%Y%m%d_%H%M%S).bak"
fi

# 念のため一時ファイル掃除
rm -f constellations_N22_7_funcid_reorder_v2_${PARAM}.bin
rm -f constellations_N22_7_funcid_reorder_v2_${PARAM}.bin.done
rm -f constellations_N22_7_funcid_reorder_v2_A.bin
rm -f constellations_N22_7_funcid_reorder_v2_B.bin
rm -f constellations_N22_7_funcid_reorder_v2_C.bin
rm -f constellations_N22_7_funcid_reorder_v2_G.bin
rm -f constellations_N22_7_funcid_reorder_v2_O.bin

set -o pipefail

{
  echo "PARAM=$PARAM"
  date
  hostname
  pwd
  df -h .
  nvidia-smi
} | tee "$LOG"

./95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe \
  -g 22 22 32 484 1 0 5 15 "$W" "$J" \
  2>&1 | tee -a "$LOG"

STATUS=${PIPESTATUS[0]}

{
  echo "EXIT=$STATUS"
  date
  echo "===== final lines ====="
  grep -E "funcid-reorder-v2-gpu-summary|funcid-reorder-v2-gpu-done|^22:" "$LOG" || true

  echo "===== progress recovery check ====="
  if [ -f "$PROG" ]; then
    awk -F'\t' '
    NR==1 {
      for (i=1; i<=NF; i++) c[$i]=i
      next
    }
    {
      rows++
      last_chunk=$(c["chunk"])
      last_m=$(c["m"])
      last_gpu_total=$(c["gpu_total"])
      last_done=$(c["done_records"])
      last_total=$(c["total_records"])
      last_remaining=$(c["remaining_records"])
      sum_ms += $(c["elapsed_ms"])
    }
    END {
      print "rows=" rows
      print "last_chunk=" last_chunk
      print "last_m=" last_m
      print "gpu_total=" last_gpu_total
      print "done_records=" last_done
      print "total_records=" last_total
      print "remaining_records=" last_remaining
      printf "sum_chunk_elapsed=%.3f sec\n", sum_ms/1000.0

      if (last_chunk==1854 && last_m==4283 && last_done==28719035 && last_total==28719035 && last_remaining==0 && last_gpu_total==2691008701644) {
        print "RECOVERED_N22_OK"
      } else {
        print "NOT_RECOVERED_OR_INCOMPLETE"
      }
    }
    ' "$PROG"
  else
    echo "NO_PROGRESS_FILE: $PROG"
  fi

  echo "===== disk after ====="
  df -h .
} | tee -a "$LOG"

exit "$STATUS"
```

期待する成功形はどちらかです。

```text
22:     2691008701644                0      ...    ok
```

または、ログ末尾が欠けても、

```text
RECOVERED_N22_OK
```

が出れば成功扱いでよいです。

## 実測後の掃除

w8_j5 が終わったら、ログとprogress TSVだけ残して `.bin` とbucketを消してください。

```bash
PARAM=w8_j5

rm -f constellations_N22_7_funcid_reorder_v2_${PARAM}.bin
rm -f constellations_N22_7_funcid_reorder_v2_${PARAM}.bin.done

rm -f constellations_N22_7_funcid_reorder_v2_A.bin
rm -f constellations_N22_7_funcid_reorder_v2_B.bin
rm -f constellations_N22_7_funcid_reorder_v2_C.bin
rm -f constellations_N22_7_funcid_reorder_v2_G.bin
rm -f constellations_N22_7_funcid_reorder_v2_O.bin

df -h .
```

## N23について

今回消したのは `hold_N23_N24_bins` の **N24** だけなので、N23がまだ残っているかもしれません。確認はこれです。

```bash
du -sh hold_N23_N24_bins 2>/dev/null
ls -lh hold_N23_N24_bins 2>/dev/null
```

N22検証だけを優先するなら、N23も消せばさらに約700MB空きます。

```bash
rm -f hold_N23_N24_bins/constellations_N23_7.bin
rm -f hold_N23_N24_bins/constellations_N23_7.bin.done
df -h .
```

ただ、今の **4.5G空きならN23は残しても大丈夫**です。

結論として、次は **w8_j5 の N22 mode15 GPU実測**で進めてください。


はい、この状態なら **`nvme1n1` 232.8G を作業用ディスクとして使う**のが一番よいです。

```text
nvme0n1   30G    → 現在の /
nvme1n1   232.8G → 未マウントの追加ディスク
```

これは「root `/` を直接広げる」というより、**大きい作業ディレクトリを `nvme1n1` 側へ移す**方針が安全です。
AWS環境では `nvme1n1` が **EBS** か **instance store** かで永続性が違います。instance store は一時ストレージで、rebootでは残りますが stop / hibernate / terminate では消える扱いです。AWS公式も instance store は一時ブロックストレージで、停止・休止・終了時にデータが保持されないと説明しています。([AWS ドキュメント][1]) ([AWS ドキュメント][2])

## 1. まず `nvme1n1` の正体と中身を確認

**まだ `mkfs` しないで**、まず確認してください。

```bash
echo "===== lsblk ====="
lsblk -f

echo "===== blkid ====="
sudo blkid /dev/nvme1n1 || true

echo "===== file signature ====="
sudo file -s /dev/nvme1n1

echo "===== nvme list ====="
sudo nvme list 2>/dev/null || true

echo "===== by-id ====="
ls -l /dev/disk/by-id/ | grep nvme1n1 || true
```

期待される空ディスクの例は、だいたい以下です。

```text
/dev/nvme1n1: data
```

または `blkid` が何も返さない状態です。

もし `file -s /dev/nvme1n1` が既存の filesystem、partition table、LVM などを表示した場合は、**そのまま止めてください**。中身がある可能性があります。

## 2. 空ディスクなら ext4 で作成

`/dev/nvme1n1` が本当に空なら、以下で作業用ディスクにします。

```bash
sudo mkfs.ext4 -F /dev/nvme1n1
```

マウント先を作ります。

```bash
sudo mkdir -p /data/nq
sudo mount /dev/nvme1n1 /data/nq
sudo chown -R suzuki:suzuki /data/nq

df -h /data/nq
```

ここで以下のように見えればOKです。

```text
/dev/nvme1n1  229G ... /data/nq
```

## 3. 再起動後もマウントする設定

`UUID` で `/etc/fstab` に入れます。`nofail` を付けておくと、何らかの理由でディスクが無い時も起動失敗しにくいです。

```bash
UUID=$(sudo blkid -s UUID -o value /dev/nvme1n1)
echo "$UUID"

sudo cp -av /etc/fstab /etc/fstab.bak.$(date +%Y%m%d_%H%M%S)

echo "UUID=$UUID /data/nq ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab

sudo umount /data/nq
sudo mount -a
df -h /data/nq
```

ただし、`nvme1n1` が **instance store** だった場合、stop / terminate 後は中身が消える可能性があります。重要なログ、最終結果、ソース、patch は root EBS、Git、S3などにも残してください。AWS公式では、instance store のデータはインスタンスのライフタイム中だけの一時ストレージとして扱われます。([AWS ドキュメント][3])

## 4. 現在の作業ディレクトリを `/data/nq` へ移す

今後の `constellations_N*.bin` や `progress_*.tsv` を root 30G に戻さないため、**作業ディレクトリごと移す**のがよいです。

まず現在地を確認します。

```bash
pwd
```

例えば現在の作業ディレクトリが `/home/suzuki/workspace` なら、以下のようにします。
`PROJECT` は自動で現在ディレクトリを使います。

```bash
PROJECT="$PWD"
BASE="$(basename "$PROJECT")"
PARENT="$(dirname "$PROJECT")"

echo "PROJECT=$PROJECT"
echo "BASE=$BASE"
echo "PARENT=$PARENT"

mkdir -p /data/nq/"$BASE"

rsync -aH --info=progress2 "$PROJECT"/ /data/nq/"$BASE"/

echo "===== copied size ====="
du -sh "$PROJECT" /data/nq/"$BASE"
```

コピー後、念のため重要ファイルを確認します。

```bash
ls -lh /data/nq/"$BASE"/95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe.py 2>/dev/null || true
ls -lh /data/nq/"$BASE"/constellations_N22_7.bin /data/nq/"$BASE"/constellations_N22_7.bin.done
ls -lh /data/nq/"$BASE"/progress_N22_7_stream_funcid_reorder_v2.tsv 2>/dev/null || true
```

問題なければ、元ディレクトリを退避して symlink にします。

```bash
cd "$PARENT"

mv "$BASE" "${BASE}.rootdisk.bak.$(date +%Y%m%d_%H%M%S)"
ln -s /data/nq/"$BASE" "$BASE"

cd "$BASE"

pwd
df -h .
```

`df -h .` が `/data/nq` 側、つまり 232.8G ディスクを指していれば成功です。

## 5. すぐ確認するコマンド

```bash
echo "===== current dir ====="
pwd

echo "===== disk ====="
df -h .

echo "===== large files ====="
du -ahx . 2>/dev/null | sort -h | tail -30

echo "===== source check ====="
ls -lh 94Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2.py
ls -lh 95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe.py 2>/dev/null || true
```

## 6. 今後の運用

新しい `/data/nq/...` 側で作業すれば、95 sweep や N22/N23/N24 の `.bin` を置いてもかなり余裕があります。

ただし、`nvme1n1` が instance store だった場合は、長期保存用には使わない方が安全です。おすすめはこの分担です。

```text
/data/nq:
  大きい .bin
  temporary bin
  bucket bin
  sweep実験用 progress
  長時間実行中の作業領域

root / Git / S3:
  ソースコード
  patch
  最終ログ
  recovered result
  重要な progress TSV
```

## 7. root側の退避ディレクトリを消すタイミング

symlink 化後にしばらく動作確認できたら、root側に残したバックアップを消せます。

```bash
cd "$PARENT"
ls -ld ${BASE}.rootdisk.bak.*

# 内容確認後に削除
rm -rf ${BASE}.rootdisk.bak.YYYYMMDD_HHMMSS
```

ここは実際のディレクトリ名を見てから削除してください。

## 結論

```text
おすすめ:
  nvme1n1 を /data/nq に ext4 でマウント
  現在の作業ディレクトリを /data/nq 側へ rsync
  元ディレクトリは symlink 化

やらない方がよい:
  nvme1n1 を root / に無理やり足す
  LVM化して root と混ぜる
  中身確認なしで mkfs

注意:
  nvme1n1 が instance store なら stop/terminate で消える可能性あり
  重要成果物は root EBS / S3 / Git に別途退避
```

まずは `sudo file -s /dev/nvme1n1` と `sudo nvme list` の結果を見て、空ディスクであることを確認してから `mkfs.ext4` に進むのが安全です。

[1]: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/InstanceStorage.html?utm_source=chatgpt.com "Instance store temporary block storage for EC2 instances"
[2]: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-store-lifetime.html?utm_source=chatgpt.com "Data persistence for Amazon EC2 instance store volumes"
[3]: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Storage.html?utm_source=chatgpt.com "Storage options for your Amazon EC2 instances"


はい。出力を見る限り、まず2点はっきりしています。

```text
現在地:
  /home/suzuki/Github

実際の作業ディレクトリ:
  /home/suzuki/Github/N-Queens/13Bit_codon

現在のディスク:
  まだ /dev/nvme0n1p1、つまり root 30G 側
```

なので、`94Py...` が見つからなかったのは正常です。1階層上で確認しているためです。

```bash
cd /home/suzuki/Github/N-Queens/13Bit_codon

ls -lh \
  94Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2.py \
  95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe.py 2>/dev/null || true

df -h .
```

## まず今すぐ消せる大物

`du` の結果では、まだここが残っています。

```text
./N-Queens/13Bit_codon/procMonitor/constellations_N22_7.bin  439M
./N-Queens/13Bit_codon/hold_N23_N24_bins/constellations_N23_7.bin  676M
./N-Queens/13Bit_codon/procMonitor/logs/health_N24_p7_GPU_20260514_023008.log  60M
```

N22の元ファイルが本体側にもあるので、`procMonitor` 側のN22は重複候補です。まず一致確認してから消すのが安全です。

```bash
cd /home/suzuki/Github/N-Queens/13Bit_codon

if [ -f constellations_N22_7.bin ] && [ -f procMonitor/constellations_N22_7.bin ]; then
  if cmp -s constellations_N22_7.bin procMonitor/constellations_N22_7.bin; then
    echo "duplicate N22 procMonitor bin: removing"
    rm -f procMonitor/constellations_N22_7.bin
  else
    echo "procMonitor N22 differs: kept"
  fi
fi

# N22検証に集中するなら、退避済みN23も削除してよい
rm -f hold_N23_N24_bins/constellations_N23_7.bin
rm -f hold_N23_N24_bins/constellations_N23_7.bin.done

# 古い監視ログ。必要なければ削除
rm -f procMonitor/logs/health_N24_p7_GPU_20260514_023008.log

df -h .
```

これで最大 **約1.1GB** 追加で空くはずです。

ただし、根本対策としては、やはり `nvme1n1` の232.8Gを使うのが正解です。

## `nvme1n1` を作業ディスクにする

`lsblk` では `nvme1n1` が未マウントに見えます。AWS Nitro系ではEBSもinstance storeもNVMeデバイスとして見えるため、まず正体確認をしてください。AWS公式も、NitroではEBS volumeがNVMe block deviceとして露出すると説明しています。instance storeは一時ブロックストレージなので、停止・終了時の保持性に注意が必要です。([AWS ドキュメント][1])

```bash
echo "===== identify nvme1n1 ====="
lsblk -f
sudo blkid /dev/nvme1n1 || true
sudo file -s /dev/nvme1n1
sudo nvme list 2>/dev/null || true
```

`sudo nvme list` の `Model` が、

```text
Amazon Elastic Block Store
```

ならEBSです。作業データを置くのにかなり安心です。

```text
Amazon EC2 NVMe Instance Storage
```

ならinstance storeです。高速な作業用scratchとしては良いですが、重要ログや最終成果物はroot EBSやS3にも残してください。AWS公式ではinstance storeを一時ストレージとして説明しています。([AWS ドキュメント][2])

## 空ディスクならフォーマットして `/data/nq` にマウント

`sudo file -s /dev/nvme1n1` が `data` のように出て、既存ファイルシステムが無いことを確認してから実行してください。`mkfs` は中身を消します。

```bash
sudo mkfs.ext4 -F /dev/nvme1n1

sudo mkdir -p /data/nq
sudo mount /dev/nvme1n1 /data/nq
sudo chown -R suzuki:suzuki /data/nq

df -h /data/nq
```

期待形はこうです。

```text
/dev/nvme1n1  229G ... /data/nq
```

再起動後もマウントしたい場合は、`UUID` で `fstab` に入れます。

```bash
UUID=$(sudo blkid -s UUID -o value /dev/nvme1n1)
echo "$UUID"

sudo cp -av /etc/fstab /etc/fstab.bak.$(date +%Y%m%d_%H%M%S)

echo "UUID=$UUID /data/nq ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab

sudo umount /data/nq
sudo mount -a
df -h /data/nq
```

## N-Queens作業ディレクトリを移す

今の `N-Queens` は全体で約1.8Gなので、丸ごと `/data/nq` に移してしまうのが簡単です。

```bash
cd /home/suzuki/Github

rsync -aH --info=progress2 N-Queens/ /data/nq/N-Queens/

echo "===== size check ====="
du -sh /home/suzuki/Github/N-Queens /data/nq/N-Queens
```

コピー確認後、root側を退避してsymlinkにします。

```bash
cd /home/suzuki/Github

mv N-Queens N-Queens.rootdisk.bak.$(date +%Y%m%d_%H%M%S)
ln -s /data/nq/N-Queens N-Queens

cd /home/suzuki/Github/N-Queens/13Bit_codon

pwd
df -h .
ls -lh 94Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2.py
ls -lh 95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe.py 2>/dev/null || true
```

ここで `df -h .` が `/data/nq` 側、つまり232.8Gディスクを指せば成功です。

## 動作確認後にroot側バックアップを削除

symlink化後、ファイルが見えて、95 safe版がビルドできることを確認したら、root側バックアップを消せます。

```bash
cd /home/suzuki/Github
ls -ld N-Queens.rootdisk.bak.*

# 実際の名前を確認してから削除
rm -rf N-Queens.rootdisk.bak.YYYYMMDD_HHMMSS

df -h /
df -h /data/nq
```

## その後の運用

移行後は、95/N22/N23/N24の `.bin` やsweep temporaryを気にせず扱いやすくなります。

```text
/data/nq/N-Queens/13Bit_codon:
  大きい .bin
  temporary bin
  bucket bin
  sweep progress
  GPU実行ログ

root側 / GitHub:
  symlinkだけ
```

重要な点は、**今の `df -h .` がまだ `/dev/nvme0n1p1` を指しているので、まだ拡張ディスク上では作業していない**ということです。次は `nvme1n1` の正体確認 → `/data/nq` マウント → `N-Queens` 移行、で進めるのが安全です。

[1]: https://docs.aws.amazon.com/ebs/latest/userguide/nvme-ebs-volumes.html?utm_source=chatgpt.com "Amazon EBS volumes and NVMe"
[2]: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/InstanceStorage.html?utm_source=chatgpt.com "Instance store temporary block storage for EC2 instances"


はい。今の状態なら、**追加ディスク拡張はほぼ完了済み**です。重要なのはここです。

```text
nvme1n1  ext4  ...  FSAVAIL 214.7G  FSUSE% 1%  MOUNTPOINTS /data/nq
```

つまり `/dev/nvme1n1` はすでに **ext4でフォーマット済み**、かつ **`/data/nq` にマウント済み**です。
ここで **`mkfs.ext4 /dev/nvme1n1` は実行しないでください**。中身を消してしまいます。

一方で、現在の作業場所はまだ root 側です。

```text
/home/suzuki/Github/N-Queens/13Bit_codon
df -h . → /dev/nvme0n1p1
```

なので、次にやることは **N-Queens作業ディレクトリを `/data/nq` 側へ移す**ことです。

## 1. まず `/data/nq` が使えるか確認

```bash
df -h /data/nq
findmnt /data/nq
touch /data/nq/.write_test && rm -f /data/nq/.write_test
```

`touch` が通れば、通常の作業領域として使えます。

`sudo file -s /dev/nvme1n1` の `needs journal recovery` は少し気になりますが、すでにマウントされていて `FSAVAIL 214.7G` が見えているので、まずはログ確認だけでよいです。

```bash
sudo dmesg | egrep -i 'nvme1n1|/data/nq|ext4|journal|error' | tail -80
```

明確な `EXT4-fs error` がなければそのまま進めてよいです。`e2fsck` をかける場合は、**必ずアンマウントしてから**ですが、今は不要だと思います。

## 2. fstab登録を確認

すでにマウントされているので、fstabに入っている可能性があります。確認します。

```bash
grep -nE 'data/nq|db3664c5-724c-4439-a9bd-1c1f3844530f|nvme1n1' /etc/fstab || echo "no fstab entry"
```

出なければ、再起動後に自動マウントされない可能性があります。登録するなら以下です。

```bash
sudo cp -av /etc/fstab /etc/fstab.bak.$(date +%Y%m%d_%H%M%S)

echo 'UUID=db3664c5-724c-4439-a9bd-1c1f3844530f /data/nq ext4 defaults,nofail 0 2' \
  | sudo tee -a /etc/fstab

sudo mount -a
df -h /data/nq
```

## 3. N-Queensを `/data/nq` へ移す

現在のプロジェクト全体は `/home/suzuki/Github/N-Queens` なので、まずこれを丸ごとコピーします。

```bash
pgrep -af '95Py|94Py|84Py|funcid_reorder|codon' || true
```

何も実行中でなければ、コピーします。

```bash
cd /home/suzuki/Github

mkdir -p /data/nq/Github

rsync -aH --info=progress2 N-Queens/ /data/nq/Github/N-Queens/

echo "===== size check ====="
du -sh /home/suzuki/Github/N-Queens /data/nq/Github/N-Queens
```

コピー後、重要ファイルを確認します。

```bash
ls -lh /data/nq/Github/N-Queens/13Bit_codon/94Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2.py
ls -lh /data/nq/Github/N-Queens/13Bit_codon/95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe.py 2>/dev/null || true
ls -lh /data/nq/Github/N-Queens/13Bit_codon/constellations_N22_7.bin
ls -lh /data/nq/Github/N-Queens/13Bit_codon/progress_N22_7_stream_funcid_reorder_v2.tsv
```

## 4. root側を退避して symlink 化

コピーが確認できたら、元の `N-Queens` をバックアップ名に変えて、`/data/nq` 側へsymlinkします。

```bash
cd /home/suzuki/Github

mv N-Queens N-Queens.rootdisk.bak.$(date +%Y%m%d_%H%M%S)
ln -s /data/nq/Github/N-Queens N-Queens

cd /home/suzuki/Github/N-Queens/13Bit_codon

echo "===== pwd ====="
pwd

echo "===== realpath ====="
realpath .

echo "===== disk ====="
df -h .
```

ここで `df -h .` が `/dev/nvme1n1` または `/data/nq` を指せば成功です。

期待形はこうです。

```text
/dev/nvme1n1  ...  /data/nq
```

## 5. 95 safe版を再確認

移動後に、safe版があるか確認してください。95 safe版は、9条件sweepを temporary bin 1本で回すための重要版です。前回ログでは、safe版の `mode16` は9条件すべて `valid=1` で完走していました。

```bash
cd /home/suzuki/Github/N-Queens/13Bit_codon

ls -lh 95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe.py
ls -lh 95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe 2>/dev/null || true
```

実行ファイルがなければビルドします。

```bash
codon build -release 95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe.py
```

## 6. root側バックアップを消すタイミング

symlink後に動作確認できるまでは、root側バックアップを残してください。
ただし、バックアップを残したままだと root `/` の空きは増えません。

動作確認後、たとえば以下が問題なければ、

```bash
cd /home/suzuki/Github/N-Queens/13Bit_codon

df -h .
ls -lh constellations_N22_7.bin
ls -lh progress_N22_7_stream_funcid_reorder_v2.tsv
./95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe --help 2>/dev/null || true
```

root側バックアップを削除します。

```bash
cd /home/suzuki/Github

ls -ld N-Queens.rootdisk.bak.*

# 表示された実名を確認してから削除
rm -rf N-Queens.rootdisk.bak.YYYYMMDD_HHMMSS

df -h /
df -h /data/nq
```

この削除で root 側に約1.8GB以上戻るはずです。

## 7. 今後の作業場所

今後は必ずこのパスで作業してください。

```bash
cd /home/suzuki/Github/N-Queens/13Bit_codon
df -h .
```

`df -h .` が `/data/nq` 側ならOKです。
これで、N22の95実測、N23/N24 `.bin`、temporary bin、bucket bin を置いてもかなり余裕があります。

## 現時点の判断

```text
nvme1n1:
  すでに /data/nq に ext4 mounted。
  空き 214.7G。
  mkfs不要。

現在の問題:
  作業ディレクトリがまだ root 30G 側。

次の一手:
  /home/suzuki/Github/N-Queens を /data/nq/Github/N-Queens に rsync。
  元ディレクトリを symlink 化。
  df -h . が /data/nq を指すことを確認。

その後:
  root側バックアップを削除して root容量を回復。
  95 safe版でN22 mode15実測へ進む。
```

これでディスク容量問題はかなり解消できます。

できています。**作業ディレクトリの `/data/nq` 移行は成功**です。

確認できている点は以下です。

```text
論理パス:
  /home/suzuki/Github/N-Queens/13Bit_codon

実体:
  /data/nq/Github/N-Queens/13Bit_codon

作業ディスク:
  /dev/nvme1n1

空き:
  215G
```

`realpath .` が `/data/nq/Github/N-Queens/13Bit_codon` を指し、`df -h .` も `/dev/nvme1n1 229G ... 215G ... /data/nq` になっているので、今後このディレクトリで作る `.bin` や sweep temporary は大容量ディスク側に出ます。

95 safe版も配置・実行ファイルとも確認できています。

```text
95Py_..._sweep_safe.py
95Py_..._sweep_safe
```

`--help` は未対応なので `Unknown option: --help` が出ていますが、usage が表示されているので実行ファイルとしては問題ありません。

## 1点だけ残作業があります

root側バックアップがまだ残っています。

```text
/home/suzuki/Github/N-Queens.rootdisk.bak.20260525_062445
```

ログでは、

```bash
rm -rf N-Queens.rootdisk.bak.YYYYMMDD_HHMMSS
```

を実行していますが、これはプレースホルダ名なので、実ファイルは消えていません。`ls -la` にバックアップが残っています。

移行先が問題なく見えているので、確認してから削除して大丈夫です。

```bash
cd /home/suzuki/Github

echo "===== symlink check ====="
ls -ld N-Queens
readlink -f N-Queens

echo "===== backup check ====="
du -sh N-Queens.rootdisk.bak.20260525_062445

echo "===== data-side check ====="
du -sh /data/nq/Github/N-Queens
ls -lh /data/nq/Github/N-Queens/13Bit_codon/constellations_N22_7.bin
ls -lh /data/nq/Github/N-Queens/13Bit_codon/95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe
```

問題なければ削除です。

```bash
rm -rf /home/suzuki/Github/N-Queens.rootdisk.bak.20260525_062445

df -h /
df -h /data/nq
```

これで root `/` 側にもさらに空きが戻るはずです。

## fstab確認もしておくと安全です

再起動後も `/data/nq` が自動マウントされるか確認してください。

```bash
grep -nE '/data/nq|db3664c5-724c-4439-a9bd-1c1f3844530f' /etc/fstab || echo "NO_FSTAB_ENTRY"
```

`NO_FSTAB_ENTRY` なら、以下を追加します。

```bash
sudo cp -av /etc/fstab /etc/fstab.bak.$(date +%Y%m%d_%H%M%S)

echo 'UUID=db3664c5-724c-4439-a9bd-1c1f3844530f /data/nq ext4 defaults,nofail 0 2' \
  | sudo tee -a /etc/fstab

sudo mount -a
df -h /data/nq
```

## 次は95のN22 mode15実測へ進めます

もう容量制約はかなり解消されました。次は予定どおり **w8_j5** のGPU実測でよいです。95 safe の mode16 は9条件すべて `valid=1` で完走済みなので、次は mode15 の実測段階です。

```bash
cd /home/suzuki/Github/N-Queens/13Bit_codon
df -h .

mkdir -p logs

W=8
J=5
PARAM="w${W}_j${J}"
LOG="logs/run95_N22_${PARAM}_mode15_gpu_$(date +%Y%m%d_%H%M%S).log"
PROG="progress_N22_7_stream_funcid_reorder_v2_${PARAM}.tsv"

if [ -f "$PROG" ]; then
  mv "$PROG" "${PROG}.$(date +%Y%m%d_%H%M%S).bak"
fi

set -o pipefail

{
  echo "PARAM=$PARAM"
  date
  hostname
  pwd
  realpath .
  df -h .
  nvidia-smi
} | tee "$LOG"

./95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe \
  -g 22 22 32 484 1 0 5 15 "$W" "$J" \
  2>&1 | tee -a "$LOG"

STATUS=${PIPESTATUS[0]}

{
  echo "EXIT=$STATUS"
  date
  echo "===== final lines ====="
  grep -E "funcid-reorder-v2-gpu-summary|funcid-reorder-v2-gpu-done|^22:" "$LOG" || true

  echo "===== progress recovery check ====="
  if [ -f "$PROG" ]; then
    awk -F'\t' '
    NR==1 {
      for (i=1; i<=NF; i++) c[$i]=i
      next
    }
    {
      rows++
      last_chunk=$(c["chunk"])
      last_m=$(c["m"])
      last_gpu_total=$(c["gpu_total"])
      last_done=$(c["done_records"])
      last_total=$(c["total_records"])
      last_remaining=$(c["remaining_records"])
      sum_ms += $(c["elapsed_ms"])
    }
    END {
      print "rows=" rows
      print "last_chunk=" last_chunk
      print "last_m=" last_m
      print "gpu_total=" last_gpu_total
      print "done_records=" last_done
      print "total_records=" last_total
      print "remaining_records=" last_remaining
      printf "sum_chunk_elapsed=%.3f sec\n", sum_ms/1000.0

      if (last_chunk==1854 && last_m==4283 && last_done==28719035 && last_total==28719035 && last_remaining==0 && last_gpu_total==2691008701644) {
        print "RECOVERED_N22_OK"
      } else {
        print "NOT_RECOVERED_OR_INCOMPLETE"
      }
    }
    ' "$PROG"
  else
    echo "NO_PROGRESS_FILE: $PROG"
  fi

  echo "===== disk after ====="
  df -h .
} | tee -a "$LOG"

exit "$STATUS"
```

成功判定はこのどちらかです。

```text
22:     2691008701644                0      ...    ok
```

または、

```text
RECOVERED_N22_OK
```

結論として、**ディスク移行は成功、95実測に進める状態**です。今やるべき残りは、root側バックアップの実名削除と、fstab確認だけです。


了解です。いまは **N22 / 95 safe / mode15 / w8_j5 実測中** という扱いで進めればよいです。

ディスク移行は成功しています。`N-Queens` は `/data/nq/Github/N-Queens` への symlink になっており、作業ディレクトリも `/dev/nvme1n1` 側、空き約215Gです。root側バックアップも削除され、`/` も6.3Gまで回復しています。
95 safe版のソースと実行ファイルも `/data/nq` 側で確認済みです。

実行中は、別窓でこれを見てください。

```bash
cd /home/suzuki/Github/N-Queens/13Bit_codon

watch -n 60 '
date
echo
echo "===== disk ====="
df -h . /
echo
echo "===== memory ====="
free -h
echo
echo "===== gpu ====="
nvidia-smi
echo
echo "===== progress w8_j5 ====="
tail -5 progress_N22_7_stream_funcid_reorder_v2_w8_j5.tsv 2>/dev/null || true
'
```

ログを追うなら、最新ログを自動で拾います。

```bash
cd /home/suzuki/Github/N-Queens/13Bit_codon

LOG=$(ls -t logs/run95_N22_w8_j5_mode15_gpu_*.log 2>/dev/null | head -1)
echo "$LOG"
tail -f "$LOG"
```

途中進捗を数値で見るならこれです。

```bash
PROG=progress_N22_7_stream_funcid_reorder_v2_w8_j5.tsv

awk -F'\t' '
NR==1 {
  for (i=1; i<=NF; i++) c[$i]=i
  next
}
{
  rows++
  last_chunk=$(c["chunk"])
  last_m=$(c["m"])
  last_gpu_total=$(c["gpu_total"])
  last_done=$(c["done_records"])
  last_total=$(c["total_records"])
  last_remaining=$(c["remaining_records"])
  sum_ms += $(c["elapsed_ms"])
}
END {
  print "rows=" rows
  print "last_chunk=" last_chunk
  print "last_m=" last_m
  print "gpu_total=" last_gpu_total
  print "done_records=" last_done
  print "total_records=" last_total
  print "remaining_records=" last_remaining
  printf "sum_chunk_elapsed=%.3f sec\n", sum_ms/1000.0
}
' "$PROG"
```

成功時は、ログにこの形が出ます。

```text
22:     2691008701644                0      ...    ok
```

または、もしまたログ末尾が欠けても、progress 側で最終的にこうなれば成功扱いできます。

```text
last_chunk=1854
last_m=4283
gpu_total=2691008701644
done_records=28719035
total_records=28719035
remaining_records=0
```

完了後はこの確認をしてください。

```bash
LOG=$(ls -t logs/run95_N22_w8_j5_mode15_gpu_*.log 2>/dev/null | head -1)
PROG=progress_N22_7_stream_funcid_reorder_v2_w8_j5.tsv

echo "LOG=$LOG"

grep -E "funcid-reorder-v2-gpu-summary|funcid-reorder-v2-gpu-done|^22:" "$LOG" || true

awk -F'\t' '
NR==1 {
  for (i=1; i<=NF; i++) c[$i]=i
  next
}
{
  rows++
  last_chunk=$(c["chunk"])
  last_m=$(c["m"])
  last_gpu_total=$(c["gpu_total"])
  last_done=$(c["done_records"])
  last_total=$(c["total_records"])
  last_remaining=$(c["remaining_records"])
  sum_ms += $(c["elapsed_ms"])
}
END {
  print "rows=" rows
  print "last_chunk=" last_chunk
  print "last_m=" last_m
  print "gpu_total=" last_gpu_total
  print "done_records=" last_done
  print "total_records=" last_total
  print "remaining_records=" last_remaining
  printf "sum_chunk_elapsed=%.3f sec\n", sum_ms/1000.0

  if (last_chunk==1854 && last_m==4283 && last_done==28719035 && last_total==28719035 && last_remaining==0 && last_gpu_total==2691008701644) {
    print "RECOVERED_N22_OK"
  } else {
    print "NOT_RECOVERED_OR_INCOMPLETE"
  }
}
' "$PROG"
```

一点だけ、実行が終わってから直すとよいものがあります。`/etc/fstab` に `/data/nq` の同じUUID行が2回入っています。現状の実行には影響しませんが、再起動・`mount -a` 周りをきれいにするため、ジョブ完了後に重複を1行へ整理してください。

確認だけなら今でも安全です。

```bash
grep -nE '/data/nq|db3664c5-724c-4439-a9bd-1c1f3844530f' /etc/fstab
```

ジョブ完了後に整理するなら、手編集が一番安全です。

```bash
sudo cp -av /etc/fstab /etc/fstab.bak.$(date +%Y%m%d_%H%M%S)
sudo vi /etc/fstab
```

残す行は1つだけです。

```text
UUID=db3664c5-724c-4439-a9bd-1c1f3844530f /data/nq ext4 defaults,nofail 0 2
```

編集後の確認です。

```bash
sudo mount -a
df -h /data/nq
```

いまの優先順位はこうです。

```text
1. w8_j5 mode15を完走させる
2. total=2691008701644 / RECOVERED_N22_OK を確認
3. ログとprogressを保存
4. param別 .bin とbucketを掃除
5. 次候補 w32_j7 または w16_j5 へ進む
6. fstab重複はジョブ完了後に整理
```

開始できているので、このまま **w8_j5の結果待ち**で大丈夫です。


w8_j5   3:40:38.068
w8_j7   3:40:32.062
w8_j11  3:40:36.610
w16_j5  3:41:25.516
w16_j7  3:44:01.870
w16_j11 3:44:01.366
w32_j5  3:41:06.090
w32_j7  3:41:09.403
w32_j11 3:40:59.730

# $ bash progressTest.sh 8 5
# $ bash progressTest.sh 8 7
# $ bash progressTest.sh 8 11
# $ bash progressTest.sh 16 5
# $ bash progressTest.sh 16 7
  $ bash progressTest.sh 16 11
# $ bash progressTest.sh 32 5
# $ bash progressTest.sh 32 7
# $ bash progressTest.sh 32 11



完成形出力
last_chunk=1854
last_m=4283
gpu_total=2691008701644
done_records=28719035
total_records=28719035
remaining_records=0



<>実行版
============

#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 W J"
  echo "Example: $0 16 5"
  exit 2
fi

case "$1" in
  8|16|32) ;;
  *) echo "W must be one of: 8 16 32"; exit 2 ;;
esac

case "$2" in
  5|7|11) ;;
  *) echo "J must be one of: 5 7 11"; exit 2 ;;
esac

cd /home/suzuki/Github/N-Queens/13Bit_codon || exit 1
df -h .

mkdir -p logs

W=$1
J=$2

PARAM="w${W}_j${J}"
TS="$(date +%Y%m%d_%H%M%S)"

LOG="logs/run95_N22_${PARAM}_mode15_gpu_${TS}.log"
PROG="progress_N22_7_stream_funcid_reorder_v2_${PARAM}.tsv"
LOCK="logs/${PARAM}.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: ${PARAM} seems to be already running. lock=${LOCK}"
  exit 3
fi

if [ -f "$PROG" ]; then
  mv "$PROG" "${PROG}.${TS}.bak"
fi

set -o pipefail

{
  echo "PARAM=$PARAM"
  echo "W=$W"
  echo "J=$J"
  echo "LOG=$LOG"
  echo "PROG=$PROG"
  echo "LOCK=$LOCK"
  date
  hostname
  pwd
  realpath .
  df -h .
  nvidia-smi
} | tee "$LOG"

./95Py_constellations_GPU_cuda_codon_dynamic_p8_stream_funcid_reorder_v2_sweep_safe \
  -g 22 22 32 484 1 0 5 15 "$W" "$J" \
  2>&1 | tee -a "$LOG"

STATUS=${PIPESTATUS[0]}

{
  echo
  echo "STATUS=$STATUS"
  date
  df -h .
} | tee -a "$LOG"

exit "$STATUS"



<>チェック版
============

#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 W J"
  echo "Example: $0 16 5"
  exit 2
fi

case "$1" in
  8|16|32) ;;
  *) echo "W must be one of: 8 16 32"; exit 2 ;;
esac

case "$2" in
  5|7|11) ;;
  *) echo "J must be one of: 5 7 11"; exit 2 ;;
esac

cd /home/suzuki/Github/N-Queens/13Bit_codon || exit 1
df -h .

mkdir -p logs

W=$1
J=$2


PROG=progress_N22_7_stream_funcid_reorder_v2_w${W}_j${J}.tsv

awk -F'\t' '
NR==1 {
  for (i=1; i<=NF; i++) c[$i]=i
  next
}
{
  rows++
  last_chunk=$(c["chunk"])
  last_m=$(c["m"])
  last_gpu_total=$(c["gpu_total"])
  last_done=$(c["done_records"])
  last_total=$(c["total_records"])
  last_remaining=$(c["remaining_records"])
  sum_ms += $(c["elapsed_ms"])
}
END {
  print "rows=" rows
  print "last_chunk=" last_chunk
  print "last_m=" last_m
  print "gpu_total=" last_gpu_total
  print "done_records=" last_done
  print "total_records=" last_total
  print "remaining_records=" last_remaining
  printf "sum_chunk_elapsed=%.3f sec\n", sum_ms/1000.0
}
' "$PROG"


はい。
その実行リストで分かることは、ひとことで言うと **`W ∈ {8,16,32}` と `J ∈ {5,7,11}` の全9通りについて、正しさが保たれたまま、どの組み合わせが一番速く・安定しているか** です。

現在の整理はこうです。

|  W | J=5  | J=7  | J=11 |
| -: | ---- | ---- | ---- |
|  8 | 完了   | これから | これから |
| 16 | 進行中  | これから | これから |
| 32 | これから | 完了   | これから |

つまり、コメントアウトされているものは既に完了済みまたは進行中で、実行するのは残り6本です。

```bash
bash progressTest.sh 8 7
bash progressTest.sh 8 11
bash progressTest.sh 16 7
bash progressTest.sh 16 11
bash progressTest.sh 32 5
bash progressTest.sh 32 11
```

これらを実行すると、最終的に次の9本がそろいます。

```text
w8_j5
w8_j7
w8_j11

w16_j5
w16_j7
w16_j11

w32_j5
w32_j7
w32_j11
```

## 1. まず「正しさ」が確認できます

各実行で、最後に以下になれば完走です。

```text
done_records      = 28719035
remaining_records = 0
```

さらに、9本すべてで最終集計が一致するかを見ます。

確認したいものは主に以下です。

```text
total_records
done_records
remaining_records
gpu_total
w2_count / w4_count / w8_count
funcid_*_count
risky_a / risky_b / risky_c / good / other
```

期待としては、**WやJを変えても最終結果は同じ**です。

つまり、

```text
w8_j5
w8_j7
w8_j11
w16_j5
...
w32_j11
```

のどれを使っても、探索対象と分類結果は一致するはずです。

もし最終集計が一致すれば、

> W/J は実行順序・並列化・chunk処理効率には影響するが、数学的な探索結果には影響していない

と判断できます。

逆に、どれか1本だけでも `gpu_total` や `funcid_*_count` や `risk/good` の最終値が違えば、かなり重要です。
その場合は、W/J依存のバグ、reorderの不整合、chunk処理漏れ、または集計ミスを疑う必要があります。

## 2. 次に「速度」が比較できます

9本がそろうと、各組み合わせについて以下を比較できます。

```text
総elapsed
records/sec
平均chunk秒数
median chunk秒数
p95 chunk秒数
p99 chunk秒数
最大chunk秒数
```

これで、単に一番速いものだけでなく、

> 平均は速いが、ときどき重いchunkが出る
> 平均は少し遅いが、p95や最大値が安定している
> Wを大きくしてもあまり速くならない
> Jを増やすとむしろ遅くなる

といった傾向が見えます。

既に見えている範囲では、

```text
w8_j5
w32_j7
```

はどちらも完走していて、総時間は `w8_j5` がわずかに速く、`w32_j7` はchunkごとのばらつきが少し小さい、という傾向でした。

ただし、この2点だけではまだ全体傾向は決められません。
残りを実行することで、

```text
W=8 が全体的に良いのか
W=16 が中間で最適なのか
W=32 がJ次第で良くなるのか
J=5 / 7 / 11 のどれが安定するのか
```

が判断できます。

## 3. W方向の比較ができます

同じJで、Wだけを変えた比較です。

例えば `J=5` なら、

```text
w8_j5
w16_j5
w32_j5
```

を比較します。

これで分かるのは、

> Wを 8 → 16 → 32 に増やしたとき、速くなるのか、遅くなるのか、安定するのか

です。

同じように、

```text
J=7:
w8_j7
w16_j7
w32_j7

J=11:
w8_j11
w16_j11
w32_j11
```

も比較できます。

ここから、Wの最適値が見えます。

例えば結果がこうなら、

```text
w8  が速い
w16 は中間
w32 は遅い
```

Wを増やす意味は薄いです。

逆に、

```text
w32 がp95やmaxで安定
```

なら、総時間だけでなく安定性重視で `w32` を採用する理由が出ます。

## 4. J方向の比較ができます

同じWで、Jだけを変えた比較です。

例えば `W=8` なら、

```text
w8_j5
w8_j7
w8_j11
```

を比較します。

これで分かるのは、

> Jを 5 → 7 → 11 に増やしたとき、速度やばらつきがどう変わるか

です。

同じように、

```text
W=16:
w16_j5
w16_j7
w16_j11

W=32:
w32_j5
w32_j7
w32_j11
```

も比較できます。

ここから、Jの最適値が見えます。

例えば、

```text
j5 が速い
j7 が安定
j11 は遅い
```

のような傾向が出る可能性があります。

## 5. WとJの相互作用が分かります

重要なのは、WとJは独立に効くとは限らないことです。

例えば、こういうことがあり得ます。

```text
W=8 では j5 が速い
W=16 では j7 が速い
W=32 では j11 が安定する
```

つまり、

> Wだけを見てもダメ
> Jだけを見てもダメ
> W×J の組み合わせで見る必要がある

ということです。

今回の9本 sweep は、その相互作用を見るための最小グリッドです。

## 6. reorder_v2 の安全性確認にもなります

今回のファイル名は、

```text
stream_funcid_reorder_v2
```

なので、chunk内・chunk間の処理順序が変わっている可能性があります。

そのため、実行ごとに chunk単位の値は違って見えることがあります。

例えば、

```text
chunk_total
funcid_*_count
risky/good のchunkごとの分布
```

は、chunk単位では違っていても問題ありません。

重要なのは、最後の累積値が一致することです。

つまり、見るべきポイントは、

```text
chunk単位で違う → OK
最終累積で一致 → OK
最終累積で不一致 → NG
```

です。

9本すべてで最終累積が一致すれば、`reorder_v2` は少なくともこのW/J範囲では安全に動いている、と判断できます。

## 7. screenが空になる問題への対策にもなります

今回の `progressTest.sh` は、

```bash
tee -a "$LOG"
```

でログを保存しているので、screenが空でもログとTSVが残ります。

見るべきものはこの2つです。

```text
logs/run95_N22_w*_j*_mode15_gpu_*.log
progress_N22_7_stream_funcid_reorder_v2_w*_j*.tsv
```

特にTSVが最後まで出ていれば、screenの表示が空だったかどうかはあまり重要ではありません。

## 8. 最終的に選べるもの

9本がそろった後、最終的には以下のような判断ができます。

### 最速重視

```text
総elapsedが最小の W/J を採用
```

### 安定性重視

```text
p95 / p99 / max chunk elapsed が小さい W/J を採用
```

### 安全性重視

```text
完走して、STATUS=0で、最終集計が全一致する W/J のみ採用
```

### 実用バランス重視

```text
総elapsedが上位で、かつp95やmaxも悪くない W/J を採用
```

たとえば、最終比較表はこういう形になります。

```text
PARAM    total_elapsed   records/sec   mean_chunk   p95_chunk   max_chunk   status   final_ok
w8_j5
w8_j7
w8_j11
w16_j5
w16_j7
w16_j11
w32_j5
w32_j7
w32_j11
```

## 9. 注意点

1回ずつの実行なので、差がごく小さい場合は誤差の可能性があります。

例えば、

```text
0.2%差
0.5%差
```

くらいなら、GPU状態、CPU負荷、温度、他プロセス、I/Oなどの影響もあり得ます。

一方で、

```text
2%以上の差
p95やmaxに明確な差
特定のW/Jだけ極端に遅い
```

なら、かなり意味のある差として見てよいと思います。

## 結論

この6本を追加実行することで、既に完了している

```text
w8_j5
w32_j7
```

と、進行中の

```text
w16_j5
```

を含めて、最終的に **9通りの完全比較**ができます。

それによって分かることは主にこの4つです。

```text
1. W/Jを変えても最終結果が一致するか
2. どのW/Jが一番速いか
3. どのW/Jがchunk時間として安定しているか
4. reorder_v2+sweep_safe の実用上の推奨パラメータはどれか
```

なので、この実行リストはかなり重要です。
ここまで揃えば、次は単なる「完走確認」ではなく、**推奨設定を決めるための比較表**に進めます。



おはようございます。w16_j11 完走ログ確認しました。
結論から言うと、**本番基準は w8_j7 で固定してよい**です。ただし、今回のログでは **w8_j7 / w8_j11 / w8_j5 の差は 4.5〜6.0秒しかなく、3時間40分級の処理ではほぼノイズ域**です。したがって、次の最速化は `j` の細かい探索ではなく、**chunk 処理そのものの短縮・計測分解・短時間ベンチ化**へ進めるのがよいです。

## 1. 今回ログの順位

添付9本の TSV の `elapsed_ms` 合計で見ると、順位は以下です。

| 順位 | config    | TSV内 elapsed 合計 | w8_j7との差 |
| -: | --------- | --------------: | -------: |
|  1 | **w8_j7** | **3:40:32.062** |   0.000s |
|  2 | w8_j11    |     3:40:36.610 |  +4.548s |
|  3 | w8_j5     |     3:40:38.068 |  +6.006s |
|  4 | w32_j11   |     3:40:59.730 | +27.668s |
|  5 | w32_j5    |     3:41:06.090 | +34.028s |
|  6 | w32_j7    |     3:41:09.403 | +37.341s |
|  7 | w16_j5    |     3:41:25.516 | +53.454s |
|  8 | w16_j11   |     3:41:28.910 | +56.848s |
|  9 | w16_j7    |     3:41:32.020 | +59.958s |

全9本とも、最終値は以下で一致しています。

```text
records          = 28,719,035
chunks           = 1,855
gpu_total        = 2,691,008,701,644
source/reordered = 28,719,035 / 28,719,035
```

つまり、w/j は総仕事量を大きく変えているというより、**重い record 群がどの chunk に配置されるかを変えている**ように見えます。

また、今回いただいた w16_j11 の wall time は `3:44:01.366`、TSV内 `elapsed_ms` 合計は `3:41:28.910` なので、TSV外に約 **2分32秒** あります。ここは初期化・読み込み・最終書き込み・flush・集計などの可能性があるため、次回から分解して見たいです。

## 2. 判断

現時点の判断はこうです。

**本番実行:**
`w8_j7` を採用でよいです。

**ただし:**
`j=5,7,11` の差は w8 系ではほぼ誤差です。`phase_jump=7` が最速ではありますが、ここをさらに細かく詰めても大きな短縮は見込みにくいです。

**次に見るべき本命:**
`window_mult` / `phase_jump` ではなく、以下です。

1. **chunk サイズ / batch サイズ**
2. **chunk 内の固定コスト**
3. **stage別計測**
4. **funcid_5 / risky_c / other 系の重い分岐**
5. **N22を全部回さない短時間ベンチ環境**

## 3. 一番重要そうな仮説

w8_j7 のログで、chunkごとの `elapsed_ms` と `chunk_total` の相関を見るとかなり強いです。
ざっくり線形近似すると、

```text
elapsed_ms ≒ 2,440ms + 3,230ms × chunk_total[10^9]
```

という感じです。

これは厳密なモデルではありませんが、意味としては大きいです。

`1 chunk あたり 2.4秒前後の固定的な重さ` がありそうです。
N22 は 1,855 chunks なので、ここが本当に固定コストなら、

```text
2.4秒 × 1,855 chunks ≒ 75分
```

に相当します。

もちろん全部が削れるわけではありませんが、**chunk数を減らす、chunk処理を融合する、同期回数を減らす、I/Oやログflushをまとめる**方向は、w/j 探索より期待値が高いです。

## 4. 次の段取り

### Step 1: w8_j7 を正式 baseline にする

まず以下を baseline として固定します。

```text
window_mult = 8
phase_jump  = 7
N           = 22
preset      = 7
chunk size  = 15488 records
block       = 32
max_blocks  = 484
```

以後の改善は、必ずこの baseline との比較にします。

比較指標は以下でよいです。

```text
elapsed_ms合計
wall time
gpu_total
source_records / reordered_records
bin hash または record-id checksum
chunkごとの elapsed_ms 分布
```

特に bin の中身が reorder 順に依存するなら、`records数一致` だけではなく、簡易 checksum を入れた方が安全です。

### Step 2: N22短時間ベンチを先に作る

今のままでは1回 3時間44分級なので、デバッグが重すぎます。
最速化の前に、まず以下の実行モードを追加するのがよいです。

```text
--start-chunk K
--max-chunks M
--chunk-list 0,8,94,...
--no-write-bin
--verify-only
--profile-stage
```

特に `--chunk-list` が欲しいです。
N22の特性を保ったまま、数分で比較できます。

私のおすすめの microbench chunk list はこれです。

```text
0,8,94,150,339,557,710,976,1073,1222,
1225,1370,1469,1471,1587,1625,1644,1706,1772,1854
```

この20 chunk は w8_j7 で約 **154秒** 分です。
早い chunk、重い chunk、w16/w32との差が出やすい chunk、末尾 chunk を混ぜています。

用途はこう分けるのがよいです。

| 用途           |                     実行量 | 目的               |
| ------------ | ----------------------: | ---------------- |
| smoke        |              3〜5 chunks | ビルド後の即確認         |
| microbench   |       20 chunks / 約2.5分 | kernel変更・分岐変更の比較 |
| mini-release | 100〜200 chunks / 10〜25分 | 候補版の安定比較         |
| full N22     |     1,855 chunks / 3時間超 | 最終確認だけ           |

これを入れるだけで、改善サイクルがかなり速くなります。

### Step 3: stage別 timer を入れる

今の TSV では chunk全体の `elapsed_ms` は見えますが、内訳が見えません。
次は chunkごとに最低限これを分けたいです。

```text
read / prepare
reorder CPU side
H2D copy
GPU kernel main
GPU sort / reduce / scan
D2H copy
write bin
log / flush
```

特に見たいのは、

```text
GPU本体が重いのか
cudaMemcpy / sync が重いのか
bin書き込みが重いのか
chunkごとの固定処理が重いのか
```

です。

今回の w16_j11 では wall と TSV 合計に約2分32秒の差があるので、TSV外の時間も stage timer に含めたいです。

### Step 4: chunk size を試す

現状はほぼ全 chunk が、

```text
m      = 15488
block  = 32
blocks = 484
```

です。

固定コストが大きそうなので、次の候補は `m` を増やすことです。

まずは full N22 ではなく、microbench でよいです。

試す候補:

```text
m = 15488   # baseline
m = 30976   # 2x
m = 61952   # 4x
```

見るべき指標:

```text
elapsed_ms / record
elapsed_ms / gpu_total
GPU memory usage
result count
bin/checksum
chunk_total の総和
```

もし `m=30976` で結果が保てて elapsed が改善するなら、かなり有望です。
`m=61952` はメモリ・occupancy・一時バッファ次第ですが、試す価値があります。

### Step 5: funcid別の重い経路を見る

w8_j7 の chunk相関を見ると、遅い chunk はだいたい次の特徴を持っています。

```text
chunk_total が大きい
funcid_5_count が多い
risky_c_ratio / other_ratio が高い
score_avg が高い
```

逆に `funcid_19_count` が多い chunk は相対的に速い傾向があります。

なので、次の kernel 最適化候補は、

```text
funcid_5 系の分岐を分ける
funcid_7 / funcid_23 も確認する
risky_c / other を別 kernel または別 path にする
branch divergence を減らす
重い funcid だけ専用処理にする
```

です。

w8/w16/w32 の違いより、こちらの方が本質的な短縮につながる可能性があります。

## 5. 追加でやるなら w 探索は小さく

今の結果を見ると、

```text
w8  >  w32  >  w16
```

です。

`w8` が明確に良いので、次に試すとしても full N22 ではなく microbench で、

```text
w4_j7
w6_j7
w8_j7
w10_j7
w12_j7
```

のように、w8 の近傍だけで十分です。
ただし、w8_j5 / w8_j7 / w8_j11 の差が数秒しかないので、`j` は当面固定でよいと思います。

優先度は、

```text
高: chunk size / stage timer / funcid重分岐
中: block size / async copy / write削減
低: phase_jump追加探索
```

です。

## 6. 実行方針まとめ

次はこの順で進めるのがよいです。

```text
1. w8_j7 を baseline として固定
2. --chunk-list / --start-chunk / --max-chunks / --no-write-bin を追加
3. 20 chunk microbench を作る
4. stage別 timer を追加
5. m=15488, 30976, 61952 を microbench で比較
6. 良い候補だけ 100〜200 chunk mini-release で確認
7. 最後に full N22 を1回だけ回す
```

今回のログからは、**w8_j7 が最速**であることは確認できています。
ただし、次の大きな短縮は `w/j` ではなく、**N22を短く試せるベンチ環境を作ってから、chunk固定コストと重い funcid 経路を削る**方向が本命だと思います。






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

VERSION_TAG:str="96 auto w8_j7 stream funcid risk balanced reorder v2 GPU runner from 95 safe"
CROSS_STRIPE_SAFE_DEFAULT:bool=False

# 96 FUNCID REORDER V2 AUTO DEFAULT:
#   Based on 94 funcid risk reorder v2. Kernel / DFS logic is intentionally unchanged.
#   Changes are limited to stream-side reordering:
#     - keep risk-balanced A/B/C/good/other quota from 93
#     - add bucket-internal phase/stripe sampling to avoid same-order heavy regions aligning
#     - bench_mode=14 simulation-only mode: build v2 reordered .bin and TSV, no GPU
#     - bench_mode=15 v2 reordered stream GPU mode: run GPU on v2 reordered .bin
#     - bench_mode=16 simulation sweep: build all w{8,16,32} x j{5,7,11} reordered bins
#     - bench_mode=14/15 short-form extra CLI args: reorder_window_mult reorder_phase_jump
#
#   Risk groups used by 95:
#     risky_a = funcid 19,22,23,24
#     risky_b = funcid 26,27
#     risky_c = funcid 20,21
#     good    = funcid 0,4,5,12,16,17,18

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

"""stream chunk の経過時間文字列を ms へ変換する。"""
def stream_elapsed_text_to_ms(elapsed_text:str)->int:
  # str(datetime_delta)[:-3] の想定:
  #   0:00:11.480
  #   1 day, 6:08:25.451
  s:str=elapsed_text
  days:int=0
  day_parts=s.split(",")
  if len(day_parts)>1:
    day_tokens=day_parts[0].strip().split()
    if len(day_tokens)>0:
      days=int(day_tokens[0])
    s=day_parts[1].strip()

  hms=s.split(":")
  if len(hms)<3:
    return 0
  hours:int=int(hms[0])
  minutes:int=int(hms[1])
  sec_ms=hms[2].split(".")
  seconds:int=int(sec_ms[0])
  millis:int=0
  if len(sec_ms)>1:
    ms_str:str=sec_ms[1]
    if len(ms_str)==1:
      millis=int(ms_str)*100
    elif len(ms_str)==2:
      millis=int(ms_str)*10
    elif len(ms_str)>=3:
      millis=int(ms_str[0:3])
  return (((days*24+hours)*60+minutes)*60+seconds)*1000+millis

"""小数3桁の平均値文字列を整数演算だけで作る。"""
def format_ratio_3(num:int,den:int)->str:
  if den<=0:
    return "0.000"
  scaled:int=(num*1000)//den
  whole:int=scaled//1000
  frac:int=scaled%1000
  frac_s:str=str(frac)
  if frac<10:
    frac_s="00"+frac_s
  elif frac<100:
    frac_s="0"+frac_s
  return str(whole)+"."+frac_s

"""92 measure2 progress のヘッダを作る。"""
def stream_measure2_progress_header()->str:
  h:str="N\tpreset\tchunk\toff\tm\tblock\tmax_blocks\tsteps\tsort_mode\telapsed\telapsed_ms\tchunk_total\tgpu_total\tdone_records\ttotal_records\tremaining_records"
  h+="\tfree_popcount_sum\tfree_popcount_avg\tfree_popcount_min\tfree_popcount_max"
  h+="\trow_sum\trow_avg\trow_min\trow_max"
  h+="\tend_sum\tend_avg\tend_min\tend_max"
  h+="\tdepth_sum\tdepth_avg\tdepth_min\tdepth_max"
  h+="\tscore_sum\tscore_avg\tscore_min\tscore_max"
  h+="\tw2_count\tw4_count\tw8_count"
  fid:int=0
  while fid<28:
    h+=f"\tfuncid_{fid}_count"
    fid+=1
  h+="\n"
  return h

"""92 measure2: chunk入力統計をTSV末尾へ足す文字列を作る。"""
def stream_measure2_stats_suffix(stats:List[int],m:int)->str:
  s:str=""
  s+=f"\t{stats[0]}\t{format_ratio_3(stats[0],m)}\t{stats[1]}\t{stats[2]}"
  s+=f"\t{stats[3]}\t{format_ratio_3(stats[3],m)}\t{stats[4]}\t{stats[5]}"
  s+=f"\t{stats[6]}\t{format_ratio_3(stats[6],m)}\t{stats[7]}\t{stats[8]}"
  s+=f"\t{stats[9]}\t{format_ratio_3(stats[9],m)}\t{stats[10]}\t{stats[11]}"
  s+=f"\t{stats[12]}\t{format_ratio_3(stats[12],m)}\t{stats[13]}\t{stats[14]}"
  s+=f"\t{stats[15]}\t{stats[16]}\t{stats[17]}"
  fid:int=0
  while fid<28:
    s+=f"\t{stats[18+fid]}"
    fid+=1
  return s

"""
92 measure2: chunk_constellations の入力特徴を既存 SoA 変換で集計する。
stats index:
  0 free_sum, 1 free_min, 2 free_max,
  3 row_sum,  4 row_min,  5 row_max,
  6 end_sum,  7 end_min,  8 end_max,
  9 depth_sum,10 depth_min,11 depth_max,
 12 score_sum,13 score_min,14 score_max,
 15 w2_count,16 w4_count,17 w8_count,
 18..45 funcid_0_count..funcid_27_count
"""
def analyze_stream_chunk_input_stats(N:int,chunk_constellations:List[Dict[str,int]])->List[int]:
  m:int=len(chunk_constellations)
  stats:List[int]=[0]*46
  if m<=0:
    return stats

  stats[1]=999999999
  stats[4]=999999999
  stats[7]=999999999
  stats[10]=999999999
  stats[13]=999999999

  soa:TaskSoA=TaskSoA(m)
  w_arr:List[u64]=[u64(0)]*m
  build_soa_for_range(N,chunk_constellations,0,m,soa,w_arr)

  i:int=0
  while i<m:
    pc:int=popcount_int(soa.free_arr[i])
    rowv:int=soa.row_arr[i]
    endv:int=soa.end_arr[i]
    depth:int=endv-rowv
    if depth<0:
      depth=0
    score:int=pc*depth

    stats[0]+=pc
    if pc<stats[1]:
      stats[1]=pc
    if pc>stats[2]:
      stats[2]=pc

    stats[3]+=rowv
    if rowv<stats[4]:
      stats[4]=rowv
    if rowv>stats[5]:
      stats[5]=rowv

    stats[6]+=endv
    if endv<stats[7]:
      stats[7]=endv
    if endv>stats[8]:
      stats[8]=endv

    stats[9]+=depth
    if depth<stats[10]:
      stats[10]=depth
    if depth>stats[11]:
      stats[11]=depth

    stats[12]+=score
    if score<stats[13]:
      stats[13]=score
    if score>stats[14]:
      stats[14]=score

    w:int=int(w_arr[i])
    if w==2:
      stats[15]+=1
    elif w==4:
      stats[16]+=1
    elif w==8:
      stats[17]+=1

    fid:int=soa.funcid_arr[i]
    if fid>=0 and fid<28:
      stats[18+fid]+=1

    i+=1

  return stats

"""stream progress を TSV に追記する。"""
def append_stream_progress(progress_fname:str,N:int,preset_queens:int,chunk_index:int,off:int,m:int,BLOCK:int,MAX_BLOCKS:int,STEPS:int,gpu_sort_mode:int,elapsed_text:str,elapsed_ms:int,chunk_total:int,gpu_total:int,total_records:int,stats:List[int])->None:
  done_records:int=off+m
  remaining_records:int=total_records-done_records
  if remaining_records<0:
    remaining_records=0
  with open(progress_fname,"a") as f:
    f.write(f"{N}\t{preset_queens}\t{chunk_index}\t{off}\t{m}\t{BLOCK}\t{MAX_BLOCKS}\t{STEPS}\t{gpu_sort_mode}\t{elapsed_text}\t{elapsed_ms}\t{chunk_total}\t{gpu_total}\t{done_records}\t{total_records}\t{remaining_records}")
    f.write(stream_measure2_stats_suffix(stats,m))
    f.write("\n")


"""93 funcid reorder: funcid を 5 bucket へ分類する。"""
def funcid_reorder_bucket(fid:int)->int:
  # 0: risky_b, 1: risky_a, 2: risky_c, 3: good, 4: other
  if fid==26 or fid==27:
    return 0
  if fid==19 or fid==22 or fid==23 or fid==24:
    return 1
  if fid==20 or fid==21:
    return 2
  if fid==0 or fid==4 or fid==5 or fid==12 or fid==16 or fid==17 or fid==18:
    return 3
  return 4

"""93 funcid reorder: bucket 名。"""
def funcid_reorder_bucket_label(g:int)->str:
  if g==0:
    return "B"
  if g==1:
    return "A"
  if g==2:
    return "C"
  if g==3:
    return "G"
  return "O"

"""93 funcid reorder: 一時 bucket bin 名。"""
def funcid_reorder_bucket_fname(N:int,preset_queens:int,g:int)->str:
  return f"constellations_N{N}_{preset_queens}_funcid_reorder_v2_{funcid_reorder_bucket_label(g)}.bin"

"""95 funcid reorder: reorder 済み bin 名。

通常 mode14/15 では window/phase を含めて保存する。
mode16 sweep ではディスク節約のため、1本の temporary bin を上書き再利用する。
"""
def funcid_reorder_output_fname(N:int,preset_queens:int)->str:
  if FUNCID_REORDER_V2_SWEEP_TEMP_OUTPUT:
    return f"constellations_N{N}_{preset_queens}_funcid_reorder_v2_sweep_tmp.bin"
  return f"constellations_N{N}_{preset_queens}_funcid_reorder_v2_w{FUNCID_REORDER_V2_WINDOW_MULT}_j{FUNCID_REORDER_V2_PHASE_JUMP}.bin"

"""93 funcid reorder: bin を空にする。"""
def truncate_plain_bin(fname:str)->None:
  with open(fname,"wb") as f:
    pass

"""93 funcid reorder progress のヘッダを作る。"""
def stream_funcid_reorder_progress_header()->str:
  h:str=stream_measure2_progress_header().strip()
  h+="\trisky_a_count\trisky_a_ratio"
  h+="\trisky_b_count\trisky_b_ratio"
  h+="\trisky_c_count\trisky_c_ratio"
  h+="\tgood_count\tgood_ratio"
  h+="\tother_count\tother_ratio"
  h+="\n"
  return h

"""93 funcid reorder: stats から risk 列を作る。"""
def stream_funcid_reorder_risk_suffix(stats:List[int],m:int)->str:
  risky_a:int=stats[18+19]+stats[18+22]+stats[18+23]+stats[18+24]
  risky_b:int=stats[18+26]+stats[18+27]
  risky_c:int=stats[18+20]+stats[18+21]
  good:int=stats[18+0]+stats[18+4]+stats[18+5]+stats[18+12]+stats[18+16]+stats[18+17]+stats[18+18]
  other:int=m-risky_a-risky_b-risky_c-good
  if other<0:
    other=0
  s:str=""
  s+=f"\t{risky_a}\t{format_ratio_3(risky_a,m)}"
  s+=f"\t{risky_b}\t{format_ratio_3(risky_b,m)}"
  s+=f"\t{risky_c}\t{format_ratio_3(risky_c,m)}"
  s+=f"\t{good}\t{format_ratio_3(good,m)}"
  s+=f"\t{other}\t{format_ratio_3(other,m)}"
  return s

"""93 funcid reorder progress を TSV に追記する。"""
def append_stream_funcid_reorder_progress(progress_fname:str,N:int,preset_queens:int,chunk_index:int,off:int,m:int,BLOCK:int,MAX_BLOCKS:int,STEPS:int,gpu_sort_mode:int,elapsed_text:str,elapsed_ms:int,chunk_total:int,gpu_total:int,total_records:int,stats:List[int])->None:
  done_records:int=off+m
  remaining_records:int=total_records-done_records
  if remaining_records<0:
    remaining_records=0
  with open(progress_fname,"a") as f:
    f.write(f"{N}\t{preset_queens}\t{chunk_index}\t{off}\t{m}\t{BLOCK}\t{MAX_BLOCKS}\t{STEPS}\t{gpu_sort_mode}\t{elapsed_text}\t{elapsed_ms}\t{chunk_total}\t{gpu_total}\t{done_records}\t{total_records}\t{remaining_records}")
    f.write(stream_measure2_stats_suffix(stats,m))
    f.write(stream_funcid_reorder_risk_suffix(stats,m))
    f.write("\n")

"""93 funcid reorder: 1レコードを bin file object から読み、out へ追加する。"""
def append_one_constellation_from_file(f,out:List[Dict[str,int]])->bool:
  raw=f.read(16)
  if len(raw)<16:
    return False
  ld:int=read_uint32_le(raw[0:4])
  rd:int=read_uint32_le(raw[4:8])
  col:int=read_uint32_le(raw[8:12])
  startijkl:int=read_uint32_le(raw[12:16])
  out.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
  return True

"""93 funcid reorder: 元 bin を funcid risk bucket bin へ分配する。"""
def build_funcid_reorder_bucket_bins(N:int,fname:str,preset_queens:int,BLOCK:int,MAX_BLOCKS:int,gpu_log_level:int=0)->List[int]:
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  g:int=0
  while g<5:
    truncate_plain_bin(funcid_reorder_bucket_fname(N,preset_queens,g))
    g+=1

  counts:List[int]=[0]*5
  chunk_index:int=0
  _read_uint32_le=read_uint32_le

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-bucket-config] N={N} bin={fname} steps={STEPS}")

  with open(fname,"rb") as f:
    while True:
      chunk_constellations:List[Dict[str,int]]=[]
      i:int=0
      while i<STEPS:
        raw=f.read(16)
        if len(raw)<16:
          break
        ld:int=_read_uint32_le(raw[0:4])
        rd:int=_read_uint32_le(raw[4:8])
        col:int=_read_uint32_le(raw[8:12])
        startijkl:int=_read_uint32_le(raw[12:16])
        chunk_constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
        i+=1

      m:int=len(chunk_constellations)
      if m==0:
        break

      soa:TaskSoA=TaskSoA(m)
      w_arr:List[u64]=[u64(0)]*m
      build_soa_for_range(N,chunk_constellations,0,m,soa,w_arr)

      bucket_b:List[Dict[str,int]]=[]
      bucket_a:List[Dict[str,int]]=[]
      bucket_c:List[Dict[str,int]]=[]
      bucket_g:List[Dict[str,int]]=[]
      bucket_o:List[Dict[str,int]]=[]

      i=0
      while i<m:
        fid:int=soa.funcid_arr[i]
        bg:int=funcid_reorder_bucket(fid)
        if bg==0:
          bucket_b.append(chunk_constellations[i])
          counts[0]+=1
        elif bg==1:
          bucket_a.append(chunk_constellations[i])
          counts[1]+=1
        elif bg==2:
          bucket_c.append(chunk_constellations[i])
          counts[2]+=1
        elif bg==3:
          bucket_g.append(chunk_constellations[i])
          counts[3]+=1
        else:
          bucket_o.append(chunk_constellations[i])
          counts[4]+=1
        i+=1

      if len(bucket_b)>0:
        append_constellations_bin(funcid_reorder_bucket_fname(N,preset_queens,0),bucket_b)
      if len(bucket_a)>0:
        append_constellations_bin(funcid_reorder_bucket_fname(N,preset_queens,1),bucket_a)
      if len(bucket_c)>0:
        append_constellations_bin(funcid_reorder_bucket_fname(N,preset_queens,2),bucket_c)
      if len(bucket_g)>0:
        append_constellations_bin(funcid_reorder_bucket_fname(N,preset_queens,3),bucket_g)
      if len(bucket_o)>0:
        append_constellations_bin(funcid_reorder_bucket_fname(N,preset_queens,4),bucket_o)

      if gpu_log_level>=2:
        print(f"[funcid-reorder-v2-bucket-chunk] chunk={chunk_index} m={m} B={len(bucket_b)} A={len(bucket_a)} C={len(bucket_c)} G={len(bucket_g)} O={len(bucket_o)}")
      chunk_index+=1

  if gpu_log_level>=1:
    total:int=counts[0]+counts[1]+counts[2]+counts[3]+counts[4]
    print(f"[funcid-reorder-v2-bucket-summary] N={N} records={total} B={counts[0]} A={counts[1]} C={counts[2]} G={counts[3]} O={counts[4]}")

  return counts

"""93 funcid reorder: 残数と chunk サイズから比例 quota を作る。"""
def funcid_reorder_make_quotas(rem_counts:List[int],total_remaining:int,m_target:int)->List[int]:
  quotas:List[int]=[0]*5
  if total_remaining<=0 or m_target<=0:
    return quotas

  g:int=0
  quota_sum:int=0
  while g<5:
    q:int=(rem_counts[g]*m_target)//total_remaining
    if q>rem_counts[g]:
      q=rem_counts[g]
    quotas[g]=q
    quota_sum+=q
    g+=1

  while quota_sum<m_target:
    best:int=-1
    best_room:int=-1
    g=0
    while g<5:
      room:int=rem_counts[g]-quotas[g]
      if room>best_room:
        best_room=room
        best=g
      g+=1
    if best<0 or best_room<=0:
      break
    quotas[best]+=1
    quota_sum+=1

  return quotas

"""95 funcid reorder v2: bucket 内を phase/stripe サンプリングするための sweep parameter。"""
FUNCID_REORDER_V2_WINDOW_MULT:int=8
FUNCID_REORDER_V2_PHASE_JUMP:int=7
FUNCID_REORDER_V2_DEFAULT_REASON:str="N22 measured best baseline w8_j7"

"""95 funcid reorder v2: mode16 temporary output flag。

False: mode14/15 用。param別の .bin を保存する。
True : mode16 sweep simulation 用。1本の .bin を上書き再利用する。
"""
FUNCID_REORDER_V2_SWEEP_TEMP_OUTPUT:bool=False

"""95 funcid reorder v2: parameter tag。"""
def funcid_reorder_param_tag()->str:
  return f"w{FUNCID_REORDER_V2_WINDOW_MULT}_j{FUNCID_REORDER_V2_PHASE_JUMP}"

"""94 funcid reorder v2: bucket file から buffer を target 件まで補充する。"""
def fill_constellation_buffer_from_file(f,buf:List[Dict[str,int]],target:int)->int:
  added:int=0
  if target<0:
    target=0
  while len(buf)<target:
    if append_one_constellation_from_file(f,buf):
      added+=1
    else:
      break
  return added

"""94 funcid reorder v2: buffer から q 件を stride 抽出し、残り buffer も返す。

93 は各 bucket 内を元順序で消費したため、bucket 内の重い位相が
同じ output chunk に揃う可能性があった。94 は q*16 程度の窓を保持し、
chunk/group ごとに異なる stripe から q 件を取る。
"""
def take_striped_records_from_buffer(buf:List[Dict[str,int]],q:int,chunk_index:int,group_id:int)->Tuple[List[Dict[str,int]],List[Dict[str,int]]]:
  taken_records:List[Dict[str,int]]=[]
  n:int=len(buf)
  if q<=0 or n<=0:
    return taken_records,buf
  if q>n:
    q=n

  selected:List[bool]=[False]*n
  step:int=n//q
  if step<=0:
    step=1

  start:int=(chunk_index*FUNCID_REORDER_V2_PHASE_JUMP+group_id*3)%step
  idx:int=start
  taken:int=0
  guard:int=0
  guard_limit:int=n*2+q+16

  while taken<q and guard<guard_limit:
    if not selected[idx]:
      selected[idx]=True
      taken_records.append(buf[idx])
      taken+=1
    idx+=step
    if idx>=n:
      idx=idx%n
    guard+=1

  # gcd(step,n) の都合などで q 件に届かない場合の安全 fallback。
  i:int=0
  while taken<q and i<n:
    if not selected[i]:
      selected[i]=True
      taken_records.append(buf[i])
      taken+=1
    i+=1

  newbuf:List[Dict[str,int]]=[]
  i=0
  while i<n:
    if not selected[i]:
      newbuf.append(buf[i])
    i+=1

  return taken_records,newbuf

"""94 funcid reorder v2: 5 bucket から取り出した part を 93 と同じ順で interleave する。"""
def interleave_funcid_reorder_parts(part_b:List[Dict[str,int]],part_a:List[Dict[str,int]],part_c:List[Dict[str,int]],part_g:List[Dict[str,int]],part_o:List[Dict[str,int]],m_target:int)->List[Dict[str,int]]:
  out:List[Dict[str,int]]=[]
  ib:int=0
  ia:int=0
  ic:int=0
  ig:int=0
  io:int=0
  while len(out)<m_target:
    progressed:bool=False
    if ib<len(part_b):
      out.append(part_b[ib])
      ib+=1
      progressed=True
    if ig<len(part_g):
      out.append(part_g[ig])
      ig+=1
      progressed=True
    if ia<len(part_a):
      out.append(part_a[ia])
      ia+=1
      progressed=True
    if io<len(part_o):
      out.append(part_o[io])
      io+=1
      progressed=True
    if ic<len(part_c):
      out.append(part_c[ic])
      ic+=1
      progressed=True
    if not progressed:
      break
  return out

"""94 funcid reorder v2: bucket bin から phase/stripe balanced reorder bin を作り、simulation TSV を出す。"""
def build_funcid_reordered_bin(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1
)->Tuple[str,int,int]:

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
  counts:List[int]=build_funcid_reorder_bucket_bins(N,fname,preset_queens,BLOCK,MAX_BLOCKS,gpu_log_level)
  counted_records:int=counts[0]+counts[1]+counts[2]+counts[3]+counts[4]
  if counted_records!=total_records:
    print(f"[funcid-reorder-v2-warning] bucket count mismatch: counted={counted_records} total_records={total_records}")
    total_records=counted_records

  reorder_fname:str=funcid_reorder_output_fname(N,preset_queens)
  truncate_plain_bin(reorder_fname)

  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_funcid_reorder_v2_{funcid_reorder_param_tag()}_sim.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_funcid_reorder_progress_header())

  rem_counts:List[int]=[0]*5
  g:int=0
  while g<5:
    rem_counts[g]=counts[g]
    g+=1

  fb=open(funcid_reorder_bucket_fname(N,preset_queens,0),"rb")
  fa=open(funcid_reorder_bucket_fname(N,preset_queens,1),"rb")
  fc=open(funcid_reorder_bucket_fname(N,preset_queens,2),"rb")
  fg=open(funcid_reorder_bucket_fname(N,preset_queens,3),"rb")
  fo=open(funcid_reorder_bucket_fname(N,preset_queens,4),"rb")

  buf_b:List[Dict[str,int]]=[]
  buf_a:List[Dict[str,int]]=[]
  buf_c:List[Dict[str,int]]=[]
  buf_g:List[Dict[str,int]]=[]
  buf_o:List[Dict[str,int]]=[]

  off:int=0
  chunk_index:int=0
  total_remaining:int=total_records

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-build-config] N={N} records={total_records} steps={STEPS} output={reorder_fname} progress={progress_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP}")

  while total_remaining>0:
    m_target:int=STEPS
    if total_remaining<STEPS:
      m_target=total_remaining

    quotas:List[int]=funcid_reorder_make_quotas(rem_counts,total_remaining,m_target)
    qb:int=quotas[0]
    qa:int=quotas[1]
    qc:int=quotas[2]
    qg:int=quotas[3]
    qo:int=quotas[4]

    t0=datetime.now()

    target_b:int=qb*FUNCID_REORDER_V2_WINDOW_MULT
    target_a:int=qa*FUNCID_REORDER_V2_WINDOW_MULT
    target_c:int=qc*FUNCID_REORDER_V2_WINDOW_MULT
    target_g:int=qg*FUNCID_REORDER_V2_WINDOW_MULT
    target_o:int=qo*FUNCID_REORDER_V2_WINDOW_MULT
    if target_b<qb:
      target_b=qb
    if target_a<qa:
      target_a=qa
    if target_c<qc:
      target_c=qc
    if target_g<qg:
      target_g=qg
    if target_o<qo:
      target_o=qo
    if target_b>rem_counts[0]:
      target_b=rem_counts[0]
    if target_a>rem_counts[1]:
      target_a=rem_counts[1]
    if target_c>rem_counts[2]:
      target_c=rem_counts[2]
    if target_g>rem_counts[3]:
      target_g=rem_counts[3]
    if target_o>rem_counts[4]:
      target_o=rem_counts[4]

    fill_constellation_buffer_from_file(fb,buf_b,target_b)
    fill_constellation_buffer_from_file(fa,buf_a,target_a)
    fill_constellation_buffer_from_file(fc,buf_c,target_c)
    fill_constellation_buffer_from_file(fg,buf_g,target_g)
    fill_constellation_buffer_from_file(fo,buf_o,target_o)

    part_b:List[Dict[str,int]]=[]
    part_a:List[Dict[str,int]]=[]
    part_c:List[Dict[str,int]]=[]
    part_g:List[Dict[str,int]]=[]
    part_o:List[Dict[str,int]]=[]

    part_b,buf_b=take_striped_records_from_buffer(buf_b,qb,chunk_index,0)
    part_a,buf_a=take_striped_records_from_buffer(buf_a,qa,chunk_index,1)
    part_c,buf_c=take_striped_records_from_buffer(buf_c,qc,chunk_index,2)
    part_g,buf_g=take_striped_records_from_buffer(buf_g,qg,chunk_index,3)
    part_o,buf_o=take_striped_records_from_buffer(buf_o,qo,chunk_index,4)

    rem_counts[0]-=len(part_b)
    rem_counts[1]-=len(part_a)
    rem_counts[2]-=len(part_c)
    rem_counts[3]-=len(part_g)
    rem_counts[4]-=len(part_o)
    if rem_counts[0]<0:
      rem_counts[0]=0
    if rem_counts[1]<0:
      rem_counts[1]=0
    if rem_counts[2]<0:
      rem_counts[2]=0
    if rem_counts[3]<0:
      rem_counts[3]=0
    if rem_counts[4]<0:
      rem_counts[4]=0

    chunk_constellations:List[Dict[str,int]]=interleave_funcid_reorder_parts(part_b,part_a,part_c,part_g,part_o,m_target)
    m:int=len(chunk_constellations)
    if m==0:
      break

    append_constellations_bin(reorder_fname,chunk_constellations)
    stats:List[int]=analyze_stream_chunk_input_stats(N,chunk_constellations)
    t1=datetime.now()
    elapsed_text:str=str(t1-t0)[:-3]
    elapsed_ms:int=stream_elapsed_text_to_ms(elapsed_text)
    append_stream_funcid_reorder_progress(progress_fname,N,preset_queens,chunk_index,off,m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_text,elapsed_ms,0,0,total_records,stats)

    if gpu_log_level>=2:
      print(f"[funcid-reorder-v2-build-chunk] chunk={chunk_index} off={off} m={m} B={stats[18+26]+stats[18+27]} A={stats[18+19]+stats[18+22]+stats[18+23]+stats[18+24]} C={stats[18+20]+stats[18+21]} G={stats[18+0]+stats[18+4]+stats[18+5]+stats[18+12]+stats[18+16]+stats[18+17]+stats[18+18]}")

    off+=m
    chunk_index+=1
    total_remaining=total_records-off

  fb.close()
  fa.close()
  fc.close()
  fg.close()
  fo.close()

  write_stream_done_count(reorder_fname+".done",off)
  reordered_records:int=count_constellations_bin_records(reorder_fname)
  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-build-summary] N={N} records={reordered_records} chunks={chunk_index} output={reorder_fname} progress={progress_fname} valid={1 if validate_bin_file(reorder_fname) else 0}")

  return reorder_fname,reordered_records,chunk_index

"""93 funcid reorder: reorder 済み bin を STEPS 件ずつ既存 GPU kernel へ投入する。"""
def exec_solutions_gpu_bin_stream_funcid_reorder(
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
  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_funcid_reorder_v2_{funcid_reorder_param_tag()}.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_funcid_reorder_progress_header())

  if chunk_only:
    if debug_chunk_start<0:
      debug_chunk_start=0
    if debug_chunk_count<=0:
      debug_chunk_count=1

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-gpu-config] N={N} records={total_records} bin={fname} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} sort_mode={gpu_sort_mode} chunk_only={1 if chunk_only else 0} progress={progress_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} inner_log_level=0")

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
        ld:int=_read_uint32_le(raw[0:4])
        rd:int=_read_uint32_le(raw[4:8])
        col:int=_read_uint32_le(raw[8:12])
        startijkl:int=_read_uint32_le(raw[12:16])
        chunk_constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
        i+=1

      m:int=len(chunk_constellations)
      if m==0:
        break

      if chunk_only:
        run_this_chunk:bool=(chunk_index>=debug_chunk_start and chunk_index<debug_chunk_start+debug_chunk_count)
        if not run_this_chunk:
          if gpu_log_level>=2:
            print(f"[funcid-reorder-v2-gpu-chunk-skip] N={N} chunk={chunk_index} off={off} m={m}")
          off+=m
          chunk_index+=1
          continue

      stats:List[int]=analyze_stream_chunk_input_stats(N,chunk_constellations)

      t0=datetime.now()
      if gpu_log_level>=1:
        print(f"[funcid-reorder-v2-gpu-chunk-start] N={N} chunk={chunk_index} off={off} m={m}")

      inner_gpu_log_level:int=0
      exec_solutions(N,chunk_constellations,True,gpu_block,gpu_max_blocks,inner_gpu_log_level,gpu_sort_mode,cross_stripe_safe)

      chunk_total:int=0
      if m>0:
        chunk_total=chunk_constellations[0]["solutions"]
      gpu_total+=chunk_total
      executed_chunks+=1
      t1=datetime.now()
      elapsed_text:str=str(t1-t0)[:-3]
      elapsed_ms:int=stream_elapsed_text_to_ms(elapsed_text)
      append_stream_funcid_reorder_progress(progress_fname,N,preset_queens,chunk_index,off,m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_text,elapsed_ms,chunk_total,gpu_total,total_records,stats)
      if gpu_log_level>=1:
        print(f"[funcid-reorder-v2-gpu-chunk-end] N={N} chunk={chunk_index} off={off} m={m} elapsed={elapsed_text} elapsed_ms={elapsed_ms} chunk_total={chunk_total} gpu_total={gpu_total}")

      off+=m
      chunk_index+=1

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-gpu-summary] N={N} records={total_records} chunks={chunk_index} executed_chunks={executed_chunks} total={gpu_total} progress={progress_fname}")

  return gpu_total

"""92 measure2 stats-only: .bin を読み、GPUを起動せず chunk 入力統計だけを書く。"""
def exec_solutions_gpu_bin_stream_stats_only(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1
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
  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_measure2_stats.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_measure2_progress_header())

  if gpu_log_level>=1:
    print(f"[stream-stats-config] N={N} records={total_records} bin={fname} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} sort_mode={gpu_sort_mode} progress={progress_fname} stats_only=1")

  off:int=0
  chunk_index:int=0
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

      t0=datetime.now()
      stats:List[int]=analyze_stream_chunk_input_stats(N,chunk_constellations)
      t1=datetime.now()
      elapsed_text:str=str(t1-t0)[:-3]
      elapsed_ms:int=stream_elapsed_text_to_ms(elapsed_text)
      append_stream_progress(progress_fname,N,preset_queens,chunk_index,off,m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_text,elapsed_ms,0,0,total_records,stats)

      if gpu_log_level>=2:
        print(f"[stream-stats-chunk] N={N} chunk={chunk_index} off={off} m={m} elapsed={elapsed_text} elapsed_ms={elapsed_ms} free_avg={format_ratio_3(stats[0],m)} depth_avg={format_ratio_3(stats[9],m)} score_avg={format_ratio_3(stats[12],m)}")

      off+=m
      chunk_index+=1

  if gpu_log_level>=1:
    print(f"[stream-stats-summary] N={N} records={total_records} chunks={chunk_index} progress={progress_fname} stats_only=1")

  return chunk_index

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
  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_measure2.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_measure2_progress_header())

  if chunk_only:
    if debug_chunk_start<0:
      debug_chunk_start=0
    if debug_chunk_count<=0:
      debug_chunk_count=1

  if gpu_log_level>=1:
    print(f"[stream-gpu-config] N={N} records={total_records} bin={fname} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} sort_mode={gpu_sort_mode} chunk_only={1 if chunk_only else 0} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} progress={progress_fname} inner_log_level=0 measure2=1")

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

      # 92 MEASURE2: compute input stats before GPU timing, then suppress inner per-chunk logs.
      stats:List[int]=analyze_stream_chunk_input_stats(N,chunk_constellations)

      t0=datetime.now()
      if gpu_log_level>=1:
        print(f"[stream-gpu-chunk-start] N={N} chunk={chunk_index} off={off} m={m}")

      inner_gpu_log_level:int=0
      exec_solutions(N,chunk_constellations,True,gpu_block,gpu_max_blocks,inner_gpu_log_level,gpu_sort_mode,cross_stripe_safe)

      chunk_total:int=0
      if m>0:
        chunk_total=chunk_constellations[0]["solutions"]
      gpu_total+=chunk_total
      executed_chunks+=1
      t1=datetime.now()
      elapsed_text:str=str(t1-t0)[:-3]
      elapsed_ms:int=stream_elapsed_text_to_ms(elapsed_text)
      append_stream_progress(progress_fname,N,preset_queens,chunk_index,off,m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_text,elapsed_ms,chunk_total,gpu_total,total_records,stats)
      if gpu_log_level>=1:
        print(f"[stream-gpu-chunk-end] N={N} chunk={chunk_index} off={off} m={m} elapsed={elapsed_text} elapsed_ms={elapsed_ms} chunk_total={chunk_total} gpu_total={gpu_total}")

      off+=m
      chunk_index+=1

  if gpu_log_level>=1:
    print(f"[stream-gpu-summary] N={N} records={total_records} chunks={chunk_index} executed_chunks={executed_chunks} total={gpu_total} progress={progress_fname} measure2=1")

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
  global FUNCID_REORDER_V2_WINDOW_MULT,FUNCID_REORDER_V2_PHASE_JUMP,FUNCID_REORDER_V2_SWEEP_TEMP_OUTPUT

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
  bench_mode:int=0  # 0:normal, 1:N20 warmup repeat, 2:N19 preheat, 3:N18+N19 preheat, 4:N20 repeat3 sweep, 5:N20 repeat2 benchmark, 6:reorder-only debug, 7:chunk-only debug, 8:boundary-classification-only, 9:boundary-solution-summary, 10:boundary-classification-only + signature prune disabled, 11:stream-bin-build-only, 13:stream-input-stats-only, 14:funcid-reorder-v2-sim-only, 15:funcid-reorder-v2-gpu, 16:funcid-reorder-v2-sim-sweep
  reorder_window_mult:int=FUNCID_REORDER_V2_WINDOW_MULT
  reorder_phase_jump:int=FUNCID_REORDER_V2_PHASE_JUMP
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
      print("Usage: nqueens [-c | -g] [nmin nmax] [gpu_block gpu_max_blocks log_level sort_mode] [preset_queens] [bench_mode] [cross_stripe_safe] [debug_chunk_start] [debug_chunk_count] [reorder_window_mult] [reorder_phase_jump]")
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
      if bench_mode<0 or (bench_mode>11 and bench_mode!=13 and bench_mode!=14 and bench_mode!=15 and bench_mode!=16):
        print(f"[warning] unknown bench_mode={bench_mode}; using 0")
        bench_mode=0
    if bench_mode>=8:
      preset_queens_arg=requested_preset_arg
    else:
      if requested_preset_arg!=5:
        print(f"[warning] preset_queens={requested_preset_arg} is disabled in 77 normal modes; using 5")
      preset_queens_arg=5
    if bench_mode==14 or bench_mode==15:
      # 96 reorder modes use a short form; if omitted, measured-best w8_j7 is used:
      #   ... [preset_queens] [bench_mode] [reorder_window_mult] [reorder_phase_jump] [cross_stripe_safe]
      # Example:
      #   -g 22 22 32 484 1 0 5 14 8 5
      #   -g 22 22 32 484 1 0 5 15        # auto w8_j7
      #   -g 22 22 32 484 1 0 5 15 16 7   # manual override
      if argc >= 11:
        reorder_window_mult=int(sys.argv[10])
      if argc >= 12:
        reorder_phase_jump=int(sys.argv[11])
      if argc >= 13:
        cross_stripe_safe=(int(sys.argv[12])!=0)
      if argc > 13:
        print("Too many arguments")
        print("Usage reorder modes: nqueens -g nmin nmax block max_blocks log_level sort_mode preset_queens bench_mode[14|15] [reorder_window_mult] [reorder_phase_jump] [cross_stripe_safe]")
        return
    elif bench_mode==16:
      # mode 16 runs the fixed simulation sweep: window_mult=8,16,32 x phase_jump=5,7,11.
      if argc > 10:
        print("Too many arguments")
        print("Usage sweep sim: nqueens -g nmin nmax block max_blocks log_level sort_mode preset_queens 16")
        return
    else:
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
    print("Usage: nqueens [-c | -g] [nmin nmax] [gpu_block gpu_max_blocks log_level sort_mode] [preset_queens] [bench_mode] [cross_stripe_safe] [debug_chunk_start] [debug_chunk_count] [reorder_window_mult] [reorder_phase_jump]")
    return

  if reorder_window_mult<=0:
    print(f"[warning] reorder_window_mult={reorder_window_mult} is invalid; using 8")
    reorder_window_mult=8
  if reorder_phase_jump<=0:
    print(f"[warning] reorder_phase_jump={reorder_phase_jump} is invalid; using 7")
    reorder_phase_jump=7
  FUNCID_REORDER_V2_WINDOW_MULT=reorder_window_mult
  FUNCID_REORDER_V2_PHASE_JUMP=reorder_phase_jump

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
    if bench_mode==13:
      print(f"stream_stats_only: mode={bench_mode} preset={preset_queens_arg}")
    if bench_mode==14:
      print(f"funcid_reorder_v2_sim: mode={bench_mode} preset={preset_queens_arg}")
    if bench_mode==15:
      print(f"funcid_reorder_v2_gpu: mode={bench_mode} preset={preset_queens_arg}")
    if bench_mode==16:
      print(f"funcid_reorder_v2_sweep_sim: mode={bench_mode} preset={preset_queens_arg}")
  if bench_mode==14 or bench_mode==15 or bench_mode==16:
    print(f"funcid_reorder_v2_params: window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} param={funcid_reorder_param_tag()} reason={FUNCID_REORDER_V2_DEFAULT_REASON}")
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

    if bench_mode==13:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      stats_chunks:int=exec_solutions_gpu_bin_stream_stats_only(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"[stream-stats-only] N={N} preset_queens={preset_queens} records={stream_records} chunks={stats_chunks} bin={stream_fname} valid={1 if validate_bin_file(stream_fname) else 0}")
      print(f"{N:2d}:{0:18d}{0:17d}{text:>21s}    stream-stats-only")
      continue

    if bench_mode==14:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      reorder_fname,reorder_records,reorder_chunks=build_funcid_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"[funcid-reorder-v2-sim-only] N={N} preset_queens={preset_queens} records={stream_records} reordered_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} valid={1 if validate_bin_file(reorder_fname) else 0}")
      print(f"{N:2d}:{0:18d}{0:17d}{text:>21s}    funcid-reorder-v2-sim-only")
      continue

    if bench_mode==16:
      # SAFE SWEEP: 9本のN22 reordered .binを残すと小容量root diskを使い切るため、
      # output .bin は1本だけを上書き再利用する。progress TSV はparam別に残す。
      FUNCID_REORDER_V2_SWEEP_TEMP_OUTPUT=True
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      sweep_windows:List[int]=[8,16,32]
      sweep_phases:List[int]=[5,7,11]
      sweep_count:int=0
      wi:int=0
      while wi<len(sweep_windows):
        pj_i:int=0
        while pj_i<len(sweep_phases):
          FUNCID_REORDER_V2_WINDOW_MULT=sweep_windows[wi]
          FUNCID_REORDER_V2_PHASE_JUMP=sweep_phases[pj_i]
          one_t0=datetime.now()
          reorder_fname,reorder_records,reorder_chunks=build_funcid_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
          one_elapsed=datetime.now()-one_t0
          one_text=str(one_elapsed)[:-3]
          print(f"[funcid-reorder-v2-sweep-sim] N={N} preset_queens={preset_queens} records={stream_records} reordered_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} temporary_bin=1 param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} valid={1 if validate_bin_file(reorder_fname) else 0} elapsed={one_text}")
          sweep_count+=1
          pj_i+=1
        wi+=1
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"[funcid-reorder-v2-sweep-summary] N={N} preset_queens={preset_queens} records={stream_records} cases={sweep_count} elapsed={text}")
      print(f"{N:2d}:{0:18d}{0:17d}{text:>21s}    funcid-reorder-v2-sweep-sim")
      continue

    if use_gpu and N>=21 and bench_mode==15:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      reorder_fname,reorder_records,reorder_chunks=build_funcid_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      total:int=exec_solutions_gpu_bin_stream_funcid_reorder(N,reorder_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,False,debug_chunk_start,debug_chunk_count)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      status:str="ok" if expected[N]==total else f"ng({total}!={expected[N]})"
      print(f"[funcid-reorder-v2-gpu-done] N={N} source_records={stream_records} reordered_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP}")
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    {status}")
      continue

    if use_gpu and N>=21 and not (bench_mode==8 or bench_mode==9 or bench_mode==10 or bench_mode==14 or bench_mode==15 or bench_mode==16):
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
