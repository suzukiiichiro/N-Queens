## 421 — 分割走行（418）を 419 のカーネルに載せ替える：複数 GPU・複数マシン・再開（カーネル変更なし、本番値不変）

### 位置づけ

- N=25 は 419-r3 で確定した（728,149,056 ms、オラクル一致。`420_README_append.md`）。次は N=26。
- N=26 は 24→25 の時間比 9.96 倍を当てると **A10G 1 枚で約 2,000 GPU 時間**（外挿、±数十%）。単発 84 日は現実的でなく、入力分割が前提。
- 418（10/26 納品・未実行）は分割の道具一式を持つが、**`418_kernel_maxd16.cu` は 417 のリネーム**で、N=25 を 0.16% 数え漏らしたスケジュール語の桁あふれを持ったまま。N≥25 に使う前に 419 へ載せ替える。それが 421。
- 費用は「GPU 時間×単価」で決まり、枚数・台数は期間を縮めるだけ。suzuki の判断（11/16）：進める。

| 構成（us-west-2、オンデマンド、2026-11-16 時点の公開価格） | 単価 | N=26 概算 | 期間 |
|---|---:|---:|---:|
| g5.12xlarge 1 台（A10G×4） | $5.672/h | 約 $2,840 | 約 21 日 |
| g5.2xlarge 4 台 | $1.212/h ×4 | 約 $2,420 | 約 21 日 |
| g5.xlarge 4 台（未測定：N=21 1 本で確認） | $1.006/h ×4 | 約 $2,010 | 約 21 日 |
| スポット（g5.xlarge、$0.53〜0.56/h、中断率 20% 超の区分） | — | 約 $1,100 | 21 日＋中断分 |

東京リージョンは g5.12xlarge $8.2261/h（約 1.45 倍）。NQ_CRunner／NQ_CRunner2 は us-west-2。

### 納品物（7 点）

| ファイル | 内容 |
|---|---|
| `421_kernel_maxd16.cu` | 419 のリネーム＋ヘッダの rev421 ノートのみ。code-region 差分 0/0、kernel sha `3b67e26c` |
| `421Py_kernel_maxd16_final.py` | 419Py から実行行 3 行（VERSION_TAG、REV_TAG、`CRunnerEntry(17, ./421_kernel_maxd16)`）＋421 ノート。419Py に無かった 418 ノートも収録 |
| `421_split_sched.py`／`421_sum_results.py` | 418 のリネーム（ロジック不変）。stride（30,720）の倍数で切る／要素ごとに足す |
| `421_multigpu_run.sh` | 418 のドライバ＋下の 3 点 |
| `421_validate.sh` | 4 GPU 機でのゲート |
| `421_README_append.md` | この節 |

ドライバの追加 3 点：

- **`WORKER_ID=i WORKER_TOTAL=W`**：チャンク k を `k % W == i` のマシンだけが走る。各マシンは同じ入力を同じ計画で分割し、自分の分を走って `partial` で終わる。実行時の共有（NFS・S3）は不要。
- **`SUM_ONLY=1 LOGDIR=<dir>`**：各マシンの `bins/part*_results.bin`・`3_part*.log`・`2_chunk_events*.tsv` を 1 つのディレクトリに集め、検査して足すだけ（GPU 不要、入力が無ければ `RECORDS=<件数>`）。
- **`RESUME=1`**：最新の `421_multigpu_N<N>_x<K>_*` を再利用（スポット再起動時の起動スクリプト用）。

ほかに、分割計画を bash 側でも計算して各チャンクのログの `records` と突き合わせる、再実行時は古い `part` の results.bin を消してから走る、を入れた。

### ゲート（`421_validate.sh`、NQ_CRunner2、約 3.8 時間・約 $22）

| | 内容 | 判定 |
|---|---|---|
| S0 | 静的：py 3/3（対 419Py）、cu 0/0（対 419）、kernel sha、ノート、PACK 定数、3 ビット語と abort の存在、split／sum の selftest、ドライバの WORKER／SUM_ONLY／RESUME、`bash -n`、gcc | ハード |
| S1 | ptxas 160／160／176／184／192 B、spill 0、regs ≤44 | ハード |
| S2〜S6 | 5 実体の SASS ＝ 419（`<425>` `4e3b9e8f`・実 454・ループ 111） | ハード |
| G0〜G3 | N=21 を各 GPU で単独に：results.bin ＝ 411 B1 `1bfab090` | ハード |
| P21x4 | N=21 を 4 分割・全 GPU：要素和 ＝ G0 バイト一致、オラクル | ハード |
| **P21w** | 同じ LOGDIR で WORKER 1/2（チャンク 1・3 だけ走り `partial` で終わること）→ WORKER なしで再実行（完了 2 を飛ばし 0・2 を走る）→ 要素和 ＝ G0 | ハード |
| **P21s** | その LOGDIR を `SUM_ONLY=1` で再合算 ＝ G0 | ハード |
| **P25Bx4** | **N=25 スライス B（307,200 件）を 4 分割：要素和 ＝ 419 の D960 results.bin（sha `c881e856`、総和 6,910,807,391,840）とバイト一致**。深さ 17・419 カーネル・別マシン・分割の端から端まで | ハード |
| S23 | N=23 一括（GPU 0）：オラクル 24,233,937,684,440 | ハード |
| P23x4／P23x8 | N=23 を 4・8 分割：要素和 ＝ S23 バイト一致、オラクル | ハード |

### 事前登録（利得は主張しない。本番値は不変）

- **W1** 分割費用＝Σkernel_ms（分割）／kernel_ms（一括）、N=23 で x4 1.02〜1.07、x8 1.04〜1.10（418 のまま。N=25 の尾は 418 の想定 5% より長い 7% だったので上限寄りを予想）。反証：x8 > 1.15、要素和の不一致。
- **W2** 4 GPU の壁時計加速 ≥ 3.4 倍。**W3** G0〜G3 の kernel_ms が互いに 1% 以内。
- **W4**（情報）P25Bx4 の Σkernel_ms ／ 3,099,327（419 の D960、別マシン）＝ 1.00〜1.10。
- 採用則：ハード全通過で **421 が本番バイナリ／`.py`**（SASS ＝ 419、本番値不変）、`421_multigpu_run.sh` が N≥25 のドライバ。N=26 のチャンク数は W1 で決める（x8 が x4＋0.03 以内なら細かい側）。

### 手元で確認したこと（GPU なし）

- `STATIC_ONLY=1 bash 421_validate.sh`：OK=23 FAIL=0（kernel sha `3b67e26c`、cu 0/0、py 3/3）。
- 模擬ランナー（スレッド別和を返す）と模擬 `nvidia-smi` で、ハーネスのセル全部（G0〜G3、P21x4、P21w、P21s、P25Bx4、S23、P23x4、P23x8）を端から端まで：OK=52 FAIL=0。
- ドライバ単体：チャンク途中で殺す→`RESUME=1` で同じディレクトリを拾って完走、2 台（WORKER 0/2・1/2、別 LOGDIR）→集めて `SUM_ONLY=1`（入力なし・`RECORDS=`）でバイト一致、part を 1 ビット壊すと FAIL、part を 1 個欠くと FAIL。
- 実機の nvcc／SASS／GPU は未確認（NQ_CRunner2 が容量不足で起動待ち）。

### 手順（NQ_CRunner2、起動できたら）

```
# 1. git pull（421 の 7 点、419_kernel_maxd16.cu、419Py_kernel_maxd16_final.py、413_slice_soa7.py が入る）
# 2. NQ_CRunner から転送するもの（bin は AMI に入っていれば不要。sha256sum で突き合わせる）
#      constellations_N21_6.bin.soa_ref_361.bin.maxd14only_363.bin.sched394f.bin              (57 MB)
#      constellations_N23_7.bin.soa_ref_361.bin.maxd16only_413.bin.sched394f.bin              (1.24 GB)
#      constellations_N25_7.bin.soa_ref_361.bin.d17only_417.bin.sched394f.bin                 (2.75 GB)
#      411_validate_20261006_150938/bins/B1_results.bin                                       (N=21 の基準 1bfab090)
#      419_validate_20261104_214321/bins/D960_results.bin                                     (スライス B の基準 c881e856)
#      419_r3_full25_20261104_232738/bins/G25_results.bin                                     (N=25 の基準 bd2de957)
#    sched25 を送らない場合は NQ_CRunner の 419_n25_sliceB_307200.bin (8.6 MB) を送り SLB=419_n25_sliceB_307200.bin を付ける
STATIC_ONLY=1 bash 421_validate.sh                                   # 約 1 分、GPU ゼロ
screen -S nq421
set -o pipefail; bash 421_validate.sh 2>&1 | tee 421_validate_stdout.log
#   約 3.8 時間（S23 2 時間 8 分、P23x4／x8 各約 35 分、P25Bx4 約 20 分）。SKIP_N23=1 なら約 35 分（W1／W2 は測れない）
# 完了後 421_validate_<ts>.tar.gz と 421_validate_stdout.log を送付。検証が済んだら NQ_CRunner2 は停止してよい
```

複数台で走らせるとき（422 以降の本番で使う形）：

```
# マシン i（i = 0..W-1、各 1 GPU）で
WORKER_ID=i WORKER_TOTAL=W CHUNKS=<K> bash 421_multigpu_run.sh 2>&1 | tee 421_multigpu_w${i}.log
#   中断されたら同じ行に RESUME=1 を付けて再実行（完了済みチャンクは飛ばす）
# 全台が終わったら 1 か所に集めて
SUM_ONLY=1 LOGDIR=<集めたディレクトリ> REF_RESULTS=<あれば基準> bash 421_multigpu_run.sh
```

### この先

- **422＝N=26**：カーネルは祖先 17 段（`PACK425_ANCESTOR 17`、`PACK425_MAXN 26`、フレーム 204→208 B 見込み）。スケジュール語（3 ビット×21 深さ）は深さ 18 を収めるが、恒久ルール 39 のとおり深さに比例する語（`future_check_mask`、`child_jmark_mask` ほか）を全部 `required_maxd` と突き合わせる。入力は preset 7 で約 1.5 億件・`required_maxd` 18 の見込み（未確認）、394f sched のチャンク化（NQ_CRunner2 は RAM 192 GB なので構築はそちらが楽）。独立参照 `419_ref_dfs.c` は深さ 31 まで。N=26 のオラクルは `.py` の表の値を使う。
- 尾の測定：P23x4／x8 のチャンク別 kernel_ms と clk から、チャンクあたりの尾を読む。N=26 のチャンク数と、尾の短縮（sched 末尾の並べ替え）をやるかどうかを決める材料。
- g5.xlarge の確認（N=21 1 本、2 分）：本番値 97,946 ms と同じなら 1 枚機は g5.xlarge が最安。
- 請求の見張り：AWS Budgets の通知。分割走行はチャンク単位で止めて再開できるので、予算に達したら止められる。

---
