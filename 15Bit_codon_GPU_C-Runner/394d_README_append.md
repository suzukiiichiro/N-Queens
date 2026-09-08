## 394d — 394c セルB(+29.7%)の段階分解ラダー

**カーネル無変更。** `394d_kernel_maxd14.cu`は394cのヘッダーのみリネーム
(コード領域sha256一致)。Codon側は`bench_mode=39`に位置引数ノブを追加
しただけ(394cPyからのコア差分 `added=71 removed=15`)。

### 動機

394cで、chunkshape148済み入力が本番stride(15,488)で**+29.7%**、968では
−1.0%。968側は`iter_sort`がcap警告で無効化されていたため、strideと
iter_sortが交絡している。データ順序だけでカーネル時間が30%動く以上、
どの段が原因かを特定しないと、この軸を活かすことも閉じることもできない。

### 変更内容(mode 39のノブ、`argv[10..15]`)

| 位置 | 内容 | 既定 |
|---|---|---|
| argv[10] | window_mult | 3 |
| argv[11] | phase_jump | 7 |
| argv[12] | cross_stripe_safe | 0 |
| argv[13] | bucket_run | 2048(1=276-338順) |
| argv[14] | iter_sort | 9(0=344順、1=346順) |
| argv[15] | **input_stage** | 2=chunkshape済み(394c)、**1=broadmarktail baseのみ** |

ノブはmode 30/31のパーサが書くのと**同じローカル**へ書き、直後に無条件で
グローバル(`CHUNKSHAPE148_BUCKET_RUN`等)へコピーされるので、キャッシュ
ファイル名(`_run2048`/`_isort9`の有無)とビルダーは30/31と同一挙動。
argc=10の呼び出しは394cと無変更。mode 39は起動時に
`crunner_reordered_params:`行で有効値を出力し、ハーネスが渡した値と
照合する(ノブが解釈されなかった場合のゲート)。

### ラダー(すべてMAX_BLOCKS=484、stride 15,488)

| セル | mode | 内容 | 加わる段 |
|---|---|---|---|
| A | 37 | 生 | (アンカー) |
| E1 | 39 stage=1 | broadmarktail base のみ | funcid w3_j7 並べ替え+rotate |
| E2 | 39 run=1 isort=0 | 276-338順 | + scorestripe(lanephase32) |
| E3 | 39 run=2048 isort=0 | 344順 | + bucket_run=2048 |
| E4 | 39 run=2048 isort=1 | 346順 | + iter_sort=1 |
| E5 | 39 run=2048 isort=9 | 350順 = 394c B | + isort9(条件付きserpentine) |

各段はパイプラインの構成要素を**1つずつ**加える。最初に+20%以上跳ねる
段が原因を名指しする。全セルN=21フル、オラクル・stride結合・ノブ反映を
ゲート。所要約30分(6実行+chunkshape生成3回)、sudo不要。

`NCU=1`(sudo)で、生とE5のchunk0スライス(743,424件=48周回=並べ替えの
周期)にSchedulerStats+WarpStateStatsを取る(各約9分)。
`Avg. Active Threads/Warp`が落ちればwarp内(レーン利用率)、
`Active Warps/Scheduler`が落ちればwarp間(launchの尻尾)。

### 事前登録予測(実測前に固定)

1. **A**は201,237を±1%で、**E5**は394cのB(261,060)を±1%で再現する。
2. **H_isort(主仮説)**: E1・E2・E3はAの±3%以内、跳ねはE4および/またはE5
   で現れる。根拠: 968側のD(iter_sort強制無効)が−1.0%だった。
3. **反証条件**: E3が既に+20%以上ならH_isortは誤りで、原因は
   scorestripe/bucket_run側。E1が既に+20%以上なら原因はbroadmarktail
   (funcid並べ替え+rotate)。
4. E4(iter_sort=1)とE5(isort9)が両方跳ねるなら、原因は「per-thread列を
   コストで単調にする」こと自体。E5だけなら条件付きserpentine固有。
5. NCU=1を回した場合: H_isortが正しければE5で`Avg. Active Threads/Warp`
   が生より下がる(warp内)と予測する。`Active Warps/Scheduler`は1.51前後で
   不変。

### 394dの後

- 原因の段が確定したら、その段を**除いた**並べ替え(例: E3 = 344順)を
  968側で再評価する。Dの−1.0%はまさに「isortなし」の構成なので、
  968での期待値は小さいが、484側で段ごとの符号が分かれば、逆方向の
  設計(C版向けのper-thread列)を395系の候補に加えられる。
- 並行して394b r3(484〜968の細分)で`MAX_BLOCKS`最適点をブラケットし、
  本番採用へ。


## 394d 結果(確定)・原因は`iter_sort`ではなく**broadmarktail baseそのもの**(+47.5%)

**実行完了。** 6段すべてオラクル一致、stride結合・ノブ反映ゲート全通過。
A=201,238(389比+0.0005%)、E5=261,063(394cのB 261,060を**0.001%**で再現)。

| セル | 内容 | `kernel_ms` | vs A | 段の寄与 |
|---|---|---:|---:|---:|
| A | 生 | 201,238 | — | — |
| **E1** | **broadmarktail base のみ**(funcid w3_j7 + rotate) | **296,902** | **+47.5%** | **+47.5pt** |
| E2 | + scorestripe(run=1, isort=0) | 295,850 | +47.0% | −0.5pt |
| E3 | + bucket_run=2048 | 288,149 | +43.2% | −3.8pt |
| E4 | + iter_sort=1 | 258,831 | +28.6% | **−14.6pt** |
| E5 | + isort9(= 394c B) | 261,063 | +29.7% | +1.1pt |

### 事前登録予測の判定

| # | 予測 | 実測 | 判定 |
|---|---|---|---|
| 1 | A・E5がアンカー±1% | +0.0005% / +0.001% | 的中 |
| 2 | **H_isort**: E1〜E3はA±3%、跳ねはE4/E5 | **E1で+47.5%** | **反証** |
| 3 | E1が+20%以上なら原因はbroadmarktail | 該当 | 反証条件が成立 |
| 4 | E4/E5両方が跳ねるなら単調化自体が原因 | 両方とも**改善**方向 | 前提不成立 |

**H_isortは反証された。** 罰の全額はbroadmarktail base(funcid w3_j7
並べ替え+rotate、rev 94-99期にN=22・当時のカーネル向けに調整された段)に
あり、後段はすべてそれを**部分的に修復**しているだけである
(scorestripe −0.5pt、bucket_run −3.8pt、iter_sort=1 −14.6pt、isort9は
isort1より+1.1pt悪い)。パイプライン全体でも生より+29.7%悪い。

### 意味

- Codon経路での並べ替えの「改善」(w3 vs w8 −1.29%、bucket_run vs RUN=1
  −4.11%、isort9 vs isort1 −0.82%)は**すべて並べ替え同士の比較**であり、
  現行カーネルで生順序と比較した記録は見当たらない。C版で観測した
  「後段は前段の修復」という構造がCodon版でも成り立っている可能性が
  ある(未検証。Codon側は`bench_mode=0`(生bin、`exec_solutions_gpu_bin_
  stream`)対mode 31で直接測れる)。
- **394cのD(968、−1.0%)の説明**: 968向けの並べ替えはisortがcapで
  無効化され、かつbase自体が別ファイル(`_m968_s30976`、66周回)で
  あった。484側のE3相当(+43.2%)と968側のD(−1.0%)がここまで違う以上、
  罰は「並べ替えの内容」ではなく「並べ替えとstrideの関係」に依存する
  ——394eで機構を切り分ける。

### 機構の候補(394eで判定)

- **M1 warp内**: baseの並べ替えがwarp内32レーンの発散を生順序より
  悪化させる。
- **M2 スレッド間**: baseの並べ替えがper-thread累積作業量(131タスク)を
  不均衡にし、rotateがそれを均しきれず、launchが長い尻尾で終わる
  (生は393 s3で常駐99.8%=尻尾なし、なので尻尾が現れる余地がある)。
  E4(iter_sort=1)が−14.6pt修復したことはM2と整合する——iter_sortの
  グループ(48周回×15,488=743,424件)全体をコストでソートして配り直す
  ので、per-thread総量を変えられる。
