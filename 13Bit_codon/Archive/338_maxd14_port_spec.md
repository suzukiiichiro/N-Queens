# 338: `kernel_dfs_iter_gpu_maxd14` CUDA C移植仕様書

リビジョン: **338 (maxd14-port-design)** / 作成日: 2026-07-27
位置づけ: **設計のみ。コードは一切追加しない**(324→325と同じパターン)
実装(`.cu`)は339以降。

---

## 0. スコープと前提

### 0.1 変えないもの(Codon側に残すもの)

| 責務 | 担当 | 根拠 |
|---|---|---|
| コンステレーション生成 | Codon (`gen_constellations_stream_to_bin`) | 実行時間に占める割合が小さく、移植の価値がない |
| broadmarktail / chunkshape148 / funcid_reorder (w3_j7) | Codon | 333で正式採用済みのホスト側パラメータ。C側に持ち込むと変数が増えすぎる |
| `.bin`キャッシュ | Codon | 337で形式を仕様確定・実データ検証済み |
| N=21フルの正当性オラクル(`314666222712`) | Codon | 既存ハーネス |

### 0.2 移植するもの

**`kernel_dfs_iter_gpu_maxd14` 一個のみ**、およびそのランチャーと、bin→SoA変換のホスト側コード。
`maxd16/18/20/21`は移植しない(N=21・split145では`required_maxd=14`が3チャンク全てで成立しており、337の実測ログでも`selected_MAXD=14`が3回。深いスケジュールへのフォールバックはCodon側にのみ存在すればよい)。

### 0.3 移植の目的(性能ではなく、まず能力の解禁)

移植そのものが速くなることを期待しているのではない。目的は Open Objectives 課題3 (tail effect) に対して、Codonでは使えない以下を解禁することである:

- warp intrinsics (`__shfl_sync` / `__ballot_sync` / `__activemask`)
- デバイスatomic (`atomicAdd` 等) — **330でCodonには存在しないことが確定済み**
- per-lineのSASS帰属 (`-lineinfo`。Codonの`-debug`は不正なPTXを生成するため永久に不可)

したがって移植版の初期性能はCodon版と同等(±数%)であれば成功とみなす。**遅くても、上記3つが使えるようになった時点で価値がある。**

### 0.4 337までに確定済みの前提(再掲、すべて実機確認済み)

| 項目 | 値 | rev |
|---|---|---|
| `nvcc` | `/usr/local/cuda/bin/nvcc` (CUDA 13.0, V13.0.88) | 334-335 |
| A10G compute capability | **8.6** → `-arch=sm_86` | 335 |
| `cuobjdump` | 同梱されていない。SASSは`sudo ncu --section SourceCounters --page source`に一本化 | 335 |
| nvccビルド+GPU実行round-trip | 成功 (`match=True`) | 336 |
| `.bin`レコード形式 | ヘッダーなし・16バイト固定長・`ld`/`rd`/`col`/`startijkl`(各u32 LE) | 337 |
| 実ファイル検証 | `records_read=2025282`(=`EXPECTED_TASKS`)、`checksum_u64=13342728758502` | 337 |

---

## 1. binレコード → SoA全配列の導出仕様

### 1.1 全体像

```
.bin (16B/record)                     GPUカーネルが読む配列
+--------------------+               +----------------------------+
| ld       (u32 LE)  |               | ld_arr       : u32[m]      |
| rd       (u32 LE)  |  ==変換==>    | rd_arr       : u32[m]      |
| col      (u32 LE)  |               | col_arr      : u32[m]      |
| startijkl(u32 LE)  |               | ctrl0_arr    : u32[m]      |
+--------------------+               | free_arr     : u32[m]      |
                                     | markctrl_arr : u32[m]      |
                                     | w_lo_arr     : u32[m]      |
                                     | w_hi_arr     : u32[m] (=0) |
                                     +----------------------------+
```

Codon側の対応関数は `build_soa_for_range`(1767行)と `symmetry`(2519行)。
**この変換はホスト側で1回行うだけであり、GPUカーネルには一切含まれない。**
したがってC側でもホスト関数として実装する(移植の難所はここではなく、正確な写経であること)。

### 1.2 前処理(レコード共通)

```c
// N: 盤面サイズ, rec: 16バイトレコードから読んだ4つのu32
const uint32_t board_mask = (N >= 32) ? 0xFFFFFFFFu : ((1u << N) - 1u);
const uint32_t small_mask = (1u << (N - 2)) - 1u;   // N >= 2
const int N1 = N - 1;
const int N2 = N - 2;

const uint32_t start_ijkl = rec.startijkl;
const int start = (int)(start_ijkl >> 20);
const int ijkl  = (int)(start_ijkl & 0xFFFFFu);

const int i = (ijkl >> 15) & 0x1F;
const int j = (ijkl >> 10) & 0x1F;
const int k = (ijkl >>  5) & 0x1F;
const int l =  ijkl        & 0x1F;

uint32_t ld  = rec.ld  >> 1;
uint32_t rd  = rec.rd  >> 1;
uint32_t col = ((rec.col >> 1) | ~small_mask) & board_mask;

const uint32_t LD = (1u << (N1 - j)) | (1u << (N1 - l));
ld |= LD >> (N - start);

if (start > k) {
    ld |= 0;                                    /* no-op: ld unchanged */
    rd |= (1u << (N1 - (start - k + 1)));
}
if (j >= 2*N - 33 - start) {
    rd |= (1u << (N1 - j)) << (N2 - start);
}

const uint32_t freev = board_mask & ~(ld | rd | col);
```

**Codonとの差異に関する注意点(必ずassertで守ること):**

1. `col` の `~small_mask` はPythonでは任意精度の負数だが、直後に `& board_mask` されるためu32演算と結果は一致する。C側では `uint32_t` で計算してよい。
2. `LD >> (N - start)`: `0 <= N - start <= N <= 21 < 32` を仮定している。**`start <= N` をassertする。**
3. `(1u << (N1 - j)) << (N2 - start)`: **`N2 - start >= 0`、すなわち `start <= N-2` を条件成立時にassertする。** Pythonは負のシフトで例外を投げるため今まで顕在化していないだけであり、Cでは未定義動作になる。この分岐がN=18/N=21の全レコードで安全であることは、移植の最初の検証項目である(§3.2 チェック1)。
4. `1u << (N1 - (start - k + 1))`: 同様に **`start - k + 1 <= N1`** をassertする。

### 1.3 funcid決定木(28葉)

`jmark`/`endmark`/`mark1`/`mark2` はすべて**初期値0**。以下は `build_soa_for_range`(1809-1939行)の完全な写像である。

```c
int jmark = 0, endmark = 0, mark1 = 0, mark2 = 0, target = 0;

const bool k_lt_l     = (k < l);
const bool start_lt_k = (start < k);
const bool start_lt_l = (start < l);
const bool l_eq_kp1   = (l == k + 1);
const bool k_eq_lp1   = (k == l + 1);
const bool j_gate     = (j > 2*N - 34 - start);

if (j < N - 3) {                       /* ---- 枝A ---- */
    jmark = j + 1;
    endmark = N2;
    if (j_gate) {
        if (k_lt_l) {
            mark1 = k - 1; mark2 = l - 1;
            if (start_lt_l) {
                if (start_lt_k) target = (!l_eq_kp1) ? 0 : 4;
                else            target = 1;
            } else              target = 2;
        } else {
            mark1 = l - 1; mark2 = k - 1;
            if (start_lt_k) {
                if (start_lt_l) target = (!k_eq_lp1) ? 5 : 7;
                else            target = 6;
            } else              target = 2;
        }
    } else {
        if (k_lt_l) { mark1 = k - 1; mark2 = l - 1; target = (!l_eq_kp1) ?  8 :  9; }
        else        { mark1 = l - 1; mark2 = k - 1; target = (!k_eq_lp1) ? 10 : 11; }
    }
} else if (j == N - 3) {               /* ---- 枝B ---- */
    endmark = N2;
    if (k_lt_l) {
        mark1 = k - 1; mark2 = l - 1;
        if (start_lt_l) {
            if (start_lt_k) target = (!l_eq_kp1) ? 12 : 15;
            else          { mark2 = l - 1; target = 13; }   /* 冗長代入(原典どおり) */
        } else            target = 14;
    } else {
        mark1 = l - 1; mark2 = k - 1;
        if (start_lt_k) {
            if (start_lt_l) target = (!k_eq_lp1) ? 16 : 18;
            else          { mark2 = k - 1; target = 17; }   /* 冗長代入(原典どおり) */
        } else            target = 14;
    }
} else if (j == N - 2) {               /* ---- 枝C ---- */
    if (k_lt_l) {
        endmark = N2;
        if (start_lt_l) {
            if (start_lt_k) {
                mark1 = k - 1;
                if (!l_eq_kp1) { mark2 = l - 1; target = 19; }
                else           { target = 22; }
            } else { mark2 = l - 1; target = 20; }          /* mark1 は 0 のまま */
        } else target = 21;
    } else {
        if (start_lt_k) {
            if (start_lt_l) {
                if (k < N2) {
                    mark1 = l - 1; endmark = N2;
                    if (!k_eq_lp1) { mark2 = k - 1; target = 23; }
                    else           { target = 24; }
                } else {
                    if (l != N - 3) { mark2 = l - 1; endmark = N - 3; target = 20; }
                    else            { endmark = N - 4;                target = 21; }
                }
            } else {
                if (k != N2) { mark2 = k - 1; endmark = N2;    target = 25; }
                else         {                endmark = N - 3; target = 21; }
            }
        } else { endmark = N2; target = 21; }
    }
} else {                               /* ---- 枝D: j == N-1 ---- */
    endmark = N2;
    if (start > k) target = 26;
    else { mark1 = k - 1; target = 27; }
}
```

**写経時の落とし穴(3つとも実際にバグの温床):**

- 枝Cの `else { mark2 = l-1; target = 20; }` では **`mark1` は設定されず0のまま**である。上位の枝Aや枝Bのように「先に mark1,mark2 をまとめて代入」しているわけではない。
- 枝Bの2箇所の `mark2 = l-1` / `mark2 = k-1` は直前の代入と同値の**冗長代入**である。原典に合わせて残すか削るかは自由だが、削るなら「同値であること」をコメントで明示する。
- `jmark` は**枝Aでのみ非ゼロ**になる。枝B/C/Dでは常に0。

### 1.4 パッキング

```c
ctrl0_arr[t]    = (uint32_t)target | ((uint32_t)start << 5);
markctrl_arr[t] = ((uint32_t)(jmark   & 31))
                | ((uint32_t)(endmark & 31) << 5)
                | ((uint32_t)(mark1   & 31) << 10)
                | ((uint32_t)(mark2   & 31) << 15);
ld_arr[t] = ld;  rd_arr[t] = rd;  col_arr[t] = col;  free_arr[t] = freev;
```

`markctrl` が使用するのは **bit 0-19 のみ**。bit 20-31 は空きである(→ Open Objectives #7)。

### 1.5 w(対称性重み)の導出

```c
static inline int rot90_eq(int ijkl, int N) {
    const int i = (ijkl >> 15) & 0x1F, j = (ijkl >> 10) & 0x1F;
    const int k = (ijkl >>  5) & 0x1F, l =  ijkl        & 0x1F;
    const int r = ((N-1-k) << 15) + ((N-1-l) << 10) + (j << 5) + i;
    return ijkl == r;
}

static inline uint64_t symmetry_w(int ijkl, int N) {
    if (rot90_eq(ijkl, N)) return 2;
    const int i = (ijkl >> 15) & 0x1F, j = (ijkl >> 10) & 0x1F;
    const int k = (ijkl >>  5) & 0x1F, l =  ijkl        & 0x1F;
    if (i == N-1-j && k == N-1-l) return 4;
    return 8;
}
```

**重要な帰結**: 値域は `{2, 4, 8}` の3値のみ。したがって

- `w_hi_arr[t]` は**常に0**(→ Open Objectives #7)
- `w = 1 << e`、`e ∈ {1,2,3}`
- カーネルの `total * w` は `total << e` に退化する

C移植版では**最初から`w_lo_arr`/`w_hi_arr`を持たず、`markctrl`のbit20-21に`e`を詰める形で実装してよい**。ただしその場合、Codon版との「同一入力→同一出力」比較(§3)は成立し続けるが、**カーネル間のバイト等価性は失われる**ため、§3の比較は必ず最終結果(`total`)と中間の`chunk_total`で行い、配列のバイト比較で行わないこと。

---

## 2. K-batchingとスタックのC側メモリレイアウト

### 2.1 起動構成(337実測ログで確認済み)

| 記号 | 値 | 出所 |
|---|---|---|
| `BLOCK` | 32 | `A10G_FINAL_DEFAULT_BLOCK` |
| `MAX_BLOCKS` (=`GRID`) | 484 | `A10G_FINAL_DEFAULT_MAX_BLOCKS` |
| `THREADS` (=`stride`) | **15488** | `BLOCK * MAX_BLOCKS` |
| `K_PER_THREAD_MAXD14` | **48** | ソース定数(294行) |
| `STEPS` | **743424** | `THREADS * K` |

**`K`は「1スレッドが必ず48個処理する」という意味ではない。** `STEPS = THREADS*K` は**チャンクサイズの上限**であり、SoA配列と`results`配列の確保サイズでもある。実際の1スレッド当たり反復回数は `ceil(m / stride)`。

337のN=21実測(`progress_full.tsv`):

| chunk | m | 反復/スレッド |
|---|---|---|
| 0 | 743424 | 48(ちょうど上限) |
| 1 | 743424 | 48 |
| 2 | 538434 | 35 |
| 計 | **2025282** | — |

C側でも同じ意味論を維持する:

```c
const int BLOCK  = 32;
const int GRID   = 484;
const int STRIDE = BLOCK * GRID;          /* 15488 */
const int K      = 48;
const int STEPS  = STRIDE * K;            /* 743424 : 配列確保サイズ */
/* kernel<<<GRID, BLOCK>>>(..., m, board_mask, n3, n4, STRIDE); */
```

### 2.2 グリッドストライドループ

```c
const int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid >= stride) return;
uint64_t thread_total = 0;
for (int idx = tid; idx < m; idx += stride) { /* ... 1コンステレーションのDFS ... */ }
results[tid] = thread_total;
```

**リダクション範囲**: ホスト側は `results[0 .. stride)` のみを合計する(`sum_count = kbatch_stride`)。`STEPS`個確保してあるが、maxd14経路では先頭`stride`個以外は常に0のまま。C側でもこの規約を守ること(`STEPS`全体を合計しても結果は同じだが、無駄。かつCodon側との差分デバッグ時に混乱する)。

### 2.3 スレッド当たりスタック

Codon: `stack = __array__[u64](MAXD14_ANCESTOR * 2)` = `u64[26]` = **208バイト/スレッド**。
`packed_stack_bytes_per_thread(14)` が返す208と一致し、337実測ログの `stack_bytes_per_thread=208` とも一致する。

```c
/* ローカル配列。__shared__ にはしないこと(現行と同じレジスタ/ローカル配置を維持) */
uint64_t stack[13 * 2];        /* MAXD14_ANCESTOR = 13 */
int stack_ptr = 0;             /* = save_sp * 2 (乗算回避のため直接維持: rev296) */
int save_sp   = 0;
```

パック規約(294-296で確定、変更しないこと):

```
stack[ptr    ] = ldrd  : lo32 = ld,  hi32 = rd
stack[ptr + 1] = colav : lo32 = col, hi32 = avail | (depth << 27)
```

`depth`は`avail`の上位ビットに相乗りしている(`avail`はboard_mask以下なのでN=21では bit0-20 しか使わず、bit27以降が空く)。

```c
/* push */
stack[stack_ptr    ] = (uint64_t)cur_ld  | ((uint64_t)cur_rd << 32);
stack[stack_ptr + 1] = (uint64_t)cur_col | ((uint64_t)(cur_avail | ((uint32_t)cur_depth << 27)) << 32);
stack_ptr += 2;  save_sp += 1;
```

### 2.4 移植で**触ってはいけない**もの(過去の失敗の記録)

| 事項 | 根拠 |
|---|---|
| ホットループの分解・カーネル分割 | 240 / 266-269 / 273 / 326 の**5戦5敗**。326は-13.7%で最悪 |
| `save_sp` / `next_depth` の除去 | 297 (+2.5%) / 298 (+18%)。命令スケジューリング用のバッファとして機能している |
| `cur_depth`の別表現 | 301/302/303 いずれも +30%級の大幅悪化 |
| `K`の変更 | 304-308で K=44〜56 が平坦な最適域と確定済み |
| `BROADMARK_VARIANT` | 309/310で variant=2 (rotate_only) 以外は +5〜6%悪化。確定 |

**C移植版は、まず「Codon版の逐語訳」を作ること。** 最適化は移植の正当性が確認できた後、単一変数実験として行う。移植と最適化を同一リビジョンで混ぜない。

---

## 3. N=18限定の正当性確認スコープ

rev84の原提案どおり、**N=18で移植版と既存Codon版の出力を突き合わせる**。N=21フルはその後。

### 3.1 なぜN=18か

| 指標 | N=18 | N=21 |
|---|---|---|
| 正解値 | **666090624** | 314666222712 |
| Codon版 実行時間 | **約2.0秒** | 約450秒 |
| 反復サイクルの現実性 | 1分に何十回でも回せる | 1日数回が限度 |

337時点のソースヘッダにあるN=18の実測値(`0:00:02.005`)がそのままオラクルとして使える。

### 3.2 突き合わせの段階(上から順に、各段が通ってから次へ)

**チェック1 — ホスト側SoA変換の一致(GPU不使用)**
N=18のbinを両実装で読み、SoA全配列のチェックサムを比較する。

- 比較対象: `ld_arr` / `rd_arr` / `col_arr` / `ctrl0_arr` / `free_arr` / `markctrl_arr` / `w` の7本
- 方式: 337の`checksum_u64`と同じ畳み込みを配列ごとに行い、7個のu64を出力して突き合わせる
- Codon側には**この7個を出力するだけの診断モードを追加する必要がある**(カーネル非接触。339の作業項目)
- ここで §1.2 のassert 3件(シフト量非負)も同時に検証される

**チェック2 — カーネル出力の一致(GPU使用、1チャンク)**
同一のSoA配列を入力として、`results[0..stride)` の合計が一致することを確認する。

- N=18は `m` が小さく1チャンクで収まる見込み。`required_maxd=14` になることを先に確認すること(なっていなければN=18ではmaxd14経路を通らず、この検証自体が無意味になる)
- 一致基準: `chunk_total` の完全一致

**チェック3 — N=18の最終値**
`666090624` に一致すること。

**チェック4 — N=21フル**
`314666222712` に一致すること。ここまで通って初めて性能比較の土俵に乗る。

### 3.3 性能比較の判定基準(先に決めておく)

移植版の初期実装に対する判定を**事前に**固定する:

| 結果 | 判定 |
|---|---|
| N=21 elapsed が 450.056s ±3% 以内 | **成功**。warp intrinsics / atomic / `-lineinfo` が解禁された時点で目的達成 |
| +3% 〜 +15% | **条件付き成功**。逐語訳の精度を疑う前に、まず`-lineinfo`でSASSを取り差分を見る(Codonでは不可能だったことが可能になっている) |
| +15%超 | 逐語訳にズレがある。§2.4の「触ってはいけないもの」を再点検 |
| 改善 | 想定外。歓迎するが、必ず正当性を再確認してから受け入れる |

**移植それ自体で速くなることは期待していない**(§0.3)。この表を先に書いておくのは、実測後に基準を動かさないためである。

---

## 4. 339以降の作業順序(暫定)

| rev | 内容 | GPU |
|---|---|---|
| 339 | Codon側にSoAチェックサム診断モードを追加(カーネル非接触) + N=18のSoAチェックサムを採取 | 不要 |
| 340 | C側 bin→SoA変換の実装 + チェック1(§3.2) | 不要 |
| 341 | C側 `kernel_dfs_iter_gpu_maxd14` の逐語訳 + チェック2/3 | 必要 |
| 342 | N=21フル(チェック4)+ 性能比較 | 必要 |

**ただし、この順序は Open Objectives #8 の測定結果によって組み替える可能性がある。** #8(warp内コスト不均質のΣmax32/Σmean32測定)は、C移植を待たずにCodon側だけで実施でき、かつ tail effect に対して移植より直接的な打ち手になりうる。#8の上限値が大きければ、#8を先に取りに行くほうが期待値が高い。

---

## 5. 付録: 移植時に手元に置いておくべき定数

```c
/* funcidビットマスク(カーネル内で使用、値はCodon版と同一) */
#define IS_BASE_MASK   69222408u
#define IS_JMARK_MASK          4u
#define IS_MARK_MASK  199209203u
#define IS_P5_MASK          3840u
#define SEL2_MASK      34742338u

#define BLOCK_CODE_B0_MASK 173707345u
#define BLOCK_CODE_B1_MASK  12689458u
#define BLOCK_CODE_B2_MASK  18088064u

#define OP_STEP3_MASK 24u   /* codes 3,4 */
#define OP_ADD1_MASK  32u   /* code 5   */
#define OP_BL1_MASK   12u   /* codes 2,3 */
#define OP_BL2_MASK   16u   /* code 4   */
#define OP_KN3_MASK   18u   /* codes 1,4 */
#define OP_KN4_MASK    8u   /* code 3   */

/* meta_next[28] (4921行と同一) */
static const unsigned char meta_next[28] = {
    1, 2, 3, 3, 2, 6, 2, 2, 0, 4, 5, 7, 13, 14,
   14,14,17,14,14,20,21,21,21,25,21,21,26,26
};

/* カーネル引数 */
/* n3 = 1u << (N-3),  n4 = 1u << (N-4),  board_mask = (1u<<N)-1 */
```

`STATIC_ONLY`静的チェックと同じ規律で、これらの定数はC側でも**行番号付きで出典を明記**しておくこと(将来Codon側が変わったときに追随漏れを検出するため)。

---

## 6. 未解決のまま残す事項

1. `-lineinfo` による per-line SASS 帰属が実際に機能するかは未検証(336のスモークテストはビルド・実行のみを確認した)。341で最初に確認する。
2. C版で `w` を `markctrl` に畳み込む場合(§1.5)、Codon版との配列レベル比較ができなくなる。§3.2チェック1の比較対象を `w` を含む7本にするか、`w`込みの`markctrl`で別扱いにするかは340で決める。
3. N=18で `required_maxd == 14` になるかは未確認。**339のチェック1と同時に必ず確認すること**(ならない場合、N=18は検証対象として使えず、N=19か20に切り替える必要がある)。
