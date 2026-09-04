# 376_frame_workqueue_spec.md

## rev376 — フレーム単位 persistent kernel + atomic ワークキュー設計仕様
## (設計専用リビジョン。`374Py_kernel_maxd14.cu`へのコード変更ゼロ)

362(`362_kernel_port_spec.md`、363実装のspec)と同じ「設計→実装」分離
パターンを踏襲する。本リビジョンは仕様策定のみで、377以降で実装に
着手する。

---

## 1. 動機(375実測の要約)

375で`374Py_kernel_maxd14.cu`を`-lineinfo`ビルドし、374確認済みの
N=21フィルタ済みデータから先頭15,488レコード(=1グリッド分、
grid-strideループが**1回も回らない**構成)を切り出してncu
SourceCountersを取得した。結果、上位ホットスポット(全体の23.5%)は
すべて`process_one_task`内の明示スタック(`stack[stack_ptr]`)の
push/pop分岐の**再収束点**(`BSYNC`/`BRA`)に集中していた。

grid-strideが1回も回っていない時点でこの偏りが出ているということは、
原因は「どのタスクをどのスレッドに割り当てるか」ではなく、**同一warp
内の32レーンが担当する32個の異なるタスクの探索木の深さ・広さが
ばらつき、一番深いレーンに他31レーンが待たされる**という、SIMT実行
モデル自体に起因する構造的な偏り(warp occupancy collapse / tail
effect)であることが実測で確定した。この診断は355/357のCodon側
知見(Divergent Branches=0、Avg Threads Executed 2〜3/32)と完全に
整合する。

**結論**: タスク単位(root task単位)でのスレッド割り当てをいくら
工夫しても(静的grid-strideでも、atomicによる動的取得でも)、1warp
32レーンが同時に処理する32タスクの複雑さのばらつきという根本原因は
解消しない。解消するには、**ワークの単位をタスク全体からDFSの
1ステップ(フレーム)まで細分化し、レーンが自分のタスクに縛られず
"今すぐ処理できるフレームなら何でも"引けるようにする**必要がある。

---

## 2. 現行アーキテクチャの整理

`process_one_task()`(161-561行目)は内部で明確に2フェーズに分かれる:

**フェーズ1: スケジュール事前計算(182-303行目)**
`ctrl0`/`markctrl`から`schedule_lo`/`schedule_hi`(各深さでの操作
nibbleを詰めたビットフィールド)・`terminal_depth`・
`terminal_base14`・`child_jmark_mask`・`future_check_mask`・
`root_action`を導出する。**タスクごとに1回だけ**実行され、以降の
DFS本体はこの結果だけを参照する(元の`ctrl0`/`markctrl`は二度と
参照しない)。

**フェーズ2: DFS本体(305-561行目)**
`root_action`による即終了(2/3)や`root_action==1`のシフト処理を経て、
`while(true)`ループで明示スタック(`stack[26]`、208バイト/スレッド)
を使い反復的に木を辿る。1ステップは概ね「1ビットを`cur_avail`から
取り出す→子状態を計算→(a)`terminal_depth`一致なら加算して終了、
(b)`nf==0`なら打ち切り、(c)それ以外は現在の残りを`push`しつつ
子へ`descend`」という形。**push(397-402行目)は1ステップにつき
高々1回**、その直後に同じスレッドが子を辿って処理を継続する
(=中断せずそのまま次のフレームへ"immediate descend")。

この「push即descend」という現行の制御フローが、まさに375で
ホットスポットとして特定された箇所である。

---

## 3. 新設計: フレームキュー方式

### 3.1 データ構造

**`TaskSchedule`(タスクごとに1回だけ計算、キャッシュ)**
```c
struct TaskSchedule {
    uint32_t schedule_lo, schedule_hi;      // 8B
    uint32_t terminal_depth;                 // 4B (int→uint32でパック)
    uint32_t terminal_base14;                // 4B (0/1のみだが将来のpaddingを避けffff32で確保)
    uint32_t child_jmark_mask;               // 4B
    uint32_t future_check_mask;              // 4B
    uint32_t w_lo;                           // 4B
    // bm/n3/n4はカーネル全体で共通の起動時引数なのでここには含めない
};                                            // 計28B/タスク
```
現行`process_one_task`のフェーズ1をこの構造体を埋める関数として
そのまま切り出す(ロジックは1バイトも変えない、戻り値の詰め方を
変えるだけ)。

**`Frame`(キューに出し入れする最小単位)**
```c
struct Frame {
    uint32_t task_id;      // TaskSchedule配列への添字
    uint32_t cur_ld, cur_rd, cur_col, cur_avail;  // 現在の盤面状態
    uint32_t cur_depth;    // 現在の深さ
};                          // 24B/フレーム
```
現行コードのスタックエントリ(`stack[stack_ptr]`/`stack[stack_ptr+1]`、
2×u64=16B、ただし`cur_depth`はavailの上位ビットに詰めている)と
ほぼ同じ情報量。フレーム単体では`TaskSchedule`を持たず、`task_id`で
間接参照する(1タスクにつき28Bを32レーン分複製せず、共有配列に
1回だけ置く)。

**グローバルワークキュー**
DFSの性質上LIFO(スタック型)がFIFOより自然(局所性が高く、深さ優先の
実行順序を保ちやすい)。固定長リングでなく、**単調増加インデックス
+ mod容量**のシンプルなリングバッファ+atomicで実装する:

```c
__device__ Frame*   g_queue;          // 事前確保した固定長バッファ
__device__ uint32_t g_queue_capacity; // 2の冪(mod演算をand化するため)
__device__ unsigned long long g_push_idx;  // atomicAdd専用、単調増加
__device__ unsigned long long g_pop_idx;   // atomicAdd専用、単調増加
__device__ int      g_active_workers; // 現在フレームを処理中のスレッド数
```

push: `idx = atomicAdd(&g_push_idx, 1); g_queue[idx % capacity] = frame;`
pop:  `idx = atomicAdd(&g_pop_idx, 1); frame = g_queue[idx % capacity];`
(単純化のため、`push_idx - pop_idx`が容量を超えないことを別途保証する
必要がある——4節で詳述)

### 3.2 カーネル構造(persistent kernel)

```c
__global__ void kernel_dfs_frame_queue(...) {
    // 起動時: stride(=BLOCK*MAX_BLOCKS=15488)スレッドのみ
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (;;) {
        Frame f;
        if (!try_pop(&f)) {
            if (atomicAdd(&g_active_workers, 0) == 0 && queue_truly_empty()) {
                break;  // 全ワーカーがidle かつ キューも空 = 本当に完了
            }
            continue;  // 他スレッドがまだ生産中の可能性 → スピン
        }
        atomicAdd(&g_active_workers, 1);

        TaskSchedule* ts = &g_schedules[f.task_id];
        // 現行DFS本体の「1ステップ分」を実行(453-530行目相当)
        //   - terminal一致 → atomicAdd(&results[f.task_id], contribution)
        //   - nf==0 → 何もせず捨てる(このフレームは終端)
        //   - それ以外 → 子フレームをpush。現行と違い、
        //     "残り"と"子"の両方をpushしてこのスレッドはキューに
        //     戻る(immediate descendしない) -- 375で特定した
        //     ホットスポットの直接的な解消策
        step_one_frame(&f, ts, meta_next, bm, n3, n4);

        atomicAdd(&g_active_workers, -1);
    }
}
```

**ここが現行との本質的な違い**: 現行は「pushした側」がそのまま
子を辿り続ける(1スレッド=1タスクの生涯所有)。新設計は**push後、
このスレッドは即座にキューに戻り、次に取れるフレームを引く**
(それが元のタスクの子かもしれないし、全く別のタスクの別の深さの
フレームかもしれない)。これによりwarp内の32レーンは「常にキュー上で
今すぐ実行可能なフレーム」を処理することになり、375で観測された
「一部レーンだけ深く沈み込み他が待つ」構造そのものを解消する。

### 3.3 初期シード

M個(最大2,025,282)のroot taskをキューへ投入する必要がある。
`TaskSchedule`をタスクごとに1回計算する前段カーネル(またはpush時に
遅延計算)を挟み、`Frame{task_id, root_ld, root_rd, root_col,
root_a, depth=0}`をM個分`push`してから本体のpersistentループを
開始する2段階起動を想定する(詳細設計は377で確定)。

---

## 4. 未解決の設計課題(377着手前に詰める必要あり)

1. **キュー容量の最悪ケース見積もり**: 1ステップで最大何フレーム
   pushされ得るか(現行は1、新設計で"残り+子"の2にするなら理論上は
   毎ステップ純増があり得る)。容量不足時の挙動(ブロック/リトライ/
   フォールバック)を定義する必要がある。
2. **終了判定のレース**: `g_active_workers==0 && queue空`を見た
   直後に別スレッドがpushする可能性(ABA問題)。多くの実装は
   「二重チェック+短いバリア」または「エポック方式」で対処するが、
   正当性への影響が大きいため慎重な設計が要る。
3. **メモリ配置**: `g_queue`をグローバルメモリに置く場合、現行の
   レジスタ/ローカルメモリ常駐スタック(208B/スレッド)と比べて
   明示的なメモリトラフィックが増える。L2キャッシュ挙動次第で
   むしろ悪化するリスクがあり、377で小規模な実験により実測する
   必要がある。
4. **`step_one_frame`への切り出し方**: 453-530行目のDFS本体1
   ステップ分を、現行のロジックを1バイトも変えずに関数分離できるか
   の精査(項目12「カーネル微修正はくじ引き」を踏まえ、コピー&
   最小限の変数名リネームのみで済ませ、ロジックの並べ替えはしない)。

---

## 5. 検証計画(377以降、段階的リスク低減)

358/359(pushガード除去で+65〜67%の壊滅的退行)の教訓を踏まえ、
以下の順で段階的に進める。**各段階で必ず正当性(363_kernel_
reference_sim.pyとのbyte-for-byte一致)を確認してからのみ次段階へ
進み、いかなる段階でも速度クレームは正当性確認後に限る**:

1. **377a**: キュー機構(push/pop/終了判定)だけを、ロジックを変えず
   ("push即descend"のまま、ただしpush/popを新queue経由に置き換えた
   だけ)実装し、キュー機構自体の正しさをまず単体検証する。この
   段階では速度改善は期待しない(むしろ悪化してよい) — 目的は
   キュー機構のバグ出しのみ。
2. **377b**: "push即descend"をやめ、真にキューから任意のフレームを
   引く persistent worker 方式に変更。ここで初めて375の
   ホットスポット解消を狙う。N=18相当の小規模データでncu
   SourceCountersを再取得し、375と同じ箇所(BSYNC/BRA)の割合が
   実際に下がっているかを確認してから、N=21全量での正当性+速度
   比較に進む。

---

## 6. まとめ

375の実測により、「タスク単位の動的スケジューリング」では解決しない
ことが判明した。真に必要なのは**フレーム単位のワークキュー**で
あり、本仕様はその構造(`TaskSchedule`/`Frame`/リングバッファ+
atomic push/pop/persistent kernel)を定義した。374のカーネルには
一切手を加えていない。377で段階的な実装に着手する。
