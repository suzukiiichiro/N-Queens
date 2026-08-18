#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Python/codon Ｎクイーン コンステレーション版 マージ

15Pyをマージした。
SQB..のバックトラック用関数をdfs関数に１本化した。
exec_solutionsの関数の呼び出しをdfsに変更した。

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


結論から言えば codon for python 17Py_ は GPU/CUDA 10Bit_CUDA/01CUDA_Bit_Symmetry.cu と同等の速度で動作します。

 $ nvcc -O3 -arch=sm_61 -m64 -ptx -prec-div=false 04CUDA_Symmetry_BitBoard.cu && POCL_DEBUG=all ./a.out -n ;
対称解除法 GPUビットボード
20:      39029188884       4878666808     000:00:02:02.52
21:     314666222712      39333324973     000:00:18:46.52
22:    2691008701644     336376244042     000:03:00:22.54
23:   24233937684440    3029242658210     001:06:03:49.29

amazon AWS m4.16xlarge x 1
$ codon build -release 15Py_constellations_optimize_codon.py && ./15Py_constellations_optimize_codon
20:      39029188884                0          0:02:52.430
21:     314666222712                0          0:24:25.554
22:    2691008701644                0          3:29:33.971
23:   24233937684440                0   1 day, 8:12:58.977

python 15py_ 以降の並列処理を除けば python でも動作します
$ python <filename.py>

codon for python ビルドしない実行方法
$ codon run <filename.py>

codon build for python ビルドすればC/C++ネイティブに変換し高速に実行します
$ codon build -release < filename.py> && ./<filename>


詳細はこちら。
【参考リンク】Ｎクイーン問題 過去記事一覧はこちらから
https://suzukiiichiro.github.io/search/?keyword=Ｎクイーン問題

エイト・クイーンのプログラムアーカイブ
Bash、Lua、C、Java、Python、CUDAまで！
https://github.com/suzukiiichiro/N-Queens
"""


"""
 N-Queens Constellations Solver (NQueens16)

 本プログラムは N-Queens を「開始コンステレーション（部分盤面）」で分割し、
 各サブ問題をビットボードで高速探索します。Zobrist ハッシュや正規化(Jasmin)により
 重複抑止とキャッシュ効率を上げています。

 ▶ 主要コンセプト / 実装の見どころ
 - ビットボード:
     左/右対角線・列をビットで管理。次行遷移で
     `(ld|bit)<<1`, `(rd|bit)>>1`, `col|bit` を用いる（例: dfs() 内）。
 - Zobrist ハッシュ:
     `_init_zobrist()` で N ごとのテーブルを作り、`zobrist_hash()` で
     各ビット集合（`ld/rd/col/LD/RD`）を **N-bit に正規化後** XOR 加算。
     例:
         `ld &= mask; rd &= mask; col &= mask; LD &= mask; RD &= mask`
     衝突耐性重視の探索済み検出に利用（軽量版 `state_hash()` も併用）。
 - コンステレーション列挙:
     `gen_constellations()` が代表盤面を列挙し、`set_pre_queens*()` が
     preset_queens 個の事前配置を再帰で展開。
     空きマスクは
         `free = ~(ld | rd | col | (LD>>(N-1-row)) | (RD<<(N-1-row))) & ((1<<N)-1)`
     を基本形とする。
 - 対称性排除&正規化:
     `check_rotations()` で 90/180/270°の回転重複を排除。
     `jasmin()`（`get_jasmin()` キャッシュ付）で代表形に写像。
 - 解探索分岐:
     `exec_solutions()` が j/k/l の位置関係・端からの距離に応じて
     分岐ラベル（"SQd0B", "SQd1BkBlB", …）を選択し、`dfs()` に委譲。
     重複補正は `symmetry()`（90°:2, 180°:4, その他:8）で掛ける。
 - キャッシュ:
     サブ星座生成の再実行防止（`set_pre_queens_cached()` で `StateKey` set）、
     さらに TXT/BIN キャッシュ（`load_or_build_constellations_*`）。

 ▶ 設計上のポイント / レビュー
 - ハッシュの N-bit 正規化:
     `zobrist_hash()` 先頭で `& ((1<<N)-1)` を徹底。負数や上位ビット汚染を回避。
 - 軽量 visited キー:
     `visited.add(self.state_hash(...))` により O(1) 計算・単一 int キーで高速＆省メモリ。
     高信頼検証時は `zobrist_hash()` に切替える A/B も容易。
 - Jasmin 正規化の指標:
     端からの距離 `ffmin(x, N-1-x)` を比較し、回転回数 `arg` を決定→必要ならミラー。
 - 分岐の抽象化:
     本来多数の再帰関数を、`dfs(funcname=...)` で集約。ロジックの核は
     「最下位 1bit 取り出し → 次行の遮蔽を作る → 先読み `_should_go_plus1()`」。
 - 先読みと追加遮蔽:
     `extra = _extra_block_for_row(...)` で j/k/l の“その行に入る瞬間の追加ブロック”を付与。
     その上で `_has_future_space_step()` により「次行が完全に詰んでいないか」を早期判定。
 - I/O の堅牢性:
     TXT/BIN の相互変換を用意。BIN は 1 レコード=16B を `validate_bin_file()` で検査。

 ▶ 使い方
     `NQueens16_constellations().main()` を実行。
     N=5..17 で検算が入り、N>5 は:
         1) gen_constellations() → 2) exec_solutions() → 3) 集計

 ▶ 引用スニペット（本ファイルより）
   - Zobrist マスク: `ld &= mask; rd &= mask; col &= mask; LD &= mask; RD &= mask`
   - 代表形写像: `for _ in range(arg): ijkl = self.rot90(ijkl, N); if self.getj(ijkl) < N-1-self.getj(ijkl): ijkl=self.mirvert(ijkl,N)`
   - 追加遮蔽: `_extra_block_for_row(row_next, mark1, mark2, jmark, N)`
   - 先読み: `_should_go_plus1(next_free, row_next, endmark, next_ld, next_rd, next_col, board_mask, extra)`
   - 対称補正: `constellation["solutions"] = cnt * self.symmetry(ijkl, N)`


amazon AWS m4.16xlarge x 1
$ codon build -release 16Py_constellations_merge_codon.py && ./16Py_constellations_merge_codon
 N:        Total       Unique        hh:mm:ss.ms
 5:           10            0         0:00:00.000
 6:            4            0         0:00:00.079
 7:           40            0         0:00:00.001
 8:           92            0         0:00:00.001
 9:          352            0         0:00:00.001
10:          724            0         0:00:00.002
11:         2680            0         0:00:00.102
12:        14200            0         0:00:00.002
13:        73712            0         0:00:00.005
14:       365596            0         0:00:00.011
15:      2279184            0         0:00:00.035
16:     14772512            0         0:00:00.078
17:     95815104            0         0:00:00.436
18:    666090624            0         0:00:02.961
19:   4968057848            0         0:00:22.049
20:  39029188884            0         0:02:52.430
21: 314666222712            0         0:24:25.554
22:2691008701644            0         3:29:33.971

top - 11:02:30 up 16:46,  4 users,  load average: 64.17, 64.13, 64.10
Tasks: 566 total,   2 running, 564 sleeping,   0 stopped,   0 zombie
%Cpu(s):100.0 us,  0.0 sy,  0.0 ni,  0.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
MiB Mem : 257899.4 total, 256292.0 free,   1156.5 used,    451.0 buff/cache
MiB Swap:      0.0 total,      0.0 free,      0.0 used. 255398.4 avail Mem

    PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND
   5634 suzuki    20   0   13.3g  58440   7384 R  6399   0.0  63650:42 15Py_constellat
  42987 suzuki    20   0  224032   4176   2956 R   0.7   0.0   0:00.09 top
     16 root      20   0       0      0      0 I   0.3   0.0   0:11.65 rcu_preempt
      1 root      20   0  171244  16904  10532 S   0.0   0.0   0:06.11 systemd
      2 root      20   0       0      0      0 S   0.0   0.0   0:00.06 kthreadd

GPU/CUDA
10Bit_CUDA/01CUDA_Bit_Symmetry.cu
19:       4968057848        621012754     000:00:00:13.80
20:      39029188884       4878666808     000:00:02:02.52
21:     314666222712      39333324973     000:00:18:46.52
22:    2691008701644     336376244042     000:03:00:22.54
23:   24233937684440    3029242658210     001:06:03:49.29
24:  227514171973736   28439272956934     012:23:38:21.02
25: 2207893435808352  275986683743434     140:07:39:29.96
"""



"""
16Py_constellations_merge_codon.py（レビュー＆注釈つき）

✅[Opt-07] Zobrist Hash による transposition / visited 状態の高速検出
ビットボード設計でも、「盤面のハッシュ」→「探索済みフラグ」で枝刈りは可能です。
例えば「既に同じビットマスク状態を訪問したか」判定、もしくは部分盤面パタ>ーンのメモ化など。

Opt-07（状態キャッシュ）の反映箇所まとめ
先頭付近（import 近く）
from typing import List, Set, Dict, Tuple
StateKey = Tuple[int, int, int, int, int, int, int, int, int, int, int]

__init__（インスタンス属性に）
self.subconst_cache: Dict[StateKey, bool] = {}
set_pre_queens(...) の引数
visited: Set[StateKey] に変更

冒頭で
key: StateKey = (ld, rd, col, row, queens, k, l, LD, RD, N, preset_queens)
if key in visited: return / visited.add(key)

set_pre_queens_cached(...) の引数
visited: Set[StateKey] に変更
subconst_cache も Dict[StateKey, bool] キーで同じタプルを使用

gen_constellations(...) の呼び出し側
visited: Set[StateKey] = set() で初期化
この構成で、Codon の型検査に対しても一貫性が取れていて、実行結果もOKです。

「探索済み状態の検出（transposition/visited）による枝刈り」は 実装済み です。
set_pre_queens / set_pre_queens_cached で、
key = (ld, rd, col, row, queens, k, l, LD, RD, N, preset_queens) を visited / subconst_cache に用いて再訪を防いでいます。

 ただし厳密な意味での 「Zobrist Hash」自体は使っていません。
現状は「タプル状態キー→visited判定」という方式です（十分に効果的で、Codon でも安定）。

いまのタプル方式でも正答・速度ともに良好なので、Opt-07（visited枝刈り）としては完了扱いでOKです。
"""

"""
✅[Opt-08]部分盤面サブ問題キャッシュ
場所: set_pre_queens_cached(...)
キー: key = (ld, rd, col, k, l, row, queens, LD, RD, N, preset_queens)
値: subconst_cache[key] = True
役割: 同じ部分状態でのサブ問題展開を一度だけにする（必ず再利用の方針に合致）。
"""

"""
✅[Opt-09]訪問済み（transposition / visited）
場所: set_pre_queens(...)
データ構造: visited（Set：実装版ではタプル or 64bit相当の圧縮キー）
役割: 再帰木を横断して同じ状態への再訪を防止。
"""

"""
✅[Opt-10]Jasmin 正規化キャッシュ
場所: get_jasmin(c, N) / jasmin_cache: Dict[Tuple[int,int], int]
役割: 盤面正規化（回転・鏡映）結果をメモ化し、同一候補の重複計算を回避。
"""

"""
✅[Opt-11]星座（コンステレーション）重複排除
場所: constellation_signatures: Set[Tuple[int,int,int,int,int,int]]
役割: 生成済み部分盤面（星座）を一意に保つための署名セット。
"""

"""
✅[Opt-12]永続キャッシュ（現状は無効化中）
場所: load_constellations(...) / pickle
インスタンス内キャッシュ（辞書／集合）
  __init__
    self.subconst_cache: Dict[StateKey, bool] = {} … サブコンステ生成の再入防止
    self.constellation_signatures: Set[Tuple[int,int,int,int,int,int]] = set() … 星座の重複署名
    self.jasmin_cache: Dict[Tuple[int,int], int] = {} … get_jasmin()の結果メモ化
    self.zobrist_tables: Dict[int, Dict[str, List[int]]] = {} … Zobristテーブル（Nごと）

✅[Opt-13]部分盤面のキャッシュ（tuple化→dict）
  set_pre_queens_cached(...)
    キー：(ld, rd, col, k, l, row, queens, LD, RD, N, preset_queens)
    既出キーなら再帰呼び出しスキップ → 指数的重複カット

✅[Opt-14]星座（コンステレーション）の重複排除
  set_pre_queens(...) 内 if queens == preset_queens: ブロック
    署名：(ld, rd, col, k, l, row) を self.constellation_signatures で判定し重複追加を抑制

✅[Opt-15]Jasmin 正規化のメモ化
  get_jasmin(c, N) → self.jasmin_cache[(c,N)]
  何度も登場する起点パターンの再計算を回避

✅[Opt-16]訪問済み状態（transposition/visited）の仕込み
  gen_constellations(...) で visited: Set[StateKey] = set() を生成し
  set_pre_queens(...) 冒頭で key: StateKey = (...) を visited に登録・参照
  ※Zobrist版 zobrist_hash(...) も実装済（今はコメントアウトでトグル可）

    # 状態ハッシュによる探索枝の枝刈り バックトラック系の冒頭に追加　やりすぎると解が合わない
    #
    # zobrist_hash
    # 各ビットを見てテーブルから XOR するため O(N)（ld/rd/col/LD/RDそれぞれで最大 N 回）。
    # とはいえ N≤17 なのでコストは小さめ。衝突耐性は高い。
    # マスク漏れや負数の扱いを誤ると不一致が起きる点に注意（先ほどの & ((1<<N)-1) 修正で解決）。
    # h: int = self.zobrist_hash(ld, rd, col, row, queens, k, l, LD, RD, N)
    #
    # state_hash
    # その場で数個の ^ と << を混ぜるだけの O(1) 計算。
    # 生成されるキーも 単一の int なので、set/dict の操作が最速＆省メモリ。
    # ただし理論上は衝突し得ます（実際はN≤17の範囲なら実害が出にくい設計にしていればOK）。
    h: int = self.state_hash(ld, rd, col, row,queens,k,l,LD,RD,N)
    if h in visited:
        return
    visited.add(h)
    #
    # StateKey（タプル）
    # 11個の整数オブジェクトを束ねるため、オブジェクト生成・GC負荷・ハッシュ合成が最も重い。
    # set の比較・保持も重く、メモリも一番食います。
    # 衝突はほぼ心配ないものの、速度とメモリ効率は最下位。
    # key: StateKey = (ld, rd, col, row, queens, k, l, LD, RD, N, preset_queens)
    # if key in visited:
    #     return
    # visited.add(key)

✅[Opt-17]星座リストの外部キャッシュ（ファイル）
  テキスト：save_constellations_txt(...) / load_constellations_txt(...)
  バイナリ：save_constellations_bin(...) / load_constellations_bin(...)
  ラッパ：load_or_build_constellations_txt(...) / load_or_build_constellations_bin(...)
    load_or_build_constellations_bin(...)
    破損チェック validate_constellation_list(...) / validate_bin_file(...) あり
"""

"""
✅[Opt-18] 星座生成（サブコンステレーション）にtuple keyでキャッシュ
set_pre_queens やサブ星座生成は、状態変数を tuple でまとめて key にできます。これで全く同じ状態での星座生成は1度だけ実行されます。

__init__ で self.subconst_cache: Dict[StateKey, bool] = {} を用意
set_pre_queens_cached(...) が tupleキー
  (ld, rd, col, k, l, row, queens, LD, RD, N, preset_queens)
  を使って self.subconst_cache を参照・更新
生成側は gen_constellations(...) から 最初の呼び出しを set_pre_queens_cached に変更済み
再帰内でも次の分岐呼び出しを set_pre_queens_cached(...) に置換しており、同一状態の再実行を回避
"""

"""
✅[Opt-19] 星座自体をtuple/hashで一意管理して重複を防ぐ
constellationsリストに追加する際、既に存在する星座を再追加しない
→ 星座自体を「tuple/int/hash」にして集合管理
これにより、異なる経路から同じ星座に到達しても重複追加を防げます。

__init__ で self.constellation_signatures: Set[Tuple[int, int, int, int, int, int]] = set() を用意。
set_pre_queens(...) 内の if queens == preset_queens: ブロックで
signature = (ld, rd, col, k, l, row) をキーに重複チェックし、未出だけ constellations.append(...) ＆ counter[0] += 1。
"""

"""
✅[Opt-20] Jasmin変換キャッシュ（クラス属性またはグローバル変数で）
（生成済み盤面の再利用）
ijkl_list_jasmin = {self.jasmin(c, N) for c in ijkl_list} も、盤面→jasmin変換は「一度計算したらdictでキャッシュ」が効果大
#グローバル変数で

def get_jasmin(self, c: int, N: int) -> int:
    key = (c, N)
    if key in jasmin_cache:
        return jasmin_cache[key]
    result = self.jasmin(c, N)
    jasmin_cache[key] = result
    return result

# 使用例:gen_constellations()内に
ijkl_list_jasmin = {self.get_jasmin(c, N) for c in ijkl_list}

__init__ に self.jasmin_cache: Dict[Tuple[int, int], int] = {}

get_jasmin(self, c: int, N: int) で (c, N) をキーに memo 化

gen_constellations() 内で

ijkl_list = { self.get_jasmin(c, N) for c in ijkl_list }
としてキャッシュ経由で Jasmin 変換しています
"""

"""
✅[Opt-21] 180°重複チェックの二重化
check_rotations() は 90/180/270°すべて見ていますが、奇数 N の中央列ブロックで check_rotations(...) と
rot180_in_set(...) を両方呼んでいますね。ここは rot180 が 重複なので、check_rotations(...)
のみでOK（微小ですが内包表記が軽くなります）。

# 修正前（中央列ブロック）
ijkl_list.update(
    self.to_ijkl(i, j, center, l)
    for l in range(center + 1, N - 1)
    for i in range(center + 1, N - 1)
    if i != (N - 1) - l
    for j in range(N - center - 2, 0, -1)
    if j != i and j != l
    if not self.check_rotations(ijkl_list, i, j, center, l, N)
    if not self.rot180_in_set(ijkl_list, i, j, center, l, N)  # ←これを削除
)

# 修正後（中央列ブロック）
ijkl_list.update(
    self.to_ijkl(i, j, center, l)
    for l in range(center + 1, N - 1)
    for i in range(center + 1, N - 1)
    if i != (N - 1) - l
    for j in range(N - center - 2, 0, -1)
    if j != i and j != l
    if not self.check_rotations(ijkl_list, i, j, center, l, N)
)
"""

"""
✅[Opt-22] visited の粒度
visited を星座ごとに新規 set() にしているので、メモリ爆発を回避できています。ハッシュに ld, rd, col, row, queens, k,
l, LD, RD, N まで混ぜているのも衝突耐性◯。
  gen_constellations() の各スタート（星座）ごとに
  visited: Set[StateKey] = set() を新規作成
  StateKey = (ld, rd, col, row, queens, k, l, LD, RD, N, preset_queens) を追加・照合
  という構成なので、
  visited のスコープが星座単位 → メモリ増大を回避できている
  衝突耐性は ld/rd/col/LD/RD の**ビット集合＋行インデックスやカウンタ（row/queens）＋分岐（k/l）**まで含むので十分に高い

  gen_constellations() の各スタート（星座）ごとに
  visited: Set[StateKey] = set() を新規作成
  StateKey = (ld, rd, col, row, queens, k, l, LD, RD, N, preset_queens) を追加・照合
  という構成なので、visited のスコープが星座単位 → メモリ増大を回避できている
  衝突耐性は ld/rd/col/LD/RD の**ビット集合＋行インデックスやカウンタ（row/queens）＋分岐（k/l）**まで含むので十分に高いでOKです。

  細かい改善ポイント（任意）：
  N と preset_queens は探索中は一定なので、キーから外しても挙動は変わりません（キーが少し短くなります）。もちろん入れたままでも正しいです。
  もし将来 state_hash() に切り替えるときも、visited を星座ごとに new にする方針はそのまま維持してください（グローバルにしない）。
"""

"""
✅[Opt-23] ビット演算のインライン化・board_mask の上位での共有・**1ビット抽出 bit = x &
-x**など、要所は押さえられています。
cnt を星座ごとにローカルで完結→solutions に掛け算（symmetry()）という流れもキャッシュに優しい設計。
これ以上を狙うなら、「星座ごと分割の並列度を広げる」か「gen_constellations の ijkl_list.update(...)
での回転重複除去を最小限に（=set操作の負荷を減らす）」の二択ですが、現状の速度を見る限り十分実用的です。

  いまの実装は
  ビット演算の徹底（bit = x & -x／board_maskの共有／blocked→next_freeの短絡判定）
  cnt を星座ローカルで完結→最後に symmetry() を掛けるフロー
  visited を星座ごとに分離
  など、ボトルネックをしっかり押さえられていて実用速度も十分です。
  さらに“やるなら”の小粒アイデア（任意）だけ置いておきます：
  symmetry(ijkl, N) の結果を小さな dict でメモ化（星座件数分の呼び出しを削減）。
  gen_constellations での set 操作を減らしたい場合は、候補を一旦 list に溜めて最後に
  （i）jasmin 変換 →
  （ii）set に流し込み（重複除去）
  という“1回だけの set 化”に寄せる（ただし回転除去の粒度は保つ）。
  並列度をもう少しだけ広げるなら、exec_solutions の @par は維持しつつ、constellations を大きめのチャンクに分割してワーカーに渡す（1件ずつよりスレッド起動回数が減るのでスケジューリング負荷が下がることがあります）。
  無理にいじるより、現状のバランス（読みやすさ×速度）を維持で十分だと思います。
"""

"""
✅[Opt-24]“先読み空き” の条件
先読み関数 _has_future_space() を使った
if next_free and ((row >= endmark-1) or _has_future_space(...)):
の形は、**「ゴール直前は先読み不要」**という意図に合っていて良い感じ。境界で row+1 >= endmark か row >= endmark-1
を使い分けている箇所も一貫しています。

  各再帰で
  next_free = board_mask & ~blocked
  if next_free and ((row >= endmark-1) or self._has_future_space(next_ld, next_rd, next_col, board_mask)):
  （1行進む再帰は row+1 >= endmark／2行進む再帰は row+2 >= endmark などに合わせて判定）
  という形になっており、
  ゴール直前は先読み不要（短絡評価で _has_future_space を呼ばない）
  それ以外は**“1行先に置ける可能性が1ビットでもあるか”**の軽量チェックでムダ分岐を削減
  がきれいに機能しています。

  軽い補足（任意）：
  「+1 進む」「+2 進む」系で row+Δ >= endmark の Δ を必ず合わせる（すでに合わせてありますが、この一貫性が重要）。
  ループ先頭で if not next_free: continue の早期スキップを入れるのも読みやすさ的に○（実測差は小さいことが多いです）。
  _has_future_space 内の式は現在の実装（board_mask & ~(((next_ld<<1)|(next_rd>>1)|next_col)) != 0）で十分速いです。
  総じて、境界条件と短絡評価の使い方が意図に合っており、問題ありません。
"""

"""
##------------------------------------------------------------------------
# 以下は対応不要、または対応できない一般的なキャッシュ対応
##------------------------------------------------------------------------

❎ 未対応 並列とキャッシュの整合
@par は exec_solutions の星座単位で独立になっているので、インスタンス属性のキャッシュは
生成段階（gen_constellations）で完結しており競合しません。jasmin_cache・subconst_cache を
インスタンス属性にしたのは正解。

  結論：今の実装（constellation → SQ＊の深い再帰、盤面は bitmask だけで保持）では “途中段階の部分対称除去”
  は基本的に入れないほうが安全 です。入れるなら設計を少し変える必要があります。
"""

"""
❎ 未対応 1行目以外の部分対称除去
jasmin/is_partial_canonicalで排除
途中段階（深さ r の盤面）を都度「辞書順最小」の canonical かどうかチェックして、そうでなければ枝刈り
→ 各 SQ〜() の再帰関数の while free: の直前にこの判定を入れ、False なら continue。
  結論：board変数にrowの配置情報を格納していないので対応不可
  結論：今の実装（constellation → SQ＊の深い再帰、盤面は bitmask だけで保持）では “途中段階の部分対称除去” は基本的に入れないほうが安全 です。入れるなら設計を少し変える必要があります。
  なぜそのままは危険／効果が薄いか
  対称代表（orbit representative）の一貫性
  すでに gen_constellations 側で 初手左右半分＋コーナー分岐＋Jasmin で代表選択をしています。この方針と、途中深さでの is_partial_canonical() の代表規則がズレると、合法枝を誤って落とす／重複排除が二重に効いて過剰枝刈りが起きます（以前の Zobrist で総数が減った現象と同系の事故になりやすい）。
  partial 変換の定義が難しい
  D4（回転・鏡映）の作用は行と列を同時に入れ替えます。bitmask だけを持つ現在の SQ 再帰状態（ld/rd/col と row、さらに j/k/l の特殊制約）に、“部分盤面を回した後も同じ制約系になるか” を正しく合成するのがかなり大変です。
  カウント重み（COUNT2/4/8）との整合
  最終的な重み付けが「代表だけ探索して最後に 2/4/8 を掛ける」設計なので、途中で代表以外を落とす規則はこの重み付けと厳密に一致していなければなりません。

# -----------------------------------
# [Opt-07] 部分盤面 canonical 判定
def is_partial_canonical(board: List[int], row: int, N: int) -> bool:
  # 現在の board[0:row] が他のミラー・回転盤面より辞書順で小さいか
  current = tuple(board[:row])
  symmetries = []
  # ミラー（左右反転）
  mirrored = [N-1 - b for b in current]
  symmetries.append(tuple(mirrored))
  # 90度回転：盤面を (col → row) に再構築する必要がある（簡略化版）
  # 完全な回転は行列転置＋ミラーが必要（時間コストあり）
  return all(current <= s for s in symmetries)
# -----------------------------------
"""

"""
❎ 未対応 軽量 is_canonical() による“部分盤面”の辞書順最小チェックを高速化（キャッシュ/軽量版）
「完成盤」だけでなく“部分盤面”用に軽量な変換（行の回転・反転は途中情報だけで可）を実装。
 is_partial_canonical() の中で zobrist_cache[hash] = True/False として使う

  「部分盤面の辞書順最小（canonical）チェック」は、基本的に
  board[row] = col_bit（＝各行に置いた列位置が順に分かる配列/スタック）
  もしくは cols_by_row = [c0, c1, ..., c(r-1)] のように「置いた列の履歴」
  を常に持っている設計で効果を発揮します。
  現在の “constellation 方式”（ld/rd/col のビットマスク＋row、k,l など）では、
  行ごとの「どの列に置いたか」の履歴が再構成できない
  回転・反転の変換を ld/rd/col に対して途中段階で正しく適用するのが難しい（ld/rd は「行進」に依存した相対ビット）
  ため、そのままでは実装が難しいです。

  いまの設計のまま：既存の対称性除去を強化する
  →あなたのコードは既に
  初手生成での左右対称除去・コーナー分岐
  Jasmin 変換（代表系化）
  180°重複チェック
  さらに set_pre_queens_cached の状態キャッシュ
  が入っているので、部分盤面 canonical は無理に足さなくても充分に強いです。
"""

"""
❎ 未対応 マクロチェス（局所パターン）    達成    violate_macro_patterns関数（導入済ならOK）
→ violate_macro_patternsのようなローカルな局所配置判定関数を挟む設計で達成
結論：board変数にrowのは位置情報を格納していないので対応不可

  violate_macro_patterns のような「局所（2～3行内）の並びで即座に弾く」系は、
  典型的には board[row] = col（行→列の履歴）を常に持っていてこそ強い枝刈りになります。
  いまの constellations 方式（ld/rd/col の集合状態＋row,k,l）だと、**「直前・直前々行でどの列に置いたか」**が直接分からないため、
  一般的な「近傍パターン」判定を素直に書くのは難しいです。
  もっとも、あなたの実装はすでに
  初手生成の厳しい対称性制約
  Jasmin 代表系化
  各種 SQ* 系の分岐（実質“マクロ手筋”をパターンとして埋め込んでいる）
  が効いているので、汎用の violate_macro_patterns を後付けする必要性は低めです。
"""



import pickle, os
from typing import List,Set,Dict,Tuple
from datetime import datetime

# 64bit マスク（Zobrist用途）
MASK64: int = (1 << 64) - 1
StateKey = Tuple[int,int,int,int,int,int,int,int,int,int,int]


class NQueens16:
  """N-Queens 分割探索（星座列挙→解探索）を担う中核クラス。
  - サブ星座生成キャッシュ: self.subconst_cache（StateKey の set）
  - 代表盤判定: constellation_signatures
  - Jasmin 正規化キャッシュ: jasmin_cache[(ijkl,N)] -> ijkl'
  - Zobrist テーブル: zobrist_tables[N] = {'ld','rd','col','LD','RD','row','queens','k','l'}
  """

  def __init__(self)->None:
    """内部キャッシュ構造の初期化。
    subconst_cache/constellation_signatures/jasmin_cache/zobrist_tables/gen_cache を準備する。
    """
    self.subconst_cache: Set[StateKey] = set()
    self.constellation_signatures: Set[ Tuple[int, int, int, int, int, int] ] = set()
    self.jasmin_cache: Dict[Tuple[int, int], int] = {}
    self.zobrist_tables: Dict[int, Dict[str, List[int]]] = {}
    self.gen_cache: Dict[Tuple[int,int,int,int,int,int,int,int], List[Dict[str,int]] ] = {}

  def _mix64(self, x: int) -> int:
    """splitmix64 終段の 64bit ミキサ。Zobrist 用疑似乱数を生成。
    - 入出力は MASK64 で正規化（`x &= MASK64`）。
    """
    # splitmix64 の最終段だけ使ったミキサ
    x &= MASK64
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & MASK64
    x ^= (x >> 31)
    return x & MASK64

  def _gen_list(self, cnt: int, seed: int) -> List[int]:
    """Zobrist 用 64bit 値を cnt 個作成。
    - 生成器を使わずリストで返し（Codon 互換性配慮）。
    - `s += 0x9E3779B97F4A7C15; out.append(self._mix64(s))`
    """
    out: List[int] = []
    s: int = seed & MASK64
    for _ in range(cnt):
        s = (s + 0x9E3779B97F4A7C15) & MASK64   # splitmix64 のインクリメント
        out.append(self._mix64(s))
    return out

  def _init_zobrist(self, N: int) -> None:
    """盤サイズ N 向け Zobrist テーブルを一度だけ初期化。
    - キー: 'ld','rd','col','LD','RD','row','queens','k','l'
    - 再入防止: 既に作成済みなら return
    """
    if N in self.zobrist_tables:
      return
    base_seed: int = (0xC0D0_0000_0000_0000 ^ (N << 32)) & MASK64
    tbl: Dict[str, List[int]] = {
        'ld'    : self._gen_list(N, base_seed ^ 0x01),
        'rd'    : self._gen_list(N, base_seed ^ 0x02),
        'col'   : self._gen_list(N, base_seed ^ 0x03),
        'LD'    : self._gen_list(N, base_seed ^ 0x04),
        'RD'    : self._gen_list(N, base_seed ^ 0x05),
        'row'   : self._gen_list(N, base_seed ^ 0x06),
        'queens': self._gen_list(N, base_seed ^ 0x07),
        'k'     : self._gen_list(N, base_seed ^ 0x08),
        'l'     : self._gen_list(N, base_seed ^ 0x09),
    }
    self.zobrist_tables[N] = tbl

  """(i,j,k,l) を 90°（時計回り）回転した signature を返す。
  - 位置写像: (r,c) -> (c, N-1-r)
  - 実装は 5bit×4 の再パックで O(1)。
  """
  def rot90(self,ijkl:int,N:int)->int: return ((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl)

  """(i,j,k,l) の 180° 回転 signature を返す（対称判定補助）。"""
  def rot180(self,ijkl:int,N:int)->int: return ((N-1-self.getj(ijkl))<<15)+((N-1-self.geti(ijkl))<<10)+((N-1-self.getl(ijkl))<<5)+(N-1-self.getk(ijkl))

  """(i,j,k,l) の 90/180/270° 回転形のいずれかが既出かを判定。
  - 既出なら True（重複排除）、未出なら False。
  - 代表集合の肥大化を初期で抑制。
  """
  def check_rotations(self,ijkl_list:Set[int],i:int,j:int,k:int,l:int,N:int)->bool: return any(rot in ijkl_list for rot in [((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i,((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+(N-1-k), (l<<15)+(k<<10)+((N-1-i)<<5)+(N-1-j)])

  """回転・ミラー対称の重複補正係数を返す。 - 90°対称: 2、180°対称: 4、それ以外: 8 """
  def symmetry(self,ijkl:int,N:int)->int: return 2 if self.symmetry90(ijkl,N) else 4 if self.geti(ijkl)==N-1-self.getj(ijkl) and self.getk(ijkl)==N-1-self.getl(ijkl) else 8

  """盤面が 90°回転で自己一致するかを判定。"""
  def symmetry90(self,ijkl:int,N:int)->bool: return ((self.geti(ijkl)<<15)+(self.getj(ijkl)<<10)+(self.getk(ijkl)<<5)+self.getl(ijkl))==(((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl))

  """(i,j,k,l) を 5bit×4 の 20bit 整数にパックして返す（signature 用）。"""
  def to_ijkl(self,i:int,j:int,k:int,l:int)->int: return (i<<15)+(j<<10)+(k<<5)+l
  """上下反転（垂直ミラー）後の signature を返す。"""
  def mirvert(self,ijkl:int,N:int)->int: return self.to_ijkl(N-1-self.geti(ijkl),N-1-self.getj(ijkl),self.getl(ijkl),self.getk(ijkl))
  """微小最適化のための min ラッパ（端から距離計算で多用）。"""
  def ffmin(self,a:int,b:int)->int: return min(a,b)

  def geti(self,ijkl:int)->int: return (ijkl>>15)&0x1F
  def getj(self,ijkl:int)->int: return (ijkl>>10)&0x1F
  def getk(self,ijkl:int)->int: return (ijkl>>5)&0x1F
  def getl(self,ijkl:int)->int: return ijkl&0x1F

  def get_jasmin(self, c: int, N: int) -> int:
    """Jasmin 正規化（代表形写像）のキャッシュ付きラッパ。
    - key=(c,N) がヒットすれば再計算を回避。
    """
    key = (c, N)
    if key in self.jasmin_cache:
        return self.jasmin_cache[key]
    result = self.jasmin(c, N)
    self.jasmin_cache[key] = result
    return result

  def jasmin(self,ijkl:int,N:int)->int:
    """(i,j,k,l) を“最も左上に近い標準形”へ写像。
    - 端から距離 `ffmin(x, N-1-x)` が最小となる軸を選び、arg 回 90°回転。
    - その後、`if self.getj(ijkl) < N-1-self.getj(ijkl): mirvert()` で上下反転。
    - 代表形の一貫性を担保し、重複圧縮に寄与。
    """
    arg=0
    min_val=self.ffmin(self.getj(ijkl),N-1-self.getj(ijkl))
    # i: 最初の行（上端） 90度回転2回
    if self.ffmin(self.geti(ijkl),N-1-self.geti(ijkl))<min_val:
      arg=2
      min_val=self.ffmin(self.geti(ijkl),N-1-self.geti(ijkl))
    # k: 最初の列（左端） 90度回転3回
    if self.ffmin(self.getk(ijkl),N-1-self.getk(ijkl))<min_val:
      arg=3
      min_val=self.ffmin(self.getk(ijkl),N-1-self.getk(ijkl))
    # l: 最後の列（右端） 90度回転1回
    if self.ffmin(self.getl(ijkl),N-1-self.getl(ijkl))<min_val:
      arg=1
      min_val=self.ffmin(self.getl(ijkl),N-1-self.getl(ijkl))
    # 90度回転を arg 回繰り返す
    for _ in range(arg):
      ijkl=self.rot90(ijkl,N)
    # 必要に応じて垂直方向のミラーリングを実行
    if self.getj(ijkl)<N-1-self.getj(ijkl):
      ijkl=self.mirvert(ijkl,N)
    return ijkl

  def file_exists(self, fname: str) -> bool:
    """ファイル存在を安全に判定（例外は False）。"""
    try:
      with open(fname, "rb"):
        return True
    except:
      return False

  def validate_constellation_list(self, constellations: List[Dict[str, int]]) -> bool:
    """TXT/BIN から復元した配列に最低限のキーが揃っているか確認。
    - 必須: "ld","rd","col","startijkl"
    """
    return all(all(k in c for k in ("ld", "rd", "col", "startijkl")) for c in constellations)

  def read_uint32_le(self, b: str) -> int:
    """LE 32bit を手組みで復元（Codon 互換: bytes ではなく str として扱う）。"""
    return (ord(b[0]) & 0xFF) | ((ord(b[1]) & 0xFF) << 8) | ((ord(b[2]) & 0xFF) << 16) | ((ord(b[3]) & 0xFF) << 24)

  def int_to_le_bytes(self,x: int) -> List[int]:
    """32bit int を LE 4バイト配列に変換（BIN 保存で使用）。"""
    return [(x >> (8 * i)) & 0xFF for i in range(4)]

  def validate_bin_file(self,fname: str) -> bool:
    """BIN のサイズ妥当性を検査（1レコード=16バイト → size % 16 == 0）。"""
    try:
      with open(fname, "rb") as f:
        f.seek(0, 2)  # ファイル末尾に移動
        size = f.tell()
      return size % 16 == 0
    except:
      return False

  def load_or_build_constellations_bin(self, ijkl_list: Set[int], constellations, N: int, preset_queens: int) -> List[Dict[str, int]]:
    """BIN キャッシュをロードし、壊れていれば生成→保存→返す。
    - `validate_bin_file()` と `validate_constellation_list()` で自己修復。
    """
    fname = f"constellations_N{N}_{preset_queens}.bin"
    if self.file_exists(fname):
      try:
        constellations = self.load_constellations_bin(fname)
        if self.validate_bin_file(fname) and self.validate_constellation_list(constellations):
          return constellations
        else:
          print(f"[警告] 不正なキャッシュ形式: {fname} を再生成します")
      except Exception as e:
        print(f"[警告] キャッシュ読み込み失敗: {fname}, 理由: {e}")
    constellations: List[Dict[str, int]] = []
    self.gen_constellations(ijkl_list, constellations, N, preset_queens)
    self.save_constellations_bin(fname, constellations)
    return constellations

  def save_constellations_txt(self, path: str, constellations: List[Dict[str, int]]) -> None:
    """コンステレーションを 1 行 5 整数（ld rd col startijkl solutions）で保存。"""
    with open(path, "w") as f:
      for c in constellations:
        ld = c["ld"]
        rd = c["rd"]
        col = c["col"]
        startijkl = c["startijkl"]
        solutions = c.get("solutions", 0)
        f.write(f"{ld} {rd} {col} {startijkl} {solutions}\n")

  def load_constellations_txt(self, path: str) -> List[Dict[str, int]]:
    """TXT から {ld,rd,col,startijkl,solutions} の配列へ復元。"""
    out: List[Dict[str, int]] = []
    with open(path, "r") as f:
      for line in f:
        parts = line.strip().split()
        if len(parts) != 5:
          continue
        ld = int(parts[0]); rd = int(parts[1]); col = int(parts[2])
        startijkl = int(parts[3]); solutions = int(parts[4])
        out.append({"ld": ld, "rd": rd, "col": col, "startijkl": startijkl, "solutions": solutions})
    return out

  def save_constellations_bin(self, fname: str, constellations: List[Dict[str, int]]) -> None:
    """BIN 形式で保存（各レコード = ld, rd, col, startijkl の LE 4B ×4）。"""
    with open(fname, "wb") as f:
      for d in constellations:
        for key in ["ld", "rd", "col", "startijkl"]:
          b = self.int_to_le_bytes(d[key])
          f.write("".join(chr(c) for c in b))  # Codonでは str がバイト文字列扱い

  def load_constellations_bin(self, fname: str) -> List[Dict[str, int]]:
    """BIN 形式をロードし、{ld,rd,col,startijkl,solutions=0} に詰め直す。"""
    constellations: List[Dict[str, int]] = []
    with open(fname, "rb") as f:
      while True:
        raw=f.read(16)
        if len(raw)<16:
          break
        ld         = self.read_uint32_le(raw[0:4])
        rd         = self.read_uint32_le(raw[4:8])
        col        = self.read_uint32_le(raw[8:12])
        startijkl  = self.read_uint32_le(raw[12:16])
        constellations.append({ "ld": ld, "rd": rd, "col": col, "startijkl": startijkl, "solutions": 0 })
    return constellations

  def load_or_build_constellations_txt(self, ijkl_list: Set[int],constellations, N: int, preset_queens: int) -> List[Dict[str, int]]:
    """TXT キャッシュをロードし、壊れていれば生成→保存→返す。
    - ロード例外やキー欠落は自動再生成で復旧。
    """
    fname = f"constellations_N{N}_{preset_queens}.txt"
    if self.file_exists(fname):
      try:
        constellations = self.load_constellations_txt(fname)
        if self.validate_constellation_list(constellations):
          return constellations
        else:
          print(f"[警告] 不正なキャッシュ形式: {fname} を再生成します")
      except Exception as e:
        print(f"[警告] キャッシュ読み込み失敗: {fname}, 理由: {e}")
    # ファイルがなければ生成・保存
    # gen_constellations() により星座を生成
    # save_constellations_txt() でファイルに保存
    # 返り値として constellations リストを返す
    constellations: List[Dict[str, int]] = []
    self.gen_constellations(ijkl_list, constellations, N, preset_queens)
    self.save_constellations_txt(fname, constellations)
    return constellations

  def set_pre_queens_cached(self, ld: int, rd: int, col: int, k: int, l: int,row: int, queens: int, LD: int, RD: int,counter:List[int], constellations: List[Dict[str, int]], N: int, preset_queens: int,visited:Set[int]) -> None:
    """サブ星座生成 `set_pre_queens()` の StateKey キャッシュ版。
    - 同一状態 key による重複呼び出しを抑制（self.subconst_cache）。
    """
    key:StateKey = (ld, rd, col, k, l, row, queens, LD, RD, N, preset_queens)
    if key in self.subconst_cache:
      # 以前に同じ状態で生成済み → 何もしない（または再利用）
      return
    # 新規実行（従来通りset_pre_queensの本体処理へ）
    self.set_pre_queens(ld, rd, col, k, l, row, queens, LD, RD, counter, constellations, N, preset_queens,visited)
    # self.subconst_cache[key] = True  # マークだけでOK
    self.subconst_cache.add(key)

  def zobrist_hash(self, ld: int, rd: int, col: int, row: int, queens: int, k: int, l: int, LD: int, RD: int, N: int) -> int:
    """Zobrist ハッシュ (64bit) を計算。
    - 先頭で **N-bit マスクを必ず適用**（`ld &= mask` など）。衝突耐性重視。
    - 立っているビット位置に対応するテーブル値を XOR し束ねる。
    """
    self._init_zobrist(N)
    tbl = self.zobrist_tables[N]
    h = 0
    mask = (1 << N) - 1
    # ★ ここが重要：Nビットに揃える（負数や上位ビットを落とす）
    ld &= mask
    rd &= mask
    col &= mask
    LD &= mask
    RD &= mask
    # 以下はそのまま
    m = ld; i = 0
    while i < N:
      if (m & 1) != 0:
        h ^= tbl['ld'][i]
      m >>= 1; i += 1
    m = rd; i = 0
    while i < N:
      if (m & 1) != 0:
        h ^= tbl['rd'][i]
      m >>= 1; i += 1
    m = col; i = 0
    while i < N:
      if (m & 1) != 0:
        h ^= tbl['col'][i]
      m >>= 1; i += 1
    m = LD; i = 0
    while i < N:
      if (m & 1) != 0:
        h ^= tbl['LD'][i]
      m >>= 1; i += 1
    m = RD; i = 0
    while i < N:
      if (m & 1) != 0:
        h ^= tbl['RD'][i]
      m >>= 1; i += 1
    if 0 <= row < N:     h ^= tbl['row'][row]
    if 0 <= queens < N:  h ^= tbl['queens'][queens]
    if 0 <= k < N:       h ^= tbl['k'][k]
    if 0 <= l < N:       h ^= tbl['l'][l]
    return h & MASK64

  def state_hash(self,ld: int, rd: int, col: int, row: int,queens:int,k:int,l:int,LD:int,RD:int,N:int) -> int:
    """O(1) の軽量ハッシュ。`visited` set のキーに用いる省メモリ・高速版。
    - `return (ld<<3) ^ (rd<<2) ^ (col<<1) ^ row ^ (queens<<7) ^ ...`
    - 厳密性 < 速度/メモリ。検証時は zobrist_hash と切替可。
    """
    # [Opt-09] Zobrist Hash（Opt-09）の導入とその用途
    # ビットボード設計でも、「盤面のハッシュ」→「探索済みフラグ」で枝刈りは可能です。
    return (ld<<3) ^ (rd<<2) ^ (col<<1) ^ row ^ (queens<<7) ^ (k<<12) ^ (l<<17) ^ (LD<<22) ^ (RD<<27) ^ (N<<1)

  def set_pre_queens(self,ld:int,rd:int,col:int,k:int,l:int,row:int,queens:int,LD:int,RD:int,counter:list,constellations:List[Dict[str,int]],N:int,preset_queens:int,visited:Set[int])->None:
    """preset_queens 個の事前配置（開始コンステレーション）を再帰列挙。
    - 初手で `h = state_hash(...)` により再訪枝を排除（`if h in visited: return`）。
    - `row==k or row==l` は必置行 → 次行へスキップ。
    - `queens==preset_queens` 到達時は signature（`(ld,rd,col,k,l,row)`）で重複を抑えつつ push。
    - 空き: `free = ~(ld|rd|col|(LD>>(N-1-row))|(RD<<(N-1-row))) & ((1<<N)-1)`
    """
    mask=(1<<N)-1  # setPreQueensで使用

    # 状態ハッシュによる探索枝の枝刈り バックトラック系の冒頭に追加　やりすぎると解が合わない
    #
    # zobrist_hash
    # 各ビットを見てテーブルから XOR するため O(N)（ld/rd/col/LD/RDそれぞれで最大 N 回）。
    # とはいえ N≤17 なのでコストは小さめ。衝突耐性は高い。
    # マスク漏れや負数の扱いを誤ると不一致が起きる点に注意（先ほどの & ((1<<N)-1) 修正で解決）。
    # h: int = self.zobrist_hash(ld, rd, col, row, queens, k, l, LD, RD, N)
    #
    # state_hash
    # その場で数個の ^ と << を混ぜるだけの O(1) 計算。
    # 生成されるキーも 単一の int なので、set/dict の操作が最速＆省メモリ。
    # ただし理論上は衝突し得ます（実際はN≤17の範囲なら実害が出にくい設計にしていればOK）。
    h: int = self.state_hash(ld, rd, col, row,queens,k,l,LD,RD,N)
    if h in visited:
      return
    visited.add(h)
    #
    # StateKey（タプル）
    # 11個の整数オブジェクトを束ねるため、オブジェクト生成・GC負荷・ハッシュ合成が最も重い。
    # set の比較・保持も重く、メモリも一番食います。
    # 衝突はほぼ心配ないものの、速度とメモリ効率は最下位。
    # key: StateKey = (ld, rd, col, row, queens, k, l, LD, RD)
    # if key in visited:
    #     return
    # visited.add(key)

    # ----------------------------
    # k行とl行はスキップ
    if row==k or row==l:
      # self.set_pre_queens(ld<<1,rd>>1,col,k,l,row+1,queens,LD,RD,counter,constellations,N,preset_queens,visited)
      self.set_pre_queens_cached(ld<<1,rd>>1,col,k,l,row+1,queens,LD,RD,counter,constellations,N,preset_queens,visited)
      return
    # クイーンの数がpreset_queensに達した場合、現在の状態を保存
    # ------------------------------------------------
    # 3. 星座のsignature重複防止
    #
    # if queens==preset_queens:
    #   constellation= {"ld": ld,"rd": rd,"col": col,"startijkl": row<<20,"solutions":0}
    #   # 新しいコンステレーションをリストに追加
    #   constellations.append(constellation)
    #   counter[0]+=1
    #   return
    if queens == preset_queens:
      # signatureの生成
      signature = (ld, rd, col, k, l, row)  # 必要な変数でOK
      # signaturesセットをクラス変数やグローバルで管理
      if not hasattr(self, "constellation_signatures"):
        self.constellation_signatures = set()
      signatures = self.constellation_signatures
      if signature not in signatures:
        constellation = {"ld": ld, "rd": rd, "col": col, "startijkl": row<<20, "solutions": 0}
        constellations.append(constellation) #星座データ追加
        signatures.add(signature)
        counter[0] += 1
      return
    # 現在の行にクイーンを配置できる位置を計算
    free=~(ld|rd|col|(LD>>(N-1-row))|(RD<<(N-1-row)))&mask
    while free:
      bit:int=free&-free
      free&=free-1
      # クイーンを配置し、次の行に進む
      # self.set_pre_queens((ld|bit)<<1,(rd|bit)>>1,col|bit,k,l,row+1,queens+1,LD,RD,counter,constellations,N,preset_queens,visited)
      self.set_pre_queens_cached((ld|bit)<<1,(rd|bit)>>1,col|bit,k,l,row+1,queens+1,LD,RD,counter,constellations,N,preset_queens,visited)

  def dfs(self,funcname:str,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    """行指向の再帰探索の総元締め（多数の SQ バリアントを funcname で切替）。
    引数:
      - ld/rd/col: 現在の遮蔽。次行で <<1 / >>1。
      - row/free: 現行行と空きビット集合（`free` は事前計算済みのときもある）。
      - jmark/endmark: j 特殊行 / 探索終端（到達で解1）。
      - mark1/mark2: 固定行（k/l の影響を受ける行）。命中時は 2～3 行スキップなどの高速化を適用。
      - board_mask/N: 盤の Nbit マスクとサイズ。
    概要:
      - `bit = avail & -avail` で最下位1ビットを取り、(ld,rd,col) を更新して次行へ。
      - その直前に `extra = _extra_block_for_row(...)` で追加遮蔽（k/l/j）を付与。
      - `_should_go_plus1(...)` で “次行で詰んでいないか” を先読みし枝刈り。
    バリアント例（funcname）:
      - "SQB": ベース版。endmark 到達で 1。
      - "SQBjrB": j 行を列0マスクで即時処理。
      - "SQBkBlBjrB" / "SQBklBjrB": k→l（あるいは隣接）命中時に 2～3 行送り。
      - "SQd0B/d1B/d2B" 系: j がコーナー～1列内側～2列内側等の距離ケース別最適化。
    """
    if funcname=="SQBkBlBjrB":
     N3:int=N-3
     blockK:int=1<<N3
     avail:int=free
     total:int=0
     while row==mark1 and avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<2
       next_rd:int=(rd|bit)>>2
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col|blockK
       next_free=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQBlBjrB",next_ld,next_rd|blockK,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQBkBlBjrB",next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       #   row_next:int=row+1
       #   extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       #   if row_next==mark1:
       #     extra |= (1<<(N-3)) #blockK
       #   if row_next == mark2:
       #     extra |= (1<<(N-3)) #blockK or blockL
       #   # jmark 系の分岐がある関数ではここでJのビットも追加する
       #   # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       #   if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
       #     total+=self.SQBkBlBjrB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQBlBjrB":
     avail:int=free
     total:int=0
     while row==mark2 and avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<2
       next_rd:int=(rd|bit)>>2
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col|1
       next_free=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQBjrB",next_ld|1,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQBlBjrB",next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       #   row_next:int=row+1
       #   extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       #   if row_next==mark1:
       #     extra |= (1<<(N-3)) #blockK
       #   if row_next == mark2:
       #     extra |= (1<<(N-3)) #blockK or blockL
       #   # jmark 系の分岐がある関数ではここでJのビットも追加する
       #   # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       #   if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
       #     total+=self.SQBlBjrB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQBjrB":
     avail:int=free
     total:int=0
     _extra_block_for_row=self._extra_block_for_row
     _should_go_plus1=self._should_go_plus1
     if row==jmark:
       avail&=~1
       ld|=1
       while avail:
         bit:int=avail&-avail
         avail&=avail-1
         next_ld:int=(ld|bit)<<1
         next_rd:int=(rd|bit)>>1
         next_col:int=col|bit
         blocked:int=next_ld|next_rd|next_col
         next_free:int=board_mask&~blocked
         # if next_free:
         #   total+=self.SQB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
         # if next_free:
         row_next:int=row+1
         # extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
         # if row_next==mark1:
         #   extra |= (1<<(N-3)) #blockK
         # if row_next == mark2:
         #   extra |= (1<<(N-3)) #blockK or blockL
         # jmark 系の分岐がある関数ではここでJのビットも追加する
         # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
         extra = _extra_block_for_row(row_next, mark1, mark2, jmark, N)
         if _should_go_plus1(next_free, row_next, endmark, next_ld, next_rd, next_col, board_mask, extra):
         # if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
           total+=self.dfs("SQB",next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       return total
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free:int=board_mask&~blocked
       # if next_free:
       #   total+=self.SQBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       row_next:int=row+1
       # extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       # if row_next==mark1:
       #   extra |= (1<<(N-3)) #blockK
       # if row_next == mark2:
       #   extra |= (1<<(N-3)) #blockK or blockL
       # jmark 系の分岐がある関数ではここでJのビットも追加する
       # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       extra = self._extra_block_for_row(row_next, mark1, mark2, jmark, N)
       if self._should_go_plus1(next_free, row_next, endmark, next_ld, next_rd, next_col, board_mask, extra):
       # if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
         total+=self.dfs("SQBjrB",next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQB":
     avail:int=free
     total:int=0
     _extra_block_for_row=self._extra_block_for_row
     _should_go_plus1=self._should_go_plus1
     if row==endmark:
       return 1
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked=next_ld|next_rd|next_col
       next_free:int=board_mask&~blocked
       # if next_free and ((row + 1 >= endmark) or self._has_future_space(next_ld, next_rd, next_col, board_mask)):
       # if next_free:
       #   total+=self.SQB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       row_next:int=row+1
       # extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       # if row_next==mark1:
       #   extra |= (1<<(N-3)) #blockK
       # if row_next == mark2:
       #   extra |= (1<<(N-3)) #blockK or blockL
       # jmark 系の分岐がある関数ではここでJのビットも追加する
       # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       extra = _extra_block_for_row(row_next, mark1, mark2, jmark, N)
       if _should_go_plus1(next_free, row_next, endmark, next_ld, next_rd, next_col, board_mask, extra):
       # if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
         total+=self.dfs("SQB",next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQBklBjrB":
     N4:int=N-4
     blockK:int=1<<N4
     avail:int=free
     total:int=0
     while row==mark1 and avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<3
       next_rd:int=(rd|bit)>>3
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col|blockK|1
       next_free:int=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQBjrB",next_ld|1,next_rd|blockK,next_col,row+3,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free:int=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQBklBjrB",next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       #   row_next:int=row+1
       #   extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       #   if row_next==mark1:
       #     extra |= (1<<(N-3)) #blockK
       #   if row_next == mark2:
       #     extra |= (1<<(N-3)) #blockK or blockL
       #   # jmark 系の分岐がある関数ではここでJのビットも追加する
       #   # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       #   if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
       #     total+=self.SQBklBjrB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQBlBkBjrB":
     avail:int=free
     total:int=0
     while row==mark1 and avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<2
       next_rd:int=(rd|bit)>>2
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col|1
       next_free:int=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQBkBjrB",next_ld|1,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free:int=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQBlBkBjrB",next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       #   row_next:int=row+1
       #   extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       #   if row_next==mark1:
       #     extra |= (1<<(N-3)) #blockK
       #   if row_next == mark2:
       #     extra |= (1<<(N-3)) #blockK or blockL
       #   # jmark 系の分岐がある関数ではここでJのビットも追加する
       #   # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       #   if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
       #     total+=self.SQBlBkBjrB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQBkBjrB":
     N3:int=N-3
     blockK:int=1<<N3
     avail:int=free
     total:int=0
     while row==mark2 and avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<2
       next_rd:int=(rd|bit)>>2
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col|blockK
       next_free:int=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQBjrB",next_ld,next_rd|blockK,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free:int=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQBkBjrB",next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       #   row_next:int=row+1
       #   extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       #   if row_next==mark1:
       #     extra |= (1<<(N-3)) #blockK
       #   if row_next == mark2:
       #     extra |= (1<<(N-3)) #blockK or blockL
       #   # jmark 系の分岐がある関数ではここでJのビットも追加する
       #   # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       #   if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
       #     total+=self.SQBkBjrB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQBlkBjrB":
     N3:int=N-3
     blockK:int=1<<N3
     avail:int=free
     total:int=0
     while row==mark1 and avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<3
       next_rd:int=(rd|bit)>>3
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col|blockK|2
       next_free:int=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQBjrB",next_ld|2,next_rd|blockK,next_col,row+3,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free:int=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQBlkBjrB",next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       #   row_next:int=row+1
       #   extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       #   if row_next==mark1:
       #     extra |= (1<<(N-3)) #blockK
       #   if row_next == mark2:
       #     extra |= (1<<(N-3)) #blockK or blockL
       #   # jmark 系の分岐がある関数ではここでJのビットも追加する
       #   # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       #   if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
       #     total+=self.SQBlkBjrB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQBjlBkBlBjrB":
     N1:int=N-1
     avail:int=free
     total:int=0
     _extra_block_for_row=self._extra_block_for_row
     _should_go_plus1=self._should_go_plus1
     if row==N1-jmark:
       rd|=1<<N1
       next_ld:int=ld<<1
       next_rd:int=rd>>1
       next_col:int=col
       blocked:int=next_ld|next_rd|next_col
       next_free=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQBkBlBjrB",next_ld,next_rd,next_col,row,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       return total
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free:int=board_mask&~blocked
       # if next_free:
       #   total+=self.SQBjlBkBlBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       row_next:int=row+1
       # extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       # if row_next==mark1:
       #   extra |= (1<<(N-3)) #blockK
       # if row_next == mark2:
       #   extra |= (1<<(N-3)) #blockK or blockL
       # jmark 系の分岐がある関数ではここでJのビットも追加する
       # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       extra = _extra_block_for_row(row_next, mark1, mark2, jmark, N)
       if _should_go_plus1(next_free, row_next, endmark, next_ld, next_rd, next_col, board_mask, extra):
       # if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
         total+=self.dfs("SQBjlBkBlBjrB",next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQBjlBklBjrB":
     N1:int=N-1
     avail:int=free
     total:int=0
     _extra_block_for_row=self._extra_block_for_row
     _should_go_plus1=self._should_go_plus1
     if row==N1-jmark:
       rd|=1<<N1
       next_ld:int=ld<<1
       next_rd:int=rd>>1
       next_col:int=col
       blocked:int=next_ld|next_rd|next_col
       next_free=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQBklBjrB",next_ld,next_rd,next_col,row,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       return total
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free:int=board_mask&~blocked
       # if next_free:
       #   total+=self.SQBjlBklBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       row_next:int=row+1
       # extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       # if row_next==mark1:
       #   extra |= (1<<(N-3)) #blockK
       # if row_next == mark2:
       #   extra |= (1<<(N-3)) #blockK or blockL
       # jmark 系の分岐がある関数ではここでJのビットも追加する
       # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       extra = _extra_block_for_row(row_next, mark1, mark2, jmark, N)
       if _should_go_plus1(next_free, row_next, endmark, next_ld, next_rd, next_col, board_mask, extra):
       # if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
         total+=self.dfs("SQBjlBklBjrB",next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQBjlBlBkBjrB":
     N1:int=N-1
     avail:int=free
     total:int=0
     _extra_block_for_row=self._extra_block_for_row
     _should_go_plus1=self._should_go_plus1
     if row==N1-jmark:
       rd|=1<<N1
       next_ld:int=ld<<1
       next_rd:int=rd>>1
       next_col:int=col
       blocked:int=next_ld|next_rd|next_col
       next_free=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQBlBkBjrB",next_ld,next_rd,next_col,row,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       return total
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free:int=board_mask&~blocked
       # if next_free:
       #   total+=self.SQBjlBlBkBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       row_next:int=row+1
       # extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       # if row_next==mark1:
       #   extra |= (1<<(N-3)) #blockK
       # if row_next == mark2:
       #   extra |= (1<<(N-3)) #blockK or blockL
       # jmark 系の分岐がある関数ではここでJのビットも追加する
       # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       extra = _extra_block_for_row(row_next, mark1, mark2, jmark, N)
       if _should_go_plus1(next_free, row_next, endmark, next_ld, next_rd, next_col, board_mask, extra):
       # if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
         total+=self.dfs("SQBjlBlBkBjrB",next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQBjlBlkBjrB":
     N1:int=N-1
     avail:int=free
     total:int=0
     _extra_block_for_row=self._extra_block_for_row
     _should_go_plus1=self._should_go_plus1
     if row==N1-jmark:
       rd|=1<<N1
       next_ld:int=ld<<1
       next_rd:int=rd>>1
       next_col:int=col
       blocked:int=next_ld|next_rd|next_col
       next_free=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQBlkBjrB",next_ld,next_rd,next_col,row,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       return total
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free:int=board_mask&~blocked
       # if next_free:
       #   total+=self.SQBjlBlkBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       row_next:int=row+1
       # extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       # if row_next==mark1:
       #   extra |= (1<<(N-3)) #blockK
       # if row_next == mark2:
       #   extra |= (1<<(N-3)) #blockK or blockL
       # jmark 系の分岐がある関数ではここでJのビットも追加する
       # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       extra = _extra_block_for_row(row_next, mark1, mark2, jmark, N)
       if _should_go_plus1(next_free, row_next, endmark, next_ld, next_rd, next_col, board_mask, extra):
       # if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
         total+=self.dfs("SQBjlBlkBjrB",next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQd2BkBlB":
     N3:int=N-3
     blockK:int=1<<N3
     avail:int=free
     total:int=0
     while row==mark1 and avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<2
       next_rd:int=(rd|bit)>>2
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col|blockK
       next_free=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd2BlB",next_ld,next_rd|blockK,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd2BkBlB",next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       #   row_next:int=row+1
       #   extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       #   if row_next==mark1:
       #     extra |= (1<<(N-3)) #blockK
       #   if row_next == mark2:
       #     extra |= (1<<(N-3)) #blockK or blockL
       #   # jmark 系の分岐がある関数ではここでJのビットも追加する
       #   # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       #   if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
       #     total+=self.SQd2BkBlB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQd2BlB":
     avail:int=free
     total:int=0
     while row==mark2 and avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<2
       next_rd:int=(rd|bit)>>2
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col|1
       next_free:int=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd2B",next_ld|1,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free:int=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd2BlB",next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       #   row_next:int=row+1
       #   extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       #   if row_next==mark1:
       #     extra |= (1<<(N-3)) #blockK
       #   if row_next == mark2:
       #     extra |= (1<<(N-3)) #blockK or blockL
       #   # jmark 系の分岐がある関数ではここでJのビットも追加する
       #   # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       #   if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
       #     total+=self.SQd2BlB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQd2B":
     avail:int=free
     total:int=0
     if row==endmark:
       if (avail&(~1))>0:
         return 1
     _extra_block_for_row=self._extra_block_for_row
     _should_go_plus1=self._should_go_plus1
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free=board_mask&~blocked
       # if next_free and ((row + 1 >= endmark) or self._has_future_space(next_ld, next_rd, next_col, board_mask)):
       # if next_free:
       #   total+=self.SQd2B(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       row_next:int=row+1
       # extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       # if row_next==mark1:
       #   extra |= (1<<(N-3)) #blockK
       # if row_next == mark2:
       #   extra |= (1<<(N-3)) #blockK or blockL
       # jmark 系の分岐がある関数ではここでJのビットも追加する
       # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       extra = _extra_block_for_row(row_next, mark1, mark2, jmark, N)
       if _should_go_plus1(next_free, row_next, endmark, next_ld, next_rd, next_col, board_mask, extra):
       # if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
         total+=self.dfs("SQd2B",next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQd2BklB":
     N4:int=N-4
     blockK:int=1<<N4
     avail:int=free
     total:int=0
     while row==mark1 and avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<3
       next_rd:int=(rd|bit)>>3
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col|blockK|1
       next_free:int=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd2B",next_ld|1,next_rd|blockK,next_col,row+3,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free:int=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd2BklB",next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       #   row_next:int=row+1
       #   extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       #   if row_next==mark1:
       #     extra |= (1<<(N-3)) #blockK
       #   if row_next == mark2:
       #     extra |= (1<<(N-3)) #blockK or blockL
       #   # jmark 系の分岐がある関数ではここでJのビットも追加する
       #   # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       #   if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
       #     total+=self.SQd2BklB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQd2BlBkB":
     avail:int=free
     total:int=0
     while row==mark1 and avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<2
       next_rd:int=(rd|bit)>>2
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col|1
       next_free:int=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd2BkB",next_ld|1,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free:int=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd2BlBkB",next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       #   row_next:int=row+1
       #   extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       #   if row_next==mark1:
       #     extra |= (1<<(N-3)) #blockK
       #   if row_next == mark2:
       #     extra |= (1<<(N-3)) #blockK or blockL
       #   # jmark 系の分岐がある関数ではここでJのビットも追加する
       #   # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       #   if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
       #     total+=self.SQd2BlBkB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQd2BkB":
     N3:int=N-3
     blockK:int=1<<N3
     avail:int=free
     total:int=0
     while row==mark2 and avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<2
       next_rd:int=(rd|bit)>>2
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col|blockK
       next_free:int=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd2B",next_ld,next_rd|blockK,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free:int=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd2BkB",next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       #   row_next:int=row+1
       #   extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       #   if row_next==mark1:
       #     extra |= (1<<(N-3)) #blockK
       #   if row_next == mark2:
       #     extra |= (1<<(N-3)) #blockK or blockL
       #   # jmark 系の分岐がある関数ではここでJのビットも追加する
       #   # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       #   if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
       #     total+=self.SQd2BkB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQd2BlkB":
     N3:int=N-3
     blockK:int=1<<N3
     avail:int=free
     total:int=0
     while row==mark1 and avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<3
       next_rd:int=(rd|bit)>>3
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col|blockK|2
       next_free=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd2B",next_ld|2,next_rd|blockK,next_col,row+3,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd2BlkB",next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       #   row_next:int=row+1
       #   extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       #   if row_next==mark1:
       #     extra |= (1<<(N-3)) #blockK
       #   if row_next == mark2:
       #     extra |= (1<<(N-3)) #blockK or blockL
       #   # jmark 系の分岐がある関数ではここでJのビットも追加する
       #   # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       #   if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
       #     total+=self.SQd2BlkB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQd1BkBlB":
     N3:int=N-3
     blockK:int=1<<N3
     avail:int=free
     total:int=0
     # _extra_block_for_row=self._extra_block_for_row
     # _should_go_plus1=self._should_go_plus1
     while row==mark1 and avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<2
       next_rd:int=(rd|bit)>>2
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col|blockK
       next_free:int=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd1BlB",next_ld,next_rd|blockK,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free:int=board_mask&~blocked
       if next_free:
       # row_next:int=row+1
       # extra = _extra_block_for_row(row_next, mark1, mark2, jmark, N)
       # if _should_go_plus1(next_free, row_next, endmark, next_ld, next_rd, next_col, board_mask, extra):
         total+=self.dfs("SQd1BkBlB",next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       #   row_next:int=row+1
       #   extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       #   if row_next==mark1:
       #     extra |= (1<<(N-3)) #blockK
       #   if row_next == mark2:
       #     extra |= (1<<(N-3)) #blockK or blockL
       #   # jmark 系の分岐がある関数ではここでJのビットも追加する
       #   # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       #   if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
       #     total+=self.SQd1BkBlB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQd1BlB":
     avail:int=free
     total:int=0
     while row==mark2 and avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<2|1
       next_rd:int=(rd|bit)>>2
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free:int=board_mask&~blocked
       # if next_free and ((row + 2 >= endmark) or self._has_future_space(next_ld, next_rd, next_col, board_mask)):
       if next_free:
         total+=self.dfs("SQd1B",next_ld,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free:int=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd1BlB",next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       #   row_next:int=row+1
       #   extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       #   if row_next==mark1:
       #     extra |= (1<<(N-3)) #blockK
       #   if row_next == mark2:
       #     extra |= (1<<(N-3)) #blockK or blockL
       #   # jmark 系の分岐がある関数ではここでJのビットも追加する
       #   # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       #   if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
       #     total+=self.SQd1BlB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQd1B":
     if row==endmark:
       return 1
     avail:int=free
     total:int=0
     _extra_block_for_row=self._extra_block_for_row
     _should_go_plus1=self._should_go_plus1
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free:int=board_mask&~blocked
       # if next_free :
       #   total+=self.SQd1B(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       # row_next:int=row+1
       # extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       # if row_next==mark1:
       #   extra |= (1<<(N-3)) #blockK
       # if row_next == mark2:
       #   extra |= (1<<(N-3)) #blockK or blockL
       # jmark 系の分岐がある関数ではここでJのビットも追加する
       # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       row_next:int=row+1
       extra = _extra_block_for_row(row_next, mark1, mark2, jmark, N)
       if _should_go_plus1(next_free, row_next, endmark, next_ld, next_rd, next_col, board_mask, extra):
       # if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
         total+=self.dfs("SQd1B",next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQd1BklB":
     N4:int=N-4
     blockK:int=1<<N4
     avail:int=free
     total:int=0
     while row==mark1 and avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<3
       next_rd:int=(rd|bit)>>3
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col|1|blockK
       next_free:int=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd1B",next_ld|1,next_rd|blockK,next_col,row+3,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free:int=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd1BklB",next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       #   row_next:int=row+1
       #   extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       #   if row_next==mark1:
       #     extra |= (1<<(N-3)) #blockK
       #   if row_next == mark2:
       #     extra |= (1<<(N-3)) #blockK or blockL
       #   # jmark 系の分岐がある関数ではここでJのビットも追加する
       #   # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       #   if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
       #     total+=self.SQd1BklB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQd1BlBkB":
     avail:int=free
     total:int=0
     while row==mark1 and avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<2
       next_rd:int=(rd|bit)>>2
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col|1
       next_free=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd1BkB",next_ld|1,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd1BlBkB",next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       #   row_next:int=row+1
       #   extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       #   if row_next==mark1:
       #     extra |= (1<<(N-3)) #blockK
       #   if row_next == mark2:
       #     extra |= (1<<(N-3)) #blockK or blockL
       #   # jmark 系の分岐がある関数ではここでJのビットも追加する
       #   # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       #   if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
       #     total+=self.SQd1BlBkB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQd1BlkB":
     N3:int=N-3
     blockK:int=1<<N3
     avail:int=free
     total:int=0
     while row==mark1 and avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<3
       next_rd:int=(rd|bit)>>3
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col|2|blockK
       next_free=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd1B",next_ld|2,next_rd|blockK,next_col,row+3,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd1BlkB",next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       #   row_next:int=row+1
       #   extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       #   if row_next==mark1:
       #     extra |= (1<<(N-3)) #blockK
       #   if row_next == mark2:
       #     extra |= (1<<(N-3)) #blockK or blockL
       #   # jmark 系の分岐がある関数ではここでJのビットも追加する
       #   # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       #   if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
       #     total+=self.SQd1BlkB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQd1BkB":
     N3:int=N-3
     blockK:int=1<<N3
     avail:int=free
     total:int=0
     while row==mark2 and avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<2
       next_rd:int=(rd|bit)>>2
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col|blockK
       next_free=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd1B",next_ld,next_rd|blockK,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd1BkB",next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       #   row_next:int=row+1
       #   extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       #   if row_next==mark1:
       #     extra |= (1<<(N-3)) #blockK
       #   if row_next == mark2:
       #     extra |= (1<<(N-3)) #blockK or blockL
       #   # jmark 系の分岐がある関数ではここでJのビットも追加する
       #   # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       #   if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
       #     total+=self.SQd1BkB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQd0B":
     if row==endmark:
       return 1
     total:int=0
     avail:int=free
     _extra_block_for_row=self._extra_block_for_row
     _should_go_plus1=self._should_go_plus1
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free:int=board_mask&~blocked
       # if next_free and ((row + 1 >= endmark) or self._has_future_space(next_ld, next_rd, next_col, board_mask)):
       # if next_free:
       #   total+=self.SQd0B(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       row_next:int=row+1
       # extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       # if row_next==mark1:
       #   extra |= (1<<(N-3)) #blockK
       # if row_next == mark2:
       #   extra |= (1<<(N-3)) #blockK or blockL
       # jmark 系の分岐がある関数ではここでJのビットも追加する
       # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       extra = _extra_block_for_row(row_next, mark1, mark2, jmark, N)
       if _should_go_plus1(next_free, row_next, endmark, next_ld, next_rd, next_col, board_mask, extra):
       # if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
         total+=self.dfs("SQd0B",next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total
    elif funcname=="SQd0BkB":
     N3:int=N-3
     blockK:int=1<<N3
     avail:int=free
     total:int=0
     while row==mark1 and avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<2
       next_rd:int=(rd|bit)>>2
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col|blockK
       next_free:int=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd0B",next_ld,next_rd|blockK,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     while avail:
       bit:int=avail&-avail
       avail&=avail-1
       next_ld:int=(ld|bit)<<1
       next_rd:int=(rd|bit)>>1
       next_col:int=col|bit
       blocked:int=next_ld|next_rd|next_col
       next_free:int=board_mask&~blocked
       if next_free:
         total+=self.dfs("SQd0BkB",next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
       # if next_free:
       #   row_next:int=row+1
       #   extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
       #   if row_next==mark1:
       #     extra |= (1<<(N-3)) #blockK
       #   if row_next == mark2:
       #     extra |= (1<<(N-3)) #blockK or blockL
       #   # jmark 系の分岐がある関数ではここでJのビットも追加する
       #   # if row_next == (N-1 - jmark): extra |= (1 << (N-1)) 等、該当関数の実装に合わせる
       #   if self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra):
       #     total+=self.SQd0BkB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
     return total

  def exec_solutions(self,constellations:List[Dict[str,int]],N:int)->None:
    """各コンステレーションに対し最適な分岐を選んで解数を計算 → symmetry で補正。
    - `startijkl` から `start` と `(i,j,k,l)` を復元。
    - `ld,rd,col` を「開始時点の遮蔽」に復元し、`free = ~(ld|rd|col)` などを設定。
    - j/k/l の関係・端からの距離で funcname を選び `dfs()` を実行。
    - `constellation["solutions"] = cnt * self.symmetry(ijkl, N)` を格納。
    """
    # jmark=j=k=l=ijkl=ld=rd=col=start_ijkl=start=free=LD=endmark=mark1=mark2=0
    N2:int=N-2
    small_mask=(1<<(N2))-1
    temp_counter=[0]
    cnt=0
    # board_mask の値が 1 ビット足りない
    # board_mask:int=(1<<(N-1))-1
    board_mask:int=(1<<N)-1
    @par
    for constellation in constellations:
      # mark1,mark2=mark1,mark2
      jmark=mark1=mark2=0
      start_ijkl=constellation["startijkl"]
      start=start_ijkl>>20
      ijkl=start_ijkl&((1<<20)-1)
      j,k,l=self.getj(ijkl),self.getk(ijkl),self.getl(ijkl)
      # 左右対角線と列の占有状況を設定
      ld,rd,col=constellation["ld"]>>1,constellation["rd"]>>1,(constellation["col"]>>1)|(~small_mask)
      LD=(1<<(N-1-j))|(1<<(N-1-l))
      ld|=LD>>(N-start)
      if start>k:
        rd|=(1<<(N-1-(start-k+1)))
      if j >= 2 * N-33-start:
        rd|=(1<<(N-1-j))<<(N2-start)
      free=~(ld|rd|col)
      # 各ケースに応じた処理
      if j<(N-3):
        jmark,endmark=j+1,N2
        if j>2 * N-34-start:
          if k<l:
            mark1,mark2=k-1,l-1
            if start<l:
              if start<k:
                if l!=k+1:
                  #cnt=self.SQBkBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
                  cnt=self.dfs("SQBkBlBjrB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
                #else: cnt=self.SQBklBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
                else: cnt=self.dfs("SQBklBjrB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              #else: cnt=self.SQBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              else: cnt=self.dfs("SQBlBjrB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
            #else: cnt=self.SQBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
            else: cnt=self.dfs("SQBjrB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
          else:
            mark1,mark2=l-1,k-1
            if start<k:
              if start<l:
                if k!=l+1:
                  #cnt=self.SQBlBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
                  cnt=self.dfs("SQBlBkBjrB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
                #else: cnt=self.SQBlkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
                else: cnt=self.dfs("SQBlkBjrB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              #else: cnt=self.SQBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              else: cnt=self.dfs("SQBkBjrB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
            #else: cnt=self.SQBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
            else: cnt=self.dfs("SQBjrB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
        else:
          if k<l:
            mark1,mark2=k-1,l-1
            if l!=k+1:
              #cnt=self.SQBjlBkBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              cnt=self.dfs("SQBjlBkBlBjrB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
            #else: cnt=self.SQBjlBklBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
            else: cnt=self.dfs("SQBjlBklBjrB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
          else:
            mark1,mark2=l-1,k-1
            if k != l+1:
              #cnt=self.SQBjlBlBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              cnt=self.dfs("SQBjlBlBkBjrB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
            #else: cnt=self.SQBjlBlkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
            else: cnt=self.dfs("SQBjlBlkBjrB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
      elif j==(N-3):
        endmark=N2
        if k<l:
          mark1,mark2=k-1,l-1
          if start<l:
            if start<k:
              #if l != k+1: cnt=self.SQd2BkBlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              if l != k+1:
                cnt=self.dfs("SQd2BkBlB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              #else: cnt=self.SQd2BklB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              else:
                cnt=self.dfs("SQd2BklB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
            else:
              mark2=l-1
              #cnt=self.SQd2BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              cnt=self.dfs("SQd2BlB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
          #else: cnt=self.SQd2B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
          else: cnt=self.dfs("SQd2B",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
        else:
          mark1,mark2=l-1,k-1
          endmark=N2
          if start<k:
            if start<l:
              if k != l+1:
                #cnt=self.SQd2BlBkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
                cnt=self.dfs("SQd2BlBkB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              #else: cnt=self.SQd2BlkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              else: cnt=self.dfs("SQd2BlkB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
            else:
              mark2=k-1
              #cnt=self.SQd2BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              cnt=self.dfs("SQd2BkB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
          #else: cnt=self.SQd2B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
          else: cnt=self.dfs("SQd2B",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
      elif j==N2: # クイーンjがコーナーからちょうど1列離れている場合
        if k<l:  # kが最初になることはない、lはクイーンの配置の関係で最後尾にはなれない
          endmark=N2
          if start<l:  # 少なくともlがまだ来ていない場合
            if start<k:  # もしkもまだ来ていないなら
              mark1=k-1
              if l != k+1:  # kとlが隣り合っている場合
                mark2=l-1
                #cnt=self.SQd1BkBlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
                cnt=self.dfs("SQd1BkBlB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              #else: cnt=self.SQd1BklB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              else: cnt=self.dfs("SQd1BklB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
            else:  # lがまだ来ていないなら
              mark2=l-1
              #cnt=self.SQd1BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              cnt=self.dfs("SQd1BlB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
          # すでにkとlが来ている場合
          #else: cnt=self.SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
          else: cnt=self.dfs("SQd1B",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
        else:  # l<k
          if start<k:  # 少なくともkがまだ来ていない場合
            if start<l:  # lがまだ来ていない場合
              if k<N2:  # kが末尾にない場合
                mark1,endmark=l-1,N2
                if k != l+1:  # lとkの間に空行がある場合
                  mark2=k-1
                  #cnt=self.SQd1BlBkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
                  cnt=self.dfs("SQd1BlBkB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
                # lとkの間に空行がない場合
                #else: cnt=self.SQd1BlkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
                else: cnt=self.dfs("SQd1BlkB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              else:  # kが末尾の場合
                if l != (N-3):  # lがkの直前でない場合
                  mark2,endmark=l-1,N-3
                  #cnt=self.SQd1BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
                  cnt=self.dfs("SQd1BlB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
                else:  # lがkの直前にある場合
                  endmark=N-4
                  #cnt=self.SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
                  cnt=self.dfs("SQd1B",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
            else:  # もしkがまだ来ていないなら
              if k != N2:  # kが末尾にない場合
                mark2,endmark=k-1,N2
                #cnt=self.SQd1BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
                cnt=self.dfs("SQd1BkB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              else:  # kが末尾の場合
                endmark=N-3
                #cnt=self.SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
                cnt=self.dfs("SQd1B",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
          else: # kとlはスタートの前
            endmark=N2
            #cnt=self.SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
            cnt=self.dfs("SQd1B",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
      else:  # クイーンjがコーナーに置かれている場合
        endmark=N2
        if start>k:
          #cnt=self.SQd0B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
          cnt=self.dfs("SQd0B",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
        else: # クイーンをコーナーに置いて星座を組み立てる方法と、ジャスミンを適用する方法
          mark1=k-1
          #cnt=self.SQd0BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
          cnt=self.dfs("SQd0BkB",ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
      # 各コンステレーションのソリューション数を更新
      # constellation["solutions"]=temp_counter[0] * self.symmetry(ijkl,N)
      constellation["solutions"]=cnt * self.symmetry(ijkl,N)
      # temp_counter[0]=0

  def gen_constellations(self,ijkl_list:Set[int],constellations:List[Dict[str,int]],N:int,preset_queens:int)->None:
    """開始コンステレーション（代表集合）を列挙 → サブ星座を生成。
    - 奇数 N は中央列を特別処理（Opt-03）。
    - 回転重複を `check_rotations()` で排除 → `jasmin()` で代表形へ。
    - 各 sc=(i,j,k,l) について `set_pre_queens_cached()` を実行。
    - 直後に生成された要素へ `startijkl |= to_ijkl(i,j,k,l)` を付与。
    """
    halfN=(N+1)//2  # Nの半分を切り上げ
    # --- [Opt-03] 中央列特別処理（奇数Nの場合のみ） ---
    if N % 2 == 1:
      center = N // 2
      ijkl_list.update(
        self.to_ijkl(i, j, center, l)
        for l in range(center + 1, N - 1)
        for i in range(center + 1, N - 1)
        if i != (N - 1) - l
        for j in range(N - center - 2, 0, -1)
        if j != i and j != l
        if not self.check_rotations(ijkl_list, i, j, center, l, N)
        # 180°回転盤面がセットに含まれていない
        # if not self.rot180_in_set(ijkl_list, i, j, center, l, N)
      )
    # --- [Opt-03] 中央列特別処理（奇数Nの場合のみ） ---
    # コーナーにクイーンがいない場合の開始コンステレーションを計算する
    ijkl_list.update(self.to_ijkl(i,j,k,l) for k in range(1,halfN) for l in range(k+1,N-1) for i in range(k+1,N-1) if i != (N-1)-l for j in range(N-k-2,0,-1) if j!=i and j!=l if not self.check_rotations(ijkl_list,i,j,k,l,N))
    # コーナーにクイーンがある場合の開始コンステレーションを計算する
    ijkl_list.update({self.to_ijkl(0,j,0,l) for j in range(1,N-2) for l in range(j+1,N-1)})
    # Jasmin変換
    # ijkl_list_jasmin = {self.jasmin(c, N) for c in ijkl_list}
    # ijkl_list_jasmin = {self.get_jasmin(c, N) for c in ijkl_list}
    # ijkl_list=ijkl_list_jasmin
    ijkl_list={self.get_jasmin(c, N) for c in ijkl_list}
    L=1<<(N-1)  # Lは左端に1を立てる
    # ローカルアクセスに変更
    geti, getj, getk, getl = self.geti, self.getj, self.getk, self.getl
    to_ijkl = self.to_ijkl
    for sc in ijkl_list:
      # ここで毎回クリア（＝この sc だけの重複抑止に限定）
      # self.constellation_signatures.clear()
      self.constellation_signatures=set()
      # i,j,k,l=self.geti(sc),self.getj(sc),self.getk(sc),self.getl(sc)
      i,j,k,l=geti(sc),getj(sc),getk(sc),getl(sc)
      # ld,rd,col=(L>>(i-1))|(1<<(N-k)),(L>>(i+1))|(1<<(l-1)),1|L|(L>>i)|(L>>j)
      # LD,RD=(L>>j)|(L>>l),(L>>j)|(1<<k)
      Lj = L >> j; Li = L >> i; Ll = L >> l
      # ld = (L >> (i-1)) | (1 << (N-k))
      ld = ((L >> (i-1)) if i > 0 else 0) | (1 << (N-k))
      rd = (L >> (i+1)) | (1 << (l-1))
      col = 1 | L | Li | Lj
      LD = Lj | Ll
      RD = Lj | (1 << k)

      counter:List[int]=[0] # サブコンステレーションを生成
      #-------------------------
      # visited:Set[StateKey]=set()
      visited:Set[int]=set()
      #-------------------------
      # self.set_pre_queens(ld,rd,col,k,l,1,3 if j==N-1 else 4,LD,RD,counter,constellations,N,preset_queens,visited)
      self.set_pre_queens_cached(ld,rd,col,k,l,1,3 if j==N-1 else 4,LD,RD,counter,constellations,N,preset_queens,visited)
      current_size=len(constellations)
      # 生成されたサブコンステレーションにスタート情報を追加
      # list(map(lambda target:target.__setitem__("startijkl",target["startijkl"]|self.to_ijkl(i,j,k,l)),(constellations[current_size-a-1] for a in range(counter[0]))))
      #
      # こちらのほうが少しだけ軽いらしい
      # for a in range(counter[0]):
      #   constellations[-1 - a]["startijkl"] |= self.to_ijkl(i, j, k, l)
      #
      # to_ijkl(i,j,k,l) はループ外で一回だけ
      # 今は毎回呼んでいるので、定数化すると少しだけ軽くなります。
      # base = self.to_ijkl(i, j, k, l)
      base = to_ijkl(i, j, k, l)
      for a in range(counter[0]):
          constellations[-1 - a]["startijkl"] |= base
  #-----------------
  # 関数プロトタイプ
  #-----------------
  @staticmethod
  def _has_future_space_step(next_ld: int, next_rd: int, next_col: int,row_next:int,endmark:int, board_mask: int, extra_block_next:int) -> bool:
    """次行 row_next で 1bit でも置ける見込みがあるかを先読み判定。
    - `row_next >= endmark` なら短絡 True。
    - `blocked_next = (next_ld<<1) | (next_rd>>1) | next_col | extra_block_next`
      を見て `board_mask & ~blocked_next != 0` を返す。
    """
    # ゴール直前は先読み不要（短絡）
    if row_next >= endmark:
        return True
    blocked_next = (next_ld << 1) | (next_rd >> 1) | next_col | extra_block_next
    return (board_mask & ~blocked_next) != 0
    #“先読み空き” を関数化します（元の式の意図に沿って、次の行での遮蔽を考慮）:
    # 次の行に進んだときに置ける可能性が1ビットでも残るか
    # return (board_mask & ~(((next_ld << 1) | (next_rd >> 1) | next_col))) != 0

  @staticmethod
  def _extra_block_for_row(row_next: int, mark1: int, mark2: int, jmark: int, N: int) -> int:
    """次行に入る“瞬間”に適用する追加遮蔽ビットを返す（k/l/j の固定効果）。
    - `blockK = 1<<(N-3)`（l も同値を流用）/ `if row_next == (N-1 - jmark): extra |= 1<<(N-1)`
    """
    extra = 0
    blockK = 1 << (N - 3)  # あなたのロジックに合わせて blockL 等も別にするなら拡張
    if row_next == mark1:
        extra |= blockK
    if row_next == mark2:
        extra |= blockK
    if row_next == (N - 1 - jmark):  # jmark 系ありの関数だけ使う
        extra |= (1 << (N - 1))
    return extra

  def _should_go_plus1( self, next_free: int, row_next: int, endmark: int, next_ld: int, next_rd: int, next_col: int, board_mask: int, extra: int,) -> bool:
    """次行へ 1 行進める価値があるかの薄いラッパ。
    - `not next_free -> False`, `row_next >= endmark -> True`、
      それ以外は `_has_future_space_step(...)` で判断。
    """
    if not next_free:
        return False
    if row_next >= endmark:
        return True
    return self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra)

class NQueens16_constellations():
  def _bit_total(self, size: int) -> int:
    """小さな N を正攻法バックトラックで全列挙（対称重み無し）。
    - 典型的な bit DP: `bitmap = mask & ~(left | down | right)` を反復。
    """
    mask = (1 << size) - 1
    total = 0

    def bt(row: int, left: int, down: int, right: int):
      nonlocal total
      if row == size:
        total += 1
        return
      bitmap = mask & ~(left | down | right)
      while bitmap:
        bit = -bitmap & bitmap
        bitmap ^= bit
        bt(row + 1, (left | bit) << 1, down | bit, (right | bit) >> 1)
    bt(0, 0, 0, 0)
    return total

  def main(self)->None:
    """エントリポイント。N=5..17 で星座列挙→解探索→集計を行う。
    - 大枠: gen_constellations → exec_solutions → sum(solutions)
    - 期待値 `expected[size]` と比較し OK/NG を表示。
    """
    nmin:int=5
    nmax:int=18
    preset_queens:int=4  # 必要に応じて変更
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax):
      start_time=datetime.now()
      if size <= 5:
        # ← フォールバック：N=5はここで正しい10を得る
        total = self._bit_total(size)
        dt = datetime.now() - start_time
        text = str(dt)[:-3]
        print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")
        continue
      ijkl_list:Set[int]=set()
      constellations:List[Dict[str,int]]=[]
      NQ=NQueens16()
      #---------------------------------
      # 星座リストそのものをキャッシュ
      #---------------------------------
      # キャッシュを使わない
      NQ.gen_constellations(ijkl_list,constellations,size,preset_queens)
      # キャッシュを使う、キャッシュの整合性もチェック
      # -- txt
      # constellations = NQ.load_or_build_constellations_txt(ijkl_list,constellations, size, preset_queens)
      # -- bin
      # constellations = NQ.load_or_build_constellations_bin(ijkl_list,constellations, size, preset_queens)
      #---------------------------------
      NQ.exec_solutions(constellations,size)
      total:int=sum(c['solutions'] for c in constellations if c['solutions']>0)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      # print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")
      expected: List[int] = [0,0,0,0,0,10,4,40,92,352,724,2680,14200,73712,365596,2279184,14772512,95815104]
      status: str = "ok" if expected[size]==total else f"ng({total}!={expected[size]})"
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}    {status}")
if __name__=="__main__":
  NQueens16_constellations().main()
