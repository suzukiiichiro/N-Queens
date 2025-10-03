#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
15Pyをマージした。
SQB..のバックトラック用関数をdfs関数に１本化した。
exec_solutionsの関数の呼び出しをdfsに変更した。

   ,     #_
   ~\_  ####_        Amazon Linux 2023
  ~~  \_#####\
  ~~     \###|
  ~~       \#/ ___   https://aws.amazon.com/linux/amazon-linux-2023
   ~~       V~' '->
    ~~~         /
      ~~._.   _/
         _/ _/
       _/m/'

amazon AWS m4.16xlarge x 1
$ codon build -release 15Py_constellations_optimize_codon.py && ./15Py_constellations_optimize_codon
コンステレーション版 キャッシュ最適化２ Ｎクイーン
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
##------------------------------------------------------------------------
# 以下は対応不要、または対応できない一般的なキャッシュ対応
##------------------------------------------------------------------------
"""
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



""""
fedora$ codon build -release 15Py_constellations_optimize_codon.py && ./15Py_constellations_optimize_codon
 N:        Total       Unique        hh:mm:ss.ms
 5:           18            0         0:00:00.005
 6:            4            0         0:00:00.000
 7:           40            0         0:00:00.002
 8:           92            0         0:00:00.002
 9:          352            0         0:00:00.001
10:          724            0         0:00:00.001
11:         2680            0         0:00:00.003
12:        14200            0         0:00:00.006
13:        73712            0         0:00:00.009
14:       365596            0         0:00:00.038
15:      2279184            0         0:00:00.092
16:     14772512            0         0:00:00.440
17:     95815104            0         0:00:02.900

fedora$ codon build -release 26Py_constellations_optimized_codon.py
fedora$ ./26Py_constellations_optimized_codon
 N:        Total       Unique        hh:mm:ss.ms
16:     14772512            0         0:00:01.503
17:     95815104            0         0:00:10.317

GPU/CUDA 11CUDA_constellation_symmetry.cu
16:         14772512               0     000:00:00:00.64
17:         95815104               0     000:00:00:03.41

"""

# import random
import pickle, os
# from operator import or_
# from functools import reduce
from typing import List,Set,Dict,Tuple
from datetime import datetime

# 64bit マスク（Zobrist用途）
MASK64: int = (1 << 64) - 1
# StateKey = Tuple[int, int, int, int, int, int, int, int, int, int, int]
StateKey = Tuple[int,int,int,int,int,int,int,int,int,int,int]
# StateKey = Tuple[int, int, int, int, int, int, int, int, int]


# pypyを使うときは以下を活かしてcodon部分をコメントアウト
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
#
class NQueens15:
  def __init__(self)->None:
    # StateKey
    # self.subconst_cache: Dict[ StateKey, bool ] = {}
    # self.subconst_cache: Dict[ Tuple[int, int, int, int, int, int, int, int, int, int, int], bool ] = {}
    self.subconst_cache: Set[StateKey] = set()
    self.constellation_signatures: Set[ Tuple[int, int, int, int, int, int] ] = set()
    self.jasmin_cache: Dict[Tuple[int, int], int] = {}
    self.zobrist_tables: Dict[int, Dict[str, List[int]]] = {}
    self.gen_cache: Dict[Tuple[int,int,int,int,int,int,int,int], List[Dict[str,int]] ] = {}

  def _mix64(self, x: int) -> int:
      # splitmix64 の最終段だけ使ったミキサ
      x &= MASK64
      x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
      x = (x ^ (x >> 27)) * 0x94D049BB133111EB & MASK64
      x ^= (x >> 31)
      return x & MASK64

  def _gen_list(self, cnt: int, seed: int) -> List[int]:
      # Zobristテーブル用の64bit値を cnt 個つくる。
      # Codonの型推論に優しいように、普通のリストで返す（ジェネレータ等は使わない）。
      out: List[int] = []
      s: int = seed & MASK64
      for _ in range(cnt):
          s = (s + 0x9E3779B97F4A7C15) & MASK64   # splitmix64 のインクリメント
          out.append(self._mix64(s))
      return out

  def _init_zobrist(self, N: int) -> None:
      # 例: self.zobrist_tables: Dict[int, Dict[str, List[int]]] を持つ前提。
      # N ごとに ['ld','rd','col','LD','RD','row','queens','k','l'] のテーブルを用意。
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
  def rot90(self,ijkl:int,N:int)->int:
    # 時計回りに90度回転
    # rot90 メソッドは、90度の右回転（時計回り）を行います
    # 元の位置 (row,col) が、回転後の位置 (col,N-1-row) になります。
    return ((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl)

  def rot180(self,ijkl:int,N:int)->int:
    # 対称性のための計算と、ijklを扱うためのヘルパー関数。
    # 開始コンステレーションが回転90に対して対称である場合
    return ((N-1-self.getj(ijkl))<<15)+((N-1-self.geti(ijkl))<<10)+((N-1-self.getl(ijkl))<<5)+(N-1-self.getk(ijkl))
  """
  # 指定した盤面 (i, j, k, l) を90度・180度・270度回転したいずれか
  # の盤面がすでにIntHashSetに存在しているかをチェックする関数
  # @param ijklList 既出盤面signature（ijkl値）の集合（HashSet）
  # @param i,j,k,l  チェック対象の盤面インデックス
  # @param N        盤面サイズ
  # @return         いずれかの回転済み盤面が登録済みなら1、なければ0
  # @details
  #   - N-Queens探索で、既存盤面の90/180/270度回転形と重複する配置
  # を高速に排除する。
  #   - 回転後のijklをそれぞれ計算し、HashSetに含まれていれば即1を
  # 返す（重複扱い）。
  #   - 真の“unique配置”のみ探索・カウントしたい場合の前処理とし
  # て必須。
  """
  def check_rotations(self,ijkl_list:Set[int],i:int,j:int,k:int,l:int,N:int)->bool:
      return any(rot in ijkl_list for rot in [((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i,((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+(N-1-k), (l<<15)+(k<<10)+((N-1-i)<<5)+(N-1-j)])
  """
  # symmetry: 回転・ミラー対称性ごとの重複補正
  # (90度:2, 180度:4, その他:8)
  """
  def symmetry(self,ijkl:int,N:int)->int:
    return 2 if self.symmetry90(ijkl,N) else 4 if self.geti(ijkl)==N-1-self.getj(ijkl) and self.getk(ijkl)==N-1-self.getl(ijkl) else 8

  def symmetry90(self,ijkl:int,N:int)->bool:
    return ((self.geti(ijkl)<<15)+(self.getj(ijkl)<<10)+(self.getk(ijkl)<<5)+self.getl(ijkl))==(((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl))
  """
  # 盤面ユーティリティ群（ビットパック式盤面インデックス変換）
  # Python実装のgeti/getj/getk/getl/toijklに対応。
  # [i, j, k, l] 各クイーンの位置情報を5ビットずつ
  # 整数値（ijkl）にパック／アンパックするためのマクロ。
  # 15ビット～0ビットまでに [i|j|k|l] を格納する設計で、
  # constellationのsignatureや回転・ミラー等の盤面操作を高速化する。
  # 例：
  #   - geti(ijkl): 上位5ビット（15-19）からiインデックスを取り出す
  #   - toijkl(i, j, k, l): 各値を5ビット単位で連結し
  # 一意な整数値（signature）に変換
  # [注意] N≦32 まで対応可能
  """
  def to_ijkl(self,i:int,j:int,k:int,l:int)->int:
    return (i<<15)+(j<<10)+(k<<5)+l

  def mirvert(self,ijkl:int,N:int)->int:
    return self.to_ijkl(N-1-self.geti(ijkl),N-1-self.getj(ijkl),self.getl(ijkl),self.getk(ijkl))

  def ffmin(self,a:int,b:int)->int:
    return min(a,b)

  def geti(self,ijkl:int)->int:
    return (ijkl>>15)&0x1F

  def getj(self,ijkl:int)->int:
    return (ijkl>>10)&0x1F

  def getk(self,ijkl:int)->int:
    return (ijkl>>5)&0x1F

  def getl(self,ijkl:int)->int:
    return ijkl&0x1F

  def get_jasmin(self, c: int, N: int) -> int:
    # 1. Jasmin変換キャッシュを導入する
    # [Opt-08] キャッシュ付き jasmin() のラッパー
    key = (c, N)
    if key in self.jasmin_cache:
        return self.jasmin_cache[key]
    result = self.jasmin(c, N)
    self.jasmin_cache[key] = result
    return result
  """
  i,j,k,lをijklに変換し、特定のエントリーを取得する関数
  各クイーンの位置を取得し、最も左上に近い位置を見つけます
  最小の値を持つクイーンを基準に回転とミラーリングを行い、配置を最も左上に近い標準形に変換します。
  最小値を持つクイーンの位置を最下行に移動させる
  i は最初の行（上端） 90度回転2回
  j は最後の行（下端） 90度回転0回
  k は最初の列（左端） 90度回転3回
  l は最後の列（右端） 90度回転1回
  優先順位が l>k>i>j の理由は？
  l は右端の列に位置するため、その位置を基準に回転させることで、配置を最も標準形に近づけることができます。
  k は左端の列に位置しますが、l ほど標準形に寄せる影響が大きくないため、次に優先されます。
  i は上端の行に位置するため、行の位置を基準にするよりも列の位置を基準にする方が配置の標準化に効果的です。
  j は下端の行に位置するため、優先順位が最も低くなります。
  """
  def jasmin(self,ijkl:int,N:int)->int:
    # 使用例:
    # ijkl_list_jasmin = {self.get_jasmin(c, N) for c in ijkl_list}
    # 最初の最小値と引数を設定
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
  #---------------------------------
  # 星座リストそのものをキャッシュ
  #---------------------------------
  def file_exists(self, fname: str) -> bool:
    try:
      with open(fname, "rb"):
        return True
    except:
      return False
  # load_or_build_constellations_txt() は、N-Queens 問題において特定
  # の盤面サイズ N と事前配置数 preset_queens に対する星座構成（部分
  # 解集合）をテキストファイルとしてキャッシュ保存し、再利用するため
  # の関数です。
  #
  # 背景：なぜキャッシュが必要？
  # 星座（コンステレーション）とは、特定のクイーンの配置から始まる
  # 枝を意味し、それぞれの星座から先を探索していくのがこのアルゴリ
  # ズムの基盤です。
  # しかしこの「星座の列挙」は、
  # 膨大な探索空間からの絞り込み
  # 対称性チェック・Jasmin判定など高コスト処理を含む
  # という特性があるため、一度生成した星座リストは保存して使い回し
  # たほうが圧倒的に効率的です。
  # --- これが Codon 向けの「ロード or 生成」関数（pickle不使用）---
  # バリデーション関数の強化（既に実装済みの場合はスキップOK）
  def validate_constellation_list(self, constellations: List[Dict[str, int]]) -> bool:
    return all(all(k in c for k in ("ld", "rd", "col", "startijkl")) for c in constellations)
  # 修正：Codon互換の from_bytes() 相当処理
  # def read_uint32_le(self, b: bytes) -> int:
  # def read_uint32_le(self, b: List[int]) -> int:
  #     return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)
  def read_uint32_le(self, b: str) -> int:
    return (ord(b[0]) & 0xFF) | ((ord(b[1]) & 0xFF) << 8) | ((ord(b[2]) & 0xFF) << 16) | ((ord(b[3]) & 0xFF) << 24)
  def int_to_le_bytes(self,x: int) -> List[int]:
    # int_to_le_bytes ヘルパー関数を定義 以下のような関数を使って int を4バイトのリトルエンディアン形式に変換できます：
    return [(x >> (8 * i)) & 0xFF for i in range(4)]
  def validate_bin_file(self,fname: str) -> bool:
    # .bin ファイルサイズチェック（1件=16バイト→行数= ilesize // 16）
    try:
      with open(fname, "rb") as f:
        f.seek(0, 2)  # ファイル末尾に移動
        size = f.tell()
      return size % 16 == 0
    except:
      return False
  def load_or_build_constellations_bin(self, ijkl_list: Set[int], constellations, N: int, preset_queens: int) -> List[Dict[str, int]]:
    # キャッシュ付きラッパー関数（.bin）
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
    # --- テキスト形式で保存（1行=5整数: ld rd col startijkl solutions）---
    with open(path, "w") as f:
      for c in constellations:
        ld = c["ld"]
        rd = c["rd"]
        col = c["col"]
        startijkl = c["startijkl"]
        solutions = c.get("solutions", 0)
        f.write(f"{ld} {rd} {col} {startijkl} {solutions}\n")
  def load_constellations_txt(self, path: str) -> List[Dict[str, int]]:
    # --- テキスト形式でロード ---
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
    # --- bin形式で保存 ---
    with open(fname, "wb") as f:
      for d in constellations:
        for key in ["ld", "rd", "col", "startijkl"]:
          b = self.int_to_le_bytes(d[key])
          f.write("".join(chr(c) for c in b))  # Codonでは str がバイト文字列扱い
  def load_constellations_bin(self, fname: str) -> List[Dict[str, int]]:
    # --- bin形式でロード ---
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
        constellations.append({
          "ld": ld, "rd": rd, "col": col,
          "startijkl": startijkl, "solutions": 0
        })
    return constellations
  def load_or_build_constellations_txt(self, ijkl_list: Set[int],constellations, N: int, preset_queens: int) -> List[Dict[str, int]]:
    # キャッシュ付きラッパー関数（.txt）
    # N と preset_queens に基づいて一意のファイル名を構成
    fname = f"constellations_N{N}_{preset_queens}.txt"
    # ファイルが存在すれば即読み込み
    # if self.file_exists(fname):
    #     return self.load_constellations_txt(fname)
    # ファイルが存在すれば読み込むが、破損チェックも行う
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
    # サブコンステレーション生成にtuple keyでキャッシュ
    # gen_constellations で set_pre_queens を呼ぶ箇所を set_pre_queens_cached に変えるだけ！
    # key = (ld, rd, col, k, l, row, queens, LD, RD, N, preset_queens)
    key:StateKey = (ld, rd, col, k, l, row, queens, LD, RD, N, preset_queens)
    if key in self.subconst_cache:
      # 以前に同じ状態で生成済み → 何もしない（または再利用）
      return
    # 新規実行（従来通りset_pre_queensの本体処理へ）
    self.set_pre_queens(ld, rd, col, k, l, row, queens, LD, RD, counter, constellations, N, preset_queens,visited)
    # self.subconst_cache[key] = True  # マークだけでOK
    self.subconst_cache.add(key)


  def zobrist_hash(self, ld: int, rd: int, col: int, row: int, queens: int, k: int, l: int, LD: int, RD: int, N: int) -> int:
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
      # [Opt-09] Zobrist Hash（Opt-09）の導入とその用途
      # ビットボード設計でも、「盤面のハッシュ」→「探索済みフラグ」で枝刈りは可能です。
      return (ld<<3) ^ (rd<<2) ^ (col<<1) ^ row ^ (queens<<7) ^ (k<<12) ^ (l<<17) ^ (LD<<22) ^ (RD<<27) ^ (N<<1)

  """
  開始コンステレーション（部分盤面）の生成関数
  N-Queens探索の初期状態を最適化するため、3つまたは4つのクイーン（presetQueens）を
  あらかじめ盤面に配置した全ての部分盤面（サブコンステレーション）を列挙・生成する。
  再帰的に呼び出され、各行ごとに可能な配置をすべて検証。

  @param ld   左対角線のビットマスク（既にクイーンがある位置は1）
  @param rd   右対角線のビットマスク
  @param col  縦方向（列）のビットマスク
  @param k    事前にクイーンを必ず置く行のインデックス1
  @param l    事前にクイーンを必ず置く行のインデックス2
  @param row  現在の再帰探索行
  @param queens 現在までに盤面に配置済みのクイーン数
  @param LD/RD 探索初期状態用のマスク（使用例次第で追記）
  @param counter 生成されたコンステレーション数を書き込むカウンタ
  @param constellations 生成したコンステレーション（部分盤面配置）のリスト
  @param N     盤面サイズ
  @details
    - row==k/lの場合は必ずクイーンを配置し次の行へ進む
    - queens==presetQueensに到達したら、現時点の盤面状態をコンステレーションとして記録
    - その他の行では、空いている位置すべてにクイーンを順次試し、再帰的に全列挙
    - 生成された部分盤面は、対称性除去・探索分割等の高速化に用いる
  """
  def set_pre_queens(self,ld:int,rd:int,col:int,k:int,l:int,row:int,queens:int,LD:int,RD:int,counter:list,constellations:List[Dict[str,int]],N:int,preset_queens:int,visited:Set[int])->None:
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
   N1:int=N-1
   N2:int=N-2
   N3:int=N-3
   N4:int=N-4
   avail:int=free
   total:int=0
   _extra_block_for_row=self._extra_block_for_row
   _should_go_plus1=self._should_go_plus1
   if funcname=="SQBkBlBjrB":
    # N3:int=N-3
    blockK:int=1<<N3
    # avail:int=free
    # total:int=0
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
    # avail:int=free
    # total:int=0
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
    # avail:int=free
    # total:int=0
    # _extra_block_for_row=self._extra_block_for_row
    # _should_go_plus1=self._should_go_plus1
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
    # avail:int=free
    # total:int=0
    # _extra_block_for_row=self._extra_block_for_row
    # _should_go_plus1=self._should_go_plus1
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
    # N4:int=N-4
    blockK:int=1<<N4
    # avail:int=free
    # total:int=0
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
    # avail:int=free
    # total:int=0
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
    # N3:int=N-3
    blockK:int=1<<N3
    # avail:int=free
    # total:int=0
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
    # N3:int=N-3
    blockK:int=1<<N3
    # avail:int=free
    # total:int=0
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
    # N1:int=N-1
    # avail:int=free
    # total:int=0
    # _extra_block_for_row=self._extra_block_for_row
    # _should_go_plus1=self._should_go_plus1
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
    # N1:int=N-1
    # avail:int=free
    # total:int=0
    # _extra_block_for_row=self._extra_block_for_row
    # _should_go_plus1=self._should_go_plus1
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
    # N1:int=N-1
    # avail:int=free
    # total:int=0
    # _extra_block_for_row=self._extra_block_for_row
    # _should_go_plus1=self._should_go_plus1
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
    # N1:int=N-1
    # avail:int=free
    # total:int=0
    # _extra_block_for_row=self._extra_block_for_row
    # _should_go_plus1=self._should_go_plus1
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
    # N3:int=N-3
    blockK:int=1<<N3
    # avail:int=free
    # total:int=0
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
    # avail:int=free
    # total:int=0
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
    # avail:int=free
    # total:int=0
    if row==endmark:
      if (avail&(~1))>0:
        return 1
    # _extra_block_for_row=self._extra_block_for_row
    # _should_go_plus1=self._should_go_plus1
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
    # N4:int=N-4
    blockK:int=1<<N4
    # avail:int=free
    # total:int=0
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
    # avail:int=free
    # total:int=0
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
    # N3:int=N-3
    blockK:int=1<<N3
    # avail:int=free
    # total:int=0
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
    # N3:int=N-3
    blockK:int=1<<N3
    # avail:int=free
    # total:int=0
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
    # N3:int=N-3
    blockK:int=1<<N3
    # avail:int=free
    # total:int=0
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
    # avail:int=free
    # total:int=0
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
    # avail:int=free
    # total:int=0
    # _extra_block_for_row=self._extra_block_for_row
    # _should_go_plus1=self._should_go_plus1
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
    # N4:int=N-4
    blockK:int=1<<N4
    # avail:int=free
    # total:int=0
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
    # avail:int=free
    # total:int=0
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
    # N3:int=N-3
    blockK:int=1<<N3
    # avail:int=free
    # total:int=0
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
    # N3:int=N-3
    blockK:int=1<<N3
    # avail:int=free
    # total:int=0
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
    # total:int=0
    # avail:int=free
    # _extra_block_for_row=self._extra_block_for_row
    # _should_go_plus1=self._should_go_plus1
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
    # N3:int=N-3
    blockK:int=1<<N3
    # avail:int=free
    # total:int=0
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










  """
  ConstellationArrayListの各Constellation（部分盤面）ごとに
  N-Queens探索を分岐し、そのユニーク解数をsolutionsフィールドに記録する関数（CPU版）
  @param constellations 解探索対象のConstellationArrayListポインタ
  @param N              盤面サイズ
  @details
    - 各Constellation（部分盤面）ごとにj, k, l, 各マスク値を展開し、
      複雑な分岐で最適な再帰ソルバー（SQ...関数群）を呼び出して解数を計算
    - 分岐ロジックは、部分盤面・クイーンの位置・コーナーからの距離などで高速化
    - 解数はtemp_counterに集約し、各Constellationのsolutionsフィールドに記録
    - symmetry(ijkl, N)で回転・ミラー重複解を補正
    - GPUバージョン(execSolutionsKernel)のCPU移植版（デバッグ・逐次確認にも活用）
  @note
    - N-Queens最適化アルゴリズムの核心部
    - temp_counterは再帰呼び出しで合計を受け渡し
    - 実運用時は、より多くの分岐パターンを組み合わせることで最大速度を発揮
  """
  def exec_solutions(self,constellations:List[Dict[str,int]],N:int)->None:
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
  """
  開始コンステレーション（部分盤面配置パターン）の列挙・重複排除を行う関数
  @param ijklList        uniqueな部分盤面signature（ijkl値）の格納先HashSet
  @param constellations  Constellation本体リスト（実際の盤面は後続で生成）
  @param N               盤面サイズ
  @details
    - コーナー・エッジ・対角・回転対称性を考慮し、「代表解」となるuniqueな開始盤面のみ抽出する。
    - forループの入れ子により、N-Queens盤面の「最小単位部分盤面」を厳密な順序で列挙。
    - k, l, i, j 各インデックスの取り方・範囲・重複排除のための判定ロジックが最適化されている。
    - checkRotations()で既出盤面（回転対称）を排除、必要なものだけをijklListに追加。
    - このunique setをもとに、後段でConstellation構造体の生成・分割探索を展開可能。
  @note
    - 「部分盤面分割＋代表解のみ探索」戦略は大規模Nの高速化の要！
    - このループ構造・排除ロジックがN-Queensソルバの根幹。
  """
  def gen_constellations(self,ijkl_list:Set[int],constellations:List[Dict[str,int]],N:int,preset_queens:int)->None:
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
  def _has_future_space_step(next_ld: int, next_rd: int, next_col: int,
                        row_next:int,endmark:int,
                        board_mask: int,
                        extra_block_next:int # 次の行で実際にORされる追加ブロック（なければ0）
                        ) -> bool:
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
      if not next_free:
          return False
      if row_next >= endmark:
          return True
      return self._has_future_space_step(next_ld, next_rd, next_col, row_next, endmark, board_mask, extra)

"""

  def SQd0B(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQd0B(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQd0BkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQd0B(next_ld,next_rd|blockK,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    while avail:
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQd0BkB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
  def SQd1BklB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQd1B(next_ld|1,next_rd|blockK,next_col,row+3,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    while avail:
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQd1BklB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
  def SQd1B(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQd1B(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQd1BkBlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQd1BlB(next_ld,next_rd|blockK,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
        total+=self.SQd1BkBlB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
  def SQd1BlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQd1B(next_ld,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    while avail:
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQd1BlB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
  def SQd1BlkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQd1B(next_ld|2,next_rd|blockK,next_col,row+3,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    while avail:
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQd1BlkB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
  def SQd1BlBkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQd1BkB(next_ld|1,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    while avail:
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQd1BlBkB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
  def SQd1BkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQd1B(next_ld,next_rd|blockK,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    while avail:
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQd1BkB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
  def SQd2BlkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQd2B(next_ld|2,next_rd|blockK,next_col,row+3,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    while avail:
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQd2BlkB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
  def SQd2BklB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQd2B(next_ld|1,next_rd|blockK,next_col,row+3,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    while avail:
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQd2BklB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
  def SQd2BkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQd2B(next_ld,next_rd|blockK,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    while avail:
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQd2BkB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
  def SQd2BlBkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQd2BkB(next_ld|1,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    while avail:
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQd2BlBkB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
  def SQd2BlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQd2B(next_ld|1,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    while avail:
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQd2BlB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
  def SQd2BkBlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQd2BlB(next_ld,next_rd|blockK,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    while avail:
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQd2BkBlB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
  def SQd2B(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQd2B(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQBlBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQBjrB(next_ld|1,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    while avail:
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQBlBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
  def SQBkBlBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQBlBjrB(next_ld,next_rd|blockK,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    while avail:
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQBkBlBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
  def SQBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
          total+=self.SQB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
        total+=self.SQBjrB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQBlBkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQBkBjrB(next_ld|1,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    while avail:
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQBlBkBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
  def SQBkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQBjrB(next_ld,next_rd|blockK,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    while avail:
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQBkBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
  def SQBklBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQBjrB(next_ld|1,next_rd|blockK,next_col,row+3,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    while avail:
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQBklBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
  def SQBlkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQBjrB(next_ld|2,next_rd|blockK,next_col,row+3,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    while avail:
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQBlkBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
  def SQBjlBkBlBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQBkBlBjrB(next_ld,next_rd,next_col,row,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
        total+=self.SQBjlBkBlBjrB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQBjlBlBkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQBlBkBjrB(next_ld,next_rd,next_col,row,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
        total+=self.SQBjlBlBkBjrB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQBjlBklBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQBklBjrB(next_ld,next_rd,next_col,row,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
        total+=self.SQBjlBklBjrB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQBjlBlkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
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
        total+=self.SQBlkBjrB(next_ld,next_rd,next_col,row,next_free,jmark,endmark,mark1,mark2,board_mask,N)
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
        total+=self.SQBjlBlkBjrB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
"""    
class NQueens15_constellations():
  def _bit_total(self, size: int) -> int:
    # 小さなNは正攻法で数える（対称重みなし・全列挙）
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
    nmin:int=5
    nmax:int=28
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
      NQ=NQueens15()
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
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")
if __name__=="__main__":
  NQueens15_constellations().main()
