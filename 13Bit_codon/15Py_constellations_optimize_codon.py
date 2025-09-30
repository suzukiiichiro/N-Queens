#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
コンステレーション版 キャッシュ最適化２ Ｎクイーン

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

"""
1. 関数内の最適化
def SQBjlBklBjrB(self, ld:int, rd:int, col:int, row:int, free:int,jmark:int, endmark:int, mark1:int, mark2:int, N:int) -> int:
    N1:int = N - 1
    # ★ 追加：内側N-2列のマスク（コーナー除去前提）
    board_mask:int = (1 << (N - 2)) - 1
    avail = free
    total = 0
    if row == N1 - jmark:
        rd |= 1 << N1
        # avail の列は内側N-2列しか持たないので、1<<N1 は範囲外 → 下の AND で自然に落ちます
        # avail &= ~(1 << N1)  # ← 実質 no-op なので不要
        # ここも ~ の後に board_mask を適用
        next_free = board_mask&~((ld << 1) | (rd >> 1) | col)
        if next_free:
            total += self.SQBklBjrB(ld, rd, col, row, free, jmark, endmark, mark1, mark2, N)
        return total
    while avail:
        bit:int = avail & -avail
        avail &= avail - 1
        # ここも ~ の後に board_mask を適用
        next_free:int = board_mask&~(
            ((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit))
        if next_free:
            total += self.SQBjlBklBjrB(
                (ld | bit) << 1, (rd | bit) >> 1, col | bit,
                row + 1, next_free, jmark, endmark, mark1, mark2, N
            )
    return total

補足（重要）
avail &= ~(1 << N1) は実質 no-op
avail は「内側 N-2 列」のビット集合、1 << (N-1) はその範囲外です。
ここで列を潰したい意図なら、内側インデックス系でビット位置を計算してください（例：左端を 0、右端を N-3 とするなど）。
ただし、board_mask を使っている限り、範囲外ビットは自然に落ちるため、通常はこの行は不要です。

もし「全 N 列」を使う設計なら
board_mask = (1 << N) - 1 を使い、コーナー列は col 側で事前に埋める（あなたの exec_solutions で既に col |= ~small_mask している方式）に統一してください。いずれにせよ next_free = board_mask&~(...) の形を守るのが肝です。


2.すべての SQ* の関数内で定義されている board_mask:int=(1<<(N-2))-1 を exec_solutions() で一度だけ定義してすべての SQ* にパラメータで渡す
3.重要：free ではなく next_free を渡す
行 1083 で次の関数へ渡しているのが free になっていますが、直前で rd を更新し、next_free を計算しています。
ここは free ではなく next_free を渡すべきです。でないと、更新後の占有状態が反映されません。

- total+=self.SQBlkBjrB(ld,rd,col,row,next_free,jmark,endmark,mark1,mark2,board_mask,N)
+ total+=self.SQBlkBjrB(ld,rd,col,row,next_free,jmark,endmark,mark1,mark2,board_mask,N)

4.一時変数を使って再計算を行わない
next_ld,next_rd,next_col = (ld|bit)<<1,(rd|bit)>>1,col|bit
next_free = board_mask & ~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)) [& ((1<<N)-1)]
if next_free and (row+1>=endmark or ~((next_ld<<1)|(next_rd>>1)|next_col)>0):
      total += self.SQd1B(next_ld,next_rd,next_col,row+1,next_free,...)
↓
blocked:int=next_ld|next_rd|next_col
next_free = board_mask & ~blocked
if next_free and (row + 1 >= endmark or (board_mask &~blocked)):
      total += self.SQd1B(next_ld,next_rd,next_col, row + 1, next_free, ...)

"""

"""
1. 並列とキャッシュの整合
@par は exec_solutions の星座単位で独立になっているので、インスタンス属性のキャッシュは 生成段階（gen_constellations）で完結しており競合しません。jasmin_cache・subconst_cache を インスタンス属性にしたのは正解。

2. 180°重複チェックの二重化
check_rotations() は 90/180/270°すべて見ていますが、奇数 N の中央列ブロックで check_rotations(...) と rot180_in_set(...) を両方呼んでいますね。ここは rot180 が 重複なので、check_rotations(...) のみでOK（微小ですが内包表記が軽くなります）。

3. visited の粒度
visited を星座ごとに新規 set() にしているので、メモリ爆発を回避できています。ハッシュに ld, rd, col, row, queens, k, l, LD, RD, N まで混ぜているのも衝突耐性◯。

4. “先読み空き” の条件
先読み関数 _has_future_space() を使った
if next_free and ((row >= endmark-1) or _has_future_space(...)):
の形は、**「ゴール直前は先読み不要」**という意図に合っていて良い感じ。境界で row+1 >= endmark か row >= endmark-1 を使い分けている箇所も一貫しています。

5. Tuple の import 確認
ファイル先頭で from typing import List, Set, Dict になっていました。__init__ の型注釈で Tuple[...] を使っているので、Tuple も import しておくと Codon の型検査で安全です：
from typing import List, Set, Dict, Tuple
（すでにビルドが通っているならOKですが、保守のために念のため。）

すでにビット演算のインライン化・board_mask の上位での共有・**1ビット抽出 bit = x & -x**など、要所は押さえられています。cnt を星座ごとにローカルで完結→solutions に掛け算（symmetry()）という流れもキャッシュに優しい設計。
これ以上を狙うなら、「星座ごと分割の並列度を広げる」か「gen_constellations の ijkl_list.update(...) での回転重複除去を最小限に（=set操作の負荷を減らす）」の二択ですが、現状の速度を見る限り十分実用的です。

"""


""""
fedora$ codon build -release 14Py_constellations_par_codon.py && ./14Py_constellations_par_codon
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
StateKey = Tuple[int, int, int, int, int, int, int, int, int, int, int]


# pypyを使うときは以下を活かしてcodon部分をコメントアウト
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
#
class NQueens15:
  def __init__(self)->None:
    # StateKey
    # self.subconst_cache: Dict[ StateKey, bool ] = {}
    self.subconst_cache: Dict[ Tuple[int, int, int, int, int, int, int, int, int, int, int], bool ] = {}
    self.constellation_signatures: Set[ Tuple[int, int, int, int, int, int] ] = set()
    self.jasmin_cache: Dict[Tuple[int, int], int] = {}
    # 追加：Zobrist テーブルキャッシュ（N ごと）
    self.zobrist_tables: Dict[int, Dict[str, List[int]]] = {}
    self.gen_cache: Dict[Tuple[int,int,int,int,int,int,int,int], List[Dict[str,int]] ] = {}

  # === NQueens15 クラス内に追加 ===
  def _init_zobrist(self, N: int) -> None:
      # ここで hasattr チェックは不要
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

  def _mix64(self, x: int) -> int:
      # splitmix64 の最終段だけ使ったミキサ
      x &= MASK64
      x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
      x = (x ^ (x >> 27)) * 0x94D049BB133111EB & MASK64
      x ^= (x >> 31)
      return x & MASK64

  def _gen_list(self, cnt: int, seed: int) -> List[int]:
      """
      Zobristテーブル用の64bit値を cnt 個つくる。
      Codonの型推論に優しいように、普通のリストで返す（ジェネレータ等は使わない）。
      """
      out: List[int] = []
      s: int = seed & MASK64
      for _ in range(cnt):
          s = (s + 0x9E3779B97F4A7C15) & MASK64   # splitmix64 のインクリメント
          out.append(self._mix64(s))
      return out

  def _init_zobrist(self, N: int) -> None:
      """
      例: self.zobrist_tables: Dict[int, Dict[str, List[int]]] を持つ前提。
      N ごとに ['ld','rd','col','LD','RD','row','queens','k','l'] のテーブルを用意。
      """
      # if not hasattr(self, "zobrist_tables"):
      #     self.zobrist_tables: Dict[int, Dict[str, List[int]]] = {}

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
      return ((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl)

  def rot180(self,ijkl:int,N:int)->int:
      return ((N-1-self.getj(ijkl))<<15)+((N-1-self.geti(ijkl))<<10)+((N-1-self.getl(ijkl))<<5)+(N-1-self.getk(ijkl))

  def rot180_in_set(self,ijkl_list:Set[int],i:int,j:int,k:int,l:int,N:int)->bool:
      return self.rot180(self.to_ijkl(i, j, k, l), N) in ijkl_list

  def check_rotations(self,ijkl_list:Set[int],i:int,j:int,k:int,l:int,N:int)->bool:
      return any(rot in ijkl_list for rot in [((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i,((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+(N-1-k), (l<<15)+(k<<10)+((N-1-i)<<5)+(N-1-j)])
    # rot90=((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i
    # rot180=((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+(N-1-k)
    # rot270=(l<<15)+(k<<10)+((N-1-i)<<5)+(N-1-j)
    # return any(rot in ijkl_list for rot in (rot90,rot180,rot270))

  def symmetry(self,ijkl:int,N:int)->int:
    return 2 if self.symmetry90(ijkl,N) else 4 if self.geti(ijkl)==N-1-self.getj(ijkl) and self.getk(ijkl)==N-1-self.getl(ijkl) else 8
  def symmetry90(self,ijkl:int,N:int)->bool:
    return ((self.geti(ijkl)<<15)+(self.getj(ijkl)<<10)+(self.getk(ijkl)<<5)+self.getl(ijkl))==(((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl))
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
  #--------------------------------------------
  def get_jasmin(self, c: int, N: int) -> int:
    key = (c, N)
    if key in self.jasmin_cache:
        return self.jasmin_cache[key]
    result = self.jasmin(c, N)
    self.jasmin_cache[key] = result
    return result
  #--------------------------------------------
  # 使用例:
  # ijkl_list_jasmin = {self.get_jasmin(c, N) for c in ijkl_list}
  #--------------------------------------------
  def jasmin(self,ijkl:int,N:int)->int:
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
  # codon では動かない
  #
  def file_exists(self,fname:str)->bool:
    # try:
    #     os.stat(fname)
    #     return True
    # except OSError:
    #     return False
    try:
      with open(fname, "rb"):
        pass
        return True
    except:
      return False
  def load_constellations(self,N:int,preset_queens:int)->list:
    fname = f"constellations_N{N}_{preset_queens}.pkl"
    if self.file_exists(fname):
        with open(fname, "rb") as f:
            return pickle.load(f)
    else:
        constellations = []
        self.gen_constellations(set(),constellations,N,preset_queens)
        with open(fname, "wb") as f:
            pickle.dump(constellations, f)
        return constellations
  # 実行時
  # main()
  #--------------------------
  # codon では動かないので以下を切り替える
  # pickleの最適化は使わない（あきらめる）
  # NQ.gen_constellations(ijkl_list,constellations,size,preset_queens)
  # codonでpickleを使う（うごかない）
  # constellations = NQ.load_constellations(size,preset_queens)
  #---------------------------------
  # def set_pre_queens_cached(self, ld: int, rd: int, col: int, k: int, l: int,row: int, queens: int, LD: int, RD: int,counter: list, constellations: List[Dict[str, int]], N: int, preset_queens: int,visited:set[StateKey]) -> None:
  def set_pre_queens_cached(self, ld: int, rd: int, col: int, k: int, l: int,row: int, queens: int, LD: int, RD: int,counter: list, constellations: List[Dict[str, int]], N: int, preset_queens: int,visited:Set[int]) -> None:
  #    key = (ld, rd, col, k, l, row, queens, LD, RD, N, preset_queens)
  #    # キャッシュの本体をdictかsetでグローバル/クラス変数に
  #    if not hasattr(self, "subconst_cache"):
  #        self.subconst_cache = {}
  #    subconst_cache = self.subconst_cache
  #    if key in subconst_cache:
  #        # 以前に同じ状態で生成済み → 何もしない（または再利用）
  #        return
  #    # 新規実行（従来通りset_pre_queensの本体処理へ）
  #    self.set_pre_queens(ld, rd, col, k, l, row, queens, LD, RD, counter, constellations, N, preset_queens,visited)
  #    subconst_cache[key] = True  # マークだけでOK
      key = (ld, rd, col, k, l, row, queens, LD, RD, N, preset_queens)
      if key in self.subconst_cache:
          return
      self.set_pre_queens(ld, rd, col, k, l, row, queens, LD, RD,
                          counter, constellations, N, preset_queens, visited)
      self.subconst_cache[key] = True
  # 呼び出し側
  # self.set_pre_queens_cached(...) とする
  #---------------------------------
  #“先読み空き” を関数化します（元の式の意図に沿って、次の行での遮蔽を考慮）:
  @staticmethod
  def _has_future_space(next_ld: int, next_rd: int, next_col: int, board_mask: int) -> bool:
      # 次の行に進んだときに置ける可能性が1ビットでも残るか
      return (board_mask & ~(((next_ld << 1) | (next_rd >> 1) | next_col))) != 0

  def zobrist_hash(self, ld: int, rd: int, col: int,
                  row: int, queens: int, k: int, l: int,
                  LD: int, RD: int, N: int) -> int:
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
      return (ld<<3) ^ (rd<<2) ^ (col<<1) ^ row ^ (queens<<7) ^ (k<<12) ^ (l<<17) ^ (LD<<22) ^ (RD<<27) ^ (N<<1)

  # def set_pre_queens(self,ld:int,rd:int,col:int,k:int,l:int,row:int,queens:int,LD:int,RD:int,counter:list,constellations:List[Dict[str,int]],N:int,preset_queens:int,visited:Set[StateKey])->None:
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
    # key: StateKey = (ld, rd, col, row, queens, k, l, LD, RD, N, preset_queens)
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
    # ------------------------------------------------

    # 現在の行にクイーンを配置できる位置を計算
    free=~(ld|rd|col|(LD>>(N-1-row))|(RD<<(N-1-row)))&mask
    while free:
      bit:int=free&-free  # 最も下位の1ビットを取得
      free&=free-1  # 使用済みビットを削除
      # クイーンを配置し、次の行に進む
      # self.set_pre_queens((ld|bit)<<1,(rd|bit)>>1,col|bit,k,l,row+1,queens+1,LD,RD,counter,constellations,N,preset_queens,visited)
      self.set_pre_queens_cached((ld|bit)<<1,(rd|bit)>>1,col|bit,k,l,row+1,queens+1,LD,RD,counter,constellations,N,preset_queens,visited)
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
                  cnt=self.SQBkBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
                else: cnt=self.SQBklBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              else: cnt=self.SQBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
            else: cnt=self.SQBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
          else:
            mark1,mark2=l-1,k-1
            if start<k:
              if start<l:
                if k!=l+1:
                  cnt=self.SQBlBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
                else: cnt=self.SQBlkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              else: cnt=self.SQBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
            else: cnt=self.SQBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
        else:
          if k<l:
            mark1,mark2=k-1,l-1
            if l!=k+1:
              cnt=self.SQBjlBkBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
            else: cnt=self.SQBjlBklBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
          else:
            mark1,mark2=l-1,k-1
            if k != l+1:
              cnt=self.SQBjlBlBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
            else: cnt=self.SQBjlBlkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
      elif j==(N-3):
        endmark=N2
        if k<l:
          mark1,mark2=k-1,l-1
          if start<l:
            if start<k:
              if l != k+1: cnt=self.SQd2BkBlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              else: cnt=self.SQd2BklB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
            else:
              mark2=l-1
              cnt=self.SQd2BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
          else: cnt=self.SQd2B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
        else:
          mark1,mark2=l-1,k-1
          endmark=N2
          if start<k:
            if start<l:
              if k != l+1:
                cnt=self.SQd2BlBkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              else: cnt=self.SQd2BlkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
            else:
              mark2=k-1
              cnt=self.SQd2BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
          else: cnt=self.SQd2B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
      elif j==N2: # クイーンjがコーナーからちょうど1列離れている場合
        if k<l:  # kが最初になることはない、lはクイーンの配置の関係で最後尾にはなれない
          endmark=N2
          if start<l:  # 少なくともlがまだ来ていない場合
            if start<k:  # もしkもまだ来ていないなら
              mark1=k-1
              if l != k+1:  # kとlが隣り合っている場合
                mark2=l-1
                cnt=self.SQd1BkBlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              else: cnt=self.SQd1BklB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
            else:  # lがまだ来ていないなら
              mark2=l-1
              cnt=self.SQd1BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
          # すでにkとlが来ている場合
          else: cnt=self.SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
        else:  # l<k
          if start<k:  # 少なくともkがまだ来ていない場合
            if start<l:  # lがまだ来ていない場合
              if k<N2:  # kが末尾にない場合
                mark1,endmark=l-1,N2
                if k != l+1:  # lとkの間に空行がある場合
                  mark2=k-1
                  cnt=self.SQd1BlBkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
                # lとkの間に空行がない場合
                else: cnt=self.SQd1BlkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              else:  # kが末尾の場合
                if l != (N-3):  # lがkの直前でない場合
                  mark2,endmark=l-1,N-3
                  cnt=self.SQd1BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
                else:  # lがkの直前にある場合
                  endmark=N-4
                  cnt=self.SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
            else:  # もしkがまだ来ていないなら
              if k != N2:  # kが末尾にない場合
                mark2,endmark=k-1,N2
                cnt=self.SQd1BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
              else:  # kが末尾の場合
                endmark=N-3
                cnt=self.SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
          else: # kとlはスタートの前
            endmark=N2
            cnt=self.SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
      else:  # クイーンjがコーナーに置かれている場合
        endmark=N2
        if start>k:
          cnt=self.SQd0B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
        else: # クイーンをコーナーに置いて星座を組み立てる方法と、ジャスミンを適用する方法
          mark1=k-1
          cnt=self.SQd0BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,N)
      # 各コンステレーションのソリューション数を更新
      # constellation["solutions"]=temp_counter[0] * self.symmetry(ijkl,N)
      constellation["solutions"]=cnt * self.symmetry(ijkl,N)
      # temp_counter[0]=0
  #---------------------------------
  # 星座リストそのものをキャッシュ
  #---------------------------------
  def file_exists(self, fname: str) -> bool:
    try:
      with open(fname, "rb"):
        return True
    except:
      return False
  # --- テキスト形式で保存（1行=5整数: ld rd col startijkl solutions）---
  def save_constellations_txt(self, path: str, constellations: List[Dict[str, int]]) -> None:
    with open(path, "w") as f:
      for c in constellations:
        ld = c["ld"]
        rd = c["rd"]
        col = c["col"]
        startijkl = c["startijkl"]
        solutions = c.get("solutions", 0)
        f.write(f"{ld} {rd} {col} {startijkl} {solutions}\n")
  # --- テキスト形式でロード ---
  def load_constellations_txt(self, path: str) -> List[Dict[str, int]]:
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
  # int_to_le_bytes ヘルパー関数を定義 以下のような関数を使って int を4バイトのリトルエンディアン形式に変換できます：
  def int_to_le_bytes(self,x: int) -> List[int]:
    return [(x >> (8 * i)) & 0xFF for i in range(4)]
  # 書き込み関数（.bin保存）
  def save_constellations_bin(self, fname: str, constellations: List[Dict[str, int]]) -> None:
    with open(fname, "wb") as f:
      for d in constellations:
        for key in ["ld", "rd", "col", "startijkl"]:
          b = self.int_to_le_bytes(d[key])
          f.write("".join(chr(c) for c in b))  # Codonでは str がバイト文字列扱い
  # 読み込み関数（.binロード）
  def load_constellations_bin(self, fname: str) -> List[Dict[str, int]]:
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
  # .bin ファイルサイズチェック（1件=16バイト→行数= ilesize // 16）
  def validate_bin_file(self,fname: str) -> bool:
   try:
     with open(fname, "rb") as f:
       f.seek(0, 2)  # ファイル末尾に移動
       size = f.tell()
     return size % 16 == 0
   except:
     return False
  # キャッシュ付きラッパー関数（.bin）
  def load_or_build_constellations_bin(self, ijkl_list: Set[int], constellations, N: int, preset_queens: int) -> List[Dict[str, int]]:
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
  # キャッシュ付きラッパー関数（.txt）
  def load_or_build_constellations_txt(self, ijkl_list: Set[int],constellations, N: int, preset_queens: int) -> List[Dict[str, int]]:
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
  #-------------------------
  # 2. サブコンステレーション生成にtuple keyでキャッシュ
  # gen_constellations で set_pre_queens を呼ぶ箇所を set_pre_queens_cached に変えるだけ！
  #---------------------------------
  # def set_pre_queens_cached(self, ld: int, rd: int, col: int, k: int, l: int,row: int, queens: int, LD: int, RD: int,counter: list, constellations: List[Dict[str, int]], N: int, preset_queens: int,visited:Set[StateKey]) -> None:
  def set_pre_queens_cached(self, ld: int, rd: int, col: int, k: int, l: int,row: int, queens: int, LD: int, RD: int,counter: list, constellations: List[Dict[str, int]], N: int, preset_queens: int,visited:set[int]) -> None:
    # key = (ld, rd, col, k, l, row, queens, LD, RD, N, preset_queens)
    key:StateKey = (ld, rd, col, k, l, row, queens, LD, RD, N, preset_queens)
    # キャッシュの本体をdictかsetでグローバル/クラス変数に
    # if not hasattr(self, "subconst_cache"):
    #   self.subconst_cache = {}
    # subconst_cache = self.subconst_cache
    if key in self.subconst_cache:
      # 以前に同じ状態で生成済み → 何もしない（または再利用）
      return
    # 新規実行（従来通りset_pre_queensの本体処理へ）
    self.set_pre_queens(ld, rd, col, k, l, row, queens, LD, RD, counter, constellations, N, preset_queens,visited)
    self.subconst_cache[key] = True  # マークだけでOK

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
        if not self.rot180_in_set(ijkl_list, i, j, center, l, N)
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
    for sc in ijkl_list:
      i,j,k,l=self.geti(sc),self.getj(sc),self.getk(sc),self.getl(sc)
      ld,rd,col=(L>>(i-1))|(1<<(N-k)),(L>>(i+1))|(1<<(l-1)),1|L|(L>>i)|(L>>j)
      LD,RD=(L>>j)|(L>>l),(L>>j)|(1<<k)
      counter=[0] # サブコンステレーションを生成
      #-------------------------
      # visited:Set[StateKey]=set()
      visited:Set[int]=set()
      #-------------------------
      # self.set_pre_queens(ld,rd,col,k,l,1,3 if j==N-1 else 4,LD,RD,counter,constellations,N,preset_queens,visited)
      self.set_pre_queens_cached(ld,rd,col,k,l,1,3 if j==N-1 else 4,LD,RD,counter,constellations,N,preset_queens,visited)
      current_size=len(constellations)
      # 生成されたサブコンステレーションにスタート情報を追加
      list(map(lambda target:target.__setitem__("startijkl",target["startijkl"]|self.to_ijkl(i,j,k,l)),(constellations[current_size-a-1] for a in range(counter[0]))))

  #-----------------
  # 関数プロトタイプ
  #-----------------
  def SQd0B(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    if row==endmark:
      # tempcounter[0]+=1
      return 1
    total:int=0
    avail:int=free
    while avail:
    # while free:
      # bit:int=free&-free  # 最下位ビットを取得
      bit:int=avail&-avail  # 最下位ビットを取得
      # free&=free-1  # 使用済みビットを削除
      avail&=avail-1  # 使用済みビットを削除
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      # NG
      # if next_free and (row>=endmark-1 or ~blocked):
      # OK（空きが1つでもあるか）
      # 1 行先を見るケース：
      # 右辺が非ゼロなら next_free が False でも通過してしまいます。
      # if next_free and (row >= endmark - 1) or ((~blocked) & board_mask):
      # 1 行先を見るケース：
      if next_free and ((row >= endmark - 1) or self._has_future_space(next_ld, next_rd, next_col, board_mask)):
        total+=self.SQd0B(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQd0BkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    N3:int=N-3
    avail:int=free
    total:int=0
    # while row==mark1 and free:
    while row==mark1 and avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<2
      next_rd:int=(rd|bit)>>2
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col|1<<N3 #<<注意
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQd0B(next_ld,next_rd|1<<N3,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    # while free:
    while avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQd0BkB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQd1BklB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    N4:int=N-4
    avail:int=free
    total:int=0
    # while row==mark1 and free:
    while row==mark1 and avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<3
      next_rd:int=(rd|bit)>>3
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col|1|1<<N4
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQd1B(next_ld|1,next_rd|1<<N4,next_col,row+3,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    # while free:
    while avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQd1BklB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQd1B(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    if row==endmark:
      # tempcounter[0]+=1
      # return
      return 1
    avail:int=free
    total:int=0
    # while free:
    while avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      # next_free:int=board_mask&~(next_ld|next_rd|next_col)&((1<<N)-1)
      next_free:int=board_mask&~blocked
      #if next_free and (row+1>=endmark or ~((next_ld<<1)|(next_rd>>1)|next_col)>0):
      # NG
      # if next_free and (row>=endmark-1 or ~((next_ld<<1)|(next_rd>>1)|(next_col))>0):
      # OK（空きが1つでもあるか）
      #if next_free and (row >= endmark - 1) or ((~blocked) & board_mask):
      # 1 行先を見るケース：
      if next_free and ((row + 1 >= endmark) or self._has_future_space(next_ld, next_rd, next_col, board_mask)):
        total+=self.SQd1B(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQd1BkBlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    N3:int=N-3
    avail:int=free
    total:int=0
    # while row==mark1 and free:
    while row==mark1 and avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<2
      next_rd:int=(rd|bit)>>2
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col|1<<N3
      next_free:int=board_mask&~blocked
      if next_free:
        # total+=self.SQd1BlB(((ld|bit)<<2),((rd|bit)>>2)|(1<<N3),col|bit,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
        total+=self.SQd1BlB(next_ld,next_rd|1<<N3,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    while avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQd1BkBlB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQd1BlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    avail:int=free
    total:int=0
    # while row==mark2 and free:
    while row==mark2 and avail:
      # Extract the rightmost available position
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<2|1
      next_rd:int=(rd|bit)>>2
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      # next_free:int=board_mask&~(next_ld|next_rd|next_col)&((1<<N)-1)
      # if next_free and (row+2>=endmark or ~blocked):
      #if next_free and (row+1>=endmark or ~((next_ld<<1)|(next_rd>>1)|next_col)>0):
      # NG
      # if next_free and (row>=endmark-1 or ~((next_ld<<1)|(next_rd>>1)|(next_col))>0):
      # OK（空きが1つでもあるか）
      #if next_free and (row >= endmark - 1) or ((~blocked) & board_mask):
      # 1 行先を見るケース：
      # if next_free and ((row >= endmark - 1) or self._has_future_space(next_ld, next_rd, next_col, board_mask)):
      # if next_free and (row+2>=endmark or ~blocked):
      # 2 行スキップ系（row+2 で終端チェックしているブロック）：
      if next_free and ((row + 2 >= endmark) or self._has_future_space(next_ld, next_rd, next_col, board_mask)):
        total+=self.SQd1B(next_ld,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    # while free: # General case when row != mark2
    while avail: # General case when row != mark2
      # bit:int=free&-free # Extract the rightmost available position
      # free&=free-1
      bit:int=avail&-avail # Extract the rightmost available position
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      # next_free:int=board_mask&~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit)) # Update diagonal and column occupancies
      next_free:int=board_mask&~blocked
      if next_free: # Recursive call if there are available positions
        total+=self.SQd1BlB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQd1BlkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    N3:int=N-3  # Precomputed value for performance
    avail:int=free
    total:int=0
    # while row==mark1 and free:
    while row==mark1 and avail:
      # bit:int=free&-free  # Extract the rightmost available position
      # free&=free-1
      bit:int=avail&-avail  # Extract the rightmost available position
      avail&=avail-1
      next_ld:int=(ld|bit)<<3
      next_rd:int=(rd|bit)>>3
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col|2|1<<N3
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQd1B(next_ld|2,next_rd|1<<N3,next_col,row+3,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    # while free:
    while avail:
      # bit:int=free&-free  # Extract the rightmost available position
      # free&=free-1
      bit:int=avail&-avail  # Extract the rightmost available position
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      # next_free=board_mask&~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQd1BlkB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQd1BlBkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    avail:int=free
    total:int=0
    # while row==mark1 and free:
    while row==mark1 and avail:
      # bit:int=free&-free  # Extract the rightmost available position
      # free&=free-1
      bit:int=avail&-avail  # Extract the rightmost available position
      avail&=avail-1
      next_ld:int=(ld|bit)<<2
      next_rd:int=(rd|bit)>>2
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col|1
      # next_free=board_mask&~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1)
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQd1BkB(next_ld|1,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    # while free:
    while avail:
      # bit:int=free&-free  # Extract the rightmost available position
      # free&=free-1
      bit:int=avail&-avail  # Extract the rightmost available position
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQd1BlBkB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQd1BkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    N3:int=N-3
    avail:int=free
    total:int=0
    # while row==mark2 and free:
    while row==mark2 and avail:
      # bit:int=free&-free  # Extract the rightmost available position
      # free&=free-1
      bit:int=avail&-avail  # Extract the rightmost available position
      avail&=avail-1
      next_ld:int=(ld|bit)<<2
      next_rd:int=(rd|bit)>>2
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col|1<<N3# Calculate the next free positions
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQd1B(next_ld,next_rd|1<<N3,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    # while free:
    while avail:
      # bit:int=free&-free  # Extract the rightmost available position
      # free&=free-1
      bit:int=avail&-avail  # Extract the rightmost available position
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      # next_free=board_mask&~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit))# Calculate the next free positions
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQd1BkB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQd2BlkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    N3:int=N-3
    avail:int=free
    total:int=0
    # while row==mark1 and free:
    while row==mark1 and avail:
      # bit:int=free&-free  # 最下位ビットを取得
      # free&=free-1  # 使用済みビットを削除
      bit:int=avail&-avail  # 最下位ビットを取得
      avail&=avail-1  # 使用済みビットを削除
      next_ld:int=(ld|bit)<<3
      next_rd:int=(rd|bit)>>3
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col|(1<<N3)|2
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQd2B(next_ld|2,next_rd|1<<N3,next_col,row+3,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    # while free:
    while avail:
      # bit:int=free&-free  # 最下位ビットを取得
      # free&=free-1  # 使用済みビットを削除
      bit:int=avail&-avail  # 最下位ビットを取得
      avail&=avail-1  # 使用済みビットを削除
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQd2BlkB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQd2BklB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    N4:int=N-4
    avail:int=free
    total:int=0
    # while row==mark1 and free:
    while row==mark1 and avail:
      # bit:int=free&-free  # 最下位のビットを取得
      # free&=free-1  # 使用済みのビットを削除
      bit:int=avail&-avail  # 最下位のビットを取得
      avail&=avail-1  # 使用済みのビットを削除
      next_ld:int=(ld|bit)<<3
      next_rd:int=(rd|bit)>>3
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col|(1<<N4)|1
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQd2B(next_ld|1,next_rd|1<<N4,next_col,row+3,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    # while free:
    while avail:
      # bit:int=free&-free  # 最下位のビットを取得
      # free&=free-1  # 使用済みのビットを削除
      bit:int=avail&-avail  # 最下位のビットを取得
      avail&=avail-1  # 使用済みのビットを削除
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQd2BklB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQd2BkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    N3:int=N-3
    avail:int=free
    total:int=0
    # while row==mark2 and free:
    while row==mark2 and avail:
      # bit:int=free&-free  # 最下位ビットを取得
      # free&=free-1  # 使用済みビットを削除
      bit:int=avail&-avail  # 最下位ビットを取得
      avail&=avail-1  # 使用済みビットを削除
      next_ld:int=(ld|bit)<<2
      next_rd:int=(rd|bit)>>2
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col|1<<N3
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQd2B(next_ld,next_rd|1<<N3,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    # while free:
    while avail:
      # bit:int=free&-free  # 最下位ビットを取得
      # free&=free-1  # 使用済みビットを削除
      bit:int=avail&-avail  # 最下位ビットを取得
      avail&=avail-1  # 使用済みビットを削除
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQd2BkB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQd2BlBkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    avail:int=free
    total:int=0
    # while row==mark1 and free:
    while row==mark1 and avail:
      # bit:int=free&-free  # Get the lowest bit
      # free&=free-1  # Remove the lowest bit
      bit:int=avail&-avail  # Get the lowest bit
      avail&=avail-1  # Remove the lowest bit
      next_ld:int=(ld|bit)<<2
      next_rd:int=(rd|bit)>>2
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col|1
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQd2BkB(next_ld|1,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    # while free:
    while avail:
      # bit:int=free&-free  # Get the lowest bit
      # free&=free-1  # Remove the lowest bit
      bit:int=avail&-avail  # Get the lowest bit
      avail&=avail-1  # Remove the lowest bit
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQd2BlBkB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQd2BlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    avail:int=free
    total:int=0
    # while row==mark2 and free:
    while row==mark2 and avail:
      # bit:int=free&-free  # Get the lowest bit
      # free&=free-1  # Remove the lowest bit
      bit:int=avail&-avail  # Get the lowest bit
      avail&=avail-1  # Remove the lowest bit
      next_ld:int=(ld|bit)<<2
      next_rd:int=(rd|bit)>>2
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col|1
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQd2B(next_ld|1,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    # while free:
    while avail:
      # bit:int=free&-free  # Get the lowest bit
      # free&=free-1  # Remove the lowest bit
      bit:int=avail&-avail  # Get the lowest bit
      avail&=avail-1  # Remove the lowest bit
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQd2BlB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQd2BkBlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    N3:int=N-3
    avail:int=free
    total:int=0
    # while row==mark1 and free:
    while row==mark1 and avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<2
      next_rd:int=(rd|bit)>>2
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col|1<<N3
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQd2BlB(next_ld,next_rd|1<<N3,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    # while free:
    while avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQd2BkBlB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQd2B(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    avail:int=free
    total:int=0
    if row==endmark:
      # if (free&(~1))>0:
      if (avail&(~1))>0:
        # tempcounter[0]+=1
        return 1
      # return
    # while free:
    while avail:
      # bit:int=free&-free  # 最も下位の1ビットを取得
      # free&=free-1  # 使用済みビットを削除
      bit:int=avail&-avail  # 最も下位の1ビットを取得
      avail&=avail-1  # 使用済みビットを削除
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free=board_mask&~blocked
      # NG
      # if next_free and (row>=endmark-1 or ~((next_ld<<1)|(next_rd>>1)|(next_col))>0):
      # OK（空きが1つでもあるか）
      # if next_free and (row >= endmark - 1) or ((~blocked) & board_mask):
      # 1 行先を見るケース：
      if next_free and ((row >= endmark - 1) or self._has_future_space(next_ld, next_rd, next_col, board_mask)):
        total+=self.SQd2B(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQBlBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    avail:int=free
    total:int=0
    # while row==mark2 and free:
    while row==mark2 and avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<2
      next_rd:int=(rd|bit)>>2
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col|1
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQBjrB(next_ld|1,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    # while free:
    while avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQBlBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQBkBlBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    N3:int=N-3
    avail:int=free
    total:int=0
    # while row==mark1 and free:
    while row==mark1 and avail:
      # bit:int=free&-free  # Isolate the rightmost 1 bit.
      # free&=free-1  # Remove the isolated bit from free.
      bit:int=avail&-avail  # Isolate the rightmost 1 bit.
      avail&=avail-1  # Remove the isolated bit from avail.
      next_ld:int=(ld|bit)<<2
      next_rd:int=(rd|bit)>>2
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col|1<<N3
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQBlBjrB(next_ld,next_rd|1<<N3,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    # while free:
    while avail:
      # bit:int=free&-free  # Isolate the rightmost 1 bit.
      # free&=free-1  # Remove the isolated bit from free.
      bit:int=avail&-avail  # Isolate the rightmost 1 bit.
      avail&=avail-1  # Remove the isolated bit from avail.
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQBkBlBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    avail:int=free
    total:int=0
    if row==jmark:
      # free&=~1  # Clear the least significant bit (mark position 0 unavailable).
      avail&=~1  # Clear the least significant bit (mark position 0 unavailable).
      ld|=1  # Mark left diagonal as occupied for position 0.
      # while free:
      while avail:
        # bit:int=free&-free  # Get the lowest bit (first free position).
        # free&=free-1  # Remove this position from the free positions.
        bit:int=avail&-avail  # Get the lowest bit (first avail position).
        avail&=avail-1  # Remove this position from the avail positions.
        next_ld:int=(ld|bit)<<1
        next_rd:int=(rd|bit)>>1
        next_col:int=col|bit
        blocked:int=next_ld|next_rd|next_col
        next_free:int=board_mask&~blocked
        if next_free:
          total+=self.SQB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
      # return
      return total
    # while free:
    while avail:
      # bit:int=free&-free  # Get the lowest bit (first free position).
      # free&=free-1  # Remove this position from the free positions.
      bit:int=avail&-avail  # Get the lowest bit (first avail position).
      avail&=avail-1  # Remove this position from the avail positions.
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    avail:int=free
    total:int=0
    if row==endmark:
      # tempcounter[0]+=1
      # return
      return 1
    # while free:
    while avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      #if next_free and (row>=endmark-1 or ~((next_ld<<1)|(next_rd>>1)|next_col)>0):
      # NG
      # if next_free and (row>=endmark-1 or ~((next_ld<<1)|(next_rd>>1)|(next_col))>0):
      # OK（空きが1つでもあるか）
      # if next_free and (row >= endmark - 1) or ((~blocked) & board_mask):
      # 1 行先を見るケース：
      if next_free and ((row >= endmark - 1) or self._has_future_space(next_ld, next_rd, next_col, board_mask)):
        total+=self.SQB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQBlBkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    avail:int=free
    total:int=0
    # while row==mark1 and free:
    while row==mark1 and avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<2
      next_rd:int=(rd|bit)>>2
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col|1
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQBkBjrB(next_ld|1,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    # while free:
    while avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQBlBkBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQBkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    N3:int=N-3
    avail:int=free
    total:int=0
    # while row==mark2 and free:
    while row==mark2 and avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<2
      next_rd:int=(rd|bit)>>2
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col|1<<N3
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQBjrB(next_ld,next_rd|1<<N3,next_col,row+2,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    # while free:
    while avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQBkBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQBklBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    N4:int=N-4
    avail:int=free
    total:int=0
    # while row==mark1 and free:
    while row==mark1 and avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<3
      next_rd:int=(rd|bit)>>3
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col|1<<N4|1
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQBjrB(next_ld|1,next_rd|1<<N4,next_col,row+3,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    # while free:
    while avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQBklBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQBlkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    N3:int=N-3
    avail:int=free
    total:int=0
    # while row==mark1 and free:
    while row==mark1 and avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<3
      next_rd:int=(rd|bit)>>3
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col|1<<N3|2
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQBjrB(next_ld|2,next_rd|1<<N3,next_col,row+3,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    # while free:
    while avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQBlkBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQBjlBkBlBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    N1:int=N-1
    avail:int=free
    total:int=0
    if row==N1-jmark:
      rd|=1<<N1
      # free&=~(1<<N1)
      next_ld:int=ld<<1
      next_rd:int=rd>>1
      next_col:int=col
      blocked:int=next_ld|next_rd|next_col
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQBkBlBjrB(next_ld,next_rd,next_col,row,next_free,jmark,endmark,mark1,mark2,board_mask,N)
      # return
      return total
    # while free:
    while avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQBjlBkBlBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQBjlBlBkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    N1:int=N-1
    avail:int=free
    total:int=0
    if row==N1-jmark:
      rd|=1<<N1
      # free&=~(1<<N1)
      next_ld:int=ld<<1
      next_rd:int=rd>>1
      next_col:int=col
      blocked:int=next_ld|next_rd|next_col
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQBlBkBjrB(next_ld,next_rd,next_col,row,next_free,jmark,endmark,mark1,mark2,board_mask,N)
      # return
      return total
    # while free:
    while avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQBjlBlBkBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQBjlBklBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    N1:int=N-1
    avail:int=free
    total:int=0
    if row==N1-jmark:
      rd|=1<<N1
      # free&=~(1<<N1)
      next_ld:int=ld<<1
      next_rd:int=rd>>1
      next_col:int=col
      blocked:int=next_ld|next_rd|next_col
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQBklBjrB(next_ld,next_rd,next_col,row,next_free,jmark,endmark,mark1,mark2,board_mask,N)
      # return
      return total
    # while free:
    while avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQBjlBklBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
  def SQBjlBlkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    #board_mask:int=(1<<(N-1))-1
    N1:int=N-1
    avail:int=free
    total:int=0
    if row==N1-jmark:
      rd|=1<<N1
      # free&=~(1<<N1)
      next_ld:int=ld<<1
      next_rd:int=rd>>1
      next_col:int=col
      blocked:int=next_ld|next_rd|next_col
      next_free=board_mask&~blocked
      if next_free:
        total+=self.SQBlkBjrB(next_ld,next_rd,next_col,row,next_free,jmark,endmark,mark1,mark2,board_mask,N)
      # return
      return total
    # while free:
    while avail:
      # bit:int=free&-free
      # free&=free-1
      bit:int=avail&-avail
      avail&=avail-1
      next_ld:int=(ld|bit)<<1
      next_rd:int=(rd|bit)>>1
      next_col:int=col|bit
      blocked:int=next_ld|next_rd|next_col
      next_free:int=board_mask&~blocked
      if next_free:
        total+=self.SQBjlBlkBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
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
