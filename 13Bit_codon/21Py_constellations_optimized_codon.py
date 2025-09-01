#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
コンステレーション版 最適化　Ｎクイーン

# 実行結果
workspace#suzuki$ codon build -release 21Py_constellations_optimized_codon.py && ./21Py_constellations_optimized_codon
 N:        Total       Unique        hh:mm:ss.ms
 5:           18            0         0:00:00.096
 6:            4            0         0:00:00.000
 7:           40            0         0:00:00.000
 8:           92            0         0:00:00.000
 9:          352            0         0:00:00.000
10:          724            0         0:00:00.000
11:         2680            0         0:00:00.007
12:        14200            0         0:00:00.001
13:        73712            0         0:00:00.003
14:       365596            0         0:00:00.010
15:      2279184            0         0:00:00.032
16:     14772512            0         0:00:00.160
17:     95815104            0         0:00:00.566
18:    666090624            0         0:00:01.960

GPU/CUDA 11CUDA_constellation_symmetry.cu
17:     95815104            0    000:00:00:03.41

$ nvcc -O3 -arch=sm_61 -m64 -ptx -prec-div=false 04CUDA_Symmetry_BitBoard.cu && POCL_DEBUG=all ./a.out -n ;
17:         95815104         11977939     000:00:00:00.26
18:        666090624         83263591     000:00:00:01.65

fedora$ codon build -release 21Py_constellations_optimized_codon.py && ./21Py_constellations_optimized_codon
 N:        Total       Unique        hh:mm:ss.ms
16:     14772512            0         0:00:00.485
17:     95815104            0         0:00:03.137
"""

# レベルA（すぐ入れられる／既に「済」でも検証コストが低い）

#済  1. [Opt-01] ビット演算による衝突枝刈り（cols/hills/dales）
#済  2. [Opt-02] 左右対称性除去（1 行目の列を 0～n//2−1 に制限）
#済  3. [Opt-03] 中央列の特別処理（N 奇数時）
#済  4. [Opt-04] 180度対象除去
#済  5. [Opt-05] 角位置（col==0）での分岐（COUNT2 偏重の明示化）
#済  6. [Opt-06] 並列処理（初手 col ごとに multiprocessing で分割）
#済  7. [Opt-07] 1 行目以外でも部分対称除去（行列単位）
#       * 途中段階（深さ r の盤面）を都度「辞書順最小」の canonical かどうかチェックして、そうでなければ枝刈り。
#不要  8. [Opt-08] is_canonical() による“部分盤面”の辞書順最小チェックを高速化（キャッシュ/軽量版）（
#       * 「完成盤」だけでなく“部分盤面”用に軽量な変換（行の回転・反転は途中情報だけで可）を実装。
#不要  9. [Opt-09] Zobrist Hash による transposition / visited 状態の高速検出
#       * N-Queens では完全一致局面の再訪は少ないですが、「部分対称 canonical チェックの結果」をハッシュ化してメモ化する用途で効果（計算の再実行を削減）。
#不要 10. [Opt-10] マクロチェス（局所パターン）による構築制限
#       * 現実装との整合や有効なパターン定義次第で効果差が大。ルール設計が難しい。 
#不要 11. [Opt-11] 「ミラー＋90°回転」による“構築時”の重複複除
#       * 完成後の対称判定より、構築途中で 8 対称性を逐次判定するのはコスト高＆実装が煩雑。部分盤面を8通り生成するコストが高く、B系の“軽量 canonical 部分盤面判定”＋Zobrist の方がバランスが良いことが多いです。 
#済 12. [Opt-12] ビット演算のインライン化

# メモリ管理・最適化
# 1. 盤面・星座の“一意シグネチャ”をZobrist hashやtupleで管理
# 今はijkl_listがSet[int]（16bit packedの盤面ID）ですが、
# 「星座の状態→Zobrist hash or tuple」も併用可能
# （星座構造が大きくなったり部分一致チェックが多いとき特に有効）
# 2. 盤面や星座の辞書キャッシュ（dict）による一意管理
# 星座リストや部分盤面ごとに、「一度作ったものはdictでキャッシュ」
# 3. Jasmin変換のキャッシュ化（生成済み盤面の再利用）【済】
# ijkl_list_jasmin = {self.jasmin(c, N) for c in ijkl_list}
# も、盤面→jasmin変換は「一度計算したらdictでキャッシュ」が効果大
# 4. 星座ごとに「hash/tuple key」を使ったキャッシュ辞書の導入
# set_pre_queensやサブコンステレーション生成時も「(ld, rd, col, ...)のtuple」や「部分盤面hash」をkeyに
# 一度作った星座はdictから即座に再利用できる構造
# 5. 星座生成全体をpickleなどで「Nごとにファイル化」して超巨大N対応
# すでに解説済ですが、gen_constellationsの全出力をconstellations_N17.pklのようなファイルでキャッシュ
# 実行時にRAM展開し、毎回再生成を回避
# ファイルI/O最小化も同時に達成

# バックトラック関数の最適化と枝狩り
# mark1 / mark2 の考慮
# mark1 や mark2（星座構成のための特定行）は、後から強制的に配置することが決まっているクイーンの位置です。
# その位置のビットは「ブロック」から除外（＝空きとして扱う）べきなので、~(1 << …) で解除します。
# 次の行が完全にブロックされているなら、その選択肢（現在の bit による配置）は探索する意味がない。
# つまり、次の row+1 にクイーンを置ける列が「ひとつも無い」場合、早期に continue。
# これは free ではなく、next_ld / next_rd / next_col による影響を見ているのがポイント。
# 効果
# この最適化により、再帰的なバックトラックの深さを減らせるケースが増え、特に解が少ない構成や途中で詰む分岐が多い盤面で効果が高く出ます。

#!/usr/bin/env python3

import os
from typing import List,Set,Dict
from datetime import datetime

# pypyを使うときは以下を活かしてcodon部分をコメントアウト
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')

# ------------------------------------------------------------
#「[Opt-03] 中央列特別処理（奇数N）」
# ------------------------------------------------------------
"""
    # --- [Opt-03] 中央列特別処理（奇数Nの場合のみ） ---
    if N % 2==1:
      center = N // 2
      ijkl_list.update(
        self.to_ijkl(i, j, center, l)
        for l in range(center + 1, N-1)
        for i in range(center + 1, N-1)
        if i != (N-1)-l
        for j in range(N-center-2, 0, -1)
        if j != i and j != l
        # 180°回転盤面がセットに含まれていない
        # if not self.check_rotations(ijkl_list, i, j, center, l, N)
        if not self.rot180_in_set(ijkl_list, i, j, center, l, N)
      )
    # --- [Opt-03] 中央列特別処理（奇数Nの場合のみ） ---
"""
# ------------------------------------------------------------
# 結果カウンタ（COUNT2/4/8 分類用）  # [Opt-05] / 対称性分類まとめ
# ------------------------------------------------------------
@dataclass
class Counts:
    c2: int = 0  # 左右対称のみ
    c4: int = 0  # 180°回転まで同一
    c8: int = 0  # 8 対称すべて異なる

    @property
    def total(self) -> int:
        return self.c2 * 2 + self.c4 * 4 + self.c8 * 8
# ------------------------------------------------------------
# [Opt-08] Zobrist Hash テーブル生成（初期化）
# Zobrist 用の乱数テーブル（部分盤面 canonical 判定のメモ化などで使用） # [Opt-09]
# ※ N は solve() で決め打ちなので、初期化は solve 側で行う前提の例
# ------------------------------------------------------------
def init_zobrist(N: int) -> List[List[int]]:
    import random
    return [[random.getrandbits(64) for _ in range(N)] for _ in range(N)]
# ---------------------------
# [Opt-08] Zobrist Hash による部分盤面ハッシュ化
# ---------------------------
def compute_zobrist_hash(board: List[int], row: int, zobrist: List[List[int]]) -> int:
    h = 0
    for r in range(row):
        h ^= zobrist[r][board[r]]
    return h
# ---------------------------
# [Opt-07+08] 部分盤面の正準性チェック + Zobristキャッシュ
# ---------------------------
def is_partial_canonical(board: List[int], row: int, N: int,zobrist: List[List[int]], zcache: dict) -> bool:
  key = compute_zobrist_hash(board, row, zobrist)
  if key in zcache:
    return zcache[key]
  current = tuple(board[:row])
  # ミラー反転のみチェック（左右対称のみ）
  mirrored = tuple(N-1-board[r] for r in range(row))
  # 必要であれば回転90/180/270 も加える（今はミラーのみ）
  minimal = min(current, mirrored)
  result = (current==minimal)
  zcache[key] = result
  return result
# ------------------------------------------------------------
# マクロチェス（ローカルパターン）ルールの例スタブ  # [Opt-10]
# ------------------------------------------------------------
def violate_macro_patterns(board: List[int], row: int, N: int) -> bool:
  # 例：最初の3行で中央寄りが密集していたら破綻しやすいため除外
  if row >= 3:
    c0 = board[row-1]
    c1 = board[row-2]
    c2 = board[row-3]
    if abs(c0-c1) <= 1 and abs(c1-c2) <= 1:
      return True
  return False

# ---------------------------
# 使い方（任意の再帰関数 backtrack(row, ...) の中で）
# ---------------------------
# if not is_partial_canonical(board, row, N, zobrist, zcache):
#   return
# if violate_macro_patterns(board, row, N):
#   return

# ※ `zobrist` は solve() 内で init_zobrist(N) により生成
# `zcache` は {} で初期化し、各 backtrack に渡す
# これらを main の backtrack 系ルーチン（例：SQdB/SQaB）に組み込みます。
# ご希望があれば、具体的な関数ごとに挿入済みコードも差し上げます。

# ------------------------------------------------------------
# 8 対称生成（完成盤面用） # [Opt-05] / [Opt-04]
# 完成盤面の重複判定・分類に使用
# ------------------------------------------------------------
def symmetries(board):
    # board: row -> col の配置（例: board[r] = c）
    n = len(board)
    def rot90(b):   # 回転 90°: (r, c) -> (c, n-1-r)
        return [b.index(i) for i in range(n-1, -1, -1)]
    def rot180(b):  # 回転 180°
        return [n-1-b[n-1-r] for r in range(n)]
    def rot270(b):  # 回転 270°
        return [b.index(n-1-i) for i in range(n)]
    def reflect(b): # ミラー（左右反転）: (r, c) -> (r, n-1-c)
        return [n-1-c for c in b]
    r0 = board
    r1 = rot90(r0)
    r2 = rot180(r0)
    r3 = rot270(r0)
    f0 = reflect(r0)
    f1 = reflect(r1)
    f2 = reflect(r2)
    f3 = reflect(r3)
    return [tuple(r0), tuple(r1), tuple(r2), tuple(r3),
            tuple(f0), tuple(f1), tuple(f2), tuple(f3)]
# ------------------------------------------------------------
# 完成盤面の対称性分類 # [Opt-05] / [Opt-04]
# ------------------------------------------------------------
def classify(board):
    syms = symmetries(board)
    mn = min(syms)
    uniq = len(set(syms))
    if uniq==1:
        return 'c2'  # 実際は c2=1, c4=0, c8=0 のように扱うなら調整可
    elif uniq==2 or uniq==4:
        return 'c4'
    else:
        return 'c8'
# ------------------------------------------------------------
# ビット演算バックトラッキング # [Opt-01]
#  -cols, hills, dales を bit で持ち、free = ~(cols|hills|dales) で高速衝突除去
#  -1 行目（row==0）は [Opt-02/03/05/06] によって枝分かれ済み
#  -途中で [Opt-07/08/09/10] を挟み込む
# ------------------------------------------------------------
def backtrack(n, row, cols, hills, dales, board, counts, is_corner=False, zobrist=None, zcache=None):
    if row==n:
        # 完成。対称分類  # [Opt-05]/[Opt-04]
        typ = classify(board)
        if typ=='c2':
            counts.c2 += 1
        elif typ=='c4':
            counts.c4 += 1
        else:
            counts.c8 += 1
        return
    # ---------------------------
    # [Opt-07]/[Opt-08] 部分盤面 canonical チェック
    if not is_partial_canonical(board, row, n, zobrist, zcache):
        return
    # ---------------------------
    # [Opt-10] マクロチェス局所パターンで除去
    if violate_macro_patterns(board, row, n):
        return
    # ---------------------------
    # 衝突していない位置（free）をビットで求める  # [Opt-01]
    mask = (1<<n)-1
    free = mask&~(cols|hills|dales)
    while free:
        bit = -free & free
        col = (bit.bit_length()-1)
        free ^= bit
        board[row] = col  # 盤面にセット
        # 180°対称や角位置分岐での特別扱いを、必要ならここに入れる # [Opt-04]/[Opt-05]
        # 例：is_corner==True の時は row>0 で (n-1-col) などの除去条件を適用 …など
        backtrack(
            n, row + 1,
            cols|bit,
            (hills|bit)<<1,
            (dales|bit)>>1,
            board,
            counts,
            is_corner=is_corner,
            zobrist=zobrist,
            zcache=zcache
        )
        board[row] = -1  # 戻す（明示的にしなくても良いが読みやすさのため）
# ------------------------------------------------------------
# 1 行目を左右対称で半分に制限し（[Opt-02]）、中央列を別処理（[Opt-03]）、
# 初手ごとに並列化（[Opt-06]）する solve_nqueens()
# ------------------------------------------------------------
def solve_nqueens(n, workers=None):
    if workers is None:
        workers = max(1, cpu_count()-1)
    # Zobrist 用意（[Opt-09] を本当に使うなら）
    zobrist = init_zobrist(n)
    manager_counts = Counts()
    # 1 行目の左半分のみ探索（左右ミラー分を除去） # [Opt-02]
    first_cols = list(range(n // 2))
    # 奇数 N の中央列は別枠 # [Opt-03]
    center_col = n // 2 if (n % 2==1) else None
    args = []
    for col in first_cols:
        args.append((n, col, True))  # col==0 のケースは is_corner=True として扱うなど # [Opt-05]
    if center_col is not None:
        args.append((n, center_col, False))  # 中央列は is_corner=False で別分類
    # 並列処理 # [Opt-06]
    with Pool(processes=workers) as pool:
        for c in pool.imap_unordered(_worker, args):
            # c は Counts
            manager_counts.c2 += c.c2
            manager_counts.c4 += c.c4
            manager_counts.c8 += c.c8
    return manager_counts
# ------------------------------------------------------------
class NQueens19:
  def __init__():
    self._rot_cache = {}
  # 1. check_rotations() を早めにフィルタする
  # 現在はジェネレータ内の末尾で check_rotations(...) を実行してい
  # ますが、もしこれが重い処理なら、事前フィルタリングかメモ化が有
  # 効です：
  #
  # check_rotations_cached: check_rotations をキャッシュして高速化
  def check_rotations_cached(self, i: int, j: int, k: int, l: int, N: int) -> bool:
    key = (i, j, k, l, N)
    if key not in self._rot_cache:
      self._rot_cache[key] = self.check_rotations(set(), i, j, k, l, N)
    return self._rot_cache[key]
  # 時計回りに90度回転
  # rot90 メソッドは、90度の右回転（時計回り）を行います
  # 元の位置 (row,col) が、回転後の位置 (col,N-1-row) になります。
  def rot90(self,ijkl:int,N:int)->int:
    return ((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl)
  # 対称性のための計算と、ijklを扱うためのヘルパー関数。
  # 開始コンステレーションが回転90に対して対称である場合
  def rot180(self,ijkl:int,N:int)->int:
    return ((N-1-self.getj(ijkl))<<15)+((N-1-self.geti(ijkl))<<10)+((N-1-self.getl(ijkl))<<5)+(N-1-self.getk(ijkl))
  def rot180_in_set(self,ijkl_list:Set[int],i:int,j:int,k:int,l:int,N:int)->bool:
    return self.rot180(self.to_ijkl(i, j, k, l), N) in ijkl_list
  # 盤面ユーティリティ群（ビットパック式盤面インデックス変換）
  # Python実装のgeti/getj/getk/getl/toijklに対応。
  # [i, j, k, l] 各クイーンの位置情報を5ビットずつ
  # 整数値（ijkl）にパック／アンパックするためのマクロ。
  # 15ビット～0ビットまでに [i|j|k|l] を格納する設計で、
  # constellationのsignatureや回転・ミラー等の盤面操作を高速化する。
  # 例：
  #  -geti(ijkl): 上位5ビット（15-19）からiインデックスを取り出す
  #  -toijkl(i, j, k, l): 各値を5ビット単位で連結し
  # 一意な整数値（signature）に変換
  # [注意] N≦32 まで対応可能
  def geti(self,ijkl:int)->int:
    return (ijkl>>15)&0x1F
  def getj(self,ijkl:int)->int:
    return (ijkl>>10)&0x1F
  def getk(self,ijkl:int)->int:
    return (ijkl>>5)&0x1F
  def getl(self,ijkl:int)->int:
    return ijkl&0x1F
  def to_ijkl(self,i:int,j:int,k:int,l:int)->int:
    return (i<<15)+(j<<10)+(k<<5)+l
  # symmetry: 回転・ミラー対称性ごとの重複補正
  # (90度:2, 180度:4, その他:8)
  def symmetry(self,ijkl:int,N:int)->int:
    return 2 if self.symmetry90(ijkl,N) else 4 if self.geti(ijkl)==N-1-self.getj(ijkl) and self.getk(ijkl)==N-1-self.getl(ijkl) else 8
  def symmetry90(self,ijkl:int,N:int)->bool:
    return ((self.geti(ijkl)<<15)+(self.getj(ijkl)<<10)+(self.getk(ijkl)<<5)+self.getl(ijkl))==(((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl))
  # 左右のミラー 与えられたクイーンの配置を左右ミラーリングします。
  # 各クイーンの位置を取得し、列インデックスを N-1 から引いた位置
  # に変更します（左右反転）。行インデックスはそのままにします。
  def mirvert(self,ijkl:int,N:int)->int:
    return self.to_ijkl(N-1-self.geti(ijkl),N-1-self.getj(ijkl),self.getl(ijkl),self.getk(ijkl))
  # 大小を比較して小さい最値を返却
  def ffmin(self,a:int,b:int)->int:
    return min(a,b)
  # 指定した盤面 (i, j, k, l) を90度・180度・270度回転したいずれか
  # の盤面がすでにIntHashSetに存在しているかをチェックする関数
  # @param ijklList 既出盤面signature（ijkl値）の集合（HashSet）
  # @param i,j,k,l  チェック対象の盤面インデックス
  # @param N        盤面サイズ
  # @return         いずれかの回転済み盤面が登録済みなら1、なければ0
  # @details
  #  -N-Queens探索で、既存盤面の90/180/270度回転形と重複する配置
  # を高速に排除する。
  #  -回転後のijklをそれぞれ計算し、HashSetに含まれていれば即1を
  # 返す（重複扱い）。
  #  -真の“unique配置”のみ探索・カウントしたい場合の前処理とし
  # て必須。
  def check_rotations(self,ijkl_list:Set[int],i:int,j:int,k:int,l:int,N:int)->bool:
      return any(rot in ijkl_list for rot in [((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i,((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+(N-1-k), (l<<15)+(k<<10)+((N-1-i)<<5)+(N-1-j)])
    # rot90=((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i
    # rot180=((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+(N-1-k)
    # rot270=(l<<15)+(k<<10)+((N-1-i)<<5)+(N-1-j)
    # return any(rot in ijkl_list for rot in (rot90,rot180,rot270))
  #--------------------------------------------
  # 1. Jasmin変換キャッシュを導入する
  # [Opt-08] キャッシュ付き jasmin() のラッパー
  #--------------------------------------------
  jasmin_cache = {}
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
  # i,j,k,lをijklに変換し、特定のエントリーを取得する関数
  # 各クイーンの位置を取得し、最も左上に近い位置を見つけます
  # 最小の値を持つクイーンを基準に回転とミラーリングを行い、配置を最も左上に近い標準形に変換します。
  # 最小値を持つクイーンの位置を最下行に移動させる
  # i は最初の行（上端） 90度回転2回
  # j は最後の行（下端） 90度回転0回
  # k は最初の列（左端） 90度回転3回
  # l は最後の列（右端） 90度回転1回
  # 優先順位が l>k>i>j の理由は？
  # l は右端の列に位置するため、その位置を基準に回転させることで、配置を最も標準形に近づけることができます。
  # k は左端の列に位置しますが、l ほど標準形に寄せる影響が大きくないため、次に優先されます。
  # i は上端の行に位置するため、行の位置を基準にするよりも列の位置を基準にする方が配置の標準化に効果的です。
  # j は下端の行に位置するため、優先順位が最も低くなります。
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
  # 4. pickleファイルで星座リストそのものをキャッシュ
  #---------------------------------
  def file_exists(self, fname: str) -> bool:
      try:
          with open(fname, "rb"):
          # f=open(fname,"rb")
          # f.close()
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
          # return self.load_constellations_bin(fname)
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
  subconst_cache = {}
  def set_pre_queens_cached(self, ld: int, rd: int, col: int, k: int, l: int,row: int, queens: int, LD: int, RD: int,counter: list, constellations: List[Dict[str, int]], N: int, preset_queens: int,visited:set[int]) -> None:
    key = (ld, rd, col, k, l, row, queens, LD, RD, N, preset_queens)
    # キャッシュの本体をdictかsetでグローバル/クラス変数に
    if not hasattr(self, "subconst_cache"):
      self.subconst_cache = {}
    subconst_cache = self.subconst_cache
    if key in subconst_cache:
      # 以前に同じ状態で生成済み → 何もしない（または再利用）
      return
    # 新規実行（従来通りset_pre_queensの本体処理へ）
    self.set_pre_queens(ld, rd, col, k, l, row, queens, LD, RD, counter, constellations, N, preset_queens,visited)
    subconst_cache[key] = True  # マークだけでOK

  constellation_signatures = set()
  #---------------------------------
  # [Opt-09] Zobrist Hash（Opt-09）の導入とその用途
  # ビットボード設計でも、「盤面のハッシュ」→「探索済みフラグ」で枝刈りは可能です。
  #---------------------------------
  def state_hash(self,ld: int, rd: int, col: int, row: int) -> int:
    if None in (ld, rd, col, row):
       return -1
    # 64ビット整数に収まるようにビット操作で圧縮
    # 単純な状態ハッシュ（高速かつ衝突率低めなら何でも可）
    return (ld * 0x9e3779b9) ^ (rd * 0x7f4a7c13) ^ (col * 0x6a5d39e9) ^ row
  #---------------------------------
  # 開始コンステレーション（部分盤面）の生成関数
  # N-Queens探索の初期状態を最適化するため、3つまたは4つのクイーン（presetQueens）を
  # あらかじめ盤面に配置した全ての部分盤面（サブコンステレーション）を列挙・生成する。
  # 再帰的に呼び出され、各行ごとに可能な配置をすべて検証。
  # @param ld   左対角線のビットマスク（既にクイーンがある位置は1）
  # @param rd   右対角線のビットマスク
  # @param col  縦方向（列）のビットマスク
  # @param k    事前にクイーンを必ず置く行のインデックス1
  # @param l    事前にクイーンを必ず置く行のインデックス2
  # @param row  現在の再帰探索行
  # @param queens 現在までに盤面に配置済みのクイーン数
  # @param LD/RD 探索初期状態用のマスク（使用例次第で追記）
  # @param counter 生成されたコンステレーション数を書き込むカウンタ
  # @param constellations 生成したコンステレーション（部分盤面配置）のリスト
  # @param N     盤面サイズ
  # @details
  #  -row==k/lの場合は必ずクイーンを配置し次の行へ進む
  #  -queens==presetQueensに到達したら、現時点の盤面状態をコンステレーションとして記録
  #  -その他の行では、空いている位置すべてにクイーンを順次試し、再帰的に全列挙
  #  -生成された部分盤面は、対称性除去・探索分割等の高速化に用いる
  def set_pre_queens(self,ld:int,rd:int,col:int,k:int,l:int,row:int,queens:int,LD:int,RD:int,counter:list,constellations:List[Dict[str,int]],N:int,preset_queens:int,visited:set[int])->None:
    mask=(1<<N)-1  # setPreQueensで使用
    # ----------------------------
    # [Opt-09] 状態ハッシュによる探索枝の枝刈り
    # バックトラック系の冒頭に追加　やりすぎると解が合わない
    h: int = self.state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    # ----------------------------
    # k行とl行はスキップ
    if row==k or row==l:
      # self.set_pre_queens(ld<<1,rd>>1,col,k,l,row+1,queens,LD,RD,counter,constellations,N,preset_queens,visited)
      self.set_pre_queens_cached(ld<<1,rd>>1,col,k,l,row+1,queens,LD,RD,counter,constellations,N,preset_queens,visited)
      return
    # クイーンの数がpreset_queensに達した場合、現在の状態を保存
    # ------------------------------------------------
    # 3. 星座のsignature重複防止
    # ------------------------------------------------
    # if queens==preset_queens:
    #   constellation= {"ld": ld,"rd": rd,"col": col,"startijkl": row<<20,"solutions":0}
    #   # 新しいコンステレーションをリストに追加
    #   constellations.append(constellation)
    #   counter[0]+=1
    #   return
    # if queens==preset_queens:
    #   # signatureの生成
    #   signature = (ld, rd, col, k, l, row)  # 必要な変数でOK
    #   # signaturesセットをクラス変数やグローバルで管理
    #   if not hasattr(self, "constellation_signatures"):
    #     self.constellation_signatures = set()
    #   signatures = self.constellation_signatures
    #   if signature not in signatures:
    #     constellation = {"ld": ld, "rd": rd, "col": col, "startijkl": row<<20, "solutions": 0}
    #     constellations.append(constellation) #星座データ追加
    #     signatures.add(signature)
    #     counter[0] += 1
    #   return
    if queens == preset_queens:
        # constellation_signatures セットの初期化（Codon対応）
        if not hasattr(self, "constellation_signatures"):
            self.constellation_signatures = set()
        # signature の生成
        signature = (ld, rd, col, k, l, row)
        # 初回の signature のみ追加
        if signature not in self.constellation_signatures:
            constellations.append({
                "ld": ld, "rd": rd, "col": col,
                "startijkl": row << 20,
                "solutions": 0
            })
            self.constellation_signatures.add(signature)
            counter[0] += 1
        return
    # ------------------------------------------------

    # 現在の行にクイーンを配置できる位置を計算
    free=mask&~(ld|rd|col|(LD>>(N-1-row))|(RD<<(N-1-row)))
    while free:
      bit:int=free&-free  # 最も下位の1ビットを取得
      free&=free-1  # 使用済みビットを削除
      # クイーンを配置し、次の行に進む
      # self.set_pre_queens((ld|bit)<<1,(rd|bit)>>1,col|bit,k,l,row+1,queens+1,LD,RD,counter,constellations,N,preset_queens,visited)
      self.set_pre_queens_cached((ld|bit)<<1,(rd|bit)>>1,col|bit,k,l,row+1,queens+1,LD,RD,counter,constellations,N,preset_queens,visited)
  # ConstellationArrayListの各Constellation（部分盤面）ごとに
  # N-Queens探索を分岐し、そのユニーク解数をsolutionsフィールドに記録する関数（CPU版）
  # @param constellations 解探索対象のConstellationArrayListポインタ
  # @param N              盤面サイズ
  # @details
  #  -各Constellation（部分盤面）ごとにj, k, l, 各マスク値を展開し、
  #     複雑な分岐で最適な再帰ソルバー（SQ...関数群）を呼び出して解数を計算
  #  -分岐ロジックは、部分盤面・クイーンの位置・コーナーからの距離などで高速化
  #  -解数はtemp_counterに集約し、各Constellationのsolutionsフィールドに記録
  #  -symmetry(ijkl, N)で回転・ミラー重複解を補正
  #  -GPUバージョン(execSolutionsKernel)のCPU移植版（デバッグ・逐次確認にも活用）
  # @note
  #  -N-Queens最適化アルゴリズムの核心部
  #  -temp_counterは再帰呼び出しで合計を受け渡し
  #  -実運用時は、より多くの分岐パターンを組み合わせることで最大速度を発揮
  def exec_solutions(self,constellations:List[Dict[str,int]],N:int)->None:
    # jmark=j=k=l=ijkl=ld=rd=col=start_ijkl=start=free=LD=endmark=mark1=mark2=0
    N2:int=N-2
    small_mask=(1<<(N2))-1
    temp_counter=[0]
    cnt=0
    board_mask:int=(1<<(N-1))-1
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
      #
      # uniqueを出そうとしたけどboardを使っていないからかな、難しいみたい
      #
      # 各コンステレーションのソリューション数を更新
      # count2,count4,count8=0,0,0
      # s:int=self.symmetry(ijkl,N)
      # u:int=cnt
      # if s==2: count2 += u
      # elif s==4: count4 += u
      # else: count8 += u
      # constellation["solutions"]=count2*2+count4*4+count8*8
      # constellation["unique"]   =count2+count4+count8
      constellation["solutions"]=cnt * self.symmetry(ijkl,N)
  # 開始コンステレーション（部分盤面配置パターン）の列挙・重複排除を行う関数
  # @param ijklList        uniqueな部分盤面signature（ijkl値）の格納先HashSet
  # @param constellations  Constellation本体リスト（実際の盤面は後続で生成）
  # @param N               盤面サイズ
  # @details
  #  -コーナー・エッジ・対角・回転対称性を考慮し、「代表解」となるuniqueな開始盤面のみ抽出する。
  #  -forループの入れ子により、N-Queens盤面の「最小単位部分盤面」を厳密な順序で列挙。
  #  -k, l, i, j 各インデックスの取り方・範囲・重複排除のための判定ロジックが最適化されている。
  #  -checkRotations()で既出盤面（回転対称）を排除、必要なものだけをijklListに追加。
  #  -このunique setをもとに、後段でConstellation構造体の生成・分割探索を展開可能。
  # @note
  #  -「部分盤面分割＋代表解のみ探索」戦略は大規模Nの高速化の要！
  #  -このループ構造・排除ロジックがN-Queensソルバの根幹。
  def gen_constellations(self,ijkl_list:Set[int],constellations:List[Dict[str,int]],N:int,preset_queens:int)->None:
    halfN=(N+1)//2  # Nの半分を切り上げ
    # --- [Opt-03] 中央列特別処理（奇数Nの場合のみ） ---
    if N % 2==1:
      center = N // 2
      ijkl_list.update(
        self.to_ijkl(i, j, center, l)
        for l in range(center + 1, N-1)
        for i in range(center + 1, N-1)
        if i != (N-1)-l
        for j in range(N-center-2, 0, -1)
        if j != i and j != l
        if not self.check_rotations(ijkl_list, i, j, center, l, N)
        # 180°回転盤面がセットに含まれていない
        if not self.rot180_in_set(ijkl_list, i, j, center, l, N)
      )
    # --- [Opt-03] 中央列特別処理（奇数Nの場合のみ） ---

    # コーナーにクイーンがいない場合の開始コンステレーション
    # ijkl_list.update(self.to_ijkl(i,j,k,l) for k in range(1,halfN) for l in range(k+1,N-1) for i in range(k+1,N-1) if i != (N-1)-l for j in range(N-k-2,0,-1) if j!=i and j!=l if not self.check_rotations(ijkl_list,i,j,k,l,N))
    # コーナーにクイーンがいない場合の開始コンステレーション
    ijkl_list.update(
        self.to_ijkl(i, j, k, l)
        for k in range(1, halfN)
        for l in range(k + 1, N - 1)
        for i in range(k + 1, N - 1)
        if i != (N - 1) - l
        for j in range(N - k - 2, 0, -1)
        if j != i and j != l
        if not self.check_rotations(ijkl_list, i, j, k, l, N)
    )
    #
    # コーナーにクイーンがある場合の開始コンステレーション
    # ijkl_list.update({self.to_ijkl(0,j,0,l) for j in range(1,N-2) for l in range(j+1,N-1)})
    # は {...} で一時 set を作っていますが、以下のように generator にすればメモリ節約・速度向上します：
    ijkl_list.update(
        self.to_ijkl(0, j, 0, l)
        for j in range(1, N - 2)
        for l in range(j + 1, N - 1)
    )
    #
    # Jasmin変換
    # ijkl_list_jasmin = {self.jasmin(c, N) for c in ijkl_list}
    # ijkl_list_jasmin = {self.get_jasmin(c, N) for c in ijkl_list}
    # ijkl_list_jasmin={self.get_jasmin(c, N) for c in ijkl_list}
    # ijkl_list=ijkl_list_jasmin
    ijkl_list={self.get_jasmin(c, N) for c in ijkl_list}
    #
    #
    #
    #
    L=1<<(N-1)  # Lは左端に1を立てる
    for sc in ijkl_list:
      i,j,k,l=self.geti(sc),self.getj(sc),self.getk(sc),self.getl(sc)
      # すべての「右辺のシフト値」が負にならないよう max(x, 0) でガード
      # ld,rd,col=(L>>(i-1))|(1<<(N-k)),(L>>(i+1))|(1<<(l-1)),1|L|(L>>i)|(L>>j)
      ld,rd,col=(L>>max(i-1,0))|(1<<max(N-k,0)),(L>>max(i+1,0))|(1<<max(l-1,0)),1|L|(L>>i)|(L>>j)
      LD,RD=(L>>j)|(L>>l),(L>>j)|(1<<k)
      counter=[0] # サブコンステレーションを生成
      #-------------------------
      # [Opt-09] 状態ハッシュによる探索枝の枝刈り
      visited:set[int]=set()
      #-------------------------
      #
      #-------------------------
      # 2. サブコンステレーション生成にtuple keyでキャッシュ
      #-------------------------
      # self.set_pre_queens(ld,rd,col,k,l,1,3 if j==N-1 else 4,LD,RD,counter,constellations,N,preset_queens,visited)
      self.set_pre_queens_cached(ld,rd,col,k,l,1,3 if j==N-1 else 4,LD,RD,counter,constellations,N,preset_queens,visited)
      current_size=len(constellations)
      # 生成されたサブコンステレーションにスタート情報を追加
      list(map(lambda target:target.__setitem__("startijkl",target["startijkl"]|self.to_ijkl(i,j,k,l)),(constellations[current_size-a-1] for a in range(counter[0]))))
  #
  # 関数プロトタイプ
  #
  def SQd0B(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    if row==endmark:
      return 1
    total:int=0
    N1:int=N-1
    rowstep:int=1+row
    while free:
      bit:int=free&-free  # 最下位ビットを取得
      free&=free-1  # 使用済みビットを削除
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      # 明らかにより細かい pruning を行っている高度な最適化条件です
      # しかし後から見返したとき、「なぜこれだけ複雑にしているのか？
      # 」という疑問が出る可能性があります。
      # if next_free and (row>=endmark-1 or mask&~(next_ld|next_rd|next_col)):
      # これは「次の行（rowstep+1）にクイーンを置けるマスが1つもな
      # ければ prune する」という強い pruning 条件であり、簡易条件
      # との重複はなく、むしろ発展系です。 
      # if next_free and not (
      #     rowstep < endmark and
      #     (
      #         mask & ~(
      #             (next_ld << 1) | (next_rd >> 1) | next_col
      #         )
      #         & ~((rowstep == mark1) << (N1 - mark1))
      #         & ~((rowstep == mark2) << (N1 - mark2))
      #     ) == 0
      # ):
      #
      # 上記を包括表記で記述したもの
      #
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd0B(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQd0BkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N3:int=N-3
    total:int=0
    N1:int=N-1
    rowstep:int=2+row
    while row==mark1 and free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<2,(rd|bit)>>2,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col|1<<N3)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd0B(next_ld,next_rd|1<<N3,next_col,row+2,next_free,jmark,endmark,mark1,mark2,mask,N)
    rowstep:int=1+row
    while free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd0BkB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQd1BklB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    N4:int=N-4
    total:int=0
    rowstep:int=3+row
    while row==mark1 and free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<3,(rd|bit)>>3,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col|1|1<<N4)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd1B(next_ld|1,next_rd|1<<N4,next_col,row+3,next_free,jmark,endmark,mark1,mark2,mask,N)
    rowstep:int=1+row
    while free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd1BklB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQd1B(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    if row==endmark:
      return 1
    total:int=0
    N1:int=N-1
    rowstep:int=1+row
    while free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      # if next_free and (row+1>=endmark or~((next_ld<<1)|(next_rd>>1)|next_col)>0):
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd1B(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQd1BkBlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    N3:int=N-3
    total:int=0
    rowstep:int=2+row
    while row==mark1 and free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<2,(rd|bit)>>2,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col|1<<N3)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd1BlB(next_ld,next_rd|1<<N3,next_col,row+2,next_free,jmark,endmark,mark1,mark2,mask,N)
    rowstep:int=1+row
    while free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd1BkBlB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQd1BlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    total:int=0
    N1:int=N-1
    rowstep:int=2+row
    while row==mark2 and free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<2|1,(rd|bit)>>2,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      # if next_free and (row+2>=endmark or~(next_ld|next_rd|next_col)):
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd1B(next_ld,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,mask,N)
    rowstep:int=1+row
    while free: # General case when row !=mark2
      bit:int=free&-free # Extract the rightmost available position
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd1BlB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQd1BlkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    N3:int=N-3  # Precomputed value for performance
    total:int=0
    rowstep:int=3+row
    while row==mark1 and free:
      bit:int=free&-free  # Extract the rightmost available position
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<3,(rd|bit)>>3,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col|2|1<<N3)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd1B(next_ld|2,next_rd|1<<N3,next_col,row+3,next_free,jmark,endmark,mark1,mark2,mask,N)
    rowstep:int=1+row
    while free:
      bit:int=free&-free  # Extract the rightmost available position
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd1BlkB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQd1BlBkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    total:int=0
    rowstep:int=2+row
    while row==mark1 and free:
      bit:int=free&-free  # Extract the rightmost available position
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<2,(rd|bit)>>2,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col|1)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd1BkB(next_ld|1,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,mask,N)
    rowstep:int=1+row
    while free:
      bit:int=free&-free  # Extract the rightmost available position
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd1BlBkB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQd1BkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    N3:int=N-3
    total:int=0
    rowstep:int=2+row
    while row==mark2 and free:
      bit:int=free&-free  # Extract the rightmost available position
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<2,(rd|bit)>>2,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col|1<<N3)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd1B(next_ld,next_rd|1<<N3,next_col,row+2,next_free,jmark,endmark,mark1,mark2,mask,N)
    rowstep:int=1+row
    while free:
      bit:int=free&-free  # Extract the rightmost available position
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd1BkB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQd2BlkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    N3:int=N-3
    total:int=0
    rowstep:int=3+row
    while row==mark1 and free:
      bit:int=free&-free  # 最下位ビットを取得
      free&=free-1  # 使用済みビットを削除
      next_ld,next_rd,next_col=(ld|bit)<<3,(rd|bit)>>3,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col|(1<<N3)|2)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd2B(next_ld|2,next_rd|1<<N3,next_col,row+3,next_free,jmark,endmark,mark1,mark2,mask,N)
    rowstep:int=1+row
    while free:
      bit:int=free&-free  # 最下位ビットを取得
      free&=free-1  # 使用済みビットを削除
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd2BlkB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQd2BklB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    N4:int=N-4
    total:int=0
    rowstep:int=3+row
    while row==mark1 and free:
      bit:int=free&-free  # 最下位のビットを取得
      free&=free-1  # 使用済みのビットを削除
      next_ld,next_rd,next_col=(ld|bit)<<3,(rd|bit)>>3,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col|(1<<N4)|1)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd2B(next_ld|1,next_rd|1<<N4,next_col,row+3,next_free,jmark,endmark,mark1,mark2,mask,N)
    rowstep:int=1+row
    while free:
      bit:int=free&-free  # 最下位のビットを取得
      free&=free-1  # 使用済みのビットを削除
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd2BklB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQd2BkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    N3:int=N-3
    total:int=0
    rowstep:int=2+row
    while row==mark2 and free:
      bit:int=free&-free  # 最下位ビットを取得
      free&=free-1  # 使用済みビットを削除
      next_ld,next_rd,next_col=(ld|bit)<<2,(rd|bit)>>2,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col|1<<N3)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd2B(next_ld,next_rd|1<<N3,next_col,row+2,next_free,jmark,endmark,mark1,mark2,mask,N)
    rowstep:int=1+row
    while free:
      bit:int=free&-free  # 最下位ビットを取得
      free&=free-1  # 使用済みビットを削除
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd2BkB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQd2BlBkB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    total:int=0
    rowstep:int=2+row
    while row==mark1 and free:
      bit:int=free&-free  # Get the lowest bit
      free&=free-1  # Remove the lowest bit
      next_ld,next_rd,next_col=(ld|bit)<<2,(rd|bit)>>2,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col|1)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd2BkB(next_ld|1,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,mask,N)
    rowstep:int=1+row
    while free:
      bit:int=free&-free  # Get the lowest bit
      free&=free-1  # Remove the lowest bit
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd2BlBkB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQd2BlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    total:int=0
    rowstep:int=2+row
    while row==mark2 and free:
      bit:int=free&-free  # Get the lowest bit
      free&=free-1  # Remove the lowest bit
      next_ld,next_rd,next_col=(ld|bit)<<2,(rd|bit)>>2,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col|1)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd2B(next_ld|1,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,mask,N)
    rowstep:int=1+row
    while free:
      bit:int=free&-free  # Get the lowest bit
      free&=free-1  # Remove the lowest bit
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd2BlB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQd2BkBlB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    N3:int=N-3
    total:int=0
    rowstep:int=2+row
    while row==mark1 and free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<2,(rd|bit)>>2,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col|1<<N3)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd2BlB(next_ld,next_rd|1<<N3,next_col,row+2,next_free,jmark,endmark,mark1,mark2,mask,N)
    rowstep:int=1+row
    while free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd2BkBlB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQd2B(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    total:int=0
    if row==endmark:
      if (free&(~1))>0:
        return 1
    rowstep:int=1+row
    while free:
      bit:int=free&-free  # 最も下位の1ビットを取得
      free&=free-1  # 使用済みビットを削除
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      # if next_free and (row>=endmark-1 or~((next_ld<<1)|(next_rd>>1)|(next_col))>0):
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQd2B(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQBlBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    total:int=0
    rowstep:int=2+row
    while row==mark2 and free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<2,(rd|bit)>>2,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col|1)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQBjrB(next_ld|1,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,mask,N)
    rowstep:int=1+row
    while free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQBlBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQBkBlBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    N3:int=N-3
    total:int=0
    rowstep:int=2+row
    while row==mark1 and free:
      bit:int=free&-free  # Isolate the rightmost 1 bit.
      free&=free-1  # Remove the isolated bit from free.
      next_ld,next_rd,next_col=(ld|bit)<<2,(rd|bit)>>2,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col|1<<N3)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQBlBjrB(next_ld,next_rd|1<<N3,next_col,row+2,next_free,jmark,endmark,mark1,mark2,mask,N)
    rowstep:int=1+row
    while free:
      bit:int=free&-free  # Isolate the rightmost 1 bit.
      free&=free-1  # Remove the isolated bit from free.
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQBkBlBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    total:int=0
    rowstep:int=1+row
    if row==jmark:
      free&=~1  # Clear the least significant bit (mark position 0 unavailable).
      ld|=1  # Mark left diagonal as occupied for position 0.
      while free:
        bit:int=free&-free  # Get the lowest bit (first free position).
        free&=free-1  # Remove this position from the free positions.
        next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
        next_free:int=mask&~(next_ld|next_rd|next_col)
        if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
          total+=self.SQB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
      return total
    rowstep:int=1+row
    while free:
      bit:int=free&-free  # Get the lowest bit (first free position).
      free&=free-1  # Remove this position from the free positions.
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    total:int=0
    if row==endmark:
      return 1
    rowstep:int=1+row
    while free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      # if next_free and (row>=endmark-1 or~((next_ld<<1)|(next_rd>>1)|next_col)>0):
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQBlBkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    total:int=0
    rowstep:int=2+row
    while row==mark1 and free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<2,(rd|bit)>>2,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col|1)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQBkBjrB(next_ld|1,next_rd,next_col,row+2,next_free,jmark,endmark,mark1,mark2,mask,N)
    rowstep:int=1+row
    while free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQBlBkBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQBkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    N3:int=N-3
    total:int=0
    rowstep:int=2+row
    while row==mark2 and free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<2,(rd|bit)>>2,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col|1<<N3)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQBjrB(next_ld,next_rd|1<<N3,next_col,row+2,next_free,jmark,endmark,mark1,mark2,mask,N)
    rowstep:int=1+row
    while free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQBkBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQBklBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    N4:int=N-4
    total:int=0
    rowstep:int=3+row
    while row==mark1 and free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<3,(rd|bit)>>3,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col|1<<N4|1)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQBjrB(next_ld|1,next_rd|1<<N4,next_col,row+3,next_free,jmark,endmark,mark1,mark2,mask,N)
    rowstep:int=1+row
    while free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQBklBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQBlkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    N3:int=N-3
    total:int=0
    rowstep:int=3+row
    while row==mark1 and free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<3,(rd|bit)>>3,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col|1<<N3|2)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQBjrB(next_ld|2,next_rd|1<<N3,next_col,row+3,next_free,jmark,endmark,mark1,mark2,mask,N)
    rowstep:int=1+row
    while free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQBlkBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQBjlBkBlBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    total:int=0
    rowstep:int=0+row
    if row==N1-jmark:
      rd|=1<<N1
      next_ld,next_rd,next_col=ld<<1,rd>>1,col
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQBkBlBjrB(next_ld,next_rd,next_col,row,next_free,jmark,endmark,mark1,mark2,mask,N)
      return total
    rowstep:int=1+row
    while free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQBjlBkBlBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQBjlBlBkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    total:int=0
    rowstep:int=0+row
    if row==N1-jmark:
      rd|=1<<N1
      next_ld,next_rd,next_col=ld<<1,rd>>1,col
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQBlBkBjrB(next_ld,next_rd,next_col,row,next_free,jmark,endmark,mark1,mark2,mask,N)
      return total
    rowstep:int=1+row
    while free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQBjlBlBkBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQBjlBklBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    total:int=0
    rowstep:int=0+row
    if row==N1-jmark:
      rd|=1<<N1
      next_ld,next_rd,next_col=ld<<1,rd>>1,col
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQBklBjrB(next_ld,next_rd,next_col,row,next_free,jmark,endmark,mark1,mark2,mask,N)
      return total
    rowstep:int=1+row
    while free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQBjlBklBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
  def SQBjlBlkBjrB(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,mask:int,N:int)->int:
    N1:int=N-1
    total:int=0
    rowstep:int=0+row
    if row==N1-jmark:
      # rd|=1<<N1
      next_ld,next_rd,next_col=ld<<1,rd>>1,col
      next_free:int=mask&~(next_ld|next_rd|next_col)
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQBlkBjrB(next_ld,next_rd,next_col,row,next_free,jmark,endmark,mark1,mark2,mask,N)
      return total
    rowstep:int=1+row
    while free:
      bit:int=free&-free
      free&=free-1
      next_ld,next_rd,next_col=(ld|bit)<<1,(rd|bit)>>1,col|bit
      next_free:int=mask&~(next_ld|next_rd|next_col)
      # if next_free:
      #   if rowstep<endmark:
      #     blocked_next=(next_ld<<1)|(next_rd>>1)|next_col
      #     if rowstep==mark1:
      #       blocked_next&=~(1<<(N1-mark1))
      #     if rowstep==mark2:
      #       blocked_next&=~(1<<(N1-mark2))
      #     if (mask&~blocked_next)==0:
      #       continue
      if next_free and not (rowstep<endmark and (mask&~(((next_ld<<1)|(next_rd>>1)|next_col)&~(int(rowstep==mark1)<<(N1-mark1))&~(int(rowstep==mark2)<<(N1-mark2))))==0): 
        total+=self.SQBjlBlkBjrB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,mask,N)
    return total
class NQueens19_constellations():
  def main(self)->None:
    nmin:int=5
    nmax:int=20
    preset_queens:int=4  # 必要に応じて変更
    total:int=0
    unique:int=0
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax):
      start_time=datetime.now()
      ijkl_list:Set[int]=set()
      constellations:List[Dict[str,int]]=[]
      total=0
      unique=0
      NQ=NQueens19()
      #---------------------------------
      # 4. pickleファイルで星座リストそのものをキャッシュ
      #---------------------------------
      # キャッシュを使わない
      # NQ.gen_constellations(ijkl_list,constellations,size,preset_queens)
      # キャッシュを使う、キャッシュの整合性もチェック
      # -- txt
      # constellations = NQ.load_or_build_constellations_txt(ijkl_list,constellations, size, preset_queens)
      # -- bin
      constellations = NQ.load_or_build_constellations_bin(ijkl_list,constellations, size, preset_queens)
      #---------------------------------
      NQ.exec_solutions(constellations,size)
      # total:int=sum(c['solutions'] for c in constellations if c['solutions']>0)
      total = sum(c["solutions"] for c in constellations if c["solutions"] > 0)
      # unique = sum(c["unique"] for c in constellations if c["unique"] > 0)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      # print(f"{size:2d}:{total:13d}{unique:13d}{text:>20s}")
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")
if __name__=="__main__":
  NQueens19_constellations().main()
