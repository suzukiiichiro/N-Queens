#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
コンステレーション版 最適化　Ｎクイーン

# 実行結果
workspace#suzuki$ codon build -release 21Py_constellations_optimized_codon.py&&./21Py_constellations_optimized_codon
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

$ nvcc -O3 -arch=sm_61 -m64 -ptx -prec-div=false 04CUDA_Symmetry_BitBoard.cu&&POCL_DEBUG=all ./a.out -n ;
17:         95815104         11977939     000:00:00:00.26
18:        666090624         83263591     000:00:00:01.65

fedora$ codon build -release 21Py_constellations_optimized_codon.py&&./21Py_constellations_optimized_codon
 N:        Total       Unique        hh:mm:ss.ms
16:     14772512            0         0:00:00.485
17:     95815104            0         0:00:03.137
18:    666090624            0         0:00:24.510

# 兄統合版
fedora$ codon build -release 21Py_constellations_optimized_codon.20250909-1_suzuki.py&&./21Py_constellations_optimized_codon.20250909-1_suzuki
17:     95815104     12041945         0:00:04.748
"""

# 枝刈り 探索木の中で、明らかに解にならない経路（無駄な配置）を早期に打ち切る手法。 
#✅[Opt-00] 対称性分類（COUNT2/4/8)　get_symmetries() → len(set(...)) による分類処理
#✅[Opt-00] 事前配置（Constellation）：クイーンの一部を固定して部分問題に分割
#✅[Opt-00] 探索関数の多様化：多数の SQ* 関数で構築時制約を細かく制御
#✅[Opt-00] 最下位ビット抽出 (x&-x) による next-free 探索
#✅[Opt-00] ルールベース分岐とマーク制御 によるブランチ削減
#  [Opt-00] 探索ループの統一（コードサイズとJIT効率の最適化）
#  現在は分岐の数が非常に多く、SQd1BlB,SQd1BkBlB,... など細かな関数に分かれています。
#  関数の数が多いとJIT最適化が効きにくくなる傾向があります。これを以下のように高階関数にまとめることで効率を改善できます。
#  [Opt-00] Constellation生成の絞り込み
#  gen_constellations() で全ての ijkl 組み合わせを列挙していますが、既に jasmin() により標準化しているなら：
#  生成時点でミラー・回転の基本形判定
#  重複除去を強化
#  組み合わせ数削減
#  メモリ圧縮
#  [Opt-00] Codon対応
#  Codonは型制約が厳しいため、以下の対応が必要：
#  List[Dict[str,int]] → @codon.struct に明示的に変換
#  Set[int] の代わりに list+in に変更（Codonは現状 set() をサポートしない）
#  [Opt-00] 64bit unsigned int ビットボード化（bitarray不要）
#  現状 int を使っていますが、np.uint64 や ctypes.c_ulonglong を使うことで、
#  メモリ占有減
#  CPUキャッシュ効率向上
#  Codonでも u64 型で静的な高速化が可能

# 基礎と対称性処理（最優先）
#✅[Opt-01] ビット演算による衝突枝刈り（cols/hills/dales） free=mask&~(left|bit<<1,right|bit>>1,col|bit)で探索
#  [Opt-02] Symmetry Breaking by Construction（構築時対称性除去） 
#  探索を始める前に以下のような除外を行う： 
#  1行目にクイーンを N // 2 より右に置かない。 
#  1行目が中央にある場合は、2行目の一部の位置のみ探索。 これにより、左右・上下対称な冗長解の大半をスキップできます。
#✅[Opt-02-1] 左右対称性除去（1 行目の列を 0～n//2−1 に制限）ミラー：1行目のクイーンを左半分に制限（ミラー対称を排除） 
#✅[Opt-02-2] 中央列の特別処理（N 奇数時） if n % 2==1: ブロックで中央列だけ個別に処理
#    中央列対応：奇数サイズで中央列に特別対応（COUNT2分類を正確に）
"""
    # 1行目は左半分に限定
    if row==0:
        limit=n // 2
        for col in range(limit):
    :
    :
    # 奇数サイズで中央列を個別処理
    if n % 2==1:
        col=n // 2
        ...
"""
#  [Opt-03] 10.Knuth法の初手以降の枝刈り強化 
#  Knuth法の基本は「初手の置き場所により対称性を徹底的に解除し、重複計算を削減」。 
#  さらに「2行目以降の探索位置にも対称性制限を厳格に適用」できることが多い。 
#  具体例： 2行目以降で初手対称による対応箇所を事前に計算し、可能配置を大幅に絞る。
#  回転・反転の対称チェックを局所化し、早期判定で枝刈り。 
#  全探索時に「探索枝ごとの重み付け（解の出現頻度）を事前計算」し加算効率化。 
#  Knuthの論文やソースコード（「Dancing Links」など）を参考にするのがベスト。
#  [Opt-  ] 「ミラー＋90度回転」で重複する形の探索枝除去
#    ミラー後90度回転して得られる配置が 元の形と一致する場合、構築時から除去可能
#    特に n=6,8,10 などの偶数盤面で顕著
#  [Opt-  ] 他の枝刈り（例えば3行目以降の枝刈り） 
#  3行目以降の枝刈りは「局所制約伝播の強化」「高度なヒューリスティック」を用いる。 
#  具体例： 行ごとに「許容される列の範囲を狭める」前処理。 「クイーンの影響度分布から到達不可能な領域を切り捨てる」技術。
#  「探索途中で解が不可能と判断した場合の早期リターン」。 こうした枝刈りは実装が複雑になるが、特に大きいNほど効果絶大。


#  [Opt-04] 180度対象除去 if row==n-1 and is_corner: により (n-1,n-1) を禁止
#   5. 斜め対称（回転180度）除去
#   角に置く場合でも、回転180度対称解が自明に重複するケースを除去できます。row=0,col=0 に置いたとき、row=n-1,col=n-1 への配置を許すと、180度回転で重複。よって、回転対称な構造を事前に構築しないように限定的に探索を行う。
"""
    mask=(1<<N)-1
    free=mask&~(cols|hills|dales)
    # 角スタート時の180度回転対称を除去：末行の右下 (n-1,n-1) を禁止
    if row==n-1 and is_corner:
      free&=~(1<<(N-1))
"""
#  [Opt-05] 角位置（col==0）での分岐（COUNT2 偏重の明示化） is_corner=True による分離集計で COUNT2 対応
#  1行目の**角（col:0 列0）**にクイーンを置くか否かの分離処理
#  🔸1行目の角（col:0 列0）にクイーンを置いた場合： ミラー・回転によって生じる COUNT2（対称解が多い）パターン が含まれる
#  🔸角に置かない場合： 通常は COUNT4 や COUNT8 の探索へと導かれやすい
# 「1行目の角（col:0 列0）にクイーンを置いた場合を別処理で分離する戦略的枝刈り」は、Knuth も推奨している有効な最適化です。
#
#  9. 1行目以外でも部分対称除去（行列単位）
# 構築途中（例：2〜n-1行）でも、回転・ミラーで過去の構成と一致する盤面が出てくる場合は prune 可能
"""
    if row==0:
        # まず列0に置くケース（角）だけを別途処理
        col=0
        bit=1<<col
        pos=row * n+col
        :
        :
        # 次に、1〜n//2-1の左半分（角以外）を処理（通常通り）
        for col in range(1,n // 2):
            bit=1<<col
            pos=row * n+col
        :
        :
        # 奇数Nで中央列を追加処理（is_corner=Falseで良い）
        if n % 2==1:
            col=n // 2
            bit=1<<col
    else:
        free=~(cols|hills|dales)&((1<<n)-1)
        while free:
            bit=free&-free
            free ^=bit
"""
#  [Opt-07] 1 行目以外でも部分対称除去（行列単位） if row==1 and c > queens[0]: continue 等で制御
#    途中段階（深さ r の盤面）を都度「辞書順最小」の canonical かどうかチェックして、そうでなければ枝刈り。
#    試験的に実装済み→正解数が減る問題あり
#  [Opt-11] 「ミラー＋90°回転」による“構築時”の重複複除 （構築時にこの対称性を安全に検出するのは難しく、導入していない）
#  * 完成後の対称判定より、構築途中で 8 対称性を逐次判定するのはコスト高実装が煩雑。
#    部分盤面を8通り生成するコストが高く、B系の“軽量 canonical 部分盤面判定”＋Zobrist の方がバランスが良いことが多いです。 


#✅[Opt-06] 並列処理（初手 col ごとに multiprocessing で分割）
"""
    @par
"""
#  [Opt-  ] 非再帰化（ループベース探索）

# 基本高速化・探索効率の向上
#  [Opt-08] is_canonical() による“部分盤面”の辞書順最小チェックを高速化（キャッシュ/軽量版）（
#  * 「完成盤」だけでなく“部分盤面”用に軽量な変換（行の回転・反転は途中情報だけで可）を実装。
"""
    def is_partial_duplicate(board: int,row: int) -> bool:
        # 部分盤面（row行まで）での対称性重複チェック
        partial_bits=row * n
        partial_board=board&((1<<partial_bits)-1)
        sym=get_partial_symmetries(partial_board,row)
        hashes=[hash_board(s,partial_bits) for s in sym]
        canonical=min(hashes)
        if canonical in partial_seen:
            return True
        partial_seen.add(canonical)
        return False
"""
#  [Opt-09] Zobrist Hash による transposition / visited 状態の高速検出 crc32() により軽量ハッシュで対称性判定を高速化
#  * N-Queens では完全一致局面の再訪は少ないですが、「部分対称 canonical チェックの結果」をハッシュ化して
#    メモ化する用途で効果（計算の再実行を削減）。
#  8. lookup table（Zobrist Hash）による重複回避
# bitarray.to01() は重い（文字列化） → zlib.crc32() などで軽量ハッシュ照合
# 対称形を辞書順最小にせずに、ハッシュで管理すると高速になる
# get_symmetries() によって生成される 8 通りの board（int）に対して、
# zlib.crc32() を用いて軽量ハッシュを取り、
# その中で 最小のハッシュ値を canonical として seen_hashes セットで判定します
"""
    def hash_board(board: int) -> int:
        byte_len=(n * n+7) // 8
        return zlib.crc32(board.to_bytes(byte_len,byteorder='big'))
    def classify_symmetry(board: int,n: int,seen_hashes: set[int]) -> str:
        sym=get_symmetries(board,n)
        hashes=[hash_board(s) for s in sym]
        canonical_hash=min(hashes)
        if canonical_hash in seen_hashes:
            return ""
        seen_hashes.add(canonical_hash)
        distinct=len(set(hashes))
        if distinct==8:
            return 'COUNT8'
        elif distinct==4:
            return 'COUNT4'
        else:
            return 'COUNT2'
"""
#  [Opt-10] マクロチェス（局所パターン）による構築制限 free==0 で安全に pruning（bin(cols) 判定は回避）
#  * 現実装との整合や有効なパターン定義次第で効果差が大。ルール設計が難しい。 
#    序盤の配置（例：1行目＋2行目）により、3行目以降のクイーン配置が詰まるパターン
#    特に cols|hills|dales が過半数を占めていると、有効配置がないことが多い
#    よって、2行目終了時点で pruning 条件を加えることで、無駄な探索を打ち切れる
#    1行目と2行目の特定パターンは、後の行にクイーンが置けなくなる 
#    例：n=8で [0,1] という開始配置（左上2マス）はその後に配置不能 
"""
    # バックトラックに以下を追加
    def backtrack(row=0,cols=0,hills=0,dales=0,board=0,is_corner=False):
      # 次の行に置ける場所がまったくないなら打ち切る（安全な pruning）
      # if row >=2:
      #   if bin(cols).count("1") > n // 2:
      #     return
      # または
      # 安全な pruning（次の行にクイーンを置ける場所がない）
      free=~(cols|hills|dales)&((1<<n)-1)
      if row >=2 and free==0:
        return
      #
      # 回転180度対称の重複除去（角スタート時のみ）
      if row==n-1 and is_corner:
        free&=~(1<<(n-1))

      while free:
        ...
"""
#  [Opt-14] ポピュレーションカウント命令（popcount）による列候補の高速抽出
#  ビット演算で生成される「配置可能な列マスク」から、候補位置を列挙する際に、
#  __builtin_ctz,__builtin_popcount（GCC/Clang）や _BitScanForward（MSVC）などの命令で 
#  次に置く列の最下位ビットを高速抽出 すると、ループ不要で高速化が図れます。
#  Pythonでも bin(mask).count('1') や (mask&-mask).bit_length()-1 を活用できますが、C/C++ほどの速度は得られません。


#✅[Opt-12] ビット演算のインライン化
#  [Opt-13] 部分解合成法 solve(col,...) を col ごとに展開
#  [Opt-15] メモリ局所性と非再帰化（ループベースバックトラッキング）
#  再帰よりもループ構造で状態スタックを管理した方が、関数呼び出しのオーバーヘッドを回避でき、
#  キャッシュ局所性も改善されて高速化します。
#  実装例として、配列で状態を管理し while ベースで探索を行う方法があります（「スタックバックトラッキング法」）。
#  [Opt-16] OpenMPやSIMD命令を用いたC/C++でのさらなる並列化
#  C/C++では #pragma omp parallel for などを使って、ループ単位で並列処理が可能です。
#  さらに、複数の盤面に対して一括でbit演算処理を行い、SIMD命令（AVX,SSE） による並列化でスループットを上げられる余地があります。
#  [Opt-17] 解の出現頻度・分布の統計解析を活用したヒューリスティクス
#  Nが大きくなると、探索空間のどの部分で解が見つかりやすいかという傾向があります。
#  統計的に解の出やすい位置を優先するように探索順序を並べ替えると、最初の解の発見や早期枝刈りが促進されます。
#  [Opt-18] SATソルバーなど汎用制約ソルバーとの比較研究
#  ビット演算による専用ソルバーが最速ですが、Z3やMiniSATなど制約ソルバーに問題を変換することで、
#  「汎用手法でどこまでできるか」という観点も参考になります。
#  特に制約プログラミング（CP-SAT）などとの比較は面白い分析材料になります。
#  [Opt-19] 結果の検証とユニーク解の正規化
#  回転・反転後に「基本形」として一意に正規化する実装が必要ですが、ここでの実装ミスは 
#  ユニーク解とTotalのカウントミスを引き起こしやすいため注意。
#  各解を「最小のビット列に変換してハッシュ」などの方式で厳密に一意性を検証するとよいです。



# キャッシュ・メモリ管理・最適化
#✅[Opt-21] 盤面・星座の“一意シグネチャ”をZobrist hashやtupleで管理
#  今はijkl_listがSet[int]（16bit packedの盤面ID）ですが、
#  「星座の状態→Zobrist hash or tuple」も併用可能
#  （星座構造が大きくなったり部分一致チェックが多いとき特に有効）
#✅[Opt-22] 盤面や星座の辞書キャッシュ（dict）による一意管理
#   星座リストや部分盤面ごとに、「一度作ったものはdictでキャッシュ」
#✅[Opt-23] Jasmin変換のキャッシュ化（生成済み盤面の再利用）【済】
#   ijkl_list_jasmin={self.jasmin(c,N) for c in ijkl_list}
#   も、盤面→jasmin変換は「一度計算したらdictでキャッシュ」が効果大
#✅[Opt-24] 星座ごとに「hash/tuple key」を使ったキャッシュ辞書の導入
#   set_pre_queensやサブコンステレーション生成時も「(ld,rd,col,...)のtuple」や「部分盤面hash」をkeyに
#   一度作った星座はdictから即座に再利用できる構造
#✅[Opt-25] 星座生成全体をpickleなどで「Nごとにファイル化」して超巨大N対応
#   すでに解説済ですが、gen_constellationsの全出力をconstellations_N17.pklのようなファイルでキャッシュ
#   実行時にRAM展開し、毎回再生成を回避
#   ファイルI/O最小化も同時に達成

# バックトラック関数の最適化と枝狩り
# mark1 / mark2 の考慮
# mark1 や mark2（星座構成のための特定行）は、後から強制的に配置することが決まっているクイーンの位置です。
# その位置のビットは「ブロック」から除外（＝空きとして扱う）べきなので、~(1<<…) で解除します。
# 次の行が完全にブロックされているなら、その選択肢（現在の bit による配置）は探索する意味がない。
# つまり、次の row+1 にクイーンを置ける列が「ひとつも無い」場合、早期に continue。
# これは free ではなく、next_ld / next_rd / next_col による影響を見ているのがポイント。
# 効果
# この最適化により、再帰的なバックトラックの深さを減らせるケースが増え、特に解が少ない構成や途中で詰む分岐が多い盤面で効果が高く出ます。





#!/usr/bin/env python3

import os
from typing import List,Set,Dict
from datetime import datetime
from typing import Dict as TDict,List as TList
# pypyを使うときは以下を活かしてcodon部分をコメントアウト
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')

class NQueens21:

  def __init__():
    self._rot_cache={}

  def rot90(self,ijkl:int,N:int)->int:
    # 時計回りに90度回転
    # rot90 メソッドは、90度の右回転（時計回り）を行います
    # 元の位置 (row,col) が、回転後の位置 (col,N-1-row) になります。
    return ((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl)

  def rot180_in_set(self,ijkl_list:Set[int],i:int,j:int,k:int,l:int,N:int)->bool:
    # 対称性のための計算と、ijklを扱うためのヘルパー関数。
    # 開始コンステレーションが回転90に対して対称である場合
    return self.rot180(self.to_ijkl(i,j,k,l),N) in ijkl_list

  def rot180(self,ijkl:int,N:int)->int:
    return ((N-1-self.getj(ijkl))<<15)+((N-1-self.geti(ijkl))<<10)+((N-1-self.getl(ijkl))<<5)+(N-1-self.getk(ijkl))

  def geti(self,ijkl:int)->int:
    # 盤面ユーティリティ群（ビットパック式盤面インデックス変換）
    # Python実装のgeti/getj/getk/getl/toijklに対応。
    # [i,j,k,l] 各クイーンの位置情報を5ビットずつ
    # 整数値（ijkl）にパック／アンパックするためのマクロ。
    # 15ビット～0ビットまでに [i|j|k|l] を格納する設計で、
    # constellationのsignatureや回転・ミラー等の盤面操作を高速化する。
    # 例：
    #  -geti(ijkl): 上位5ビット（15-19）からiインデックスを取り出す
    #  -toijkl(i,j,k,l): 各値を5ビット単位で連結し
    # 一意な整数値（signature）に変換
    # [注意] N≦32 まで対応可能
    return (ijkl>>15)&0x1F

  def getj(self,ijkl:int)->int:
    return (ijkl>>10)&0x1F

  def getk(self,ijkl:int)->int:
    return (ijkl>>5)&0x1F

  def getl(self,ijkl:int)->int:
    return ijkl&0x1F

  def getijkl(self,ijkl:int)->Tuple[int,int,int,int]:
    return (ijkl>>15)&0x1F,(ijkl>>10)&0x1F,(ijkl>>5)&0x1F,ijkl&0x1F

  def to_ijkl(self,i:int,j:int,k:int,l:int)->int:
    return (i<<15)+(j<<10)+(k<<5)+l

  def symmetry(self,ijkl:int,N:int)->int:
    # symmetry: 回転・ミラー対称性ごとの重複補正 (90度:2,180度:4,その他:8)
    return 2 if self.symmetry90(ijkl,N) else 4 if self.geti(ijkl)==N-1-self.getj(ijkl) and self.getk(ijkl)==N-1-self.getl(ijkl) else 8

  def symmetry90(self,ijkl:int,N:int)->bool:
    return ((self.geti(ijkl)<<15)+(self.getj(ijkl)<<10)+(self.getk(ijkl)<<5)+self.getl(ijkl))==(((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl))

  def check_rotations(self,ijkl_list:Set[int],i:int,j:int,k:int,l:int,N:int)->bool:
    # 指定した盤面 (i,j,k,l) を90度・180度・270度回転したいずれか
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
    # rot90=((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i
    # rot180=((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+(N-1-k)
    # rot270=(l<<15)+(k<<10)+((N-1-i)<<5)+(N-1-j)
    # return any(rot in ijkl_list for rot in (rot90,rot180,rot270))
      return any(rot in ijkl_list for rot in [((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i,((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+(N-1-k),(l<<15)+(k<<10)+((N-1-i)<<5)+(N-1-j)])

  jasmin_cache={}
  def get_jasmin(self,c: int,N: int) -> int:
    key=(c,N)
    if key in self.jasmin_cache: return self.jasmin_cache[key]
    result=self.jasmin(c,N)
    self.jasmin_cache[key]=result
    return result

  def jasmin(self,ijkl:int,N:int)->int:
    #--------------------------------------------
    # 1. Jasmin変換キャッシュを導入する
    # [Opt-08] キャッシュ付き jasmin() のラッパー
    # 使用例: 
    # ijkl_list_jasmin={self.get_jasmin(c,N) for c in ijkl_list}
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
    # 最初の最小値と引数を設定
    arg,min_val=0,min(self.getj(ijkl),N-1-self.getj(ijkl))
    # i: 最初の行（上端） 90度回転2回
    if min(self.geti(ijkl),N-1-self.geti(ijkl))<min_val:
      arg,min_val=2,min(self.geti(ijkl),N-1-self.geti(ijkl))
    # k: 最初の列（左端） 90度回転3回
    if min(self.getk(ijkl),N-1-self.getk(ijkl))<min_val:
      arg,min_val=3,min(self.getk(ijkl),N-1-self.getk(ijkl))
    # l: 最後の列（右端） 90度回転1回
    if min(self.getl(ijkl),N-1-self.getl(ijkl))<min_val:
      arg,min_val=1,min(self.getl(ijkl),N-1-self.getl(ijkl))
    # 90度回転を arg 回繰り返す
    for _ in range(arg): ijkl=self.rot90(ijkl,N)
    # 必要に応じて垂直方向のミラーリングを実行
    if self.getj(ijkl)<N-1-self.getj(ijkl): ijkl=self.to_ijkl(N-1-self.geti(ijkl),N-1-self.getj(ijkl),self.getl(ijkl),self.getk(ijkl))
    return ijkl

  def file_exists(self,fname: str) -> bool:
    #---------------------------------
    # 4. ファイルで星座リストそのものをキャッシュ
    #---------------------------------
    try:
      with open(fname,"rb"):
        return True
    except:
      return False

  def validate_constellation_list(self,constellations: List[Dict[str,int]]) -> bool:
    # バリデーション関数の強化（既に実装済みの場合はスキップOK）
    return all(all(k in c for k in ("ld","rd","col","startijkl")) for c in constellations)

  def load_or_build_constellations_txt(self,ijkl_list: Set[int],constellations,N: int,preset_queens: int) -> List[Dict[str,int]]:
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
    # キャッシュ付きラッパー関数（.txt）
    # N と preset_queens に基づいて一意のファイル名を構成
    fname=f"constellations_N{N}_{preset_queens}.txt"
    # ファイルが存在すれば即読み込み
    # if self.file_exists(fname):
    #     return self.load_constellations_txt(fname)
    # ファイルが存在すれば読み込むが、破損チェックも行う
    if self.file_exists(fname):
      try:
        constellations=self.load_constellations_txt(fname)
        if self.validate_constellation_list(constellations): return constellations
        else: print(f"[警告] 不正なキャッシュ形式: {fname} を再生成します")
      except Exception as e:
        print(f"[警告] キャッシュ読み込み失敗: {fname},理由: {e}")
    # ファイルがなければ生成・保存
    # gen_constellations() により星座を生成
    # save_constellations_txt() でファイルに保存
    # 返り値として constellations リストを返す
    constellations: List[Dict[str,int]]=[]
    self.gen_constellations(ijkl_list,constellations,N,preset_queens)
    self.save_constellations_txt(fname,constellations)
    return constellations

  def save_constellations_txt(self,path: str,constellations: List[Dict[str,int]]) -> None:
    #------------------------------------------
    # txt
    #------------------------------------------
    # テキスト形式で保存（1行=5整数: ld rd col startijkl solutions）
    with open(path,"w") as f:
      for c in constellations:
        ld,rd,col=c["ld"],c["rd"],c["col"]
        startijkl=c["startijkl"]
        solutions=c.get("solutions",0)
        f.write(f"{ld} {rd} {col} {startijkl} {solutions}\n")

  def load_constellations_txt(self,path: str) -> List[Dict[str,int]]:
    # --- テキスト形式でロード ---
    out: List[Dict[str,int]]=[]
    with open(path,"r") as f:
      for line in f:
        parts=line.strip().split()
        if len(parts) !=5: continue
        ld,rd,col=int(parts[0]),int(parts[1]),int(parts[2])
        startijkl=int(parts[3])
        solutions=int(parts[4])
        out.append({"ld": ld,"rd": rd,"col": col,"startijkl": startijkl,"solutions": solutions})
    return out

  def read_uint32_le(self,b: str) -> int:
    #------------------------------------------
    # bin
    #------------------------------------------
    # Codon互換の from_bytes() 相当処理
    return (ord(b[0])&0xFF)|((ord(b[1])&0xFF)<<8)|((ord(b[2])&0xFF)<<16)|((ord(b[3])&0xFF)<<24)

  def int_to_le_bytes(self,x: int) -> List[int]:
    # int_to_le_bytes ヘルパー関数を定義 以下のような関数を使って int を4バイトのリトルエンディアン形式に変換できます：
    return [(x >> (8 * i))&0xFF for i in range(4)]

  def validate_bin_file(self,fname: str) -> bool:
    # .bin ファイルサイズチェック（1件=16バイト→行数=ilesize // 16）
   try:
     with open(fname,"rb") as f:
       f.seek(0,2)  # ファイル末尾に移動
       size=f.tell()
     return size % 16==0
   except:
     return False

  def load_or_build_constellations_bin(self,ijkl_list: Set[int],constellations,N: int,preset_queens: int) -> List[Dict[str,int]]:
    # キャッシュ付きラッパー関数（.bin）
    fname=f"constellations_N{N}_{preset_queens}.bin"
    if self.file_exists(fname):
      try:
        constellations=self.load_constellations_bin(fname)
        if self.validate_bin_file(fname) and self.validate_constellation_list(constellations): return constellations
        else: print(f"[警告] 不正なキャッシュ形式: {fname} を再生成します")
      except Exception as e:
        print(f"[警告] キャッシュ読み込み失敗: {fname},理由: {e}")
    constellations: List[Dict[str,int]]=[]
    self.gen_constellations(ijkl_list,constellations,N,preset_queens)
    self.save_constellations_bin(fname,constellations)
    return constellations

  def save_constellations_bin(self,fname: str,constellations: List[Dict[str,int]]) -> None:
    # bin形式で保存
    with open(fname,"wb") as f:
      for d in constellations:
        for key in ["ld","rd","col","startijkl"]:
          b=self.int_to_le_bytes(d[key])
          # Codonでは str がバイト文字列扱い
          f.write("".join(chr(c) for c in b))

  def load_constellations_bin(self,fname: str)->List[Dict[str,int]]:
    # 読み込み関数（.binロード）
    constellations: List[Dict[str,int]]=[]
    with open(fname,"rb") as f:
      while True:
        raw=f.read(16)
        if len(raw)<16: break
        ld,rd,col=self.read_uint32_le(raw[0:4]),self.read_uint32_le(raw[4:8]),self.read_uint32_le(raw[8:12])
        startijkl=self.read_uint32_le(raw[12:16])
        constellations.append({
          "ld": ld,"rd": rd,"col": col,
          "startijkl": startijkl,"solutions": 0
        })
    return constellations

  subconst_cache={}
  def set_pre_queens_cached(self,ld: int,rd: int,col: int,k: int,l: int,row: int,queens: int,LD: int,RD: int,constellations: List[Dict[str,int]],N: int,preset_queens: int,visited:set[int]) -> None:
    #-------------------------
    # 2. サブコンステレーション生成にtuple keyでキャッシュ
    # gen_constellations で set_pre_queens を呼ぶ箇所を set_pre_queens_cached に変えるだけ！
    #---------------------------------
    key=(ld,rd,col,k,l,row,queens,LD,RD,N,preset_queens)
    # キャッシュの本体をdictかsetでグローバル/クラス変数に
    if not hasattr(self,"subconst_cache"): self.subconst_cache={}
    subconst_cache=self.subconst_cache
    # 以前に同じ状態で生成済み → 何もしない（または再利用）
    if key in subconst_cache: return
    # 新規実行（従来通りset_pre_queensの本体処理へ）
    # self.set_pre_queens(ld,rd,col,k,l,row,queens,LD,RD,counter,constellations,N,preset_queens,visited)
    self.set_pre_queens(ld,rd,col,k,l,row,queens,LD,RD,constellations,N,preset_queens,visited)
    subconst_cache[key]=True  # マークだけでOK

  def state_hash(self,ld: int,rd: int,col: int,row: int) -> int:
    #---------------------------------
    # [Opt-09] Zobrist Hash（Opt-09）の導入とその用途
    # ビットボード設計でも、「盤面のハッシュ」→「探索済みフラグ」で枝刈りは可能です。
    #---------------------------------
    if None in (ld,rd,col,row): return -1
    # 64ビット整数に収まるようにビット操作で圧縮
    # 単純な状態ハッシュ（高速かつ衝突率低めなら何でも可）
    return (ld * 0x9e3779b9) ^ (rd * 0x7f4a7c13) ^ (col * 0x6a5d39e9) ^ row

  constellation_signatures=set()
  def set_pre_queens(self,ld:int,rd:int,col:int,k:int,l:int,row:int,queens:int,LD:int,RD:int,constellations:List[Dict[str,int]],N:int,preset_queens:int,visited:set[int])->None:
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
    mask=(1<<N)-1  # setPreQueensで使用
    # ----------------------------
    # [Opt-09] 状態ハッシュによる探索枝の枝刈り
    # バックトラック系の冒頭に追加　やりすぎると解が合わない
    h: int=self.state_hash(ld,rd,col,row)
    if h in visited: return
    visited.add(h)
    # ----------------------------
    # k行とl行はスキップ
    if row==k or row==l:
      # cachedしない
      # self.set_pre_queens(ld<<1,rd>>1,col,k,l,row+1,queens,LD,RD,constellations,N,preset_queens,visited)
      # cachedする
      self.set_pre_queens_cached(ld<<1,rd>>1,col,k,l,row+1,queens,LD,RD,constellations,N,preset_queens,visited)
      return
    # クイーンの数がpreset_queensに達した場合、現在の状態を保存
    # ------------------------------------------------
    # 3. 星座のsignature重複防止
    # ------------------------------------------------
    if queens==preset_queens:
      # constellation_signatures セットの初期化（Codon対応）
      if not hasattr(self,"constellation_signatures"):
        self.constellation_signatures=set()
      # signature の生成
      signature=(ld,rd,col,k,l,row)
      # 初回の signature のみ追加
      if signature not in self.constellation_signatures:
        constellations.append({
          "ld": ld,"rd": rd,"col": col,
          # 既存の startijkl は互換のため残す（row を 20bit 左に入れている）
          "startijkl": row<<20,
          # ここから新規に保存（行依存制約を再現するために必要）
          "row": row,"k": k,"l": l,"LD": LD,"RD": RD,
          "solutions": 0
        })
        self.constellation_signatures.add(signature)
      return
    # ------------------------------------------------
    # 現在の行にクイーンを配置できる位置を計算
    free=mask&~(ld|rd|col|(LD>>(N-1-row))|(RD<<(N-1-row)))
    while free:
      bit:int=free&-free
      free&=free-1
      self.set_pre_queens_cached((ld|bit)<<1,(rd|bit)>>1,col|bit,k,l,row+1,queens+1,LD,RD,constellations,N,preset_queens,visited)

  def dfs(self,N:int,mask:int,ld:int,rd:int,col:int,row:int,need:int,k:int,l:int,LD:int,RD:int,idx:int)->int:
    # ConstellationArrayListの各Constellation（部分盤面）ごとに
    # N-Queens探索を分岐し、そのユニーク解数をsolutionsフィールドに記録する関数（CPU版）
    # @param constellations 解探索対象のConstellationArrayListポインタ
    # @param N              盤面サイズ
    # @details
    #  -各Constellation（部分盤面）ごとにj,k,l,各マスク値を展開し、
    #     複雑な分岐で最適な再帰ソルバー（SQ...関数群）を呼び出して解数を計算
    #  -分岐ロジックは、部分盤面・クイーンの位置・コーナーからの距離などで高速化
    #  -解数はtemp_counterに集約し、各Constellationのsolutionsフィールドに記録
    #  -symmetry(ijkl,N)で回転・ミラー重複解を補正
    #  -GPUバージョン(execSolutionsKernel)のCPU移植版（デバッグ・逐次確認にも活用）
    # @note
    #  -N-Queens最適化アルゴリズムの核心部
    #  -temp_counterは再帰呼び出しで合計を受け渡し
    #  -実運用時は、より多くの分岐パターンを組み合わせることで最大速度を発揮
    # mask:int=(1<<N)-1
    if need==0: return 1
    # k/l の行は“すでに置いてある行”なのでスキップ
    # if row==k or row==l:
    #     return self.dfs(N,mask,(ld<<1)&mask,rd >> 1,col,row+1,need,k,l,LD,RD,idx)
    # k/l 行は“すでに置いてある”のでループで一気にスキップ
    while row<N and (row==k or row==l): ld,rd,row=(ld<<1)&mask,rd>>1,row+1
    # 力尽きた
    if row>=N: return 0
    # 上限枝刈り：残りの可処理行が need に満たなければ打ち切り
    # （k/l がこの先に残っていれば、その分は実質スキップ行）
    remain=N-row
    if row<=k: remain-=1
    if row<=l: remain-=1
    if need>remain: return 0
    # この行で禁止される列（LD/RD は行ごとに動的）
    forbid:int=mask&(ld|rd|col|LD>>(N-1-row)|RD<<(N-1-row))
    avail:int=mask&~forbid
    if not avail: return 0
    total:int=0
    # 典型の LSB 展開：bit=avail&-avail; avail&=avail-1
    while avail:
        bit=-avail&avail
        avail&=avail-1
        total+=self.dfs(N,mask, (ld|bit)<<1, (rd|bit)>>1, col|bit, row+1, need-1, k,l,LD,RD,idx)
    return total

  def exec_solutions(self,constellations:List[Dict[str,int]],N:int):
    # 再実装（ズレ修正版）:
    #  set_pre_queens が保存した (ld,rd,col,start=row) をそのまま継続
    #  圧縮盤を使わず N ビットで DFS
    #  LD/RD を (L>>j)|(L>>l),(L>>j)|(1<<k) で復元し、各行で動的に適用
    mask:int=(1<<N)-1
    L:int=1<<(N-1)
    @par
    for idx,c in enumerate(constellations):
        s:int=int(c.get("startijkl",0))
        start:int=s>>20
        ijkl:int=s&((1<<20)-1)
        j,k,l=self.getj(ijkl),self.getk(ijkl),self.getl(ijkl)
        ld,rd,col=int(c.get("ld",0))&mask,int(c.get("rd",0))&mask,int(c.get("col",0))&mask
        placed:int=sum(1 for i in range(N) if ((col&mask) >> i)&1)
        need:int=N-sum(1 for i in range(N) if (col>>i)&1)
        # gen_constellations と同じ LD/RD を復元
        LD,RD=L>>j|L>>l,L>>j|1<<k
        unique_cnt:int=self.dfs(N,mask,ld,rd,col,start,need,k,l,LD,RD,idx)
        c["unique"]=unique_cnt
        c["solutions"]=unique_cnt*self.symmetry(ijkl,N)

  def gen_constellations(self,ijkl_list:Set[int],constellations:List[Dict[str,int]],N:int,preset_queens:int)->None:
    # 開始コンステレーション（部分盤面配置パターン）の列挙・重複排除を行う関数
    # @param ijklList        uniqueな部分盤面signature（ijkl値）の格納先HashSet
    # @param constellations  Constellation本体リスト（実際の盤面は後続で生成）
    # @param N               盤面サイズ
    # @details
    #  -コーナー・エッジ・対角・回転対称性を考慮し、「代表解」となるuniqueな開始盤面のみ抽出する。
    #  -forループの入れ子により、N-Queens盤面の「最小単位部分盤面」を厳密な順序で列挙。
    #  -k,l,i,j 各インデックスの取り方・範囲・重複排除のための判定ロジックが最適化されている。
    #  -checkRotations()で既出盤面（回転対称）を排除、必要なものだけをijklListに追加。
    #  -このunique setをもとに、後段でConstellation構造体の生成・分割探索を展開可能。
    # @note
    #  -「部分盤面分割＋代表解のみ探索」戦略は大規模Nの高速化の要！
    #  -このループ構造・排除ロジックがN-Queensソルバの根幹。
    halfN=(N+1)//2  # Nの半分を切り上げ
    # --- [Opt-03] 中央列特別処理（奇数Nの場合のみ） ---
    if N % 2==1:
      center=N // 2
      ijkl_list.update(
        self.to_ijkl(i,j,center,l)
        for l in range(center+1,N-1)
        for i in range(center+1,N-1)
        if i !=(N-1)-l
        for j in range(N-center-2,0,-1)
        if j !=i and j !=l
        if not self.check_rotations(ijkl_list,i,j,center,l,N)
        # 180°回転盤面がセットに含まれていない
        if not self.rot180_in_set(ijkl_list,i,j,center,l,N)
      )
    # --- [Opt-03] 中央列特別処理（奇数Nの場合のみ） ---

    # コーナーにクイーンがいない場合の開始コンステレーション
    # ijkl_list.update(self.to_ijkl(i,j,k,l) for k in range(1,halfN) for l in range(k+1,N-1) for i in range(k+1,N-1) if i !=(N-1)-l for j in range(N-k-2,0,-1) if j!=i and j!=l if not self.check_rotations(ijkl_list,i,j,k,l,N))
    # コーナーにクイーンがいない場合の開始コンステレーション
    ijkl_list.update(
      self.to_ijkl(i,j,k,l)
      for k in range(1,halfN)
      for l in range(k+1,N-1)
      for i in range(k+1,N-1)
      if i !=(N-1)-l
      for j in range(N-k-2,0,-1)
      if j !=i and j !=l
      if not self.check_rotations(ijkl_list,i,j,k,l,N)
    )
    # コーナーにクイーンがある場合の開始コンステレーション
    # ijkl_list.update({self.to_ijkl(0,j,0,l) for j in range(1,N-2) for l in range(j+1,N-1)})
    # は {...} で一時 set を作っていますが、以下のように generator にすればメモリ節約・速度向上します：
    ijkl_list.update(
      self.to_ijkl(0,j,0,l)
      for j in range(1,N-2)
      for l in range(j+1,N-1)
    )
    # Jasmin変換
    # ijkl_list_jasmin={self.jasmin(c,N) for c in ijkl_list}
    # ijkl_list_jasmin={self.get_jasmin(c,N) for c in ijkl_list}
    # ijkl_list_jasmin={self.get_jasmin(c,N) for c in ijkl_list}
    # ijkl_list=ijkl_list_jasmin
    ijkl_list={self.get_jasmin(c,N) for c in ijkl_list}
    #
    L=1<<(N-1)  # Lは左端に1を立てる
    for sc in ijkl_list:
      # i,j,k,l=self.geti(sc),self.getj(sc),self.getk(sc),self.getl(sc)
      i,j,k,l=self.getijkl(sc)
      # すべての「右辺のシフト値」が負にならないよう max(x,0) でガード
      ld,rd,col=(L>>max(i-1,0))|(1<<max(N-k,0)),(L>>max(i+1,0))|(1<<max(l-1,0)),1|L|(L>>i)|(L>>j)
      LD,RD=(L>>j)|(L>>l),(L>>j)|(1<<k)
      #-------------------------
      # [Opt-09] 状態ハッシュによる探索枝の枝刈り
      visited:set[int]=set()
      #-------------------------
      # 2. サブコンステレーション生成にtuple keyでキャッシュ
      #-------------------------
      # counter=[0] # サブコンステレーションを生成
      before=len(constellations)
      self.set_pre_queens_cached(ld,rd,col,k,l,1,3 if j==N-1 else 4,LD,RD,constellations,N,preset_queens,visited)
      # 今回新たに追加された件数
      created=len(constellations)-before
      # created=len(self.constellation_signatures)
      # 追加分に startijkl を付与（最後に追加された created 個）
      start=self.to_ijkl(i,j,k,l)
      # 生成されたサブコンステレーションにスタート情報を追加
      for idx in range(created):
          constellations[before+idx]["startijkl"] |=start
#
class NQueens21_constellations():
  def main(self)->None:
    nmin:int=5
    nmax:int=19
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
      NQ=NQueens21()
      #---------------------------------
      # 星座リストそのものをキャッシュ
      #---------------------------------
      # キャッシュを使わない
      NQ.gen_constellations(ijkl_list,constellations,size,preset_queens)
      # キャッシュを使う、キャッシュの整合性もチェック
      # -- txt
      # constellations=NQ.load_or_build_constellations_txt(ijkl_list,constellations,size,preset_queens)
      # -- bin
      # constellations=NQ.load_or_build_constellations_bin(ijkl_list,constellations,size,preset_queens)
      #
      #---------------------------------
      NQ.exec_solutions(constellations,size)
      total=sum(c["solutions"] for c in constellations if c["solutions"] > 0)
      unique=sum(c["unique"] for c in constellations if c["unique"] > 0)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"{size:2d}:{total:13d}{unique:13d}{text:>20s}")

if __name__=="__main__":
  NQueens21_constellations().main()

