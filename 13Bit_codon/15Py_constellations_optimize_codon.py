#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Python/codon Ｎクイーン コンステレーション版 キャッシュ最適化２


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


""""
 N-Queens Constellations Solver (Bitboard + Zobrist + 分割探索)

 本プログラムは N-Queens を「開始コンステレーション（部分盤面）」に
 分割し、各コンステレーションを高速に探索する実装です。

 主要コンセプト
 - ビットボード: 左/右対角線・列をビットで管理
     例) set_pre_queens() 内の free 計算:
         free = ~(ld | rd | col | (LD>>(N-1-row)) | (RD<<(N-1-row))) & ((1<<N)-1)
 - 開始コンステレーション列挙: gen_constellations()
     - 位置 (i,j,k,l) を 5bit パック (to_ijkl) で signature 化
     - 回転・ミラー対称除去 (check_rotations / jasmin)
 - 標準形（Jasmin正規化）: jasmin()
     - (i,j,k,l) を回転・ミラーで正規形へ写像し、重複を圧縮
 - Zobrist ハッシュ: zobrist_hash(), _mix64(), _gen_list(), _init_zobrist()
     - 盤面状態の64bitハッシュを計算し、高速な探索済み検出に利用
     - 実運用では state_hash() の O(1) 簡易ハッシュも併用
 - サブコンステレーション生成キャッシュ: set_pre_queens_cached()
     - StateKey（行/ビットボード/パラメタ）で生成済みをスキップ
 - 解探索分岐: exec_solutions() → SQ... 群
     - コーナー/境界/特殊行（j, k, l）に応じて最適な再帰ソルバを選択
     - symmetry() で 90/180/ミラーの重複補正(2/4/8倍)

 使い方
 - NQueens15_constellations().main() で N=5..19 を走査
 - 事前配置数 preset_queens は main() の preset_queens で設定

 実装上の注意
 - 64bit マスク MASK64 で Zobrist 値の範囲を明確化
 - N に応じた N-bit マスク ((1<<N)-1) を都度適用（符号や上位ビット汚染を防ぐ）
 - pickle の代わりに TXT/BIN キャッシュもサポート（Codon 互換考慮）
 - @par とある行は並列化の目印コメント（実行環境により扱いを調整）

 参考コード断片
 - 5bit パック: to_ijkl(i,j,k,l)
 - 回転90度:    rot90(ijkl, N)
 - 垂直ミラー:  mirvert(ijkl, N)
 - 標準形:      jasmin(ijkl, N)
 - 開始列挙:    gen_constellations(...)
 - 探索本体:    exec_solutions(...) → SQ...()


レビューの要点（コード中からの引用付き）

N-bit 正規化の徹底
zobrist_hash() 冒頭で
ld &= mask; rd &= mask; col &= mask; LD &= mask; RD &= mask
としており、負数や上位ビット汚染を確実に排除できています（👍とても重要）。

状態キャッシュの軽量化
visited に state_hash() の単一 int を入れる戦略は、
「StateKey タプル」より圧倒的に省メモリ・速い一方で、衝突の懸念は残ります。
N≤17 という範囲なら実害はまず出ない設計ですが、厳密性を求める検証 runでは
一時的に zobrist_hash() を使って比較する A/B 実験がおすすめ。

Jasmin 正規化
jasmin() は “最も端に近い軸”を選んで 90°回転を繰り返し、最後にミラー判定する明快設計。
ffmin(self.getj(ijkl), N-1-self.getj(ijkl)) のような端から距離の採用が効いています。

回転重複の即時排除
check_rotations() を gen_constellations() の列挙内に内挿しているため、
集合のサイズ増加を初期で抑制できています。

SQ 系の見通し
次行先読みを _extra_block_for_row() / _should_go_plus1() の2段で関数化し、
多数の SQ バリアントにまたがる可読性と再利用性を維持。
row==mark1 や <<2 / >>2 の2行スキップなど“固定行の強制通過”最適化も良いです。

I/O キャッシュの堅牢化
TXT/BIN 両対応、validate_*() と try/except を絡めた自己修復的ロードは実運用で有益。
BIN は 16 バイト境界のサイズ検査で破損検出しているのも OK。

amazon AWS m4.16xlarge x 1
$ codon build -release 15Py_constellations_optimize_codon.py && ./15Py_constellations_optimize_codon
 N:            Total       Unique        hh:mm:ss.ms
 5:               10            0         0:00:00.000
 6:                4            0         0:00:00.079
 7:               40            0         0:00:00.001
 8:               92            0         0:00:00.001
 9:              352            0         0:00:00.001
10:              724            0         0:00:00.002
11:             2680            0         0:00:00.102
12:            14200            0         0:00:00.002
13:            73712            0         0:00:00.005
14:           365596            0         0:00:00.011
15:          2279184            0         0:00:00.035
16:         14772512            0         0:00:00.078
17:         95815104            0         0:00:00.436
18:        666090624            0         0:00:02.961
19:       4968057848            0         0:00:22.049
20:      39029188884            0         0:02:52.430
21:     314666222712            0         0:24:25.554
22:    2691008701644            0         3:29:33.971
23:   24233937684440            0  1  day,8:12:58.977

top-10:29:32 up 1 day,16:13, 4 users, load average: 64.39,64.21,64.12
Tasks: 563 total,  2 running,561 sleeping,  0 stopped,  0 zombie
%Cpu(s):100.0 us, 0.0 sy, 0.0 ni, 0.0 id, 0.0 wa, 0.0 hi, 0.0 si, 0.0 st
MiB Mem : 257899.4 total,256193.4 free,  1225.5 used,   480.5 buff/cache
MiB Swap:      0.0 total,     0.0 free,     0.0 used. 255314.6 avail Mem
    PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND
   5634 suzuki    20   0   13.4g  70056   7384 R  6399   0.0 148411:55 15Py_constellat

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
✅[Opt-08]部分盤面サブ問題キャッシュ
場所: set_pre_queens_cached(...)
キー: key=(ld,rd,col,k,l,row,queens,LD,RD,N,preset_queens)
値: subconst_cache[key]=True
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
場所: get_jasmin(c,N) / jasmin_cache: Dict[Tuple[int,int],int]
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
    self.subconst_cache: Dict[StateKey,bool]={} … サブコンステ生成の再入防止
    self.constellation_signatures: Set[Tuple[int,int,int,int,int,int]]=set() … 星座の重複署名
    self.jasmin_cache: Dict[Tuple[int,int],int]={} … get_jasmin()の結果メモ化
    self.zobrist_tables: Dict[int,Dict[str,List[int]]]={} … Zobristテーブル（Nごと）

✅[Opt-13]部分盤面のキャッシュ（tuple化→dict）
  set_pre_queens_cached(...)
    キー：(ld,rd,col,k,l,row,queens,LD,RD,N,preset_queens)
    既出キーなら再帰呼び出しスキップ → 指数的重複カット

✅[Opt-14]星座（コンステレーション）の重複排除
  set_pre_queens(...) 内 if queens==preset_queens: ブロック
    署名：(ld,rd,col,k,l,row) を self.constellation_signatures で判定し重複追加を抑制

✅[Opt-15]Jasmin 正規化のメモ化
  get_jasmin(c,N) → self.jasmin_cache[(c,N)]
  何度も登場する起点パターンの再計算を回避

✅[Opt-16]訪問済み状態（transposition/visited）の仕込み
  gen_constellations(...) で visited: Set[StateKey]=set() を生成し
  set_pre_queens(...) 冒頭で key: StateKey=(...) を visited に登録・参照
  ※Zobrist版 zobrist_hash(...) も実装済（今はコメントアウトでトグル可）

    # 状態ハッシュによる探索枝の枝刈り バックトラック系の冒頭に追加　やりすぎると解が合わない
    #
    # zobrist_hash
    # 各ビットを見てテーブルから XOR するため O(N)（ld/rd/col/LD/RDそれぞれで最大 N 回）。
    # とはいえ N≤17 なのでコストは小さめ。衝突耐性は高い。
    # マスク漏れや負数の扱いを誤ると不一致が起きる点に注意（先ほどの&((1<<N)-1) 修正で解決）。
    # h:int=self.zobrist_hash(ld,rd,col,row,queens,k,l,LD,RD,N)
    #
    # state_hash
    # その場で数個の ^ と<<を混ぜるだけの O(1) 計算。
    # 生成されるキーも 単一の int なので、set/dict の操作が最速＆省メモリ。
    # ただし理論上は衝突し得ます（実際はN≤17の範囲なら実害が出にくい設計にしていればOK）。
    h:int=self.state_hash(ld,rd,col,row,queens,k,l,LD,RD,N)
    if h in visited:
        return
    visited.add(h)
    #
    # StateKey（タプル）
    # 11個の整数オブジェクトを束ねるため、オブジェクト生成・GC負荷・ハッシュ合成が最も重い。
    # set の比較・保持も重く、メモリも一番食います。
    # 衝突はほぼ心配ないものの、速度とメモリ効率は最下位。
    # key: StateKey=(ld,rd,col,row,queens,k,l,LD,RD,N,preset_queens)
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

__init__ で self.subconst_cache: Dict[StateKey,bool]={} を用意
set_pre_queens_cached(...) が tupleキー
  (ld,rd,col,k,l,row,queens,LD,RD,N,preset_queens)
  を使って self.subconst_cache を参照・更新
生成側は gen_constellations(...) から 最初の呼び出しを set_pre_queens_cached に変更済み
再帰内でも次の分岐呼び出しを set_pre_queens_cached(...) に置換しており、同一状態の再実行を回避
"""

"""
✅[Opt-19] 星座自体をtuple/hashで一意管理して重複を防ぐ
constellationsリストに追加する際、既に存在する星座を再追加しない
→ 星座自体を「tuple/int/hash」にして集合管理
これにより、異なる経路から同じ星座に到達しても重複追加を防げます。

__init__ で self.constellation_signatures: Set[Tuple[int,int,int,int,int,int]]=set() を用意。
set_pre_queens(...) 内の if queens==preset_queens: ブロックで
signature=(ld,rd,col,k,l,row) をキーに重複チェックし、未出だけ constellations.append(...) ＆ counter[0]+=1。
"""

"""
✅[Opt-20] Jasmin変換キャッシュ（クラス属性またはグローバル変数で）
（生成済み盤面の再利用）
ijkl_list_jasmin={self.jasmin(c,N) for c in ijkl_list} も、盤面→jasmin変換は「一度計算したらdictでキャッシュ」が効果大
#グローバル変数で

def get_jasmin(self,c:int,N:int) -> int:
    key=(c,N)
    if key in jasmin_cache:
        return jasmin_cache[key]
    result=self.jasmin(c,N)
    jasmin_cache[key]=result
    return result

# 使用例:gen_constellations()内に
ijkl_list_jasmin={self.get_jasmin(c,N) for c in ijkl_list}

__init__ に self.jasmin_cache: Dict[Tuple[int,int],int]={}

get_jasmin(self,c:int,N:int) で (c,N) をキーに memo 化

gen_constellations() 内で

ijkl_list={ self.get_jasmin(c,N) for c in ijkl_list }
としてキャッシュ経由で Jasmin 変換しています
"""

"""
✅[Opt-21] 180°重複チェックの二重化
check_rotations() は 90/180/270°すべて見ていますが、奇数 N の中央列ブロックで check_rotations(...) と
rot180_in_set(...) を両方呼んでいますね。ここは rot180 が 重複なので、check_rotations(...)
のみでOK（微小ですが内包表記が軽くなります）。

# 修正前（中央列ブロック）
ijkl_list.update(
    self.to_ijkl(i,j,center,l)
    for l in range(center+1,N-1)
    for i in range(center+1,N-1)
    if i != (N-1)-l
    for j in range(N-center-2,0,-1)
    if j != i and j != l
    if not self.check_rotations(ijkl_list,i,j,center,l,N)
    if not self.rot180_in_set(ijkl_list,i,j,center,l,N)  # ←これを削除
)

# 修正後（中央列ブロック）
ijkl_list.update(
    self.to_ijkl(i,j,center,l)
    for l in range(center+1,N-1)
    for i in range(center+1,N-1)
    if i != (N-1)-l
    for j in range(N-center-2,0,-1)
    if j != i and j != l
    if not self.check_rotations(ijkl_list,i,j,center,l,N)
)
"""

"""
✅[Opt-22] visited の粒度
visited を星座ごとに新規 set() にしているので、メモリ爆発を回避できています。ハッシュに ld,rd,col,row,queens,k,
l,LD,RD,N まで混ぜているのも衝突耐性◯。
  gen_constellations() の各スタート（星座）ごとに
  visited: Set[StateKey]=set() を新規作成
  StateKey=(ld,rd,col,row,queens,k,l,LD,RD,N,preset_queens) を追加・照合
  という構成なので、
  visited のスコープが星座単位 → メモリ増大を回避できている
  衝突耐性は ld/rd/col/LD/RD の**ビット集合＋行インデックスやカウンタ（row/queens）＋分岐（k/l）**まで含むので十分に高い

  gen_constellations() の各スタート（星座）ごとに
  visited: Set[StateKey]=set() を新規作成
  StateKey=(ld,rd,col,row,queens,k,l,LD,RD,N,preset_queens) を追加・照合
  という構成なので、visited のスコープが星座単位 → メモリ増大を回避できている
  衝突耐性は ld/rd/col/LD/RD の**ビット集合＋行インデックスやカウンタ（row/queens）＋分岐（k/l）**まで含むので十分に高いでOKです。

  細かい改善ポイント（任意）：
  N と preset_queens は探索中は一定なので、キーから外しても挙動は変わりません（キーが少し短くなります）。もちろん入れたままでも正しいです。
  もし将来 state_hash() に切り替えるときも、visited を星座ごとに new にする方針はそのまま維持してください（グローバルにしない）。
"""

"""
✅[Opt-23] ビット演算のインライン化・board_mask の上位での共有・**1ビット抽出 bit=x &
-x**など、要所は押さえられています。
cnt を星座ごとにローカルで完結→solutions に掛け算（symmetry()）という流れもキャッシュに優しい設計。
これ以上を狙うなら、「星座ごと分割の並列度を広げる」か「gen_constellations の ijkl_list.update(...)
での回転重複除去を最小限に（=set操作の負荷を減らす）」の二択ですが、現状の速度を見る限り十分実用的です。

  いまの実装は
  ビット演算の徹底（bit=x&-x／board_maskの共有／blocked→next_freeの短絡判定）
  cnt を星座ローカルで完結→最後に symmetry() を掛けるフロー
  visited を星座ごとに分離
  など、ボトルネックをしっかり押さえられていて実用速度も十分です。
  さらに“やるなら”の小粒アイデア（任意）だけ置いておきます：
  symmetry(ijkl,N) の結果を小さな dict でメモ化（星座件数分の呼び出しを削減）。
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
  next_free=board_mask&~blocked
  if next_free and ((row >= endmark-1) or self._has_future_space(next_ld,next_rd,next_col,board_mask)):
  （1行進む再帰は row+1 >= endmark／2行進む再帰は row+2 >= endmark などに合わせて判定）
  という形になっており、
  ゴール直前は先読み不要（短絡評価で _has_future_space を呼ばない）
  それ以外は**“1行先に置ける可能性が1ビットでもあるか”**の軽量チェックでムダ分岐を削減
  がきれいに機能しています。

  軽い補足（任意）：
  「+1 進む」「+2 進む」系で row+Δ >= endmark の Δ を必ず合わせる（すでに合わせてありますが、この一貫性が重要）。
  ループ先頭で if not next_free: continue の早期スキップを入れるのも読みやすさ的に○（実測差は小さいことが多いです）。
  _has_future_space 内の式は現在の実装（board_mask&~(((next_ld<<1)|(next_rd>>1)|next_col)) != 0）で十分速いです。
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
def is_partial_canonical(board: List[int],row:int,N:int) -> bool:
  # 現在の board[0:row] が他のミラー・回転盤面より辞書順で小さいか
  current=tuple(board[:row])
  symmetries=[]
  # ミラー（左右反転）
  mirrored=[N-1-b for b in current]
  symmetries.append(tuple(mirrored))
  # 90度回転：盤面を (col → row) に再構築する必要がある（簡略化版）
  # 完全な回転は行列転置＋ミラーが必要（時間コストあり）
  return all(current<=s for s in symmetries)
# -----------------------------------
"""

"""
❎ 未対応 軽量 is_canonical() による“部分盤面”の辞書順最小チェックを高速化（キャッシュ/軽量版）
「完成盤」だけでなく“部分盤面”用に軽量な変換（行の回転・反転は途中情報だけで可）を実装。
 is_partial_canonical() の中で zobrist_cache[hash]=True/False として使う

  「部分盤面の辞書順最小（canonical）チェック」は、基本的に
  board[row]=col_bit（＝各行に置いた列位置が順に分かる配列/スタック）
  もしくは cols_by_row=[c0,c1,...,c(r-1)] のように「置いた列の履歴」
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
  典型的には board[row]=col（行→列の履歴）を常に持っていてこそ強い枝刈りになります。
  いまの constellations 方式（ld/rd/col の集合状態＋row,k,l）だと、**「直前・直前々行でどの列に置いたか」**が直接分からないため、
  一般的な「近傍パターン」判定を素直に書くのは難しいです。
  もっとも、あなたの実装はすでに
  初手生成の厳しい対称性制約
  Jasmin 代表系化
  各種 SQ* 系の分岐（実質“マクロ手筋”をパターンとして埋め込んでいる）
  が効いているので、汎用の violate_macro_patterns を後付けする必要性は低めです。
"""



# import random
import pickle,os
# from operator import or_
# from functools import reduce
from typing import List,Set,Dict,Tuple
from datetime import datetime

# 64bit マスク（Zobrist用途）
MASK64:int=(1<<64)-1
# StateKey=Tuple[int,int,int,int,int,int,int,int,int,int,int]
StateKey=Tuple[int,int,int,int,int,int,int,int,int,int,int]
# StateKey=Tuple[int,int,int,int,int,int,int,int,int]


# pypyを使うときは以下を活かしてcodon部分をコメントアウト
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
#
class NQueens15:
  """N-Queens の探索・前処理（Zobrist, 対称除去, 開始盤面生成など）を担う中核クラス。

  属性:
      subconst_cache (Set[StateKey]): サブコンステレーション生成の重複防止キャッシュ。
      constellation_signatures (Set[Tuple[int,...]]): 生成済み星座の重複抑止用 signature 集。
      jasmin_cache (Dict[Tuple[int,int],int]): jasmin() 正規化の結果キャッシュ (key=(ijkl,N)).
      zobrist_tables (Dict[int,Dict[str,List[int]]]): Nごとの Zobrist テーブル。
      gen_cache (Dict[...]): 生成系の任意キャッシュ（拡張用）。
  """


  def __init__(self)->None:
    # StateKey
    # self.subconst_cache: Dict[ StateKey,bool ]={}
    # self.subconst_cache: Dict[ Tuple[int,int,int,int,int,int,int,int,int,int,int],bool ]={}
    self.subconst_cache: Set[StateKey]=set()
    self.constellation_signatures: Set[ Tuple[int,int,int,int,int,int] ]=set()
    self.jasmin_cache: Dict[Tuple[int,int],int]={}
    self.zobrist_tables: Dict[int,Dict[str,List[int]]]={}
    self.gen_cache: Dict[Tuple[int,int,int,int,int,int,int,int],List[Dict[str,int]] ]={}

  def _mix64(self,x:int) -> int:
    """splitmix64 の終段に相当する 64bit ミキサ。

    目的:
        Zobrist テーブル用の擬似乱数を生成する内部関数。
    参照:
        MASK64, _gen_list()
    """

    # splitmix64 の最終段だけ使ったミキサ
    x &= MASK64
    x=(x ^ (x>>30)) * 0xBF58476D1CE4E5B9&MASK64
    x=(x ^ (x>>27)) * 0x94D049BB133111EB&MASK64
    x ^= (x>>31)
    return x&MASK64

  def _gen_list(self,cnt:int,seed:int) -> List[int]:
    """Zobrist テーブルに使う 64bit 値を cnt 個生成して返す。
    引数:
        cnt: 生成個数
        seed: 64bit シード（splitmix64 の加算項を利用）
    戻り値:
        64bit 整数のリスト
    関連:
        _init_zobrist() で 'ld','rd','col','LD','RD','row','queens','k','l' に割当。
    """
    out: List[int]=[]
    s:int=seed&MASK64
    for _ in range(cnt):
      s=(s+0x9E3779B97F4A7C15)&MASK64   # splitmix64 のインクリメント
      out.append(self._mix64(s))
    return out

  def _init_zobrist(self,N:int) -> None:
    """盤サイズ N 用の Zobrist テーブルを初期化（1度だけ）。
    テーブル要素:
        'ld','rd','col','LD','RD','row','queens','k','l'
    再入防止:
        既に self.zobrist_tables[N] があれば何もしない。
    """
    # 例: self.zobrist_tables: Dict[int,Dict[str,List[int]]] を持つ前提。
    # N ごとに ['ld','rd','col','LD','RD','row','queens','k','l'] のテーブルを用意。
    if N in self.zobrist_tables:
      return
    base_seed:int=(0xC0D0_0000_0000_0000 ^ (N<<32))&MASK64
    tbl: Dict[str,List[int]]={
        'ld'    : self._gen_list(N,base_seed ^ 0x01),
        'rd'    : self._gen_list(N,base_seed ^ 0x02),
        'col'   : self._gen_list(N,base_seed ^ 0x03),
        'LD'    : self._gen_list(N,base_seed ^ 0x04),
        'RD'    : self._gen_list(N,base_seed ^ 0x05),
        'row'   : self._gen_list(N,base_seed ^ 0x06),
        'queens': self._gen_list(N,base_seed ^ 0x07),
        'k'     : self._gen_list(N,base_seed ^ 0x08),
        'l'     : self._gen_list(N,base_seed ^ 0x09),
    }
    self.zobrist_tables[N]=tbl

  def rot90(self,ijkl:int,N:int)->int:
    """(i,j,k,l) の盤面を 90°（時計回り）回転させた signature を返す。
    位置対応:
        (row, col) → (col, N-1-row)
    実装:
        geti/getj/getk/getl を再配置して 20bit に組み直し。
    """
    return ((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl)

  def rot180(self,ijkl:int,N:int)->int:
    """ 対称性のための計算と、ijklを扱うためのヘルパー関数。
    開始コンステレーションが回転90に対して対称である場合
    """
    return ((N-1-self.getj(ijkl))<<15)+((N-1-self.geti(ijkl))<<10)+((N-1-self.getl(ijkl))<<5)+(N-1-self.getk(ijkl))


  def check_rotations(self,ijkl_list:Set[int],i:int,j:int,k:int,l:int,N:int)->bool:
    """(i,j,k,l) の 90/180/270° 回転形のいずれかが既に集合に含まれているかを判定。
    戻り値:
        True: 既出（重複） / False: 未出（追加候補）
    用途:
        gen_constellations() で回転対称の重複を除外。
    """
    return any(rot in ijkl_list for rot in [((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i,((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+(N-1-k),(l<<15)+(k<<10)+((N-1-i)<<5)+(N-1-j)])

  def symmetry(self,ijkl:int,N:int)->int:
    """解の重複補正係数を返す（90°:2, 180°:4, その他:8）。
    用途:
        exec_solutions() で各コンステレーションの解数に補正を掛ける。
    """
    return 2 if self.symmetry90(ijkl,N) else 4 if self.geti(ijkl)==N-1-self.getj(ijkl) and self.getk(ijkl)==N-1-self.getl(ijkl) else 8

  def symmetry90(self,ijkl:int,N:int)->bool:
    """盤面が 90° 回転で自己一致するか（90°回転対称）を判定。"""
    return ((self.geti(ijkl)<<15)+(self.getj(ijkl)<<10)+(self.getk(ijkl)<<5)+self.getl(ijkl))==(((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl))

  def to_ijkl(self,i:int,j:int,k:int,l:int)->int:
    """(i,j,k,l) を 5bit ×4 の 20bit 整数にパックして返す（signature 用）。"""
    return (i<<15)+(j<<10)+(k<<5)+l

  def mirvert(self,ijkl:int,N:int)->int:
    """垂直ミラー（上下反転）後の signature を返す。"""
    return self.to_ijkl(N-1-self.geti(ijkl),N-1-self.getj(ijkl),self.getl(ijkl),self.getk(ijkl))

  def ffmin(self,a:int,b:int)->int:
    """微小高速のための min ラッパ（命名は Fast&Friendly の意）。"""
    return min(a,b)

  def geti(self,ijkl:int)->int: return (ijkl>>15)&0x1F
  def getj(self,ijkl:int)->int: return (ijkl>>10)&0x1F
  def getk(self,ijkl:int)->int: return (ijkl>>5)&0x1F
  def getl(self,ijkl:int)->int: return ijkl&0x1F

  def get_jasmin(self,c:int,N:int) -> int:
    """jasmin() による正規化にキャッシュを噛ませたラッパ。
    key:
        (c, N) をキーに結果を self.jasmin_cache に保存。
    1. Jasmin変換キャッシュを導入する
    [Opt-08] キャッシュ付き jasmin() のラッパー
    """
    key=(c,N)
    if key in self.jasmin_cache:
        return self.jasmin_cache[key]
    result=self.jasmin(c,N)
    self.jasmin_cache[key]=result
    return result

  def jasmin(self,ijkl:int,N:int)->int:
    """(i,j,k,l) を回転・ミラーで“最も左上に近い標準形”へ写像する。
    ポイント:
        - まず j/i/k/l の「端からの距離」の最小を持つ軸を優先し、arg 回 90°回転。
        - その後、j が上側に来るよう必要なら垂直ミラー。
    戻り値:
        正規化後の signature（20bit）
    補足:
        「l>k>i>j の優先順位」の考え方をコードに反映（実装コメント参照）。
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

  def file_exists(self,fname: str) -> bool:
    """ファイル存在を安全に判定（例外も握りつぶして False を返す）。"""
    try:
      with open(fname,"rb"):
        return True
    except:
      return False

  def validate_constellation_list(self,constellations: List[Dict[str,int]]) -> bool:
    """読み込んだコンステレーション配列が最低限のキーを持つかを検証。"""
    return all(all(k in c for k in ("ld","rd","col","startijkl")) for c in constellations)

  def read_uint32_le(self,b: str) -> int:
    """4バイト（文字列扱い/Codon互換）から little-endian 32bit int を復元。"""
    return (ord(b[0])&0xFF)|((ord(b[1])&0xFF)<<8)|((ord(b[2])&0xFF)<<16)|((ord(b[3])&0xFF)<<24)

  def int_to_le_bytes(self,x:int) -> List[int]:
    """32bit int を little-endian の4バイト配列に変換。"""
    return [(x>>(8 * i))&0xFF for i in range(4)]

  def validate_bin_file(self,fname: str) -> bool:
    """BINのサイズ妥当性を確認（1レコード=16バイト: ld,rd,col,startijkl）。"""
    try:
      with open(fname,"rb") as f:
        f.seek(0,2)  # ファイル末尾に移動
        size=f.tell()
      return size % 16==0
    except:
      return False

  def load_or_build_constellations_bin(self,ijkl_list: Set[int],constellations,N:int,preset_queens:int) -> List[Dict[str,int]]:
    """BINキャッシュを読み込み、壊れていれば再生成して保存して返すラッパ。"""
    fname=f"constellations_N{N}_{preset_queens}.bin"
    if self.file_exists(fname):
      try:
        constellations=self.load_constellations_bin(fname)
        if self.validate_bin_file(fname) and self.validate_constellation_list(constellations):
          return constellations
        else:
          print(f"[警告] 不正なキャッシュ形式: {fname} を再生成します")
      except Exception as e:
        print(f"[警告] キャッシュ読み込み失敗: {fname},理由: {e}")
    constellations: List[Dict[str,int]]=[]
    self.gen_constellations(ijkl_list,constellations,N,preset_queens)
    self.save_constellations_bin(fname,constellations)
    return constellations

  def save_constellations_txt(self,path: str,constellations: List[Dict[str,int]]) -> None:
    """コンステレーションをテキスト (ld rd col startijkl solutions) の1行形式で保存。"""
    with open(path,"w") as f:
      for c in constellations:
        ld=c["ld"]
        rd=c["rd"]
        col=c["col"]
        startijkl=c["startijkl"]
        solutions=c.get("solutions",0)
        f.write(f"{ld} {rd} {col} {startijkl} {solutions}\n")

  def load_constellations_txt(self,path: str) -> List[Dict[str,int]]:
    """テキスト形式のコンステレーションをロードし、辞書配列に復元。"""
    out: List[Dict[str,int]]=[]
    with open(path,"r") as f:
      for line in f:
        parts=line.strip().split()
        if len(parts) != 5:
          continue
        ld=int(parts[0]); rd=int(parts[1]); col=int(parts[2])
        startijkl=int(parts[3]); solutions=int(parts[4])
        out.append({"ld": ld,"rd": rd,"col": col,"startijkl": startijkl,"solutions": solutions})
    return out

  def save_constellations_bin(self,fname: str,constellations: List[Dict[str,int]]) -> None:
    """コンステレーションを BIN (ld, rd, col, startijkl) × n レコードで保存。"""
    with open(fname,"wb") as f:
      for d in constellations:
        for key in ["ld","rd","col","startijkl"]:
          b=self.int_to_le_bytes(d[key])
          f.write("".join(chr(c) for c in b))  # Codonでは str がバイト文字列扱い

  def load_constellations_bin(self,fname: str) -> List[Dict[str,int]]:
    """BIN形式のコンステレーションをロードして辞書配列に復元。"""
    constellations: List[Dict[str,int]]=[]
    with open(fname,"rb") as f:
      while True:
        raw=f.read(16)
        if len(raw)<16:
          break
        ld        =self.read_uint32_le(raw[0:4])
        rd        =self.read_uint32_le(raw[4:8])
        col       =self.read_uint32_le(raw[8:12])
        startijkl =self.read_uint32_le(raw[12:16])
        constellations.append({ "ld": ld,"rd": rd,"col": col,"startijkl": startijkl,"solutions": 0 })
    return constellations

  def load_or_build_constellations_txt(self,ijkl_list: Set[int],constellations,N:int,preset_queens:int) -> List[Dict[str,int]]:
    """TXTキャッシュを読み込み、壊れていれば再生成して保存して返すラッパ。"""
    fname=f"constellations_N{N}_{preset_queens}.txt"
    if self.file_exists(fname):
      try:
        constellations=self.load_constellations_txt(fname)
        if self.validate_constellation_list(constellations):
          return constellations
        else:
          print(f"[警告] 不正なキャッシュ形式: {fname} を再生成します")
      except Exception as e:
        print(f"[警告] キャッシュ読み込み失敗: {fname},理由: {e}")
    constellations: List[Dict[str,int]]=[]
    self.gen_constellations(ijkl_list,constellations,N,preset_queens)
    self.save_constellations_txt(fname,constellations)
    return constellations

  def set_pre_queens_cached(self,ld:int,rd:int,col:int,k:int,l:int,row:int,queens:int,LD:int,RD:int,counter:List[int],constellations: List[Dict[str,int]],N:int,preset_queens:int,visited:Set[int]) -> None:
    """set_pre_queens() に StateKey キャッシュを噛ませたラッパ。
    効果:
        同一状態の再実行をスキップして生成時間を短縮。
    """
    key:StateKey=(ld,rd,col,k,l,row,queens,LD,RD,N,preset_queens)
    if key in self.subconst_cache:
      # 以前に同じ状態で生成済み → 何もしない（または再利用）
      return
    # 新規実行（従来通りset_pre_queensの本体処理へ）
    self.set_pre_queens(ld,rd,col,k,l,row,queens,LD,RD,counter,constellations,N,preset_queens,visited)
    # self.subconst_cache[key]=True  # マークだけでOK
    self.subconst_cache.add(key)

  def zobrist_hash(self,ld:int,rd:int,col:int,row:int,queens:int,k:int,l:int,LD:int,RD:int,N:int) -> int:
    """Zobrist ハッシュ (64bit) を計算して返す。
    説明:
        各ビット集合（ld/rd/col/LD/RD）を N-bit に正規化した上で、
        立っているビット位置ごとにテーブル値を XOR。行(row)/個数(queens)/k/l も反映。
    注意:
        必ず N-bit マスクを事前適用（負数や上位ビット汚染を回避）。
    """
    self._init_zobrist(N)
    tbl=self.zobrist_tables[N]
    h=0
    mask=(1<<N)-1
    # ★ ここが重要：Nビットに揃える（負数や上位ビットを落とす）
    ld &= mask
    rd &= mask
    col &= mask
    LD &= mask
    RD &= mask
    # 以下はそのまま
    m=ld; i=0
    while i < N:
      if (m&1) != 0:
        h ^= tbl['ld'][i]
      m >>= 1; i+=1
    m=rd; i=0
    while i < N:
      if (m&1) != 0:
        h ^= tbl['rd'][i]
      m >>= 1; i+=1
    m=col; i=0
    while i < N:
      if (m&1) != 0:
        h ^= tbl['col'][i]
      m >>= 1; i+=1
    m=LD; i=0
    while i < N:
      if (m&1) != 0:
        h ^= tbl['LD'][i]
      m >>= 1; i+=1
    m=RD; i=0
    while i < N:
      if (m&1) != 0:
        h ^= tbl['RD'][i]
      m >>= 1; i+=1
    if 0<=row < N:     h ^= tbl['row'][row]
    if 0<=queens < N:  h ^= tbl['queens'][queens]
    if 0<=k < N:       h ^= tbl['k'][k]
    if 0<=l < N:       h ^= tbl['l'][l]
    return h&MASK64

  def state_hash(self,ld:int,rd:int,col:int,row:int,queens:int,k:int,l:int,LD:int,RD:int,N:int) -> int:
    """O(1) の軽量ハッシュ（探索の高速枝刈り用）。
    使い分け:
        - 衝突耐性重視: zobrist_hash()
        - 速度/省メモリ重視: state_hash()（この実装）
    """
    # [Opt-09] Zobrist Hash（Opt-09）の導入とその用途
    # ビットボード設計でも、「盤面のハッシュ」→「探索済みフラグ」で枝刈りは可能です。
    return (ld<<3) ^ (rd<<2) ^ (col<<1) ^ row ^ (queens<<7) ^ (k<<12) ^ (l<<17) ^ (LD<<22) ^ (RD<<27) ^ (N<<1)

  def set_pre_queens(self,ld:int,rd:int,col:int,k:int,l:int,row:int,queens:int,LD:int,RD:int,counter:list,constellations:List[Dict[str,int]],N:int,preset_queens:int,visited:Set[int])->None:
    """開始コンステレーション（preset_queens 個のクイーン配置）を再帰的に列挙する。
    振る舞い:
        - row が k/l の行はスキップ（必置行）
        - queens == preset_queens で現状態を constellations に push
        - free = ~(ld|rd|col|(LD>>(N-1-row))|(RD<<(N-1-row))) & ((1<<N)-1)
          の各ビットにクイーンを立てながら深掘り
    最適化:
        - visited に state_hash を登録し、再訪を枝刈り
        - constellation_signatures で (ld,rd,col,k,l,row) 重複を抑止
    出力:
        counter[0] に生成件数を加算、constellations に辞書を追加（startijkl は後で付与）
    """
    mask=(1<<N)-1  # setPreQueensで使用
    # 状態ハッシュによる探索枝の枝刈り バックトラック系の冒頭に追加　やりすぎると解が合わない
    #
    # zobrist_hash
    # 各ビットを見てテーブルから XOR するため O(N)（ld/rd/col/LD/RDそれぞれで最大 N 回）。
    # とはいえ N≤17 なのでコストは小さめ。衝突耐性は高い。
    # マスク漏れや負数の扱いを誤ると不一致が起きる点に注意（先ほどの&((1<<N)-1) 修正で解決）。
    # h:int=self.zobrist_hash(ld,rd,col,row,queens,k,l,LD,RD,N)
    #
    # state_hash
    # その場で数個の ^ と<<を混ぜるだけの O(1) 計算。
    # 生成されるキーも 単一の int なので、set/dict の操作が最速＆省メモリ。
    # ただし理論上は衝突し得ます（実際はN≤17の範囲なら実害が出にくい設計にしていればOK）。
    h:int=self.state_hash(ld,rd,col,row,queens,k,l,LD,RD,N)
    if h in visited:
      return
    visited.add(h)
    #
    # StateKey（タプル）
    # 11個の整数オブジェクトを束ねるため、オブジェクト生成・GC負荷・ハッシュ合成が最も重い。
    # set の比較・保持も重く、メモリも一番食います。
    # 衝突はほぼ心配ないものの、速度とメモリ効率は最下位。
    # key: StateKey=(ld,rd,col,row,queens,k,l,LD,RD)
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
    if queens==preset_queens:
      # signatureの生成
      signature=(ld,rd,col,k,l,row)  # 必要な変数でOK
      # signaturesセットをクラス変数やグローバルで管理
      if not hasattr(self,"constellation_signatures"):
        self.constellation_signatures=set()
      signatures=self.constellation_signatures
      if signature not in signatures:
        constellation={"ld": ld,"rd": rd,"col": col,"startijkl": row<<20,"solutions": 0}
        constellations.append(constellation) #星座データ追加
        signatures.add(signature)
        counter[0]+=1
      return
    # 現在の行にクイーンを配置できる位置を計算
    free=~(ld|rd|col|(LD>>(N-1-row))|(RD<<(N-1-row)))&mask
    while free:
      bit:int=free&-free
      free&=free-1
      # クイーンを配置し、次の行に進む
      # self.set_pre_queens((ld|bit)<<1,(rd|bit)>>1,col|bit,k,l,row+1,queens+1,LD,RD,counter,constellations,N,preset_queens,visited)
      self.set_pre_queens_cached((ld|bit)<<1,(rd|bit)>>1,col|bit,k,l,row+1,queens+1,LD,RD,counter,constellations,N,preset_queens,visited)

  def exec_solutions(self,constellations:List[Dict[str,int]],N:int)->None:
    """各コンステレーションに対して最適な SQ... 再帰ソルバを選び、解数を計算して書き込む。
    仕組み:
        - startijkl から j,k,l 等を復元し、境界条件/位置関係に応じて SQ... 関数を選択
        - 返値 cnt に symmetry(ijkl,N) を掛けて constellation['solutions'] に格納
    注意:
        - board_mask は (1<<N)-1 を用いる（上位ビット汚染注意）
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

  def gen_constellations(self,ijkl_list:Set[int],constellations:List[Dict[str,int]],N:int,preset_queens:int)->None:
    """開始コンステレーションの代表集合を生成（回転・ミラー重複排除込み）。

    ステップ:
        1) (i,j,k,l) を厳密順序で走査し、回転重複は check_rotations() で除去
        2) 奇数Nは中央列を特別扱い（Opt-03）
        3) jasmin()（get_jasmin）で標準形に写像 → ijkl_list を正規化
        4) 各 signature に対し set_pre_queens_cached() でサブ星座を列挙
        5) 列挙結果の 'startijkl' に基底 to_ijkl(i,j,k,l) を OR 付与
    出力:
        constellations に {ld,rd,col,startijkl,solutions} の辞書を詰める
    """
    halfN=(N+1)//2  # Nの半分を切り上げ
    # --- [Opt-03] 中央列特別処理（奇数Nの場合のみ） ---
    if N % 2==1:
      center=N // 2
      ijkl_list.update(
        self.to_ijkl(i,j,center,l)
        for l in range(center+1,N-1)
        for i in range(center+1,N-1)
        if i != (N-1)-l
        for j in range(N-center-2,0,-1)
        if j != i and j != l
        if not self.check_rotations(ijkl_list,i,j,center,l,N)
        # 180°回転盤面がセットに含まれていない
        # if not self.rot180_in_set(ijkl_list,i,j,center,l,N)
      )
    # --- [Opt-03] 中央列特別処理（奇数Nの場合のみ） ---
    # コーナーにクイーンがいない場合の開始コンステレーションを計算する
    ijkl_list.update(self.to_ijkl(i,j,k,l) for k in range(1,halfN) for l in range(k+1,N-1) for i in range(k+1,N-1) if i != (N-1)-l for j in range(N-k-2,0,-1) if j!=i and j!=l if not self.check_rotations(ijkl_list,i,j,k,l,N))
    # コーナーにクイーンがある場合の開始コンステレーションを計算する
    ijkl_list.update({self.to_ijkl(0,j,0,l) for j in range(1,N-2) for l in range(j+1,N-1)})
    # Jasmin変換
    # ijkl_list_jasmin={self.jasmin(c,N) for c in ijkl_list}
    # ijkl_list_jasmin={self.get_jasmin(c,N) for c in ijkl_list}
    # ijkl_list=ijkl_list_jasmin
    ijkl_list={self.get_jasmin(c,N) for c in ijkl_list}
    L=1<<(N-1)  # Lは左端に1を立てる
    # ローカルアクセスに変更
    geti,getj,getk,getl=self.geti,self.getj,self.getk,self.getl
    to_ijkl=self.to_ijkl
    for sc in ijkl_list:
      # ここで毎回クリア（＝この sc だけの重複抑止に限定）
      # self.constellation_signatures.clear()
      self.constellation_signatures=set()
      # i,j,k,l=self.geti(sc),self.getj(sc),self.getk(sc),self.getl(sc)
      i,j,k,l=geti(sc),getj(sc),getk(sc),getl(sc)
      # ld,rd,col=(L>>(i-1))|(1<<(N-k)),(L>>(i+1))|(1<<(l-1)),1|L|(L>>i)|(L>>j)
      # LD,RD=(L>>j)|(L>>l),(L>>j)|(1<<k)
      Lj=L>>j; Li=L>>i; Ll=L>>l
      # ld=(L>>(i-1))|(1<<(N-k))
      ld=((L>>(i-1)) if i > 0 else 0)|(1<<(N-k))
      rd=(L>>(i+1))|(1<<(l-1))
      col=1|L|Li|Lj
      LD=Lj|Ll
      RD=Lj|(1<<k)

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
      #   constellations[-1-a]["startijkl"]|=self.to_ijkl(i,j,k,l)
      #
      # to_ijkl(i,j,k,l) はループ外で一回だけ
      # 今は毎回呼んでいるので、定数化すると少しだけ軽くなります。
      # base=self.to_ijkl(i,j,k,l)
      base=to_ijkl(i,j,k,l)
      for a in range(counter[0]):
          constellations[-1-a]["startijkl"]|=base

  @staticmethod
  def _has_future_space_step(next_ld:int,next_rd:int,next_col:int,row_next:int,endmark:int,board_mask:int,extra_block_next:int) -> bool:
    """次行 row_next で少なくとも 1bit 置ける見込みがあるかを判定（短絡枝刈り）。
    仕様:
        - row_next >= endmark の場合は True（ゴール直前）
        - それ以外は blocked = (next_ld<<1)|(next_rd>>1)|next_col|extra_block_next を見て判定
    """
    # ゴール直前は先読み不要（短絡）
    if row_next >= endmark:
        return True
    blocked_next=(next_ld<<1)|(next_rd>>1)|next_col|extra_block_next
    return (board_mask&~blocked_next) != 0

  @staticmethod
  def _extra_block_for_row(row_next:int,mark1:int,mark2:int,jmark:int,N:int) -> int:
    """次行に入るときに適用すべき“追加遮蔽”ビットを返す（k/l/j の固定影響をモデル化）。

    仕様:
        - row_next == mark1 or mark2: (1<<(N-3)) を追加（blockK 相当）
        - row_next == (N-1-jmark): (1<<(N-1)) を追加（j行系）
    """
    extra=0
    blockK=1<<(N-3)  # あなたのロジックに合わせて blockL 等も別にするなら拡張
    if row_next==mark1:
        extra|=blockK
    if row_next==mark2:
        extra|=blockK
    if row_next==(N-1-jmark):  # jmark 系ありの関数だけ使う
        extra|=(1<<(N-1))
    return extra

  def _should_go_plus1( self,next_free:int,row_next:int,endmark:int,next_ld:int,next_rd:int,next_col:int,board_mask:int,extra:int,) -> bool:
    """次行へ進む前に“行っても無駄にならないか”を先読み判定する薄いラッパ。"""
    if not next_free:
        return False
    if row_next >= endmark:
        return True
    return self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra)

  def SQd0B(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,N:int)->int:
    """行指向の再帰探索（分岐版）。行ごとに 1bit（クイーン）を選んで次へ進む。
    共通引数:
        ld, rd, col (int): 左/右対角線・列のビット占有。次行遷移で <<1 / >>1 を付与。
        row (int): 現在の行インデックス。
        free (int): 現行行で置ける位置のビット集合（board_mask & ~blocked）。
        jmark, endmark (int): 特殊行 j の目印 / ゴール直前行の目印（最深行）。
        mark1, mark2 (int): 事前固定行（k,l 起因）の“追加ブロック”適用行。
        board_mask (int): 盤サイズ N の Nbit マスク((1<<N)-1)。
        N (int): 盤面サイズ。

    共通ロジック:
        - avail の最下位ビットを取り出して配置 → 次行へ（ld<<1 / rd>>1 / col|bit）
        - _extra_block_for_row() で次行に入るときの追加遮蔽（k/l/jの固定影響）を考慮
        - _should_go_plus1() で「次行に少なくとも1bit置ける見込みか」を先読みして枝刈り
        - 行末(endmark)到達で 1 返し、和を積算

    命名規則のヒント:
        SQ[d?][B?][k?][l?][jr?]B
          d0/d1/d2 : コーナー距離/ケース種別（exec_solutions の分岐条件に対応）
          k / l    : 固定行の通過タイミング（mark1/mark2 が効く）
          jr       : j 行（コーナー相対の特別行）を即時処理するバリアント
          B        : “Bitboard step（再帰）”の意味的接尾辞

    # 例: SQd0B …… d0ケースのベース版（特殊行処理なしの基本遷移）
    # 例: SQd0BkB …… d0 + k固定行をヒットしたら 2行スキップで進める版
    # 例: SQd1B …… d1ケースのベース版。endmark 到達で 1 を返す。
    # 例: SQBjrB …… j 行に入ったら列0をマスクして即時処理する版
    # 例: SQBjlBkBlBjrB …… j行を“1行前倒し”で処理しつつ、l→k 順の固定行考慮版
    """
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
      # if next_free and ((row+1 >= endmark) or self._has_future_space(next_ld,next_rd,next_col,board_mask)):
      # if next_free:
      #   total+=self.SQd0B(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
      # if next_free:
      row_next:int=row+1
      # extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
      # if row_next==mark1:
      #   extra|=(1<<(N-3)) #blockK
      # if row_next==mark2:
      #   extra|=(1<<(N-3)) #blockK or blockL
      # jmark 系の分岐がある関数ではここでJのビットも追加する
      # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      extra=_extra_block_for_row(row_next,mark1,mark2,jmark,N)
      if _should_go_plus1(next_free,row_next,endmark,next_ld,next_rd,next_col,board_mask,extra):
      # if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #     extra|=(1<<(N-3)) #blockK
      #   if row_next==mark2:
      #     extra|=(1<<(N-3)) #blockK or blockL
      #   # jmark 系の分岐がある関数ではここでJのビットも追加する
      #   # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      #   if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #     extra|=(1<<(N-3)) #blockK
      #   if row_next==mark2:
      #     extra|=(1<<(N-3)) #blockK or blockL
      #   # jmark 系の分岐がある関数ではここでJのビットも追加する
      #   # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      #   if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #   extra|=(1<<(N-3)) #blockK
      # if row_next==mark2:
      #   extra|=(1<<(N-3)) #blockK or blockL
      # jmark 系の分岐がある関数ではここでJのビットも追加する
      # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      row_next:int=row+1
      extra=_extra_block_for_row(row_next,mark1,mark2,jmark,N)
      if _should_go_plus1(next_free,row_next,endmark,next_ld,next_rd,next_col,board_mask,extra):
      # if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      # extra=_extra_block_for_row(row_next,mark1,mark2,jmark,N)
      # if _should_go_plus1(next_free,row_next,endmark,next_ld,next_rd,next_col,board_mask,extra):
        total+=self.SQd1BkBlB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
      # if next_free:
      #   row_next:int=row+1
      #   extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
      #   if row_next==mark1:
      #     extra|=(1<<(N-3)) #blockK
      #   if row_next==mark2:
      #     extra|=(1<<(N-3)) #blockK or blockL
      #   # jmark 系の分岐がある関数ではここでJのビットも追加する
      #   # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      #   if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      # if next_free and ((row+2 >= endmark) or self._has_future_space(next_ld,next_rd,next_col,board_mask)):
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
      #     extra|=(1<<(N-3)) #blockK
      #   if row_next==mark2:
      #     extra|=(1<<(N-3)) #blockK or blockL
      #   # jmark 系の分岐がある関数ではここでJのビットも追加する
      #   # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      #   if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #     extra|=(1<<(N-3)) #blockK
      #   if row_next==mark2:
      #     extra|=(1<<(N-3)) #blockK or blockL
      #   # jmark 系の分岐がある関数ではここでJのビットも追加する
      #   # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      #   if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #     extra|=(1<<(N-3)) #blockK
      #   if row_next==mark2:
      #     extra|=(1<<(N-3)) #blockK or blockL
      #   # jmark 系の分岐がある関数ではここでJのビットも追加する
      #   # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      #   if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #     extra|=(1<<(N-3)) #blockK
      #   if row_next==mark2:
      #     extra|=(1<<(N-3)) #blockK or blockL
      #   # jmark 系の分岐がある関数ではここでJのビットも追加する
      #   # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      #   if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #     extra|=(1<<(N-3)) #blockK
      #   if row_next==mark2:
      #     extra|=(1<<(N-3)) #blockK or blockL
      #   # jmark 系の分岐がある関数ではここでJのビットも追加する
      #   # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      #   if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #     extra|=(1<<(N-3)) #blockK
      #   if row_next==mark2:
      #     extra|=(1<<(N-3)) #blockK or blockL
      #   # jmark 系の分岐がある関数ではここでJのビットも追加する
      #   # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      #   if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #     extra|=(1<<(N-3)) #blockK
      #   if row_next==mark2:
      #     extra|=(1<<(N-3)) #blockK or blockL
      #   # jmark 系の分岐がある関数ではここでJのビットも追加する
      #   # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      #   if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #     extra|=(1<<(N-3)) #blockK
      #   if row_next==mark2:
      #     extra|=(1<<(N-3)) #blockK or blockL
      #   # jmark 系の分岐がある関数ではここでJのビットも追加する
      #   # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      #   if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #     extra|=(1<<(N-3)) #blockK
      #   if row_next==mark2:
      #     extra|=(1<<(N-3)) #blockK or blockL
      #   # jmark 系の分岐がある関数ではここでJのビットも追加する
      #   # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      #   if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #     extra|=(1<<(N-3)) #blockK
      #   if row_next==mark2:
      #     extra|=(1<<(N-3)) #blockK or blockL
      #   # jmark 系の分岐がある関数ではここでJのビットも追加する
      #   # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      #   if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      # if next_free and ((row+1 >= endmark) or self._has_future_space(next_ld,next_rd,next_col,board_mask)):
      # if next_free:
      #   total+=self.SQd2B(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
      # if next_free:
      row_next:int=row+1
      # extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
      # if row_next==mark1:
      #   extra|=(1<<(N-3)) #blockK
      # if row_next==mark2:
      #   extra|=(1<<(N-3)) #blockK or blockL
      # jmark 系の分岐がある関数ではここでJのビットも追加する
      # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      extra=_extra_block_for_row(row_next,mark1,mark2,jmark,N)
      if _should_go_plus1(next_free,row_next,endmark,next_ld,next_rd,next_col,board_mask,extra):
      # if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #     extra|=(1<<(N-3)) #blockK
      #   if row_next==mark2:
      #     extra|=(1<<(N-3)) #blockK or blockL
      #   # jmark 系の分岐がある関数ではここでJのビットも追加する
      #   # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      #   if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #     extra|=(1<<(N-3)) #blockK
      #   if row_next==mark2:
      #     extra|=(1<<(N-3)) #blockK or blockL
      #   # jmark 系の分岐がある関数ではここでJのビットも追加する
      #   # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      #   if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
        #   extra|=(1<<(N-3)) #blockK
        # if row_next==mark2:
        #   extra|=(1<<(N-3)) #blockK or blockL
        # jmark 系の分岐がある関数ではここでJのビットも追加する
        # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
        extra=_extra_block_for_row(row_next,mark1,mark2,jmark,N)
        if _should_go_plus1(next_free,row_next,endmark,next_ld,next_rd,next_col,board_mask,extra):
        # if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #   extra|=(1<<(N-3)) #blockK
      # if row_next==mark2:
      #   extra|=(1<<(N-3)) #blockK or blockL
      # jmark 系の分岐がある関数ではここでJのビットも追加する
      # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      extra=self._extra_block_for_row(row_next,mark1,mark2,jmark,N)
      if self._should_go_plus1(next_free,row_next,endmark,next_ld,next_rd,next_col,board_mask,extra):
      # if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      # if next_free and ((row+1 >= endmark) or self._has_future_space(next_ld,next_rd,next_col,board_mask)):
      # if next_free:
      #   total+=self.SQB(next_ld,next_rd,next_col,row+1,next_free,jmark,endmark,mark1,mark2,board_mask,N)
      # if next_free:
      row_next:int=row+1
      # extra=0 # 次の行が特殊行なら、その行で実際にORされる追加ブロックを足す
      # if row_next==mark1:
      #   extra|=(1<<(N-3)) #blockK
      # if row_next==mark2:
      #   extra|=(1<<(N-3)) #blockK or blockL
      # jmark 系の分岐がある関数ではここでJのビットも追加する
      # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      extra=_extra_block_for_row(row_next,mark1,mark2,jmark,N)
      if _should_go_plus1(next_free,row_next,endmark,next_ld,next_rd,next_col,board_mask,extra):
      # if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #     extra|=(1<<(N-3)) #blockK
      #   if row_next==mark2:
      #     extra|=(1<<(N-3)) #blockK or blockL
      #   # jmark 系の分岐がある関数ではここでJのビットも追加する
      #   # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      #   if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #     extra|=(1<<(N-3)) #blockK
      #   if row_next==mark2:
      #     extra|=(1<<(N-3)) #blockK or blockL
      #   # jmark 系の分岐がある関数ではここでJのビットも追加する
      #   # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      #   if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #     extra|=(1<<(N-3)) #blockK
      #   if row_next==mark2:
      #     extra|=(1<<(N-3)) #blockK or blockL
      #   # jmark 系の分岐がある関数ではここでJのビットも追加する
      #   # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      #   if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #     extra|=(1<<(N-3)) #blockK
      #   if row_next==mark2:
      #     extra|=(1<<(N-3)) #blockK or blockL
      #   # jmark 系の分岐がある関数ではここでJのビットも追加する
      #   # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      #   if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #   extra|=(1<<(N-3)) #blockK
      # if row_next==mark2:
      #   extra|=(1<<(N-3)) #blockK or blockL
      # jmark 系の分岐がある関数ではここでJのビットも追加する
      # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      extra=_extra_block_for_row(row_next,mark1,mark2,jmark,N)
      if _should_go_plus1(next_free,row_next,endmark,next_ld,next_rd,next_col,board_mask,extra):
      # if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #   extra|=(1<<(N-3)) #blockK
      # if row_next==mark2:
      #   extra|=(1<<(N-3)) #blockK or blockL
      # jmark 系の分岐がある関数ではここでJのビットも追加する
      # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      extra=_extra_block_for_row(row_next,mark1,mark2,jmark,N)
      if _should_go_plus1(next_free,row_next,endmark,next_ld,next_rd,next_col,board_mask,extra):
      # if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #   extra|=(1<<(N-3)) #blockK
      # if row_next==mark2:
      #   extra|=(1<<(N-3)) #blockK or blockL
      # jmark 系の分岐がある関数ではここでJのビットも追加する
      # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      extra=_extra_block_for_row(row_next,mark1,mark2,jmark,N)
      if _should_go_plus1(next_free,row_next,endmark,next_ld,next_rd,next_col,board_mask,extra):
      # if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
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
      #   extra|=(1<<(N-3)) #blockK
      # if row_next==mark2:
      #   extra|=(1<<(N-3)) #blockK or blockL
      # jmark 系の分岐がある関数ではここでJのビットも追加する
      # if row_next==(N-1-jmark): extra|=(1<<(N-1)) 等、該当関数の実装に合わせる
      extra=_extra_block_for_row(row_next,mark1,mark2,jmark,N)
      if _should_go_plus1(next_free,row_next,endmark,next_ld,next_rd,next_col,board_mask,extra):
      # if self._has_future_space_step(next_ld,next_rd,next_col,row_next,endmark,board_mask,extra):
        total+=self.SQBjlBlkBjrB(next_ld,next_rd,next_col,row_next,next_free,jmark,endmark,mark1,mark2,board_mask,N)
    return total
class NQueens15_constellations():
  """小さなNの総数検算や、バッチ実行エントリ(main)を持つ補助クラス。"""

  def _bit_total(self,size:int) -> int:
    """小さな N（例: N≤5）を正攻法バックトラックで全列挙して総数を返す。"""
    mask=(1<<size)-1
    total=0

    def bt(row:int,left:int,down:int,right:int):
      nonlocal total
      if row==size:
        total+=1
        return
      bitmap=mask&~(left|down|right)
      while bitmap:
        bit=-bitmap&bitmap
        bitmap ^= bit
        bt(row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
    bt(0,0,0,0)
    return total

  def main(self)->None:
    """N=5..19 を対象に、(開始列挙 → 解探索) の一括実行を行い、所要時間を表示。

    振る舞い:
        - N≤5 は _bit_total() で検算
        - 通常は NQueens15().gen_constellations(...), exec_solutions(...) の順に実行
        - solutions 合計と経過時間を出力
    """
    nmin:int=5
    nmax:int=20
    preset_queens:int=4  # 必要に応じて変更
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax):
      start_time=datetime.now()
      if size<=5:
        # ← フォールバック：N=5はここで正しい10を得る
        total=self._bit_total(size)
        dt=datetime.now()-start_time
        text=str(dt)[:-3]
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
      # constellations=NQ.load_or_build_constellations_txt(ijkl_list,constellations,size,preset_queens)
      # -- bin
      # constellations=NQ.load_or_build_constellations_bin(ijkl_list,constellations,size,preset_queens)
      #---------------------------------
      NQ.exec_solutions(constellations,size)
      total:int=sum(c['solutions'] for c in constellations if c['solutions']>0)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")
if __name__=="__main__":
  NQueens15_constellations().main()
