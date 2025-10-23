#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Python/codon Ｎクイーン コンステレーション版 インテグレート

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
N-Queens 高速ソルバ (NQueens17) — レビュー & 実装ノート
=====================================================

■ 全体像
- 本実装は「星座 (constellation)」= 事前に 3〜4 個のクイーンを置いた部分盤面を起点に、
  それぞれを独立に深さ優先探索 (DFS) で解数を数える分割統治法。
  - 代表列挙: `gen_constellations()` が対称性・回転対称を除去したサブ盤面を生成
    （`check_rotations()`, `jasmin()` による正規化）。
  - 解探索: `exec_solutions()` がサブ盤面ごとに最適化された分岐を選び、
    汎用化された `dfs()` の 1 本化ループでカウント。
  - 対称補正: `symmetry()` で 90°, 180°, ミラーを勘案して重複補正。

■ 主要データ表現
- 盤面ビットボード: 列 / 左右対角の占有をビットで持つ（`ld`, `rd`, `col`）。
- 4 点の代表座標 (i, j, k, l) を 5bit × 4 = 20bit にパック: `to_ijkl()`。
- Zobrist Hash を導入しうる設計（`mix64()`, `gen_list()`, `init_zobrist()`, `zobrist_hash()`）。

■ 速度の肝
- 「分岐の前計算」を `exec_solutions()` で行い、DFS 本体は `dfs()` 1 本化。
  - `func_meta = [(next_funcid, funcptn, availptn), ...]` により
    「mark 行での step=2/3」「jmark 行での特殊」「ゴール判定」「+1 先読み」などを
    汎用ループに畳み込み、関数分岐のコストを削減。
- サブコンステレーション生成では、行固定 (k,l) を強制配置して分枝を極限まで減らす（`set_pre_queens()`）。

■ よくできている点
- 代表列挙の中で「Jasmin 正規化（回転/ミラーの規約化）」を行うことで、探索開始点を最左上へ寄せ、
  さらに `check_rotations()` で回転重複を抑止している。
- DFS の「未来 1 行の先読み」(`use_future`) による早期枝刈り。
- SoA（配列 of フィールド）化して、並列化区間のループ体を軽量化。

■ 注意/改善ポイント（重要）
1) Zobrist 初期化とキャッシュが実質無効
   - `init_zobrist(self, N, zobrist_tables)` は `self.zobrist_tables` を参照しますが、
     クラスに `self.zobrist_tables` が存在しません。
     しかも `zobrist_hash()` 側で毎回 `zobrist_tables = {}` を新規作成しており、
     テーブルがキャッシュされません。
     → 【修正案】
       - コンストラクタで `self.zobrist_tables: Dict[int, Dict[str, List[int]]] = {}` を持つ。
       - `init_zobrist()` は `self.zobrist_tables` を埋める形に（返り値でなく副作用）。
       - `zobrist_hash()` は `self.init_zobrist(N)` だけ呼び、`tbl = self.zobrist_tables[N]` を使用。

2) `get_jasmin()` のキャッシュが毎回空
   - `jasmin_cache` をローカルで `={}` 初期化しているため、呼ぶたびに空になります。
     → 【修正案】`self.jasmin_cache: Dict[Tuple[int,int], int] = {}` を用意し、そこを使う。

3) `set_pre_queens_cached()` のサブコンステキャッシュが毎回空
   - `subconst_cache: Set[StateKey] = set()` を関数ローカルで初期化しており効果なし。
     → 【修正案】`self.subconst_cache: Set[StateKey]` をコンストラクタで確保・再利用。

4) `init_zobrist()` の返り値と `return` がちぐはぐ
   - `init_zobrist()` は `tbl` を返していますが、`if N in self.zobrist_tables: return` と
     `None` を返す経路があり、呼び出し側の `tbl = self.init_zobrist(...)` が破綻します。
     → 【修正案】副作用専用にして返さない。呼び出し側は `self.zobrist_tables[N]` を参照。

5) 未使用/混乱しやすい import
   - `import pickle, os` は現状未使用。使わないなら削除推奨。

6) バイナリ I/O の注意（Codon 想定コメントは残しつつ）
   - `save_constellations_bin()` で `f.write("".join(chr(c) for c in b))` は
     CPython では `TypeError`（`bytes` を書くべき）。用途に応じて分岐/注釈を。

■ 代表的な修正パッチ（抜粋）
- コンストラクタを追加し、キャッシュをフィールド化：

    class NQueens17:
        def __init__(self) -> None:
            self.zobrist_tables: Dict[int, Dict[str, List[int]]] = {}
            self.jasmin_cache: Dict[Tuple[int, int], int] = {}
            self.subconst_cache: Set[StateKey] = set()

- init_zobrist を副作用化：

    def init_zobrist(self, N: int) -> None:
        MASK64 = (1 << 64) - 1
        if N in self.zobrist_tables:
            return
        base_seed = (0xC0D0_0000_0000_0000 ^ (N << 32)) & MASK64
        g = self.gen_list
        self.zobrist_tables[N] = {
            'ld': g(N, base_seed ^ 0x01),
            'rd': g(N, base_seed ^ 0x02),
            'col': g(N, base_seed ^ 0x03),
            'LD': g(N, base_seed ^ 0x04),
            'RD': g(N, base_seed ^ 0x05),
            'row': g(N, base_seed ^ 0x06),
            'queens': g(N, base_seed ^ 0x07),
            'k': g(N, base_seed ^ 0x08),
            'l': g(N, base_seed ^ 0x09),
        }

- zobrist_hash からローカル dict を排除：

    def zobrist_hash(...):
        self.init_zobrist(N)
        tbl = self.zobrist_tables[N]
        ...

- get_jasmin のキャッシュをフィールドへ：

    def get_jasmin(self, c: int, N: int) -> int:
        key = (c, N)
        if key in self.jasmin_cache:
            return self.jasmin_cache[key]
        v = self.jasmin(c, N)
        self.jasmin_cache[key] = v
        return v

- set_pre_queens_cached のキャッシュをフィールドへ：

    def set_pre_queens_cached(...):
        key: StateKey = (...)
        if key in self.subconst_cache:
            return
        self.set_pre_queens(...)
        self.subconst_cache.add(key)

■ テスト観点
- 代表値: N=5..17 の既知総数に一致（`expected` 比較）。
- キャッシュ有効化後も一致を満たすこと（衝突や誤枝刈りがないか）。
- .bin/.txt の保存/復元の相互一致。
- `jasmin()` 正規化の冪等性（jasmin(jasmin(x)) == jasmin(x)）。
"""


"""
17Py_constellations_integrate_codon.py（レビュー＆注釈つき）

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
23:   24233937684440            0  1 day, 8:12:58.977



workspace#suzuki$ bash MAIN.SH 15Py_constellations_optimize_codon.py
 N:        Total       Unique        hh:mm:ss.ms
17:     95815104            0         0:00:02.987
18:    666090624            0         0:00:21.549
19:   4968057848            0         0:02:43.514

workspace#suzuki$ bash MAIN.SH 17Py_constellations_integrate_codon_20251017_suzuki.py
 N:        Total       Unique        hh:mm:ss.ms
17:     95815104            0         0:00:04.037    ok
18:    666090624            0         0:00:29.301    ok
19:   4968057848            0         0:03:41.853    ok
"""

from typing import List,Set,Dict,Tuple
from datetime import datetime

StateKey=Tuple[int,int,int,int,int,int,int,int,int,int,int]

class NQueens17:
  def mix64(self,x:int)->int:
    """splitmix64 のミキサ最終段。64bit 値 x を (>>/XOR/乗算) の 3 段で拡散して返す。 Zobrist テーブルの乱数品質を担保するために使用。"""
    MASK64:int=(1<<64)-1 # 64bit マスク（Zobrist用途）
    x&=MASK64
    x=(x^(x>>30))*0xBF58476D1CE4E5B9&MASK64
    x=(x^(x>>27))*0x94D049BB133111EB&MASK64
    x^=(x>>31)
    return x&MASK64

  def gen_list(self,cnt:int,seed:int)->List[int]:
    """Zobrist テーブル用の 64bit 乱数を cnt 個生成してリストで返す。 seed は splitmix64 のインクリメント規約 (0x9E3779B97F4A7C15) に従って更新。"""
    MASK64:int=(1<<64)-1 # 64bit マスク（Zobrist用途）
    out:List[int]=[]
    s:int=seed&MASK64
    _mix64=self.mix64
    for _ in range(cnt):
      s=(s+0x9E3779B97F4A7C15)&MASK64   # splitmix64 のインクリメント
      # out.append(self._mix64(s))
      out.append(_mix64(s))
    return out

  def init_zobrist(self,N:int,zobrist_tables:Dict[int,Dict[str,List[int]]])->Dict[int,Dict[str,List[int]]]:
    """Zobrist テーブルを N ごとに初期化する。キーは 'ld','rd','col','LD','RD','row','queens','k','l'。 ※ キャッシュは self.zobrist_tables[N] に格納して再利用する。"""
    MASK64:int=(1<<64)-1 # 64bit マスク（Zobrist用途）
    if N in self.zobrist_tables:
      return
    base_seed:int=(0xC0D0_0000_0000_0000^(N<<32))&MASK64
    gen_list=self.gen_list
    tbl:Dict[str,List[int]]={
      'ld':gen_list(N,base_seed^0x01),
      'rd':gen_list(N,base_seed^0x02),
      'col':gen_list(N,base_seed^0x03),
      'LD':gen_list(N,base_seed^0x04),
      'RD':gen_list(N,base_seed^0x05),
      'row':gen_list(N,base_seed^0x06),
      'queens':gen_list(N,base_seed^0x07),
      'k':gen_list(N,base_seed^0x08),
      'l':gen_list(N,base_seed^0x09),
    }
    # self.zobrist_tables[N]=tbl
    return tbl

  def zobrist_hash(self,ld:int,rd:int,col:int,row:int,queens:int,k:int,l:int,LD:int,RD:int,N:int)->int:
    """(ld, rd, col, LD, RD, row, queens, k, l) を Zobrist Hash によって 64bit にまとめる。 各マスクは N ビットに正規化してから XOR 混合する。衝突確率の低い探索済み判定に利用可能。"""
    MASK64:int=(1<<64)-1
    zobrist_tables:Dict[int,Dict[str,List[int]]]={}
    tbl=self.init_zobrist(N,zobrist_tables)
    h=0
    mask=(1<<N)-1
    ld&=mask
    rd&=mask
    col&=mask
    LD&=mask
    RD&=mask
    m=ld;i=0
    while i<N:
      if (m&1)!=0:
        h^=tbl['ld'][i]
      m>>=1;i+=1
    m=rd;i=0
    while i<N:
      if (m&1)!=0:
        h^=tbl['rd'][i]
      m>>=1;i+=1
    m=col;i=0
    while i<N:
      if (m&1)!=0:
        h^=tbl['col'][i]
      m>>=1;i+=1
    m=LD;i=0
    while i<N:
      if (m&1)!=0:
        h^=tbl['LD'][i]
      m>>=1;i+=1
    m=RD;i=0
    while i<N:
      if (m&1)!=0:
        h^=tbl['RD'][i]
      m>>=1;i+=1
    if 0<=row<N:h^=tbl['row'][row]
    if 0<=queens<N:h^=tbl['queens'][queens]
    if 0<=k<N:h^=tbl['k'][k]
    if 0<=l<N:h^=tbl['l'][l]
    return h&MASK64

  """(i,j,k,l) を 5bit×4=20bit にパック/アンパックするユーティリティ。 mirvert は上下ミラー（行: N-1-?）＋ (k,l) の入れ替えで左右ミラー相当を実現。"""
  def to_ijkl(self,i:int,j:int,k:int,l:int)->int:return (i<<15)+(j<<10)+(k<<5)+l
  def mirvert(self,ijkl:int,N:int)->int:return self.to_ijkl(N-1-self.geti(ijkl),N-1-self.getj(ijkl),self.getl(ijkl),self.getk(ijkl))
  def ffmin(self,a:int,b:int)->int:return min(a,b)
  def geti(self,ijkl:int)->int:return (ijkl>>15)&0x1F
  def getj(self,ijkl:int)->int:return (ijkl>>10)&0x1F
  def getk(self,ijkl:int)->int:return (ijkl>>5)&0x1F
  def getl(self,ijkl:int)->int:return ijkl&0x1F
  """(i,j,k,l) パック値に対して盤面 90°/180° 回転を適用した新しいパック値を返す。 回転の定義: (r,c) -> (c, N-1-r)。対称性チェック・正規化に利用。"""
  def rot90(self,ijkl:int,N:int)->int:return ((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl)
  def rot180(self,ijkl:int,N:int)->int:return ((N-1-self.getj(ijkl))<<15)+((N-1-self.geti(ijkl))<<10)+((N-1-self.getl(ijkl))<<5)+(N-1-self.getk(ijkl))
  def symmetry(self,ijkl:int,N:int)->int:return 2 if self.symmetry90(ijkl,N) else 4 if self.geti(ijkl)==N-1-self.getj(ijkl) and self.getk(ijkl)==N-1-self.getl(ijkl) else 8
  def symmetry90(self,ijkl:int,N:int)->bool:return ((self.geti(ijkl)<<15)+(self.getj(ijkl)<<10)+(self.getk(ijkl)<<5)+self.getl(ijkl))==(((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl))
  """与えた (i,j,k,l) の 90/180/270° 回転形が既出集合 ijkl_list に含まれるかを判定する。"""
  def check_rotations(self,ijkl_list:Set[int],i:int,j:int,k:int,l:int,N:int)->bool:return any(rot in ijkl_list for rot in [((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i,((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+(N-1-k),(l<<15)+(k<<10)+((N-1-i)<<5)+(N-1-j)])

  """Jasmin 正規化のキャッシュ付ラッパ。盤面パック値 c を回転/ミラーで規約化した代表値を返す。 ※ キャッシュは self.jasmin_cache[(c,N)] に保持。"""
  """ [Opt-08] キャッシュ付き jasmin() のラッパー """
  def get_jasmin(self,c:int,N:int)->int:
    jasmin_cache:Dict[Tuple[int,int],int]={}
    key=(c,N)
    if key in jasmin_cache:
        return jasmin_cache[key]
    result=self.jasmin(c,N)
    jasmin_cache[key]=result
    return result

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
    _rot90=self.rot90
    for _ in range(arg):
      # ijkl=self.rot90(ijkl,N)
      ijkl=_rot90(ijkl,N)
    # 必要に応じて垂直方向のミラーリングを実行
    if self.getj(ijkl)<N-1-self.getj(ijkl):
      ijkl=self.mirvert(ijkl,N)
    return ijkl

  def dfs(self,functionid:int,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,board_mask:int,blockK_by_funcid:List[int],blockl_by_funcid:List[int],func_meta:List[Tuple[int,int,int]],N:int,N1:int,NK:int,NJ:int)->int:
    """汎用 DFS カーネル。古い SQ???? 関数群を 1 本化し、func_meta の記述に従って
  (1) mark 行での step=2/3 + 追加ブロック、(2) jmark 特殊、(3) ゴール判定、(4) +1 先読み
  を切り替える。引数:
    functionid: 現在の分岐モード ID（次の ID, パターン, 先読み有無は func_meta で決定）
    ld/rd/col:   斜線/列の占有
    row/free:    現在行と空きビット
    jmark/endmark/mark1/mark2: 特殊行/探索終端
    board_mask:  盤面全域のビットマスク
    blockK_by_funcid/blockl_by_funcid: 関数 ID に紐づく追加ブロック
    func_meta:   (next_id, funcptn, availptn) のメタ情報配列
    """
    # ローカル束縛（属性/関数参照を減らす）
    _dfs=self.dfs
    next_funcid,funcptn,avail_flag=func_meta[functionid]
    bit:int=0
    avail:int=free
    total:int=0
    step:int=1
    add1:int=0
    row_step:int=row+step
    use_blocks:bool=False       # blockK/blockl を噛ませるか
    use_future:bool=(avail_flag==1)  # _should_go_plus1 を使うか
    blockK:int=0
    blockl:int=0
    local_next_funcid:int=functionid    # 既定は自分
    # ==============================
    # ここから 1 本化した共通配置ループ # 既定値（通常の +1 前進ループ）
    # ==============================
    # --- P1/P2/P3: mark 行での step=2/3 ＋ block 適用を共通ループへ設定
    if funcptn in (0,1,2):
      at_mark:bool=(row==mark1) if funcptn in (0,2) else (row==mark2)
      if at_mark and avail:
        step:int=2 if funcptn in (0,1) else 3
        add1:int=1 if (funcptn==1 and functionid==20) else 0  # SQd1BlB のときだけ1
        row_step=row+step
        blockK:int=blockK_by_funcid[functionid]
        blockl:int=blockl_by_funcid[functionid]
        use_blocks:bool=True
        use_future:bool=False          # ここは従来どおり next_free だけで分岐
        local_next_funcid=next_funcid
    # ---- P6: endmark 基底
    elif funcptn==5 and row==endmark:
      if functionid==14:# SQd2B
        return 1 if (avail&(~1))>0 else 0
      return 1
    # --- P4: jmark 特殊を共通ループの前処理に畳み込む
    elif funcptn==3 and row==jmark:
      avail&=~1   # 列0禁止
      ld|=1       # 左斜線 LSB を立てる
      local_next_funcid=next_funcid  # 次関数へ
    # ---- P5: N-1-jmark 入口（行据え置きの一手前処理）
    elif funcptn==4:
      if row==N1-jmark:
        rd|=NJ # rd |= 1 << N1
        next_free:int=board_mask&~((ld<<1)|(rd>>1)|col)
        if next_free:
          total+=_dfs(next_funcid,ld<<1,rd>>1,col,row,next_free,jmark,endmark,mark1,mark2,board_mask,blockK_by_funcid,blockl_by_funcid,func_meta,N,N1,NK,NJ)
        return total # 続行（通常ループへ）

    #use_blocks / use_future の分岐ごとにループを分ける
    # ブーリアン分岐を内側ループに残すより、「使う/使わない」でループを2本に分けると分岐予測も最短経路になります
    if use_blocks:
      while avail:
        bit:int=avail&-avail
        avail&=avail-1
        next_ld,next_rd,next_col=(((ld|bit)<<step)|add1)|blockl,((rd|bit)>>step)|blockK,col|bit
        next_free:int=board_mask&~(next_ld|next_rd|next_col)
        if next_free:
          total+=_dfs(local_next_funcid,next_ld,next_rd,next_col,row_step,next_free,jmark,endmark,mark1,mark2,board_mask,blockK_by_funcid,blockl_by_funcid,func_meta,N,N1,NK,NJ)
    else:
      # “素の +1” だけ（先読みなし） # “+1 with 先読み”
      if use_future:
        while avail:
          bit:int=avail&-avail
          avail&=avail-1
          next_ld,next_rd,next_col=((ld|bit)<<step)|add1,(rd|bit)>>step,col|bit
          next_free:int=board_mask&~(next_ld|next_rd|next_col)
          if not next_free:
            continue
          if row_step>=endmark:
            total+=_dfs(local_next_funcid,next_ld,next_rd,next_col,row_step,next_free,jmark,endmark,mark1,mark2,board_mask,blockK_by_funcid,blockl_by_funcid,func_meta,N,N1,NK,NJ)
            continue
          # 先読み（+1）の中
          m1=1 if row_step==mark1 else 0
          m2=1 if row_step==mark2 else 0
          use_j=(funcptn==4)            # ★ P5ファミリのみ J 行を有効化
          mj=1 if (use_j and row_step==(N1-jmark)) else 0
          extra=((m1|m2)*NK)|(mj*NJ)
          future=board_mask&~(((next_ld<<1)|(next_rd>>1)|next_col)|extra)
          if future:
              total+=_dfs(local_next_funcid,next_ld,next_rd,next_col,row_step,next_free,jmark,endmark,mark1,mark2,board_mask,blockK_by_funcid,blockl_by_funcid,func_meta,N,N1,NK,NJ)
      else:
        while avail:
          bit:int=avail&-avail
          avail&=avail-1
          next_ld,next_rd,next_col=((ld|bit)<<step)|add1,(rd|bit)>>step,col|bit
          next_free:int=board_mask&~(next_ld|next_rd|next_col)
          if next_free:
              total+=_dfs(local_next_funcid,next_ld,next_rd,next_col,row_step,next_free,jmark,endmark,mark1,mark2,board_mask,blockK_by_funcid,blockl_by_funcid,func_meta,N,N1,NK,NJ)
    return total

  def exec_solutions(self,constellations:List[Dict[str,int]],N:int)->None:
    """各 Constellation（部分盤面）ごとに最適分岐（functionid）を選び、`dfs()` で解数を取得。 結果は `solutions` に書き込み、最後に `symmetry()` の重みで補正する。前段で SoA 展開し 並列化区間のループ体を軽量化。"""
    N2:int=N-2
    small_mask:int=(1<<N2)-1
    board_mask:int=(1<<N)-1
    dfs=self.dfs
    symmetry=self.symmetry
    getj,getk,getl=self.getj,self.getk,self.getl
    FUNC_CATEGORY={
      # N-3
      "SQBkBlBjrB":3,"SQBlkBjrB":3,"SQBkBjrB":3,
      "SQd2BkBlB":3,"SQd2BkB":3,"SQd2BlkB":3,
      "SQd1BkBlB":3,"SQd1BlkB":3,"SQd1BkB":3,"SQd0BkB":3,
      # N-4
      "SQBklBjrB":4,"SQd2BklB":4,"SQd1BklB":4,
      # 0（上記以外）
      "SQBlBjrB":0,"SQBjrB":0,"SQB":0,"SQBlBkBjrB":0,
      "SQBjlBkBlBjrB":0,"SQBjlBklBjrB":0,"SQBjlBlBkBjrB":0,"SQBjlBlkBjrB":0,
      "SQd2BlB":0,"SQd2B":0,"SQd2BlBkB":0,
      "SQd1BlB":0,"SQd1B":0,"SQd1BlBkB":0,"SQd0B":0
    }
    FID={
      "SQBkBlBjrB":0,"SQBlBjrB":1,"SQBjrB":2,"SQB":3,
      "SQBklBjrB":4,"SQBlBkBjrB":5,"SQBkBjrB":6,"SQBlkBjrB":7,
      "SQBjlBkBlBjrB":8,"SQBjlBklBjrB":9,"SQBjlBlBkBjrB":10,"SQBjlBlkBjrB":11,
      "SQd2BkBlB":12,"SQd2BlB":13,"SQd2B":14,"SQd2BklB":15,"SQd2BlBkB":16,
      "SQd2BkB":17,"SQd2BlkB":18,"SQd1BkBlB":19,"SQd1BlB":20,"SQd1B":21,
      "SQd1BklB":22,"SQd1BlBkB":23,"SQd1BlkB":24,"SQd1BkB":25,"SQd0B":26,"SQd0BkB":27
    }
    # next_funcid, funcptn, availptn の3つだけ持つ
    func_meta=[
      (1,0,0),#  0 SQBkBlBjrB   -> P1, 先読みなし
      (2,1,0),#  1 SQBlBjrB     -> P2, 先読みなし
      (3,3,1),#  2 SQBjrB       -> P4, 先読みあり
      (3,5,1),#  3 SQB          -> P6, 先読みあり
      (2,2,0),#  4 SQBklBjrB    -> P3, 先読みなし
      (6,0,0),#  5 SQBlBkBjrB   -> P1, 先読みなし
      (2,1,0),#  6 SQBkBjrB     -> P2, 先読みなし
      (2,2,0),#  7 SQBlkBjrB    -> P3, 先読みなし
      (0,4,1),#  8 SQBjlBkBlBjrB-> P5, 先読みあり
      (4,4,1),#  9 SQBjlBklBjrB -> P5, 先読みあり
      (5,4,1),# 10 SQBjlBlBkBjrB-> P5, 先読みあり
      (7,4,1),# 11 SQBjlBlkBjrB -> P5, 先読みあり
      (13,0,0),# 12 SQd2BkBlB    -> P1, 先読みなし
      (14,1,0),# 13 SQd2BlB      -> P2, 先読みなし
      (14,5,1),# 14 SQd2B        -> P6, 先読みあり（avail 特例）
      (14,2,0),# 15 SQd2BklB     -> P3, 先読みなし
      (17,0,0),# 16 SQd2BlBkB    -> P1, 先読みなし
      (14,1,0),# 17 SQd2BkB      -> P2, 先読みなし
      (14,2,0),# 18 SQd2BlkB     -> P3, 先読みなし
      (20,0,0),# 19 SQd1BkBlB    -> P1, 先読みなし
      (21,1,0),# 20 SQd1BlB      -> P2, 先読みなし（add1=1 は dfs 内で特別扱い）
      (21,5,1),# 21 SQd1B        -> P6, 先読みあり
      (21,2,0),# 22 SQd1BklB     -> P3, 先読みなし
      (25,0,0),# 23 SQd1BlBkB    -> P1, 先読みなし
      (21,2,0),# 24 SQd1BlkB     -> P3, 先読みなし
      (21,1,0),# 25 SQd1BkB      -> P2, 先読みなし
      (26,5,1),# 26 SQd0B        -> P6, 先読みあり
      (26,0,0),# 27 SQd0BkB      -> P1, 先読みなし
    ]
    n3=1<<max(0,N-3)   # 念のため負シフト防止
    n4=1<<max(0,N-4)
    size=max(FID.values())+1
    blockK_by_funcid=[0]*size
    blockl_by_funcid=[0,1,0,0,1,1,0,2,0,0,0,0,0,1,0,1,1,0,2,0,0,0,1,1,2,0,0,0]
    for fn,cat in FUNC_CATEGORY.items():# FUNC_CATEGORY: {関数名: 3 or 4 or 0}
      fid=FID[fn]
      blockK_by_funcid[fid]=n3 if cat==3 else (n4 if cat==4 else 0)
    # ===== 前処理ステージ（単一スレッド） =====
    m=len(constellations)
    # SoA（Structure of Arrays）に展開：並列本体が軽くなる
    ld_arr=[0]*m;rd_arr=[0]*m;col_arr=[0]*m
    row_arr=[0]*m;free_arr=[0]*m
    jmark_arr=[0]*m;end_arr=[0]*m
    mark1_arr=[0]*m;mark2_arr=[0]*m
    funcid_arr=[0]*m
    ijkl_arr=[0]*m   # symmetry 計算用
    N1=N-1
    NK=1<<(N-3)
    NJ=1<<N1
    results=[0]*m
    target:int=0
    for i,constellation in enumerate(constellations):
      jmark=mark1=mark2=0
      start_ijkl=constellation["startijkl"]
      start=start_ijkl>>20
      ijkl=start_ijkl&((1<<20)-1)
      j,k,l=getj(ijkl),getk(ijkl),getl(ijkl)
      ld,rd,col=constellation["ld"]>>1,constellation["rd"]>>1,(constellation["col"]>>1)|(~small_mask)
      LD=(1<<(N1-j))|(1<<(N1-l))
      ld|=LD>>(N-start)
      if start>k:
        rd|=(1<<(N1-(start-k+1)))
      if j>=2*N-33-start:
        rd|=(1<<(N1-j))<<(N2-start)
      free=~(ld|rd|col)
      if j<(N-3):
        jmark,endmark=j+1,N2
        if j>2*N-34-start:
          if k<l:
            mark1,mark2=k-1,l-1
            if start<l:
              if start<k:
                if l!=k+1:target=0 # SQBkBlBjrB
                else:target=4 # SQBklBjrB
              else:target=1 #_SQBlBjrB
            else:target=2 # SQBjrB
          else:
            mark1,mark2=l-1,k-1
            if start<k:
              if start<l:
                if k!=l+1:target=5 # SQBlBkBjrB
                else:target=7 # SQBlkBjrB
              else:target=6 # SQBkBjrB
            else:target=2 # SQBjrB
        else:
          if k<l:
            mark1,mark2=k-1,l-1
            if l!=k+1:target=8 # SQBjlBkBlBjrB
            else:target=9 # SQBjlBklBjrB
          else:
            mark1,mark2=l-1,k-1
            if k!=l+1:target=10 # SQBjlBlBkBjrB
            else:target=11 # SQBjlBlkBjrB
      elif j==(N-3):
        endmark=N2
        if k<l:
          mark1,mark2=k-1,l-1
          if start<l:
            if start<k:
              if l!=k+1:target=12 # SQd2BkBlB
              else:target=15 # SQd2BklB
            else:
              mark2=l-1
              target=13 # SQd2BlB
          else:target=14 # SQd2B
        else:
          mark1,mark2=l-1,k-1
          if start<k:
            if start<l:
              if k!=l+1:target=16 # SQd2BlBkB
              else:target=18 # SQd2BlkB
            else:
              mark2=k-1
              target=17 # SQd2BkB
          else:target=14 # SQd2B
      elif j==N2:# jがコーナーから1列内側
        if k<l:
          endmark=N2
          if start<l:
            if start<k:
              mark1=k-1
              if l!=k+1:
                mark2=l-1
                target=19 # SQd1BkBlB
              else:target=22 # SQd1BklB
            else:
              mark2=l-1
              target=20 # SQd1BlB
          else:target=21 # SQd1B
        else:# l < k
          if start<k:
            if start<l:
              if k<N2:
                mark1,endmark=l-1,N2
                if k!=l+1:
                  mark2=k-1
                  target=23 # SQd1BlBkB
                else:target=24 # SQd1BlkB
              else:
                if l!=(N-3):
                  mark2,endmark=l-1,N-3
                  target=20 # SQd1BlB
                else:
                  endmark=N-4
                  target=21 # SQd1B
            else:
              if k!=N2:
                mark2,endmark=k-1,N2
                target=25 # SQd1BkB
              else:
                endmark=N-3
                target=21 # SQd1B
          else:
            endmark=N2
            target=21 # SQd1B
      else:# j がコーナー
        endmark=N2
        if start>k:
          target=26 # SQd0B
        else:
          mark1=k-1
          target=27 # SQd0BkB
      # 配列へ格納
      ld_arr[i],rd_arr[i],col_arr[i]=ld,rd,col
      row_arr[i],free_arr[i]=start,free
      jmark_arr[i],end_arr[i]=jmark,endmark
      mark1_arr[i],mark2_arr[i]=mark1,mark2
      funcid_arr[i]=target
      ijkl_arr[i]=ijkl
    # ===== 並列ステージ：計算だけ =====
    # cnt=dfs(target,ld,rd,col,start,free,jmark,endmark,mark1,mark2,board_mask,blockK_by_funcid,blockl_by_funcid,func_meta,N)
    @par
    for i in range(m):
      cnt=dfs(funcid_arr[i],ld_arr[i],rd_arr[i],col_arr[i],row_arr[i],free_arr[i],jmark_arr[i],end_arr[i],mark1_arr[i],mark2_arr[i],board_mask,blockK_by_funcid,blockl_by_funcid,func_meta,N,N1,NK,NJ)
      results[i]=cnt*symmetry(ijkl_arr[i],N)

    # ===== 書き戻し（単一スレッド） =====
    # constellation["solutions"]=cnt*symmetry(ijkl,N)
    for i,constellation in enumerate(constellations):
      constellation["solutions"]=results[i]

  def set_pre_queens_cached(self,ld:int,rd:int,col:int,k:int,l:int,row:int,queens:int,LD:int,RD:int,counter:List[int],constellations:List[Dict[str,int]],N:int,preset_queens:int,visited:Set[int],constellation_signatures:Set[Tuple[int,int,int,int,int,int]]):
    """サブコンステレーション生成のキャッシュ付ラッパ。StateKey で一意化し、 同一状態での重複再帰を回避して生成量を抑制する。"""
    key:StateKey=(ld,rd,col,k,l,row,queens,LD,RD,N,preset_queens)
    subconst_cache:Set[StateKey]=set()
    if key in subconst_cache:
      # 以前に同じ状態で生成済み → 何もしない（または再利用）
      return
    # 新規実行（従来通りset_pre_queensの本体処理へ）
    self.set_pre_queens(ld,rd,col,k,l,row,queens,LD,RD,counter,constellations,N,preset_queens,visited,constellation_signatures)
    # subconst_cache[key] = True  # マークだけでOK
    subconst_cache.add(key)

  def set_pre_queens(self,ld:int,rd:int,col:int,k:int,l:int,row:int,queens:int,LD:int,RD:int,counter:list,constellations:List[Dict[str,int]],N:int,preset_queens:int,visited:Set[int],constellation_signatures:Set[Tuple[int,int,int,int,int,int]])->None:
    """事前に置く行 (k,l) を強制しつつ、queens==preset_queens に到達するまで再帰列挙。 `visited` には軽量な `state_hash` を入れて枝刈り。到達時は {ld,rd,col,startijkl} を constellation に追加。"""
    mask=(1<<N)-1  # setPreQueensで使用
    # ---------------------------------------------------------------------
    # 状態ハッシュによる探索枝の枝刈り バックトラック系の冒頭に追加　やりすぎると解が合わない
    # <>zobrist_hash
    # 各ビットを見てテーブルから XOR するため O(N)（ld/rd/col/LD/RDそれぞれで最大 N 回）。
    # とはいえ N≤17 なのでコストは小さめ。衝突耐性は高い。
    # マスク漏れや負数の扱いを誤ると不一致が起きる点に注意（先ほどの & ((1<<N)-1) 修正で解決）。
    # h: int = self.zobrist_hash(ld, rd, col, row, queens, k, l, LD, RD, N)
    # <>state_hash
    # その場で数個の ^ と << を混ぜるだけの O(1) 計算。
    # 生成されるキーも 単一の int なので、set/dict の操作が最速＆省メモリ。
    # ただし理論上は衝突し得ます（実際はN≤17の範囲なら実害が出にくい設計にしていればOK）。
    # [Opt-09] Zobrist Hash（Opt-09）の導入とその用途
    # ビットボード設計でも、「盤面のハッシュ」→「探索済みフラグ」で枝刈りは可能です。
    state_hash:int=(ld<<3)^(rd<<2)^(col<<1)^row^(queens<<7)^(k<<12)^(l<<17)^(LD<<22)^(RD<<27)^(N<<1)
    h:int=state_hash
    if h in visited:
      return
    visited.add(h)
    # <>StateKey（タプル）
    # 11個の整数オブジェクトを束ねるため、オブジェクト生成・GC負荷・ハッシュ合成が最も重い。
    # set の比較・保持も重く、メモリも一番食います。
    # 衝突はほぼ心配ないものの、速度とメモリ効率は最下位。
    # key: StateKey = (ld, rd, col, row, queens, k, l, LD, RD)
    # if key in visited:
    #     return
    # visited.add(key)
    # ---------------------------------------------------------------------
    # k行とl行はスキップ
    if row==k or row==l:
      self.set_pre_queens_cached(ld<<1,rd>>1,col,k,l,row+1,queens,LD,RD,counter,constellations,N,preset_queens,visited,constellation_signatures)
      return
    # クイーンの数がpreset_queensに達した場合、現在の状態を保存
    if queens==preset_queens:
      # signatureの生成
      signature=(ld,rd,col,k,l,row)  # 必要な変数でOK
      if not hasattr(self,"constellation_signatures"):
        constellation_signatures=set()
      signatures=constellation_signatures
      if signature not in signatures:
        constellation={"ld":ld,"rd":rd,"col":col,"startijkl":row<<20,"solutions":0}
        constellations.append(constellation) #星座データ追加
        signatures.add(signature)
        counter[0]+=1
      return
    # 現在の行にクイーンを配置できる位置を計算
    free=~(ld|rd|col|(LD>>(N-1-row))|(RD<<(N-1-row)))&mask
    _set_pre_queens_cached=self.set_pre_queens_cached
    while free:
      bit:int=free&-free
      free&=free-1
      _set_pre_queens_cached((ld|bit)<<1,(rd|bit)>>1,col|bit,k,l,row+1,queens+1,LD,RD,counter,constellations,N,preset_queens,visited,constellation_signatures)

  def gen_constellations(self,ijkl_list:Set[int],constellations:List[Dict[str,int]],N:int,preset_queens:int)->None:
    """開始コンステレーション（代表部分盤面）の列挙。中央列（奇数 N）特例、回転重複排除 （`check_rotations`）、Jasmin 正規化（`get_jasmin`）を経て、各 sc から `set_pre_queens_cached` でサブ構成を作る。"""
    halfN=(N+1)//2  # Nの半分を切り上げ
    N1:int=N-1
    constellation_signatures: Set[ Tuple[int, int, int, int, int, int] ] = set()
    # --- [Opt-03] 中央列特別処理（奇数Nの場合のみ） ---
    if N%2==1:
      center=N//2
      ijkl_list.update(
        self.to_ijkl(i,j,center,l)
        for l in range(center+1,N1)
        for i in range(center+1,N1)
        if i!=(N1)-l
        for j in range(N-center-2,0,-1)
        if j!=i and j!=l
        if not self.check_rotations(ijkl_list,i,j,center,l,N)
      )
    # --- [Opt-03] 中央列特別処理（奇数Nの場合のみ） ---
    # コーナーにクイーンがいない場合の開始コンステレーションを計算する
    ijkl_list.update(self.to_ijkl(i,j,k,l) for k in range(1,halfN) for l in range(k+1,N1) for i in range(k+1,N-1) if i!=(N-1)-l for j in range(N-k-2,0,-1) if j!=i and j!=l if not self.check_rotations(ijkl_list,i,j,k,l,N))
    # コーナーにクイーンがある場合の開始コンステレーションを計算する
    ijkl_list.update({self.to_ijkl(0,j,0,l) for j in range(1,N-2) for l in range(j+1,N1)})
    # Jasmin変換
    ijkl_list={self.get_jasmin(c,N) for c in ijkl_list}
    L=1<<(N1)  # Lは左端に1を立てる
    # ローカルアクセスに変更
    geti,getj,getk,getl=self.geti,self.getj,self.getk,self.getl
    to_ijkl=self.to_ijkl
    _set_pre_queens_cached=self.set_pre_queens_cached
    for sc in ijkl_list:
      # ここで毎回クリア（＝この sc だけの重複抑止に限定）
      constellation_signatures=set()
      i,j,k,l=geti(sc),getj(sc),getk(sc),getl(sc)
      Lj=L>>j;Li=L>>i;Ll=L>>l
      ld=((L>>(i-1)) if i>0 else 0)|(1<<(N-k))
      rd=(L>>(i+1))|(1<<(l-1))
      col=1|L|Li|Lj
      LD=Lj|Ll
      RD=Lj|(1<<k)
      counter:List[int]=[0] # サブコンステレーションを生成
      visited:Set[int]=set()
      _set_pre_queens_cached(ld,rd,col,k,l,1,3 if j==N1 else 4,LD,RD,counter,constellations,N,preset_queens,visited,constellation_signatures)
      current_size=len(constellations)
      base=to_ijkl(i,j,k,l)
      for a in range(counter[0]):
        constellations[-1-a]["startijkl"]|=base

  """constellations の各要素が {ld, rd, col, startijkl} を全て持つかを検証する。"""
  def validate_constellation_list(self,constellations:List[Dict[str,int]])->bool: return all(all(k in c for k in ("ld","rd","col","startijkl")) for c in constellations)

  """32bit little-endian の相互変換ヘルパ。Codon/CPython の差異に注意。"""
  def read_uint32_le(self,b:str)->int: return (ord(b[0])&0xFF)|((ord(b[1])&0xFF)<<8)|((ord(b[2])&0xFF)<<16)|((ord(b[3])&0xFF)<<24)
  def int_to_le_bytes(self,x:int)->List[int]: return [(x>>(8*i))&0xFF for i in range(4)]

  def file_exists(self,fname:str)->bool:
    """ファイル存在チェック（読み取り open の可否で判定）。"""
    try:
      with open(fname,"rb"):
        return True
    except:
      return False

  def validate_bin_file(self,fname:str)->bool:
    """bin キャッシュのサイズ妥当性確認（1 レコード 16 バイトの整数倍か）。"""
    try:
      with open(fname,"rb") as f:
        f.seek(0,2)  # ファイル末尾に移動
        size=f.tell()
      return size%16==0
    except:
      return False

  """テキスト形式で constellations を保存/復元する（1 行 5 数値: ld rd col startijkl solutions）。"""
  def save_constellations_txt(self,path:str,constellations:List[Dict[str,int]])->None:
    with open(path,"w") as f:
      for c in constellations:
        ld=c["ld"]
        rd=c["rd"]
        col=c["col"]
        startijkl=c["startijkl"]
        solutions=c.get("solutions",0)
        f.write(f"{ld} {rd} {col} {startijkl} {solutions}\n")

  """テキスト形式で constellations を保存/復元する（1 行 5 数値: ld rd col startijkl solutions）。"""
  def load_constellations_txt(self,path:str)->List[Dict[str,int]]:
    out:List[Dict[str,int]]=[]
    with open(path,"r") as f:
      for line in f:
        parts=line.strip().split()
        if len(parts)!=5:
          continue
        ld=int(parts[0]);rd=int(parts[1]);col=int(parts[2])
        startijkl=int(parts[3]);solutions=int(parts[4])
        out.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions": solutions})
    return out

  """テキストキャッシュを読み込み。壊れていれば `gen_constellations()` で再生成して保存する。"""
  def load_or_build_constellations_txt(self,ijkl_list:Set[int],constellations,N:int,preset_queens:int)->List[Dict[str,int]]:
    # N と preset_queens に基づいて一意のファイル名を構成
    fname=f"constellations_N{N}_{preset_queens}.txt"
    # ファイルが存在すれば読み込むが、破損チェックも行う
    if self.file_exists(fname):
      try:
        constellations=self.load_constellations_txt(fname)
        if self.validate_constellation_list(constellations):
          return constellations
        else:
          print(f"[警告] 不正なキャッシュ形式: {fname} を再生成します")
      except Exception as e:
        print(f"[警告] キャッシュ読み込み失敗: {fname}, 理由: {e}")
    # ファイルがなければ生成・保存
    constellations:List[Dict[str,int]]=[]
    self.gen_constellations(ijkl_list,constellations,N,preset_queens)
    self.save_constellations_txt(fname,constellations)
    return constellations

  """bin 形式で constellations を保存/復元。Codon では str をバイト列として扱う前提のため、CPython では bytes で書き込むよう分岐/注意が必要。"""
  def save_constellations_bin(self,fname:str,constellations:List[Dict[str,int]])->None:
    _int_to_le_bytes=self.int_to_le_bytes
    with open(fname,"wb") as f:
      for d in constellations:
        for key in ["ld","rd","col","startijkl"]:
          b=_int_to_le_bytes(d[key])
          _int_to_le_bytes(d[key])
          f.write("".join(chr(c) for c in b))  # Codonでは str がバイト文字列扱い

  """bin 形式で constellations を保存/復元。Codon では str をバイト列として扱う前提のため、CPython では bytes で書き込むよう分岐/注意が必要。"""
  def load_constellations_bin(self,fname:str)->List[Dict[str,int]]:
    constellations:List[Dict[str,int]]=[]
    _read_uint32_le=self.read_uint32_le
    with open(fname,"rb") as f:
      while True:
        raw=f.read(16)
        if len(raw)<16:
          break
        ld=self.read_uint32_le(raw[0:4])
        rd=self.read_uint32_le(raw[4:8])
        col=self.read_uint32_le(raw[8:12])
        startijkl=_read_uint32_le(raw[12:16])
        constellations.append({ "ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0 })
    return constellations

  def load_or_build_constellations_bin(self,ijkl_list:Set[int],constellations,N:int,preset_queens:int)->List[Dict[str,int]]:
    """bin キャッシュを読み込み。検証に失敗した場合は再生成して保存し、その結果を返す。"""
    fname=f"constellations_N{N}_{preset_queens}.bin"
    if self.file_exists(fname):
      constellations=self.load_constellations_bin(fname)
      # ファイルが存在すれば読み込むが、破損チェックも行う
      try:
        constellations=self.load_constellations_bin(fname)
        if self.validate_bin_file(fname) and self.validate_constellation_list(constellations):
          return constellations
        else:
          print(f"[警告] 不正なキャッシュ形式: {fname} を再生成します")
      except Exception as e:
        print(f"[警告] キャッシュ読み込み失敗: {fname}, 理由: {e}")
    # ファイルがなければ生成・保存
    constellations:List[Dict[str,int]]=[]
    self.gen_constellations(ijkl_list,constellations,N,preset_queens)
    self.save_constellations_bin(fname,constellations)
    return constellations

class NQueens17_constellations():
  # 小さなNは正攻法で数える（対称重みなし・全列挙）
  def _bit_total(self,size:int)->int:
    """小さな N 用の素朴な全列挙（対称重みなし）。ビットボードで列/斜線の占有を管理して再帰的に合計を返す。検算/フォールバック用。"""
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
        bitmap^=bit
        bt(row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
    bt(0,0,0,0)
    return total

  def main(self)->None:
    """N=5..17 の合計解を計測。N<=5 は `_bit_total()` のフォールバック、それ以外は星座キャッシュ（.bin/.txt）→ `exec_solutions()` → 合計→既知解 `expected` と照合。"""
    nmin:int=5
    nmax:int=18
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
      # NQ=NQueens17(size)
      NQ=NQueens17()
      #---------------------------------
      # 星座リストそのものをキャッシュ
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
      total:int=sum(c['solutions'] for c in constellations if c['solutions']>0)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      # print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")
      expected:List[int]=[0,0,0,0,0,10,4,40,92,352,724,2680,14200,73712,365596,2279184,14772512,95815104,666090624,4968057848]
      status:str="ok" if expected[size]==total else f"ng({total}!={expected[size]})"
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}    {status}")

if __name__=="__main__":
  NQueens17_constellations().main()
