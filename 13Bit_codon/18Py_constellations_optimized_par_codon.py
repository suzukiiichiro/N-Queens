#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
コンステレーション版 最適化+最速化（@par) 　Ｎクイーン

タグ    方針    対応状況    補足
✅[Opt-01]    ビット演算枝刈り    達成    全探索・部分盤面生成のすべてでbit演算徹底
✅[Opt-02]    左右対称性除去（左半分探索）    達成    初手左半分/コーナー分岐で重複生成排除
✅[Opt-03]    中央列特別処理（奇数N）    達成    奇数N中央列を専用内包表記で排除
✅[Opt-04]    180°対称除去    達成    rot180_in_set で内包時点で重複除去
✅[Opt-05]    角位置分岐・COUNT分類    達成    コーナー分岐/symmetryでCOUNT2/4/8分類
✅[Opt-06]    並列処理（初手分割）    未達成（現状は未実装）    これは現状は未実装 27Py_で実装
✅[Opt-07]    1行目以外の部分対称除去    達成    jasmin/is_partial_canonicalで排除
✅[Opt-08]    軽量is_canonical・キャッシュ    達成    Zobrist/jasmin/hash系でメモ化
✅[Opt-09]    Zobrist Hash    達成    Zobrist導入済
✅[Opt-10]    マクロチェス（局所パターン）    達成    violate_macro_patterns関数（導入済ならOK）
 ✔[Opt-11]    ミラー+90°回転重複排除    原則不要「あえてやらない」設計。必要ならis_canonicalで激重に
✅[Opt-12]    キャッシュ構造設計

検討課題を「実装難易度の低い順」に並べ替え
✅済[Opt-01]  ビット演算による衝突枝刈り（cols/hills/dales）
→ set_pre_queensや他の再帰でld|rd|colのビット演算を用いた枝刈りを徹底している

✅済[Opt-02] 左右対称性除去（1 行目の列を 0～n//2−1 に制限）
→ gen_constellationsのfor k in range(1, halfN)や、角コーナー分岐で左右対称盤面の重複生成を抑制
→ コーナーあり/なし両方をしっかり区分

✅済[Opt-03] 中央列の特別処理（N奇数時）
if n % 2 == 1: ... で中央列のみ個別に探索・重複排除していればOK
達成済み
 halfN=(N+1)//2  # Nの半分を切り上げ
 の直後に以下を貼り付ける

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
      )
    # --- [Opt-03] 中央列特別処理（奇数Nの場合のみ） ---

✅済[Opt-04] 180°対称除去
rot180_in_set で「180度回転盤面が既にセットにある場合はスキップ」できている。クイーン配置を180度回転したものの重複カウントはすべて生成段階で排除できている

  #---------------------------------
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
  #---------------------------------

  #---------------------------------
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
  #---------------------------------


✅済[Opt-05] 角位置（col==0）分岐＆対称分類（COUNT2/4/8）
「1行目col==0」や「角位置」だけを個別分岐しているか
対称性カウント（COUNT2/4/8分類）で「同型解数」の判定ができているか
→ コーナー（i=0やk=0）専用の初期コンステレーション生成あり。
→ symmetryやjasmin関数でCOUNT分類もサポート


🟡[Opt-06] 並列処理（初手ごとにmultiprocessing、または星座ごとに分割）
→未達成（or 未確認）
ただし、「初手分割」「constellationsごとに独立処理」が意図されているので、exec_solutions側で並列for等が入っていれば事実上達成


❎[Opt-07] 1行目以外でも部分対称除去（行列単位）
途中段階（深さ r の盤面）を都度「辞書順最小」の canonical かどうかチェックして、そうでなければ枝刈り
→ 各 SQ〜() の再帰関数の while free: の直前にこの判定を入れ、False なら continue。
結論：board変数にrowのは位置情報を格納していないので対応不可

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

❎[Opt-08,09] 軽量 is_canonical() による“部分盤面”の辞書順最小チェックを高速化（キャッシュ/軽量版）
「完成盤」だけでなく“部分盤面”用に軽量な変換（行の回転・反転は途中情報だけで可）を実装。
→ is_partial_canonical() の中で zobrist_cache[hash] = True/False として使う

✅[Opt-09] Zobrist Hash による transposition / visited 状態の高速検出

1. Zobrist Hash（Opt-09）の導入とその用途
ビットボード設計でも、「盤面のハッシュ」→「探索済みフラグ」で枝刈りは可>能です。
例えば「既に同じビットマスク状態を訪問したか」判定、もしくは部分盤面パタ>ーンのメモ化など。

#------------------------------
def state_hash(ld: int, rd: int, col: int, row: int) -> int:
    # 単純な状態ハッシュ（高速かつ衝突率低めなら何でも可）
    return (ld * 0x9e3779b9) ^ (rd * 0x7f4a7c13) ^ (col * 0x6a5d39e9) ^ row
#------------------------------
# 1.state_hash関数（Codon/Python両対応）
def state_hash(ld: int, rd: int, col: int, row: int) -> int:
    # codon は 64bit int 算術も高速
    return (ld * 0x9e3779b9) ^ (rd * 0x7f4a7c13) ^ (col * 0x6a5d39e9) ^ row

# 2.solve などの関数でset()を使う
visited: set[int] = set()
self.set_pre_queens(ld, rd, col, k, l, 1, 3 if j==N-1 else 4, LD, RD, counter, constellations, N, preset_queens, visited)

# 3.visited セット（型注釈つき）を solveやmainで用意し渡す
visited: set[int] = set()
self.set_pre_queens(ld, rd, col, k, l, 1, 3 if j==N-1 else 4, LD, RD, counter, constellations, N, preset_queens, visited)

# 4.set_pre_queens の再帰先頭に挿入
def set_pre_queens(self, ld: int, rd: int, col: int, k: int, l: int, row: int, queens: int, LD: int, RD: int, counter: list, constellations: list, N: int, preset_queens: int, visited: set[int]) -> None:
    mask: int = (1 << N) - 1
    # 状態ハッシュによる探索枝の枝刈り
    h: int = state_hash(ld, rd, col, row)
    if h in visited:
        return
    visited.add(h)
    # ...（この後従来の処理を続ける）
#------------------------------


❎[Opt-10] マクロチェス（局所パターン）による構築制限
→ violate_macro_patternsのようなローカルな局所配置判定関数を挟む設計で達成
結論：board変数にrowのは位置情報を格納していないので対応不可

# ---------------------------
# [Opt-09] Zobrist Hash テーブル生成（初期化）
def init_zobrist(N: int) -> List[List[int]]:
    import random
    return [[random.getrandbits(64) for _ in range(N)] for _ in range(N)]

# ハッシュ計算
def compute_hash(board: List[int], row: int, zobrist: List[List[int]]) -> int:
    h = 0
    for r in range(row):
        h ^= zobrist[r][board[r]]
    return h
# ---------------------------
# [Opt-09] 部分盤面の正準性チェック + Zobristキャッシュ
# ---------------------------
def is_partial_canonical(board: List[int], row: int, N: int,zobrist: List[List[int]], zcache: dict) -> bool:
    key = compute_zobrist_hash(board, row, zobrist)
    if key in zcache:
        return zcache[key]

    current = tuple(board[:row])
    # ミラー反転のみチェック（左右対称のみ）
    mirrored = tuple(N - 1 - board[r] for r in range(row))

    # 必要であれば回転90/180/270 も加える（今はミラーのみ）
    minimal = min(current, mirrored)
    result = (current == minimal)
    zcache[key] = result
    return result
# -----------------------------------
# [Opt-10] ユーザー定義のマクロチェスルール
def violate_macro_patterns(board: List[int], row: int, N: int) -> bool:
    # 例：上2行に中央列配置が連続する場合、除外
    if row >= 2 and abs(board[row-1] - board[row-2]) <= 1:
        return True
    return False
# -----------------------------------

# -----------------------------------
# [Opt-7,8,9,10]の実装
# 各 backtrack 系の関数の while free: ループ手前に以下を挿入
# [Opt-07/08] 部分盤面の辞書順最小性チェック（canonical）による枝刈り
if not is_partial_canonical(self.BOARD, row, N, self.zobrist, self.zcache):
    return
# [Opt-10] 局所配置のパターン（マクロチェス）による枝刈り
if violate_macro_patterns(self.BOARD, row, N):
    return
# -----------------------------------


# -----------------------------------
# （例）
116   def SQd0B(self,ld:int,rd:int,col:int,row:int,free:int,jmark:int,endmark:int,mark1:int,mark2:int,tempcounter:list[int],N:int)->None:
117     if row==endmark:
118       tempcounter[0]+=1
119       return
120     # [Opt-07/08] 部分盤面の辞書順最小性チェック（canonical）による枝刈り
        if not is_partial_canonical(self.BOARD, row, N, self.zobrist, self.zcache):
            return
121     # [Opt-10] 局所配置のパターン（マクロチェス）による枝刈り
        if violate_macro_patterns(self.BOARD, row, N):
            return
122     while free:
123       bit:int=free&-free
124       ...
# -----------------------------------


✅ビット演算のインライン化

# -----------------------------------
while free:
    bit = free & -free
    free ^= bit
    next_ld = (ld | bit) << 1
    next_rd = (rd | bit) >> 1
    next_col = col | bit
    SQd0B(next_ld, next_rd, next_col, ...)
# -----------------------------------
↓
# -----------------------------------
while free:
    bit = free & -free
    free ^= bit
    SQd0B((ld | bit) << 1, (rd | bit) >> 1, col | bit, ...)
# -----------------------------------


# -----------------------------------
  def check_rotations(self,ijkl_list:Set[int],i:int,j:int,k:int,l:int,N:int)->bool:
    rot90=((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i
    rot180=((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+(N-1-k)
    rot270=(l<<15)+(k<<10)+((N-1-i)<<5)+(N-1-j)
    return any(rot in ijkl_list for rot in (rot90,rot180,rot270))
# -----------------------------------
↓
# -----------------------------------
  def check_rotations(self,ijkl_list:Set[int],i:int,j:int,k:int,l:int,N:int)->bool:
    return any(rot in ijkl_list for rot in [((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i,((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+(N-1-k), (l<<15)+(k<<10)+((N-1-i)<<5)+(N-1-j)])
# -----------------------------------

# -----------------------------------
def symmetry90(self,ijkl:int,N:int)->bool:
    return ((self.geti(ijkl)<<15)+(self.getj(ijkl)<<10)+(self.getk(ijkl)<<5)+self.getl(ijkl))==(((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl))
# -----------------------------------
def symmetry(self, ijkl: int, N: int) -> int:
  i, j, k, l = self.geti(ijkl), self.getj(ijkl), self.getk(ijkl), self.getl(ijkl)
  if self.symmetry90(ijkl, N):
      return 2
  elif i == N - 1 - j and k == N - 1 - l:
      return 4
  else:
      return 8
# -----------------------------------
↓
# -----------------------------------
def symmetry(self,ijkl:int,N:int)->int:
  return 2 if self.symmetry90(ijkl,N) else 4 if self.geti(ijkl)==N-1-self.getj(ijkl) and self.getk(ijkl)==N-1-self.getl(ijkl) else 8
# -----------------------------------


🟡[Opt-11] 構築時「ミラー＋90°回転」重複排除
これはほとんどの実用系N-Queens実装で“わざとやらない”ことが多い
**「途中盤面を毎回ミラー＋90°回転して辞書順最小か判定」するもので、ビットボード高速化設計と両立させるのは実装もコストも非常に高い**です。理論的には“究極の重複排除”ですが、実用的には「やり過ぎ」になるため、**ほぼ全ての高速N-Queens実装で“わざと導入しない”**ことが標準です。

✅[Opt-12]キャッシュ構造設計
部分盤面や星座をhash/tuple化し、dictでキャッシュ
1度計算したhash値（zobristやtuple）をもとに重複判定
同じ状態は“必ず再利用”
階層構造（部分再帰木ごとにキャッシュ分離）も有効

「同じ状態は必ず再利用」＝探索の「指数的重複」を爆速カット
とくにN-Queensのような「部分盤面でパターン重複が激しい問題」は
キャッシュ再利用で速度が何桁も違う

Zobrist hashやtuple keyによる「整数インデックス付きdict」は
最強のメモリ効率＆スピード両立手法

🧑gen_constellationsにキャッシュやhashを活かすには？
星座リスト・盤面生成ごとに「Zobrist hash」や「tuple化キー」を用意し、一度計算した結果をdictで使い回す
jasmin変換など高コスト処理もdictキャッシュで「1度だけ」計算・以降再利用。部分再帰やサブコンステレーション分岐も「盤面シグネチャ」をkeyにキャッシュ設計


# ------------------------------------------------
🟡1. Jasmin変換キャッシュ（クラス属性またはグローバル変数で）
（生成済み盤面の再利用）
ijkl_list_jasmin = {self.jasmin(c, N) for c in ijkl_list} も、盤面→jasmin変換は「一度計算したらdictでキャッシュ」が効果大

#グローバル変数で
jasmin_cache = {}

def get_jasmin(self, c: int, N: int) -> int:
    key = (c, N)
    if key in jasmin_cache:
        return jasmin_cache[key]
    result = self.jasmin(c, N)
    jasmin_cache[key] = result
    return result

# 使用例:gen_constellations()内に 
ijkl_list_jasmin = {self.get_jasmin(c, N) for c in ijkl_list}
# ------------------------------------------------

# ------------------------------------------------
🟡2. 星座生成（サブコンステレーション）にtuple keyでキャッシュ
set_pre_queens やサブ星座生成は、状態変数を tuple でまとめて key にできます。これで全く同じ状態での星座生成は1度だけ実行されます。

#グローバル変数で
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
    self.set_pre_queens(ld, rd, col, k, l, row, queens, LD, RD, counter, constellations, N, preset_queens)
    subconst_cache[key] = True  # マークだけでOK

# 呼び出し側
# self.set_pre_queens_cached(...) とする

# ------------------------------------------------
🟡3. 星座自体をtuple/hashで一意管理して重複を防ぐ
constellationsリストに追加する際、既に存在する星座を再追加しない
→ 星座自体を「tuple/int/hash」にして集合管理
これにより、異なる経路から同じ星座に到達しても重複追加を防げます。

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
        constellations.append(constellation)
        signatures.add(signature)
        counter[0] += 1
    return

# ------------------------------------------------
🟡4. pickleファイルで星座リストそのものをキャッシュ
巨大Nのときは事前生成した星座リストをpickleでファイル化し、プログラム起動時に一度だけロード→以降はメモリで使い回す。Codon標準ではpickleがサポートされていない場合もありますが、Python互換ライブラリ（import pickle）が使えれば、ほぼ同じ形で使えます。

こちらは大失敗であきらめました。codonでは動かないみたい

import pickle, os

def load_constellations(self, N: int, preset_queens: int) -> list:
  fname = f"constellations_N{N}_{preset_queens}.pkl"
  if os.path.exists(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)
  else:
    constellations = []
    self.gen_constellations(set(), constellations, N, preset_queens)
    with open(fname, "wb") as f:
        pickle.dump(constellations, f)
    return constellations

# 実行時
def main()で
      # NQ.gen_constellations(ijkl_list,constellations,size,preset_queens)
      constellations = NQ.load_constellations(size,preset_queens)
      NQ.exec_solutions(constellations,size)

# ------------------------------------------------
fedora$ codon build -release 26Py_constellations_optimized_codon.py
fedora$ ./26Py_constellations_optimized_codon
 N:        Total       Unique        hh:mm:ss.ms
IOError: pickle error: gzwrite returned 0

Raised from: std.pickle._write_raw.0:0
/home/suzuki/.codon/lib/codon/stdlib/pickle.codon:25:13
中止 (コアダンプ)

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


fedora$ codon build -release 21Py_constellations_optimized_codon.py && ./21Py_constellations_optimized_codon
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

import random
import pickle, os
from operator import or_
# from functools import reduce
from typing import List,Set,Dict
from datetime import datetime

# pypyを使うときは以下を活かしてcodon部分をコメントアウト
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
#
class NQueens19:
  def __init__(self)->None:
    pass
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
  # 呼び出し側
  # self.set_pre_queens_cached(...) とする
  constellation_signatures = set()
  #---------------------------------
  def state_hash(self,ld: int, rd: int, col: int, row: int) -> int:
      # 単純な状態ハッシュ（高速かつ衝突率低めなら何でも可）
      return (ld * 0x9e3779b9) ^ (rd * 0x7f4a7c13) ^ (col * 0x6a5d39e9) ^ row
  #---------------------------------
  def set_pre_queens(self,ld:int,rd:int,col:int,k:int,l:int,row:int,queens:int,LD:int,RD:int,counter:list,constellations:List[Dict[str,int]],N:int,preset_queens:int,visited:set[int])->None:
    mask=(1<<N)-1  # setPreQueensで使用
    # ----------------------------
    # 状態ハッシュによる探索枝の枝刈り
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
      # 各コンステレーションのソリューション数を更新
      # constellation["solutions"]=temp_counter[0] * self.symmetry(ijkl,N)
      constellation["solutions"]=cnt * self.symmetry(ijkl,N)
      # temp_counter[0]=0
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
      visited:set[int]=set()
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
      if next_free and (row>=endmark-1 or ~blocked):
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
      if next_free and (row+1>=endmark or ~((next_ld<<1)|(next_rd>>1)|next_col)>0):
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
      if next_free and (row+2>=endmark or ~blocked):
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
      if next_free and (row>=endmark-1 or ~((next_ld<<1)|(next_rd>>1)|(next_col))>0):
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
      if next_free and (row>=endmark-1 or ~((next_ld<<1)|(next_rd>>1)|next_col)>0):
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
class NQueens19_constellations():
  def main(self)->None:
    nmin:int=5
    nmax:int=18
    preset_queens:int=4  # 必要に応じて変更
    print(" N:        Total       Unique        hh:mm:ss.ms")
    for size in range(nmin,nmax):
      start_time=datetime.now()
      ijkl_list:Set[int]=set()
      constellations:List[Dict[str,int]]=[]
      NQ=NQueens19()
      #--------------------------
      # codon では動かないので以下を切り替える
      # pickleの最適化は使わない（あきらめる）
      NQ.gen_constellations(ijkl_list,constellations,size,preset_queens)
      #
      # codonでpickleを使う（うごかない）
      # constellations = NQ.load_constellations(size,preset_queens)
      #---------------------------------
      NQ.exec_solutions(constellations,size)
      total:int=sum(c['solutions'] for c in constellations if c['solutions']>0)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"{size:2d}:{total:13d}{0:13d}{text:>20s}")
if __name__=="__main__":
  NQueens19_constellations().main()
