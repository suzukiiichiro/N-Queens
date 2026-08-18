#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Python/codon Ｎクイーン コンステレーション版 最適化+最速化（@par)


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
N-Queens(ビットボード + 星座分割 + 対称性除去) — 実装レビュー / 使い方 / 主要ポイント

■概要
- 本実装は N-Queens の総解数を、ビット演算・対称性・部分状態(星座/constellation)分割で高速数え上げする。
- 探索の二段構成:
    (1) `gen_constellations()` で盤の代表配置(開始星座)を生成・正規化（Jasmin 変換＋回転鏡像除去）。
    (2) `exec_solutions()` で各星座から下流をビットバックトラック（多数の SQ* 関数群）して解数を集計。

■主な最適化と設計
- ビットボード表現と LSB 抽出:
    - 置ける位置: `free = board_mask & ~(ld | rd | col)`
    - LSB: `bit = free & -free`
    - 伝播: `next_ld = (ld | bit) << 1`, `next_rd = (rd | bit) >> 1`, `next_col = col | bit`
- “先読み空き”(詰みの早期検出)の関数化:
    - `_has_future_space()` → `if next_free and ((row >= endmark - 1) or self._has_future_space(...)):` の形で無駄再帰を削減。
- 対称性の倍率(2/4/8):
    - `symmetry()` / `symmetry90()` で 90°自己同型・対角自己同型・一般を判定し、
      `solutions = cnt * self.symmetry(ijkl, N)` で最後に倍率を掛ける。
- Jasmin 変換（開始星座の正規代表化）:
    - `jasmin()` で「盤端からの近さ」を比較 → 必要回数の 90°回転 (`rot90`)＋上下鏡像 (`mirvert`) を適用。
    - メモ化 `get_jasmin()` で再計算を回避。
- サブ状態生成の重複抑止:
    - `set_pre_queens_cached()` のキー: `(ld, rd, col, k, l, row, queens, LD, RD, N, preset_queens)`
    - `state_hash()` と `visited` セットで探索枝の再訪をブロック。
    - `constellation_signatures` で星座辞書の重複 `{"ld","rd","col","startijkl"}` 追加を防止。
- 盤幅の厳密マスク:
    - `board_mask = (1<<N) - 1` を導入し、**補数(~)を使う箇所は必ず `board_mask & ~...` でクリップ**
      （例: `free = board_mask & ~(ld|rd|col)`）。
- 実装上の注意:
    - ソース中の `@par` は通常 Python では未定義（並列化の名残）。そのままなら削除/No-Op 化が必要。
    - codon/pypy 用のコメント切替があるため、実行環境に合わせて `pickle` 使用箇所を選択。

■使い方
    python3 NQueens14.py
出力:
    N:        Total       Unique        hh:mm:ss.ms

■引用メモ（本ソースの核）
- 90°回転: `rot90(): return ((N-1-self.getk(ijkl))<<15)+...+(self.getj(ijkl)<<5)+self.geti(ijkl)`
- 自己同型倍率: `symmetry(): return 2 if self.symmetry90(...) else 4 if ... else 8`
- 先読み空き: `_has_future_space(): return (board_mask & ~(((next_ld<<1)|(next_rd>>1)|next_col))) != 0`
- 星座正規化: `for _ in range(arg): ijkl = self.rot90(ijkl, N); if self.getj(ijkl) < N-1-self.getj(ijkl): ijkl = self.mirvert(ijkl, N)`


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


"""
✅[Opt-02-1]    左右対称性除去（初手左半分/コーナー分岐で重複生成排除）
1 行目の列を 0～n//2−1 に制限
→ gen_constellationsのfor k in range(1, halfN)や、角コーナー分岐で左右対称盤面の重複生成を抑制
→ コーナーあり/なし両方をしっかり区分

1) 1 行目を「左半分」に制限（左右対称の半分だけ生成）
関数: gen_constellations(...)

halfN = (N+1)//2
# コーナーにクイーンがいない開始コンステレーション
ijkl_list.update(
  self.to_ijkl(i, j, k, l)
  for k in range(1, halfN)                  # ← 左半分だけ
  for l in range(k+1, N-1)
  for i in range(k+1, N-1)
  if i != (N-1) - l
  for j in range(N-k-2, 0, -1)
  if j != i and j != l
  if not self.check_rotations(ijkl_list, i, j, k, l, N)
)

k in range(1, halfN) が「最上段の列を左半分に制限」に相当（鏡像を作らない）。
これにより左右対称の重複生成をそもそも発生させない設計です。

"""

"""
✅[Opt-02-2]    中央列特別処理（奇数N）    達成    奇数N中央列を専用内包表記で排除
2) 奇数盤での中央列（対称軸）を特別処理

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
    if not self.rot180_in_set(ijkl_list, i, j, center, l, N)
  )

奇数 N の“中央列”は左右対称の軸になるため、専用の生成ルールで重複を抑止。
rot180_in_set により 180°回転の重複まで排除。
"""

"""
✅[Opt-02-3]) コーナーあり／なしを明確に分岐

# コーナーにクイーンがある開始コンステレーション
ijkl_list.update({ self.to_ijkl(0, j, 0, l) for j in range(1, N-2) for l in range(j+1, N-1) })

「コーナーあり」を別の集合内包で明確に分けて生成。
「コーナーなし」との混在による重複を避けつつ、ケース分岐を保ったまま探索を開始できます。
"""

"""
✅[Opt-02-4]) 生成後に“正準化”で回転・鏡像を一本化（最終ダメ押し）

ijkl_list = { self.get_jasmin(c, N) for c in ijkl_list }  # ← Jasmin で正準形へ

jasmin（90°回転＋垂直ミラー含む）で代表形をとり、左右対称・回転対称の重複を根こそぎ圧縮。
さらに check_rotations / rot180_in_set も前段で併用しており、多重に重複防止が効く構成。
"""

"""
✅[Opt-02-5])カウント側は対称性の重み付けで整合
関数: exec_solutions(...)

constellation["solutions"] = cnt * self.symmetry(ijkl, N)

生成側で左右（および回転）を抑えたうえで、対称群の位数に応じた重み付け（2/4/8）で最終トータルを復元。
生成制約とカウント重みの整合が取れているので、ダブりも欠落もなし。

結論
「最上段は左半分だけ」「奇数盤の中央列は特別処理」「コーナーあり／なしの分岐」「Jasmin 正準化」の四段構えで、Opt-02 は十分に適用済みです。
そのうえで symmetry() による重み付けで総数が正しく積み上がっています。
次の最適化項目、どうぞ！同じ調子で該当コードを指差し＆必要なら改善ポイントまで洗い出します。
"""

"""
✅[Opt-03]    角位置分岐・COUNT分類 コーナー分岐/symmetryでCOUNT2/4/8分類

A) 角（コーナー）あり／なしの分岐（生成段階）
関数: gen_constellations(...)

# コーナーにクイーンがいない場合の開始コンステレーション
ijkl_list.update(
  self.to_ijkl(i,j,k,l)
  for k in range(1, halfN) ...
  if not self.check_rotations(ijkl_list, i, j, k, l, N)
)

# コーナーにクイーンがある場合の開始コンステレーション
ijkl_list.update({ self.to_ijkl(0, j, 0, l) for j in range(1, N-2) for l in range(j+1, N-1) })

ここで**「コーナーなし群」と「コーナーあり群」**を分岐して初期星座を構成しています。
COUNT分類（2/4/8）は、後述の symmetry(...) で与える倍率に反映される前提で、分岐自体は重複生成の回避と探索空間の整理を担っています。

B) COUNT 2/4/8 の分類ロジック（対称クラス）
関数: symmetry(ijkl: int, N: int) -> int / symmetry90(...)

def symmetry(self, ijkl: int, N: int) -> int:
    return 2 if self.symmetry90(ijkl, N) else \
           4 if self.geti(ijkl) == N-1-self.getj(ijkl) and self.getk(ijkl) == N-1-self.getl(ijkl) else 8

def symmetry90(self, ijkl: int, N: int) -> bool:
    return ((self.geti(ijkl)<<15) + (self.getj(ijkl)<<10) + (self.getk(ijkl)<<5) + self.getl(ijkl)) \
           == (((N-1-self.getk(ijkl))<<15) + ((N-1-self.getl(ijkl))<<10) + (self.getj(ijkl)<<5) + self.geti(ijkl))

symmetry(...) が盤面の対称性に応じて 2/4/8 を返し、これが COUNT分類の核。
symmetry90(...) は 90°回転不変（4回転対称）の特別ケースを検出。
次に「主対角/副対角のミラーと整合する配置」（i == N-1-j かつ k == N-1-l）を COUNT=4、それ以外を COUNT=8 に分類。

C) 分類倍率の適用（集計段階）
関数: exec_solutions(...)

cnt = ...  # 各サブルーチンで得た代表解のカウント
constellation["solutions"] = cnt * self.symmetry(ijkl, N)  # ← ここで 2/4/8 の倍率を適用

生成時に正準化（Jasmin）等で代表形だけを数え、最後に対称クラスの大きさ（2/4/8）で増幅して総数に反映しています。
これにより、左右対称・回転対称などを重複生成せず、かつ過不足なく合算できています。

メモ（任意の改善ポイント）
symmetry(...) の判定は実装としてコンパクトで高速ですが、条件の意味づけコメント（「ここは 90°回転不変 → COUNT=2」「ここは主/副対角ミラー整合 → COUNT=4」など）を数行入れておくと、将来の保守で安心です。
角分岐の生成セット（to_ijkl(0, j, 0, l)）はコーナーの代表置きに限定され、後段の symmetry() 倍率で全体を補完しているので整合しています。
"""

"""
✅[Opt-04]    180°対称除去
rot180_in_set で内包時点で重複除去
rot180_in_set で「180度回転盤面が既にセットにある場合はスキップ」できている。クイーン配置を180度回転したものの重複カウントはすべて生成段階で排除できている

A) 一般ケース（コーナーなしの生成）
関数: gen_constellations(...)

ijkl_list.update(
  self.to_ijkl(i, j, k, l)
  ...
  if not self.check_rotations(ijkl_list, i, j, k, l, N)   # ← ここで 90/180/270 まとめて重複排除
)

check_rotations(...) 内で rot90 / rot180 / rot270 をすべて生成済み集合と照合しています。
→ 180°回転（rot180）も ここで除去 されています。

B) 奇数盤の中央列・特別処理

if N % 2 == 1:
  ...
  ijkl_list.update(
    self.to_ijkl(i, j, center, l)
    ...
    if not self.check_rotations(ijkl_list, i, j, center, l, N)  # ← 90/180/270
    if not self.rot180_in_set(ijkl_list, i, j, center, l, N)    # ← 180°を明示的に追加チェック
  )

ここでは check_rotations に加えて rot180_in_set を重ねがけ。
→ 180°対称の重複は 二重のガード で確実に弾かれます（やや冗長ではありますが安全）。

C) 仕上げの正準化（万一の取りこぼし防止）
ijkl_list = { self.get_jasmin(c, N) for c in ijkl_list }  # 正準形へ

Jasmin（回転＋ミラーの正準化）で生成後の代表形に畳み込み。
→ 生成段階での取りこぼしが仮にあっても、ここで同一形が統合されます。

ひと言アドバイス（任意）
中央列ブロックの if not self.check_rotations(...) と if not self.rot180_in_set(...) は、機能としては重複です（check_rotations がすでに rot180 を含むため）。
パフォーマンス重視で条件判定を軽くしたいなら、どちらか片方に寄せるのがスッキリです。
一方で「中央列だけは180°対称だけ即時除去したい」という意図的な最適化なら、このままでもOKです。
"""

"""
✅[Opt-05]    並列処理（初手分割）    @par
A) 代表盤面ごとの独立タスクに分割
関数: exec_solutions(...)
該当（抜粋）:

@par
for constellation in constellations:
    ...
    cnt = ...  # 各サブルーチンで代表解を数える
    constellation["solutions"] = cnt * self.symmetry(ijkl, N)

constellations の各要素（= 初手構成ごとの代表盤面）をループ粒度で並列化しています。
各反復が独立（共有状態を持たない）になるよう、使用する変数をループ内ローカルにしており、constellation["solutions"] も各要素専有のスロットに書き戻すだけなのでレースになりにくい構造です。

B) 集計は並列後に一括
関数: NQueens14_constellations.main(...)

NQ.exec_solutions(constellations, size)
total = sum(c['solutions'] for c in constellations if c['solutions'] > 0)

並列区間外で合計を一括で取り、削減（reduction）はシンプルに保っています。

ちょい改善の余地（任意）
exec_solutions 内で辞書へ直接書き戻す代わりに、ローカル変数 sol = ... を計算→最後に代入にしておくと、可読性と安全性が少し上がります（実質的な性能差はほぼゼロ）。
もし将来、並列ループ内で共有キャッシュ（例：jasmin_cache 等）を触れる変更を入れるなら、生成前段でキャッシュ完了→exec_solutions では読み取りのみにする方針を維持すると安心です。

"""

"""
✅済[Opt-06] 角位置（col==0）分岐＆対称分類（COUNT2/4/8）
「1行目col==0」や「角位置」だけを個別分岐しているか
対称性カウント（COUNT2/4/8分類）で「同型解数」の判定ができているか
→ コーナー（i=0やk=0）専用の初期コンステレーション生成あり。
→ symmetryやjasmin関数でCOUNT分類もサポート

角位置（col==0）分岐
  初期コンステレーション生成（コーナー専用）
  関数: gen_constellations(...)

# コーナーにクイーンがある場合の開始コンステレーションを計算する
ijkl_list.update({self.to_ijkl(0, j, 0, l) for j in range(1, N-2) for l in range(j+1, N-1)})

ここで「i=0 かつ k=0」のケースを別分岐で投入しています（= 角位置起点の盤面を明示生成）。
なお、直後の初期ビットマスク構築で col に 1 をOR しており、col==0列が常に占有される状態を担保しています。

L = 1 << (N-1)
...
col = 1 | L | (L >> i) | (L >> j)  # ← 1 が列0ビット


非コーナー分岐（左右対称の左半分に制限）
  同じ関数内で、

ijkl_list.update(
    self.to_ijkl(i, j, k, l)
    for k in range(1, halfN)        # ← 左半分
    for l in range(k+1, N-1)
    for i in range(k+1, N-1)
    if i != (N-1) - l
    for j in range(N-k-2, 0, -1)
    if j != i and j != l
    if not self.check_rotations(ijkl_list, i, j, k, l, N)
)

で初手を左右半分に制限（対称除去）しつつ、コーナー無しの系を別に扱っています。

COUNT2/4/8 の対称分類
  分類ロジック
  関数: symmetry(ijkl, N) / symmetry90(ijkl, N)

def symmetry(self, ijkl: int, N: int) -> int:
    return 2 if self.symmetry90(ijkl, N) else \
           4 if self.geti(ijkl) == N-1-self.getj(ijkl) and self.getk(ijkl) == N-1-self.getl(ijkl) else 8

  90°不変 → COUNT2
  主対角/副対角の位置関係が対応 → COUNT4
  それ以外 → COUNT8

分類の利用（同型解数の反映）
  関数: exec_solutions(...)

cnt = ...  # 代表盤面での数え上げ
constellation["solutions"] = cnt * self.symmetry(ijkl, N)

  代表解に対称重み（2/4/8）を乗算して最終解数に反映しています。

  #（補足）Jasmin 正規化
    関数: jasmin(...) / get_jasmin(...)
    ijkl_list = { self.get_jasmin(c, N) for c in ijkl_list } の形で回転・鏡映の正準化を行い、生成段階で重複を抑えています（分類そのものではないですが、COUNT分類との相性が良い実装です）。
"""

"""
✅[Opt-07] Zobrist Hash による transposition / visited 状態の高速検出
ビットボード設計でも、「盤面のハッシュ」→「探索済みフラグ」で枝刈りは可能です。
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
"""


"""
❎[Opt-07]    1行目以外の部分対称除去
jasmin/is_partial_canonicalで排除
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

❎[Opt-11]    ミラー+90°回転重複排除    原則不要「あえてやらない」設計。必要ならis_canonicalで激重に

❎[Opt-10]    マクロチェス（局所パターン）    達成    violate_macro_patterns関数（導入済ならOK）
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


✅[Opt-12]    キャッシュ構造設計

ちょい改善の余地（任意）
exec_solutions 内で辞書へ直接書き戻す代わりに、ローカル変数 sol = ... を計算→最後に代入にしておくと、可読性と安全性が少し上がります（実質的な性能差はほぼゼロ）。
もし将来、並列ループ内で共有キャッシュ（例：jasmin_cache 等）を触れる変更を入れるなら、生成前段でキャッシュ完了→exec_solutions では読み取りのみにする方針を維持すると安心です。

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

6. 使っていない（or Codon 非対応の）コード
pickle 系ユーティリティはコメントで「Codon では動かない」と明記済み。将来の混乱防止のため、if TYPE_CHECKING: か try/except ImportError で囲うか、CODON 判定で読み飛ばすガードにしておくとベター。実行パスからは既に外してあるので実害はありません。

すでにビット演算のインライン化・board_mask の上位での共有・**1ビット抽出 bit = x & -x**など、要所は押さえられています。cnt を星座ごとにローカルで完結→solutions に掛け算（symmetry()）という流れもキャッシュに優しい設計。
これ以上を狙うなら、「星座ごと分割の並列度を広げる」か「gen_constellations の ijkl_list.update(...) での回転重複除去を最小限に（=set操作の負荷を減らす）」の二択ですが、現状の速度を見る限り十分実用的です。

"""



# import random
import pickle, os
# from operator import or_
# from functools import reduce
from typing import List, Set, Dict, Tuple
from datetime import datetime

# pypyを使うときは以下を活かしてcodon部分をコメントアウト
# import pypyjit
# pypyjit.set_param('max_unroll_recursion=-1')
#
class NQueens14:

  def __init__(self)->None:
    """
    内部キャッシュの初期化。

    - `subconst_cache`: サブ星座生成(set_pre_queens)の再実行を防ぐキー集合。
      key = (ld, rd, col, k, l, row, queens, LD, RD, N, preset_queens)
    - `constellation_signatures`: 生成済み星座(辞書)の重複登録を防止。
      signature = (ld, rd, col, k, l, row)
    - `jasmin_cache`: Jasmin 変換のメモ化 {(packed_ijkl, N): packed_ijkl'}
    """

    # インスタンス専用に上書き（共有を避ける）
    self.subconst_cache: Dict[ Tuple[int, int, int, int, int, int, int, int, int, int, int], bool ] = {}
    self.constellation_signatures: Set[ Tuple[int, int, int, int, int, int] ] = set()
    self.jasmin_cache: Dict[Tuple[int, int], int] = {}

  def rot90(self,ijkl:int,N:int)->int:
    """
    (i,j,k,l) を 90°回転した座標に変換（5bit × 4 のパック整数表現）。

        return ((N-1-self.getk(ijkl))<<15) + ((N-1-self.getl(ijkl))<<10) + (self.getj(ijkl)<<5) + self.geti(ijkl)

    盤上の 0-origin 座標系で、回転後の (i',j',k',l') をパックして返す。
    """
    return ((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl)

  def rot180(self,ijkl:int,N:int)->int:
    """
    (i,j,k,l) を 180°回転。`rot90()` の 2 回分に相当するが、直接計算で高速化。
    """
    return ((N-1-self.getj(ijkl))<<15)+((N-1-self.geti(ijkl))<<10)+((N-1-self.getl(ijkl))<<5)+(N-1-self.getk(ijkl))

  def rot180_in_set(self,ijkl_list:Set[int],i:int,j:int,k:int,l:int,N:int)->bool:
    """
    与えた (i,j,k,l) の 180°回転結果が `ijkl_list` に既に含まれるかを判定。
    """
    return self.rot180(self.to_ijkl(i, j, k, l), N) in ijkl_list

  def check_rotations(self,ijkl_list:Set[int],i:int,j:int,k:int,l:int,N:int)->bool:
    """
    90°/180°/270°回転のいずれかが `ijkl_list` に存在するかを判定（重複星座の生成を抑制）。
    """
    return any(rot in ijkl_list for rot in [((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i,((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+(N-1-k), (l<<15)+(k<<10)+((N-1-i)<<5)+(N-1-j)])

  def symmetry(self,ijkl:int,N:int)->int:
    """
    自己同型(回転/鏡像)の種類に応じた倍率(2/4/8)を返す。

    - 90°自己同型: 2
    - 対角自己同型（i=~j, k=~l が成り立つ）: 4
    - 一般: 8
    """
    return 2 if self.symmetry90(ijkl,N) else 4 if self.geti(ijkl)==N-1-self.getj(ijkl) and self.getk(ijkl)==N-1-self.getl(ijkl) else 8

  def symmetry90(self,ijkl:int,N:int)->bool:
    """
    90°回転で自身と一致する(自己同型)かを判定。
    """
    return ((self.geti(ijkl)<<15)+(self.getj(ijkl)<<10)+(self.getk(ijkl)<<5)+self.getl(ijkl))==(((N-1-self.getk(ijkl))<<15)+((N-1-self.getl(ijkl))<<10)+(self.getj(ijkl)<<5)+self.geti(ijkl))

  def to_ijkl(self,i:int,j:int,k:int,l:int)->int:
    """
    4 つの 5-bit 値 (i,j,k,l) を 20bit の整数にパックして返す。
        return (i<<15) + (j<<10) + (k<<5) + l
    """
    return (i<<15)+(j<<10)+(k<<5)+l

  def mirvert(self,ijkl:int,N:int)->int:
    """
    垂直方向の鏡像（上下反転）を適用した (i,j,k,l) を返す。
        return self.to_ijkl(N-1-i, N-1-j, l, k)
    """
    return self.to_ijkl(N-1-self.geti(ijkl),N-1-self.getj(ijkl),self.getl(ijkl),self.getk(ijkl))

  def ffmin(self,a:int,b:int)->int:
    """`min(a,b)` の薄いラッパー（可読性/インライン展開目的）。"""
    return min(a,b)

  """パックされた `ijkl` から i(上位 5bit) を取り出す。 (ijkl>>15)&0x1F"""
  def geti(self,ijkl:int)->int: return (ijkl>>15)&0x1F
  def getj(self,ijkl:int)->int: return (ijkl>>10)&0x1F
  def getk(self,ijkl:int)->int: return (ijkl>>5)&0x1F
  def getl(self,ijkl:int)->int: return ijkl&0x1F

  def get_jasmin(self, c: int, N: int) -> int:
    """
    Jasmin 変換のメモ化版。キー `(c, N)` がキャッシュされていればそれを返し、
    なければ `jasmin()` を実行して保存する。
    """
    key = (c, N)
    if key in self.jasmin_cache:
        return self.jasmin_cache[key]
    result = self.jasmin(c, N)
    self.jasmin_cache[key] = result
    return result

  def jasmin(self,ijkl:int,N:int)->int:
    """
    (i,j,k,l) の“盤端からの近さ”を比較して回転回数 arg∈{0,1,2,3} を選び、
    その後に必要なら上下鏡像を適用して開始星座の正規代表を得る。

    主要断片:
        arg = 0
        min_val = min(j, N-1-j)
        if min(i, N-1-i) < min_val: arg = 2; min_val = ...
        if min(k, N-1-k) < min_val: arg = 3; ...
        if min(l, N-1-l) < min_val: arg = 1; ...
        for _ in range(arg): ijkl = self.rot90(ijkl, N)
        if self.getj(ijkl) < N-1-self.getj(ijkl): ijkl = self.mirvert(ijkl, N)
    """
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

  def file_exists(self,fname:str)->bool:
    """pickle ファイル存在を try/except で判定（読み取り確認）。"""
    try:
      with open(fname, "rb"):
        pass
        return True
    except:
      return False

  def load_constellations(self,N:int,preset_queens:int)->list:
    """
    事前計算した星座を pickle からロード。無ければ `gen_constellations()` で生成して保存。
    ファイル名: f"constellations_N{N}_{preset_queens}.pkl"
    """
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

  def set_pre_queens_cached(self, ld: int, rd: int, col: int, k: int, l: int,row: int, queens: int, LD: int, RD: int,counter: list, constellations: List[Dict[str, int]], N: int, preset_queens: int,visited:set[int]) -> None:
    """
    `set_pre_queens()` の結果を (ld,rd,col,k,l,row,queens,LD,RD,N,preset_queens) でメモ化。
    既に同一キーが実行済みならスキップする。
    """
    key = (ld, rd, col, k, l, row, queens, LD, RD, N, preset_queens)
    if key in self.subconst_cache:
        return
    self.set_pre_queens(ld, rd, col, k, l, row, queens, LD, RD,
                        counter, constellations, N, preset_queens, visited)
    self.subconst_cache[key] = True

  @staticmethod
  def _has_future_space(next_ld: int, next_rd: int, next_col: int, board_mask: int) -> bool:
    """
    1 行先（ld<<1, rd>>1 を適用済み状態）で候補が 1 ビットでも残るか。
        return (board_mask & ~(((next_ld << 1) | (next_rd >> 1) | next_col))) != 0
    """
    # 次の行に進んだときに置ける可能性が1ビットでも残るか
    return (board_mask & ~(((next_ld << 1) | (next_rd >> 1) | next_col))) != 0

  def state_hash(self,ld: int, rd: int, col: int, row: int,queens:int,k:int,l:int,LD:int,RD:int,N:int) -> int:
    """
    set_pre_queens 系の枝刈り用ハッシュ。衝突低めかつ計算軽量な XOR/SHIFT 混合。
        return (ld<<3) ^ (rd<<2) ^ (col<<1) ^ row ^ (queens<<7) ^ (k<<12) ^ (l<<17) ^ (LD<<22) ^ (RD<<27) ^ (N<<1)
    """
    return (ld<<3) ^ (rd<<2) ^ (col<<1) ^ row ^ (queens<<7) ^ (k<<12) ^ (l<<17) ^ (LD<<22) ^ (RD<<27) ^ (N<<1)

  def set_pre_queens(self,ld:int,rd:int,col:int,k:int,l:int,row:int,queens:int,LD:int,RD:int,counter:list,constellations:List[Dict[str,int]],N:int,preset_queens:int,visited:set[int])->None:
    """
    事前に `preset_queens` 個の Q を安全に配置した星座（部分状態）を列挙。

    流れ:
      1) 再訪枝刈り: `h = self.state_hash(...)`; `if h in visited: return`
      2) 行スキップ: `if row == k or row == l: ...`
      3) 完了: `if queens == preset_queens:` → signature 重複を避けて `constellations.append(...)`
      4) 進展:
         `free = ~(ld|rd|col|(LD>>(N-1-row))|(RD<<(N-1-row))) & mask`
         `bit = free & -free`
         再帰へ (`(ld|bit)<<1`, `(rd|bit)>>1`, `col|bit`)

    注意:
      - `free` の算出は盤幅 `mask=(1<<N)-1` でクリップ。後段は `board_mask` を使用。
    """
    mask=(1<<N)-1  # setPreQueensで使用
    # ----------------------------
    # 状態ハッシュによる探索枝の枝刈り
    # バックトラック系の冒頭に追加　やりすぎると解が合わない
    h: int = self.state_hash(ld, rd, col, row,queens,k,l,LD,RD,N)
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
    """
    生成済み星座ごとに SQ* ルーチンを分岐実行し、`solutions` を埋める。

    要点:
      - `board_mask = (1<<N) - 1` を用意し、**常に** `free = board_mask & ~(ld|rd|col)` で初期化。
      - `j,k,l`/`start` から jmark/endmark/mark1/mark2 を決め、該当する SQ* を選択。
      - 結果 `cnt` に対称倍率 `self.symmetry(ijkl, N)` を掛けて `constellation["solutions"]` に格納。

    備考:
      - ソースの `@par` は通常 Python では未定義。並列化しない場合は削除可。
    """

    N2:int=N-2
    small_mask=(1<<(N2))-1
    temp_counter=[0]
    cnt=0
    board_mask:int=(1<<N)-1
    @par
    for constellation in constellations:
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
      # free=~(ld|rd|col)
      free = board_mask&~(ld|rd|col)
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
    """
    開始星座 (i,j,k,l) の集合を作り、Jasmin 正規化→ `set_pre_queens_cached()` でサブ星座を生成。

    ポイント:
      - 奇数 N の中央列を特別扱い（Opt-03）。180°回転の重複も除去:
            if N % 2 == 1: ... if not self.rot180_in_set(...):
      - 既存星座との回転同型を `check_rotations()` で弾く。
      - Jasmin 正規化はメモ化版 `get_jasmin()` を用いる:
            ijkl_list = { self.get_jasmin(c, N) for c in ijkl_list }
      - `ld,rd,col,LD,RD` の初期値をビットで構築し、`set_pre_queens_cached()` へ。
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
    """
    SQ* 系バックトラック（部分状態から葉までの全列挙・カウント用）。

    目的:
        - 盤幅 `board_mask` 上で `free`（配置可能ビット）を順に展開し、葉に達したら 1 を返す。
        - 分岐規則（mark1/mark2/jmark/endmark など）で特定の行/列をスキップ or 確定し、探索幅を削減。
    主要断片:
        avail = free
        while avail:
            bit = avail & -avail           # LSB 抽出
            avail &= avail - 1             # LSB 除去
            next_ld  = (ld  | bit) << 1
            next_rd  = (rd  | bit) >> 1
            next_col =  col | bit
            blocked  = next_ld | next_rd | next_col
            next_free = board_mask & ~blocked
            # 先読み空き: 1 行先で候補が残らないノードを弾く
            if next_free and ((row >= endmark - 1) or self._has_future_space(next_ld, next_rd, next_col, board_mask)):
                total += self.<再帰呼び先>(...)

    終端条件:
        - 通常: `if row == endmark: return 1`
        - バリエーション: `if row == endmark and (avail & ~1) > 0: return 1` など（末尾制約あり）

    引数:
        ld, rd, col : 左斜線/右斜線/列の占有ビット集合（次行で <<1 / >>1 へシフト）
        row         : 今から置く行インデックス
        free        : 現行行の候補集合（必ず `board_mask & ~blocked` で算出済み）
        jmark, endmark, mark1, mark2 : 分岐・スキップ・確定のための境界/マーカー
        board_mask  : (1<<N)-1 の盤幅マスク
        N           : 盤サイズ

    注意:
        - **next_free は必ず `board_mask & ~blocked` でクリップ**（~blocked 単体は NG）。
        - `row == mark?` ブロックでは 2 行 or 3 行進める最適化あり（例: `row==mark1` → `row+2`/`row+3`）。
    """

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

class NQueens14_constellations:
  """
  実行エントリ＋小 N 用のフォールバック全列挙を持つ薄いランナー。

  - `_bit_total(size)` : 小さな N では素のビット DFS で正しい総数を返す。
  - `main()`           : N をループし、星座分割→探索→計測→出力までを行う。
  """

  def _bit_total(self, size: int) -> int:
    """
    小 N 用の純粋ビットバックトラック（対称重みなし・全列挙）。

    主要断片:
        mask = (1 << size) - 1
        def bt(row, left, down, right):
            if row == size: total += 1; return
            bitmap = mask & ~(left | down | right)
            while bitmap:
                bit = -bitmap & bitmap
                bitmap ^= bit
                bt(row+1, (left|bit)<<1, down|bit, (right|bit)>>1)
    """

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
    """
    ベンチ実行。`size <= 5` は `_bit_total` で、以降は星座分割探索。
    代表ループ:
        for size in range(nmin, nmax):
            start = datetime.now()
            if size <= 5: ... else:
                NQ = NQueens14()
                NQ.gen_constellations(...)
                NQ.exec_solutions(...)
                total = sum(c['solutions'] for c in constellations if c['solutions'] > 0)
            経過時間を "hh:mm:ss.ms" で表示。
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
      NQ=NQueens14()
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
  NQueens14_constellations().main()
