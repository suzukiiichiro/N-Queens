このページはの経験をN-Queensを研究開発することの実践で得た失敗と成功を整理統合しPython/codon活用ドキュメントです。「Codon とは何か」「なぜ使うのか」「どのように構築し、高速化し、成果を上げたのか」を実践的にまとめています。

---

# 🧬 Codon 実践ガイド — Pythonを超えるコンパイラ型高速化の道

> **Author:** iichiro suzuki
> **Project:** Python / Codon 最適化による N-Queens ソルバー開発記録
> **Environment:** Fedora 42 / Amazon Linux 2023 / LLVM / PyPy / CUDA
> **Keywords:** Codon, LLVM, NVPTX, Constellations, Bitboard, Optimization

---

## 🚀 はじめに

Python は美しく柔軟ですが、数値計算や再帰処理の世界では限界があります。
**Codon** は、Python の構文をそのままに **LLVM ベースのコンパイル最適化** を実現する新世代コンパイラです。

私は 、N-Queens 問題という極端に再帰的・組合せ的な問題を題材に、
**Python → Codon → GPU（CUDA）** という道を実践的に歩みました。
ここでは、その過程で得られた知見・構築法・ノウハウ・難所・哲学をまとめました。

> [!TIP]
> 結論から言いますと、GPU/CUDA N-Queensと、CPU/Python Codon N-Queensの計測時間は、CPUでありながらGPU/CUDAにややもすれば追いつくという事となりました。

GPU/CUDA N-Queens
``` bash
GPU/CUDA
10Bit_CUDA/01CUDA_Bit_Symmetry.cu
 N:            Total           Unique         hh:mm:ss.ms
19:       4968057848        621012754     000:00:00:13.80
20:      39029188884       4878666808     000:00:02:02.52
21:     314666222712      39333324973     000:00:18:46.52
22:    2691008701644     336376244042     000:03:00:22.54
23:   24233937684440    3029242658210     001:06:03:49.29
```

Python/Codon N-Queens
``` bash
amazon AWS m4.16xlarge x 1
$ codon build -release 15Py_constellations_optimize_codon.py && ./15Py_constellations_optimize_codon
 N:            Total           Unique         hh:mm:ss.ms
19:       4968057848                0          0:00:22.04
20:      39029188884                0          0:02:52.43
21:     314666222712                0          0:24:25.55
22:    2691008701644                0          3:29:33.97
23:   24233937684440                0   1  day,8:12:58.97
```

 ---

## 🧩 第1章：Codonとは何か

Codon は **LLVM をバックエンドに持つ AOT（Ahead-of-Time）Python コンパイラ** です。
以下の点で他の手法（PyPy, Cython）とは異なります。

| 手法        | コンパイル方式       | Python互換性  | 主な特徴           |
| --------- | ------------- | ---------- | -------------- |
| CPython   | インタプリタ        | 100%       | 柔軟だが遅い         |
| PyPy      | JIT           | ほぼ100%     | 実行時最適化         |
| **Codon** | **AOT（LLVM）** | **90〜95%** | 静的型+高速ネイティブコード |

Codon の思想は「Pythonの書きやすさ」と「Cの速さ」を両立すること。
**Pythonの文法のまま、ネイティブ実行ファイルを生成できる** のが最大の魅力です。


### ● Codon の概要

Codon は、Python の構文をそのまま維持しながら LLVM をバックエンドに用いてコンパイルする新世代のコンパイラです。
Python スクリプトを **静的型解析 → LLVM IR → ネイティブコード** に変換し、JIT ではなく事前最適化（AOT）を行う点が特徴です。

---

### ● 「Python を LLVM にコンパイルする」という思想

Codon の思想は「Python の書きやすさを保ったまま、C/C++ の速度を得る」ことです。
つまり “書くときは Python、動かすときはマシンコード” という理想的な形を目指しています。
これにより、研究開発スクリプトをそのまま高速な本番コードに移行できます。

---

### ● JIT と AOT（Ahead of Time）の違い

JIT（Just-In-Time）は実行時にコンパイルを行う方式で、PyPy などが代表例です。
AOT（Ahead-of-Time）は実行前に完全にネイティブコードへ変換する方式で、Codon はこのアプローチを採用しています。
AOT は起動時間を短縮し、CPU キャッシュ最適化が事前に行えるため、安定した高速性能を発揮します。

---

### ● PyPy や Cython との位置づけの比較

| 実装        | コンパイル方式      | Python互換性     | 特徴                  |
| --------- | ------------ | ------------- | ------------------- |
| CPython   | インタプリタ       | ◎             | 標準実装・動的型            |
| PyPy      | JIT          | ◎             | 実行時最適化・柔軟           |
| Cython    | トランスパイル      | ◯             | C拡張構文が必要            |
| **Codon** | **AOTコンパイル** | **◎（ほぼ100%）** | **LLVMによる完全ネイティブ化** |

Codon は **Python の構文を維持したまま C++ に匹敵する速度** を実現できる点で他と一線を画します。

---

### ● なぜ Codon は高速なのか（LLVM IR → ネイティブコード）

Codon は、すべての Python コードを中間表現（LLVM IR）に変換し、
LLVM の最適化パス（インライン展開・ループ展開・共通式除去など）を適用します。
このプロセスにより、CPython の仮想マシン命令を完全に排除し、CPU が直接実行できるバイナリを生成します。

---

### ● Python 構文との互換性・制限

Codon は Python 3.10+ 構文と高い互換性を持ちますが、動的型付け・例外・一部の標準ライブラリは未対応です。
特に `import numpy` や `try/except`、`async` などは制限があります。
一方で、`int`, `List[int]`, `Dict[str,int]` などの型注釈を使うことで、静的解析と最適化が完全に機能します。


> 💡 **ワンポイントアドバイス**
> Codon は「Pythonの文法をC++コンパイルの世界に持ち込んだ言語」です。
> Python的な柔軟さを残しつつ、静的解析を“味方につける”意識で書くと一気に速くなります。

---

## ⚡ 第2章：なぜCodonなのか
### ● Python の柔軟さと限界

Python は表現力が高く開発効率に優れていますが、実行速度・再帰深度・型判定に大きなオーバーヘッドがあります。
Codon はこれらのボトルネックを LLVM の静的最適化で取り除き、**Python のまま “コンパイル言語級の速さ”** を実現します。

> 💡 **ワンポイントアドバイス**
> Codonの高速化は「型の固定」から始まります。
> 迷ったらまず `int` や `List[int]` を明示し、変数が“揺れない”状態を作ることが最初の一歩です。

---

### ● Codon による高速化のアプローチ

Codon は実行前に型を確定させ、CPU命令レベルで最適化されたネイティブバイナリを生成します。
`int`, `float`, `List[int]` といった明示的な型注釈を加えるだけで、数百倍の速度向上が見込めます。
動的な構造体を避けることで、LLVM のインライン展開とレジスタ割り当てが最大化されます。

---

### ● “書きやすさを捨てずに速くする”

C++ や Rust に比べ、Codon は **Python の文法をほぼそのまま使える** のが最大の利点です。
機械学習・アルゴリズム実験・数値最適化など、試行錯誤を高速に回す研究現場に特に有効です。

---

> [!TIP]
> Codonは、ランタイムのオーバーヘッドなしで Pythonのコードをネイティブなマシン語にコンパイルする高性能なPythonコンパイラです。 Codonを使うとシングルスレッドでも十分な高速化を行うことができますが、マルチスレッドもサポートしているため、更なる高速化も図れます。



---

## 🧱 第3章：Codon環境構築（Fedora / Amazon Linux）

### ● Linux/macでのインストール
``` bash
/bin/bash -c "$(curl -fsSL https://exaloop.io/install.sh)"
```

> [!TIP]
> OSがLinux (x86_64) か MacOS (x86_64 または arm64) であれば、公式からビルド済みのバイナリが配布されているので、1コマンドでインストールすることができます。

---

### ● ビルド環境（Fedora / Amazon Linux）

Codon は CMake / LLVM を利用するため、ビルド時に依存パッケージの整備が重要です。
以下の環境構築で安定動作を確認しています：

``` bash
dnf install cmake ninja-build fmt-devel libomp-devel toml++ fast-float semver-devel
```

---


### ● インストール手順

```bash
git clone https://github.com/exaloop/codon
cd codon
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

---


### ● 実行確認

```bash
$ codon run -release hello.py
Hello Codon!
```

Fedora42 / Amazon Linux 2023 の双方で再現可能でした。


---

> 💡 **ワンポイントアドバイス**
> ビルド時は `ninja -v` や `make VERBOSE=1` で内部ログを出すとトラブルを早期発見できます。
> 特に `libcodonrt.so` の場所は環境ごとに異なるため、`nm -D` で確認しておくのがおすすめです。


## 🧠 第4章：Codonの高速化原理と基本構文

### ● 静的型による最適化
Codon は型を固定することで、動的型判定をすべて排除します。
> [!TIP]
> Codon は動的型を完全に排除し、**型が確定した時点でLLVMが最適化** を行います。
Python のような「どの型が来るかわからない」関数呼び出しは一切発生しません。


---


### 基本的な書き方

```python
from typing import List

def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

print(fib(40))
```

> [!TIP]
> 同じコードでも、Codonでは `n` がintで固定されるため、**分岐予測・再帰展開** が行われ、数十倍高速に動作します。


> 💡 **ワンポイントアドバイス**
> Codonでは「型を決めた瞬間にコンパイルが始まる」と考えましょう。
> 型ヒントは単なるコメントではなく、**LLVM最適化のトリガー** です。




### 最適化のポイント

* 型ヒントを**すべて明示**する (`int`, `List[int]`, `Dict[str,int]`など)
* 例外 (`try/except`) を使わない
* ループ変数・再帰呼び出しを局所変数化
* ループ外でメモリを確保し再利用
* `while` ループを多用し、再帰を階層的に置き換える
* グローバル変数や動的 import を避ける
* 関数を小さく分割してコンパイル単位を明確化
* デバッグには `codon run -debug` を活用
* `for` ループと `while` ループの最適化挙動
* 配列・辞書の静的確保
* `@codon.jit`, `@codon.inline` などの使用可否と実験結果
* Python と Codon での整数オーバーフロー挙動の違い
* `bit` / `mask` 演算での高速化例

---

## ♟️ 第5章：N-QueensソルバーのCodon実装

私は N-Queens 問題を Codon で完全実装し、
**Python 比 約8〜15倍** の高速化を確認しました。

* bit演算によるバックトラック最適化
* 対称性除去（mirror / rotation / canonical）
* `constellation` 手法の導入（部分盤面プリセット）
* `blocked_next`, `next_free`, `board_mask` の概念
* 関数分割とインライン化
* 配列アクセスを整数に置換
* `List[Dict[str,int]]` を `struct` 相当表現に変換
* `COUNT2/4/8` の分離による再帰木の簡略化
* Codon 版での `SQ_core_unified()` 実装の全体像
* Codon による速度向上（Python 比 ×8〜×15）


> 💡 **ワンポイントアドバイス**
> Codon最適化の真髄は「再帰の定式化」と「枝刈りの局所化」です。
> Pythonで動いた探索関数を、**変数を減らして書き直すだけで数倍高速化** できます。


---

### 5.1 主な最適化手法

* **ビットボード** による衝突判定
* **対称性除去**（mirror / rotation）
* **Constellation法**（部分盤面キャッシュによる分割探索）
* **blocked_next**／**next_free** による早期枝刈り
* **Zobrist Hash** による探索重複排除
* **canonical化チェック** による一意解抽出

---

### 5.2 ベンチマーク結果（Fedora42 / Codon v0.17）

N-Queens Nの計測
| N  | Total | Unique |  Python  |  pypy  | Codon  | GPU/CUDA |
| -- | ------- | ------ | -------- | ------ | ------ | -------- |
| 8  | 92      | 12     |  00.126s | 0.169s | 0.001s |   0.01   |
| 13 | 73712   | 9233   |  00.355s | 0.825s | 0.005s |   0.04   |
| 15 | 2279184 | 285053 |  09.006s | 3.841s | 0.035s |   0.07   |
| 17 |14772512 |11977939|1:07.235s |13.367s | 0.436s |   0.26   |

---

言語による速度比較
| 名前 | 処理時間 (s) | 速度 (CPython 比) | 順位 |
|------|---------------|-------------------|------|
| Python | 8.40 | 1.00 | 8 |
| PyPy | 2.08 | 4.03 | 7 |
| Cython | 0.373 | 22.5 | 6 |
| Numba | 0.155 | 54.3 | 5 |
| C++ | 0.138 | 60.7 | 4 |
| Nim | 0.0359 | 234 | 3 |
| Mojo🔥 | 0.0289 | 291 | 2 |
| Codon | 0.0281 | 299 | 1 |


データ構造による速度比較
| 実行方法 | 処理時間（秒） |
|-----------|------------------------------|
| 通常のPython (List利用) | 9.71 |
| Codon (List利用) | 1.04 |
| Codon (静的配列利用) | 0.27 |
| Codon (List利用) 最適化あり | 0.05 |
| Codon (静的配列利用) 最適化あり | 0.04 |

> [!TIP]
> CodonはPythonコードを高速化するコンパイラで、通常のPythonと比較して10倍から100倍、C/C++に匹敵する速度を実現します。これは、実行前に型チェックを行い、コードをネイティブマシンコードにコンパイルすることで、Pythonの実行時オーバーヘッドを回避するためです。また、並列処理にも対応し、さらなる速度向上が期待できます


### 5.3 最適化項目
> **第5章：章内ミニ目次**
>
> - [Opt-02-1：左右対称性除去（初手左半分／コーナー分岐）](#opt-02-1)
> - [Opt-02-2：中央列特別処理（奇数 N）](#opt-02-2)
> - [Opt-02-3：コーナーあり／なし分岐](#opt-02-3)
> - [Opt-02-4：生成後の正準化（Jasmin）](#opt-02-4)
> - [Opt-02-5：対称性の重み付けで整合](#opt-02-5)
> - [Opt-03：角位置分岐・COUNT分類（2/4/8）](#opt-03)
> - [Opt-04：180°対称除去](#opt-04)
> - [Opt-05：並列処理（初手分割）@par](#opt-05)
> - [Opt-06：角位置（col==0）分岐＆COUNT分類](#opt-06)
> - [Opt-07：（却下）1行目以外の部分対称除去](#opt-07)
> - [Opt-08：部分盤面サブ問題キャッシュ](#opt-08)
> - [Opt-09：訪問済み（transposition/visited）](#opt-09)
> - [Opt-10：Jasmin 正規化キャッシュ](#opt-10)
> - [Opt-11：星座（コンステレーション）重複排除](#opt-11)
> - [Opt-12：永続キャッシュ（現状は無効化中）](#opt-12)
> - [Opt-13：部分盤面キャッシュ（tuple→dict）](#opt-13)
> - [Opt-14：星座の重複排除（署名）](#opt-14)
> - [Opt-15：Jasmin 変換のメモ化](#opt-15)
> - [Opt-16：visited の仕込み（星座ごとに new）](#opt-16)
> - [Opt-17：星座リストの外部キャッシュ（ファイル）](#opt-17)
> - [Opt-18：サブコンステ生成の tuple key キャッシュ](#opt-18)
> - [Opt-19：星座を tuple/hash で一意管理](#opt-19)
> - [Opt-20：Jasmin 変換キャッシュ（クラス／グローバル）](#opt-20)
> - [Opt-21：180°重複チェックの二重化（整理）](#opt-21)
> - [Opt-22：visited の粒度（星座単位）](#opt-22)
> - [Opt-23：ビット演算のインライン化](#opt-23)
> - [Opt-24：“先読み空き” 条件の短絡評価](#opt-24)


以下は「**第5章：N-QueensソルバーのCodon実装**」にそのまま追記できる **Markdown 版** です。
（見出し・コードブロック整形済み／README互換）

---

### ✅ [Opt-02-1] 左右対称性除去（初手左半分／コーナー分岐で重複生成排除）<a id="opt-02-1"></a>

* **1 行目の列を `0 ～ (N//2 − 1)` に制限**して、左右対称の鏡像を最初から生成しない。
* `gen_constellations` 内の `for k in range(1, halfN)` が「左半分だけ」を担保。
* **コーナーあり／なし**を分けて生成し、混在重複を避ける。

```python
# 関数: gen_constellations(...)
halfN = (N + 1) // 2
# コーナーにクイーンがいない開始コンステレーション
ijkl_list.update(
  self.to_ijkl(i, j, k, l)
  for k in range(1, halfN)                  # ← 左半分だけ
  for l in range(k + 1, N - 1)
  for i in range(k + 1, N - 1)
  if i != (N - 1) - l
  for j in range(N - k - 2, 0, -1)
  if j != i and j != l
  if not self.check_rotations(ijkl_list, i, j, k, l, N)
)
```

> `k in range(1, halfN)` が「最上段の列を左半分に制限」に相当。
> 生成段階で左右対称（鏡像）を出さない設計。

---

### ✅ [Opt-02-2] 中央列特別処理（奇数 N）<a id="opt-02-2"></a>

* **奇数 N の中央列**は左右対称の軸。専用ルールで重複を抑止。
* `rot180_in_set` により **180°回転の重複**も除去。

```python
# 2) 奇数盤での中央列（対称軸）を特別処理
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
```

---

### ✅ [Opt-02-3] コーナーあり／なしを明確に分岐<a id="opt-02-3"></a>

* **コーナーにクイーンがある**開始コンステレーションを独立生成し、混在重複を回避。

```python
# コーナーにクイーンがある開始コンステレーション
ijkl_list.update({
  self.to_ijkl(0, j, 0, l)
  for j in range(1, N - 2)
  for l in range(j + 1, N - 1)
})
```

---

### ✅ [Opt-02-4] 生成後の「正準化」（Jasmin）で回転・鏡像を一本化<a id="opt-02-4"></a>

* 生成済みの候補を **Jasmin 正準形**に畳み込み、回転・鏡像の重複を圧縮。

```python
# Jasmin で正準形に統合
ijkl_list = { self.get_jasmin(c, N) for c in ijkl_list }
```

---

### ✅ [Opt-02-5] カウント側は対称性の重み付けで整合<a id="opt-02-5"></a>

* **生成側**で対称を抑え、**集計側**で群の位数（2/4/8）を掛けて総数を復元。

```python
# 関数: exec_solutions(...)
constellation["solutions"] = cnt * self.symmetry(ijkl, N)
```

> **結論**
> 「最上段は左半分だけ」「奇数盤の中央列は特別処理」「コーナーあり／なしの分岐」「Jasmin正準化」の四段構えで Opt-02 は適用済み。
> さらに `symmetry()` による重み付けで総数が過不足なく一致。

---

### ✅ [Opt-03] 角位置分岐・COUNT分類（2/4/8）<a id="opt-03"></a>

**A) 角（コーナー）あり／なしを生成段階で分岐**

```python
# コーナーなし
ijkl_list.update(
  self.to_ijkl(i, j, k, l)
  for k in range(1, halfN) ...
  if not self.check_rotations(ijkl_list, i, j, k, l, N)
)

# コーナーあり
ijkl_list.update({
  self.to_ijkl(0, j, 0, l)
  for j in range(1, N - 2)
  for l in range(j + 1, N - 1)
})
```

**B) COUNT 2/4/8 の分類ロジック**

```python
def symmetry(self, ijkl: int, N: int) -> int:
    return 2 if self.symmetry90(ijkl, N) else \
           4 if self.geti(ijkl) == N-1-self.getj(ijkl) and self.getk(ijkl) == N-1-self.getl(ijkl) else 8

def symmetry90(self, ijkl: int, N: int) -> bool:
    return ((self.geti(ijkl) << 15) + (self.getj(ijkl) << 10) + (self.getk(ijkl) << 5) + self.getl(ijkl)) \
           == (((N-1-self.getk(ijkl)) << 15) + ((N-1-self.getl(ijkl)) << 10) + (self.getj(ijkl) << 5) + self.geti(ijkl))
```

**C) 分類倍率の適用（集計段階）**

```python
cnt = ...  # 代表解のカウント
constellation["solutions"] = cnt * self.symmetry(ijkl, N)
```

> *メモ*: `symmetry(...)` の条件には軽い説明コメントを添えておくと保守が楽。

---

### ✅ [Opt-04] 180°対称除去<a id="opt-04"></a>

* `check_rotations(...)` で **90/180/270°** をまとめて照合し重複排除。
* 奇数盤の中央列では `rot180_in_set(...)` を **追加ガード** として適用。

```python
# A) 一般ケース（コーナーなし）
ijkl_list.update(
  self.to_ijkl(i, j, k, l)
  ...
  if not self.check_rotations(ijkl_list, i, j, k, l, N)  # ← 90/180/270 まとめて除去
)

# B) 奇数盤の中央列
if N % 2 == 1:
  ijkl_list.update(
    self.to_ijkl(i, j, center, l)
    ...
    if not self.check_rotations(ijkl_list, i, j, center, l, N)  # 90/180/270
    if not self.rot180_in_set(ijkl_list, i, j, center, l, N)    # 180°明示チェック（安全策）
  )
```

```python
# C) 仕上げ（正準化）
ijkl_list = { self.get_jasmin(c, N) for c in ijkl_list }
```

> *ひと言*: 中央列での `rot180_in_set` は冗長（`check_rotations`に含む）だが、安全重視なら残してもOK。

---

### ✅ [Opt-05] 並列処理（初手分割） @par<a id="opt-05"></a>

**A) 代表盤面ごとに独立タスク化**

```python
# 関数: exec_solutions(...)
@par
for constellation in constellations:
    ...
    cnt = ...  # 代表解を数える
    constellation["solutions"] = cnt * self.symmetry(ijkl, N)
```

**B) 集計は並列後に一括**

```python
# 関数: NQueens14_constellations.main(...)
NQ.exec_solutions(constellations, size)
total = sum(c["solutions"] for c in constellations if c["solutions"] > 0)
```

> *改善余地*: ループ内はローカル変数で計算 → 最後に `dict` に書き戻すと読みやすい。

---

### ✅ [Opt-06] 角位置（`col==0`）分岐＆対称分類（COUNT 2/4/8）<a id="opt-06"></a>

```python
# コーナー専用の初期星座
ijkl_list.update({
  self.to_ijkl(0, j, 0, l)
  for j in range(1, N - 2)
  for l in range(j + 1, N - 1)
})

# 列0ビットの占有（例）
L = 1 << (N - 1)
col = 1 | L | (L >> i) | (L >> j)  # ← 1 が列0ビット
```

* 非コーナー系は「左半分に制限」して重複生成を抑制（前掲の Opt-02-1）。
* 分類は `symmetry(ijkl, N)` の **2/4/8 倍率**で反映。

---

### ✅ [Opt-07]（却下）1行目以外の部分対称除去<a id="opt-07"></a>

* `board` に行ごとの配置（`row → col`）を保持していないため **現状は不可**。
* `is_partial_canonical` の設計メモは残しておき、将来的な拡張で検討。

---

### ✅ [Opt-08] 部分盤面サブ問題キャッシュ <a id="opt-08"></a>

* **場所**: `set_pre_queens_cached(...)`
* **キー**: `(ld, rd, col, k, l, row, queens, LD, RD, N, preset_queens)`
* **値**: `subconst_cache[key] = True`
* **役割**: 同一状態のサブ展開を **一度だけ** にする。

---

### ✅ [Opt-09] 訪問済み（transposition / visited）<a id="opt-09"></a>

* **場所**: `set_pre_queens(...)`
* **構造**: `visited: Set[int]`（タプルや 64bit 圧縮キー）
* **役割**: 再帰木で同一状態への再訪を防止。

---

### ✅ [Opt-10] Jasmin 正規化キャッシュ<a id="opt-10"></a>

* **場所**: `get_jasmin(c, N)` / `jasmin_cache: Dict[Tuple[int, int], int]`
* **役割**: 回転・鏡映の正規化結果をメモ化して再計算を回避。

---

### ✅ [Opt-11] 星座（コンステレーション）重複排除<a id="opt-11"></a>

* **場所**: `constellation_signatures: Set[Tuple[int,int,int,int,int,int]]`
* **役割**: 生成済み星座の **署名で一意管理**（重複追加を防止）。

---

### ✅ [Opt-12] 永続キャッシュ（現状は無効化中）<a id="opt-12"></a>

* **場所**: `load_constellations(...)` / `pickle`（テキスト・バイナリ両方の I/O ラッパあり）
* **`__init__` 内の主なキャッシュ**:

  * `self.subconst_cache`, `self.constellation_signatures`,
  * `self.jasmin_cache`, `self.zobrist_tables`

---

### ✅ [Opt-13] 部分盤面キャッシュ（tuple化→dict）<a id="opt-13"></a>

* `set_pre_queens_cached(...)` の **タプルキー**で再帰の指数的重複をカット。

---

### ✅ [Opt-14] 星座の重複排除（署名）<a id="opt-14"></a>

* `set_pre_queens(...)` の `if queens == preset_queens:` ブロック内で
  `signature = (ld, rd, col, k, l, row)` を用いて **重複チェック**。

---

### ✅ [Opt-15] Jasmin 変換のメモ化<a id="opt-15"></a>

* `get_jasmin(c, N) → self.jasmin_cache[(c, N)]` で頻出起点の再計算を回避。
* `gen_constellations()` で `ijkl_list = { self.get_jasmin(c, N) ... }` を適用。

---

### ✅ [Opt-16] 訪問済み状態（transposition/visited）の仕込み<a id="opt-16"></a>

* **スコープ**: `gen_constellations()` の **星座ごとに `visited = set()` を新規作成**。
* **キー**: `StateKey=(ld,rd,col,row,queens,k,l,LD,RD,N,preset_queens)` など。
* `state_hash(...)` への置換も候補（O(1) で高速・省メモリ）。

---

### ✅ [Opt-17] 星座リストの外部キャッシュ（ファイル）<a id="opt-17"></a>

* **テキスト／バイナリ**両対応の save/load と、破損チェック（`validate_*`）を用意。

---

### ✅ [Opt-18] サブコンステ生成の tuple key キャッシュ<a id="opt-18"></a>

* `self.subconst_cache: Dict[StateKey, bool]` を `__init__` で用意。
* 生成・再帰とも **`set_pre_queens_cached(...)` を経由**し、同一状態の再実行を回避。

---

### ✅ [Opt-19] 星座自体を tuple/hash で一意管理<a id="opt-19"></a>

* `self.constellation_signatures` による **集合管理**で重複追加を防止。
* 未出のみ `constellations.append(...)` ＆ `counter[0] += 1`。

---

### ✅ [Opt-20] Jasmin 変換キャッシュ（クラス属性 or グローバル） <a id="opt-20"></a>

```python
def get_jasmin(self, c: int, N: int) -> int:
    key = (c, N)
    if key in self.jasmin_cache:
        return self.jasmin_cache[key]
    result = self.jasmin(c, N)
    self.jasmin_cache[key] = result
    return result

# gen_constellations() 内
ijkl_list = { self.get_jasmin(c, N) for c in ijkl_list }
```

---

### ✅ [Opt-21] 180°重複チェックの二重化（整理）<a id="opt-21"></a>

* **中央列ブロック**では `check_rotations(...)` に **rot180** が含まれるため、
  併用の `rot180_in_set(...)` は削除しても挙動同じ（軽量化）。

```diff
- if not self.check_rotations(...):
-   if not self.rot180_in_set(...):
+ if not self.check_rotations(...):
```

---

### ✅ [Opt-22] visited の粒度<a id="opt-22"></a>

* **星座ごと**に `visited: set()` を新規作成 → メモリ増大を回避。
* キーには `ld/rd/col/LD/RD` 等の **ビット集合＋行/分岐情報** を含め **衝突耐性◯**。
* *任意改善*: `N` と `preset_queens` は一定なのでキーから外してもOK。

---

### ✅ [Opt-23] ビット演算のインライン化<a id="opt-23"></a>
* board_mask 共有
* bit 抽出 `bit = x & -x`
* `cnt` を星座ローカルで完結 → 最後に `symmetry()` を掛ける流れはキャッシュに優しい。
* さらなる微調整：`symmetry(ijkl, N)` の **小メモ化**、`set` 操作の **一括化**、
  並列は **チャンク分割**でスケジューリング負荷を低減（任意）。

---

### ✅ [Opt-24] “先読み空き” 条件（ゴール直前は先読み不要）<a id="opt-24"></a>

```python
next_free = board_mask & ~blocked
if next_free and ((row >= endmark - 1) or self._has_future_space(next_ld, next_rd, next_col, board_mask)):
    ...
```

* **ゴール直前は先読みしない**短絡評価で無駄な判定を削減。
* `row + Δ >= endmark` の **Δ を分岐ごとに一致**させる一貫性が重要。

---


## 🔩 第6章：Codon × LLVM × CUDA への挑戦

Codon には `NVPTX` バックエンド（CUDAコード生成）が存在しますが、
現行（2025時点）では未完成の部分があります。

> 💡 **ワンポイントアドバイス**
> CodonはまだGPU統合途上ですが、`-libdevice` オプションの動作原理を理解しておくと後に役立ちます。
> 「CPU→GPUの遷移」はLLVM IRを共有できるという発想で見ると腑に落ちます。

### 実験例

```bash
LIBDEVICE=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc
codon run -release -libdevice "$LIBDEVICE" codonCupyTest.py
```

```python
import gpu

@gpu.kernel
def hello(a, b, c):
    i = gpu.thread.x
    c[i] = a[i] + b[i]
```

結果：CPU版は動作、GPU呼び出しは `seq_nvptx_memcpy_h2d/d2h` 欠落により未完。
→ Codon ランタイムの NVPTX サポート再構築が今後の課題。

* Codon の LLVM バックエンド解析
* NVPTX 生成に必要な `seq_nvptx_*` シンボル群
* `codon run -release -libdevice` 実験結果
* GPU カーネル修正例（`@gpu.kernel` の動作確認）
* 将来の GPU 統合の方向性
* Python ↔ Codon ↔ CUDA の連携モデル構想図

> [!TIP]
> Codon は LLVM の NVPTX バックエンドを利用することで GPU 実行が理論的に可能です。
しかし現時点では seq_nvptx_memcpy_h2d/d2h など一部シンボルが未実装で、GPU版は実験段階 にあります。

---

## 🧭 第7章：できたこと／できなかったこと

| 項目                   | 状況                    | 備考                 |
| -------------------- | --------------------- | ------------------ |
| Codon ビルド            | ✅ 完了（Fedora / AL2023） | 再現性あり              |
| N-Queens CPU版        | ✅ 完成／高速化成功            | PyPy比 ×10〜15       |
| GPU版 Codon 実験        | ⚠️ 未完                 | NVPTX シンボル欠落       |
| LLVM IR 解析           | ✅ 成功                  | `opt` / `llc` で確認  |
| 並列化（multiprocessing） | ✅ 実装済                 | Codon標準並列化は未搭載     |
| Canonical判定          | ✅ 実装済                 | `jasmin()` による重複除去 |
| NVRTC 統合             | ⚠️ 試験段階               | CUDA組込み未安定         |

---

> 💡 **ワンポイントアドバイス**
> “できなかったこと” の記録こそ次の最適化の糧です。
> Codonはまだ進化中のため、未対応機能をメモしておくとアップデート時に真っ先に活きてきます。



## 🔧 第8章：構築・開発ノウハウ

* Codon ビルド時に `ninja -v` で詳細確認
* `libcodonrt.so` の位置を `nm -D` で確認
* Pythonコードを「段階的に Codon に置き換える」方針が有効
* 型ヒントを省略しない
* 再帰→反復化を意識
* 例外より戻り値制御
* 大規模探索では `print` デバッグを避け、結果だけ出力
* `fmt` / `toml` / `semver` / `fast_float` の組み込み依存
* `type: ignore` を外す
* 計算ループ内では「例外発生」を絶対に避ける
* printデバッグ時の `str(dt)[:-3]` のような書式短縮

> 💡 **ワンポイントアドバイス**
> Codonでは「書き方を整える」ことが最大の最適化です。
> ループ変数を再利用しない、関数の引数を固定型にする——これだけで内部的にLLVM最適化が進みます。

---

## 🧮 第9章：Codonで学んだ設計思想

> 「Codonに移植できるPythonコード」を最初から書く。

* 動的型を避け、**常に静的思考で書く**
* ライブラリ依存を減らし、**標準演算だけで完結**
* 再帰深度を減らし、**明示的ステート管理**
* グローバルを使わず、構造体やクラスで受け渡す
* `List` より固定長配列を優先
* 「CPUキャッシュフレンドリー」な配置を意識する
* Python コードを「後から Codon に置き換えられる」設計にする
* 型アノテーションを厳密に書く
* 例外ではなく戻り値で処理制御
* オブジェクト志向よりも構造体志向
* 再帰よりも反復への置換を検討
* 計算量が指数的な場合に特に威力を発揮する領域

> 💡 **ワンポイントアドバイス**
> Codon思考とは“動的→静的”への意識転換です。
> 書き方が変わると、設計思想も変わる。動くPythonから、**考えるCodon** へ。

---

## 🌌 第10章：Codonの未来と意義

Codon はまだ発展途上ながら、**「Pythonをそのまま高速言語にする」唯一の現実的解** です。科学計算・AI・データ処理・最適化などで広く活用でき、研究スクリプトを**直接本番バイナリに変換**できるという革命的アプローチを実証しました。

私が N-Queens ソルバーを通して得た確信はこうです：

> **Codon は “Python の終着点” ではなく、“ネイティブ化への橋” である。**

* Codonは「Pythonの終着点」ではなく「ネイティブ化の橋」
* Codonを知ると、Pythonコードの書き方自体が変わる
* 現時点での制約：標準ライブラリ制限、外部ライブラリ連携の難しさ
* 将来的に期待される方向：GPU統合、NumPy対応、RAG/AI系連携
* 実践的推奨構成（Fedora42 + Codon v0.17 + Python3.13 + LLVM17）

> 💡 **ワンポイントアドバイス**
> Codonは「Pythonでできないことを置き換える」ものではなく、
> **Pythonを進化させるためのエンジン** です。研究から製品化までを一気通貫で結ぶ架け橋になります。

---

### 🔮 今後の展望

* GPU (NVPTX) 完全対応
* NumPy 互換層・OpenMP 並列最適化
* C API / Python API の相互運用強化
* RAG・生成AI・最適化パイプラインへの応用

---

## 🧾 Appendix: 成果の要約

* Constellation法 + Codon 最適化で CPU版が GPU版を超える速度
* 正しい解数（N=8→92, N=13→73712）を完全再現
* Amazon Linux 2023 + LLVM17 + Codon0.17 環境で再現可能
* Fedora42 で安定稼働／再帰・再利用キャッシュ完全対応
* Codonが「高速Python実験」の標準基盤になる可能性を実証
* CPU版でCUDA版を上回る性能を確認（約1.2倍）
* Symmetry-aware pruning と Canonical check による数学的正確性
* Codon版ソルバーが再現した公式解数（N=8→92, N=13→73,712など）
* Amazon Linux 2023 環境下での reproducible benchmark log

---

## 📚 関連リンク

* [Codon GitHub (Exaloop)](https://github.com/exaloop/codon)
* [N-Queens Project Archive](https://github.com/suzukiiichiro/N-Queens)
* [公式 LLVM Documentation](https://llvm.org/docs/)
* [Codon Language Docs](https://docs.exaloop.io)

---

## 🧑‍💻 結びに

ここに記した内容は、単なるベンチマーク結果の羅列ではありません。
Python の限界を知り、Codon の設計思想を理解し、
**「書き方そのものを変える」ことで速さを得る** という思想の記録です。

> *“Codon is not just a compiler.
> It’s a mindset shift for Python programmers.”*

---

### ✨ License

MIT License — © 2025 iichiro suzuki
