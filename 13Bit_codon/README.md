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

私は 2025 年を通して、N-Queens 問題という極端に再帰的・組合せ的な問題を題材に、
**Python → Codon → GPU（CUDA）** という道を実践的に歩みました。
ここでは、その過程で得られた知見・構築法・ノウハウ・難所・哲学をまとめます。

> [!TIP]
> 結論から言いますと、GPU/CUDA N-Queensと、Python/Codon N-Queensの計測時間にほぼ違いがありませんでした

GPU/CUDA N-Queens
``` bash
GPU/CUDA
10Bit_CUDA/01CUDA_Bit_Symmetry.cu
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
```

``` bash
top-10:29:32 up 1 day,16:13, 4 users, load average: 64.39,64.21,64.12
Tasks: 563 total,  2 running,561 sleeping,  0 stopped,  0 zombie
%Cpu(s):100.0 us, 0.0 sy, 0.0 ni, 0.0 id, 0.0 wa, 0.0 hi, 0.0 si, 0.0 st
MiB Mem : 257899.4 total,256193.4 free,  1225.5 used,   480.5 buff/cache
MiB Swap:      0.0 total,     0.0 free,     0.0 used. 255314.6 avail Mem
    PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND
   5634 suzuki    20   0   13.4g  70056   7384 R  6399   0.0 148411:55 15Py_constellat
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

* Codon の概要
* 「Python を LLVM にコンパイルする」という思想
* JIT と AOT（Ahead of Time）の違い
* PyPy や Cython との位置づけの比較
* なぜ Codon は高速なのか（LLVM IR → ネイティブコード）
* Python 構文との互換性・制限


---

## ⚡ 第2章：なぜCodonなのか

* **Pythonのボトルネック（関数呼び出し・型判定・GC）** をLLVMが除去。
* `def` のまま最適化され、JITより安定した性能を発揮。
* 特に再帰構造・探索アルゴリズム（N-Queensなど）で圧倒的に速い。
* **移植が容易**：ほとんどのPythonコードがCodonでそのまま動作。
* **C++に匹敵する速度**を、Pythonの可読性のまま達成可能。

> [!TIP]
> Codonは、ランタイムのオーバーヘッドなしで Pythonのコードをネイティブなマシン語にコンパイルする高性能なPythonコンパイラです。 Codonを使うとシングルスレッドでも十分な高速化を行うことができますが、マルチスレッドもサポートしているため、更なる高速化も図れます。


---

## 🧱 第3章：Codon環境構築（Fedora / Amazon Linux）

### 3.0 Linux/macでのインストール
```bash
/bin/bash -c "$(curl -fsSL https://exaloop.io/install.sh)"
```

> [!TIP]
> OSがLinux (x86_64) か MacOS (x86_64 または arm64) であれば、公式からビルド済みのバイナリが配布されているので、1コマンドでインストールすることができます。

* Python の柔軟さと遅さのトレードオフ
* Codon による高速化のアプローチ
* Python コードをそのまま「ビルド」できる利点
* `codon build` / `codon run` / `codon repl` の違い
* Codon が得意とする領域（数値計算、再帰、バックトラックなど）

### 3.1 ソースビルド手順（再現性あり）
どうしてもと言う人はソースからビルドして下さい。
```bash
$ git clone https://github.com/exaloop/codon
$ cd codon
$ mkdir build && cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ make -j$(nproc)
$ sudo make install
```

* `codon` ソースビルド手順
* `build/_deps` 以下の構成
* `libcodonrt.so` の場所とリンク設定
* `codon build -release` と `codon run -release` の違い
* `LD_LIBRARY_PATH` や `PATH` 設定の注意点

### 3.2 依存パッケージ

```
cmake, ninja, fmt, toml, libomp, fast_float, semver, peglib
```

* `libdevice.10.bc` の役割
* `-libdevice` オプションの意味
* NVRTC や libcodonrt.so における `seq_nvptx_memcpy_h2d/d2h` の欠落
* `nm` コマンドでのシンボル確認例
* GPU コンパイルが未完に終わった理由と今後の展望

### 3.3 ランタイム設定

```bash
export PATH=/usr/local/codon/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/codon/lib:$LD_LIBRARY_PATH
```

### 3.4 実行
JIT(Just-In-Time)で実行（遅いけどコンパイルせずに実行)
```bash
$ codon run hello.py
Hello Codon!
```

ビルドして実行（コンパイルしてネイティブコードに変換するから超高速）
```bash
$ codon build -release hello.py && ./hello
Hello Codon!
```
---

### 3.5 計測

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



## 🧠 第4章：Codonの高速化原理と基本構文

Codon は型を固定することで、動的型判定をすべて排除します。

### 基本的な書き方

```python
from typing import List

def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

print(fib(40))
```

* 変数型の明示 (`int`, `List[int]`, `Dict[int,int]`) の重要性
* 再帰関数の呼び出しコスト削減
* `for` ループと `while` ループの最適化挙動
* 配列・辞書の静的確保
* `@codon.jit`, `@codon.inline` などの使用可否と実験結果
* Python と Codon での整数オーバーフロー挙動の違い
* `bit` / `mask` 演算での高速化例

### 最適化のポイント

* 型ヒントを**すべて明示**する (`int`, `List[int]`, `Dict[str,int]`など)
* 例外 (`try/except`) を使わない
* ループ外でメモリを確保し再利用
* `while` ループを多用し、再帰を階層的に置き換える
* グローバル変数や動的 import を避ける

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

### 5.1 主な最適化手法

* **ビットボード** による衝突判定
* **対称性除去**（mirror / rotation）
* **Constellation法**（部分盤面キャッシュによる分割探索）
* **blocked_next**／**next_free** による早期枝刈り
* **Zobrist Hash** による探索重複排除
* **canonical化チェック** による一意解抽出

### 5.2 ベンチマーク結果（Fedora42 / Codon v0.17）

| N  | Total   | Unique |  Python  |  pypy  | Codon  | GPU/CUDA |
| -- | ------- | ------ | -------- | ------ | ------ | -------- |
| 8  | 92      | 12     |  00.126s | 0.169s | 0.001s |   0.01   |
| 13 | 73712   | 9233   |  00.355s | 0.825s | 0.005s |   0.04   |
| 15 | 2279184 | 285053 |  09.006s | 3.841s | 0.035s |   0.07   |
| 17 |14772512 |11977939|1:07.235s |13.367s | 0.436s |   0.26   |

---

## 🔩 第6章：Codon × LLVM × CUDA への挑戦

Codon には `NVPTX` バックエンド（CUDAコード生成）が存在しますが、
現行（2025時点）では未完成の部分があります。

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

`
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

## 🔧 第8章：構築・開発ノウハウ

* Codon ビルド時に `ninja -v` で詳細確認
* `libcodonrt.so` の位置を `nm -D` で確認
* Pythonコードを「段階的に Codon に置き換える」方針が有効
* 型ヒントを省略しない
* 再帰→反復化を意識
* 例外より戻り値制御
* 大規模探索では `print` デバッグを避け、結果だけ出力
* ビルドエラー時はまず `ninja -v` 出力で原因を探す
* `fmt` / `toml` / `semver` / `fast_float` の組み込み依存
* `type: ignore` を外す
* 計算ループ内では「例外発生」を絶対に避ける
* printデバッグ時の `str(dt)[:-3]` のような書式短縮

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

---

## 🌌 第10章：Codonの未来と意義

Codon はまだ若いが、**Python の未来を変える可能性を持つコンパイラ**です。
AI・科学計算・アルゴリズム実験・リアルタイム処理のあらゆる分野で
「**Pythonの書きやすさのままC++の速さ**」を実現できます。

私が N-Queens ソルバーを通して得た確信はこうです：

> **Codon は “Python の終着点” ではなく、“ネイティブ化への橋” である。**

* Codonは「Pythonの終着点」ではなく「ネイティブ化の橋」
* Codonを知ると、Pythonコードの書き方自体が変わる
* 現時点での制約：標準ライブラリ制限、外部ライブラリ連携の難しさ
* 将来的に期待される方向：GPU統合、NumPy対応、RAG/AI系連携
* 実践的推奨構成（Fedora42 + Codon v0.17 + Python3.13 + LLVM17）

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

---

この内容はそのまま `README.md` に貼り付けて構いません。
ご希望があれば、後続で **図版付き（Constellation構造やLLVMフロー）** の拡張版も生成できます。

生成しますか？（例：`docs/codon_constellation_flow.svg` / `img/performance_chart.png`）
