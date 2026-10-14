## 411 — 現段階の最速版を1番号に：408のカーネル＋`.py` から `NQ_EXTRA_CTX=1` を外す

408の結果と採用は `408_results_README_append.md`、10/6午後の探索・診断・A/Bは `408_followup_README_append.md` に記録済み。

### 位置づけ

- **カーネルは408とバイト同一**（コード領域差分 0/0、kernel sha `ab483e70`、SASS `<402>` `821302f2`・446・109／`<403>` `e940cb96`・445・108）。
  ファイル名を揃えるためのリネームで、ヘッダのノートだけ追加。
- **変更は `.py` の1点**：`CRunnerEntry` の `env_prefix` を `"NQ_EXTRA_CTX=1 NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "` →
  `"NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "`。実行行の差分は3行（VERSION_TAG、REV_TAG、CRunnerEntry のパスと env_prefix）。

### 根拠（A/B、2026-10-06 13:58、`408_ab_extra_ctx.sh`）

408 N=21 直接、3セルとも `results.bin` 一致・オラクル・SM 1710 平ら：A0 なし 97,913.1／A1 =1 97,906.3／A2 なし 97,908.6。
d = −0.005%、同一セッションの幅 0.005% → 事前登録どおり**不活性**。プロセス内の余分な CUDA コンテキストはデバイスメモリ
256 MB（free_mb 21,890 → 21,634）を使うだけだった。405以降の直接セルはすべて `NQ_EXTRA_CTX` なしで走っている。

### ゲート

| | 内容 |
|---|---|
| S0 | 静的：`.cu` のコード領域＝408 とバイト一致（cmp）、指紋 0/0、kernel sha、`.py` 指紋 3/3、テーブル項目が `./411_kernel_maxd14` ＋ `"NQ_HELPER_CTX=1 NQ_HELPER_MB=128 "`、`NQ_EXTRA_CTX` を持つテーブル項目が無い、gcc、ノート |
| S1 | ptxas 2実体 160 B／spill 0／レジスタ≤44 |
| S2／S3 | SASS sha が408と同一（`821302f2`／`e940cb96`） |
| Y3 | rc=3×2（Y2 は同一カーネルなので省略） |

### セル（直接＝BLOCK 32／MB 960／ヘルパー1＋128 MB、`NQ_EXTRA_CTX` なし）

| セル | バイナリ | 内容 | 所要 |
|---|---|---|---:|
| B0 | 408 | N=21 auto（layout 403）。アンカー 408 A0 97,913.1（1710 平ら） | 2分 |
| B1 | 411 | N=21 auto。**results.bin＝B0（ハード）**。同一SASSなので時間差＝セッション内のふらつき | 2分 |
| G21 | 411Py | `-g 21 21`。**dispatch.log の env_prefix に `NQ_EXTRA_CTX` が無い（ハード）**、crunner ログ `[gpu-ctx] extra_ctx=0`（ハード）、ヘルパー1、オラクル・layout 403（ハード）、free_mb ≥ 21,379＋200（ハード）、`/tmp/crunner_N21_results.bin` 対 B0 は情報 | 2分 |
| G22 | 411Py | `-g 22 22`。env_prefix・extra_ctx=0・オラクル・layout 403（ハード） | 約14.3分 |

全セルで `norm_ms = kernel_ms × sm_mean / 1710` を併記。

### 事前登録（実測前に固定）

- **利得は主張しない。** R1：|G21_norm／B1_norm − 1| ≤ 0.3% **かつ** |G22_norm／854,582 − 1| ≤ 0.5%（N=22 はセッションを
  またぐ比較なので広め）。R4：全セル 1710±2%。
- **採用則**：ハード全通過＋R1 → `411` が本番（N=21 本番値＝G21、N=22 本番値＝G22、どちらも 408 の ±0.3〜0.5% 以内の見込み）。

所要：約21分（`SKIP_G=1` は B0・B1 のみ約5分、採用不可）。

```
STATIC_ONLY=1 bash 411_validate.sh      # OK=15
              bash 411_validate.sh      # 約21分
```

**事前条件**：`408_kernel_maxd14.cu`／`408Py_kernel_maxd14_final.py`／`408_kernel_maxd14`（無ければ `.cu` から再ビルド）、
`rev386_validation_helpers.py`、N=21／N=22入力。
