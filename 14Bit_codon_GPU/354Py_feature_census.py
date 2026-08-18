#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
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


Python/codon Ｎクイーン コンステレーション版 CUDA 高速ソルバ

================================================================================
## 現在の未解決課題 (Open Objectives) -- 最終更新: 352 (2026-07-30)

このセクションはリビジョンごとに更新されるサマリです。詳細な経緯は下の
年代順ログ、および対応するREADME.mdの同名セクションを参照してください。

1. [結論・確定・採用] 非コアレッシングメモリアクセス(w_arrのSoA分割)
   328で実装、N=21フル実行で正当性(314666222712)と456.036秒(327比
   ノイズ内)を確認、329の実機ncu SourceCounters再解析でw_lo_arr/
   両配列の読み込み3箇所全てのL2 Excessive=0を確認し、正式にADOPT済み。
   329自身のフル実行(456.187秒、328比-0.033%)でも再現を確認。
   **351で上位半分を削除した**(項目7)。分割そのもの(=密なu32配列を
   読むこと)がコアレッシングの効いている部分であり、上位配列は恒等的に
   ゼロだったのでロードごと不要である。下位配列はそのまま残る。

2. [撤回・保留] kernel内future_check_mask 1軸専用化
   326で実装したホットループ複製は正当性一致・517.563秒(+13.7%)で撤回。
   240/266-269/273と合わせてGPU側kernel/ループ分解は5戦5敗。保留継続。

3. [結論・確定・330で対策経路を確定] Stall Branch Resolving
   (カーネル全体の約19〜23%、stall_branch_resolvingの65.5%が2箇所の
   BRA/BSYNCに集中、実体はwarp占有率崩壊/tail effect: Divergent
   Branches=0かつAvg Threads Executed 2〜3/32 -- 329で確定)。
   330でrev292以来未解決だった「Codonの@gpu.kernelはatomic演算を
   サポートするか」を実機調査で解決: **サポートしない**(確定)。
   /home/suzuki/.codon/lib/codon/stdlib/gpu.codon にatomicは0ヒット、
   internal/gpu.codonの2ヒットはinternal.gc.atomic(GCアロケータの
   型特性チェック: 型がポインタを含まずGC追跡不要かの判定)であり、
   CUDAデバイスatomic(atomicAdd等)とは無関係。したがって292案C-1
   (persistent kernel + グローバルatomicワークキュー)は純Codonでは
   実装不可能と確定し、正式にクローズ。残る対策経路は、warp intrinsics
   (__shfl_sync/__ballot_sync)やatomicを使えるCUDA Cランナー
   (rev84で提案、未着手: コンステレーション生成/並べ替え/binキャッシュ
   はCodonのまま、カーネルとランチャーのみ.cuで書き同じbinを読む構成)
   のみ。tail effectへの唯一の残存対策として中期候補に位置づける。

4. [新規発見・低優先度・未対応] kernel epilogueのSTG書き込み超過
   329のncu再解析で検出(Excessive=1936×2)。スレッド当たり1回のみの
   実行で影響は無視できる規模。記録のみ。

5. [クローズ・333で正式採用] funcid_reorder w/jのN21再調整 → w3_j7採用
   発掘案C-3の3段スイープが完結した。330(W∈{4,6,8,10,12,16})でW=4を
   候補として発見、331(W∈{2,3,4,5,8})で**W=3が両側ブラケット済みの
   内部最適点(kernel_reduce_ms 448,789、W=8比-1.294%)**と確定、
   332のJ側スイープ((3,3)(3,5)(3,7)(3,11)+(8,7)対照+(1,7)保険、
   全点正当性OK)でJ=3/5/11がいずれも大差で悪化しJ=7の最適性を確認、
   (1,7)は-0.08%で低側第2極小なしも確認した。(3,7)アンカーは331比
   2ms差(448,791)で再現し、キャッシュ済みbinでのelapsed 450.113sは
   事前予測(~450.1s)と一致。(8,7)対照点は4セッション連続で
   454,672〜677ms(全幅5ms=0.001%)。**333で
   A10G_FINAL_DEFAULT_REORDER_WINDOW_MULT/FUNCID_REORDER_V2_WINDOW_MULT
   を8→3に変更し、w3_j7をソースデフォルトとして正式採用**(PHASE_JUMPは
   7のまま)。94-99時代のN22基準チューニングの置き換えであり、
   カーネルロジックには一切触れない(ホスト側の並べ替えパラメータのみ)。
   期待値: elapsed約450.1s(旧456.0s比約-1.3%)、正当性314666222712
   不変。**333の単発フルvalidateで確定した**: 正当性314666222712一致、
   elapsed 450.667s(332直接測定450.113s比-0.123%、ノイズ内)、
   kernel_reduce_ms合計448,920ms(331/332比±0.03%以内)。旧w8_j7
   ベースライン(456.036/456.187s)比 **+1.18%改善**を確認し、発掘案
   C-3は完全クローズ。330→331→332→333の4リビジョン・1日で完結。

6. [設計中・338でC側移植仕様を確定] CUDA Cランナー(案C-2)スパイク設計
   課題3(tail effect)への唯一の残存対策経路。方針は334以来不変:
   コンステレーション生成・並べ替え(broadmarktail/chunkshape148/
   funcid_reorder w3_j7)・binキャッシュはCodonのまま維持し、
   `kernel_dfs_iter_gpu_maxd14`一個のみを.cuに移植する最小構成とする。
   **334-337で土台が全て揃った(確定)**: nvcc=`/usr/local/cuda/bin/nvcc`
   (CUDA 13.0/V13.0.88)、A10G compute_cap=**8.6**(`-arch=sm_86`)、
   cuobjdumpは同梱されておらずSASSは`sudo ncu --section SourceCounters
   --page source`に一本化(334-335)。nvccビルド+GPU実行round-tripを
   実機実証(336、`gpu_sum=cpu_sum=22898104320`、match=True)。bin形式
   (ヘッダーなし・16バイト固定長レコード、`ld`/`rd`/`col`/`startijkl`
   各u32 LE)をソース精読で確定し、C側リーダーで実ファイルに対し検証
   (337、`records_read=2025282`=`EXPECTED_TASKS`、`checksum_u64=
   13342728758502`)。337のN=21フルも正当性314666222712・elapsed
   450.056sで再現済み。
   **338はコードを一切追加しない設計リビジョンである**(324→325と同じ
   パターン)。移植仕様を`338_maxd14_port_spec.md`に文書化した:
   (1) binレコード(ld/rd/col/startijkl)からSoA全配列
       (ld_arr/rd_arr/col_arr/ctrl0_arr/free_arr/markctrl_arr/w)を
       導出する完全な手続き(`build_soa_for_range`と`symmetry`の
       C言語仕様への写像。28分岐のfuncid決定木を含む)
   (2) K-batching(K=48、grid-stride、stride=grid*block=484*32=15488)
       とスレッド当たりスタック(`__array__[u64](13*2)`=208バイト、
       `stack_ptr`/`save_sp`/`cur_depth`/`next_depth`)のC側表現
   (3) N=18限定の突き合わせスコープ(rev84の原提案どおり。N=21フルは
       その後)
   実装(`.cu`)は339以降に回す。

7. [解決・351で採用・352で軸をクローズ] 上位半分は恒等的にゼロ
   `symmetry()`の戻り値は`u64(2)`/`u64(4)`/`u64(8)`の3値のみであり、
   328で分割した上位u32配列は全要素が恒等的に0であった。351でこれを
   5カーネルとディスパッチャから除去し、**フルlaunchで -0.19〜-0.21%**
   を得て採用した。352はソースを一行も変えず、記録の訂正と本項目の
   クローズのみを行う。

   **測定(351)**: chunk0を3つの独立した対照に対して比較し、
   -0.2087%(同一セッション対照)/ -0.1998%(0728の350)/
   -0.1998%(ncuアタッチ下)。**3回とも -0.20% 前後で一致**。
   elapsed は -0.29%(同一セッション)〜-0.17%(前セッション)。

   **【訂正】351で私は「u64乗算はそのまま残る。上位オペランドが常に
   ゼロであることをコンパイラは知りようがない」と4箇所に書いたが、
   これは誤りであった。** `u64(w_lo_arr[i])`はu32ロードのゼロ拡張
   なので上位がゼロであることは中間表現の段階で証明済みであり、
   352のSASS比較で乗算が実際に縮んでいることを確認した。

     350: IMAD.WIDE.U32(lo*lo) + IMAD(hi_w*lo_t) + IMAD(lo_w*hi_t)  3命令
     351: IMAD.WIDE.U32(lo*lo) + IMAD(lo_w*hi_t)                    2命令

   64x64が64x32に退化し、さらに累算が`IMAD.WIDE`の加数へ融合された。
   **351は、352の仕事として切り分けたものの一部を既に取っていた。**

   **352のSASS比較(N=18 launch 0、同一入力・同一グリッド)**:
     構造一致率(レジスタ番号を無視) 650/696 = 93.4%
     命令数 696 -> 688 (-8)。内訳: エピローグ3箇所 -6、それ以外 -2
     LDG.E 13 -> 10。消えたのは3組のペアの片方のみ。他は1対1で保存
     STL/LDL 6 = 6(スピル変化なし)、Stack Size 1024 = 1024
     制御フロー(BRA/BSSY/BSYNC/BREAK)の差分ゼロ
     Registers Per Thread 47 -> 45、ただし Block Limit Registers 40、
     Theoretical Occupancy 33.33% はいずれも不変(レジスタは非律速)

   **-0.20%は生成コードでは説明できない。** 除去した作業はコンステレーション
   1件当たり高々数百サイクル、実測の節約は 8.4x10^6 サイクル/件で
   **4桁足りない**。スピル・占有率・制御フロー・ホットループの命令構成は
   いずれも排除された。

   **同一N=18 launchのストール比較**(実行命令数が2,251,673,225 対
   2,251,672,257 で一致するため、両者は互いに比較可能):
     no_instruction  61,122 -> 64,830  **+6.07%**
     wait           611,867 -> 607,907  -0.65%
     他は全て +-0.7% 以内。総サンプル数は -0.014%(N=18では速くならない)

   カーネルは128バイト短くなり、**全域でレジスタ番号が付け替わった**
   (命令テキストの完全一致率は21.6%)。命令キャッシュ上の整列と
   レジスタバンク競合のパターンが両方とも変わる。`no_instruction`が
   6%動いたことは前者が実在の摂動である直接的証拠である。
   **ただしこれは候補であって検証していない。** 347の轍は踏まない。

   **したがって -0.20% はレバーではなくコンパイルのくじである**(項目12)。
   e詰め込みの解析的期待値は残り2命令程度で約0.00002%であり、
   **測って何が出てもそれはくじの目**である。この軸はここで閉じる。

8. [解決・344で採用] warp内コスト不均質(tail effect) -- CHUNKSHAPE148_BUCKET_RUN=2048

   329で「Divergent Branches=0、Avg Threads Executed 2〜3/32」として特定
   されたtail effectに対し、338で仮説を立て、339で実装、340-343で測定を
   完了し、**344で採用した**。カーネルには一度も触れていない。

   **最終成績(N=21、同一マシン、対照RUN=1=450.183s)**:
     採用値 RUN=2048 -> **431.677s = -4.111%**
     chunk0 -3.853% / chunk1 -4.529% / chunk2 -3.836%(3チャンク同時改善)
     328(SoA採用時)の456.036sからの累積では **-5.341%**

   **確定した3つの規則**:
   (1) **32整列則**(340) 非32倍数の16/48/80は規模によらず-0.26〜-0.34%の
       狭い帯に落ちる。warp幅への整列が必要条件。
   (2) **2冪則**(342) 32整列だが2冪でない96(-0.609%)・384(-0.690%)は
       整列群の底に留まり、隣接2冪の64/128/512に及ばない。
   (3) **飽和は2048**(343) 2048/4096/8192/16384/32768の5値が生成する
       shaped binは**バイト同一**(md5 283b038c...、32,404,512 bytes)。
       タイミングも431.677〜431.764sの0.020%幅に収まる。**RUN軸は
       完全に出し切った。**

   **343で私が書いた「342の推定は誤り」という記述は、それ自体が過剰
   だった**。342は飽和点を`quotas[b] ≈ STEPS/8 ≈ 1936`付近と見積もった
   が、実測の飽和点は2048であり、この見積もりは正しかった。343で述べた
   「保証された飽和点は15488」は十分条件として正しいが必要条件ではなく、
   実際のquota分布はほぼ均等なので2048で飽和する。また342の上限8192は
   最良点2048を上回っており、**341(上限256=最良点)と違って制約に
   なっていなかった**。343の自己批判は事実として不正確だったため訂正する。
   なお343で追加した「上限がBLOCK*MAX_BLOCKS以上であること」の静的
   チェックは、無害かつ将来の設定変更に対する保険として残す。

   **344の実測(2026-07-27)**: 431.944s、同一セッションのRUN=1対照
   450.067s に対して **-4.027%**(chunk0 -3.810% / chunk1 -4.523% /
   chunk2 -3.824%)。343の飽和値431.677sとは+0.062%、RUN=1対照も340の
   450.183sと-0.026%で、いずれもノイズ帯。**採用を確定**。

9. [解決・346で採用] ソートのスコープ -- CHUNKSHAPE148_ITER_SORT=1

   345で0/1/2/3/4をスイープし、346で採用した。カーネルには一度も
   触れていない。

   **効いているのは粒度ではなくスコープである。** mode 1 と mode 3 は
   ソートが完全に同一(どちらも key>>5 の昇順・安定ソート)であり、
   違うのはスコープだけである。それだけで24.991%の差がついた。

     mode 0  off                          431.983s   (アンカー)
     mode 1  launch群 743424 昇順         402.460s   -6.834%
     mode 4  launch群 743424 全key昇順    402.758s   -6.765%
     mode 2  launch群 743424 降順         403.331s   -6.633%
     mode 3  イテレーション 15488 昇順    503.038s  +16.449%

   **なぜスコープが効くのか**: grid-strideのlaunchはwarp wに順位
   `15488*j + 32*w + lane` を渡す。したがって launch群全体をソート
   すると、warp w は **48個の等幅ランク層から1つずつ**サンプリング
   され、warp間が構造的に均衡する。一方イテレーション内だけソート
   すると、warp 0 は毎回最軽量32件、warp 483 は毎回最重量32件を
   受け取り、48回積み上がって不均衡が最大化する。

   **事前登録した4つの予測はすべて的中した**。mode 1 は改善、mode 3 は
   退行、mode 2 は方向対称性により mode 1 と同等、mode 4 は下位5bitが
   同一スコア級内の並べ替えにしか効かないため mode 1 と同等。
   mode 1/2/4 は0.216%幅に収まる。ノイズ床(全体0.009%、per-chunk
   ±0.04%)を上回るので mode 1 が最良であること自体は実在の差だが僅差。

   **最終成績(N=21)**: 345で402.460s、346の採用ランで402.258s(同一
   セッションのmode 0対照431.812s比 -6.844%、chunk0 -7.853% /
   chunk1 -6.412% / chunk2 -6.007%)。344 RUN=1対照450.067s比
   -10.623%、328(SoA採用時)456.036s比で **-11.792%**。

   **再現性が3セッションにまたがって確定した**。同一構成(mode 0相当)の
   独立した3回の測定は 344:431.944s / 345アンカー:431.983s /
   346対照:431.812s で幅0.040%。mode 1側も402.460s→402.258sで-0.050%。
   -6.8%の効果はノイズの100倍以上のマージンを持つ。

   **未解明の観測を1件記録する**: mode 2(降順)のチャンク別内訳が
   分裂する。chunk0 -8.736% / chunk1 -6.843% は mode 1 を上回るのに、
   chunk2 は -3.372% に落ちる。chunk2 は唯一の部分launch(538434件
   = 34.76イテレーション、chunk0/1は48丁度)なので末尾の半端な
   イテレーションとの相互作用を疑ったが、**合成分布のシミュレーションでは
   方向の非対称性を再現できなかった**。したがって機構は確定していない。
   仮説として記録するに留める。347のmode 6がこの点を突く。

10. [解決・350で採用] 蛇行と半端ストラタム -- ITER_SORT=9

   347-349で測定し、350で採用した。カーネルには一度も触れていない。

     349 同一セッション(アンカー = mode 1 = 402.301s)
       mode 1                402.301s       --
       mode 5 (常に蛇行)     401.007s  -0.322%
       mode 9 (条件付き蛇行) 398.988s  -0.824%      <- 採用
       mode 9 vs mode 5                -0.503%

   **タイミングを読む前にバイトで検証した。** mode 9 はフルlaunchでは
   mode 5 と、部分launchでは mode 1 と要素単位で一致するので、shaped bin
   は両者のオフセット23789568でのバイト連結でなければならない。`cmp` が
   mode 9 の2ラン両方でそれを確認した。タイミングも構造どおりに出た:
   chunk0/chunk1 が mode 5 と -0.022% / +0.047%、chunk2 が mode 1 と
   +0.020%、いずれもノイズ床0.04%内。

   **再現性は異常な水準に達している。** mode 1 アンカーは348の402.296sに
   対し **402.301s**、差は**5ミリ秒**(+0.001%)。実行前の投影値398.769s
   に対し実測398.988s、+0.055%。

   **規則の性格を明記する。** mode 9 は「群長がiter_lenの倍数のときだけ
   蛇行する」。**これは経験則である。** 347と348は蛇行が部分launchに
   約1.87%の損失を与えることを2回測定し(0.002ポイントで再現)、しかし
   **その機構は理解されていない**。347で述べた自己補償説明は348の要因
   計画で反証され撤回済みである。規則自体はN固有の定数ではなくlaunch形状
   の判定なので他のNにも一般化するが、**導出ではなく測定に基づく**という
   点は記録に残し続ける。

   **最終成績(N=21)**: 398.988s。346採用値402.258s比 -0.813%、
   344 RUN=1対照450.067s比 -11.349%、328(SoA採用時)456.036s比で
   **-12.510%**。

11. [未解明・要調査] 部分launchにおける蛇行ペナルティ

   蛇行はフルlaunch(48ストラタム)を約1.25%改善し、部分launch(34.76
   ストラタム)を約1.87%悪化させる。347と348で独立に再現済み。348の
   2x2要因計画では light tail 回転も両方の行で chunk2 を悪化させた
   (+1.509% / +4.402%)。**どちらの機構も分かっていない。**
   これはレバーではなく機械についての問いであり、ncuでの調査対象として
   記録する。345で未解明としたmode 2のchunk2分裂も同根の可能性がある。

12. [新規・352で確定] カーネル微修正は「くじ引き」である
    351で確定した事実の一般化。この作業負荷では、カーネルへの小さな
    変更が**その変更が除去した作業とは無関係に**約0.2%の壁時計変動を
    生む。351では削減量と実測が4桁乖離し、SASS比較でスピル・占有率・
    制御フロー・ホットループ命令構成のすべてが排除された。残る候補は
    命令キャッシュ整列とレジスタバンク競合で、どちらも**カーネルに次の
    一手を入れれば引き直しになる**。

    **運用上の帰結**: カーネル微修正の結果は「測れるが帰属できない」。
    採否は実測で決めてよいが、**機構の説明を付けてはならない**し、
    その効果が次の変更後も残ると期待してはならない。同じ0.2%が
    出るたびに同じ議論をしないよう、ここに記録する。


このセクション自体は通常カーネルロジックとは独立しています。
352は**記録の訂正リビジョン**です。**実行コードは351と1バイトも
変わりません。** 変更はVERSION_TAG、WHI_ELIM_REASON、および2つの
docstringのみです。

  訂正   「u64乗算は残る」という351の記述(4箇所)-> 実際は3命令から
         2命令へ縮んでいた。352のSASS比較で確認
  確定   351の -0.19〜-0.21%(フルlaunch、3つの独立した対照に対して
         0.01ポイント以内で一致)
  クローズ 項目7(エピローグ軸)。e詰め込みは行わない
  新設   項目12(カーネル微修正はくじである)

`.sh`の静的チェックは、**docstringとVERSION_TAG/WHI_ELIM_REASONを
除いた実行コードが351とバイト単位で完全一致すること**を検査します
(`source_code_identical_to_351`)。したがって再測定は確認であって
発見ではありません。ホスト側の並べ替え、K=48、SoA、w3_j7、
`CHUNKSHAPE148_BUCKET_RUN=2048`、`CHUNKSHAPE148_ITER_SORT=9`は
すべて351と同一で、shaped binも同一ファイル(`..._isort9.bin`)です。


# ビルド
codon build -release 115Py_range_default_clean_cg_v2.py

# CPU実行
stdbuf -oL -eL ./115Py_range_default_clean_cg_v2 -c 2>&1 | tee 115Py_cpu_range_$(date +%Y%m%d_%H%M%S).log

# GPU実行
stdbuf -oL -eL ./115Py_range_default_clean_cg_v2 -c 2>&1 | tee 115Py_cpu_range_$(date +%Y%m%d_%H%M%S).log


suzuki@cudacodon$ nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv -l 5
clocks.current.sm [MHz], clocks.max.sm [MHz]
1320 MHz, 1710 MHz
suzuki@cudacodon$ date
2026年  7月 23日 木曜日 02:45:14 UTC
suzuki@cudacodon./328Py_warr_soa_split_implement -g
GPU mode selected
 N:             Total           Unique         hh:mm:ss.ms
 5:                10                0          0:00:00.000
 6:                 4                0          0:00:00.003    ok
 7:                40                0          0:00:00.003    ok
 8:                92                0          0:00:00.002    ok
 9:               352                0          0:00:00.002    ok
10:               724                0          0:00:00.003    ok
11:              2680                0          0:00:00.007    ok
12:             14200                0          0:00:00.010    ok
13:             73712                0          0:00:00.012    ok
14:            365596                0          0:00:00.018    ok
15:           2279184                0          0:00:00.032    ok
16:          14772512                0          0:00:00.074    ok
17:          95815104                0          0:00:00.260    ok
18:         666090624                0          0:00:02.005    ok
19:        4968057848                0          0:00:10.887    ok
20:       39029188884                0          0:01:25.200    ok
21:      314666222712                0          0:07:36.107    ok

304Py_K48_sweep_probe.py - elapsed = **351.070s** (0:05:51.070)


2023/11/22 これまでの最高速実装（CUDA GPU 使用/C）
C/CUDA NVIDIA(GPU)
$ nvcc -O3 -arch=sm_61 -m64 -ptx -prec-div=false 04CUDA_Symmetry_BitBoard.cu && POCL_DEBUG=all ./a.out -n ;
対称解除法 GPUビットボード
18:         666090624        83263591    000:00:00:01.65
19:        4968057848       621012754    000:00:00:13.80
20:       39029188884      4878666808    000:00:02:02.52
21:      314666222712     39333324973    000:00:18:46.52
22:     2691008701644    336376244042    000:03:00:22.54
23:    24233937684440   3029242658210    001:06:03:49.29
24:   227514171973736  28439272956934    012:23:38:21.02
25:  2207893435808352 275986683743434    140:07:39:29.96

"""

"""
**354の設計です。** 353の結論(lane_tail 1.14〜1.16倍でCUDA C移植の
workload不均衡側の根拠は消えた、host-side reorderingの床は6.3%、
現行キーの順位相関は−0.36で再設計余地あり)を受けて、353が選んだ軸
——**ホスト側キー再設計**——に進む。**カーネルは一切触らない。**

## なぜ352の上に乗せるか(353の上ではない)

自分たちで決めた運用規則そのものである。「354のカーネルは352から
再開する。計数器を載せたまま先へ進めると項目12の引き直しになる」と
353のVERSION_TAGと`.sh`の両方に明記した。354はカーネルに用がないので、
352のバイト一致カーネルに、**CPU専用のダンプ経路だけ**を足す。

## 何を測るか

現行の並べ替えキーは `chunkshape148_score_key_from_soa`(352の4335行目)
であり、これが読む入力は `TaskSoA` の6フィールド——`funcid_arr` /
`free_arr` / `end_arr` / `row_arr` / `mark1_arr` / `mark2_arr`——**だけ**
である。この6フィールドを作る `build_soa_for_range` は**GPU非依存の
純CPU関数**であることをソースで確認した(`@gpu.kernel`デコレータなし、
`gpu.raw`呼び出しなし)。353の実測でも該当ステージ(`soa_ms`)は
743424件のチャンクで約111msだった。**つまりこの軸の検証にGPUは不要**
であり、N=21のフル実行(400秒級)も不要である。

354は新設の `bench_mode=32` で、既存のshaped bin読み込み経路
(`exec_solutions_gpu_bin_stream_split145` と同じチャンク分割・
`chunk_only`/`chunk_list`の意味論)を再利用しつつ、GPUディスパッチの
代わりに `build_soa_for_range` を呼んで6フィールドと**現行キーの実値**
(関数を直接呼び出す、再実装しない)を `features354_..._chunk<i>.bin`
に書き出す。census353と**同じチャンク境界**(K=48込みのSTEPS=743424)
で書くので、位置で1対1に対応する。

## なぜ導出値ではなく生フィールドを保存するか

`pc`(popcount)や`depth`、`mark_gap`などの**導出値**を354側で計算して
保存する設計も検討したが、それは production の計算式を二重に持つこと
になり、どちらかが将来ズレても気づけない。そこで**6つの生フィールド**
と、**関数呼び出しで得た本物のキー値**の両方を保存する。導出値は
オフライン(Python)でいつでも再計算でき、しかも本物のキー値という
正解と突き合わせて検算できる。

## 検算(このリビジョンの正しさの基準)

354は正解値オラクル(`314666222712`)を持たない——タスク集合にも
GPU計算にも触れないからである。代わりに、**現行キーとの相関を
再現できるか**が正しさの基準になる。

`.sh`は、featuresファイルから再計算したキー値の順位が、shaped bin上の
実際の並び順(position)とどれだけ整合するかを検算し、353が実測した
順位相関(ρ≈−0.36、position vs 実コスト)の**符号と大きさが food
矛盾しないこと**を機械的に確認する。ここが崩れていれば、それは
候補キー探索の前に直すべき「特徴量抽出のバグ」である。

## 候補キー探索(すべてオフライン、GPU不要)

`.sh`内のCPythonヒアドキュメントが、features354とcensus353を位置で
結合し、

1. 現行キー(生の`raw`部分)と実コスト(trips)の相関
2. 候補キーいくつか(`pc`/`depth`/`mark_gap`の重み付けを変えたもの)
   それぞれについて、LPT下界に基づく推定makespan
3. 353が確定した床(modelS headroom 6.3%)に対して、各候補が
   どこまで近づくか

を出力する。**GPUもCodonの再ビルドも要らない。** 候補が現行より
明確に良ければ355でshaped bin再構築(`build_chunkshape148_reordered_bin`
のキー式差し替え)に進み、そこで初めてN=21フルの実測検証を行う。

## 実行手順

```
STATIC_ONLY=1    bash 354Py_feature_census_validate_N21_full_once.sh
FEATURE_SMOKE=1  bash 354Py_feature_census_validate_N21_full_once.sh   # chunk0のみ、GPU不使用、数秒
                 bash 354Py_feature_census_validate_N21_full_once.sh   # 全チャンク、GPU不使用、数秒〜十数秒
```

**壁時計はtiming行にしない。** そもそも比較対象がない(GPUを使わない
リビジョンを352/353の壁時計と比べても無意味)。

## 355はどこから始めるか

**キー再設計が有望と出た場合のみ**、355は352のカーネル(不変)の上で
`chunkshape148_score_key_from_soa`の式だけを差し替え、
`build_chunkshape148_reordered_bin`でshaped binを再構築し、N=21フルで
実測する。**354自体はレバーではない。** 353と対になる「レバー製造機」
の後半であり、355のための候補選定である。

"""

import gpu
import sys
from typing import List,Set,Dict,Tuple
from datetime import datetime

MAXD14:Static[int]=14
MAXD14_ANCESTOR:Static[int]=13
MAXD16:Static[int]=16
MAXD18:Static[int]=18
MAXD20:Static[int]=20
MAXD21:Static[int]=21
SCHED_WORDS14:Static[int]=0  # 210 MAXD14 keeps scalar u32 nibble schedule fields, not local u32 schedule words
SCHED_WORDS16:Static[int]=4
SCHED_WORDS18:Static[int]=5
SCHED_WORDS20:Static[int]=5
SCHED_WORDS21:Static[int]=6

# 292: number of constellations each GPU thread processes sequentially in
# kernel_dfs_iter_gpu_maxd14 via a grid-stride loop (stride = BLOCK*MAX_BLOCKS,
# i.e. the already block/grid-tuned 32x484 launch is unchanged). K=1 reproduces
# 291 exactly (one task per thread, no batching). Edit this constant to sweep
# K=2/4/8/... ; selected_maxd>14 chunks always fall back to the original
# 1-task-per-thread launch regardless of this value (see
# exec_solutions_gpu_chunk_split145).
K_PER_THREAD_MAXD14:Static[int]=48


VERSION_TAG:str="354 feature-census: MEASUREMENT ONLY REVISION, NOT A CANDIDATE FOR ADOPTION, CPU ONLY, NO KERNEL CHANGE. 354 sits on top of 352, not 353: 353 explicitly restarts the kernel lineage from 352 in its own VERSION_TAG, and 354 has no business in the kernel at all, so it inherits the byte identical 352 kernel and adds only a host side dump path. NEW bench_mode 32 reuses the existing shaped bin read loop and chunk_only/chunk_list semantics from exec_solutions_gpu_bin_stream_split145, but in place of a GPU dispatch it calls build_soa_for_range, the exact production host side function, CPU only, no gpu.kernel anywhere in the call chain, confirmed by source inspection and by the 353 smoke log where the equivalent stage measured about 111ms for a 743424 record chunk. THE MOTIVATION IS THE 353 RESULT: lane_tail measured 1.14 to 1.16x, which is far below the port-justifying threshold of 6x fixed in the 353 header, so the workload imbalance case for the CUDA C port is not supported by this evidence; modelS, cross warp imbalance, the achievable ceiling given SIMT lockstep, measured a headroom of 6.300 percent, above the 3 percent threshold, so host side reordering still has ammunition; and rank_corr_position_vs_trips measured about minus 0.36, below the 0.85 threshold that would have closed the key axis. 354 is the offline tool for spending that 6.3 percent. WHAT IS DUMPED: for every task, in the exact chunk boundaries census353 used, the six raw fields chunkshape148_score_key_from_soa reads, funcid, free, end, row, mark1, mark2, PLUS the real production key value obtained by calling that function verbatim rather than reimplementing its arithmetic a second time. Storing raw fields rather than derived ones, popcount, depth, mark_gap, row_to_end, row_to_mark1, means any bug in a candidate key formula tried offline is caught by comparison against the real key column instead of being baked silently into the dump twice. THIS REVISION HAS NO CORRECTNESS ORACLE: it touches neither the task set nor the GPU, so 314666222712 does not apply. Its correctness gate is instead a reproduction check, that the rank correlation recomputed from the dumped fields agrees in sign and rough magnitude with the value 353 measured directly from the real key column, about minus 0.36; a mismatch means the feature extraction is wrong and must be fixed before any candidate key is trusted. CANDIDATE SEARCH IS ENTIRELY OFFLINE, GPU AND N=21 FULL RUN NOT REQUIRED for this revision: several reweightings of the existing raw ingredients are scored against census353 by LPT lower bound projection, ranked against the 6.300 percent ceiling 353 fixed. 355, if this axis produces a winner, restarts from the unchanged 352 kernel and rebuilds the shaped bin with the winning key formula, and only THEN spends an N=21 full run to verify it against 314666222712 and against the timing baseline. WALL CLOCK FROM THIS BUILD IS NOT RECORDED AS A TIMING ROW: there is no comparable baseline for a GPU-free revision."
WHI_ELIM_REASON:str="351 removed the identically zero high half of the SoA w split introduced in 328, and 352 corrects the record without touching a line of executable code. symmetry() yields only 2, 4 or 8, so the high 32 bits of every w value were always zero and the three loads of them in each kernel epilogue were pure waste. Five kernel signatures lose one pointer parameter, the dispatcher builds one array instead of two, and each epilogue reads a single u32 and widens it. CORRECTION FROM 352: 351 said the u64 multiply was left alone because the compiler could not know the high operand was zero. That was wrong. The widened load is a provable zero extension, so the multiply fell from three IMADs to two and the accumulation folded into the widening MAD. A host-side guard ORs every element of w_arr and aborts if the high half is ever nonzero, so the invariant is checked rather than assumed. Host-side reordering is byte identical to 350, so the shaped bin is the same file and is reused. The measured effect on the full launches was 0.19 to 0.21 percent across three independent controls, but the SASS comparison rules out the generated code as the cause, so the epilogue axis is closed and the result is recorded as a measurement without a mechanism."
FEATURE354_REASON:str="354 adds bench_mode 32, a CPU-only pass over the shaped bin that calls build_soa_for_range, the exact production host-side field builder, and dumps six raw fields plus the real production sort key value (via a verbatim call to chunkshape148_score_key_from_soa, not a reimplementation) for every task, in the same chunk boundaries census353 used. No GPU kernel is touched or launched by this path. The correctness gate is a rank-correlation reproduction check against the minus 0.36 value 353 measured directly; 314666222712 does not apply because this revision does not touch the task set. Diagnostic only; 355 restarts from the unchanged 352 kernel if a candidate key is adopted."
CROSS_STRIPE_SAFE_DEFAULT:bool=False

A10G_FINAL_DEFAULT_N:int=22
A10G_FINAL_DEFAULT_BLOCK:int=32
A10G_FINAL_DEFAULT_MAX_BLOCKS:int=484
A10G_FINAL_DEFAULT_LOG_LEVEL:int=0
A10G_FINAL_DEFAULT_SORT_MODE:int=0
A10G_FINAL_DEFAULT_PRESET:int=7
A10G_FINAL_DEFAULT_BENCH_MODE:int=31  # 241: bare -g stays split145 mode31; 240 fid14 launch split rejected
A10G_FINAL_DEFAULT_REORDER_WINDOW_MULT:int=3
A10G_FINAL_DEFAULT_REORDER_PHASE_JUMP:int=7
A10G_FINAL_DEFAULT_CROSS_STRIPE_SAFE:bool=False
A10G_FINAL_DEFAULT_WORKER_ID:int=0
A10G_FINAL_DEFAULT_WORKER_COUNT:int=1
A10G_FINAL_DEFAULT_BROADMARK_VARIANT:int=2
A10G_FINAL_DEFAULT_CHUNKSHAPE148_BUCKET_RUN:int=2048  # 344: ADOPTED (was 1). See CHUNKSHAPE148_BUCKET_RUN.
A10G_FINAL_DEFAULT_CHUNKSHAPE148_ITER_SORT:int=9  # 350: ADOPTED (was 1, and 0 before 346). See CHUNKSHAPE148_ITER_SORT.
CPU_FINAL_DEFAULT_N:int=22
DEFAULT_RANGE_NMIN:int=5
DEFAULT_RANGE_NMAX_EXCLUSIVE:int=24  # range() upper bound; outputs N=5..23

DISABLE_CONSTELLATION_SIGNATURE_PRUNE:bool=False

# 241: keep runtime globals required by broadmarktail/chunkshape148/split145
# core modes. These are not old diagnostics; mode28/29/30/31 and bare -g
# still depend on them for cache names, shaping order, and CLI overrides.
FUNCID_REORDER_V2_WINDOW_MULT:int=3
FUNCID_REORDER_V2_PHASE_JUMP:int=7
FUNCID_REORDER_V2_DEFAULT_REASON:str="N21 measured best w3_j7 (330-332 sweeps, kernel_reduce_ms 448789-448791 vs w8_j7 454672-454677, -1.294%; replaces the rev94-99-era N22-tuned w8_j7)"
BROAD_MARKDIST_TAIL_REORDER_VERSION:str="v4"
BROAD_MARKDIST_TAIL_REORDER_DEFAULT_REASON:str="333 w3j7-adopt: host-side parameter adoption only, kernel/dispatcher identical to 328-332; w3_j7 replaces the rev94-99-era N22-tuned w8_j7 as the funcid_reorder default based on the 330-332 three-stage sweep (W=3 bracketed interior optimum, J=7 confirmed, kernel_reduce_ms -1.294%, metric reproducible to 0.001% across four sessions); expected elapsed ~450.1s vs 456.0s baseline, correctness 314666222712 unchanged (reordering does not alter the task set)"
BROAD_MARKDIST_TAIL_VARIANT:int=2
BROAD_MARKDIST_TAIL_PHASE_SALT:int=53
BROAD_MARKDIST_TAIL_CELL_SALT:int=17
BROAD_MARKDIST_TAIL_RISK_SALT:int=11

class TaskSoA:
  def __init__(self,m:int)->None:
    self.ld_arr:List[u32]=[u32(0)]*m
    self.rd_arr:List[u32]=[u32(0)]*m
    self.col_arr:List[u32]=[u32(0)]*m
    self.row_arr:List[int]=[0]*m
    self.ctrl0_arr:List[u32]=[u32(0)]*m
    self.free_arr:List[u32]=[u32(0)]*m
    self.markctrl_arr:List[u32]=[u32(0)]*m
    self.jmark_arr:List[int]=[0]*m
    self.end_arr:List[int]=[0]*m
    self.mark1_arr:List[int]=[0]*m
    self.mark2_arr:List[int]=[0]*m
    self.funcid_arr:List[int]=[0]*m
    self.ijkl_arr:List[int]=[0]*m

def schedule_depth_for_task(ctrl0:u32,markctrl:u32,meta_next:List[u8])->int:
  IS_BASE_MASK_I:int=69222408
  IS_JMARK_MASK_I:int=4
  IS_MARK_MASK_I:int=199209203
  IS_P5_MASK_I:int=3840
  SEL2_MASK_I:int=34742338
  STP3_MASK_I:int=21266576

  raw:int=int(ctrl0)
  marks:int=int(markctrl)
  jmark:int=marks&31
  endm:int=(marks>>5)&31
  mark1:int=(marks>>10)&31
  mark2:int=(marks>>15)&31
  depth:int=0

  while True:
    fu:int=raw&31
    rowv:int=(raw>>5)&31

    if ((IS_P5_MASK_I>>fu)&1)!=0 and rowv==mark1:
      fu=int(meta_next[fu])

    if ((IS_BASE_MASK_I>>fu)&1)!=0 and rowv==endm:
      return depth

    stepv:int=1
    nextfid:int=fu
    if ((IS_MARK_MASK_I>>fu)&1)!=0:
      markv:int=mark2 if ((SEL2_MASK_I>>fu)&1)!=0 else mark1
      if rowv==markv:
        stepv=3 if ((STP3_MASK_I>>fu)&1)!=0 else 2
        nextfid=int(meta_next[fu])

    if ((IS_JMARK_MASK_I>>fu)&1)!=0 and rowv==jmark:
      nextfid=int(meta_next[fu])

    child_row:int=rowv+stepv
    depth+=1
    if depth>21 or child_row>31:
      return 22
    raw=nextfid|(child_row<<5)

def max_schedule_depth_of_tasks(soa:TaskSoA,m:int,meta_next:List[u8])->int:
  required_maxd:int=0
  i:int=0
  while i<m:
    d:int=schedule_depth_for_task(soa.ctrl0_arr[i],soa.markctrl_arr[i],meta_next)
    if d>required_maxd:
      required_maxd=d
    i+=1
  return required_maxd

def select_static_maxd(required_maxd:int)->int:
  if required_maxd<=14:
    return 14
  if required_maxd<=16:
    return 16
  if required_maxd<=18:
    return 18
  if required_maxd<=20:
    return 20
  if required_maxd<=21:
    return 21
  return 0

def packed_schedule_words_for_maxd(selected_maxd:int)->int:
  if selected_maxd==14:
    return 0
  if selected_maxd==16:
    return 4
  if selected_maxd==18:
    return 5
  if selected_maxd==20:
    return 5
  if selected_maxd==21:
    return 6
  return 0

def packed_stack_bytes_per_thread(selected_maxd:int)->int:
  if selected_maxd==14:
    return 208
  words:int=packed_schedule_words_for_maxd(selected_maxd)
  if words==0:
    return 0
  return selected_maxd*16+words*4

@gpu.kernel
def kernel_dfs_iter_gpu_maxd14(
    ld_arr:Ptr[u32],rd_arr:Ptr[u32],col_arr:Ptr[u32],ctrl0_arr:Ptr[u32],free_arr:Ptr[u32],
    markctrl_arr:Ptr[u32],w_lo_arr:Ptr[u32],
    meta_next:Ptr[u8],
    results:Ptr[u64],
    m:int,board_mask:u32,
    n3:u32,n4:u32,
    stride:int,
)->None:
    IS_BASE_MASK:u32=u32(69222408)
    IS_JMARK_MASK:u32=u32(4)
    IS_MARK_MASK:u32=u32(199209203)
    IS_P5_MASK:u32=u32(3840)
    SEL2_MASK:u32=u32(34742338)

    BLOCK_CODE_B0_MASK:u32=u32(173707345)
    BLOCK_CODE_B1_MASK:u32=u32(12689458)
    BLOCK_CODE_B2_MASK:u32=u32(18088064)

    OP_STEP3_MASK:u32=u32(24)  # codes 3,4
    OP_ADD1_MASK:u32=u32(32)   # code 5
    OP_BL1_MASK:u32=u32(12)    # codes 2,3
    OP_BL2_MASK:u32=u32(16)    # code 4
    OP_KN3_MASK:u32=u32(18)    # codes 1,4
    OP_KN4_MASK:u32=u32(8)     # code 3

    # 292: stack arrays are allocated once per GPU thread and reused across
    # every constellation the thread processes in the grid-stride loop below.
    # 294: pack 4x u32 into 2x u64 (ldrd/colav). 295: merge into single
    # u64 array of size MAXD14_ANCESTOR*2 so push/pop accesses two adjacent
    # words at indices save_sp*2 and save_sp*2+1, improving cache-line
    # locality (both words land in same 128-bit cache line entry).
    # stack[ptr]   = ldrd  (lo32:ld  | hi32:rd)  where ptr=stack_ptr
    # stack[ptr+1] = colav (lo32:col | hi32:avail|(depth<<27))
    # 296: stack_ptr maintained directly (=save_sp*2) to avoid multiply.
    stack=__array__[u64](MAXD14_ANCESTOR*2)
    bm:u32=board_mask
    tid:int=(gpu.block.x*gpu.block.dim.x)+gpu.thread.x
    if tid>=stride:return

    # 292: grid-stride loop over K=ceil(m/stride) constellations per thread.
    # stride == grid*block, i.e. the already-tuned 32x484 launch config is
    # unchanged; only the number of constellations processed per thread grows.
    thread_total:u64=u64(0)
    idx:int=tid
    while idx<m:
      markctrl:u32=markctrl_arr[idx]
      jmark:u32=markctrl&u32(31)
      endm:u32=(markctrl>>u32(5))&u32(31)
      mark1:u32=(markctrl>>u32(10))&u32(31)
      mark2:u32=(markctrl>>u32(15))&u32(31)
      total:u64=u64(0)

      root_ld:u32=ld_arr[idx]
      root_rd:u32=rd_arr[idx]
      root_col:u32=col_arr[idx]
      root_a:u32=free_arr[idx]&bm
      if root_a==u32(0):
        idx+=stride
        continue

      schedule_raw:u32=ctrl0_arr[idx]
      schedule_depth:int=0
      schedule_lo:u32=u32(0)
      schedule_hi:u32=u32(0)
      child_jmark_mask:u32=u32(0)
      future_check_mask:u32=u32(0)
      terminal_parent_depth:int=0
      terminal_is_base14:u32=u32(0)
      root_action:u32=u32(0)
      while True:
        schedule_fu:u32=schedule_raw&u32(31)
        schedule_rowv:u32=(schedule_raw>>u32(5))&u32(31)

        if ((IS_P5_MASK>>schedule_fu)&u32(1))!=u32(0):
          if schedule_rowv==mark1:
            schedule_fu=u32(meta_next[int(schedule_fu)])

        frame_action:u32=u32(0)
        frame_nibble:u32=u32(0)
        frame_raw:u32=u32(0)
        schedule_isbu:u32=(IS_BASE_MASK>>schedule_fu)&u32(1)
        if schedule_isbu!=u32(0) and schedule_rowv==endm:
          frame_action=u32(3) if schedule_fu==u32(14) else u32(2)
        else:
          schedule_ismu:u32=(IS_MARK_MASK>>schedule_fu)&u32(1)
          schedule_block_code:u32=u32(0)
          schedule_stepv:u32=u32(1)
          schedule_use_futureu:u32=u32(1)-schedule_ismu
          schedule_nextfidu:u32=schedule_fu

          if schedule_ismu!=u32(0):
            schedule_markv:u32=mark2 if ((SEL2_MASK>>schedule_fu)&u32(1))!=u32(0) else mark1
            if schedule_rowv==schedule_markv:
              schedule_block_code=(
                ((BLOCK_CODE_B0_MASK>>schedule_fu)&u32(1))
                |(((BLOCK_CODE_B1_MASK>>schedule_fu)&u32(1))<<u32(1))
                |(((BLOCK_CODE_B2_MASK>>schedule_fu)&u32(1))<<u32(2))
              )
              schedule_stepv=u32(2)+((OP_STEP3_MASK>>schedule_block_code)&u32(1))
              schedule_use_futureu=u32(0)
              schedule_nextfidu=u32(meta_next[int(schedule_fu)])

          schedule_isju:u32=(IS_JMARK_MASK>>schedule_fu)&u32(1)
          if schedule_isju!=u32(0):
            if schedule_rowv==jmark:
              frame_action=u32(1)
              schedule_nextfidu=u32(meta_next[int(schedule_fu)])

          schedule_child_rowu:u32=schedule_rowv+schedule_stepv
          schedule_fcvu:u32=u32(0)
          if schedule_use_futureu!=u32(0) and schedule_child_rowu<endm:
            schedule_fcvu=u32(1)
          frame_nibble=schedule_block_code|(schedule_fcvu<<u32(3))
          frame_raw=schedule_nextfidu|(schedule_child_rowu<<u32(5))

        if schedule_depth==0:
          root_action=frame_action
        else:
          parent_depth:int=schedule_depth-1
          if frame_action==u32(1):
            child_jmark_mask|=u32(1)<<u32(parent_depth)
          elif frame_action>=u32(2):
            terminal_parent_depth=parent_depth
            terminal_is_base14=u32(1) if frame_action==u32(3) else u32(0)

        if frame_action>=u32(2):
          break

        if schedule_fcvu!=u32(0):
          future_check_mask|=u32(1)<<u32(schedule_depth)

        if schedule_depth<8:
          schedule_lo|=frame_nibble<<u32(schedule_depth*4)
        else:
          schedule_hi|=frame_nibble<<u32((schedule_depth-8)*4)
        schedule_raw=frame_raw
        schedule_depth+=1

      if root_action==u32(2):
        thread_total+=u64(w_lo_arr[idx])
        idx+=stride
        continue
      if root_action==u32(3):
        total+=u64(1) if ((root_a&~u32(1))!=u32(0)) else u64(0)
        thread_total+=total*u64(w_lo_arr[idx])
        idx+=stride
        continue
      if root_action==u32(1):
        root_a&=~u32(1)
        if root_a==u32(0):
          idx+=stride
          continue
        root_ld|=u32(1)

      terminal_depth:int=terminal_parent_depth
      terminal_base14:u32=terminal_is_base14

      save_sp:int=0
      stack_ptr:int=0
      cur_depth:int=0
      cur_ld:u32=root_ld
      cur_rd:u32=root_rd
      cur_col:u32=root_col
      cur_avail:u32=root_a

      root_rest:u32=cur_avail&(cur_avail-u32(1))
      root_second:u32=root_rest&(u32(0)-root_rest)
      root_after_second:u32=root_rest^root_second

      if root_after_second==u32(0):
        root_first:u32=cur_avail&(u32(0)-cur_avail)
        pr_nibble_op:u32=schedule_lo&u32(15)
        pr_block_code:u32=pr_nibble_op&u32(7)
        pr_bit:u32=root_first

        pr_nld:u32=u32(0)
        pr_nrd:u32=u32(0)
        if pr_block_code!=u32(0):
          pr_stepu:u32=u32(2)+((OP_STEP3_MASK>>pr_block_code)&u32(1))
          pr_addvu:u32=(OP_ADD1_MASK>>pr_block_code)&u32(1)
          pr_bLiu:u32=(
            ((OP_BL1_MASK>>pr_block_code)&u32(1))
            |(((OP_BL2_MASK>>pr_block_code)&u32(1))<<u32(1))
          )
          pr_ktu:u32=(
            ((OP_KN3_MASK>>pr_block_code)&u32(1))
            |(((OP_KN4_MASK>>pr_block_code)&u32(1))<<u32(1))
          )
          pr_bKu:u32=(n3&(u32(0)-(pr_ktu&u32(1))))|(n4&(u32(0)-(pr_ktu>>u32(1))))
          pr_nld=((cur_ld|pr_bit)<<pr_stepu)|pr_addvu|pr_bLiu
          pr_nrd=((cur_rd|pr_bit)>>pr_stepu)|pr_bKu
        else:
          pr_nld=(cur_ld|pr_bit)<<u32(1)
          pr_nrd=(cur_rd|pr_bit)>>u32(1)
        pr_ncol:u32=cur_col|pr_bit
        pr_nf:u32=bm&~(pr_nld|pr_nrd|pr_ncol)
        pr_descend:u32=u32(1)
        if pr_nf==u32(0):
          pr_descend=u32(0)
        if pr_descend!=u32(0):
          if future_check_mask!=u32(0):
            if (pr_nibble_op&u32(8))!=u32(0):
              if (bm&~((pr_nld<<u32(1))|(pr_nrd>>u32(1))|pr_ncol))==u32(0):
                pr_descend=u32(0)

        if pr_descend!=u32(0):
          if terminal_depth==0:
            if terminal_base14==u32(0):
              total+=u64(1)
            else:
              total+=u64(1) if ((pr_nf&~u32(1))!=u32(0)) else u64(0)
            pr_descend=u32(0)

        if pr_descend!=u32(0):
          pr_child_jmark:u32=child_jmark_mask&u32(1)
          if pr_child_jmark!=u32(0):
            pr_nf&=~u32(1)
            if pr_nf==u32(0):
              pr_descend=u32(0)
            else:
              pr_nld|=u32(1)

        cur_avail=root_rest
        if pr_descend!=u32(0):
          if cur_avail!=u32(0):
            stack[stack_ptr]=u64(cur_ld)|(u64(cur_rd)<<u64(32))
            stack[stack_ptr+1]=u64(cur_col)|(u64(cur_avail|(u32(cur_depth)<<u32(27)))<<u64(32))
            stack_ptr+=2
            save_sp+=1
          cur_ld=pr_nld
          cur_rd=pr_nrd
          cur_col=pr_ncol
          cur_avail=pr_nf
          cur_depth=1

      while True:
        if cur_avail==u32(0):
          if save_sp==0:
            break
          save_sp-=1
          stack_ptr-=2
          packed_ldrd:u64=stack[stack_ptr]
          packed_colav:u64=stack[stack_ptr+1]
          cur_ld=u32(packed_ldrd)
          cur_rd=u32(packed_ldrd>>u64(32))
          cur_col=u32(packed_colav)
          saved_avail:u32=u32(packed_colav>>u64(32))
          cur_avail=saved_avail&bm
          cur_depth=int(saved_avail>>u32(27))
          continue

        nibble_op:u32=u32(0)
        if cur_depth<8:
          nibble_op=(schedule_lo>>u32(cur_depth*4))&u32(15)
        else:
          nibble_op=(schedule_hi>>u32((cur_depth-8)*4))&u32(15)
        bit:u32=cur_avail&(u32(0)-cur_avail)
        cur_avail=cur_avail^bit

        # 291: keep 289 normal-default nld/nrd + ncol-only early, but delay
        # block_code scalar creation to the special branch only.
        # nf is still computed once after the branch; 288 nf-default is intentionally not used.
        nld:u32=(cur_ld|bit)<<u32(1)
        nrd:u32=(cur_rd|bit)>>u32(1)
        ncol:u32=cur_col|bit
        if (nibble_op&u32(7))!=u32(0):
          block_code:u32=nibble_op&u32(7)
          stepu:u32=u32(2)+((OP_STEP3_MASK>>block_code)&u32(1))
          addvu:u32=(OP_ADD1_MASK>>block_code)&u32(1)
          bLiu:u32=(
            ((OP_BL1_MASK>>block_code)&u32(1))
            |(((OP_BL2_MASK>>block_code)&u32(1))<<u32(1))
          )
          ktu:u32=(
            ((OP_KN3_MASK>>block_code)&u32(1))
            |(((OP_KN4_MASK>>block_code)&u32(1))<<u32(1))
          )
          bKu:u32=(n3&(u32(0)-(ktu&u32(1))))|(n4&(u32(0)-(ktu>>u32(1))))
          nld=((cur_ld|bit)<<stepu)|addvu|bLiu
          nrd=((cur_rd|bit)>>stepu)|bKu
        nf:u32=bm&~(nld|nrd|ncol)
        if nf==u32(0):
          continue
        if future_check_mask!=u32(0):
          if (nibble_op&u32(8))!=u32(0):
            if (bm&~((nld<<u32(1))|(nrd>>u32(1))|ncol))==u32(0):
              continue

        if cur_depth==terminal_depth:
          if terminal_base14==u32(0):
            total+=u64(1)
          else:
            total+=u64(1) if ((nf&~u32(1))!=u32(0)) else u64(0)
          continue

        child_jmark:u32=(child_jmark_mask>>u32(cur_depth))&u32(1)
        if child_jmark!=u32(0):
          nf&=~u32(1)
          if nf==u32(0):
            continue
          nld|=u32(1)

        next_depth:int=cur_depth+1
        if cur_avail!=u32(0):
          stack[stack_ptr]=u64(cur_ld)|(u64(cur_rd)<<u64(32))
          stack[stack_ptr+1]=u64(cur_col)|(u64(cur_avail|(u32(cur_depth)<<u32(27)))<<u64(32))
          stack_ptr+=2
          save_sp+=1
        cur_ld=nld
        cur_rd=nrd
        cur_col=ncol
        cur_avail=nf
        cur_depth=next_depth
      thread_total+=total*u64(w_lo_arr[idx])
      idx+=stride
    results[tid]=thread_total

@gpu.kernel
def kernel_dfs_iter_gpu_maxd16(
    ld_arr:Ptr[u32],rd_arr:Ptr[u32],col_arr:Ptr[u32],ctrl0_arr:Ptr[u32],free_arr:Ptr[u32],
    markctrl_arr:Ptr[u32],w_lo_arr:Ptr[u32],
    meta_next:Ptr[u8],
    results:Ptr[u64],
    m:int,board_mask:u32,
    n3:u32,n4:u32,
)->None:
    IS_BASE_MASK:u32=u32(69222408)
    IS_JMARK_MASK:u32=u32(4)
    IS_MARK_MASK:u32=u32(199209203)
    IS_P5_MASK:u32=u32(3840)
    SEL2_MASK:u32=u32(34742338)

    BLOCK_CODE_B0_MASK:u32=u32(173707345)
    BLOCK_CODE_B1_MASK:u32=u32(12689458)
    BLOCK_CODE_B2_MASK:u32=u32(18088064)

    OP_STEP3_MASK:u32=u32(24)  # codes 3,4
    OP_ADD1_MASK:u32=u32(32)   # code 5
    OP_BL1_MASK:u32=u32(12)    # codes 2,3
    OP_BL2_MASK:u32=u32(16)    # code 4
    OP_KN3_MASK:u32=u32(18)    # codes 1,4
    OP_KN4_MASK:u32=u32(8)     # code 3

    ld=__array__[u32](MAXD16)
    rd=__array__[u32](MAXD16)
    col=__array__[u32](MAXD16)
    avail=__array__[u32](MAXD16)
    packed_schedule=__array__[u32](SCHED_WORDS16)
    bm:u32=board_mask
    i:int=(gpu.block.x*gpu.block.dim.x)+gpu.thread.x
    if i>=m:return

    markctrl:u32=markctrl_arr[i]
    jmark:u32=markctrl&u32(31)
    endm:u32=(markctrl>>u32(5))&u32(31)
    mark1:u32=(markctrl>>u32(10))&u32(31)
    mark2:u32=(markctrl>>u32(15))&u32(31)
    total:u64=u64(0)

    root_ld:u32=ld_arr[i]
    root_rd:u32=rd_arr[i]
    root_col:u32=col_arr[i]
    root_a:u32=free_arr[i]&bm
    if root_a==u32(0):
      results[i]=u64(0)
      return

    schedule_raw:u32=ctrl0_arr[i]
    schedule_depth:int=0
    pack_word:u32=u32(0)
    pack_word_index:int=0
    root_action:u32=u32(0)
    while True:
      schedule_fu:u32=schedule_raw&u32(31)
      schedule_rowv:u32=(schedule_raw>>u32(5))&u32(31)

      if ((IS_P5_MASK>>schedule_fu)&u32(1))!=u32(0):
        if schedule_rowv==mark1:
          schedule_fu=u32(meta_next[int(schedule_fu)])

      frame_action:u32=u32(0)
      frame_opcode:u32=u32(0)
      frame_raw:u32=u32(0)
      schedule_isbu:u32=(IS_BASE_MASK>>schedule_fu)&u32(1)
      if schedule_isbu!=u32(0) and schedule_rowv==endm:
        frame_action=u32(3) if schedule_fu==u32(14) else u32(2)
      else:
        schedule_ismu:u32=(IS_MARK_MASK>>schedule_fu)&u32(1)
        schedule_block_code:u32=u32(0)
        schedule_stepv:u32=u32(1)
        schedule_use_futureu:u32=u32(1)-schedule_ismu
        schedule_nextfidu:u32=schedule_fu

        if schedule_ismu!=u32(0):
          schedule_markv:u32=mark2 if ((SEL2_MASK>>schedule_fu)&u32(1))!=u32(0) else mark1
          if schedule_rowv==schedule_markv:
            schedule_block_code=(
              ((BLOCK_CODE_B0_MASK>>schedule_fu)&u32(1))
              |(((BLOCK_CODE_B1_MASK>>schedule_fu)&u32(1))<<u32(1))
              |(((BLOCK_CODE_B2_MASK>>schedule_fu)&u32(1))<<u32(2))
            )
            schedule_stepv=u32(2)+((OP_STEP3_MASK>>schedule_block_code)&u32(1))
            schedule_use_futureu=u32(0)
            schedule_nextfidu=u32(meta_next[int(schedule_fu)])

        schedule_isju:u32=(IS_JMARK_MASK>>schedule_fu)&u32(1)
        if schedule_isju!=u32(0):
          if schedule_rowv==jmark:
            frame_action=u32(1)
            schedule_nextfidu=u32(meta_next[int(schedule_fu)])

        schedule_child_rowu:u32=schedule_rowv+schedule_stepv
        schedule_fcvu:u32=u32(0)
        if schedule_use_futureu!=u32(0) and schedule_child_rowu<endm:
          schedule_fcvu=u32(1)
        frame_opcode=schedule_block_code|(schedule_fcvu<<u32(3))
        frame_raw=schedule_nextfidu|(schedule_child_rowu<<u32(5))

      if schedule_depth==0:
        root_action=frame_action
      else:
        parent_lane:int=(schedule_depth-1)&3
        parent_action_shift:u32=u32(parent_lane*8+4)
        pack_word|=frame_action<<parent_action_shift
        if (schedule_depth&3)==0:
          packed_schedule[pack_word_index]=pack_word
          pack_word_index+=1
          pack_word=u32(0)

      if frame_action>=u32(2):
        if schedule_depth>0 and (schedule_depth&3)!=0:
          packed_schedule[pack_word_index]=pack_word
        break

      opcode_shift:u32=u32((schedule_depth&3)*8)
      pack_word|=frame_opcode<<opcode_shift
      schedule_raw=frame_raw
      schedule_depth+=1

    if root_action==u32(2):
      results[i]=u64(w_lo_arr[i])
      return
    if root_action==u32(3):
      total+=u64(1) if ((root_a&~u32(1))!=u32(0)) else u64(0)
      results[i]=total*u64(w_lo_arr[i])
      return
    if root_action==u32(1):
      root_a&=~u32(1)
      if root_a==u32(0):
        results[i]=u64(0)
        return
      root_ld|=u32(1)

    sp:int=0
    ld[0]=root_ld
    rd[0]=root_rd
    col[0]=root_col
    avail[0]=root_a

    while True:
      a:u32=avail[sp]
      if a==u32(0):
        if sp==0:
          break
        sp-=1
        continue

      opcode_word:u32=packed_schedule[sp>>2]
      opcode:u32=(opcode_word>>u32((sp&3)*8))&u32(255)
      block_code:u32=opcode&u32(7)
      bit:u32=a&(u32(0)-a)
      avail[sp]=a^bit

      nld:u32=u32(0)
      nrd:u32=u32(0)
      if block_code!=u32(0):
        stepu:u32=u32(2)+((OP_STEP3_MASK>>block_code)&u32(1))
        addvu:u32=(OP_ADD1_MASK>>block_code)&u32(1)
        bLiu:u32=(
          ((OP_BL1_MASK>>block_code)&u32(1))
          |(((OP_BL2_MASK>>block_code)&u32(1))<<u32(1))
        )
        ktu:u32=(
          ((OP_KN3_MASK>>block_code)&u32(1))
          |(((OP_KN4_MASK>>block_code)&u32(1))<<u32(1))
        )
        bKu:u32=(n3&(u32(0)-(ktu&u32(1))))|(n4&(u32(0)-(ktu>>u32(1))))
        nld=((ld[sp]|bit)<<stepu)|addvu|bLiu
        nrd=((rd[sp]|bit)>>stepu)|bKu
      else:
        nld=(ld[sp]|bit)<<u32(1)
        nrd=(rd[sp]|bit)>>u32(1)
      ncol:u32=col[sp]|bit
      nf:u32=bm&~(nld|nrd|ncol)
      if nf==u32(0):
        continue
      if (opcode&u32(8))!=u32(0):
        if (bm&~((nld<<u32(1))|(nrd>>u32(1))|ncol))==u32(0):
          continue

      child_action:u32=(opcode>>u32(4))&u32(3)
      if child_action>=u32(2):
        if child_action==u32(2):
          total+=u64(1)
        else:
          total+=u64(1) if ((nf&~u32(1))!=u32(0)) else u64(0)
        continue
      if child_action==u32(1):
        nf&=~u32(1)
        if nf==u32(0):
          continue
        nld|=u32(1)

      sp+=1
      ld[sp]=nld
      rd[sp]=nrd
      col[sp]=ncol
      avail[sp]=nf
    results[i]=total*u64(w_lo_arr[i])

@gpu.kernel
def kernel_dfs_iter_gpu_maxd18(
    ld_arr:Ptr[u32],rd_arr:Ptr[u32],col_arr:Ptr[u32],ctrl0_arr:Ptr[u32],free_arr:Ptr[u32],
    markctrl_arr:Ptr[u32],w_lo_arr:Ptr[u32],
    meta_next:Ptr[u8],
    results:Ptr[u64],
    m:int,board_mask:u32,
    n3:u32,n4:u32,
)->None:
    IS_BASE_MASK:u32=u32(69222408)
    IS_JMARK_MASK:u32=u32(4)
    IS_MARK_MASK:u32=u32(199209203)
    IS_P5_MASK:u32=u32(3840)
    SEL2_MASK:u32=u32(34742338)

    BLOCK_CODE_B0_MASK:u32=u32(173707345)
    BLOCK_CODE_B1_MASK:u32=u32(12689458)
    BLOCK_CODE_B2_MASK:u32=u32(18088064)

    OP_STEP3_MASK:u32=u32(24)  # codes 3,4
    OP_ADD1_MASK:u32=u32(32)   # code 5
    OP_BL1_MASK:u32=u32(12)    # codes 2,3
    OP_BL2_MASK:u32=u32(16)    # code 4
    OP_KN3_MASK:u32=u32(18)    # codes 1,4
    OP_KN4_MASK:u32=u32(8)     # code 3

    ld=__array__[u32](MAXD18)
    rd=__array__[u32](MAXD18)
    col=__array__[u32](MAXD18)
    avail=__array__[u32](MAXD18)
    packed_schedule=__array__[u32](SCHED_WORDS18)
    bm:u32=board_mask
    i:int=(gpu.block.x*gpu.block.dim.x)+gpu.thread.x
    if i>=m:return

    markctrl:u32=markctrl_arr[i]
    jmark:u32=markctrl&u32(31)
    endm:u32=(markctrl>>u32(5))&u32(31)
    mark1:u32=(markctrl>>u32(10))&u32(31)
    mark2:u32=(markctrl>>u32(15))&u32(31)
    total:u64=u64(0)

    root_ld:u32=ld_arr[i]
    root_rd:u32=rd_arr[i]
    root_col:u32=col_arr[i]
    root_a:u32=free_arr[i]&bm
    if root_a==u32(0):
      results[i]=u64(0)
      return

    schedule_raw:u32=ctrl0_arr[i]
    schedule_depth:int=0
    pack_word:u32=u32(0)
    pack_word_index:int=0
    root_action:u32=u32(0)
    while True:
      schedule_fu:u32=schedule_raw&u32(31)
      schedule_rowv:u32=(schedule_raw>>u32(5))&u32(31)

      if ((IS_P5_MASK>>schedule_fu)&u32(1))!=u32(0):
        if schedule_rowv==mark1:
          schedule_fu=u32(meta_next[int(schedule_fu)])

      frame_action:u32=u32(0)
      frame_opcode:u32=u32(0)
      frame_raw:u32=u32(0)
      schedule_isbu:u32=(IS_BASE_MASK>>schedule_fu)&u32(1)
      if schedule_isbu!=u32(0) and schedule_rowv==endm:
        frame_action=u32(3) if schedule_fu==u32(14) else u32(2)
      else:
        schedule_ismu:u32=(IS_MARK_MASK>>schedule_fu)&u32(1)
        schedule_block_code:u32=u32(0)
        schedule_stepv:u32=u32(1)
        schedule_use_futureu:u32=u32(1)-schedule_ismu
        schedule_nextfidu:u32=schedule_fu

        if schedule_ismu!=u32(0):
          schedule_markv:u32=mark2 if ((SEL2_MASK>>schedule_fu)&u32(1))!=u32(0) else mark1
          if schedule_rowv==schedule_markv:
            schedule_block_code=(
              ((BLOCK_CODE_B0_MASK>>schedule_fu)&u32(1))
              |(((BLOCK_CODE_B1_MASK>>schedule_fu)&u32(1))<<u32(1))
              |(((BLOCK_CODE_B2_MASK>>schedule_fu)&u32(1))<<u32(2))
            )
            schedule_stepv=u32(2)+((OP_STEP3_MASK>>schedule_block_code)&u32(1))
            schedule_use_futureu=u32(0)
            schedule_nextfidu=u32(meta_next[int(schedule_fu)])

        schedule_isju:u32=(IS_JMARK_MASK>>schedule_fu)&u32(1)
        if schedule_isju!=u32(0):
          if schedule_rowv==jmark:
            frame_action=u32(1)
            schedule_nextfidu=u32(meta_next[int(schedule_fu)])

        schedule_child_rowu:u32=schedule_rowv+schedule_stepv
        schedule_fcvu:u32=u32(0)
        if schedule_use_futureu!=u32(0) and schedule_child_rowu<endm:
          schedule_fcvu=u32(1)
        frame_opcode=schedule_block_code|(schedule_fcvu<<u32(3))
        frame_raw=schedule_nextfidu|(schedule_child_rowu<<u32(5))

      if schedule_depth==0:
        root_action=frame_action
      else:
        parent_lane:int=(schedule_depth-1)&3
        parent_action_shift:u32=u32(parent_lane*8+4)
        pack_word|=frame_action<<parent_action_shift
        if (schedule_depth&3)==0:
          packed_schedule[pack_word_index]=pack_word
          pack_word_index+=1
          pack_word=u32(0)

      if frame_action>=u32(2):
        if schedule_depth>0 and (schedule_depth&3)!=0:
          packed_schedule[pack_word_index]=pack_word
        break

      opcode_shift:u32=u32((schedule_depth&3)*8)
      pack_word|=frame_opcode<<opcode_shift
      schedule_raw=frame_raw
      schedule_depth+=1

    if root_action==u32(2):
      results[i]=u64(w_lo_arr[i])
      return
    if root_action==u32(3):
      total+=u64(1) if ((root_a&~u32(1))!=u32(0)) else u64(0)
      results[i]=total*u64(w_lo_arr[i])
      return
    if root_action==u32(1):
      root_a&=~u32(1)
      if root_a==u32(0):
        results[i]=u64(0)
        return
      root_ld|=u32(1)

    sp:int=0
    ld[0]=root_ld
    rd[0]=root_rd
    col[0]=root_col
    avail[0]=root_a

    while True:
      a:u32=avail[sp]
      if a==u32(0):
        if sp==0:
          break
        sp-=1
        continue

      opcode_word:u32=packed_schedule[sp>>2]
      opcode:u32=(opcode_word>>u32((sp&3)*8))&u32(255)
      block_code:u32=opcode&u32(7)
      bit:u32=a&(u32(0)-a)
      avail[sp]=a^bit

      nld:u32=u32(0)
      nrd:u32=u32(0)
      if block_code!=u32(0):
        stepu:u32=u32(2)+((OP_STEP3_MASK>>block_code)&u32(1))
        addvu:u32=(OP_ADD1_MASK>>block_code)&u32(1)
        bLiu:u32=(
          ((OP_BL1_MASK>>block_code)&u32(1))
          |(((OP_BL2_MASK>>block_code)&u32(1))<<u32(1))
        )
        ktu:u32=(
          ((OP_KN3_MASK>>block_code)&u32(1))
          |(((OP_KN4_MASK>>block_code)&u32(1))<<u32(1))
        )
        bKu:u32=(n3&(u32(0)-(ktu&u32(1))))|(n4&(u32(0)-(ktu>>u32(1))))
        nld=((ld[sp]|bit)<<stepu)|addvu|bLiu
        nrd=((rd[sp]|bit)>>stepu)|bKu
      else:
        nld=(ld[sp]|bit)<<u32(1)
        nrd=(rd[sp]|bit)>>u32(1)
      ncol:u32=col[sp]|bit
      nf:u32=bm&~(nld|nrd|ncol)
      if nf==u32(0):
        continue
      if (opcode&u32(8))!=u32(0):
        if (bm&~((nld<<u32(1))|(nrd>>u32(1))|ncol))==u32(0):
          continue

      child_action:u32=(opcode>>u32(4))&u32(3)
      if child_action>=u32(2):
        if child_action==u32(2):
          total+=u64(1)
        else:
          total+=u64(1) if ((nf&~u32(1))!=u32(0)) else u64(0)
        continue
      if child_action==u32(1):
        nf&=~u32(1)
        if nf==u32(0):
          continue
        nld|=u32(1)

      sp+=1
      ld[sp]=nld
      rd[sp]=nrd
      col[sp]=ncol
      avail[sp]=nf
    results[i]=total*u64(w_lo_arr[i])

@gpu.kernel
def kernel_dfs_iter_gpu_maxd20(
    ld_arr:Ptr[u32],rd_arr:Ptr[u32],col_arr:Ptr[u32],ctrl0_arr:Ptr[u32],free_arr:Ptr[u32],
    markctrl_arr:Ptr[u32],w_lo_arr:Ptr[u32],
    meta_next:Ptr[u8],
    results:Ptr[u64],
    m:int,board_mask:u32,
    n3:u32,n4:u32,
)->None:
    IS_BASE_MASK:u32=u32(69222408)
    IS_JMARK_MASK:u32=u32(4)
    IS_MARK_MASK:u32=u32(199209203)
    IS_P5_MASK:u32=u32(3840)
    SEL2_MASK:u32=u32(34742338)

    BLOCK_CODE_B0_MASK:u32=u32(173707345)
    BLOCK_CODE_B1_MASK:u32=u32(12689458)
    BLOCK_CODE_B2_MASK:u32=u32(18088064)

    OP_STEP3_MASK:u32=u32(24)  # codes 3,4
    OP_ADD1_MASK:u32=u32(32)   # code 5
    OP_BL1_MASK:u32=u32(12)    # codes 2,3
    OP_BL2_MASK:u32=u32(16)    # code 4
    OP_KN3_MASK:u32=u32(18)    # codes 1,4
    OP_KN4_MASK:u32=u32(8)     # code 3

    ld=__array__[u32](MAXD20)
    rd=__array__[u32](MAXD20)
    col=__array__[u32](MAXD20)
    avail=__array__[u32](MAXD20)
    packed_schedule=__array__[u32](SCHED_WORDS20)
    bm:u32=board_mask
    i:int=(gpu.block.x*gpu.block.dim.x)+gpu.thread.x
    if i>=m:return

    markctrl:u32=markctrl_arr[i]
    jmark:u32=markctrl&u32(31)
    endm:u32=(markctrl>>u32(5))&u32(31)
    mark1:u32=(markctrl>>u32(10))&u32(31)
    mark2:u32=(markctrl>>u32(15))&u32(31)
    total:u64=u64(0)

    root_ld:u32=ld_arr[i]
    root_rd:u32=rd_arr[i]
    root_col:u32=col_arr[i]
    root_a:u32=free_arr[i]&bm
    if root_a==u32(0):
      results[i]=u64(0)
      return

    schedule_raw:u32=ctrl0_arr[i]
    schedule_depth:int=0
    pack_word:u32=u32(0)
    pack_word_index:int=0
    root_action:u32=u32(0)
    while True:
      schedule_fu:u32=schedule_raw&u32(31)
      schedule_rowv:u32=(schedule_raw>>u32(5))&u32(31)

      if ((IS_P5_MASK>>schedule_fu)&u32(1))!=u32(0):
        if schedule_rowv==mark1:
          schedule_fu=u32(meta_next[int(schedule_fu)])

      frame_action:u32=u32(0)
      frame_opcode:u32=u32(0)
      frame_raw:u32=u32(0)
      schedule_isbu:u32=(IS_BASE_MASK>>schedule_fu)&u32(1)
      if schedule_isbu!=u32(0) and schedule_rowv==endm:
        frame_action=u32(3) if schedule_fu==u32(14) else u32(2)
      else:
        schedule_ismu:u32=(IS_MARK_MASK>>schedule_fu)&u32(1)
        schedule_block_code:u32=u32(0)
        schedule_stepv:u32=u32(1)
        schedule_use_futureu:u32=u32(1)-schedule_ismu
        schedule_nextfidu:u32=schedule_fu

        if schedule_ismu!=u32(0):
          schedule_markv:u32=mark2 if ((SEL2_MASK>>schedule_fu)&u32(1))!=u32(0) else mark1
          if schedule_rowv==schedule_markv:
            schedule_block_code=(
              ((BLOCK_CODE_B0_MASK>>schedule_fu)&u32(1))
              |(((BLOCK_CODE_B1_MASK>>schedule_fu)&u32(1))<<u32(1))
              |(((BLOCK_CODE_B2_MASK>>schedule_fu)&u32(1))<<u32(2))
            )
            schedule_stepv=u32(2)+((OP_STEP3_MASK>>schedule_block_code)&u32(1))
            schedule_use_futureu=u32(0)
            schedule_nextfidu=u32(meta_next[int(schedule_fu)])

        schedule_isju:u32=(IS_JMARK_MASK>>schedule_fu)&u32(1)
        if schedule_isju!=u32(0):
          if schedule_rowv==jmark:
            frame_action=u32(1)
            schedule_nextfidu=u32(meta_next[int(schedule_fu)])

        schedule_child_rowu:u32=schedule_rowv+schedule_stepv
        schedule_fcvu:u32=u32(0)
        if schedule_use_futureu!=u32(0) and schedule_child_rowu<endm:
          schedule_fcvu=u32(1)
        frame_opcode=schedule_block_code|(schedule_fcvu<<u32(3))
        frame_raw=schedule_nextfidu|(schedule_child_rowu<<u32(5))

      if schedule_depth==0:
        root_action=frame_action
      else:
        parent_lane:int=(schedule_depth-1)&3
        parent_action_shift:u32=u32(parent_lane*8+4)
        pack_word|=frame_action<<parent_action_shift
        if (schedule_depth&3)==0:
          packed_schedule[pack_word_index]=pack_word
          pack_word_index+=1
          pack_word=u32(0)

      if frame_action>=u32(2):
        if schedule_depth>0 and (schedule_depth&3)!=0:
          packed_schedule[pack_word_index]=pack_word
        break

      opcode_shift:u32=u32((schedule_depth&3)*8)
      pack_word|=frame_opcode<<opcode_shift
      schedule_raw=frame_raw
      schedule_depth+=1

    if root_action==u32(2):
      results[i]=u64(w_lo_arr[i])
      return
    if root_action==u32(3):
      total+=u64(1) if ((root_a&~u32(1))!=u32(0)) else u64(0)
      results[i]=total*u64(w_lo_arr[i])
      return
    if root_action==u32(1):
      root_a&=~u32(1)
      if root_a==u32(0):
        results[i]=u64(0)
        return
      root_ld|=u32(1)

    sp:int=0
    ld[0]=root_ld
    rd[0]=root_rd
    col[0]=root_col
    avail[0]=root_a

    while True:
      a:u32=avail[sp]
      if a==u32(0):
        if sp==0:
          break
        sp-=1
        continue

      opcode_word:u32=packed_schedule[sp>>2]
      opcode:u32=(opcode_word>>u32((sp&3)*8))&u32(255)
      block_code:u32=opcode&u32(7)
      bit:u32=a&(u32(0)-a)
      avail[sp]=a^bit

      nld:u32=u32(0)
      nrd:u32=u32(0)
      if block_code!=u32(0):
        stepu:u32=u32(2)+((OP_STEP3_MASK>>block_code)&u32(1))
        addvu:u32=(OP_ADD1_MASK>>block_code)&u32(1)
        bLiu:u32=(
          ((OP_BL1_MASK>>block_code)&u32(1))
          |(((OP_BL2_MASK>>block_code)&u32(1))<<u32(1))
        )
        ktu:u32=(
          ((OP_KN3_MASK>>block_code)&u32(1))
          |(((OP_KN4_MASK>>block_code)&u32(1))<<u32(1))
        )
        bKu:u32=(n3&(u32(0)-(ktu&u32(1))))|(n4&(u32(0)-(ktu>>u32(1))))
        nld=((ld[sp]|bit)<<stepu)|addvu|bLiu
        nrd=((rd[sp]|bit)>>stepu)|bKu
      else:
        nld=(ld[sp]|bit)<<u32(1)
        nrd=(rd[sp]|bit)>>u32(1)
      ncol:u32=col[sp]|bit
      nf:u32=bm&~(nld|nrd|ncol)
      if nf==u32(0):
        continue
      if (opcode&u32(8))!=u32(0):
        if (bm&~((nld<<u32(1))|(nrd>>u32(1))|ncol))==u32(0):
          continue

      child_action:u32=(opcode>>u32(4))&u32(3)
      if child_action>=u32(2):
        if child_action==u32(2):
          total+=u64(1)
        else:
          total+=u64(1) if ((nf&~u32(1))!=u32(0)) else u64(0)
        continue
      if child_action==u32(1):
        nf&=~u32(1)
        if nf==u32(0):
          continue
        nld|=u32(1)

      sp+=1
      ld[sp]=nld
      rd[sp]=nrd
      col[sp]=ncol
      avail[sp]=nf
    results[i]=total*u64(w_lo_arr[i])

@gpu.kernel
def kernel_dfs_iter_gpu_maxd21(
    ld_arr:Ptr[u32],rd_arr:Ptr[u32],col_arr:Ptr[u32],ctrl0_arr:Ptr[u32],free_arr:Ptr[u32],
    markctrl_arr:Ptr[u32],w_lo_arr:Ptr[u32],
    meta_next:Ptr[u8],
    results:Ptr[u64],
    m:int,board_mask:u32,
    n3:u32,n4:u32,
)->None:
    IS_BASE_MASK:u32=u32(69222408)
    IS_JMARK_MASK:u32=u32(4)
    IS_MARK_MASK:u32=u32(199209203)
    IS_P5_MASK:u32=u32(3840)
    SEL2_MASK:u32=u32(34742338)

    BLOCK_CODE_B0_MASK:u32=u32(173707345)
    BLOCK_CODE_B1_MASK:u32=u32(12689458)
    BLOCK_CODE_B2_MASK:u32=u32(18088064)

    OP_STEP3_MASK:u32=u32(24)  # codes 3,4
    OP_ADD1_MASK:u32=u32(32)   # code 5
    OP_BL1_MASK:u32=u32(12)    # codes 2,3
    OP_BL2_MASK:u32=u32(16)    # code 4
    OP_KN3_MASK:u32=u32(18)    # codes 1,4
    OP_KN4_MASK:u32=u32(8)     # code 3

    ld=__array__[u32](MAXD21)
    rd=__array__[u32](MAXD21)
    col=__array__[u32](MAXD21)
    avail=__array__[u32](MAXD21)
    packed_schedule=__array__[u32](SCHED_WORDS21)
    bm:u32=board_mask
    i:int=(gpu.block.x*gpu.block.dim.x)+gpu.thread.x
    if i>=m:return

    markctrl:u32=markctrl_arr[i]
    jmark:u32=markctrl&u32(31)
    endm:u32=(markctrl>>u32(5))&u32(31)
    mark1:u32=(markctrl>>u32(10))&u32(31)
    mark2:u32=(markctrl>>u32(15))&u32(31)
    total:u64=u64(0)

    root_ld:u32=ld_arr[i]
    root_rd:u32=rd_arr[i]
    root_col:u32=col_arr[i]
    root_a:u32=free_arr[i]&bm
    if root_a==u32(0):
      results[i]=u64(0)
      return

    schedule_raw:u32=ctrl0_arr[i]
    schedule_depth:int=0
    pack_word:u32=u32(0)
    pack_word_index:int=0
    root_action:u32=u32(0)
    while True:
      schedule_fu:u32=schedule_raw&u32(31)
      schedule_rowv:u32=(schedule_raw>>u32(5))&u32(31)

      if ((IS_P5_MASK>>schedule_fu)&u32(1))!=u32(0):
        if schedule_rowv==mark1:
          schedule_fu=u32(meta_next[int(schedule_fu)])

      frame_action:u32=u32(0)
      frame_opcode:u32=u32(0)
      frame_raw:u32=u32(0)
      schedule_isbu:u32=(IS_BASE_MASK>>schedule_fu)&u32(1)
      if schedule_isbu!=u32(0) and schedule_rowv==endm:
        frame_action=u32(3) if schedule_fu==u32(14) else u32(2)
      else:
        schedule_ismu:u32=(IS_MARK_MASK>>schedule_fu)&u32(1)
        schedule_block_code:u32=u32(0)
        schedule_stepv:u32=u32(1)
        schedule_use_futureu:u32=u32(1)-schedule_ismu
        schedule_nextfidu:u32=schedule_fu

        if schedule_ismu!=u32(0):
          schedule_markv:u32=mark2 if ((SEL2_MASK>>schedule_fu)&u32(1))!=u32(0) else mark1
          if schedule_rowv==schedule_markv:
            schedule_block_code=(
              ((BLOCK_CODE_B0_MASK>>schedule_fu)&u32(1))
              |(((BLOCK_CODE_B1_MASK>>schedule_fu)&u32(1))<<u32(1))
              |(((BLOCK_CODE_B2_MASK>>schedule_fu)&u32(1))<<u32(2))
            )
            schedule_stepv=u32(2)+((OP_STEP3_MASK>>schedule_block_code)&u32(1))
            schedule_use_futureu=u32(0)
            schedule_nextfidu=u32(meta_next[int(schedule_fu)])

        schedule_isju:u32=(IS_JMARK_MASK>>schedule_fu)&u32(1)
        if schedule_isju!=u32(0):
          if schedule_rowv==jmark:
            frame_action=u32(1)
            schedule_nextfidu=u32(meta_next[int(schedule_fu)])

        schedule_child_rowu:u32=schedule_rowv+schedule_stepv
        schedule_fcvu:u32=u32(0)
        if schedule_use_futureu!=u32(0) and schedule_child_rowu<endm:
          schedule_fcvu=u32(1)
        frame_opcode=schedule_block_code|(schedule_fcvu<<u32(3))
        frame_raw=schedule_nextfidu|(schedule_child_rowu<<u32(5))

      if schedule_depth==0:
        root_action=frame_action
      else:
        parent_lane:int=(schedule_depth-1)&3
        parent_action_shift:u32=u32(parent_lane*8+4)
        pack_word|=frame_action<<parent_action_shift
        if (schedule_depth&3)==0:
          packed_schedule[pack_word_index]=pack_word
          pack_word_index+=1
          pack_word=u32(0)

      if frame_action>=u32(2):
        if schedule_depth>0 and (schedule_depth&3)!=0:
          packed_schedule[pack_word_index]=pack_word
        break

      opcode_shift:u32=u32((schedule_depth&3)*8)
      pack_word|=frame_opcode<<opcode_shift
      schedule_raw=frame_raw
      schedule_depth+=1

    if root_action==u32(2):
      results[i]=u64(w_lo_arr[i])
      return
    if root_action==u32(3):
      total+=u64(1) if ((root_a&~u32(1))!=u32(0)) else u64(0)
      results[i]=total*u64(w_lo_arr[i])
      return
    if root_action==u32(1):
      root_a&=~u32(1)
      if root_a==u32(0):
        results[i]=u64(0)
        return
      root_ld|=u32(1)

    sp:int=0
    ld[0]=root_ld
    rd[0]=root_rd
    col[0]=root_col
    avail[0]=root_a

    while True:
      a:u32=avail[sp]
      if a==u32(0):
        if sp==0:
          break
        sp-=1
        continue

      opcode_word:u32=packed_schedule[sp>>2]
      opcode:u32=(opcode_word>>u32((sp&3)*8))&u32(255)
      block_code:u32=opcode&u32(7)
      bit:u32=a&(u32(0)-a)
      avail[sp]=a^bit

      nld:u32=u32(0)
      nrd:u32=u32(0)
      if block_code!=u32(0):
        stepu:u32=u32(2)+((OP_STEP3_MASK>>block_code)&u32(1))
        addvu:u32=(OP_ADD1_MASK>>block_code)&u32(1)
        bLiu:u32=(
          ((OP_BL1_MASK>>block_code)&u32(1))
          |(((OP_BL2_MASK>>block_code)&u32(1))<<u32(1))
        )
        ktu:u32=(
          ((OP_KN3_MASK>>block_code)&u32(1))
          |(((OP_KN4_MASK>>block_code)&u32(1))<<u32(1))
        )
        bKu:u32=(n3&(u32(0)-(ktu&u32(1))))|(n4&(u32(0)-(ktu>>u32(1))))
        nld=((ld[sp]|bit)<<stepu)|addvu|bLiu
        nrd=((rd[sp]|bit)>>stepu)|bKu
      else:
        nld=(ld[sp]|bit)<<u32(1)
        nrd=(rd[sp]|bit)>>u32(1)
      ncol:u32=col[sp]|bit
      nf:u32=bm&~(nld|nrd|ncol)
      if nf==u32(0):
        continue
      if (opcode&u32(8))!=u32(0):
        if (bm&~((nld<<u32(1))|(nrd>>u32(1))|ncol))==u32(0):
          continue

      child_action:u32=(opcode>>u32(4))&u32(3)
      if child_action>=u32(2):
        if child_action==u32(2):
          total+=u64(1)
        else:
          total+=u64(1) if ((nf&~u32(1))!=u32(0)) else u64(0)
        continue
      if child_action==u32(1):
        nf&=~u32(1)
        if nf==u32(0):
          continue
        nld|=u32(1)

      sp+=1
      ld[sp]=nld
      rd[sp]=nrd
      col[sp]=ncol
      avail[sp]=nf
    results[i]=total*u64(w_lo_arr[i])

def launch_kernel_dfs_iter_gpu_static_maxd(
  selected_maxd:int,
  soa:TaskSoA,
  w_arr:List[u64],
  meta_next:List[u8],
  results:List[u64],
  m:int,
  board_mask_gpu:u32,
  n3_gpu:u32,
  n4_gpu:u32,
  grid_size:int,
  block_size:int,
  stride:int=0
)->bool:
  # 328 introduced the SoA split: w_arr (host-side List[u64], built by
  # build_soa_for_range -- UNCHANGED) was split into two densely-packed u32
  # arrays right here, at the GPU dispatch boundary only. That took L2
  # Theoretical Sectors Global Excessive on the w read from about 25000 to
  # zero, verified on the real kernel by the 329 ncu SourceCounters re-run.
  #
  # 351 REMOVES THE HIGH HALF. symmetry() returns only u64(2), u64(4) or
  # u64(8), so every element of w_arr is at most 8 and the high array was
  # identically zero: the three loads of it in each kernel epilogue always
  # returned 0. The parameter is gone from all five kernel signatures and
  # each epilogue now reads a single u32 and widens it. Nothing about
  # w_arr's generation, build_soa_for_range, or the CPU verification path
  # changes -- only this GPU-facing view. The u64 multiply is untouched;
  # folding the exponent into the unused markctrl bits 20-21 so that the
  # multiply degenerates into a shift is 352, kept separate on purpose so
  # that a regression in either step can be attributed to that step.
  #
  # 351 INVARIANT GUARD. The correctness of the removal rests entirely on
  # the claim that symmetry() stays inside 32 bits. The N=21 oracle would
  # catch a violation only as a wrong total, and only for N=21, so check
  # the invariant directly instead of assuming it. One OR pass over w_arr
  # costs well under a millisecond per launch against a ~400s run, and it
  # converts a silent wrong answer into a loud abort. Elements past m are
  # still zero and cannot trip it.
  w_hi_or:u64=u64(0)
  for v in w_arr:
    w_hi_or|=v
  if (w_hi_or>>u64(32))!=u64(0):
    print(f"[whi-error] w_arr high half is nonzero (or_all {int(w_hi_or)}); 351 assumes symmetry() stays within 32 bits")
    return False
  w_lo_arr:List[u32]=[u32(v&u64(0xffffffff)) for v in w_arr]
  if selected_maxd==14:
    kbatch_stride:int=stride if stride>0 else (grid_size*block_size)
    kernel_dfs_iter_gpu_maxd14(
      gpu.raw(soa.ld_arr),gpu.raw(soa.rd_arr),gpu.raw(soa.col_arr),
      gpu.raw(soa.ctrl0_arr),gpu.raw(soa.free_arr),
      gpu.raw(soa.markctrl_arr),gpu.raw(w_lo_arr),gpu.raw(meta_next),gpu.raw(results),
      m,board_mask_gpu,n3_gpu,n4_gpu,kbatch_stride,grid=grid_size,block=block_size
    )
    return True
  if selected_maxd==16:
    kernel_dfs_iter_gpu_maxd16(
      gpu.raw(soa.ld_arr),gpu.raw(soa.rd_arr),gpu.raw(soa.col_arr),
      gpu.raw(soa.ctrl0_arr),gpu.raw(soa.free_arr),
      gpu.raw(soa.markctrl_arr),gpu.raw(w_lo_arr),gpu.raw(meta_next),gpu.raw(results),
      m,board_mask_gpu,n3_gpu,n4_gpu,grid=grid_size,block=block_size
    )
    return True
  if selected_maxd==18:
    kernel_dfs_iter_gpu_maxd18(
      gpu.raw(soa.ld_arr),gpu.raw(soa.rd_arr),gpu.raw(soa.col_arr),
      gpu.raw(soa.ctrl0_arr),gpu.raw(soa.free_arr),
      gpu.raw(soa.markctrl_arr),gpu.raw(w_lo_arr),gpu.raw(meta_next),gpu.raw(results),
      m,board_mask_gpu,n3_gpu,n4_gpu,grid=grid_size,block=block_size
    )
    return True
  if selected_maxd==20:
    kernel_dfs_iter_gpu_maxd20(
      gpu.raw(soa.ld_arr),gpu.raw(soa.rd_arr),gpu.raw(soa.col_arr),
      gpu.raw(soa.ctrl0_arr),gpu.raw(soa.free_arr),
      gpu.raw(soa.markctrl_arr),gpu.raw(w_lo_arr),gpu.raw(meta_next),gpu.raw(results),
      m,board_mask_gpu,n3_gpu,n4_gpu,grid=grid_size,block=block_size
    )
    return True
  if selected_maxd==21:
    kernel_dfs_iter_gpu_maxd21(
      gpu.raw(soa.ld_arr),gpu.raw(soa.rd_arr),gpu.raw(soa.col_arr),
      gpu.raw(soa.ctrl0_arr),gpu.raw(soa.free_arr),
      gpu.raw(soa.markctrl_arr),gpu.raw(w_lo_arr),gpu.raw(meta_next),gpu.raw(results),
      m,board_mask_gpu,n3_gpu,n4_gpu,grid=grid_size,block=block_size
    )
    return True
  return False

def dfs_iter(
  meta:List[Tuple[int,int,int]],blockK:List[int],blockL:List[int],board_mask:int,
  functionid:int,ld:int,rd:int,col:int,row:int,free:int,
  jmark:int,endmark:int,mark1:int,mark2:int
)->u64:
  total:u64=u64(0)

  stack:List[Tuple[int,int,int,int,int,int]]=[(functionid,ld,rd,col,row,free)]

  while stack:
    functionid,ld,rd,col,row,free=stack.pop()

    if not free:
      continue

    next_funcid,funcptn,avail_flag=meta[functionid]
    avail:int=free

    if funcptn==5 and row==endmark:
      if functionid==14:
        total+=u64(1) if (avail>>1) else u64(0)
      else:
        total+=u64(1)
      continue

    step:int=1
    add1:int=0
    row_step:int=row+1

    use_blocks:bool=False
    use_future:bool=(avail_flag==1)

    local_next_funcid:int=functionid

    _blockK:int=0
    _blockL:int=0

    if funcptn in (0,1,2):
      at_mark:bool=(row==mark1) if funcptn in (0,2) else (row==mark2)

      if at_mark and avail:
        step=2 if funcptn in (0,1) else 3
        add1=1 if (funcptn==1 and functionid==20) else 0
        row_step=row+step

        _blockK=blockK[functionid]
        _blockL=blockL[functionid]

        use_blocks=True
        use_future=False
        local_next_funcid=next_funcid

    elif funcptn==3 and row==jmark:
      avail&=~1

      ld|=1

      local_next_funcid=next_funcid

      if not avail:
        continue

    elif funcptn==4 and row==mark1:
      stack.append((next_funcid,ld,rd,col,row,avail))
      continue

    if use_blocks:
      while avail:
        bit:int=avail&-avail
        avail&=avail-1

        nld:int=((ld|bit)<<step)|add1|_blockL
        nrd:int=((rd|bit)>>step)|_blockK
        ncol:int=col|bit

        nf:int=board_mask&~(nld|nrd|ncol)

        if nf:
          stack.append((local_next_funcid,nld,nrd,ncol,row_step,nf))

      continue

    if not use_future:
      while avail:
        bit:int=avail&-avail
        avail&=avail-1

        nld:int=(ld|bit)<<1
        nrd:int=(rd|bit)>>1
        ncol:int=col|bit

        nf:int=board_mask&~(nld|nrd|ncol)

        if nf:
          stack.append((local_next_funcid,nld,nrd,ncol,row_step,nf))

      continue

    if row_step>=endmark:
      while avail:
        bit:int=avail&-avail
        avail&=avail-1

        nld:int=(ld|bit)<<1
        nrd:int=(rd|bit)>>1
        ncol:int=col|bit

        nf:int=board_mask&~(nld|nrd|ncol)

        if nf:
          stack.append((local_next_funcid,nld,nrd,ncol,row_step,nf))

      continue

    while avail:
      bit:int=avail&-avail
      avail&=avail-1

      nld:int=(ld|bit)<<1
      nrd:int=(rd|bit)>>1
      ncol:int=col|bit

      nf:int=board_mask&~(nld|nrd|ncol)

      if not nf:
        continue

      if board_mask&~((nld<<1)|(nrd>>1)|ncol):
        stack.append((local_next_funcid,nld,nrd,ncol,row_step,nf))

  return total

# 241: recursive CPU dfs fallback remains removed. CPU path uses dfs_iter only.

def build_soa_for_range(
    N:int,
    constellations:List[Dict[str,int]],
    off:int,
    m:int,
    soa:TaskSoA,
    w_arr:List[u64]
)->Tuple[TaskSoA,List[u64]]:
    board_mask:int=(1<<N)-1

    small_mask:int=(1<<max(0,N-2))-1

    N1:int=N-1
    N2:int=N-2

    for t in range(m):
        constellation:Dict[str,int]=constellations[off+t]

        jmark:int=0
        mark1:int=0
        mark2:int=0

        start_ijkl:int=constellation["startijkl"]
        start:int=start_ijkl>>20
        ijkl:int=start_ijkl&((1<<20)-1)

        j,k,l=getj(ijkl),getk(ijkl),getl(ijkl)

        ld:int=constellation["ld"]>>1
        rd:int=constellation["rd"]>>1

        col:int=(constellation["col"]>>1)|(~small_mask)

        col&=board_mask

        LD:int=(1<<(N1-j))|(1<<(N1-l))

        ld|=LD>>(N-start)

        if start>k:
            rd|=(1<<(N1-(start-k+1)))

        if j>=2*N-33-start:
            rd|=(1<<(N1-j))<<(N2-start)

        free:int=board_mask&~(ld|rd|col)

        endmark:int=0
        target:int=0

        j_lt_N3:bool=(j<N-3)
        j_eq_N3:bool=(j==N-3)
        j_eq_N2:bool=(j==N-2)

        k_lt_l:bool=(k<l)
        start_lt_k:bool=(start<k)
        start_lt_l:bool=(start<l)

        l_eq_kp1:bool=(l==k+1)
        k_eq_lp1:bool=(k==l+1)

        j_gate:bool=(j>2*N-34-start)

        if j_lt_N3:
            jmark=j+1

            endmark=N2

            if j_gate:
                if k_lt_l:
                    mark1,mark2=k-1,l-1

                    if start_lt_l:
                        if start_lt_k:
                            target:int=0 if (not l_eq_kp1) else 4
                        else:
                            target=1
                    else:
                        target=2
                else:
                    mark1,mark2=l-1,k-1

                    if start_lt_k:
                        if start_lt_l:
                            target=5 if (not k_eq_lp1) else 7
                        else:
                            target=6
                    else:
                        target=2
            else:
                if k_lt_l:
                    mark1,mark2=k-1,l-1
                    target=8 if (not l_eq_kp1) else 9
                else:
                    mark1,mark2=l-1,k-1
                    target=10 if (not k_eq_lp1) else 11

        elif j_eq_N3:
            endmark=N2

            if k_lt_l:
                mark1,mark2=k-1,l-1

                if start_lt_l:
                    if start_lt_k:
                        target=12 if (not l_eq_kp1) else 15
                    else:
                        mark2=l-1
                        target=13
                else:
                    target=14
            else:
                mark1,mark2=l-1,k-1

                if start_lt_k:
                    if start_lt_l:
                        target=16 if (not k_eq_lp1) else 18
                    else:
                        mark2=k-1
                        target=17
                else:
                    target=14

        elif j_eq_N2:
            if k_lt_l:
                endmark=N2
                if start_lt_l:
                    if start_lt_k:
                        mark1=k-1
                        if not l_eq_kp1:
                            mark2=l-1
                            target=19
                        else:
                            target=22
                    else:
                        mark2=l-1
                        target=20
                else:
                    target=21
            else:
                if start_lt_k:
                    if start_lt_l:
                        if k<N2:
                            mark1,endmark=l-1,N2
                            if not k_eq_lp1:
                                mark2=k-1
                                target=23
                            else:
                                target=24
                        else:
                            if l!=(N-3):
                                mark2,endmark=l-1,N-3
                                target=20
                            else:
                                endmark=N-4
                                target=21
                    else:
                        if k!=N2:
                            mark2,endmark=k-1,N2
                            target=25
                        else:
                            endmark=N-3
                            target=21
                else:
                    endmark=N2
                    target=21

        else:
            endmark=N2
            if start>k:
                target=26
            else:
                mark1=k-1
                target=27

        soa.ld_arr[t]=u32(ld)
        soa.rd_arr[t]=u32(rd)
        soa.col_arr[t]=u32(col)
        soa.row_arr[t]=start
        soa.ctrl0_arr[t]=u32(target)|(u32(start)<<u32(5))
        soa.free_arr[t]=u32(free)
        soa.markctrl_arr[t]=(
          u32(jmark&31)
          |(u32(endmark&31)<<u32(5))
          |(u32(mark1&31)<<u32(10))
          |(u32(mark2&31)<<u32(15))
        )
        soa.jmark_arr[t]=jmark
        soa.end_arr[t]=endmark
        soa.mark1_arr[t]=mark1
        soa.mark2_arr[t]=mark2
        soa.funcid_arr[t]=target
        soa.ijkl_arr[t]=ijkl

    @par
    for t in range(m):
        w_arr[t]=symmetry(soa.ijkl_arr[t],N)

    return soa,w_arr

def auto_sort_mode(N:int)->int:
  if N==20 or N==21:
    return 9
  return 0

def validate_chunk_range(label:str,start:int,end:int,total:int)->bool:
  ok:bool=True
  if start<0:
    print(f"[cross-stripe-safe][error] {label}: start < 0 start={start} total={total}")
    ok=False
  if end<start:
    print(f"[cross-stripe-safe][error] {label}: end < start start={start} end={end} total={total}")
    ok=False
  if end>total:
    print(f"[cross-stripe-safe][error] {label}: end > total start={start} end={end} total={total}")
    ok=False
  if ok and start==end:
    print(f"[cross-stripe-safe][warn] {label}: empty range start={start} end={end} total={total}")
  return ok

def validate_reordered_count(label:str,expected:int,actual:int)->bool:
  if expected!=actual:
    print(f"[stripe-reorder][error] {label}: reordered count mismatch expected={expected} actual={actual}")
    return False
  return True

def validate_reordered_indices(label:str,expected:int,idxs:List[int])->bool:
  if not validate_reordered_count(label,expected,len(idxs)):
    return False
  seen:List[int]=[0]*expected
  for v in idxs:
    if v<0 or v>=expected:
      print(f"[cross-stripe-safe][error] {label}: index out of range idx={v} expected={expected}")
      return False
    if seen[v]!=0:
      print(f"[cross-stripe-safe][error] {label}: duplicated index idx={v}")
      return False
    seen[v]=1
  missing:int=0
  first_missing:int=-1
  for i in range(expected):
    if seen[i]==0:
      missing+=1
      if first_missing<0:
        first_missing=i
  if missing!=0:
    print(f"[cross-stripe-safe][error] {label}: missing count={missing} first_missing={first_missing}")
    return False
  return True

def popcount_int(x:int)->int:
  c:int=0
  while x:
    x&=x-1
    c+=1
  return c

def exec_solutions(N:int,constellations:List[Dict[str,int]],use_gpu:bool,gpu_block:int=32,gpu_max_blocks:int=484,gpu_log_level:int=0,gpu_sort_mode:int=-1,cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT,reorder_only:bool=False,chunk_only:bool=False,debug_chunk_start:int=0,debug_chunk_count:int=1)->None:
  N1:int=N-1
  N2:int=N-2
  board_mask:int=(1<<N)-1

  if gpu_sort_mode < 0:
    gpu_sort_mode = auto_sort_mode(N)

  FUNC_CATEGORY={
    "SQBkBlBjrB":3,"SQBlkBjrB":3,"SQBkBjrB":3,
    "SQd2BkBlB":3,"SQd2BkB":3,"SQd2BlkB":3,
    "SQd1BkBlB":3,"SQd1BlkB":3,"SQd1BkB":3,"SQd0BkB":3,
    "SQBklBjrB":4,"SQd2BklB":4,"SQd1BklB":4,
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
  F=len(func_meta)
  funcptn_by_fid:List[int]=[0]*F
  for f,(nxt,ptn,aflag) in enumerate(func_meta):
      funcptn_by_fid[f]=ptn
  is_base=[0]*F   # ptn==5
  is_jmark=[0]*F   # ptn==3
  is_mark=[0]*F   # ptn in {0,1,2}

  mark_sel=[0]*F  # 0:none 1:mark1 2:mark2
  mark_step=[1]*F  # 1 or 2 or 3
  mark_add1=[0]*F  # 0/1
  for f,(nxt,ptn,aflag) in enumerate(func_meta):
      if ptn==5:
          is_base[f]=1
      elif ptn==3:
          is_jmark[f]=1
      elif ptn==0 or ptn==1 or ptn==2:
          is_mark[f]=1
          if ptn==1:
              mark_sel[f]=2
              mark_step[f]=2
              if f==20:
                  mark_add1[f]=1
          else:
              mark_sel[f]=1
              mark_step[f]=2 if ptn==0 else 3

  n3=1<<max(0,N-3)   # 念のため負シフト防止
  n4=1<<max(0,N-4)   # N3,N4とは違います
  m=len(constellations)
  BLOCK=gpu_block
  MAX_BLOCKS=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS=BLOCK*MAX_BLOCKS
  m_all=len(constellations)

  if use_gpu:
    stripe_chunks:bool=(gpu_sort_mode==5 or gpu_sort_mode==6 or gpu_sort_mode==7 or gpu_sort_mode==8 or gpu_sort_mode==9 or gpu_sort_mode==10)
    balanced_stripe:bool=(gpu_sort_mode==7 or gpu_sort_mode==8)
    cross_stripe:bool=(gpu_sort_mode==9 or gpu_sort_mode==10)
    local_sort_mode:int=gpu_sort_mode
    work_constellations:List[Dict[str,int]]=constellations
    if reorder_only and not stripe_chunks:
      print(f"[reorder-only] N={N} sort_mode={gpu_sort_mode} stripe=0 original={m_all} steps={STEPS} ok")
      if m_all>0:
        constellations[0]["solutions"]=0
      return
    if stripe_chunks:
      n_chunks_est:int=(m_all + STEPS - 1)//STEPS
      reordered:List[Dict[str,int]]=[]
      validate_reorder:bool=(cross_stripe_safe or reorder_only)
      reordered_idx:List[int]=[]
      if cross_stripe:
        out_ch:int=0
        while out_ch<n_chunks_est:
          base:int=0
          while base*n_chunks_est<STEPS:
            src_ch:int=0
            while src_ch<n_chunks_est:
              rem:int=(src_ch + out_ch) % n_chunks_est
              within:int=base*n_chunks_est + rem
              if within<STEPS:
                idx:int=src_ch*STEPS+within
                if idx<m_all:
                  reordered.append(constellations[idx])
                  if validate_reorder:
                    reordered_idx.append(idx)
              src_ch+=1
            base+=1
          out_ch+=1
      elif balanced_stripe:
        out_ch:int=0
        while out_ch<n_chunks_est:
          slot:int=0
          while slot<STEPS:
            src_ch:int=slot % n_chunks_est
            base:int=slot // n_chunks_est
            within:int=(out_ch + base*n_chunks_est) % STEPS
            idx:int=src_ch*STEPS+within
            if idx<m_all:
              reordered.append(constellations[idx])
              if validate_reorder:
                reordered_idx.append(idx)
            slot+=1
          out_ch+=1
      else:
        within:int=0
        while within<STEPS:
          ch:int=0
          while ch<n_chunks_est:
            idx:int=ch*STEPS+within
            if idx<m_all:
              reordered.append(constellations[idx])
              if validate_reorder:
                reordered_idx.append(idx)
            ch+=1
          within+=1
      work_constellations=reordered
      if not validate_reordered_count("stripe_reorder",m_all,len(work_constellations)):
        return
      if validate_reorder:
        if not validate_reordered_indices("stripe_reorder",m_all,reordered_idx):
          return
      if reorder_only:
        print(f"[reorder-only] N={N} sort_mode={gpu_sort_mode} original={m_all} reordered={len(work_constellations)} chunks={n_chunks_est} steps={STEPS} ok")
        if m_all>0:
          constellations[0]["solutions"]=0
        return
      if gpu_sort_mode==5 or gpu_sort_mode==8 or gpu_sort_mode==10:
        local_sort_mode=4
      else:
        local_sort_mode=0
    if gpu_log_level>=1:
      print(f"[gpu-config] N={N} original=1 mixed32=1 hotpath=1 trunk75={BLOCK}x{MAX_BLOCKS} sort_mode={gpu_sort_mode} local_sort={local_sort_mode} stripe={1 if stripe_chunks else 0} balanced={1 if balanced_stripe else 0} cross={1 if cross_stripe else 0} cross_safe={1 if cross_stripe_safe else 0} reorder_only={1 if reorder_only else 0} chunk_only={1 if chunk_only else 0} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} log_level={gpu_log_level}")
    soa:TaskSoA=TaskSoA(STEPS)
    results:List[u64]=[u64(0)]*STEPS
    gpu_total:int=0
    w_arr:List[u64]=[u64(0)]*STEPS

    sort_soa:TaskSoA=TaskSoA(STEPS)
    sort_w_arr:List[u64]=[u64(0)]*STEPS
    order:List[int]=[0]*STEPS

    meta_next: List[u8] = [ u8(1),u8(2),u8(3),u8(3),u8(2),u8(6),u8(2),u8(2), u8(0),u8(4),u8(5),u8(7),u8(13),u8(14),u8(14),u8(14), u8(17),u8(14),u8(14),u8(20),u8(21),u8(21),u8(21),u8(25), u8(21),u8(21),u8(26),u8(26) ]
    off = 0
    board_mask_gpu:u32=u32(board_mask)
    n3_gpu:u32=u32(1)<<u32(N-3)
    n4_gpu:u32=u32(1)<<u32(N-4)
    chunks:int=0
    executed_chunks:int=0
    if chunk_only:
      if debug_chunk_start<0:
        debug_chunk_start=0
      if debug_chunk_count<=0:
        debug_chunk_count=1
      if gpu_log_level>=1:
        print(f"[chunk-only] mode=7 executes selected chunks only; start={debug_chunk_start} count={debug_chunk_count}")
    while off < m_all:
      m = min(STEPS, m_all - off)
      chunk_index:int=chunks
      chunks+=1
      if chunk_only:
        run_this_chunk:bool=(chunk_index>=debug_chunk_start and chunk_index<debug_chunk_start+debug_chunk_count)
        if not run_this_chunk:
          if gpu_log_level>=2:
            print(f"[gpu-chunk-skip] N={N} chunk={chunk_index} off={off} m={m}")
          off += m
          continue
      executed_chunks+=1
      if cross_stripe_safe:
        if not validate_chunk_range("gpu_chunk",off,off+m,m_all):
          return
      if gpu_log_level>=2:
        t0=datetime.now()
      build_soa_for_range(N,work_constellations, off, m,soa,w_arr)
      if gpu_log_level>=2:
        t1=datetime.now()
      use_sorted:bool=(local_sort_mode==1 or local_sort_mode==2 or local_sort_mode==3 or local_sort_mode==4)
      if gpu_log_level>=2:
        ts0=datetime.now()
      if use_sorted:
        nb:int=28
        if local_sort_mode==2:
          nb=6
        if local_sort_mode==3:
          nb=24  # funcptn 6種 x work bucket 4種
        if local_sort_mode==4:
          nb=48  # funcptn 6種 x work bucket 8種
        counts:List[int]=[0]*64
        pos:List[int]=[0]*64
        cur:List[int]=[0]*64
        for i in range(m):
          fid0:int=soa.funcid_arr[i]
          key:int=fid0
          if local_sort_mode==2:
            key=funcptn_by_fid[fid0]
          if local_sort_mode==3:
            ptn:int=funcptn_by_fid[fid0]
            depth:int=soa.end_arr[i]-soa.row_arr[i]
            if depth<0:
              depth=0
            pc:int=popcount_int(int(soa.free_arr[i]))
            wb:int=0
            if pc>=3:
              wb+=1
            if depth>=12:
              wb+=2
            key=ptn*4+wb
          if local_sort_mode==4:
            ptn:int=funcptn_by_fid[fid0]
            depth:int=soa.end_arr[i]-soa.row_arr[i]
            if depth<0:
              depth=0
            pc:int=popcount_int(int(soa.free_arr[i]))
            wb:int=0
            if pc>=2:
              wb+=1
            if pc>=4:
              wb+=2
            if depth>=10:
              wb+=4
            key=ptn*8+wb
          counts[key]+=1
        run:int=0
        for b in range(nb):
          pos[b]=run
          cur[b]=run
          run+=counts[b]
        for i in range(m):
          fid0:int=soa.funcid_arr[i]
          key:int=fid0
          if local_sort_mode==2:
            key=funcptn_by_fid[fid0]
          if local_sort_mode==3:
            ptn:int=funcptn_by_fid[fid0]
            depth:int=soa.end_arr[i]-soa.row_arr[i]
            if depth<0:
              depth=0
            pc:int=popcount_int(int(soa.free_arr[i]))
            wb:int=0
            if pc>=3:
              wb+=1
            if depth>=12:
              wb+=2
            key=ptn*4+wb
          if local_sort_mode==4:
            ptn:int=funcptn_by_fid[fid0]
            depth:int=soa.end_arr[i]-soa.row_arr[i]
            if depth<0:
              depth=0
            pc:int=popcount_int(int(soa.free_arr[i]))
            wb:int=0
            if pc>=2:
              wb+=1
            if pc>=4:
              wb+=2
            if depth>=10:
              wb+=4
            key=ptn*8+wb
          p:int=cur[key]
          order[p]=i
          cur[key]+=1
        for p in range(m):
          q:int=order[p]
          sort_soa.ld_arr[p]=soa.ld_arr[q]
          sort_soa.rd_arr[p]=soa.rd_arr[q]
          sort_soa.col_arr[p]=soa.col_arr[q]
          sort_soa.row_arr[p]=soa.row_arr[q]
          sort_soa.ctrl0_arr[p]=soa.ctrl0_arr[q]
          sort_soa.free_arr[p]=soa.free_arr[q]
          sort_soa.markctrl_arr[p]=soa.markctrl_arr[q]
          sort_soa.funcid_arr[p]=soa.funcid_arr[q]
          sort_soa.ijkl_arr[p]=soa.ijkl_arr[q]
          sort_w_arr[p]=w_arr[q]
      if gpu_log_level>=2:
        ts1=datetime.now()
      GRID = (m + BLOCK - 1) // BLOCK

      if use_sorted:
        required_maxd:int=max_schedule_depth_of_tasks(sort_soa,m,meta_next)
        selected_maxd:int=select_static_maxd(required_maxd)
        if gpu_log_level>=2:
          print(f"[maxd-dispatch] N={N} scope=exec chunk={chunk_index} m={m} required_maxd={required_maxd} selected_MAXD={selected_maxd} schedule_words={packed_schedule_words_for_maxd(selected_maxd)} stack_bytes_per_thread={packed_stack_bytes_per_thread(selected_maxd)} capacity_check=OK")
        if not launch_kernel_dfs_iter_gpu_static_maxd(selected_maxd,sort_soa,sort_w_arr,meta_next,results,m,board_mask_gpu,n3_gpu,n4_gpu,GRID,BLOCK):
          print(f"[maxd-error] unsupported required_maxd={required_maxd}; supported maximum is 21")
          return
      else:
        required_maxd:int=max_schedule_depth_of_tasks(soa,m,meta_next)
        selected_maxd:int=select_static_maxd(required_maxd)
        if gpu_log_level>=2:
          print(f"[maxd-dispatch] N={N} scope=exec chunk={chunk_index} m={m} required_maxd={required_maxd} selected_MAXD={selected_maxd} schedule_words={packed_schedule_words_for_maxd(selected_maxd)} stack_bytes_per_thread={packed_stack_bytes_per_thread(selected_maxd)} capacity_check=OK")
        if not launch_kernel_dfs_iter_gpu_static_maxd(selected_maxd,soa,w_arr,meta_next,results,m,board_mask_gpu,n3_gpu,n4_gpu,GRID,BLOCK):
          print(f"[maxd-error] unsupported required_maxd={required_maxd}; supported maximum is 21")
          return

      if gpu_log_level>=2:
        t2=datetime.now()
      chunk_total:int=0
      for i in range(m):
        chunk_total += int(results[i])
      gpu_total += chunk_total
      if gpu_log_level>=2:
        t3=datetime.now()
        print(f"[gpu-chunk] N={N} chunk={chunk_index} off={off} m={m} grid={GRID} sort={gpu_sort_mode}/{local_sort_mode} build={str(t1-t0)[:-3]} sort_time={str(ts1-ts0)[:-3]} kernel={str(t2-ts1)[:-3]} sum={str(t3-t2)[:-3]} partial_total={chunk_total}")
      off += m

    if m_all>0:
      constellations[0]["solutions"] = gpu_total
    if gpu_log_level>=1:
      print(f"[gpu-summary] N={N} constellations={m_all} chunks={chunks} executed_chunks={executed_chunks} direct_total=1 stripe={1 if stripe_chunks else 0} balanced={1 if balanced_stripe else 0} cross={1 if cross_stripe else 0} cross_safe={1 if cross_stripe_safe else 0} chunk_only={1 if chunk_only else 0}")
    return
  else:
    soa:TaskSoA = TaskSoA(m_all)
    results: List[u64] = [u64(0)] * m_all
    w_arr: List[u64] = [u64(0)] * m_all

    size=max(FID.values())+1
    blockK_by_funcid=[0]*size
    blockL_by_funcid=[0,1,0,0,1,1,0,2,0,0,0,0,0,1,0,1,1,0,2,0,0,0,1,1,2,0,0,0]
    for fn,cat in FUNC_CATEGORY.items():# FUNC_CATEGORY: {関数名: 3 or 4 or 0}
      fid=FID[fn]
      blockK_by_funcid[fid]=n3 if cat==3 else (n4 if cat==4 else 0)

    m_all = len(constellations) # CPUは全件を1回で SoA + w_arr を作る（これがないと壊れる）
    if m_all == 0:
      return
    soa, w_arr = build_soa_for_range(N,constellations, 0, m_all, soa, w_arr)
    results:List[u64] = [u64(0)] * m_all
    @par
    for i in range(m_all):
      cnt:u64 = dfs_iter(
          func_meta,
          blockK_by_funcid,blockL_by_funcid,
          board_mask,
          soa.funcid_arr[i],
          int(soa.ld_arr[i]), int(soa.rd_arr[i]), int(soa.col_arr[i]),
          soa.row_arr[i],int(soa.free_arr[i]),
          soa.jmark_arr[i], soa.end_arr[i],
          soa.mark1_arr[i], soa.mark2_arr[i])
      results[i]=cnt*w_arr[i]
  out = results
  for i, constellation in enumerate(constellations):
    constellation["solutions"] = int(out[i])

def mix64(x:u64)->u64:
  x=(x^(x>>u64(30)))*u64(0xBF58476D1CE4E5B9)
  x=(x^(x>>u64(27)))*u64(0x94D049BB133111EB)
  x^=(x>>u64(31))
  return x

def gen_list(cnt:int,seed:u64)->List[u64]:
  out:List[u64]=[]
  s:u64=seed
  for _ in range(cnt):
    s=s+u64(0x9E3779B97F4A7C15)
    out.append(mix64(s))
  return out

def init_zobrist(N:int,zobrist_hash_tables: Dict[int, Dict[str, List[u64]]])->Dict[str,List[u64]]:
  if N in zobrist_hash_tables:
    return zobrist_hash_tables[N]
  base_seed:u64=(u64(0xC0D0_0000_0000_0000)^(u64(N)<<u64(32)))
  tbl:Dict[str,List[u64]]={
    'ld':gen_list(N,base_seed^u64(0x01)),
    'rd':gen_list(N,base_seed^u64(0x02)),
    'col':gen_list(N,base_seed^u64(0x03)),
    'LD':gen_list(N,base_seed^u64(0x04)),
    'RD':gen_list(N,base_seed^u64(0x05)),
    'row':gen_list(N,base_seed^u64(0x06)),
    'queens':gen_list(N,base_seed^u64(0x07)),
    'k':gen_list(N,base_seed^u64(0x08)),
    'l':gen_list(N,base_seed^u64(0x09)),
  }
  zobrist_hash_tables[N]=tbl
  return tbl

def zobrist_hash(N:int, ld: int, rd: int, col: int, row: int, queens: int, k: int, l: int, LD: int, RD: int,zobrist_hash_tables:Dict[int, Dict[str, List[u64]]]) -> u64:
  tbl: Dict[str, List[u64]] = init_zobrist(N,zobrist_hash_tables)

  ld_tbl  = tbl["ld"]    # List[u64]
  rd_tbl  = tbl["rd"]    # List[u64]
  col_tbl = tbl["col"]   # List[u64]
  LD_tbl  = tbl["LD"]    # List[u64]
  RD_tbl  = tbl["RD"]    # List[u64]
  row_tbl = tbl["row"]   # List[u64]
  q_tbl   = tbl["queens"]# List[u64]
  k_tbl   = tbl["k"]     # List[u64]
  l_tbl   = tbl["l"]     # List[u64]

  mask: u64 = (u64(1) << u64(N)) - u64(1)

  ld64: u64  = u64(ld)  & mask
  rd64: u64  = u64(rd)  & mask
  col64: u64 = u64(col) & mask
  LD64: u64  = u64(LD)  & mask
  RD64: u64  = u64(RD)  & mask

  h: u64 = u64(0)

  m: u64 = ld64
  i: int = 0
  while i < N:
    if (m & u64(1)) != u64(0):
      h ^= u64(ld_tbl[i])
    m >>= u64(1)
    i += 1

  m = rd64; i = 0
  while i < N:
    if (m & u64(1)) != u64(0):
      h ^= u64(rd_tbl[i])
    m >>= u64(1)
    i += 1

  m = col64; i = 0
  while i < N:
    if (m & u64(1)) != u64(0):
      h ^= u64(col_tbl[i])
    m >>= u64(1)
    i += 1

  m = LD64; i = 0
  while i < N:
    if (m & u64(1)) != u64(0):
      h ^= u64(LD_tbl[i])
    m >>= u64(1)
    i += 1

  m = RD64; i = 0
  while i < N:
    if (m & u64(1)) != u64(0):
      h ^= u64(RD_tbl[i])
    m >>= u64(1)
    i += 1

  if 0 <= row < N:
    h ^= u64(row_tbl[row])
  if 0 <= queens < N:
    h ^= u64(q_tbl[queens])
  if 0 <= k < N:
    h ^= u64(k_tbl[k])
  if 0 <= l < N:
    h ^= u64(l_tbl[l])

  return h

def to_ijkl(i:int,j:int,k:int,l:int)->int:return (i<<15)+(j<<10)+(k<<5)+l
def mirvert(ijkl:int,N:int)->int:return to_ijkl(N-1-geti(ijkl),N-1-getj(ijkl),getl(ijkl),getk(ijkl))
def ffmin(a:int,b:int)->int:return min(a,b)
def geti(ijkl:int)->int:return (ijkl>>15)&0x1F
def getj(ijkl:int)->int:return (ijkl>>10)&0x1F
def getk(ijkl:int)->int:return (ijkl>>5)&0x1F
def getl(ijkl:int)->int:return ijkl&0x1F

def rot90(ijkl:int,N:int)->int:return ((N-1-getk(ijkl))<<15)+((N-1-getl(ijkl))<<10)+(getj(ijkl)<<5)+geti(ijkl)
def symmetry(ijkl:int,N:int)->u64:return u64(2) if symmetry90(ijkl,N) else u64(4) if geti(ijkl)==N-1-getj(ijkl) and getk(ijkl)==N-1-getl(ijkl) else u64(8)
def symmetry90(ijkl:int,N:int)->bool:return ((geti(ijkl)<<15)+(getj(ijkl)<<10)+(getk(ijkl)<<5)+getl(ijkl))==(((N-1-getk(ijkl))<<15)+((N-1-getl(ijkl))<<10)+(getj(ijkl)<<5)+geti(ijkl))

def check_rotations(ijkl_list:Set[int],i:int,j:int,k:int,l:int,N:int)->bool:
  return any(rot in ijkl_list for rot in [((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i,((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+(N-1-k),(l<<15)+(k<<10)+((N-1-i)<<5)+(N-1-j)])

jasmin_cache_global:Dict[Tuple[int,int],int]={}

def get_jasmin(N:int,c:int)->int:
  key=(c,N)
  if key in jasmin_cache_global:
    return jasmin_cache_global[key]
  result=jasmin(c,N)
  jasmin_cache_global[key]=result
  return result

def jasmin(ijkl:int,N:int)->int:
  arg=0
  min_val=ffmin(getj(ijkl),N-1-getj(ijkl))
  if ffmin(geti(ijkl),N-1-geti(ijkl))<min_val:
    arg=2
    min_val=ffmin(geti(ijkl),N-1-geti(ijkl))
  if ffmin(getk(ijkl),N-1-getk(ijkl))<min_val:
    arg=3
    min_val=ffmin(getk(ijkl),N-1-getk(ijkl))
  if ffmin(getl(ijkl),N-1-getl(ijkl))<min_val:
    arg=1
    min_val=ffmin(getl(ijkl),N-1-getl(ijkl))
  _rot90=rot90
  for _ in range(arg):
    ijkl=_rot90(ijkl,N)
  if getj(ijkl)<N-1-getj(ijkl):
    ijkl=mirvert(ijkl,N)
  return ijkl

def set_pre_queens_cached(
  N:int,
  ijkl_list:Set[int],
  subconst_cache:set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],
  ld:int, rd:int, col:int,
  k:int, l:int,
  row:int, queens:int,
  LD:int, RD:int,
  counter:List[int],
  constellations:List[Dict[str,int]],
  preset_queens:int,
  visited:Set[int],
  constellation_signatures:Set[Tuple[int,int,int,int,int,int]],
  zobrist_hash_tables: Dict[int, Dict[str, List[u64]]]
)->Tuple[Set[int], Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]], List[Dict[str,int]], int]:
  if preset_queens>=7:
    ijkl_list, subconst_cache, constellations, preset_queens = set_pre_queens(
      N, ijkl_list, subconst_cache,
      ld, rd, col,
      k, l,
      row, queens,
      LD, RD,
      counter, constellations, preset_queens,
      visited, constellation_signatures,
      zobrist_hash_tables
    )
    return ijkl_list, subconst_cache, constellations, preset_queens

  key:Tuple[int,int,int,int,int,int,int,int,int,int,int] = (
    ld, rd, col, k, l, row, queens, LD, RD, N, preset_queens
  )

  if key in subconst_cache:
    return ijkl_list, subconst_cache, constellations, preset_queens

  subconst_cache.add(key)

  ijkl_list, subconst_cache, constellations, preset_queens = set_pre_queens(
    N, ijkl_list, subconst_cache,
    ld, rd, col,
    k, l,
    row, queens,
    LD, RD,
    counter, constellations, preset_queens,
    visited, constellation_signatures,
    zobrist_hash_tables
  )
  return ijkl_list, subconst_cache, constellations, preset_queens

use_visited_prune:bool=False
def set_pre_queens(N:int,ijkl_list:Set[int],subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],ld:int,rd:int,col:int,k:int,l:int,row:int,queens:int,LD:int,RD:int,counter:list,constellations:List[Dict[str,int]],preset_queens:int,visited:Set[int],constellation_signatures:Set[Tuple[int,int,int,int,int,int]],zobrist_hash_tables: Dict[int, Dict[str, List[u64]]])->Tuple[Set[int], Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]], List[Dict[str,int]], int]:
  board_mask= (1<<N)-1
  if use_visited_prune:
    h: int = int(zobrist_hash(N,ld & board_mask, rd & board_mask, col & board_mask, row, queens, k, l, LD & board_mask, RD & board_mask,zobrist_hash_tables))
    if h in visited:
      return ijkl_list, subconst_cache, constellations, preset_queens
    visited.add(h)

  if row==k or row==l:
    ijkl_list, subconst_cache, constellations, preset_queens = set_pre_queens_cached(N,ijkl_list,subconst_cache,ld<<1,rd>>1,col,k,l,row+1,queens,LD,RD,counter,constellations,preset_queens,visited,constellation_signatures,zobrist_hash_tables)
    return ijkl_list, subconst_cache, constellations, preset_queens
  if queens == preset_queens:
    if (not DISABLE_CONSTELLATION_SIGNATURE_PRUNE) and preset_queens <= 5:
      sig = (ld, rd, col, k, l, row)    # これが signature (tuple)
      if sig in constellation_signatures:
        return ijkl_list, subconst_cache, constellations, preset_queens
      constellation_signatures.add(sig)
    constellation={"ld":ld,"rd":rd,"col":col,"startijkl":row<<20,"solutions":0}
    constellations.append(constellation) #星座データ追加
    counter[0]+=1
    return ijkl_list, subconst_cache, constellations, preset_queens
  free=~(ld|rd|col|(LD>>(N-1-row))|(RD<<(N-1-row)))&board_mask
  while free:
    bit:int=free&-free
    free&=free-1
    ijkl_list, subconst_cache, constellations, preset_queens = set_pre_queens_cached(N,ijkl_list,subconst_cache,(ld|bit)<<1,(rd|bit)>>1,col|bit,k,l,row+1,queens+1,LD,RD,counter,constellations,preset_queens,visited,constellation_signatures,zobrist_hash_tables)

  return ijkl_list, subconst_cache, constellations, preset_queens

def gen_constellations(
  N:int,
  ijkl_list:Set[int],
  subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],
  constellations:List[Dict[str,int]],
  preset_queens:int
)->Tuple[
  Set[int],
  Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],
  List[Dict[str,int]],
  int
]:
  halfN = (N + 1) // 2        # N の半分（切り上げ）。開始星座生成の範囲を絞るために使う
  N1:int = N - 1              # 最終列 index
  N2:int = N - 2

  subconst_cache.clear()

  constellation_signatures: Set[Tuple[int,int,int,int,int,int]] = set()

  if N % 2 == 1:
    center = N // 2
    ijkl_list.update(
      to_ijkl(i, j, center, l)
      for l in range(center + 1, N1)
      for i in range(center + 1, N1)
      if i != (N1) - l
      for j in range(N - center - 2, 0, -1)
      if j != i and j != l
      if not check_rotations(ijkl_list, i, j, center, l, N)
    )

  ijkl_list.update(
    to_ijkl(i, j, k, l)
    for k in range(1, halfN)
    for l in range(k + 1, N1)
    for i in range(k + 1, N1)
    if i != (N1) - l
    for j in range(N - k - 2, 0, -1)
    if j != i and j != l
    if not check_rotations(ijkl_list, i, j, k, l, N)
  )

  ijkl_list.update({to_ijkl(0, j, 0, l) for j in range(1, N2) for l in range(j + 1, N1)})

  ijkl_list = {get_jasmin(N, c) for c in ijkl_list}

  L = 1 << (N1)

  for sc in ijkl_list:
    subconst_cache.clear()

    constellation_signatures = set()

    i, j, k, l = geti(sc), getj(sc), getk(sc), getl(sc)

    Lj = L >> j
    Li = L >> i
    Ll = L >> l

    ld = (((L >> (i - 1)) if i > 0 else 0) | (1 << (N - k)))
    rd = ((L >> (i + 1)) | (1 << (l - 1)))
    col = (1 | L | Li | Lj)

    LD = (Lj | Ll)
    RD = (Lj | (1 << k))

    counter: List[int] = [0]     # set_pre_queens 側が増やす
    visited: Set[int] = set()    # 枝刈り用 visited（hash を入れる設計）

    zobrist_hash_tables: Dict[int, Dict[str, List[u64]]] = {}

    ijkl_list, subconst_cache, constellations, preset_queens = set_pre_queens_cached(
      N, ijkl_list, subconst_cache,
      ld, rd, col,
      k, l,
      1,
      3 if j == N1 else 4,
      LD, RD,
      counter, constellations, preset_queens,
      visited, constellation_signatures,
      zobrist_hash_tables
    )

    base = to_ijkl(i, j, k, l)

    for a in range(counter[0]):
      constellations[-1 - a]["startijkl"] |= base

  return ijkl_list, subconst_cache, constellations, preset_queens

def read_uint32_le(b:str)->int:
  return (ord(b[0])&0xFF)|((ord(b[1])&0xFF)<<8)|((ord(b[2])&0xFF)<<16)|((ord(b[3])&0xFF)<<24)

def int_to_le_bytes(x:int)->List[int]:
  return [(x>>(8*i))&0xFF for i in range(4)]

def file_exists(fname:str)->bool:
  try:
    with open(fname,"rb"):
      return True
  except:
    return False

def validate_bin_file(fname:str)->bool:
  try:
    with open(fname,"rb") as f:
      f.seek(0,2)  # ファイル末尾に移動
      size=f.tell()
    return size%16==0
  except:
    return False

def count_constellations_bin_records(fname:str)->int:
  try:
    with open(fname,"rb") as f:
      f.seek(0,2)
      size:int=f.tell()
    if size%16!=0:
      return 0
    return size//16
  except:
    return 0

def read_stream_done_count(fname:str)->int:
  try:
    with open(fname,"r") as f:
      text:str=f.read().strip()
    if text=="":
      return -1
    return int(text)
  except:
    return -1

def write_stream_done_count(fname:str,count:int)->None:
  with open(fname,"w") as f:
    f.write(str(count))
    f.write("\n")

def truncate_constellations_bin(fname:str)->None:
  with open(fname,"wb") as f:
    pass
  write_stream_done_count(fname+".done",0)

def append_constellations_bin(fname:str,constellations:List[Dict[str,int]])->None:
  with open(fname,"ab") as f:
    for d in constellations:
      for key in ["ld","rd","col","startijkl"]:
        b=int_to_le_bytes(d[key])
        f.write("".join(chr(c) for c in b))

def gen_constellations_stream_to_bin(
  N:int,
  ijkl_list:Set[int],
  subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],
  fname:str,
  preset_queens:int,
  gpu_log_level:int=0
)->Tuple[Set[int],Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],int,int]:

  halfN:int=(N+1)//2
  N1:int=N-1
  N2:int=N-2
  subconst_cache.clear()

  constellation_signatures:Set[Tuple[int,int,int,int,int,int]]=set()

  if N%2==1:
    center:int=N//2
    ijkl_list.update(
      to_ijkl(i,j,center,l)
      for l in range(center+1,N1)
      for i in range(center+1,N1)
      if i!=(N1)-l
      for j in range(N-center-2,0,-1)
      if j!=i and j!=l
      if not check_rotations(ijkl_list,i,j,center,l,N)
    )

  ijkl_list.update(
    to_ijkl(i,j,k,l)
    for k in range(1,halfN)
    for l in range(k+1,N1)
    for i in range(k+1,N1)
    if i!=(N1)-l
    for j in range(N-k-2,0,-1)
    if j!=i and j!=l
    if not check_rotations(ijkl_list,i,j,k,l,N)
  )

  ijkl_list.update({to_ijkl(0,j,0,l) for j in range(1,N2) for l in range(j+1,N1)})
  ijkl_list={get_jasmin(N,c) for c in ijkl_list}

  L:int=1<<(N1)
  total_count:int=0
  sc_index:int=0
  truncate_constellations_bin(fname)

  for sc in ijkl_list:
    subconst_cache.clear()
    constellation_signatures=set()

    i,j,k,l=geti(sc),getj(sc),getk(sc),getl(sc)
    Lj:int=L>>j
    Li:int=L>>i
    Ll:int=L>>l

    ld:int=(((L>>(i-1)) if i>0 else 0)|(1<<(N-k)))
    rd:int=((L>>(i+1))|(1<<(l-1)))
    col:int=(1|L|Li|Lj)
    LD:int=(Lj|Ll)
    RD:int=(Lj|(1<<k))

    counter:List[int]=[0]
    visited:Set[int]=set()
    zobrist_hash_tables:Dict[int,Dict[str,List[u64]]]={}
    sc_constellations:List[Dict[str,int]]=[]

    ijkl_list,subconst_cache,sc_constellations,preset_queens=set_pre_queens_cached(
      N,ijkl_list,subconst_cache,
      ld,rd,col,
      k,l,
      1,
      3 if j==N1 else 4,
      LD,RD,
      counter,sc_constellations,preset_queens,
      visited,constellation_signatures,
      zobrist_hash_tables
    )

    base:int=to_ijkl(i,j,k,l)
    for a in range(counter[0]):
      sc_constellations[-1-a]["startijkl"]|=base

    if counter[0]>0:
      append_constellations_bin(fname,sc_constellations)
      total_count+=counter[0]

    if gpu_log_level>=2:
      print(f"[stream-build-sc] N={N} sc_index={sc_index} added={counter[0]} total={total_count}")

    sc_index+=1

  write_stream_done_count(fname+".done",total_count)
  if gpu_log_level>=1:
    print(f"[stream-build-summary] N={N} preset_queens={preset_queens} sc={sc_index} records={total_count} bin={fname}")

  return ijkl_list,subconst_cache,total_count,preset_queens

def ensure_constellations_bin_stream(N:int,ijkl_list:Set[int],subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],preset_queens:int,gpu_log_level:int=0)->Tuple[Set[int],Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]],int,int,str]:
  fname:str=f"constellations_N{N}_{preset_queens}.bin"
  records:int=count_constellations_bin_records(fname)
  done_count:int=read_stream_done_count(fname+".done")
  if records>0 and done_count==records:
    if gpu_log_level>=1:
      print(f"[stream-cache-hit] N={N} preset_queens={preset_queens} records={records} bin={fname}")
    return ijkl_list,subconst_cache,records,preset_queens,fname

  if file_exists(fname):
    print(f"[stream-cache-warning] invalid/incomplete bin cache: {fname}; records={records} done={done_count}; rebuilding")
  else:
    if gpu_log_level>=1:
      print(f"[stream-cache-miss] N={N} preset_queens={preset_queens} bin={fname}; building")

  ijkl_list,subconst_cache,records,preset_queens=gen_constellations_stream_to_bin(N,ijkl_list,subconst_cache,fname,preset_queens,gpu_log_level)
  return ijkl_list,subconst_cache,records,preset_queens,fname

def stream_elapsed_text_to_ms(elapsed_text:str)->int:
  s:str=elapsed_text
  days:int=0
  day_parts=s.split(",")
  if len(day_parts)>1:
    day_tokens=day_parts[0].strip().split()
    if len(day_tokens)>0:
      days=int(day_tokens[0])
    s=day_parts[1].strip()

  hms=s.split(":")
  if len(hms)<3:
    return 0
  hours:int=int(hms[0])
  minutes:int=int(hms[1])
  sec_ms=hms[2].split(".")
  seconds:int=int(sec_ms[0])
  millis:int=0
  if len(sec_ms)>1:
    ms_str:str=sec_ms[1]
    if len(ms_str)==1:
      millis=int(ms_str)*100
    elif len(ms_str)==2:
      millis=int(ms_str)*10
    elif len(ms_str)>=3:
      millis=int(ms_str[0:3])
  return (((days*24+hours)*60+minutes)*60+seconds)*1000+millis

def format_ratio_3(num:int,den:int)->str:
  if den<=0:
    return "0.000"
  scaled:int=(num*1000)//den
  whole:int=scaled//1000
  frac:int=scaled%1000
  frac_s:str=str(frac)
  if frac<10:
    frac_s="00"+frac_s
  elif frac<100:
    frac_s="0"+frac_s
  return str(whole)+"."+frac_s

# 241: these helpers are used by the kept broadmarktail/split145
# cache-generation path. They are not removed diagnostics; removing them breaks
# mode28/29/30/31 and bare -g cache-miss builds.
def stream_funcid_reorder_risk_suffix(stats:List[int],m:int)->str:
  risky_a:int=stats[18+19]+stats[18+22]+stats[18+23]+stats[18+24]
  risky_b:int=stats[18+26]+stats[18+27]
  risky_c:int=stats[18+20]+stats[18+21]
  good:int=stats[18+0]+stats[18+4]+stats[18+5]+stats[18+12]+stats[18+16]+stats[18+17]+stats[18+18]
  other:int=m-risky_a-risky_b-risky_c-good
  if other<0:
    other=0
  s:str=""
  s+=f"\t{risky_a}\t{format_ratio_3(risky_a,m)}"
  s+=f"\t{risky_b}\t{format_ratio_3(risky_b,m)}"
  s+=f"\t{risky_c}\t{format_ratio_3(risky_c,m)}"
  s+=f"\t{good}\t{format_ratio_3(good,m)}"
  s+=f"\t{other}\t{format_ratio_3(other,m)}"
  return s

def funcid_reorder_make_quotas(rem_counts:List[int],total_remaining:int,m_target:int)->List[int]:
  quotas:List[int]=[0]*5
  if total_remaining<=0 or m_target<=0:
    return quotas

  g:int=0
  quota_sum:int=0
  while g<5:
    q:int=(rem_counts[g]*m_target)//total_remaining
    if q>rem_counts[g]:
      q=rem_counts[g]
    quotas[g]=q
    quota_sum+=q
    g+=1

  while quota_sum<m_target:
    best:int=-1
    best_room:int=-1
    g=0
    while g<5:
      room:int=rem_counts[g]-quotas[g]
      if room>best_room:
        best_room=room
        best=g
      g+=1
    if best<0 or best_room<=0:
      break
    quotas[best]+=1
    quota_sum+=1

  return quotas

def interleave_funcid_reorder_parts(part_b:List[Dict[str,int]],part_a:List[Dict[str,int]],part_c:List[Dict[str,int]],part_g:List[Dict[str,int]],part_o:List[Dict[str,int]],m_target:int)->List[Dict[str,int]]:
  out:List[Dict[str,int]]=[]
  ib:int=0
  ia:int=0
  ic:int=0
  ig:int=0
  io:int=0
  while len(out)<m_target:
    progressed:bool=False
    if ib<len(part_b):
      out.append(part_b[ib])
      ib+=1
      progressed=True
    if ig<len(part_g):
      out.append(part_g[ig])
      ig+=1
      progressed=True
    if ia<len(part_a):
      out.append(part_a[ia])
      ia+=1
      progressed=True
    if io<len(part_o):
      out.append(part_o[io])
      io+=1
      progressed=True
    if ic<len(part_c):
      out.append(part_c[ic])
      ic+=1
      progressed=True
    if not progressed:
      break
  return out

def stream_measure2_progress_header()->str:
  h:str="N\tpreset\tchunk\toff\tm\tblock\tmax_blocks\tsteps\tsort_mode\telapsed\telapsed_ms\tchunk_total\tgpu_total\tdone_records\ttotal_records\tremaining_records"
  h+="\tfree_popcount_sum\tfree_popcount_avg\tfree_popcount_min\tfree_popcount_max"
  h+="\trow_sum\trow_avg\trow_min\trow_max"
  h+="\tend_sum\tend_avg\tend_min\tend_max"
  h+="\tdepth_sum\tdepth_avg\tdepth_min\tdepth_max"
  h+="\tscore_sum\tscore_avg\tscore_min\tscore_max"
  h+="\tw2_count\tw4_count\tw8_count"
  fid:int=0
  while fid<28:
    h+=f"\tfuncid_{fid}_count"
    fid+=1
  h+="\n"
  return h

def stream_measure2_stats_suffix(stats:List[int],m:int)->str:
  s:str=""
  s+=f"\t{stats[0]}\t{format_ratio_3(stats[0],m)}\t{stats[1]}\t{stats[2]}"
  s+=f"\t{stats[3]}\t{format_ratio_3(stats[3],m)}\t{stats[4]}\t{stats[5]}"
  s+=f"\t{stats[6]}\t{format_ratio_3(stats[6],m)}\t{stats[7]}\t{stats[8]}"
  s+=f"\t{stats[9]}\t{format_ratio_3(stats[9],m)}\t{stats[10]}\t{stats[11]}"
  s+=f"\t{stats[12]}\t{format_ratio_3(stats[12],m)}\t{stats[13]}\t{stats[14]}"
  s+=f"\t{stats[15]}\t{stats[16]}\t{stats[17]}"
  fid:int=0
  while fid<28:
    s+=f"\t{stats[18+fid]}"
    fid+=1
  return s

def analyze_stream_chunk_input_stats(N:int,chunk_constellations:List[Dict[str,int]])->List[int]:
  m:int=len(chunk_constellations)
  stats:List[int]=[0]*46
  if m<=0:
    return stats

  stats[1]=999999999
  stats[4]=999999999
  stats[7]=999999999
  stats[10]=999999999
  stats[13]=999999999

  soa:TaskSoA=TaskSoA(m)
  w_arr:List[u64]=[u64(0)]*m
  build_soa_for_range(N,chunk_constellations,0,m,soa,w_arr)

  i:int=0
  while i<m:
    pc:int=popcount_int(int(soa.free_arr[i]))
    rowv:int=soa.row_arr[i]
    endv:int=soa.end_arr[i]
    depth:int=endv-rowv
    if depth<0:
      depth=0
    score:int=pc*depth

    stats[0]+=pc
    if pc<stats[1]:
      stats[1]=pc
    if pc>stats[2]:
      stats[2]=pc

    stats[3]+=rowv
    if rowv<stats[4]:
      stats[4]=rowv
    if rowv>stats[5]:
      stats[5]=rowv

    stats[6]+=endv
    if endv<stats[7]:
      stats[7]=endv
    if endv>stats[8]:
      stats[8]=endv

    stats[9]+=depth
    if depth<stats[10]:
      stats[10]=depth
    if depth>stats[11]:
      stats[11]=depth

    stats[12]+=score
    if score<stats[13]:
      stats[13]=score
    if score>stats[14]:
      stats[14]=score

    w:int=int(w_arr[i])
    if w==2:
      stats[15]+=1
    elif w==4:
      stats[16]+=1
    elif w==8:
      stats[17]+=1

    fid:int=soa.funcid_arr[i]
    if fid>=0 and fid<28:
      stats[18+fid]+=1

    i+=1

  return stats

def append_stream_progress(progress_fname:str,N:int,preset_queens:int,chunk_index:int,off:int,m:int,BLOCK:int,MAX_BLOCKS:int,STEPS:int,gpu_sort_mode:int,elapsed_text:str,elapsed_ms:int,chunk_total:int,gpu_total:int,total_records:int,stats:List[int])->None:
  done_records:int=off+m
  remaining_records:int=total_records-done_records
  if remaining_records<0:
    remaining_records=0
  with open(progress_fname,"a") as f:
    f.write(f"{N}\t{preset_queens}\t{chunk_index}\t{off}\t{m}\t{BLOCK}\t{MAX_BLOCKS}\t{STEPS}\t{gpu_sort_mode}\t{elapsed_text}\t{elapsed_ms}\t{chunk_total}\t{gpu_total}\t{done_records}\t{total_records}\t{remaining_records}")
    f.write(stream_measure2_stats_suffix(stats,m))
    f.write("\n")

def funcid_reorder_bucket(fid:int)->int:
  if fid==26 or fid==27:
    return 0
  if fid==19 or fid==22 or fid==23 or fid==24:
    return 1
  if fid==20 or fid==21:
    return 2
  if fid==0 or fid==4 or fid==5 or fid==12 or fid==16 or fid==17 or fid==18:
    return 3
  return 4

def funcid_reorder_bucket_label(g:int)->str:
  if g==0:
    return "B"
  if g==1:
    return "A"
  if g==2:
    return "C"
  if g==3:
    return "G"
  return "O"

def truncate_plain_bin(fname:str)->None:
  with open(fname,"wb") as f:
    pass

def stream_funcid_reorder_progress_header()->str:
  h:str=stream_measure2_progress_header().strip()
  h+="\trisky_a_count\trisky_a_ratio"
  h+="\trisky_b_count\trisky_b_ratio"
  h+="\trisky_c_count\trisky_c_ratio"
  h+="\tgood_count\tgood_ratio"
  h+="\tother_count\tother_ratio"
  h+="\n"
  return h

def profile_elapsed_ms_between(t0:datetime,t1:datetime)->int:
  return stream_elapsed_text_to_ms(str(t1-t0)[:-3])

def analyze_stream_chunk_input_stats_from_soa(soa:TaskSoA,w_arr:List[u64],m:int)->List[int]:
  stats:List[int]=[0]*46
  if m<=0:
    return stats

  stats[1]=999999999
  stats[4]=999999999
  stats[7]=999999999
  stats[10]=999999999
  stats[13]=999999999

  i:int=0
  while i<m:
    pc:int=popcount_int(int(soa.free_arr[i]))
    rowv:int=soa.row_arr[i]
    endv:int=soa.end_arr[i]
    depth:int=endv-rowv
    if depth<0:
      depth=0
    score:int=pc*depth

    stats[0]+=pc
    if pc<stats[1]:
      stats[1]=pc
    if pc>stats[2]:
      stats[2]=pc

    stats[3]+=rowv
    if rowv<stats[4]:
      stats[4]=rowv
    if rowv>stats[5]:
      stats[5]=rowv

    stats[6]+=endv
    if endv<stats[7]:
      stats[7]=endv
    if endv>stats[8]:
      stats[8]=endv

    stats[9]+=depth
    if depth<stats[10]:
      stats[10]=depth
    if depth>stats[11]:
      stats[11]=depth

    stats[12]+=score
    if score<stats[13]:
      stats[13]=score
    if score>stats[14]:
      stats[14]=score

    w:int=int(w_arr[i])
    if w==2:
      stats[15]+=1
    elif w==4:
      stats[16]+=1
    elif w==8:
      stats[17]+=1

    fid:int=soa.funcid_arr[i]
    if fid>=0 and fid<28:
      stats[18+fid]+=1

    i+=1

  return stats

def append_stream_funcid_reorder_progress(progress_fname:str,N:int,preset_queens:int,chunk_index:int,off:int,m:int,BLOCK:int,MAX_BLOCKS:int,STEPS:int,gpu_sort_mode:int,elapsed_text:str,elapsed_ms:int,chunk_total:int,gpu_total:int,total_records:int,stats:List[int])->None:
  done_records:int=off+m
  remaining_records:int=total_records-done_records
  if remaining_records<0:
    remaining_records=0
  with open(progress_fname,"a") as f:
    f.write(f"{N}\t{preset_queens}\t{chunk_index}\t{off}\t{m}\t{BLOCK}\t{MAX_BLOCKS}\t{STEPS}\t{gpu_sort_mode}\t{elapsed_text}\t{elapsed_ms}\t{chunk_total}\t{gpu_total}\t{done_records}\t{total_records}\t{remaining_records}")
    f.write(stream_measure2_stats_suffix(stats,m))
    f.write(stream_funcid_reorder_risk_suffix(stats,m))
    f.write("\n")

def broad_markdist_tail_variant_tag()->str:
  v:int=BROAD_MARKDIST_TAIL_VARIANT
  if v==0:
    return "v2base"
  if v==1:
    return "phase_only"
  if v==2:
    return "rotate_only"
  if v==3:
    return "wide_only"
  if v==4:
    return "phase_rotate"
  if v==5:
    return "wide_phase_rotate"
  return "unknown"

def broad_markdist_tail_variant_desc()->str:
  v:int=BROAD_MARKDIST_TAIL_VARIANT
  if v==0:
    return "112/111-like v2 baseline: boost=1, simple tail phase, fixed interleave"
  if v==1:
    return "phase only: boost=1, cell/risk-aware tail phase, fixed interleave"
  if v==2:
    return "rotate only: boost=1, simple tail phase, rotating F17/GH/R interleave"
  if v==3:
    return "wide only: boost=2, simple tail phase, fixed interleave"
  if v==4:
    return "phase+rotate: boost=1, cell/risk-aware tail phase, rotating interleave"
  if v==5:
    return "113-like full: boost=2, cell/risk-aware tail phase, rotating interleave"
  return "unknown broadmarktail variant"

def broad_markdist_tail_window_boost_value()->int:
  v:int=BROAD_MARKDIST_TAIL_VARIANT
  if v==3 or v==5:
    return 2
  return 1

def broad_markdist_tail_phase_salt_value()->int:
  v:int=BROAD_MARKDIST_TAIL_VARIANT
  if v==1 or v==4 or v==5:
    return 53
  return 31

def broad_markdist_tail_use_phase_mix()->bool:
  v:int=BROAD_MARKDIST_TAIL_VARIANT
  return v==1 or v==4 or v==5

def broad_markdist_tail_use_rotating_interleave()->bool:
  v:int=BROAD_MARKDIST_TAIL_VARIANT
  return v==2 or v==4 or v==5

def parse_chunk_list_spec(spec:str)->List[int]:
  out:List[int]=[]
  s:str=spec.strip()
  if s=="" or s=="-":
    return out
  parts:List[str]=s.split(",")
  for p in parts:
    t:str=p.strip()
    if t=="":
      continue
    v:int=int(t)
    if v>=0:
      out.append(v)
  return out

def funcid_mark_effective_values_from_soa(soa:TaskSoA,idx:int)->List[int]:
  ijkl:int=soa.ijkl_arr[idx]
  k:int=getk(ijkl)
  l:int=getl(ijkl)
  lo:int=k
  hi:int=l
  if lo>hi:
    tmp:int=lo
    lo=hi
    hi=tmp
  mark1:int=lo-1
  mark2:int=hi-1
  if mark1<0:
    mark1=0
  if mark2<0:
    mark2=0
  rowv:int=soa.row_arr[idx]
  mark_gap:int=mark2-mark1
  row_to_mark1:int=mark1-rowv
  row_to_mark2:int=mark2-rowv
  return [mark1,mark2,mark_gap,row_to_mark1,row_to_mark2,soa.jmark_arr[idx],soa.end_arr[idx]]

def chunk_list_contains(chunks:List[int],chunk_index:int)->bool:
  for v in chunks:
    if v==chunk_index:
      return True
  return False

def chunk_list_max(chunks:List[int])->int:
  mx:int=-1
  for v in chunks:
    if v>mx:
      mx=v
  return mx

def chunk_list_to_string(chunks:List[int])->str:
  s:str=""
  first:bool=True
  for v in chunks:
    if first:
      s=str(v)
      first=False
    else:
      s+=","+str(v)
  return s


def funcid_reorder_param_tag()->str:
  return f"w{FUNCID_REORDER_V2_WINDOW_MULT}_j{FUNCID_REORDER_V2_PHASE_JUMP}"

def stream_chunk_param_tag(block:int,max_blocks:int)->str:
  b:int=block
  mb:int=max_blocks
  if b<=0:
    b=32
  if mb<=0:
    mb=484
  steps:int=b*mb
  if steps<=0:
    steps=15488
  return f"b{b}_m{mb}_s{steps}"

def funcid_reorder_run_param_tag(block:int,max_blocks:int)->str:
  return f"{funcid_reorder_param_tag()}_{stream_chunk_param_tag(block,max_blocks)}"

def take_striped_records_from_buffer(buf:List[Dict[str,int]],q:int,chunk_index:int,group_id:int)->Tuple[List[Dict[str,int]],List[Dict[str,int]]]:
  taken_records:List[Dict[str,int]]=[]
  n:int=len(buf)
  if q<=0 or n<=0:
    return taken_records,buf
  if q>n:
    q=n

  selected:List[bool]=[False]*n
  step:int=n//q
  if step<=0:
    step=1

  start:int=(chunk_index*FUNCID_REORDER_V2_PHASE_JUMP+group_id*3)%step
  idx:int=start
  taken:int=0
  guard:int=0
  guard_limit:int=n*2+q+16

  while taken<q and guard<guard_limit:
    if not selected[idx]:
      selected[idx]=True
      taken_records.append(buf[idx])
      taken+=1
    idx+=step
    if idx>=n:
      idx=idx%n
    guard+=1

  i:int=0
  while taken<q and i<n:
    if not selected[i]:
      selected[i]=True
      taken_records.append(buf[i])
      taken+=1
    i+=1

  newbuf:List[Dict[str,int]]=[]
  i=0
  while i<n:
    if not selected[i]:
      newbuf.append(buf[i])
    i+=1

  return taken_records,newbuf

def funcid_markdist_risk_reorder_bucket_label(g:int)->str:
  if g==0:
    return "X"   # extreme exact tail
  if g==1:
    return "T"   # tail
  if g==2:
    return "H"   # heavy
  if g==3:
    return "M"   # medium / known-heavy fallback
  return "O"     # other

def funcid_markdist_risk_score(fid:int,mark_gap:int,d1:int)->int:
  if fid==5:
    if mark_gap==5 and d1==4:
      return 368385
    if mark_gap==3 and d1==5:
      return 283091
    if mark_gap==8 and d1==0:
      return 263545
    if mark_gap==5 and d1==5:
      return 239714
    if mark_gap==4 and d1==5:
      return 212174
    if mark_gap==7 and d1==0:
      return 170038
    if mark_gap==4 and d1==4:
      return 160513
    if mark_gap==2 and d1==4:
      return 81086
    if mark_gap==4 and d1==2:
      return 52036
    return 1928

  if fid==0:
    if mark_gap==3 and d1==0:
      return 278909
    if mark_gap==2 and d1==1:
      return 229048
    if mark_gap==2 and d1==2:
      return 21911
    if mark_gap==2 and d1==0:
      return 7626
    return 12654

  if fid==1:
    if mark_gap==3 and d1==-2:
      return 197600
    if mark_gap==4 and d1==-2:
      return 30436
    if mark_gap==2 and d1==-2:
      return 15648
    return 23653

  if fid==4:
    if mark_gap==1 and d1==2:
      return 23208
    if mark_gap==1 and d1==3:
      return 16755
    if mark_gap==1 and d1==1:
      return 15180
    if mark_gap==1 and d1==0:
      return 13930
    return 10456

  if fid==7:
    if mark_gap==1 and d1==1:
      return 22703
    if mark_gap==1 and d1==4:
      return 21049
    if mark_gap==1 and d1==3:
      return 20638
    if mark_gap==1 and d1==2:
      return 13311
    return 9606

  if fid==15:
    if mark_gap==1 and d1==3:
      return 237308
    if mark_gap==1 and d1==1:
      return 12596
    return 19458

  if fid==19:
    if mark_gap==5 and d1==0:
      return 61714
    if mark_gap==2 and d1==3:
      return 40407
    if mark_gap==2 and d1==0:
      return 7701
    return 10278

  if fid==22:
    if mark_gap==1 and d1==0:
      return 34009
    if mark_gap==1 and d1==4:
      return 27746
    if mark_gap==1 and d1==1:
      return 8152
    return 12353

  if fid==24:
    if mark_gap==1 and d1==0:
      return 94216
    if mark_gap==1 and d1==5:
      return 30767
    if mark_gap==1 and d1==2:
      return 5466
    return 9473

  if fid==12 or fid==16 or fid==17 or fid==18:
    return 5000
  if fid==20 or fid==21 or fid==23 or fid==25 or fid==26 or fid==27:
    return 3000
  return 1000

def funcid_markdist_risk_bucket_from_score(score:int)->int:
  if score>=200000:
    return 0
  if score>=50000:
    return 1
  if score>=15000:
    return 2
  if score>=5000:
    return 3
  return 4

def funcid_markdist_risk_bucket(fid:int,mark_gap:int,d1:int)->int:
  return funcid_markdist_risk_bucket_from_score(funcid_markdist_risk_score(fid,mark_gap,d1))

def stream_funcid_markdist_risk_reorder_progress_header()->str:
  h:str=stream_funcid_reorder_progress_header().strip()
  h+="\tmarkrisk_x_count\tmarkrisk_x_ratio"
  h+="\tmarkrisk_t_count\tmarkrisk_t_ratio"
  h+="\tmarkrisk_h_count\tmarkrisk_h_ratio"
  h+="\tmarkrisk_m_count\tmarkrisk_m_ratio"
  h+="\tmarkrisk_o_count\tmarkrisk_o_ratio"
  h+="\tmarkrisk_score_sum\tmarkrisk_score_avg\tmarkrisk_score_min\tmarkrisk_score_max"
  h+="\n"
  return h

def analyze_markdist_risk_stats_from_soa(soa:TaskSoA,m:int)->List[int]:
  out:List[int]=[0]*8
  if m<=0:
    return out
  out[6]=999999999
  out[7]=0
  i:int=0
  while i<m:
    fid:int=soa.funcid_arr[i]
    vals:List[int]=funcid_mark_effective_values_from_soa(soa,i)
    score:int=funcid_markdist_risk_score(fid,vals[2],vals[3])
    b:int=funcid_markdist_risk_bucket_from_score(score)
    if b<0 or b>4:
      b=4
    out[b]+=1
    out[5]+=score
    if score<out[6]:
      out[6]=score
    if score>out[7]:
      out[7]=score
    i+=1
  if out[6]==999999999:
    out[6]=0
  return out

def analyze_markdist_risk_stats(N:int,chunk_constellations:List[Dict[str,int]])->List[int]:
  m:int=len(chunk_constellations)
  stats:List[int]=[0]*8
  if m<=0:
    return stats
  soa:TaskSoA=TaskSoA(m)
  w_arr:List[u64]=[u64(0)]*m
  build_soa_for_range(N,chunk_constellations,0,m,soa,w_arr)
  stats=analyze_markdist_risk_stats_from_soa(soa,m)
  return stats

def stream_funcid_markdist_risk_reorder_suffix(risk_stats:List[int],m:int)->str:
  s:str=""
  s+=f"\t{risk_stats[0]}\t{format_ratio_3(risk_stats[0],m)}"
  s+=f"\t{risk_stats[1]}\t{format_ratio_3(risk_stats[1],m)}"
  s+=f"\t{risk_stats[2]}\t{format_ratio_3(risk_stats[2],m)}"
  s+=f"\t{risk_stats[3]}\t{format_ratio_3(risk_stats[3],m)}"
  s+=f"\t{risk_stats[4]}\t{format_ratio_3(risk_stats[4],m)}"
  s+=f"\t{risk_stats[5]}\t{format_ratio_3(risk_stats[5],m)}\t{risk_stats[6]}\t{risk_stats[7]}"
  return s

def append_stream_funcid_markdist_risk_reorder_progress(progress_fname:str,N:int,preset_queens:int,chunk_index:int,off:int,m:int,BLOCK:int,MAX_BLOCKS:int,STEPS:int,gpu_sort_mode:int,elapsed_text:str,elapsed_ms:int,total_records:int,stats:List[int],risk_stats:List[int])->None:
  done_records:int=off+m
  remaining_records:int=total_records-done_records
  if remaining_records<0:
    remaining_records=0
  with open(progress_fname,"a") as f:
    f.write(f"{N}\t{preset_queens}\t{chunk_index}\t{off}\t{m}\t{BLOCK}\t{MAX_BLOCKS}\t{STEPS}\t{gpu_sort_mode}\t{elapsed_text}\t{elapsed_ms}\t{0}\t{0}\t{done_records}\t{total_records}\t{remaining_records}")
    f.write(stream_measure2_stats_suffix(stats,m))
    f.write(stream_funcid_reorder_risk_suffix(stats,m))
    f.write(stream_funcid_markdist_risk_reorder_suffix(risk_stats,m))
    f.write("\n")

def interleave_funcid_markdist_risk_reorder_parts(part_x:List[Dict[str,int]],part_t:List[Dict[str,int]],part_h:List[Dict[str,int]],part_m:List[Dict[str,int]],part_o:List[Dict[str,int]],m_target:int)->List[Dict[str,int]]:
  out:List[Dict[str,int]]=[]
  ix:int=0
  it:int=0
  ih:int=0
  im:int=0
  io:int=0
  while len(out)<m_target:
    progressed:bool=False
    if ix<len(part_x):
      out.append(part_x[ix])
      ix+=1
      progressed=True
    if io<len(part_o):
      out.append(part_o[io])
      io+=1
      progressed=True
    if it<len(part_t):
      out.append(part_t[it])
      it+=1
      progressed=True
    if im<len(part_m):
      out.append(part_m[im])
      im+=1
      progressed=True
    if ih<len(part_h):
      out.append(part_h[ih])
      ih+=1
      progressed=True
    if not progressed:
      break
  return out

def make_broad_markdist_cell_buffers()->List[List[Dict[str,int]]]:
  out:List[List[Dict[str,int]]]=[]
  i:int=0
  while i<25:
    one:List[Dict[str,int]]=[]
    out.append(one)
    i+=1
  return out

def fill_constellation_buffer_from_bin_range(fname:str,buf:List[Dict[str,int]],off_record:int,target:int)->Tuple[List[Dict[str,int]],int]:
  if target<0:
    target=0
  while len(buf)<target:
    need:int=target-len(buf)
    chunk:List[Dict[str,int]]=read_constellations_bin_range(fname,off_record,need)
    got:int=len(chunk)
    if got<=0:
      break
    i:int=0
    while i<got:
      buf.append(chunk[i])
      i+=1
    off_record+=got
  return buf,off_record

def analyze_broad_markdist_cell_stats_from_soa(soa:TaskSoA,m:int)->List[int]:
  out:List[int]=[0]*25
  i:int=0
  while i<m:
    fid:int=soa.funcid_arr[i]
    broad:int=funcid_reorder_bucket(fid)
    vals:List[int]=funcid_mark_effective_values_from_soa(soa,i)
    risk:int=funcid_markdist_risk_bucket(fid,vals[2],vals[3])
    if broad<0 or broad>4:
      broad=4
    if risk<0 or risk>4:
      risk=4
    out[(broad*5+risk)]+=1
    i+=1
  return out

def analyze_broad_markdist_cell_stats(N:int,chunk_constellations:List[Dict[str,int]])->List[int]:
  m:int=len(chunk_constellations)
  out:List[int]=[0]*25
  if m<=0:
    return out
  soa:TaskSoA=TaskSoA(m)
  w_arr:List[u64]=[u64(0)]*m
  build_soa_for_range(N,chunk_constellations,0,m,soa,w_arr)
  out=analyze_broad_markdist_cell_stats_from_soa(soa,m)
  return out

def stream_broad_markdist_reorder_progress_header()->str:
  h:str=stream_funcid_markdist_risk_reorder_progress_header().strip()
  broad:int=0
  while broad<5:
    risk:int=0
    while risk<5:
      h+=f"\tcell_{funcid_reorder_bucket_label(broad)}_{funcid_markdist_risk_reorder_bucket_label(risk)}_count"
      risk+=1
    broad+=1
  h+="\n"
  return h

def stream_broad_markdist_cell_suffix(cell_stats:List[int])->str:
  s:str=""
  i:int=0
  while i<25:
    v:int=0
    if i<len(cell_stats):
      v=cell_stats[i]
    s+=f"\t{v}"
    i+=1
  return s

def append_stream_broad_markdist_reorder_progress(progress_fname:str,N:int,preset_queens:int,chunk_index:int,off:int,m:int,BLOCK:int,MAX_BLOCKS:int,STEPS:int,gpu_sort_mode:int,elapsed_text:str,elapsed_ms:int,total_records:int,stats:List[int],risk_stats:List[int],cell_stats:List[int])->None:
  done_records:int=off+m
  remaining_records:int=total_records-done_records
  if remaining_records<0:
    remaining_records=0
  with open(progress_fname,"a") as f:
    f.write(f"{N}\t{preset_queens}\t{chunk_index}\t{off}\t{m}\t{BLOCK}\t{MAX_BLOCKS}\t{STEPS}\t{gpu_sort_mode}\t{elapsed_text}\t{elapsed_ms}\t{0}\t{0}\t{done_records}\t{total_records}\t{remaining_records}")
    f.write(stream_measure2_stats_suffix(stats,m))
    f.write(stream_funcid_reorder_risk_suffix(stats,m))
    f.write(stream_funcid_markdist_risk_reorder_suffix(risk_stats,m))
    f.write(stream_broad_markdist_cell_suffix(cell_stats))
    f.write("\n")

def broad_markdist_make_cell_quotas(rem_counts:List[int],total_remaining:int,m_target:int)->List[int]:
  quotas:List[int]=[0]*25
  if total_remaining<=0 or m_target<=0:
    return quotas

  broad_rem:List[int]=[0]*5
  broad:int=0
  while broad<5:
    risk:int=0
    while risk<5:
      broad_rem[broad]+=rem_counts[broad*5+risk]
      risk+=1
    broad+=1

  broad_quotas:List[int]=funcid_reorder_make_quotas(broad_rem,total_remaining,m_target)

  broad=0
  while broad<5:
    bq:int=broad_quotas[broad]
    if bq>0 and broad_rem[broad]>0:
      cell_rem:List[int]=[0]*5
      risk=0
      while risk<5:
        cell_rem[risk]=rem_counts[broad*5+risk]
        risk+=1
      cell_q:List[int]=funcid_reorder_make_quotas(cell_rem,broad_rem[broad],bq)
      risk=0
      while risk<5:
        quotas[broad*5+risk]=cell_q[risk]
        risk+=1
    broad+=1

  qsum:int=0
  cell:int=0
  while cell<25:
    qsum+=quotas[cell]
    cell+=1
  while qsum<m_target:
    best:int=-1
    best_room:int=-1
    cell=0
    while cell<25:
      room:int=rem_counts[cell]-quotas[cell]
      if room>best_room:
        best_room=room
        best=cell
      cell+=1
    if best<0 or best_room<=0:
      break
    quotas[best]+=1
    qsum+=1

  return quotas

def interleave_broad_markdist_secondary_parts(parts:List[List[Dict[str,int]]],broad_quotas:List[int],m_target:int)->List[Dict[str,int]]:
  part_b:List[Dict[str,int]]=interleave_funcid_markdist_risk_reorder_parts(parts[0],parts[1],parts[2],parts[3],parts[4],broad_quotas[0])
  part_a:List[Dict[str,int]]=interleave_funcid_markdist_risk_reorder_parts(parts[5],parts[6],parts[7],parts[8],parts[9],broad_quotas[1])
  part_c:List[Dict[str,int]]=interleave_funcid_markdist_risk_reorder_parts(parts[10],parts[11],parts[12],parts[13],parts[14],broad_quotas[2])
  part_g:List[Dict[str,int]]=interleave_funcid_markdist_risk_reorder_parts(parts[15],parts[16],parts[17],parts[18],parts[19],broad_quotas[3])
  part_o:List[Dict[str,int]]=interleave_funcid_markdist_risk_reorder_parts(parts[20],parts[21],parts[22],parts[23],parts[24],broad_quotas[4])
  return interleave_funcid_reorder_parts(part_b,part_a,part_c,part_g,part_o,m_target)

def broad_markdist_tail_label(tail:int)->str:
  if tail==0:
    return "F17"
  if tail==1:
    return "GH"
  return "R"

def broad_markdist_tail_reorder_subcell_fname(N:int,preset_queens:int,broad:int,risk:int,tail:int)->str:
  return f"constellations_N{N}_{preset_queens}_broadmarktail_reorder_{BROAD_MARKDIST_TAIL_REORDER_VERSION}_{funcid_reorder_bucket_label(broad)}_{funcid_markdist_risk_reorder_bucket_label(risk)}_{broad_markdist_tail_label(tail)}.bin"

def broad_markdist_tail_reorder_output_fname(N:int,preset_queens:int,block:int=32,max_blocks:int=484)->str:
  return f"constellations_N{N}_{preset_queens}_broadmarktail_reorder_{BROAD_MARKDIST_TAIL_REORDER_VERSION}_{broad_markdist_tail_variant_tag()}_{funcid_reorder_run_param_tag(block,max_blocks)}.bin"

CHUNKSHAPE148_REORDER_VERSION:str="scorestripe_v9_lanephase32_octetfirstpairlock29"
CHUNKSHAPE148_DEFAULT_REASON:str="276 restores 274/271 final validated scorestripe_v9 task order/cache after 275 bucket diagnostic; CUDA kernels unchanged; 275 bucket diagnostic, 273 root0 direct kernel, 270 rootpre2flag, and 240 fid14 launch split rejected"

def chunkshape148_reorder_output_fname(N:int,preset_queens:int,block:int=32,max_blocks:int=484)->str:
  return f"constellations_N{N}_{preset_queens}_chunkshape148_{CHUNKSHAPE148_REORDER_VERSION}_{BROAD_MARKDIST_TAIL_REORDER_VERSION}_{broad_markdist_tail_variant_tag()}_{funcid_reorder_run_param_tag(block,max_blocks)}{chunkshape148_bucket_run_tag()}{chunkshape148_iter_sort_tag()}.bin"

CHUNKSHAPE148_SCORE_KEY_MAX:int=32767
CHUNKSHAPE148_LANE_COUNT:int=32
CHUNKSHAPE148_LANE_MASK:int=31

# 339: number of records emitted CONSECUTIVELY from one score bucket before
# rotating to the next bucket in interleave_order. 1 == the 276/274/271
# scorestripe_v9 behaviour that has been in place since 276 (one record per
# bucket, so an aligned group of 32 output records spans all 8 buckets).
#
# Why this knob exists (Open Objectives #8): a warp's 32 lanes process 32
# CONSECUTIVE records per grid-stride iteration (idx = n*stride + 32w + j).
# With RUN=1 every warp-iteration is deliberately made maximally
# heterogeneous in cost, which is exactly the intra-warp trip-count variance
# that 329 measured as the tail effect (Divergent Branches=0, Avg Threads
# Executed 2-3/32). With RUN=32 each aligned 32-record group is drawn from a
# SINGLE score bucket, so lanes inside a warp get similar-cost work.
#
# Inter-warp balance is preserved either way: build_chunkshape148_reordered_bin
# works in units of STEPS=BLOCK*MAX_BLOCKS=15488, i.e. exactly ONE grid-stride
# iteration, and rotates the bucket order by out_ch each iteration
# (order_pos=(oi+out_ch)%8). With RUN=32, warp w at iteration out_ch receives
# interleave_order[(w+out_ch)%8], so across the 131 iterations of an N=21 run
# every warp still sees all 8 buckets near-uniformly. This is what makes
# RUN=32 a strictly intra-warp change rather than a trade of one imbalance
# for another.
#
# RUN=1 reproduces the current cache byte-for-byte AND keeps the existing
# output filename unchanged (see chunkshape148_reorder_output_fname), so the
# 333-338 baseline stays reproducible without rebuilding anything.
# 344: ADOPTED. Changed from 1 (the 276-338 emit order) to 2048 after the
# 339-343 measurement campaign. 2048 is the smallest value at which the
# reordering SATURATES: 343 confirmed at the data level that the shaped bins
# produced by 2048, 4096, 8192, 16384 and 32768 are byte-identical (md5
# 283b038c2573848f6a5b945b20ec6102, 32404512 bytes, all five), so no larger
# value can do anything further. Measured N=21 elapsed 431.677s against the
# in-session RUN=1 control of 450.183s, i.e. -4.111%, with all three chunks
# improving together (-3.853%, -4.529%, -3.836%).
# Setting this back to 1 reproduces the 276-338 order exactly, including the
# cached shaped-bin filename, so the old baseline stays reachable at any time.
CHUNKSHAPE148_BUCKET_RUN:int=2048
# 343: raised again, 8192 -> 65536. 342 raised 256 -> 8192 on the reasoning
# that saturation would arrive near quotas[b] ~ STEPS/8 = 15488/8 ~ 1936.
# THAT REASONING WAS WRONG: 1936 is the AVERAGE quota, but quotas[] is
# proportional to each bucket's remaining population and merely SUMS to
# m_target = STEPS = 15488, so a single bucket's quota can be as large as
# 15488. A run emits min(RUN, quotas[b], remaining) records, therefore the
# guaranteed saturation point is RUN >= 15488, not ~1936 -- and 8192 was
# still BELOW it. 65536 clears 15488 with room to spare.
# The saturation claim is now a hard, falsifiable prediction: every
# RUN >= 15488 must emit every bucket's whole quota as one unbroken block,
# so all such values must produce BYTE-IDENTICAL shaped bins. 343 checks
# that by comparing file digests, which costs no GPU time at all.
# Raising the cap changes nothing for any RUN <= 8192, so every shaped-bin
# cache built in 339-342 stays valid.
CHUNKSHAPE148_BUCKET_RUN_MAX:int=65536
CHUNKSHAPE148_BUCKET_RUN_REASON:str="344 ADOPTED the bucket run-length knob at 2048, changed from the 339-343 source default of 1. 2048 is the saturation point: 343 showed that the shaped bins for 2048, 4096, 8192, 16384 and 32768 are byte-identical, so nothing above 2048 can differ. Measured -4.111% on N=21 against the in-session RUN=1 control. Passing 1 explicitly still reproduces the 276-338 order and its cache filename."

def chunkshape148_bucket_run_value()->int:
  r:int=CHUNKSHAPE148_BUCKET_RUN
  if r<1:
    r=1
  if r>CHUNKSHAPE148_BUCKET_RUN_MAX:
    r=CHUNKSHAPE148_BUCKET_RUN_MAX
  return r

def chunkshape148_bucket_run_tag()->str:
  # 339: RUN=1 must produce the SAME filename as 276-338 so the already-cached
  # shaped bin keeps being reused for the baseline. Only non-default values
  # add a suffix (and therefore get their own cache file).
  r:int=chunkshape148_bucket_run_value()
  if r==1:
    return ""
  return f"_run{r}"

# 345: sort scope knob. See the module docstring, Open Objectives item 9.
#   0  off. Byte-for-byte the 344 emit order, and the cache filename is
#      unchanged so the already-built 344 shaped bin keeps being reused.
#   1  sort one whole launch group (CHUNKSHAPE148_SORT_GROUP output chunks,
#      i.e. 48*15488 = 743424 records) ascending by key>>5, stable.
#   2  same, descending.
#   3  sort only inside each 15488-record output chunk, ascending by key>>5,
#      stable. This is the CONTROL: it has the same granularity as mode 1 but
#      the narrow scope, so it should reintroduce warp imbalance.
#   4  sort one whole launch group ascending by the full key, low 5 lane bits
#      included. This is the only mode that breaks the 276-era lane diffusion
#      inside a score class.
#   5  347: mode 1 followed by a SERPENTINE pass -- the order inside every
#      odd numbered 15488-record stratum is reversed. Mode 1 hands warp w the
#      ranks 32*w .. 32*w+31 in every stratum, so warp 0 always draws the
#      floor and warp 483 always draws the ceiling and the heaviest warp is
#      fixed. Serpentine makes each warp draw the ceiling half the time.
#   6  347: mode 2 followed by the same serpentine pass. Once the systematic
#      offset is removed, direction should become genuinely symmetric, so
#      this also probes the unexplained chunk2 direction dependence seen in
#      345 under mode 2.
#   7  348: mode 5 plus the LIGHT TAIL rotation -- serpentine AND the lightest
#      records placed in the ragged final stratum.
#   8  348: mode 1 plus the LIGHT TAIL rotation, with NO serpentine. Together
#      with modes 1, 5 and 7 this completes a 2x2 factorial over
#      {serpentine, none} x {heavy tail, light tail}. Both rotations are
#      no-ops when the group length is an exact multiple of iter_len, so on
#      the two full launches mode 8 is IDENTICAL to mode 1 and mode 7 is
#      IDENTICAL to mode 5, and only chunk2 can move.
#   9  349: CONDITIONAL serpentine. Serpentine is applied only when the group
#      length is an exact multiple of iter_len, so chunk0 and chunk1 get it
#      and chunk2 does not. By construction mode 9 is element-for-element
#      IDENTICAL to mode 5 on the two full launches and IDENTICAL to mode 1 on
#      the partial one, so its shaped bin is the concatenation of theirs and
#      its timing is fully determined by measurements that already exist.
# In every non-zero mode ONLY the write order changes: the picked indices are
# buffered and reordered, while quotas, the lane scan, phase_seed and
# written_by_bucket are untouched, so membership is preserved exactly.
# 350: ADOPTED at 9, the conditional serpentine. 346 adopted 1 after 345 swept
# 0/1/2/3/4 and established that SORT SCOPE is the mechanism. 347 and 348 then
# measured, twice, that serpentine helps a full launch and hurts the partial
# one, and 349 measured mode 9, which applies serpentine only to groups that
# are an exact multiple of iter_len: 398.988s against an in-session mode 1
# anchor of 402.301s, -0.824%, and -0.503% against mode 5.
# Two earlier defaults stay reachable: 1 reproduces 346 with its cached bin,
# and 0 reproduces the 344 order exactly, filename included.
CHUNKSHAPE148_ITER_SORT:int=9
CHUNKSHAPE148_ITER_SORT_MAX:int=9
# 345: group size in OUTPUT CHUNKS. Pinned to K_PER_THREAD_MAXD14 so that one
# group is exactly one GPU launch: shaping STEPS is BLOCK*MAX_BLOCKS = 15488
# while the launch STEPS is BLOCK*MAX_BLOCKS*K = 743424, so 48 output chunks
# are consumed by one launch. Keeping the group pinned to the launch is what
# makes the warp stratification argument hold and also keeps per-launch record
# counts identical to 344.
CHUNKSHAPE148_SORT_GROUP:int=48
# 345: stable-sort packing. A group holds at most 48*15488 = 743424 records,
# which needs 20 bits, so a sequence number is packed into the low 20 bits and
# the sort key above it. Sorting the packed integers is therefore stable by
# construction without relying on the sort implementation being stable.
CHUNKSHAPE148_SORT_SEQ_BITS:int=20
CHUNKSHAPE148_SORT_SEQ_MASK:int=1048575
CHUNKSHAPE148_ITER_SORT_REASON:str="350 ADOPTED the sort layout knob at 9, the conditional serpentine, changed from the 346 to 349 default of 1. 345 established that sort SCOPE is the mechanism and 346 adopted mode 1. 347 and 348 then measured twice that serpentine helps a full launch by about 1.25 percent and hurts the partial one by about 1.87 percent, replicating to two thousandths of a point, while the 348 factorial refuted the light tail idea in both rows and withdrew the self-compensation explanation. Mode 9 therefore applies serpentine only when the group length is an exact multiple of iter_len, which makes it element-for-element mode 5 on the full launches and mode 1 on the partial one; cmp confirmed the shaped bin is exactly that concatenation. Measured 398.988s against an in-session mode 1 anchor of 402.301s. The rule is EMPIRICAL: the mechanism behind the partial-launch penalty is not understood. Passing 1 reproduces 346 and passing 0 reproduces the 344 order and its cache filename."

def chunkshape148_iter_sort_value()->int:
  m:int=CHUNKSHAPE148_ITER_SORT
  if m<0:
    m=0
  if m>CHUNKSHAPE148_ITER_SORT_MAX:
    m=CHUNKSHAPE148_ITER_SORT_MAX
  return m

def chunkshape148_iter_sort_tag()->str:
  # 345: mode 0 must produce the SAME filename as 344 so the already-cached
  # shaped bin keeps being reused for the anchor run. Only non-zero modes add
  # a suffix and therefore get their own cache file.
  m:int=chunkshape148_iter_sort_value()
  if m==0:
    return ""
  return f"_isort{m}"

def chunkshape148_sort_group_chunks()->int:
  g:int=CHUNKSHAPE148_SORT_GROUP
  if g<1:
    g=1
  return g

def chunkshape148_reorder_group(picked:List[int],score_key_by_idx:List[int],mode:int,iter_len:int)->List[int]:
  # 345: reorder the indices picked for one launch group. Membership is never
  # changed here -- the returned list is a permutation of the input.
  n:int=len(picked)
  if mode<=0 or n<=1:
    return picked
  if n>(CHUNKSHAPE148_SORT_SEQ_MASK+1):
    print(f"[chunkshape148-warning] group too large for stable packing: n={n} cap={CHUNKSHAPE148_SORT_SEQ_MASK+1}; iter_sort disabled for this group")
    return picked

  span:int=iter_len
  if mode!=3 or span<1:
    span=n

  out:List[int]=[]
  base:int=0
  while base<n:
    seg:int=span
    if base+seg>n:
      seg=n-base
    packed:List[int]=[]
    j:int=0
    while j<seg:
      key:int=score_key_by_idx[picked[base+j]]
      part:int=0
      if mode==4:
        part=key
      elif mode==2 or mode==6:
        part=(CHUNKSHAPE148_SCORE_KEY_MAX>>5)-(key>>5)
        if part<0:
          part=0
      else:
        part=key>>5
      packed.append((part<<CHUNKSHAPE148_SORT_SEQ_BITS)|j)
      j+=1
    packed.sort()
    j=0
    while j<seg:
      out.append(picked[base+(packed[j]&CHUNKSHAPE148_SORT_SEQ_MASK)])
      j+=1
    base+=seg

  # 348: LIGHT TAIL rotation for modes 7 and 8.
  # grid-stride fixes WHERE the ragged stratum sits -- it is always the final
  # partial stride step -- but not WHICH records land in it. With a plain
  # ascending sort the ragged stratum receives the HEAVIEST records, and they
  # go to the low-index threads. In mode 1 that happened to be self
  # compensating, because those same threads drew the floor of every other
  # stratum, and 347 showed that serpentine destroys the compensation and
  # costs chunk2 +1.867%. Rotating the sorted group left by the remainder puts
  # the LIGHTEST records in the ragged stratum instead, which turns an accident
  # into a design. When the group length is an exact multiple of iter_len the
  # remainder is zero and this is a no-op, so full launches are untouched.
  if (mode==7 or mode==8) and iter_len>=1:
    rem:int=len(out)%iter_len
    if rem>0:
      rot:List[int]=[]
      ri:int=rem
      while ri<len(out):
        rot.append(out[ri])
        ri+=1
      ri=0
      while ri<rem:
        rot.append(out[ri])
        ri+=1
      out=rot

  # 349: CONDITIONAL serpentine, mode 9. 347 and 348 both measured the same
  # thing: serpentine helps a FULL launch and hurts the PARTIAL one. chunk0 and
  # chunk1 improved by about 1.25% while chunk2 regressed by 1.867% in 347 and
  # 1.865% in 348, a replication to two thousandths of a point. 348 also showed
  # that the light tail rotation makes chunk2 worse in BOTH rows, so the
  # self-compensation story offered in 347 was wrong and has been withdrawn.
  # The mechanism behind the chunk2 penalty is still NOT understood. Mode 9
  # therefore applies the treatment only where it measurably helps: serpentine
  # runs when the group is an exact multiple of iter_len and is skipped when a
  # ragged stratum is present. This is an EMPIRICAL rule, not a derived one.
  # It is stated in terms of launch shape rather than any N-specific constant,
  # so it carries over to other N unchanged.
  serp_on:bool=(mode==5 or mode==6 or mode==7)
  if mode==9 and iter_len>=1 and (len(out)%iter_len)==0:
    serp_on=True

  # 347: serpentine (boustrophedon) post-pass for modes 5, 6, 7 and 9.
  # A plain sorted launch group hands warp w the ranks 32*w .. 32*w+31 inside
  # EVERY stratum, so warp 0 always draws the floor of each stratum and warp
  # 483 always draws the ceiling. That fixes which warp is heaviest and
  # therefore fixes the makespan. Reversing the order inside every odd
  # numbered stratum makes each warp draw the ceiling in half the strata and
  # the floor in the other half, which equalises the warp totals. The reversal
  # is applied to whole iter_len slices, so the 32 alignment established in 340
  # is preserved exactly.
  if serp_on and iter_len>=1:
    serp:List[int]=[]
    sbase:int=0
    sidx:int=0
    while sbase<len(out):
      sseg:int=iter_len
      if sbase+sseg>len(out):
        sseg=len(out)-sbase
      if (sidx&1)==1:
        k:int=sseg-1
        while k>=0:
          serp.append(out[sbase+k])
          k-=1
      else:
        k2:int=0
        while k2<sseg:
          serp.append(out[sbase+k2])
          k2+=1
      sbase+=sseg
      sidx+=1
    out=serp
  return out

def chunkshape148_score_key_from_soa(soa:TaskSoA,idx:int,global_idx:int)->int:
  fid:int=soa.funcid_arr[idx]
  pc:int=popcount_int(int(soa.free_arr[idx]))
  depth:int=soa.end_arr[idx]-soa.row_arr[idx]
  if depth<0:
    depth=0

  mark_gap:int=soa.mark2_arr[idx]-soa.mark1_arr[idx]
  if mark_gap<0:
    mark_gap=-mark_gap
  row_to_end:int=soa.end_arr[idx]-soa.row_arr[idx]
  if row_to_end<0:
    row_to_end=0
  row_to_mark1:int=soa.mark1_arr[idx]-soa.row_arr[idx]
  if row_to_mark1<0:
    row_to_mark1=0

  raw:int=0
  raw+=pc*12
  raw+=depth*7
  raw+=row_to_end*3

  if fid==26 or fid==27:
    raw+=96
  elif fid==19 or fid==22 or fid==23 or fid==24:
    raw+=72
  elif fid==20 or fid==21:
    raw+=56
  elif fid==17:
    raw+=42
  elif fid==14:
    raw+=36
  elif fid==0 or fid==4 or fid==5 or fid==12 or fid==16 or fid==18:
    raw+=20
  else:
    raw+=8

  if pc>=5:
    raw+=20
  elif pc>=4:
    raw+=12
  elif pc>=3:
    raw+=6

  if depth>=13:
    raw+=20
  elif depth>=11:
    raw+=12
  elif depth>=9:
    raw+=6

  if mark_gap>=3:
    raw+=8
  if row_to_mark1>=4:
    raw+=4

  tie:int=(global_idx*13 + soa.ijkl_arr[idx]*7 + fid*5 + pc*3 + depth) & 31
  key:int=raw*32 + tie
  if key<0:
    key=0
  if key>CHUNKSHAPE148_SCORE_KEY_MAX:
    key=CHUNKSHAPE148_SCORE_KEY_MAX
  return key

def chunkshape148_build_thresholds_from_hist(hist:List[int],total_records:int)->List[int]:
  thresholds:List[int]=[0]*8
  if total_records<=0:
    thresholds[7]=CHUNKSHAPE148_SCORE_KEY_MAX
    return thresholds

  b:int=0
  cum:int=0
  k:int=0
  while k<=CHUNKSHAPE148_SCORE_KEY_MAX:
    cum+=hist[k]
    while b<7:
      target:int=(total_records*(b+1))//8
      if target<=0:
        target=1
      if cum>=target:
        thresholds[b]=k
        b+=1
      else:
        break
    k+=1

  while b<7:
    thresholds[b]=CHUNKSHAPE148_SCORE_KEY_MAX
    b+=1
  thresholds[7]=CHUNKSHAPE148_SCORE_KEY_MAX
  return thresholds

def chunkshape148_bucket_from_key(key:int,thresholds:List[int])->int:
  b:int=0
  while b<7:
    if key<=thresholds[b]:
      return b
    b+=1
  return 7

def chunkshape148_lane_from_key(key:int)->int:
  lane:int=key&CHUNKSHAPE148_LANE_MASK
  if lane<0:
    lane=0
  if lane>=CHUNKSHAPE148_LANE_COUNT:
    lane=lane&CHUNKSHAPE148_LANE_MASK
  return lane

def chunkshape148_make_quotas(rem_counts:List[int],total_remaining:int,m_target:int)->List[int]:
  quotas:List[int]=[0]*8
  if total_remaining<=0 or m_target<=0:
    return quotas

  b:int=0
  quota_sum:int=0
  while b<8:
    q:int=(rem_counts[b]*m_target)//total_remaining
    if q>rem_counts[b]:
      q=rem_counts[b]
    quotas[b]=q
    quota_sum+=q
    b+=1

  while quota_sum<m_target:
    best:int=-1
    best_room:int=-1
    b=0
    while b<8:
      room:int=rem_counts[b]-quotas[b]
      if room>best_room:
        best_room=room
        best=b
      b+=1
    if best<0 or best_room<=0:
      break
    quotas[best]+=1
    quota_sum+=1

  return quotas

def build_chunkshape148_reordered_bin(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0
)->Tuple[str,int,int]:
  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  total_records:int=count_constellations_bin_records(fname)
  n_chunks:int=0
  if total_records>0:
    n_chunks=(total_records+STEPS-1)//STEPS

  out_fname:str=chunkshape148_reorder_output_fname(N,preset_queens,BLOCK,MAX_BLOCKS)
  progress_fname:str=f"progress_N{N}_{preset_queens}_chunkshape148_{CHUNKSHAPE148_REORDER_VERSION}_{BROAD_MARKDIST_TAIL_REORDER_VERSION}_{broad_markdist_tail_variant_tag()}_{funcid_reorder_run_param_tag(BLOCK,MAX_BLOCKS)}_sim.tsv"

  if gpu_log_level>=1:
    print(f"[chunkshape148-build-config] N={N} source={fname} records={total_records} chunks={n_chunks} steps={STEPS} bucket_run={chunkshape148_bucket_run_value()} iter_sort={chunkshape148_iter_sort_value()} sort_group={chunkshape148_sort_group_chunks()} output={out_fname} progress={progress_fname} reason={CHUNKSHAPE148_DEFAULT_REASON}")

  with open(progress_fname,"w") as pf:
    lane_header:str=""
    lane_i:int=0
    while lane_i<CHUNKSHAPE148_LANE_COUNT:
      lane_header+=f"\tl{lane_i}"
      lane_i+=1
    pf.write("N\tpreset\tchunk\tout_records\ttotal_records\telapsed\telapsed_ms\tb0\tb1\tb2\tb3\tb4\tb5\tb6\tb7"+lane_header+"\n")

  if total_records<=0:
    truncate_plain_bin(out_fname)
    write_stream_done_count(out_fname+".done",0)
    return out_fname,0,0

  data:str=""
  with open(fname,"rb") as f:
    data=f.read()

  expected_bytes:int=total_records*16
  if len(data)<expected_bytes:
    print(f"[chunkshape148-warning] source byte count shorter than expected: bytes={len(data)} expected={expected_bytes}")
    total_records=len(data)//16
    n_chunks=(total_records+STEPS-1)//STEPS

  truncate_plain_bin(out_fname)
  write_stream_done_count(out_fname+".done",0)

  score_key_by_idx:List[int]=[0]*total_records
  key_hist:List[int]=[0]*(CHUNKSHAPE148_SCORE_KEY_MAX+1)
  off:int=0
  classify_chunk:int=0
  key_min:int=CHUNKSHAPE148_SCORE_KEY_MAX
  key_max:int=0
  while off<total_records:
    m:int=STEPS
    remain:int=total_records-off
    if remain<m:
      m=remain

    chunk_constellations:List[Dict[str,int]]=[]
    i:int=0
    while i<m:
      p:int=(off+i)*16
      ld:int=read_uint32_le(data[p:p+4])
      rd:int=read_uint32_le(data[p+4:p+8])
      col:int=read_uint32_le(data[p+8:p+12])
      startijkl:int=read_uint32_le(data[p+12:p+16])
      chunk_constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
      i+=1

    soa:TaskSoA=TaskSoA(m)
    w_arr:List[u64]=[u64(0)]*m
    build_soa_for_range(N,chunk_constellations,0,m,soa,w_arr)

    i=0
    while i<m:
      key:int=chunkshape148_score_key_from_soa(soa,i,off+i)
      if key<0:
        key=0
      if key>CHUNKSHAPE148_SCORE_KEY_MAX:
        key=CHUNKSHAPE148_SCORE_KEY_MAX
      score_key_by_idx[off+i]=key
      key_hist[key]+=1
      if key<key_min:
        key_min=key
      if key>key_max:
        key_max=key
      i+=1

    if gpu_log_level>=2:
      print(f"[chunkshape148-classify-chunk] N={N} chunk={classify_chunk} off={off} m={m}")
    off+=m
    classify_chunk+=1

  score_thresholds:List[int]=chunkshape148_build_thresholds_from_hist(key_hist,total_records)
  bucket_counts:List[int]=[0]*8
  lane_counts:List[int]=[0]*CHUNKSHAPE148_LANE_COUNT
  idx_count:int=0
  while idx_count<total_records:
    b0:int=chunkshape148_bucket_from_key(score_key_by_idx[idx_count],score_thresholds)
    lane0:int=chunkshape148_lane_from_key(score_key_by_idx[idx_count])
    bucket_counts[b0]+=1
    lane_counts[lane0]+=1
    idx_count+=1

  if gpu_log_level>=1:
    print(f"[chunkshape148-score-key] N={N} records={total_records} key_min={key_min} key_max={key_max} t0={score_thresholds[0]} t1={score_thresholds[1]} t2={score_thresholds[2]} t3={score_thresholds[3]} t4={score_thresholds[4]} t5={score_thresholds[5]} t6={score_thresholds[6]}")
    print(f"[chunkshape148-score-buckets] N={N} records={total_records} b0={bucket_counts[0]} b1={bucket_counts[1]} b2={bucket_counts[2]} b3={bucket_counts[3]} b4={bucket_counts[4]} b5={bucket_counts[5]} b6={bucket_counts[6]} b7={bucket_counts[7]}")
    lane_counts_text:str=""
    lane_i:int=0
    while lane_i<CHUNKSHAPE148_LANE_COUNT:
      lane_counts_text+=f" l{lane_i}={lane_counts[lane_i]}"
      lane_i+=1
    print(f"[chunkshape148-lanes] N={N} records={total_records}{lane_counts_text}")

  bucket_lane_indices:List[List[int]]=[]
  flat:int=0
  while flat<(8*CHUNKSHAPE148_LANE_COUNT):
    one:List[int]=[]
    bucket_lane_indices.append(one)
    flat+=1

  out_ch:int=0
  staged_total:int=0
  while out_ch<n_chunks:
    base:int=0
    while base*n_chunks<STEPS:
      src_ch:int=0
      while src_ch<n_chunks:
        rem:int=(src_ch+out_ch)%n_chunks
        within:int=base*n_chunks+rem
        if within<STEPS:
          idx:int=src_ch*STEPS+within
          if idx<total_records:
            bucket:int=chunkshape148_bucket_from_key(score_key_by_idx[idx],score_thresholds)
            lane_val:int=chunkshape148_lane_from_key(score_key_by_idx[idx])
            bucket_lane_indices[bucket*CHUNKSHAPE148_LANE_COUNT+lane_val].append(idx)
            staged_total+=1
        src_ch+=1
      base+=1
    out_ch+=1

  if staged_total!=total_records:
    print(f"[chunkshape148-warning] staged count mismatch: staged={staged_total} source={total_records}")

  bucket_lane_pos:List[int]=[0]*(8*CHUNKSHAPE148_LANE_COUNT)
  bucket_lane_rem:List[int]=[0]*(8*CHUNKSHAPE148_LANE_COUNT)
  bucket_rem:List[int]=[0]*8
  b:int=0
  while b<8:
    lane:int=0
    while lane<CHUNKSHAPE148_LANE_COUNT:
      flat=b*CHUNKSHAPE148_LANE_COUNT+lane
      cnt:int=len(bucket_lane_indices[flat])
      bucket_lane_rem[flat]=cnt
      bucket_rem[b]+=cnt
      lane+=1
    b+=1

  interleave_order:List[int]=[7,0,6,1,5,2,4,3]
  bucket_run:int=chunkshape148_bucket_run_value()  # 339
  iter_sort_mode:int=chunkshape148_iter_sort_value()  # 345
  sort_group_chunks:int=chunkshape148_sort_group_chunks()  # 345
  group_picked:List[int]=[]  # 345: picked indices buffered for one launch group
  group_chunks_held:int=0
  group_index:int=0

  written_total:int=0
  out_ch=0
  with open(out_fname,"ab") as out:
    while out_ch<n_chunks:
      t0:datetime=datetime.now()
      m_target:int=STEPS
      total_remaining:int=total_records-written_total
      if total_remaining<m_target:
        m_target=total_remaining

      quotas:List[int]=chunkshape148_make_quotas(bucket_rem,total_remaining,m_target)
      written_by_bucket:List[int]=[0]*8
      written_by_lane:List[int]=[0]*CHUNKSHAPE148_LANE_COUNT
      written_chunk:int=0

      quartet_index:int=out_ch//4
      octet_first_pair_phase_bias:int=0
      if (quartet_index&1)==0 and (out_ch&3)<2 and (((quartet_index+1)*4)+1)<n_chunks:
        octet_first_pair_phase_bias=29

      while written_chunk<m_target:
        made:int=0
        oi:int=0
        while oi<8:
          order_pos:int=(oi+out_ch)%8
          b=interleave_order[order_pos]
          # 339: emit up to bucket_run records from this bucket before
          # rotating. bucket_run==1 is byte-for-byte the 276-338 behaviour:
          # the loop body below is unchanged, it simply runs once.
          rep:int=0
          while rep<bucket_run:
            if quotas[b]<=0:
              break
            pair_phase_bias:int=0
            if (out_ch&1)==0:
              pair_phase_bias=11 if (out_ch&3)==0 else 1
            phase_seed:int=(out_ch*11 + quartet_index*17 + b*13 + pair_phase_bias + octet_first_pair_phase_bias) & CHUNKSHAPE148_LANE_MASK
            start_lane:int=(phase_seed + written_by_bucket[b]*5) & CHUNKSHAPE148_LANE_MASK
            found:int=-1
            scan:int=0
            while scan<CHUNKSHAPE148_LANE_COUNT:
              lane=(start_lane+scan) & CHUNKSHAPE148_LANE_MASK
              flat=b*CHUNKSHAPE148_LANE_COUNT+lane
              if bucket_lane_rem[flat]>0:
                found=lane
                break
              scan+=1
            if found<0:
              break
            flat=b*CHUNKSHAPE148_LANE_COUNT+found
            pick_idx:int=bucket_lane_indices[flat][bucket_lane_pos[flat]]
            # 345: buffer the pick instead of writing it here. The selection
            # rules below are byte-for-byte the 344 ones, so membership is
            # unchanged; only the eventual write order can differ.
            group_picked.append(pick_idx)
            bucket_lane_pos[flat]+=1
            bucket_lane_rem[flat]-=1
            bucket_rem[b]-=1
            quotas[b]-=1
            written_by_bucket[b]+=1
            written_by_lane[found]+=1
            written_chunk+=1
            made+=1
            rep+=1
            if written_chunk>=m_target:
              break
          if written_chunk>=m_target:
            break
          oi+=1
        if made==0:
          break

      written_total+=written_chunk
      t1:datetime=datetime.now()
      elapsed_text:str=str(t1-t0)[:-3]
      elapsed_ms:int=stream_elapsed_text_to_ms(elapsed_text)
      with open(progress_fname,"a") as pf:
        lane_values:str=""
        lane_i:int=0
        while lane_i<CHUNKSHAPE148_LANE_COUNT:
          lane_values+=f"\t{written_by_lane[lane_i]}"
          lane_i+=1
        pf.write(f"{N}\t{preset_queens}\t{out_ch}\t{written_chunk}\t{total_records}\t{elapsed_text}\t{elapsed_ms}\t{written_by_bucket[0]}\t{written_by_bucket[1]}\t{written_by_bucket[2]}\t{written_by_bucket[3]}\t{written_by_bucket[4]}\t{written_by_bucket[5]}\t{written_by_bucket[6]}\t{written_by_bucket[7]}"+lane_values+"\n")
      if gpu_log_level>=2:
        lane_debug:str=""
        lane_i:int=0
        while lane_i<CHUNKSHAPE148_LANE_COUNT:
          lane_debug+=f" l{lane_i}={written_by_lane[lane_i]}"
          lane_i+=1
        print(f"[chunkshape148-build-chunk] N={N} chunk={out_ch} written={written_chunk} total_written={written_total} b0={written_by_bucket[0]} b1={written_by_bucket[1]} b2={written_by_bucket[2]} b3={written_by_bucket[3]} b4={written_by_bucket[4]} b5={written_by_bucket[5]} b6={written_by_bucket[6]} b7={written_by_bucket[7]}{lane_debug}")
      out_ch+=1

      # 345: flush one launch group. The boundary is pinned to
      # CHUNKSHAPE148_SORT_GROUP output chunks, which is exactly one GPU
      # launch, and the group is also flushed when the source runs out.
      group_chunks_held+=1
      if group_chunks_held>=sort_group_chunks or out_ch>=n_chunks or written_total>=total_records:
        ordered:List[int]=chunkshape148_reorder_group(group_picked,score_key_by_idx,iter_sort_mode,STEPS)
        if len(ordered)!=len(group_picked):
          print(f"[chunkshape148-warning] group reorder changed length: in={len(group_picked)} out={len(ordered)}")
        gi:int=0
        while gi<len(ordered):
          pick_p:int=ordered[gi]*16
          out.write(data[pick_p:pick_p+16])
          gi+=1
        if gpu_log_level>=1:
          gk_min:int=CHUNKSHAPE148_SCORE_KEY_MAX
          gk_max:int=0
          gi=0
          while gi<len(ordered):
            gk:int=score_key_by_idx[ordered[gi]]
            if gk<gk_min:
              gk_min=gk
            if gk>gk_max:
              gk_max=gk
            gi+=1
          head_key:int=score_key_by_idx[ordered[0]] if len(ordered)>0 else 0
          tail_key:int=score_key_by_idx[ordered[len(ordered)-1]] if len(ordered)>0 else 0
          print(f"[chunkshape148-group-flush] N={N} group={group_index} chunks={group_chunks_held} records={len(ordered)} iter_sort={iter_sort_mode} key_min={gk_min} key_max={gk_max} head_key={head_key} tail_key={tail_key}")
        group_picked=[0]*0
        group_chunks_held=0
        group_index+=1

  write_stream_done_count(out_fname+".done",written_total)
  reordered_records:int=count_constellations_bin_records(out_fname)
  if written_total!=total_records:
    print(f"[chunkshape148-warning] record count mismatch: written={written_total} source={total_records}")
  if reordered_records!=written_total:
    print(f"[chunkshape148-warning] output count mismatch: file_records={reordered_records} written={written_total}")
  if gpu_log_level>=1:
    print(f"[chunkshape148-build-summary] N={N} records={reordered_records} chunks={n_chunks} output={out_fname} progress={progress_fname} valid={1 if validate_bin_file(out_fname) else 0}")

  return out_fname,reordered_records,n_chunks

def broad_markdist_tail_subcell_index(broad:int,risk:int,tail:int)->int:
  return (broad*5+risk)*3+tail

def make_broad_markdist_tail_subcell_buffers()->List[List[Dict[str,int]]]:
  out:List[List[Dict[str,int]]]=[]
  i:int=0
  while i<75:
    one:List[Dict[str,int]]=[]
    out.append(one)
    i+=1
  return out

def broad_markdist_tail_bucket(fid:int,broad:int,risk:int)->int:
  if fid==17:
    return 0
  if broad==3 and risk==2:
    return 1
  return 2

def broad_markdist_tail_phase_group_id(subcell:int,broad:int,risk:int,tail:int)->int:
  gid:int=subcell
  if broad_markdist_tail_use_phase_mix():
    gid+=broad*BROAD_MARKDIST_TAIL_CELL_SALT
    gid+=risk*BROAD_MARKDIST_TAIL_RISK_SALT
    if tail==0:
      gid+=broad_markdist_tail_phase_salt_value()
    elif tail==1:
      gid+=broad_markdist_tail_phase_salt_value()*2
  else:
    if tail==0 or tail==1:
      gid+=broad_markdist_tail_phase_salt_value()
  return gid

def analyze_broad_markdist_tail_summary_from_soa(soa:TaskSoA,m:int)->List[int]:
  out:List[int]=[0]*4
  i:int=0
  while i<m:
    fid:int=soa.funcid_arr[i]
    broad:int=funcid_reorder_bucket(fid)
    vals:List[int]=funcid_mark_effective_values_from_soa(soa,i)
    risk:int=funcid_markdist_risk_bucket(fid,vals[2],vals[3])
    if broad<0 or broad>4:
      broad=4
    if risk<0 or risk>4:
      risk=4
    score:int=0
    if fid==17:
      out[0]+=1
      score+=4
    if broad==3 and risk==2:
      out[1]+=1
      score+=3
    if risk==2:
      out[2]+=1
      score+=1
    out[3]+=score
    i+=1
  return out

def analyze_broad_markdist_tail_summary(N:int,chunk_constellations:List[Dict[str,int]])->List[int]:
  m:int=len(chunk_constellations)
  out:List[int]=[0]*4
  if m<=0:
    return out
  soa:TaskSoA=TaskSoA(m)
  w_arr:List[u64]=[u64(0)]*m
  build_soa_for_range(N,chunk_constellations,0,m,soa,w_arr)
  out=analyze_broad_markdist_tail_summary_from_soa(soa,m)
  return out

def stream_broad_markdist_tail_reorder_progress_header()->str:
  h:str=stream_broad_markdist_reorder_progress_header().strip()
  h+="\ttail_funcid17_count"
  h+="\ttail_cell_G_H_count"
  h+="\ttail_markrisk_H_count"
  h+="\ttail_proxy_sum"
  h+="\ttail_proxy_avg"
  h+="\n"
  return h

def stream_broad_markdist_tail_summary_suffix(tail_stats:List[int],m:int)->str:
  f17:int=0
  gh:int=0
  hcnt:int=0
  proxy:int=0
  if len(tail_stats)>0:
    f17=tail_stats[0]
  if len(tail_stats)>1:
    gh=tail_stats[1]
  if len(tail_stats)>2:
    hcnt=tail_stats[2]
  if len(tail_stats)>3:
    proxy=tail_stats[3]
  return f"\t{f17}\t{gh}\t{hcnt}\t{proxy}\t{format_ratio_3(proxy,m)}"

def append_stream_broad_markdist_tail_reorder_progress(progress_fname:str,N:int,preset_queens:int,chunk_index:int,off:int,m:int,BLOCK:int,MAX_BLOCKS:int,STEPS:int,gpu_sort_mode:int,elapsed_text:str,elapsed_ms:int,total_records:int,stats:List[int],risk_stats:List[int],cell_stats:List[int],tail_stats:List[int])->None:
  done_records:int=off+m
  remaining_records:int=total_records-done_records
  if remaining_records<0:
    remaining_records=0
  with open(progress_fname,"a") as f:
    f.write(f"{N}\t{preset_queens}\t{chunk_index}\t{off}\t{m}\t{BLOCK}\t{MAX_BLOCKS}\t{STEPS}\t{gpu_sort_mode}\t{elapsed_text}\t{elapsed_ms}\t{0}\t{0}\t{done_records}\t{total_records}\t{remaining_records}")
    f.write(stream_measure2_stats_suffix(stats,m))
    f.write(stream_funcid_reorder_risk_suffix(stats,m))
    f.write(stream_funcid_markdist_risk_reorder_suffix(risk_stats,m))
    f.write(stream_broad_markdist_cell_suffix(cell_stats))
    f.write(stream_broad_markdist_tail_summary_suffix(tail_stats,m))
    f.write("\n")

def build_broad_markdist_tail_reorder_subcell_bins(N:int,fname:str,preset_queens:int,BLOCK:int,MAX_BLOCKS:int,gpu_log_level:int=0)->List[int]:
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  subcell:int=0
  while subcell<75:
    cell:int=subcell//3
    broad:int=cell//5
    risk:int=cell%5
    tail:int=subcell%3
    truncate_plain_bin(broad_markdist_tail_reorder_subcell_fname(N,preset_queens,broad,risk,tail))
    subcell+=1

  counts:List[int]=[0]*75
  chunk_index:int=0
  _read_uint32_le=read_uint32_le

  if gpu_log_level>=1:
    print(f"[broadmarktail-reorder-subcell-config] N={N} bin={fname} steps={STEPS} reason={BROAD_MARKDIST_TAIL_REORDER_DEFAULT_REASON}")

  with open(fname,"rb") as f:
    while True:
      chunk_constellations:List[Dict[str,int]]=[]
      i:int=0
      while i<STEPS:
        raw:str=f.read(16)
        if len(raw)<16:
          break
        ld:int=_read_uint32_le(raw[0:4])
        rd:int=_read_uint32_le(raw[4:8])
        col:int=_read_uint32_le(raw[8:12])
        startijkl:int=_read_uint32_le(raw[12:16])
        chunk_constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
        i+=1

      m:int=len(chunk_constellations)
      if m==0:
        break

      soa:TaskSoA=TaskSoA(m)
      w_arr:List[u64]=[u64(0)]*m
      build_soa_for_range(N,chunk_constellations,0,m,soa,w_arr)

      buckets:List[List[Dict[str,int]]]=make_broad_markdist_tail_subcell_buffers()
      i=0
      while i<m:
        fid:int=soa.funcid_arr[i]
        broad=funcid_reorder_bucket(fid)
        vals:List[int]=funcid_mark_effective_values_from_soa(soa,i)
        risk=funcid_markdist_risk_bucket(fid,vals[2],vals[3])
        if broad<0 or broad>4:
          broad=4
        if risk<0 or risk>4:
          risk=4
        tail=broad_markdist_tail_bucket(fid,broad,risk)
        subcell=broad_markdist_tail_subcell_index(broad,risk,tail)
        buckets[subcell].append(chunk_constellations[i])
        counts[subcell]+=1
        i+=1

      subcell=0
      while subcell<75:
        if len(buckets[subcell])>0:
          cell=subcell//3
          broad=cell//5
          risk=cell%5
          tail=subcell%3
          append_constellations_bin(broad_markdist_tail_reorder_subcell_fname(N,preset_queens,broad,risk,tail),buckets[subcell])
        subcell+=1

      if gpu_log_level>=2:
        tail_stats:List[int]=analyze_broad_markdist_tail_summary_from_soa(soa,m)
        print(f"[broadmarktail-reorder-subcell-chunk] chunk={chunk_index} m={m} funcid17={tail_stats[0]} cell_G_H={tail_stats[1]} markrisk_H={tail_stats[2]} proxy_avg={format_ratio_3(tail_stats[3],m)}")
      chunk_index+=1

  if gpu_log_level>=1:
    total:int=0
    subcell=0
    while subcell<75:
      total+=counts[subcell]
      subcell+=1
    print(f"[broadmarktail-reorder-subcell-summary] N={N} records={total} funcid17={counts[broad_markdist_tail_subcell_index(3,0,0)]+counts[broad_markdist_tail_subcell_index(3,1,0)]+counts[broad_markdist_tail_subcell_index(3,2,0)]+counts[broad_markdist_tail_subcell_index(3,3,0)]+counts[broad_markdist_tail_subcell_index(3,4,0)]} cell_G_H={counts[broad_markdist_tail_subcell_index(3,2,0)]+counts[broad_markdist_tail_subcell_index(3,2,1)]+counts[broad_markdist_tail_subcell_index(3,2,2)]}")

  return counts

def broad_markdist_tail_make_subcell_quotas(rem_counts:List[int],total_remaining:int,m_target:int)->List[int]:
  quotas:List[int]=[0]*75
  if total_remaining<=0 or m_target<=0:
    return quotas

  cell_rem:List[int]=[0]*25
  subcell:int=0
  while subcell<75:
    cell:int=subcell//3
    cell_rem[cell]+=rem_counts[subcell]
    subcell+=1

  cell_quotas:List[int]=broad_markdist_make_cell_quotas(cell_rem,total_remaining,m_target)

  cell=0
  while cell<25:
    cq:int=cell_quotas[cell]
    if cq>0 and cell_rem[cell]>0:
      sub_rem:List[int]=[0]*5
      sub_rem[0]=rem_counts[cell*3]
      sub_rem[1]=rem_counts[cell*3+1]
      sub_rem[2]=rem_counts[cell*3+2]
      sub_q:List[int]=funcid_reorder_make_quotas(sub_rem,cell_rem[cell],cq)
      quotas[cell*3]=sub_q[0]
      quotas[cell*3+1]=sub_q[1]
      quotas[cell*3+2]=sub_q[2]
    cell+=1

  qsum:int=0
  subcell=0
  while subcell<75:
    qsum+=quotas[subcell]
    subcell+=1
  while qsum<m_target:
    best:int=-1
    best_room:int=-1
    subcell=0
    while subcell<75:
      room:int=rem_counts[subcell]-quotas[subcell]
      if room>best_room:
        best_room=room
        best=subcell
      subcell+=1
    if best<0 or best_room<=0:
      break
    quotas[best]+=1
    qsum+=1

  return quotas

def interleave_broad_markdist_tail_subparts(part_f17:List[Dict[str,int]],part_gh:List[Dict[str,int]],part_r:List[Dict[str,int]],m_target:int,cell:int,chunk_index:int)->List[Dict[str,int]]:
  out:List[Dict[str,int]]=[]
  i17:int=0
  ig:int=0
  ir:int=0
  phase:int=0
  if broad_markdist_tail_use_rotating_interleave():
    phase=(chunk_index+cell)%4
  while len(out)<m_target:
    progressed:bool=False

    if phase==0:
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True
      if i17<len(part_f17) and len(out)<m_target:
        out.append(part_f17[i17])
        i17+=1
        progressed=True
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True
      if ig<len(part_gh) and len(out)<m_target:
        out.append(part_gh[ig])
        ig+=1
        progressed=True
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True
    elif phase==1:
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True
      if ig<len(part_gh) and len(out)<m_target:
        out.append(part_gh[ig])
        ig+=1
        progressed=True
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True
      if i17<len(part_f17) and len(out)<m_target:
        out.append(part_f17[i17])
        i17+=1
        progressed=True
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True
    elif phase==2:
      if i17<len(part_f17) and len(out)<m_target:
        out.append(part_f17[i17])
        i17+=1
        progressed=True
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True
      if ig<len(part_gh) and len(out)<m_target:
        out.append(part_gh[ig])
        ig+=1
        progressed=True
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True
    else:
      if ig<len(part_gh) and len(out)<m_target:
        out.append(part_gh[ig])
        ig+=1
        progressed=True
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True
      if i17<len(part_f17) and len(out)<m_target:
        out.append(part_f17[i17])
        i17+=1
        progressed=True
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True
      if ir<len(part_r) and len(out)<m_target:
        out.append(part_r[ir])
        ir+=1
        progressed=True

    if not progressed:
      break
  return out

def build_broad_markdist_tail_reordered_bin(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1
)->Tuple[str,int,int]:

  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  total_records:int=count_constellations_bin_records(fname)
  counts:List[int]=build_broad_markdist_tail_reorder_subcell_bins(N,fname,preset_queens,BLOCK,MAX_BLOCKS,gpu_log_level)
  counted_records:int=0
  subcell:int=0
  while subcell<75:
    counted_records+=counts[subcell]
    subcell+=1
  if counted_records!=total_records:
    print(f"[broadmarktail-reorder-warning] subcell count mismatch: counted={counted_records} total_records={total_records}")
    total_records=counted_records

  reorder_fname:str=broad_markdist_tail_reorder_output_fname(N,preset_queens,BLOCK,MAX_BLOCKS)
  truncate_plain_bin(reorder_fname)

  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_broadmarktail_reorder_{BROAD_MARKDIST_TAIL_REORDER_VERSION}_{broad_markdist_tail_variant_tag()}_{funcid_reorder_run_param_tag(BLOCK,MAX_BLOCKS)}_sim.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_broad_markdist_tail_reorder_progress_header())

  rem_counts:List[int]=[0]*75
  read_offsets:List[int]=[0]*75
  subcell_buffers:List[List[Dict[str,int]]]=make_broad_markdist_tail_subcell_buffers()
  subcell=0
  while subcell<75:
    rem_counts[subcell]=counts[subcell]
    read_offsets[subcell]=0
    subcell+=1

  off:int=0
  chunk_index:int=0
  total_remaining:int=total_records

  if gpu_log_level>=1:
    print(f"[broadmarktail-reorder-build-config] N={N} records={total_records} steps={STEPS} output={reorder_fname} progress={progress_fname} param={funcid_reorder_run_param_tag(BLOCK,MAX_BLOCKS)} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} weak_tail_window_boost={broad_markdist_tail_window_boost_value()} tail_variant={broad_markdist_tail_variant_tag()} reason={BROAD_MARKDIST_TAIL_REORDER_DEFAULT_REASON}")

  while total_remaining>0:
    m_target:int=STEPS
    if total_remaining<STEPS:
      m_target=total_remaining

    quotas:List[int]=broad_markdist_tail_make_subcell_quotas(rem_counts,total_remaining,m_target)
    cell_quotas:List[int]=[0]*25
    broad_quotas:List[int]=[0]*5
    subcell=0
    while subcell<75:
      cell:int=subcell//3
      broad:int=cell//5
      cell_quotas[cell]+=quotas[subcell]
      broad_quotas[broad]+=quotas[subcell]
      subcell+=1

    t0:datetime=datetime.now()

    parts:List[List[Dict[str,int]]]=make_broad_markdist_tail_subcell_buffers()
    subcell=0
    while subcell<75:
      q:int=quotas[subcell]
      if q>0:
        cell=subcell//3
        broad=cell//5
        risk:int=cell%5
        tail:int=subcell%3
        target:int=q*FUNCID_REORDER_V2_WINDOW_MULT
        if tail==0 or tail==1:
          target=target*broad_markdist_tail_window_boost_value()
        if target<q:
          target=q
        if target>rem_counts[subcell]:
          target=rem_counts[subcell]
        fname_subcell:str=broad_markdist_tail_reorder_subcell_fname(N,preset_queens,broad,risk,tail)
        newbuf:List[Dict[str,int]]=[]
        subcell_buffers[subcell],read_offsets[subcell]=fill_constellation_buffer_from_bin_range(fname_subcell,subcell_buffers[subcell],read_offsets[subcell],target)
        group_id:int=broad_markdist_tail_phase_group_id(subcell,broad,risk,tail)
        parts[subcell],newbuf=take_striped_records_from_buffer(subcell_buffers[subcell],q,chunk_index,group_id)
        subcell_buffers[subcell]=newbuf
        rem_counts[subcell]-=len(parts[subcell])
        if rem_counts[subcell]<0:
          rem_counts[subcell]=0
      subcell+=1

    cell_parts:List[List[Dict[str,int]]]=make_broad_markdist_cell_buffers()
    cell=0
    while cell<25:
      qcell:int=cell_quotas[cell]
      cell_parts[cell]=interleave_broad_markdist_tail_subparts(parts[cell*3],parts[cell*3+1],parts[cell*3+2],qcell,cell,chunk_index)
      cell+=1

    chunk_constellations:List[Dict[str,int]]=interleave_broad_markdist_secondary_parts(cell_parts,broad_quotas,m_target)
    m:int=len(chunk_constellations)
    if m==0:
      break

    append_constellations_bin(reorder_fname,chunk_constellations)
    stats:List[int]=analyze_stream_chunk_input_stats(N,chunk_constellations)
    risk_stats:List[int]=analyze_markdist_risk_stats(N,chunk_constellations)
    cell_stats:List[int]=analyze_broad_markdist_cell_stats(N,chunk_constellations)
    tail_stats:List[int]=analyze_broad_markdist_tail_summary(N,chunk_constellations)
    t1:datetime=datetime.now()
    elapsed_text:str=str(t1-t0)[:-3]
    elapsed_ms:int=stream_elapsed_text_to_ms(elapsed_text)
    append_stream_broad_markdist_tail_reorder_progress(progress_fname,N,preset_queens,chunk_index,off,m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_text,elapsed_ms,total_records,stats,risk_stats,cell_stats,tail_stats)

    if gpu_log_level>=2:
      print(f"[broadmarktail-reorder-build-chunk] chunk={chunk_index} off={off} m={m} B={stats[18+26]+stats[18+27]} A={stats[18+19]+stats[18+22]+stats[18+23]+stats[18+24]} C={stats[18+20]+stats[18+21]} G={stats[18+0]+stats[18+4]+stats[18+5]+stats[18+12]+stats[18+16]+stats[18+17]+stats[18+18]} O={m-(stats[18+26]+stats[18+27]+stats[18+19]+stats[18+22]+stats[18+23]+stats[18+24]+stats[18+20]+stats[18+21]+stats[18+0]+stats[18+4]+stats[18+5]+stats[18+12]+stats[18+16]+stats[18+17]+stats[18+18])} X={risk_stats[0]} T={risk_stats[1]} H={risk_stats[2]} M={risk_stats[3]} R={risk_stats[4]} funcid17={tail_stats[0]} cell_G_H={tail_stats[1]} tail_proxy_avg={format_ratio_3(tail_stats[3],m)}")

    off+=m
    chunk_index+=1
    total_remaining=total_records-off

  write_stream_done_count(reorder_fname+".done",off)
  reordered_records:int=count_constellations_bin_records(reorder_fname)
  if gpu_log_level>=1:
    print(f"[broadmarktail-reorder-build-summary] N={N} records={reordered_records} chunks={chunk_index} output={reorder_fname} progress={progress_fname} valid={1 if validate_bin_file(reorder_fname) else 0}")

  return reorder_fname,reordered_records,chunk_index

def int_to_le_bytes8(x:int)->List[int]:
  return [(x>>(8*i))&0xFF for i in range(8)]

def write_features_chunk_cpu(fname:str,soa:TaskSoA,m:int,global_base:int)->int:
  # 354: dumps the six RAW fields that chunkshape148_score_key_from_soa reads
  # (fid, free, end, row, mark1, mark2), plus the ACTUAL production key value
  # obtained by calling that function verbatim -- not a reimplementation. depth,
  # popcount, mark_gap, row_to_end and row_to_mark1 are all derivable from the
  # six raw fields offline; storing the raw fields instead of the derived ones
  # means any bug in a candidate derivation is caught by comparison against the
  # real key column rather than silently baked into the dump.
  # Record layout, 32 bytes, little endian:
  #   fid:i32  free:u32  end:i32  row:i32  mark1:i32  mark2:i32  key:i64
  wrote:int=0
  buf:List[str]=[]
  with open(fname,"wb") as f:
    i:int=0
    while i<m:
      key:int=chunkshape148_score_key_from_soa(soa,i,global_base+i)
      if key<0:
        key=0
      if key>CHUNKSHAPE148_SCORE_KEY_MAX:
        key=CHUNKSHAPE148_SCORE_KEY_MAX
      rec:List[int]=[]
      rec+=int_to_le_bytes(int(soa.funcid_arr[i]))
      rec+=int_to_le_bytes(int(soa.free_arr[i]))
      rec+=int_to_le_bytes(int(soa.end_arr[i]))
      rec+=int_to_le_bytes(int(soa.row_arr[i]))
      rec+=int_to_le_bytes(int(soa.mark1_arr[i]))
      rec+=int_to_le_bytes(int(soa.mark2_arr[i]))
      rec+=int_to_le_bytes8(key)
      buf.append("".join(chr(c) for c in rec))
      wrote+=1
      i+=1
      if len(buf)>=65536:
        f.write("".join(buf))
        buf=[]
    if len(buf)>0:
      f.write("".join(buf))
  return wrote

def exec_features_cpu_bin_stream_split145(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  chunk_only:bool=False,
  debug_chunk_start:int=0,
  debug_chunk_count:int=0,  # 354: 0 means ALL chunks. Unlike bucket_run/iter_sort
                             # style knobs this is not clamped to 1 -- a feature
                             # census with no counterpart to "run all N=21" would
                             # be a smoke test wearing a full-run's clothes.
  chunk_list_spec:str=""
)->int:
  # 354: CPU ONLY. No GPU kernel launch anywhere in this function or in anything
  # it calls. build_soa_for_range is the exact production host-side function
  # that populates the fields chunkshape148_score_key_from_soa reads; calling it
  # here on the shaped bin's own final order costs about 110ms per 743424-record
  # chunk (measured inside exec_solutions_gpu_chunk_split145's soa_ms stage on
  # this same workload), so the whole N=21 population processes in low seconds
  # with zero GPU time.
  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS*K_PER_THREAD_MAXD14
  if STEPS<=0:
    STEPS=15488*K_PER_THREAD_MAXD14
  run_param_tag:str=funcid_reorder_run_param_tag(BLOCK,MAX_BLOCKS)

  selected_chunks:List[int]=parse_chunk_list_spec(chunk_list_spec)
  use_chunk_list:bool=(len(selected_chunks)>0)
  if chunk_only:
    if debug_chunk_start<0:
      debug_chunk_start=0
    if debug_chunk_count<=0:
      debug_chunk_count=1
  stop_after_chunk:int=-1
  if use_chunk_list:
    stop_after_chunk=chunk_list_max(selected_chunks)
  elif chunk_only:
    stop_after_chunk=debug_chunk_start+debug_chunk_count-1

  total_records:int=count_constellations_bin_records(fname)
  if gpu_log_level>=1:
    print(f"[feature354-config] N={N} records={total_records} bin={fname} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} chunk_only={1 if (chunk_only or use_chunk_list) else 0} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={chunk_list_to_string(selected_chunks)}")

  processed:int=0
  off:int=0
  chunk_index:int=0
  _read_uint32_le=read_uint32_le
  with open(fname,"rb") as f:
    while True:
      if stop_after_chunk>=0 and chunk_index>stop_after_chunk:
        break
      chunk_constellations:List[Dict[str,int]]=[]
      i:int=0
      while i<STEPS:
        raw:str=f.read(16)
        if len(raw)<16:
          break
        ld:int=_read_uint32_le(raw[0:4])
        rd:int=_read_uint32_le(raw[4:8])
        col:int=_read_uint32_le(raw[8:12])
        startijkl:int=_read_uint32_le(raw[12:16])
        chunk_constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
        i+=1
      m:int=len(chunk_constellations)
      if m==0:
        break
      run_this_chunk:bool=True
      if use_chunk_list:
        run_this_chunk=chunk_list_contains(selected_chunks,chunk_index)
      elif chunk_only:
        run_this_chunk=(chunk_index>=debug_chunk_start and chunk_index<debug_chunk_start+debug_chunk_count)
      if not run_this_chunk:
        if gpu_log_level>=2:
          print(f"[feature354-chunk-skip] N={N} chunk={chunk_index} off={off} m={m}")
        off+=m; chunk_index+=1; continue

      t0=datetime.now()
      soa:TaskSoA=TaskSoA(m)
      w_arr:List[u64]=[u64(0)]*m
      build_soa_for_range(N,chunk_constellations,0,m,soa,w_arr)
      features_fname:str=f"features354_N{N}_{preset_queens}_{run_param_tag}_k{K_PER_THREAD_MAXD14}_chunk{chunk_index}.bin"
      written:int=write_features_chunk_cpu(features_fname,soa,m,off)
      processed+=written
      t1=datetime.now()
      elapsed_text:str=str(t1-t0)[:-3]
      if gpu_log_level>=1:
        print(f"[feature354-chunk-done] N={N} chunk={chunk_index} off={off} m={m} elapsed={elapsed_text} records={written} file={features_fname}")
      off+=m
      chunk_index+=1
  if gpu_log_level>=1:
    print(f"[feature354-summary] N={N} records={total_records} chunks={chunk_index} processed={processed}")
  return processed

def exec_solutions_gpu_bin_stream_funcid_reorder(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1,
  cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT,
  chunk_only:bool=False,
  debug_chunk_start:int=0,
  debug_chunk_count:int=1,
  chunk_list_spec:str="",
  progress_suffix:str="",
  worker_id:int=0,
  worker_count:int=1
)->int:

  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  if worker_count<=0:
    print(f"[worker-warning] invalid worker_count={worker_count}; using 1")
    worker_count=1
  if worker_id<0:
    print(f"[worker-warning] invalid worker_id={worker_id}; using 0")
    worker_id=0
  if worker_id>=worker_count:
    print(f"[worker-error] worker_id must be smaller than worker_count: worker_id={worker_id} worker_count={worker_count}")
    return 0

  total_records:int=count_constellations_bin_records(fname)
  run_param_tag:str=funcid_reorder_run_param_tag(BLOCK,MAX_BLOCKS)
  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_funcid_reorder_v2_{run_param_tag}.tsv"
  if progress_suffix!="":
    progress_fname=f"progress_N{N}_{preset_queens}_stream_funcid_reorder_v2_{run_param_tag}_{progress_suffix}.tsv"
  if worker_count>1:
    if progress_suffix!="":
      progress_fname=f"progress_N{N}_{preset_queens}_stream_funcid_reorder_v2_{run_param_tag}_{progress_suffix}_worker{worker_id}of{worker_count}.tsv"
    else:
      progress_fname=f"progress_N{N}_{preset_queens}_stream_funcid_reorder_v2_{run_param_tag}_worker{worker_id}of{worker_count}.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_funcid_reorder_progress_header())

  selected_chunks:List[int]=parse_chunk_list_spec(chunk_list_spec)
  use_chunk_list:bool=(len(selected_chunks)>0)
  if chunk_only:
    if debug_chunk_start<0:
      debug_chunk_start=0
    if debug_chunk_count<=0:
      debug_chunk_count=1

  stop_after_chunk:int=-1
  if use_chunk_list:
    stop_after_chunk=chunk_list_max(selected_chunks)
  elif chunk_only:
    stop_after_chunk=debug_chunk_start+debug_chunk_count-1

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-gpu-config] N={N} records={total_records} bin={fname} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} sort_mode={gpu_sort_mode} chunk_only={1 if (chunk_only or use_chunk_list) else 0} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={chunk_list_to_string(selected_chunks)} worker={worker_id}/{worker_count} progress={progress_fname} param={funcid_reorder_param_tag()} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} inner_log_level=0")

  gpu_total:int=0
  off:int=0
  chunk_index:int=0
  executed_chunks:int=0
  _read_uint32_le=read_uint32_le

  with open(fname,"rb") as f:
    while True:
      if stop_after_chunk>=0 and chunk_index>stop_after_chunk:
        break
      chunk_constellations:List[Dict[str,int]]=[]
      i:int=0
      while i<STEPS:
        raw:str=f.read(16)
        if len(raw)<16:
          break
        ld:int=_read_uint32_le(raw[0:4])
        rd:int=_read_uint32_le(raw[4:8])
        col:int=_read_uint32_le(raw[8:12])
        startijkl:int=_read_uint32_le(raw[12:16])
        chunk_constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
        i+=1

      m:int=len(chunk_constellations)
      if m==0:
        break

      if chunk_only or use_chunk_list:
        run_this_chunk:bool=True
        if use_chunk_list:
          run_this_chunk=chunk_list_contains(selected_chunks,chunk_index)
        else:
          run_this_chunk=(chunk_index>=debug_chunk_start and chunk_index<debug_chunk_start+debug_chunk_count)
        if not run_this_chunk:
          if gpu_log_level>=2:
            print(f"[funcid-reorder-v2-gpu-chunk-skip] N={N} chunk={chunk_index} off={off} m={m}")
          off+=m
          chunk_index+=1
          continue

      if worker_count>1:
        run_worker_chunk:bool=((chunk_index % worker_count)==worker_id)
        if not run_worker_chunk:
          if gpu_log_level>=2:
            print(f"[funcid-reorder-v2-gpu-worker-skip] N={N} worker={worker_id}/{worker_count} chunk={chunk_index} off={off} m={m}")
          off+=m
          chunk_index+=1
          continue

      stats:List[int]=analyze_stream_chunk_input_stats(N,chunk_constellations)

      t0=datetime.now()
      if gpu_log_level>=1:
        print(f"[funcid-reorder-v2-gpu-chunk-start] N={N} worker={worker_id}/{worker_count} chunk={chunk_index} off={off} m={m}")

      inner_gpu_log_level:int=0
      exec_solutions(N,chunk_constellations,True,gpu_block,gpu_max_blocks,inner_gpu_log_level,gpu_sort_mode,cross_stripe_safe)

      chunk_total:int=0
      if m>0:
        chunk_total=chunk_constellations[0]["solutions"]
      gpu_total+=chunk_total
      executed_chunks+=1
      t1=datetime.now()
      elapsed_text:str=str(t1-t0)[:-3]
      elapsed_ms:int=stream_elapsed_text_to_ms(elapsed_text)
      append_stream_funcid_reorder_progress(progress_fname,N,preset_queens,chunk_index,off,m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_text,elapsed_ms,chunk_total,gpu_total,total_records,stats)
      if gpu_log_level>=1:
        print(f"[funcid-reorder-v2-gpu-chunk-end] N={N} worker={worker_id}/{worker_count} chunk={chunk_index} off={off} m={m} elapsed={elapsed_text} elapsed_ms={elapsed_ms} chunk_total={chunk_total} gpu_total={gpu_total}")

      off+=m
      chunk_index+=1

  if gpu_log_level>=1:
    print(f"[funcid-reorder-v2-gpu-summary] N={N} records={total_records} chunks={chunk_index} executed_chunks={executed_chunks} total={gpu_total} progress={progress_fname}")
    if worker_count>1:
      print(f"[worker-summary] N={N} worker={worker_id}/{worker_count} records={total_records} chunks={chunk_index} executed_chunks={executed_chunks} partial_total={gpu_total} progress={progress_fname}")

  return gpu_total

def exec_solutions_gpu_chunk_split145(
  N:int,
  chunk_constellations:List[Dict[str,int]],
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_sort_mode:int=-1,
  cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT,
  split_mode:int=0,
  gpu_log_level:int=0,
  k_per_thread:int=1  # 292: constellations per GPU thread for the maxd14 grid-stride kernel
)->Tuple[int,List[int],List[int],str,int]:
  board_mask:int=(1<<N)-1

  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  if k_per_thread<=0:
    k_per_thread=1
  THREADS:int=BLOCK*MAX_BLOCKS
  if THREADS<=0:
    THREADS=15488
  STEPS:int=THREADS*k_per_thread

  m:int=len(chunk_constellations)
  soa:TaskSoA=TaskSoA(STEPS)
  w_arr:List[u64]=[u64(0)]*STEPS
  results:List[u64]=[u64(0)]*STEPS

  t0=datetime.now()
  build_soa_for_range(N,chunk_constellations,0,m,soa,w_arr)
  t1=datetime.now()
  stats:List[int]=analyze_stream_chunk_input_stats_from_soa(soa,w_arr,m)
  t2=datetime.now()

  d2base14_m:int=0
  d0_m:int=0
  i:int=0
  while i<m:
    fid:int=soa.funcid_arr[i]
    if fid==14:
      d2base14_m+=1
    if fid==26 or fid==27:
      d0_m+=1
    i+=1

  t3=datetime.now()

  board_mask_gpu:u32=u32(board_mask)
  n3_gpu:u32=u32(1)<<u32(N-3)
  n4_gpu:u32=u32(1)<<u32(N-4)
  chunk_total:int=0

  if gpu_log_level>=1:
    print(f"[split145-buckets] N={N} m={m} generic={m} d2base14={d2base14_m} d0={d0_m} rest=0 split_mode={split_mode} specialized=0")

  meta_next:List[u8]=[u8(1),u8(2),u8(3),u8(3),u8(2),u8(6),u8(2),u8(2),u8(0),u8(4),u8(5),u8(7),u8(13),u8(14),u8(14),u8(14),u8(17),u8(14),u8(14),u8(20),u8(21),u8(21),u8(21),u8(25),u8(21),u8(21),u8(26),u8(26)]

  if m>0:
    required_maxd:int=max_schedule_depth_of_tasks(soa,m,meta_next)
    selected_maxd:int=select_static_maxd(required_maxd)
    if gpu_log_level>=1:
      print(f"[maxd-dispatch] N={N} scope=split145 m={m} required_maxd={required_maxd} selected_MAXD={selected_maxd} schedule_words={packed_schedule_words_for_maxd(selected_maxd)} stack_bytes_per_thread={packed_stack_bytes_per_thread(selected_maxd)} capacity_check=OK")
    # 292: only kernel_dfs_iter_gpu_maxd14 supports the grid-stride K-batch
    # loop. When a chunk needs deeper schedule (rare; selected_maxd>14),
    # fall back to the original 1-task-per-thread launch so those kernels
    # (unmodified) keep covering every task in [0,m) correctly.
    GRID:int=0
    kbatch_stride:int=0
    if selected_maxd==14:
      GRID=MAX_BLOCKS
      kbatch_stride=THREADS
    else:
      GRID=(m+BLOCK-1)//BLOCK
      kbatch_stride=0
    if not launch_kernel_dfs_iter_gpu_static_maxd(selected_maxd,soa,w_arr,meta_next,results,m,board_mask_gpu,n3_gpu,n4_gpu,GRID,BLOCK,kbatch_stride):
      print(f"[maxd-error] unsupported required_maxd={required_maxd}; supported maximum is 21")
      error_stages:List[int]=[0]*10
      return 0,stats,error_stages,"0:00:00.000",0
    sum_count:int=kbatch_stride if selected_maxd==14 else m
    i=0
    while i<sum_count:
      chunk_total+=int(results[i])
      i+=1

  t4=datetime.now()
  stage_soa_ms:int=profile_elapsed_ms_between(t0,t1)
  stage_stats_ms:int=profile_elapsed_ms_between(t1,t2)
  stage_split_ms:int=profile_elapsed_ms_between(t2,t3)
  stage_kernel_reduce_ms:int=profile_elapsed_ms_between(t3,t4)
  stage_compute_ms:int=stage_soa_ms+stage_split_ms+stage_kernel_reduce_ms
  stage_no_read_ms:int=stage_compute_ms+stage_stats_ms
  elapsed_text:str=str(t4-t0)[:-3]
  elapsed_ms:int=stage_no_read_ms
  stages:List[int]=[stage_soa_ms,stage_stats_ms,stage_split_ms,stage_kernel_reduce_ms,stage_compute_ms,stage_no_read_ms,0,d0_m,d2base14_m,m]
  return chunk_total,stats,stages,elapsed_text,elapsed_ms

def exec_solutions_gpu_bin_stream_split145(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1,
  cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT,
  chunk_only:bool=False,
  debug_chunk_start:int=0,
  debug_chunk_count:int=1,
  chunk_list_spec:str="",
  progress_suffix:str="split145",
  worker_id:int=0,
  worker_count:int=1,
  split_mode:int=2,
  k_per_thread:int=1  # 292: constellations per GPU thread (grid-stride K-batch); 1 = original 291 behavior
)->int:
  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  if k_per_thread<=0:
    k_per_thread=1
  STEPS:int=BLOCK*MAX_BLOCKS*k_per_thread
  if STEPS<=0:
    STEPS=15488
  if worker_count<=0:
    print(f"[worker-warning] invalid worker_count={worker_count}; using 1")
    worker_count=1
  if worker_id<0:
    print(f"[worker-warning] invalid worker_id={worker_id}; using 0")
    worker_id=0
  if worker_id>=worker_count:
    print(f"[worker-error] worker_id must be smaller than worker_count: worker_id={worker_id} worker_count={worker_count}")
    return 0

  total_records:int=count_constellations_bin_records(fname)
  run_param_tag:str=funcid_reorder_run_param_tag(BLOCK,MAX_BLOCKS)
  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_split145_{run_param_tag}.tsv"
  if progress_suffix!="":
    progress_fname=f"progress_N{N}_{preset_queens}_stream_split145_{run_param_tag}_{progress_suffix}.tsv"
  if worker_count>1:
    if progress_suffix!="":
      progress_fname=f"progress_N{N}_{preset_queens}_stream_split145_{run_param_tag}_{progress_suffix}_worker{worker_id}of{worker_count}.tsv"
    else:
      progress_fname=f"progress_N{N}_{preset_queens}_stream_split145_{run_param_tag}_worker{worker_id}of{worker_count}.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_funcid_reorder_progress_header())

  selected_chunks:List[int]=parse_chunk_list_spec(chunk_list_spec)
  use_chunk_list:bool=(len(selected_chunks)>0)
  if chunk_only:
    if debug_chunk_start<0:
      debug_chunk_start=0
    if debug_chunk_count<=0:
      debug_chunk_count=1
  stop_after_chunk:int=-1
  if use_chunk_list:
    stop_after_chunk=chunk_list_max(selected_chunks)
  elif chunk_only:
    stop_after_chunk=debug_chunk_start+debug_chunk_count-1

  if gpu_log_level>=1:
    print(f"[split145-gpu-config] N={N} records={total_records} bin={fname} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} sort_mode={gpu_sort_mode} chunk_only={1 if (chunk_only or use_chunk_list) else 0} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={chunk_list_to_string(selected_chunks)} worker={worker_id}/{worker_count} split_mode={split_mode} progress={progress_fname}")

  gpu_total:int=0
  off:int=0
  chunk_index:int=0
  executed_chunks:int=0
  _read_uint32_le=read_uint32_le
  with open(fname,"rb") as f:
    while True:
      if stop_after_chunk>=0 and chunk_index>stop_after_chunk:
        break
      chunk_constellations:List[Dict[str,int]]=[]
      i:int=0
      while i<STEPS:
        raw:str=f.read(16)
        if len(raw)<16:
          break
        ld:int=_read_uint32_le(raw[0:4])
        rd:int=_read_uint32_le(raw[4:8])
        col:int=_read_uint32_le(raw[8:12])
        startijkl:int=_read_uint32_le(raw[12:16])
        chunk_constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
        i+=1
      m:int=len(chunk_constellations)
      if m==0:
        break
      if chunk_only or use_chunk_list:
        run_this_chunk:bool=True
        if use_chunk_list:
          run_this_chunk=chunk_list_contains(selected_chunks,chunk_index)
        else:
          run_this_chunk=(chunk_index>=debug_chunk_start and chunk_index<debug_chunk_start+debug_chunk_count)
        if not run_this_chunk:
          if gpu_log_level>=2:
            print(f"[split145-gpu-chunk-skip] N={N} chunk={chunk_index} off={off} m={m}")
          off+=m; chunk_index+=1; continue
      if worker_count>1:
        run_worker_chunk:bool=((chunk_index % worker_count)==worker_id)
        if not run_worker_chunk:
          if gpu_log_level>=2:
            print(f"[split145-gpu-worker-skip] N={N} worker={worker_id}/{worker_count} chunk={chunk_index} off={off} m={m}")
          off+=m; chunk_index+=1; continue

      t0=datetime.now()
      if gpu_log_level>=1:
        print(f"[split145-gpu-chunk-start] N={N} worker={worker_id}/{worker_count} chunk={chunk_index} off={off} m={m}")
      chunk_total:int=0
      stats:List[int]=[0]*46
      stages_inner:List[int]=[0,0,0,0,0,0,0]
      elapsed_text:str="0:00:00.000"
      elapsed_ms:int=0
      chunk_total,stats,stages_inner,elapsed_text,elapsed_ms=exec_solutions_gpu_chunk_split145(N,chunk_constellations,BLOCK,MAX_BLOCKS,gpu_sort_mode,cross_stripe_safe,split_mode,gpu_log_level,k_per_thread)
      gpu_total+=chunk_total
      executed_chunks+=1
      t1=datetime.now()
      elapsed_outer_text:str=str(t1-t0)[:-3]
      elapsed_outer_ms:int=stream_elapsed_text_to_ms(elapsed_outer_text)
      append_stream_funcid_reorder_progress(progress_fname,N,preset_queens,chunk_index,off,m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_outer_text,elapsed_outer_ms,chunk_total,gpu_total,total_records,stats)
      if gpu_log_level>=1:
        print(f"[split145-gpu-chunk-end] N={N} worker={worker_id}/{worker_count} chunk={chunk_index} off={off} m={m} elapsed={elapsed_outer_text} inner={elapsed_text} elapsed_ms={elapsed_outer_ms} chunk_total={chunk_total} gpu_total={gpu_total} soa_ms={stages_inner[0]} stats_ms={stages_inner[1]} split_ms={stages_inner[2]} kernel_reduce_ms={stages_inner[3]}")
      off+=m
      chunk_index+=1
  if gpu_log_level>=1:
    print(f"[split145-gpu-summary] N={N} records={total_records} chunks={chunk_index} executed_chunks={executed_chunks} total={gpu_total} split_mode={split_mode} progress={progress_fname}")
    if worker_count>1:
      print(f"[worker-summary] N={N} worker={worker_id}/{worker_count} records={total_records} chunks={chunk_index} executed_chunks={executed_chunks} partial_total={gpu_total} progress={progress_fname}")
  return gpu_total

def read_constellations_bin_range(fname:str,off_record:int,max_records:int)->List[Dict[str,int]]:
  out:List[Dict[str,int]]=[]
  if off_record<0:
    off_record=0
  if max_records<=0:
    return out
  _read_uint32_le=read_uint32_le
  with open(fname,"rb") as f:
    f.seek(off_record*16,0)
    i:int=0
    while i<max_records:
      raw:str=f.read(16)
      if len(raw)<16:
        break
      ld:int=_read_uint32_le(raw[0:4])
      rd:int=_read_uint32_le(raw[4:8])
      col:int=_read_uint32_le(raw[8:12])
      startijkl:int=_read_uint32_le(raw[12:16])
      out.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
      i+=1
  return out

def exec_solutions_gpu_bin_stream(
  N:int,
  fname:str,
  preset_queens:int,
  gpu_block:int=32,
  gpu_max_blocks:int=484,
  gpu_log_level:int=0,
  gpu_sort_mode:int=-1,
  cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT,
  chunk_only:bool=False,
  debug_chunk_start:int=0,
  debug_chunk_count:int=1
)->int:

  BLOCK:int=gpu_block
  MAX_BLOCKS:int=gpu_max_blocks
  if BLOCK<=0:
    BLOCK=32
  if MAX_BLOCKS<=0:
    MAX_BLOCKS=484
  STEPS:int=BLOCK*MAX_BLOCKS
  if STEPS<=0:
    STEPS=15488

  total_records:int=count_constellations_bin_records(fname)
  progress_fname:str=f"progress_N{N}_{preset_queens}_stream_measure2.tsv"
  with open(progress_fname,"w") as pf:
    pf.write(stream_measure2_progress_header())

  if chunk_only:
    if debug_chunk_start<0:
      debug_chunk_start=0
    if debug_chunk_count<=0:
      debug_chunk_count=1

  if gpu_log_level>=1:
    print(f"[stream-gpu-config] N={N} records={total_records} bin={fname} block={BLOCK} max_blocks={MAX_BLOCKS} steps={STEPS} sort_mode={gpu_sort_mode} chunk_only={1 if chunk_only else 0} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} progress={progress_fname} inner_log_level=0 measure2=1")

  gpu_total:int=0
  off:int=0
  chunk_index:int=0
  executed_chunks:int=0
  _read_uint32_le=read_uint32_le

  with open(fname,"rb") as f:
    while True:
      chunk_constellations:List[Dict[str,int]]=[]
      i:int=0
      while i<STEPS:
        raw:str=f.read(16)
        if len(raw)<16:
          break
        ld=_read_uint32_le(raw[0:4])
        rd=_read_uint32_le(raw[4:8])
        col=_read_uint32_le(raw[8:12])
        startijkl=_read_uint32_le(raw[12:16])
        chunk_constellations.append({"ld":ld,"rd":rd,"col":col,"startijkl":startijkl,"solutions":0})
        i+=1

      m:int=len(chunk_constellations)
      if m==0:
        break

      if chunk_only:
        run_this_chunk:bool=(chunk_index>=debug_chunk_start and chunk_index<debug_chunk_start+debug_chunk_count)
        if not run_this_chunk:
          if gpu_log_level>=2:
            print(f"[stream-gpu-chunk-skip] N={N} chunk={chunk_index} off={off} m={m}")
          off+=m
          chunk_index+=1
          continue

      stats:List[int]=analyze_stream_chunk_input_stats(N,chunk_constellations)

      t0=datetime.now()
      if gpu_log_level>=1:
        print(f"[stream-gpu-chunk-start] N={N} chunk={chunk_index} off={off} m={m}")

      inner_gpu_log_level:int=0
      exec_solutions(N,chunk_constellations,True,gpu_block,gpu_max_blocks,inner_gpu_log_level,gpu_sort_mode,cross_stripe_safe)

      chunk_total:int=0
      if m>0:
        chunk_total=chunk_constellations[0]["solutions"]
      gpu_total+=chunk_total
      executed_chunks+=1
      t1=datetime.now()
      elapsed_text:str=str(t1-t0)[:-3]
      elapsed_ms:int=stream_elapsed_text_to_ms(elapsed_text)
      append_stream_progress(progress_fname,N,preset_queens,chunk_index,off,m,BLOCK,MAX_BLOCKS,STEPS,gpu_sort_mode,elapsed_text,elapsed_ms,chunk_total,gpu_total,total_records,stats)
      if gpu_log_level>=1:
        print(f"[stream-gpu-chunk-end] N={N} chunk={chunk_index} off={off} m={m} elapsed={elapsed_text} elapsed_ms={elapsed_ms} chunk_total={chunk_total} gpu_total={gpu_total}")

      off+=m
      chunk_index+=1

  if gpu_log_level>=1:
    print(f"[stream-gpu-summary] N={N} records={total_records} chunks={chunk_index} executed_chunks={executed_chunks} total={gpu_total} progress={progress_fname} measure2=1")

  return gpu_total

def select_dynamic_preset_queens(N:int,preset_queens:int)->int:
  if N>=5 and N<=17:
    return 5
  elif N>=18 and N<=21:
    return 6
  elif N>=22 and N<=24:
    return 7
  elif N>=25 and N<=27:
    return 8
  return preset_queens

def _bit_total(N:int)->int:
  mask:int=(1<<N)-1
  def bt(row:int,left:int,down:int,right:int)->int:
    if row==N:
      return 1
    total:int=0
    bitmap:int=mask&~(left|down|right)
    while bitmap:
      bit:int=-bitmap&bitmap
      bitmap^=bit
      total+=bt(row+1,(left|bit)<<1,down|bit,(right|bit)>>1)
    return total
  return bt(0,0,0,0)

def main()->None:
  global FUNCID_REORDER_V2_WINDOW_MULT,FUNCID_REORDER_V2_PHASE_JUMP
  global BROAD_MARKDIST_TAIL_VARIANT
  global CHUNKSHAPE148_BUCKET_RUN
  global CHUNKSHAPE148_ITER_SORT

  expected:List[int]=[0,0,0,0,0,10,4,40,92,352,724,2680,14200,73712,365596,2279184,14772512,95815104,666090624,4968057848,39029188884,314666222712,2691008701644,24233937684440,227514171973736,2207893435808352,22317699616364044,234907967154122528]
  nmin:int=DEFAULT_RANGE_NMIN
  nmax:int=DEFAULT_RANGE_NMAX_EXCLUSIVE
  use_gpu:bool=False
  gpu_block:int=32
  gpu_max_blocks:int=484
  gpu_log_level:int=0
  gpu_sort_mode:int=-1
  cross_stripe_safe:bool=CROSS_STRIPE_SAFE_DEFAULT
  debug_chunk_start:int=0
  debug_chunk_count:int=1
  split_probe_chunk_list_spec:str=""
  bench_mode:int=0  # kept modes: 0 normal, 11 stream-bin-build-only, 28 broadmarktail sim, 29 broadmarktail gpu, 30 split145 probe/cache-build, 31 split145 full gpu
  reorder_window_mult:int=FUNCID_REORDER_V2_WINDOW_MULT
  reorder_phase_jump:int=FUNCID_REORDER_V2_PHASE_JUMP
  worker_id:int=0
  worker_count:int=1
  broadmark_tail_variant:int=BROAD_MARKDIST_TAIL_VARIANT
  chunkshape148_bucket_run:int=CHUNKSHAPE148_BUCKET_RUN
  chunkshape148_iter_sort:int=CHUNKSHAPE148_ITER_SORT
  preset_queens_arg:int=5
  requested_preset_arg:int=5
  argc:int=len(sys.argv)

  if argc == 1:
    use_gpu=False
    nmin=CPU_FINAL_DEFAULT_N
    nmax=CPU_FINAL_DEFAULT_N+1
    bench_mode=0
    print("CPU auto mode selected")
    print("[241-default] no arguments: CPU N22 default; use -g for A10G range mode")
  elif argc >= 2:
    arg=sys.argv[1]
    if arg == "-c":
      use_gpu=False
      print("CPU mode selected")
    elif arg == "-g":
      use_gpu=True
      print("GPU mode selected")
      if argc == 2:
        gpu_block=A10G_FINAL_DEFAULT_BLOCK
        gpu_max_blocks=A10G_FINAL_DEFAULT_MAX_BLOCKS
        gpu_log_level=A10G_FINAL_DEFAULT_LOG_LEVEL
        gpu_sort_mode=A10G_FINAL_DEFAULT_SORT_MODE
        requested_preset_arg=A10G_FINAL_DEFAULT_PRESET
        preset_queens_arg=A10G_FINAL_DEFAULT_PRESET
        bench_mode=A10G_FINAL_DEFAULT_BENCH_MODE
        reorder_window_mult=A10G_FINAL_DEFAULT_REORDER_WINDOW_MULT
        reorder_phase_jump=A10G_FINAL_DEFAULT_REORDER_PHASE_JUMP
        cross_stripe_safe=A10G_FINAL_DEFAULT_CROSS_STRIPE_SAFE
        worker_id=A10G_FINAL_DEFAULT_WORKER_ID
        worker_count=A10G_FINAL_DEFAULT_WORKER_COUNT
        broadmark_tail_variant=A10G_FINAL_DEFAULT_BROADMARK_VARIANT
        chunkshape148_bucket_run=A10G_FINAL_DEFAULT_CHUNKSHAPE148_BUCKET_RUN
        chunkshape148_iter_sort=A10G_FINAL_DEFAULT_CHUNKSHAPE148_ITER_SORT
    else:
      print(f"Unknown option: {arg}")
      print("Usage: nqueens [-c | -g] [nmin nmax] [gpu_block gpu_max_blocks log_level sort_mode] [preset_queens] [bench_mode] [window] [phase] [cross_stripe_safe] [worker_id worker_count variant]")
      return

    if argc >= 4:
      nmin=int(sys.argv[2])
      nmax=int(sys.argv[3])+1
    if argc >= 5:
      gpu_block=int(sys.argv[4])
    if argc >= 6:
      gpu_max_blocks=int(sys.argv[5])
    if argc >= 7:
      gpu_log_level=int(sys.argv[6])
    if argc >= 8:
      gpu_sort_mode=int(sys.argv[7])
    if argc >= 9:
      requested_preset_arg=int(sys.argv[8])
    if argc >= 10:
      bench_mode=int(sys.argv[9])
      if not (bench_mode==0 or bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32):
        print(f"[warning] bench_mode={bench_mode} was removed in 276 restore274/coretrim; using 0")
        bench_mode=0

    if bench_mode==11 or bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32:
      preset_queens_arg=requested_preset_arg
    else:
      if requested_preset_arg!=5:
        print(f"[warning] preset_queens={requested_preset_arg} is disabled in normal modes; using 5")
      preset_queens_arg=5

    if bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31 or bench_mode==32:
      if argc >= 11:
        reorder_window_mult=int(sys.argv[10])
      if argc >= 12:
        reorder_phase_jump=int(sys.argv[11])
      if argc >= 13:
        cross_stripe_safe=(int(sys.argv[12])!=0)
      if bench_mode==28:
        if argc >= 14:
          broadmark_tail_variant=int(sys.argv[13])
      if bench_mode==29 or bench_mode==31:
        if argc >= 14:
          worker_id=int(sys.argv[13])
        if argc >= 15:
          worker_count=int(sys.argv[14])
        if argc >= 16:
          broadmark_tail_variant=int(sys.argv[15])
        if argc >= 17:
          chunkshape148_bucket_run=int(sys.argv[16])
        if argc >= 18:
          chunkshape148_iter_sort=int(sys.argv[17])
      if bench_mode==30 or bench_mode==32:
        if argc >= 14:
          debug_chunk_start=int(sys.argv[13])
        if argc >= 15:
          debug_chunk_count=int(sys.argv[14])
        if argc >= 16:
          split_probe_chunk_list_spec=sys.argv[15]
        if argc >= 17:
          broadmark_tail_variant=int(sys.argv[16])

  FUNCID_REORDER_V2_WINDOW_MULT=reorder_window_mult
  FUNCID_REORDER_V2_PHASE_JUMP=reorder_phase_jump
  BROAD_MARKDIST_TAIL_VARIANT=broadmark_tail_variant
  CHUNKSHAPE148_BUCKET_RUN=chunkshape148_bucket_run
  CHUNKSHAPE148_ITER_SORT=chunkshape148_iter_sort

  if use_gpu and gpu_log_level>=1:
    if bench_mode==28:
      print(f"broadmarktail_reorder_sim: mode={bench_mode} preset={preset_queens_arg}")
    if bench_mode==29:
      print(f"broadmarktail_reorder_gpu: mode={bench_mode} preset={preset_queens_arg} worker={worker_id}/{worker_count}")
    if bench_mode==30:
      print(f"split291_final_probe: mode={bench_mode} preset={preset_queens_arg} chunk_start={debug_chunk_start} chunk_count={debug_chunk_count} chunk_list={split_probe_chunk_list_spec}")
    if bench_mode==31:
      print(f"split291_final_full_gpu: mode={bench_mode} preset={preset_queens_arg} worker={worker_id}/{worker_count}")
    if bench_mode==28 or bench_mode==29 or bench_mode==30 or bench_mode==31:
      print(f"funcid_reorder_v2_params: window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} param={funcid_reorder_param_tag()} reason={FUNCID_REORDER_V2_DEFAULT_REASON}")
      print(f"broadmarktail_params: version={BROAD_MARKDIST_TAIL_REORDER_VERSION} variant={BROAD_MARKDIST_TAIL_VARIANT} tag={broad_markdist_tail_variant_tag()} window_boost={broad_markdist_tail_window_boost_value()} phase_mix={1 if broad_markdist_tail_use_phase_mix() else 0} rotate_interleave={1 if broad_markdist_tail_use_rotating_interleave() else 0} phase_salt={broad_markdist_tail_phase_salt_value()} reason={BROAD_MARKDIST_TAIL_REORDER_DEFAULT_REASON}")
    if bench_mode==30 or bench_mode==31 or bench_mode==32:
      print(f"chunkshape148_params: version={CHUNKSHAPE148_REORDER_VERSION} bucket_run={chunkshape148_bucket_run_value()} run_tag={chunkshape148_bucket_run_tag()} reason={CHUNKSHAPE148_DEFAULT_REASON} run_reason={CHUNKSHAPE148_BUCKET_RUN_REASON}")
      print(f"chunkshape148_sort_params: iter_sort={chunkshape148_iter_sort_value()} iter_sort_tag={chunkshape148_iter_sort_tag()} sort_group={chunkshape148_sort_group_chunks()} k_per_thread={K_PER_THREAD_MAXD14} iter_sort_reason={CHUNKSHAPE148_ITER_SORT_REASON}")
      print(f"whi_elim_params: whi_elim=removed guard=or_all_high32 reason={WHI_ELIM_REASON}")
      print(f"feature354_params: feature_census=on scope=cpu_only kernel=unchanged_352 reason={FEATURE354_REASON}")

  print(" N:             Total           Unique         hh:mm:ss.ms")
  for N in range(nmin,nmax):
    start_time=datetime.now()
    if N<=5:
      total=_bit_total(N)
      dt=datetime.now()-start_time
      text=str(dt)[:-3]
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}")
      continue

    ijkl_list:Set[int]=set()
    constellations:List[Dict[str,int]]=[]
    subconst_cache:Set[Tuple[int,int,int,int,int,int,int,int,int,int,int]]=set()
    preset_queens:int=select_dynamic_preset_queens(N,preset_queens_arg)

    if gpu_log_level>=1:
      print(f"[dynamic-preset] N={N} preset_queens={preset_queens}")

    if bench_mode==11:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"[stream-cache-only] N={N} preset_queens={preset_queens} records={stream_records} bin={stream_fname} valid={1 if validate_bin_file(stream_fname) else 0}")
      print(f"{N:2d}:{0:18d}{0:17d}{text:>21s}    stream-cache-only")
      continue

    if bench_mode==28:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      reorder_fname,reorder_records,reorder_chunks=build_broad_markdist_tail_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      print(f"[broadmarktail-reorder-sim-only] N={N} preset_queens={preset_queens} records={stream_records} reordered_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} valid={1 if validate_bin_file(reorder_fname) else 0}")
      print(f"{N:2d}:{0:18d}{0:17d}{text:>21s}    broadmarktail-reorder-sim-only")
      continue

    if use_gpu and N>=21 and bench_mode==29:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      reorder_fname:str=broad_markdist_tail_reorder_output_fname(N,preset_queens,gpu_block,gpu_max_blocks)
      reorder_records:int=count_constellations_bin_records(reorder_fname)
      steps_for_count:int=gpu_block*gpu_max_blocks
      if steps_for_count<=0:
        steps_for_count=15488
      reorder_chunks:int=0
      if reorder_records>0:
        reorder_chunks=(reorder_records + steps_for_count - 1)//steps_for_count
      done_count:int=read_stream_done_count(reorder_fname+".done")
      if not (reorder_records==stream_records and done_count==stream_records and validate_bin_file(reorder_fname)):
        if gpu_log_level>=1:
          print(f"[broadmarktail-reorder-gpu-build] N={N} stream_records={stream_records} existing_records={reorder_records} done_count={done_count} bin={reorder_fname}")
        reorder_fname,reorder_records,reorder_chunks=build_broad_markdist_tail_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      elif gpu_log_level>=1:
        print(f"[broadmarktail-reorder-gpu-reuse] N={N} records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)}")
      progress_suffix:str=f"broadmarktail_{BROAD_MARKDIST_TAIL_REORDER_VERSION}_{broad_markdist_tail_variant_tag()}_gpu"
      total:int=exec_solutions_gpu_bin_stream_funcid_reorder(N,reorder_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,False,debug_chunk_start,debug_chunk_count,"",progress_suffix,worker_id,worker_count)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      status:str="ok" if expected[N]==total else f"ng({total}!={expected[N]})"
      if worker_count>1:
        status=f"partial-worker-{worker_id}-of-{worker_count}"
      if gpu_log_level>=1:
        print(f"[broadmarktail-reorder-gpu-done] N={N} source_records={stream_records} reordered_records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} variant={broad_markdist_tail_variant_tag()} worker={worker_id}/{worker_count} total={total}")
        if worker_count>1:
          print(f"[worker-done] N={N} worker={worker_id}/{worker_count} partial_total={total} expected_total={expected[N]}")
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    {status}")
      continue

    if use_gpu and N>=21 and (bench_mode==30 or bench_mode==31 or bench_mode==32):
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      reorder_fname:str=broad_markdist_tail_reorder_output_fname(N,preset_queens,gpu_block,gpu_max_blocks)
      reorder_records:int=count_constellations_bin_records(reorder_fname)
      steps_for_count:int=gpu_block*gpu_max_blocks
      if steps_for_count<=0:
        steps_for_count=15488
      reorder_chunks:int=0
      if reorder_records>0:
        reorder_chunks=(reorder_records + steps_for_count - 1)//steps_for_count
      done_count:int=read_stream_done_count(reorder_fname+".done")
      if not (reorder_records==stream_records and done_count==stream_records and validate_bin_file(reorder_fname)):
        if gpu_log_level>=1:
          print(f"[split291-base-build] N={N} stream_records={stream_records} existing_records={reorder_records} done_count={done_count} bin={reorder_fname}")
        reorder_fname,reorder_records,reorder_chunks=build_broad_markdist_tail_reordered_bin(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode)
      elif gpu_log_level>=1:
        print(f"[split291-base-reuse] N={N} records={reorder_records} chunks={reorder_chunks} bin={reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)}")
      base_reorder_fname:str=reorder_fname
      base_reorder_records:int=reorder_records
      shaped_fname:str=chunkshape148_reorder_output_fname(N,preset_queens,gpu_block,gpu_max_blocks)
      shaped_records:int=count_constellations_bin_records(shaped_fname)
      shaped_chunks:int=0
      if shaped_records>0:
        shaped_chunks=(shaped_records+steps_for_count-1)//steps_for_count
      shaped_done:int=read_stream_done_count(shaped_fname+".done")
      if not (shaped_records==base_reorder_records and shaped_done==base_reorder_records and validate_bin_file(shaped_fname)):
        if gpu_log_level>=1:
          print(f"[chunkshape148-build] N={N} source_records={base_reorder_records} existing_records={shaped_records} done_count={shaped_done} bin={shaped_fname}")
        shaped_fname,shaped_records,shaped_chunks=build_chunkshape148_reordered_bin(N,base_reorder_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level)
      elif gpu_log_level>=1:
        print(f"[chunkshape148-reuse] N={N} records={shaped_records} chunks={shaped_chunks} bin={shaped_fname} source_bin={base_reorder_fname}")
      if bench_mode==32:
        # 354: CPU-only feature census. No kernel, no correctness oracle --
        # this bin does not change the task set or touch the GPU, so there is
        # nothing for expected[N] to check. debug_chunk_count==0 means every
        # chunk (the full-run path); >0 means a chunk-bounded smoke run.
        feature_chunk_only:bool=(debug_chunk_count>0)
        processed:int=exec_features_cpu_bin_stream_split145(N,shaped_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,feature_chunk_only,debug_chunk_start,debug_chunk_count,split_probe_chunk_list_spec if feature_chunk_only else "")
        time_elapsed=datetime.now()-start_time
        text=str(time_elapsed)[:-3]
        status:str="feature-census-cpu-only"
        if gpu_log_level>=1:
          print(f"[feature354-done] N={N} source_records={stream_records} shaped_records={shaped_records} chunks={shaped_chunks} bin={shaped_fname} processed={processed}")
        print(f"{N:2d}:{processed:18d}{0:17d}{text:>21s}    {status}")
        continue
      progress_suffix:str=f"split291_{'probe' if bench_mode==30 else 'full'}_{CHUNKSHAPE148_REORDER_VERSION}_{BROAD_MARKDIST_TAIL_REORDER_VERSION}_{broad_markdist_tail_variant_tag()}"
      chunk_only:bool=(bench_mode==30)
      total:int=exec_solutions_gpu_bin_stream_split145(N,shaped_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,chunk_only,debug_chunk_start,debug_chunk_count,split_probe_chunk_list_spec if chunk_only else "",progress_suffix,worker_id if bench_mode==31 else 0,worker_count if bench_mode==31 else 1,0,K_PER_THREAD_MAXD14)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      status:str="ok" if expected[N]==total else f"ng({total}!={expected[N]})"
      if bench_mode==30:
        status="split145-probe"
      if bench_mode==31 and worker_count>1:
        status=f"partial-worker-{worker_id}-of-{worker_count}"
      if gpu_log_level>=1:
        print(f"[split291-{'probe' if bench_mode==30 else 'full'}-done] N={N} source_records={stream_records} base_reordered_records={base_reorder_records} shaped_records={shaped_records} chunks={shaped_chunks} bin={shaped_fname} base_bin={base_reorder_fname} param={funcid_reorder_run_param_tag(gpu_block,gpu_max_blocks)} chunkshape={CHUNKSHAPE148_REORDER_VERSION} window_mult={FUNCID_REORDER_V2_WINDOW_MULT} phase_jump={FUNCID_REORDER_V2_PHASE_JUMP} variant={broad_markdist_tail_variant_tag()} worker={worker_id}/{worker_count} total={total}")
        if bench_mode==31 and worker_count>1:
          print(f"[worker-done] N={N} worker={worker_id}/{worker_count} partial_total={total} expected_total={expected[N]}")
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    {status}")
      continue

    if use_gpu and N>=21:
      ijkl_list,subconst_cache,stream_records,preset_queens,stream_fname=ensure_constellations_bin_stream(N,ijkl_list,subconst_cache,preset_queens,gpu_log_level)
      total:int=exec_solutions_gpu_bin_stream(N,stream_fname,preset_queens,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe,False,0,1)
      time_elapsed=datetime.now()-start_time
      text=str(time_elapsed)[:-3]
      status:str="ok" if expected[N]==total else f"ng({total}!={expected[N]})"
      print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    {status}")
      continue

    ijkl_list,subconst_cache,constellations,preset_queens=gen_constellations(N,ijkl_list,subconst_cache,constellations,preset_queens)
    if gpu_log_level>=1:
      print(f"[constellation-config] N={N} preset_queens={preset_queens} constellations={len(constellations)}")
    exec_solutions(N,constellations,use_gpu,gpu_block,gpu_max_blocks,gpu_log_level,gpu_sort_mode,cross_stripe_safe)
    total:int=sum(c['solutions'] for c in constellations if c['solutions']>0)
    time_elapsed=datetime.now()-start_time
    text=str(time_elapsed)[:-3]
    status:str="ok" if expected[N]==total else f"ng({total}!={expected[N]})"
    print(f"{N:2d}:{total:18d}{0:17d}{text:>21s}    {status}")

if __name__=="__main__":
  main()
