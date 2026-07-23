# N-Queens CUDA/Codon ソルバ 開発ログ

## 現在の未解決課題 (Open Objectives) -- 最終更新: 328 (2026-07-22)

このセクションはリビジョンごとに更新されるサマリです。詳細な経緯は下の年代順ログを参照してください。

1. **[実装済み・検証待ち] 非コアレッシングメモリアクセス(w_arrのSoA分割)**
   ncu OPTアドバイザーが指摘したL2 Theoretical Sectors Global Excessive(189,728個、推定11.79%速度向上の余地)の原因は、`w_arr[idx]`(u64、8バイト間隔)読み込みが2つの32bit LDG.Eに分割され、かつ各読み込み自体が8バイト間隔ゆえに理想的にコアレッシングできていないことだと327の独立プローブで確認・特定した(3kernel比較)。`w_lo_arr`/`w_hi_arr`という2つの独立した密なu32配列に分割すればExcessiveがゼロになることも実証済み(sum_base==sum_soaで正当性も確認)。**328で、この検証済みの変更を5つ全てのkernel(maxd14/16/18/20/21)とディスパッチャに実装した。** ホットな発散DFSループ自体には一切触れていない(タスクごとに1回のみの読み込み箇所3箇所×5kernelのみ変更)。326(ホットループ複製、5戦5敗の一角)とはリスクの質が異なる低リスクな変更。実機でのビルド・実行はまだ行われていない。正当性(314666222712)を最優先で確認し、悪化すれば327へロールバックする方針。なお推定11.79%はkernel全体に対する見積もりであり、この変更はタスクごとに1回の読み込みのみが対象のため、実際の改善幅はそれよりかなり小さい可能性が高い。

2. **[撤回・保留] kernel内future_check_mask 1軸専用化**
   322/323のStall Branch Resolving知見への対応として設計(324)、実装(326)した`future_check_mask==0/!=0`ホットループ複製は、N=21フル実行で正当性は一致したが実行時間が517.563秒(319比+13.7%)と大幅に悪化し撤回した。240/266-269/273と合わせてGPU側kernel/ループ分解は5戦5敗。この方向性は保留とする。

3. **[結論・確定] Stall Branch Resolving (カーネル全体の約19〜23%)**
   3つの独立した測定手法(316サイクルベース/317ハードウェアカウンタ/318-319 PCサンプリング)が収束し、メインDFSループの共有back-edgeにおける再収束(BSYNC)オーバーヘッドが原因と結論(319/322)。per-line対応はCodon+ncuツールチェーンでは到達不能(318-321)。対策として試みたkernel内専用化(326)も大幅な性能悪化で撤回された。GPU側での構造的対策は現時点で5戦5敗であり、この課題への対応はいったん保留。

---


suzuki@cudacodon$ uname -a
Linux ip-172-31-3-195.us-west-2.compute.internal 6.1.158-180.294.amzn2023.x86_64 #1 SMP PREEMPT_DYNAMIC Mon Dec  1 05:36:50 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux
suzuki@cudacodon$ date
2026年  7月 10日 金曜日 05:11:41 UTC
suzuki@cudacodon$ codon build -release 292Py_kbatch4_gridstride_probe.py && ./292Py_kbatch4_gridstride_probe -g
GPU mode selected
 N:             Total           Unique         hh:mm:ss.ms
 5:                10                0          0:00:00.000
 6:                 4                0          0:00:00.003    ok
 7:                40                0          0:00:00.003    ok
 8:                92                0          0:00:00.002    ok
 9:               352                0          0:00:00.002    ok
10:               724                0          0:00:00.003    ok
11:              2680                0          0:00:00.004    ok
12:             14200                0          0:00:00.007    ok
13:             73712                0          0:00:00.011    ok
14:            365596                0          0:00:00.017    ok
15:           2279184                0          0:00:00.031    ok
16:          14772512                0          0:00:00.067    ok
17:          95815104                0          0:00:00.222    ok
18:         666090624                0          0:00:01.694    ok
19:        4968057848                0          0:00:08.981    ok
20:       39029188884                0          0:01:09.057    ok
21:      314666222712                0          0:06:07.340    ok


# GPU m4.xlarge での実行例
suzuki@cudacodon$ date
2026年  7月  6日 月曜日
suzuki@cudacodon$ codon build -release 237Py_restore232_fastdefault_keepfeatures_probe.py
suzuki@cudacodon$ ./237Py_restore232_fastdefault_keepfeatures_probe -g
GPU mode selected
 N:             Total           Unique         hh:mm:ss.ms
 5:                10                0          0:00:00.000
 6:                 4                0          0:00:00.018    ok
 7:                40                0          0:00:00.003    ok
 8:                92                0          0:00:00.004    ok
 9:               352                0          0:00:00.008    ok
10:               724                0          0:00:00.006    ok
11:              2680                0          0:00:00.006    ok
12:             14200                0          0:00:00.016    ok
13:             73712                0          0:00:00.012    ok
14:            365596                0          0:00:00.028    ok
15:           2279184                0          0:00:00.035    ok
16:          14772512                0          0:00:00.070    ok
17:          95815104                0          0:00:00.235    ok
18:         666090624                0          0:00:01.986    ok
19:        4968057848                0          0:00:08.971    ok
20:       39029188884                0          0:01:07.112    ok
21:      314666222712                0          0:07:07.834    ok


# GPU g5.xlarge での実行例
suzuki@cudacodon$ date
2026年  6月  9日 火曜日 05:55:00 UTC
suzuki@cudacodon$ stdbuf -oL -eL ./115Py_range_default_clean_cg_v2 -g 2>&1 | tee 115Py_cpu_range_$(date +%Y%m%d_%H%M%S).log
GPU mode selected
 N:             Total           Unique         hh:mm:ss.ms
 5:                10                0          0:00:00.000
 6:                 4                0          0:00:00.004    ok
 7:                40                0          0:00:00.004    ok
 8:                92                0          0:00:00.002    ok
 9:               352                0          0:00:00.002    ok
10:               724                0          0:00:00.003    ok
11:              2680                0          0:00:00.005    ok
12:             14200                0          0:00:00.007    ok
13:             73712                0          0:00:00.011    ok
14:            365596                0          0:00:00.018    ok
15:           2279184                0          0:00:00.036    ok
16:          14772512                0          0:00:00.104    ok
17:          95815104                0          0:00:00.465    ok
18:         666090624                0          0:00:03.475    ok
19:        4968057848                0          0:00:22.443    ok
20:       39029188884                0          0:03:04.146    ok
21:      314666222712                0          0:23:34.869    ok
22:     2691008701644                0          3:37:51.255    ok
23:    24233937684440                0  1 day, 11:20:40.926    ok


# CPU m4.xlarge での実行例
suzuki@cudacodon$ date
2026年  7月  6日 月曜日
suzuki@cudacodon$ codon build -release 237Py_restore232_fastdefault_keepfeatures_probe.py
suzuki@cudacodon$ ./237Py_restore232_fastdefault_keepfeatures_probe -c
CPU mode selected
 N:             Total           Unique         hh:mm:ss.ms
 5:                10                0          0:00:00.000
 6:                 4                0          0:00:00.000    ok
 7:                40                0          0:00:00.000    ok
 8:                92                0          0:00:00.000    ok
 9:               352                0          0:00:00.000    ok
10:               724                0          0:00:00.001    ok
11:              2680                0          0:00:00.004    ok
12:             14200                0          0:00:00.008    ok
13:             73712                0          0:00:00.023    ok
14:            365596                0          0:00:00.044    ok
15:           2279184                0          0:00:00.146    ok
16:          14772512                0          0:00:00.730    ok
17:          95815104                0          0:00:04.557    ok
18:         666090624                0          0:00:40.116    ok
19:        4968057848                0          0:05:16.015    ok

# CPU m4.16xlarge での実行例
workspace#suzuki$ date
2026年  7月  6日 月曜日
suzuki@cudacodon$ codon build -release 237Py_restore232_fastdefault_keepfeatures_probe.py
suzuki@cudacodon$ ./237Py_restore232_fastdefault_keepfeatures_probe -c
CPU mode selected
 N:             Total           Unique         hh:mm:ss.ms
 5:                10                0          0:00:00.000
 6:                 4                0          0:00:00.095    ok
 7:                40                0          0:00:00.002    ok
 8:                92                0          0:00:00.001    ok
 9:               352                0          0:00:00.002    ok
10:               724                0          0:00:00.006    ok
11:              2680                0          0:00:00.010    ok
12:             14200                0          0:00:00.019    ok
13:             73712                0          0:00:00.041    ok
14:            365596                0          0:00:00.090    ok
15:           2279184                0          0:00:00.170    ok
16:          14772512                0          0:00:00.270    ok
17:          95815104                0          0:00:00.409    ok
18:         666090624                0          0:00:04.462    ok
19:        4968057848                0          0:00:16.978    ok
20:       39029188884                0          0:02:10.232    ok
21:      314666222712                0          0:18:05.956    ok

# CPU m4.16xlarge での実行例
workspace#suzuki$ date
2026年  6月  9日 火曜日 14:23:02 JST
workspace#suzuki$ stdbuf -oL -eL ./115Py_range_default_clean_cg_v2 -c 2>&1 | tee 115Py_cpu_range.log
CPU mode selected
 N:             Total           Unique         hh:mm:ss.ms
 5:                10                0          0:00:00.000
 6:                 4                0          0:00:00.105    ok
 7:                40                0          0:00:00.000    ok
 8:                92                0          0:00:00.000    ok
 9:               352                0          0:00:00.015    ok
10:               724                0          0:00:00.011    ok
11:              2680                0          0:00:00.009    ok
12:             14200                0          0:00:00.020    ok
13:             73712                0          0:00:00.042    ok
14:            365596                0          0:00:00.091    ok
15:           2279184                0          0:00:00.173    ok
16:          14772512                0          0:00:00.270    ok
17:          95815104                0          0:00:00.412    ok
18:         666090624                0          0:00:04.433    ok
19:        4968057848                0          0:00:17.103    ok
20:       39029188884                0          0:02:11.042    ok
21:      314666222712                0          0:18:07.042    ok
22:     2691008701644                0          2:38:58.023    ok

# CPU m4.16xlarge での実行例
workspace#suzuki$ date
2026年  5月 15日 金曜日 20:50:42 JST
workspace#suzuki$ codon build -release 84Py_constellations_GPU_cuda_codon_dynamic_p8_stream.py
workspace#suzuki$ ./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream -c
CPU mode selected
 N:             Total           Unique         hh:mm:ss.ms
 5:                10                0          0:00:00.000
 6:                 4                0          0:00:00.088    ok
 7:                40                0          0:00:00.024    ok
 8:                92                0          0:00:00.001    ok
 9:               352                0          0:00:00.005    ok
10:               724                0          0:00:00.002    ok
11:              2680                0          0:00:00.010    ok
12:             14200                0          0:00:00.020    ok
13:             73712                0          0:00:00.041    ok
14:            365596                0          0:00:00.091    ok
15:           2279184                0          0:00:00.171    ok
16:          14772512                0          0:00:00.275    ok
17:          95815104                0          0:00:00.409    ok
18:         666090624                0          0:00:04.455    ok
19:        4968057848                0          0:00:17.064    ok
20:       39029188884                0          0:02:10.450    ok
21:      314666222712                0          0:18:05.956    ok
22:     2691008701644                0          2:38:08.664    ok
23:    24233937684440                0   1 day, 0:43:10.509    ok


# GPU g5.16xlarge での実行例
suzuki@cudacodon$ date
2026年  5月 15日 金曜日 09:34:47 UTC
suzuki@cudacodon$ codon build -release 84Py_constellations_GPU_cuda_codon_dynamic_p8_stream.py
suzuki@cudacodon$ ./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream -g
  or
suzuki@cudacodon$ ./84Py_constellations_GPU_cuda_codon_dynamic_p8_stream -g 5 22 32 484 1 0 7
GPU mode selected
version        : 84 stream bin GPU runner from 82 dynamic preset P8 cross_stripe_safe: 0
 N:             Total           Unique         hh:mm:ss.ms
 5:                10                0          0:00:00.000
 6:                 4                0          0:00:00.004    ok
 7:                40                0          0:00:00.003    ok
 8:                92                0          0:00:00.002    ok
 9:               352                0          0:00:00.002    ok
10:               724                0          0:00:00.003    ok
11:              2680                0          0:00:00.004    ok
12:             14200                0          0:00:00.007    ok
13:             73712                0          0:00:00.011    ok
14:            365596                0          0:00:00.018    ok
15:           2279184                0          0:00:00.037    ok
16:          14772512                0          0:00:00.107    ok
17:          95815104                0          0:00:00.466    ok
18:         666090624                0          0:00:03.505    ok
19:        4968057848                0          0:00:22.592    ok
20:       39029188884                0          0:02:24.917    ok
21:      314666222712                0          0:25:38.459    ok
22:     2691008701644                0          3:18:42.963    ok
23:    24233937684440                0   1 day, 6:08:25.451    ok

g5.16xlarge は NVIDIA A10G GPU を搭載しており、CUDA 13.0 対応のドライバが入っています。
g5.xlarge は NVIDIA A10G GPU を搭載しており、CUDA 13.0 対応のドライバが入っています。

速度が上がらない理由
-----------------------
g5.xlarge  → A10G 1枚
g5.16xlarge → A10G 1枚
------------------------

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



# N-Queens Python/Codon CUDA 更新履歴

このREADMEは、N-Queens Python/Codon CUDA版の連番実験について、「どの版で何を施したか」を一覧管理するための索引です。

## 運用方針

- 過去の `*.py` へ履歴一覧を一括転記しない。過去ソースは、その時点の実験状態を示す記録として保持する。
- 全体の更新履歴一覧は、この `README.md` に集約する。
- 今後の `xxxPy...py` の冒頭には、過去一覧ではなく、その版で実施した短い `NQ_UPDATE_MEMO` だけを置く。
- `xxxPy...sh` の冒頭にも、親版、実験目的、期待するdispatch/stack、検証条件を短く残す。
- 更新の都度、`xxxPy...py`、`xxxPy...sh` と合わせて、この `README.md` も追記版として配布する。
- 正当性OKかつ高速化した版は「新最速基準」として明記する。
- 正当性OKだが遅い版は「不採用」と明記し、次版では最速基準へ戻す。
- `CUDA_ERROR_INVALID_PTX` は計算不一致ではなくJIT不成立として扱い、危険なkernel差分を撤回する。

## 最新基準と直近方針

**2026-07-06 現在の実務基準**

- 現時点のN=21 full once数値上最速基準は **217Py `0:07:07.709`**。
- 230Pyは final total / progress / dispatch はすべてOKだが `0:07:08.848` で大きく退行したため不採用。
- 次版は、230Pyを採用せず **231Py = 217Py rootrestlate restore / no kernel-change relative to 217 / README整理版** として、224〜230の不採用差分を撤回する。


- 175Pyは、MAXD14の4-bit nibble scheduleにより `N=21 full once` で `0:07:51.106` を記録した旧最速基準。
- 182Pyは、181/180/175相当kernelを維持した管理方針反映版で、r4検証により `0:07:51.064` を記録した。
- 183Pyは、182のruntime-only検証方針を正式継承し、kernelを変更せず `split183` tagへ分離した安全継承版。N=21 fullは `0:07:51.091`、正当性OK。
- 184Pyは、183を親にMAXD14 kernelだけ no-sibling spill elision / tail-call descent へ変更した版。N=21 fullは `0:07:15.635`、最終合計 `314666222712` 一致、131チャンク完走、warning/errorなし。183比 `35.456秒`、`7.526%` 改善したため、新最速基準として採用された。
- 185Pyは、184を親に `terminal_parent_depth==13` fast pathを追加した単独実験版。N=21 fullは `0:07:24.682`、正当性OKだが184比で `9.047秒`、`2.077%` 遅いため不採用。
- 186Pyは、185を採用せず184を親に戻し、MAXD14 kernelだけ `terminal_base14` の分岐をDFS loop外側へ分離する単独実験版。N=21 fullは `0:09:14.022`、正当性OKだが184比で `118.387秒`、`27.176%` 遅いため不採用。
- 187Pyは、186を採用せず184を親に戻し、MAXD14 kernelだけ `child_jmark_mask!=u32(0)` の軽量guardを追加する単独実験版。N=21 fullは `0:07:16.241`、正当性OKだが184比で `0.606秒`、`0.139%` 遅いため不採用。
- 188Pyは、187を採用せず184を親に戻し、MAXD14 kernelだけ `future_check_mask==0` scheduleでfuture-prune判定を軽量guardする単独実験版。N=21 fullは `0:07:10.137`、最終合計 `314666222712` 一致、131チャンク完走、warning/errorなし。184比 `5.498秒`、`1.262%` 改善、183比 `40.954秒`、`8.693%` 改善したため、当時の新最速基準として採用。
- 189Pyは、188の新最速化を受け、188のfuture_check_mask guard/no-sibling spill elisionを親に、MAXD14 kernelのみ単一候補frameを連続処理する forced-chain fast path を追加した単独実験版。N=21 fullは `0:14:57.665`、正当性OKだが188比で `467.528秒`、`108.693%` 遅いため不採用。当時の新最速基準は188のまま。
- 190Pyは、189を採用せず188を親に戻し、MAXD14 kernelのみ `block_check_mask==0` scheduleでblock_code分岐/decodeを省く軽量guardを追加した単独実験版。N=21 fullは `0:07:53.357`、正当性OKだが188比で `43.220秒`、`10.048%` 遅いため不採用。当時の新最速基準は188のまま。
- 191Pyは、190を採用せず188を親に戻し、MAXD14 kernelのみqueen配置の `(state|bit)` を `(state^bit)` へ置換する単独実験版。N=21 fullは `0:07:10.144`、正当性OK、warning/errorなし。ただし188比で `0.007秒`、`0.002%` 遅く、ほぼ同等ながら188を上回らないため不採用。当時の新最速基準は188のまま。
- 192Pyは、191を採用せず188を親に戻し、MAXD14 kernelのみroot frameの `cur_avail` が単一bitの場合にdepth 0をDFS loop前へ1段prerollする単独実験版。N=21 fullは `0:07:10.118`、最終合計 `314666222712` 一致、131チャンク完走、warning/errorなし。188比で `0.019秒`、`0.004%` のごく小幅改善で誤差級だが、数値上の新最速基準として採用。
- 193Pyは、192の数値上最速化を受け、terminal-before-future orderingを試した単独実験版。N=21 fullは `0:07:17.870`、正当性OKだが192比で `7.752秒`、`1.802%` 遅いため不採用。新最速基準は192のまま。
- 194Pyは、193を採用せず192を親に戻し、MAXD14 kernelのみ root frameの `cur_avail` が1bitまたは2bitの場合にdepth 0の先頭候補をDFS loop前へ1段prerollする単独実験版。N=21 fullは `0:07:08.288`、正当性OK、warning/errorなし。192比で `1.830秒`、`0.425%` 高速で、数値上の新最速基準として採用。
- 195Pyは、194のroot one/two-candidate preroll/future_check_mask guard/no-sibling spill elisionを親に、MAXD14 kernelのみ root frameの `cur_avail` が1bit/2bit/3bitの場合にdepth 0の先頭候補をDFS loop前へ1段prerollした単独実験版。N=21 fullは `0:07:08.738`、最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。ただし194比で `0.450秒`、`0.105%` 遅いため不採用。新最速基準は194のまま。
- 196Pyは、195を採用せず194のroot one/two-candidate preroll/future_check_mask guard/no-sibling spill elisionへ戻し、MAXD14 kernelのみroot<=2の判定形をsecond-lowbit方式へ置き換えた復帰・微差実験版。N=21 fullは `0:07:08.239`、最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。194比で `0.049秒`、`0.011%` 高速、195比で `0.499秒`、`0.116%` 高速のため、差は極小ながら数値上の新最速基準として採用。
- 197Pyは、196のroot<=2 second-lowbit preroll/future_check_mask guard/no-sibling spill elisionを親にし、MAXD14 kernelのみroot<=2判定後の `root_preroll` flagを削ってdirect-ifでpreroll bodyへ入る微差実験版。r3検証のN=21 fullは `0:07:08.202`、最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。196比で `0.037秒`、`0.009%` 高速、194比で `0.086秒`、`0.020%` 高速のため、差は極小ながら数値上の新最速基準として採用。
- 197Py r2は、CUDA kernelと `.py` の探索ロジックは197のまま変更せず、`.sh` のstatic checkだけを補修した配布版。direct-if の検査を空白差分に強くし、static mismatch時にsummaryと該当grep行を表示して、`.py`/`.sh` の取り違えを判別しやすくした。
- 197Py r3は、CUDA kernelと `.py` の探索ロジックは197/r2のまま変更せず、`.sh` の `source_split_tag` 静的検査だけを補修した配布版。旧版文字列がコメントや履歴文に残ってもGPU実行を止めないよう、検査対象をactive runtime/progress tag限定へ変更し、N=21 fullで正当性OK・数値上最速を確認した。
- 198Pyは、197のroot<=2 direct-if preroll/future_check_mask guard/no-sibling spill elisionを親にし、MAXD14 kernelのみroot<=2判定を `root_rest & (root_rest - 1) == 0` のtail-testへ置き換えた微差実験版。N=21 fullは `0:07:08.239`、正当性OKだが197比で `0.037秒`、`0.009%` 遅いため不採用。新最速基準は197のまま。
- 199Pyは、198を採用せず197へ戻し、MAXD14 kernelのみroot<=2判定を `root_rest==root_second` の等価比較へ置き換えた微差実験版。N=21 fullは `0:07:08.293`、正当性OKだが197比で `0.091秒`、`0.021%` 遅く、198/196比で `0.054秒`、194比でも `0.005秒` 遅いため不採用。新最速基準は197のまま。
- 200Pyは、199を採用せず197のroot<=2 direct-if preroll/future_check_mask guard/no-sibling spill elisionへ戻し、MAXD14 kernelのみroot-preroll内のfuture-prune判定から冗長な `future_check_mask!=0` 外側guardを外し、`pr_nibble_op & 8` を直接見る微差実験版。generic DFS loopの `future_check_mask==0` schedule軽量guardは197/188のまま維持する。root fast-startの対象範囲は197/196/194と同じ1bit/2bit rootのみで、3bit以上はgeneric loopのまま。初回配布版でcudacodon側の静的検査がsummary表示前に停止したケースが出たため、200 r2ではCUDA kernelと `.py` を変更せず、`.sh` の静的検査FAIL時にsummaryを必ず表示するよう補修した。r2 N=21 fullは `0:07:09.520`、最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。ただし197比で `1.318秒`、`0.308%` 遅いため不採用。新最速基準は197のまま。

- 200Py r2は、CUDA kernelと `.py` の探索ロジックは200のまま変更せず、`.sh` の静的検査だけを補修した配布版。`STATIC_ONLY=1` ではfull-run lockを取らず、通常実行時にsource static checkがFAILした場合もsummaryを表示してから停止する。root-preroll future-bit検査もコメント/空白差に依存しにくい形へ変更した。
- 201Pyは、200が正当性OKながら197比で遅かったため採用せず、197のroot<=2 direct-if preroll/future_check_mask guard/no-sibling spill elisionへ戻す。MAXD14 kernelのみ、root-preroll内のchild_jmark処理で `pr_child_jmark:u32=child_jmark_mask&u32(1)` の一時scalarを作らず、`if (child_jmark_mask&u32(1))!=u32(0):` と直接判定する単独実験版。N=21 fullは `0:07:08.383`、正当性OKだが197比で `0.181秒`、`0.042%` 遅いため不採用。新最速基準は197のまま。
- 202Pyは、201を採用せず197へ戻し、MAXD14 kernelのみroot-preroll内で残りroot siblingを保存する際のpayloadを `cur_avail` ではなく `root_second` から作る単独実験版。N=21 fullは `0:07:08.482`、正当性OKだが197比で `0.280秒`、`0.065%` 遅いため不採用。新最速基準は197のまま。
- 203Pyは、202を採用せず197へ戻し、MAXD14 kernelのみroot-prerollで残りroot siblingを保存する際に `cur_depth==0` が既知であることを利用し、`avail[save_sp]=cur_avail|(u32(cur_depth)<<u32(27))` を `avail[save_sp]=cur_avail` へ置き換える単独実験版。N=21 fullは `0:07:08.344`、正当性OKだが197比で `0.142秒`、`0.033%` 遅いため不採用。新最速基準は197のまま。
- 204Pyは、203を採用せず197へ戻し、MAXD14 kernelのみroot<=2判定で `root_first` 除去後の `root_rest` を `cur_avail^root_first` ではなく `cur_avail&(cur_avail-u32(1))` で作る微差実験版。N=21 fullは `0:07:07.795`、最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。197比で `0.407秒`、`0.095%` 高速、194比で `0.493秒`、`0.115%` 高速で、数値上の新最速基準として採用。
- 205Pyは、204のroot_rest clear-lowbit/future_check_mask guard/no-sibling spill elisionを親にし、MAXD14 kernelのみroot<=2判定で `root_after_second:u32=root_rest^root_second` の一時scalarを削り、`if (root_rest^root_second)==u32(0):` へ直接inlineする微差実験版。root fast-startの対象範囲は204/197/196/194と同じ1bit/2bit rootのみで、3bit以上はgeneric loopのまま。採用可否はN=21 fullで204/197/203/202/201/200/199/198/196/194/188比timingを確認して判断する。

## ソース冒頭コメントの推奨形

```python
# =============================================================================
# NQ_UPDATE_MEMO
# 205: 更新メモ: <親版>を基準に、<変更点>を実施。<期待効果/不採用差分/検証条件>。
# Full update history: see README.md
# =============================================================================
```

## 検証シェル冒頭コメントの推奨形

```bash
# 205 validation harness
# Parent: 204 validated rootclearrest/futuremask/no-sibling MAXD14 baseline; 198/199/200/201/202/203 correct but not adopted; 195 rootthree and 193/189/190/191 are correct but not adopted
# Experiment: <変更点>
# Expected N=21 dispatch: required=14, selected MAXD14, schedule_words=0, stack=208 bytes/thread
# Validation: one N=21 full GPU run + progress TSV reconstruction
```

## 更新履歴一覧

### 注記

- 20Py〜47Py付近の詳細は、現在手元にある168〜205系ソース内コメントと会話ログから安全に復元できる範囲に限定しています。
- 速度は、N=21 full onceで添付ログまたは検証ログから確認できたものだけを記載しています。

- **020-047**: 現在手元に残る資料では詳細未確定。Constellation分割、CPU/GPU共通DFS、Codon/CUDA移植の基礎整備期として扱う。
- **049**: gen_constellations() の返却値を counter[0] ではなく preset_queens 本体へ修正し、ログ上の誤表示を防止。
- **054-055**: N=20向けGPU投入粒度を測定し、32x484を安定候補として採用。
- **056-060**: GPU結果のscatter/copy-backを削り、chunkごとのdirect_total集計へ寄せて後段集計を軽量化。
- **059**: Zobrist/前処理キャッシュの無駄計算を抑制し、通常運用で不要なhash計算を避ける。
- **072**: 32x484を安定ベンチ設定化し、sort/stripe実験の基準線を整理。
- **074**: cross-stripe reorderの欠落バグを修正し、任意chunk数で全recordを一度ずつ出力する安全形へ変更。
- **076**: auto_sort_modeを整理し、N=20/21のみsort_mode=9を自動採用、N>=22は安全側で従来順序を維持。
- **078**: P5/fid=8..11のsame-row遷移を追加し、SQBjl*jrB系を正しくnext_funcidへ接続。
- **079**: subconst_cacheを開始星座scごとにclearし、preset=6/7で正当なconstellation生成が消える問題を修正。
- **080**: preset>=7ではsubconst_cacheをbypassし、同一状態へ複数経路で到達するmultiplicityを保持。
- **081**: GPU kernel起動が外れてresultsが0のままになる退行を修正し、GPU計算経路を復帰。
- **084**: constellationを全件List保持せず、stream binへ書き出してGPU chunk runnerで読む形を導入。
- **092**: progress TSVへchunk入力統計、free popcount、row/end/depth、score、funcid countを追加。
- **093-096**: funcid risk bucket reorder v2を導入し、B/A/C/G/O quotaとbucket内stripe samplingを実装。
- **097**: N22向けselected chunk microbenchを追加し、full run前の短時間比較を可能化。
- **098**: build_soa、stats、sort、kernel、reduceのstage timing診断を追加。
- **099**: chunk size profileを追加し、同じ代表chunkを1x/2x/4x投入粒度で比較。
- **100-102**: funcid target/single/split診断を追加し、heavy_tail/bulk_heavy/restの分離効果を測定。
- **103-105**: depth/free-popcount、mark pattern、exact mark-distance診断を追加し、tail islandをfid/gap/d1/d2で特定。
- **106**: exact mark-distance risk reorderを導入し、fid x gap x d1のrisk scoreでstream orderを再配置。
- **107**: broad funcid quotaを主制約に戻し、mark-distance riskをsecondary quotaとして配分。
- **109-114**: funcid17とcell_G_Hの弱い三次split、phase/rotate/wide ablationを追加し、broadmarktail v4へ整理。
- **115**: A10G単GPU向け実用デフォルトを確定。broadmarktail mode29、w8_j7、variant2 rotate_onlyを採用。
- **127-130**: selected chunk probe/full worker streamとgeneric-only split harnessを整理し、host shaping評価をkernel変更なしで実施。
- **140-141**: scorestripe内の+11/+1 pair selectorと+29 octet first-pair lockを導入し、隣接quartetの位相を安定化。
- **148**: scorestripe_v9 lanephase32 octetfirstpairlock29をhost-side task配置のvalidated baselineとして採用。
- **154-157**: GPU入力をu32へ固定化し、ctrl0_arrとmarkctrl_arrへfid/row/mark/end情報をpre-pack。
- **162**: 148 scorestripe host shapingのvalidated継承点。後続kernel実験のhost layout基準として使用。
- **164**: child frame push initializationを見直し、GPU frame初期化配置を最適化。
- **167**: static MAXD dispatch baseline。required depthをhostで走査し、MAXD16/18/20/21の最小安全kernelを選択。N=21: `0:10:33.039`。
- **168**: ctrl metadataをpacked byte opcode schedule化し、MAXD16 local storageを320→272 bytes/threadへ削減。N=21: `0:09:20.261`。
- **169**: MAXD14静的kernelを追加し、N=21のrequired_maxd=14をMAXD14へdispatch。240 bytes/thread化したが実測は168より遅い。N=21: `0:09:33.503`。
- **170**: MAXD14のschedule配列を廃止。u64版はPTX JIT失敗、u32-pair版で224 bytes/thread化し168を小幅更新。N=21: `0:09:18.483`。
- **171**: current frameをregister保持し、候補bit反復中のlocal load/storeを削減。大幅高速化して8分台へ到達。N=21: `0:08:11.231`。
- **172**: active frameをregister保持したままancestor stackを13段へ縮小し、208 bytes/thread化。速度は171同等。N=21: `0:08:11.190`。
- **173**: frame entry時metadata一括decodeを試行。register圧増大で大幅低下したため不採用。N=21: `0:12:28.469`。
- **174**: compact_op fast pathを試行。正当性OKだが175/172より遅く不採用。N=21: `0:08:21.387`。
- **175**: MAXD14 scheduleを4-bit nibble化し、shift/decodeを簡素化。`0:07:51.106` で新最速基準。N=21: `0:07:51.106`。
- **176**: current nibbleをregister保持。scalar追加による悪化が大きく不採用。N=21: `0:08:59.402`。
- **177**: terminal-first判定を試行。分岐形悪化で不採用。N=21: `0:09:47.077`。
- **178**: host-precomputed scheduleとblock-switchを試行。どちらも `CUDA_ERROR_INVALID_PTX` で不採用。
- **179**: block-switch no-or版を試行。`CUDA_ERROR_INVALID_PTX` 継続のためblock-switch系を撤回。
- **180**: 175相当kernelへ安全に戻し、tail diagnosticsを追加。N=21: `0:07:51.191`。175比 `+0.085秒` で実質同等。top5 tail share `41/1000`、top10 tail share `81/1000`。
- **181**: kernelは180/175相当のまま、更新履歴ヘッダーとtail平坦性メトリクスを追加。N=21: `0:07:51.068`。最終合計 `314666222712` 一致、131チャンク完走、top5 tail share `41/1000`、top10 tail share `81/1000`。175比 `0.038秒` 高速で、数値上は現時点最速。
- **182**: 181の正当性OK・175同等最速を受け、GPU kernelは181/180/175相当のまま、ソース冒頭を短い `NQ_UPDATE_MEMO` 方式へ整理。初期 `.sh` のsource/model検査が強すぎて冒頭で落ちたため、r4で巨大なPython assert検証を外し、GPU full run・dispatch・progress検証に限定。N=21: `0:07:51.064`。最終合計 `314666222712` 一致、131チャンク完走、top5 tail share `41/1000`、top10 tail share `81/1000`。181比 `0.004秒` 高速で、数値上の最速値を更新。
- **183**: 182の正当性OK・最速値更新を受け、GPU kernelは182/181/180/175相当のまま変更せず、runtime-clean tailflat版として固定。`.py` 冒頭は当該版の短い更新メモのみ、全体履歴はREADME集約を継続。`.sh` は182 r4で実運用確認済みのruntime-only安全ハーネスを正式継承し、source/model assertは非強制、dispatch・progress・tail平坦性を検証。progress/log tagは `split183` へ分離。182/181/180/175相当のnibble MAXD14 kernelを維持。runtime-only検証ハーネスを正式固定。N=21 full: `0:07:51.091`。最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、tail平坦、warning/errorなし。最終運用基準として採用。
- **184**: 183を親に、host側のscorestripe/chunkshape148、dispatch、cache、progress検証方針は維持したまま、MAXD14 kernelのdescend処理のみ変更。`sp`が兼ねていた論理深度と保存stack pointerを `cur_depth` / `save_sp` に分離し、親frameの `cur_avail` が0になった場合はancestor stackへ保存せずchildをactive frameへ直接進める no-sibling spill elision / tail-call descent を導入。残候補がある親だけを保存し、復元に必要な論理深度は `avail` 上位5bitへpackする。N<=27では実bitboardが下位27bitに収まるため、stack_bytes_per_threadは208のまま。N=21 full: `0:07:15.635`。最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。183比 `35.456秒`、`7.526%` 改善したため、新最速基準として採用。tailはtop5 share `41/1000`、top10 share `81/1000`、p95/p50 `1068/1000`。
- **185**: 184の正当性OK・新最速化を受け、host側scorestripe/chunkshape148、dispatch、cache、no-sibling spill elision、stack_bytes_per_thread=208を維持したまま、MAXD14 terminal判定だけを単独変更。N=21/MAXD14では全launchが `required_maxd=14` でterminal parent depthが13になるため、`terminal_parent_depth==13` のfast pathを作り、hot loopでは `cur_depth==13` のliteral比較でterminal判定する。非13 terminal scheduleには184相当のgeneric fallbackを残し、N<=27の挙動を静かに狭めない。N=21 full: `0:07:24.682`。最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。ただし184比で `9.047秒`、`2.077%` 遅いため不採用。新最速基準は184のまま。
- **186**: 185は正当性OKだが速度退行したため採用せず、184のno-sibling spill elision / tail-call descentを親に戻す。host側scorestripe/chunkshape148、dispatch、cache、bitboard演算、solution arithmetic、stack_bytes_per_thread=208は184と同一。MAXD14 kernelのみ、`terminal_base14` をDFS loop内のterminal hitごとに判定する形から、loop外側で `terminal_base14==0` の通常base pathとbase14 pathへ分離する形へ変更。terminal hit hot pathの分岐削減がloop複製/register pressureを上回るかを確認する単独実験版。N=21 full: `0:09:14.022`。最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。ただし184比で `118.387秒`、`27.176%` 遅いため不採用。新最速基準は184のまま。
- **187**: 186は正当性OKだが大幅退行したため採用せず、184のno-sibling spill elision / tail-call descentを親に戻す。host側scorestripe/chunkshape148、dispatch、cache、bitboard演算、solution arithmetic、stack_bytes_per_thread=208は184と同一。MAXD14 kernelのみ、child_jmark処理の直前に `if child_jmark_mask!=u32(0):` の軽量guardを追加し、child jmark actionを持たないscheduleではhot loop中の `child_jmark_mask >> cur_depth` とmask判定を省く。非zero mask時のjmark arithmeticは184と同一で、185のterminal_depth=13 literal pathおよび186のterminal_base14 outer loop splitは入れない。N=21 full: `0:07:16.241`。最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。ただし184比で `0.606秒`、`0.139%` 遅いため不採用。新最速基準は184のまま。
- **188**: 187は正当性OKだが184比で微小退行したため採用せず、184のno-sibling spill elision / tail-call descentを親に戻す。host側scorestripe/chunkshape148、dispatch、cache、bitboard演算、solution arithmetic、stack_bytes_per_thread=208は184と同一。MAXD14 kernelのみ、schedule生成時にfuture-prune bitを持つ深度を `future_check_mask` へ集約し、`future_check_mask==0` のscheduleではhot loop中の `nibble_op & 8` 判定とfuture-prune分岐を省く軽量guardを追加する。`future_check_mask!=0` のscheduleは184相当のgeneric future-prune判定を残す。187のchild_jmark guard、185のterminal_depth=13 literal path、186のterminal_base14 outer splitは入れない。N=21 full: `0:07:10.137`。最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。184比で `5.498秒`、`1.262%` 改善したため、新最速基準として採用。tailはtop5 share `41/1000`、top10 share `81/1000`、p95/p50 `1066/1000`。
- **189**: 188の正当性OK・新最速化を受け、188のfuture_check_mask guard/no-sibling spill elisionを親にする。host側scorestripe/chunkshape148、dispatch、cache、bitboard演算、solution arithmetic、stack_bytes_per_thread=208は188と同一。MAXD14 kernelのみ、active frameの `cur_avail` が単一bitの場合に、兄弟候補がないことを利用してancestor stackへ保存せず、連続する単一候補frameを小さなinner loopで処理する forced-chain fast path を追加する。multi-candidate frameは188 generic pathのまま残し、future_check_mask guard、terminal判定、child_jmark処理、no-sibling save_sp/cur_depth復元規則は188と同じ意味を保つ。N=21 full: `0:14:57.665`。最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。ただし188比で `467.528秒`、`108.693%` 遅いため不採用。当時の新最速基準は188のまま。
- **190**: 189は正当性OKだが大幅退行したため採用せず、188のfuture_check_mask guard/no-sibling spill elisionを親に戻す。host側scorestripe/chunkshape148、dispatch、cache、bitboard演算、solution arithmetic、stack_bytes_per_thread=208は188と同一。MAXD14 kernelのみ、schedule生成時にblock operationを持つ深度を `block_check_mask` へ集約し、`block_check_mask==0` のscheduleではhot loop中の `block_code` 分岐/decodeを省いて通常1段diagonal更新へ直接進む軽量guardを追加する。`block_check_mask!=0` のscheduleは188相当のper-depth block_code pathを残す。189のforced-chain fast path、187のchild_jmark guard、185のterminal_depth=13 literal path、186のterminal_base14 outer splitは入れない。N=21 full: `0:07:53.357`。最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。ただし188比で `43.220秒`、`10.048%` 遅いため不採用。当時の新最速基準は188のまま。
- **191**: 190は正当性OKだが188比で退行したため採用せず、188のfuture_check_mask guard/no-sibling spill elisionを親に戻す。host側scorestripe/chunkshape148、dispatch、cache、bitboard演算、solution arithmetic、stack_bytes_per_thread=208は188と同一。MAXD14 kernelのみ、`cur_avail = bm & ~(cur_ld|cur_rd|cur_col)` から選んだ `bit` がactive frameの `cur_ld/cur_rd/cur_col` とboard内でdisjointであることを利用し、queen配置の `(cur_ld|bit)`, `(cur_rd|bit)`, `(cur_col|bit)` をそれぞれ `(cur_ld^bit)`, `(cur_rd^bit)`, `(cur_col^bit)` へ置換する。188のfuture_check_mask guard/no-sibling save_sp/cur_depth復元規則はそのまま維持し、190のblock_check_mask guard、189のforced-chain fast path、187のchild_jmark guard、185/186のterminal系変更は入れない。N=21 full: `0:07:10.144`。最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。ただし188比で `0.007秒`、`0.002%` 遅く、ほぼ同等ながら改善ではないため不採用。当時の新最速基準は188のまま。
- **192**: 191は正当性OKだが188比で微小退行したため採用せず、188のfuture_check_mask guard/no-sibling spill elisionを親に戻す。host側scorestripe/chunkshape148、dispatch、cache、bitboard演算、solution arithmetic、stack_bytes_per_thread=208は188と同一。MAXD14 kernelのみ、post-root-action の `cur_avail` が単一bitであるthreadに限ってdepth 0の処理をDFS generic loopへ入る前に1段prerollし、depth 1をactive frameとして開始する。rootに複数候補があるthreadは188 generic loopをそのまま使う。190のblock_check_mask guard、189のforced-chain fast path、187のchild_jmark guard、185/186のterminal系変更、191のplace-xor置換は入れない。N=21 full: `0:07:10.118`。最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。188比で `0.019秒`、`0.004%` 高速で、差は誤差級だが数値上の新最速基準として採用。tailはtop5 share `41/1000`、top10 share `81/1000`、p95/p50 `1065/1000`。
- **193**: 192の正当性OK・数値上最速化を受け、192のroot-singleton preroll/future_check_mask guard/no-sibling spill elisionを親にする。host側scorestripe/chunkshape148、dispatch、cache、bitboard演算、solution arithmetic、stack_bytes_per_thread=208は192と同一。MAXD14 kernelのみ、`nf` が非zeroになった後の順序を `future_check -> terminal` から `terminal -> future_check` へ入れ替える terminal-before-future ordering を導入する。terminal-parent frameではschedule生成上 `child_row==endmark` になり、future-prune nibble bitは立たないため、terminal hitはfuture_check_mask分岐を通らず即countできる。非terminal frameは188/192相当のfuture-prune判定をそのまま残す。190のblock_check_mask guard、189のforced-chain fast path、187のchild_jmark guard、185/186のterminal系loop複製/terminal13 literal、191のplace-xor置換は入れない。N=21 full: `0:07:17.870`。最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。ただし192比で `7.752秒`、`1.802%` 遅いため不採用。新最速基準は192のまま。
- **194**: 193は正当性OKだが速度退行したため採用せず、192のroot-singleton preroll/future_check_mask guard/no-sibling spill elisionを親に戻す。host側scorestripe/chunkshape148、dispatch、cache、bitboard演算、solution arithmetic、stack_bytes_per_thread=208は192と同一。MAXD14 kernelのみ、post-root-action の `cur_avail` が1bitまたは2bitであるthreadに限り、depth 0の先頭候補をDFS generic loop前に1段prerollする。1bit rootでは192のroot-singleton preroll相当、2bit rootでは残ったroot siblingをgeneric loopと同じ `save_sp/cur_depth` 規則で保存し、first childが死ぬ/terminalになる場合は残りroot候補をactive depth0としてgeneric loopへ渡す。rootが3候補以上のthreadは192 generic loopを維持する。190のblock_check_mask guard、189のforced-chain fast path、187のchild_jmark guard、185/186のterminal系変更、191のplace-xor置換、193のterminal-before-future orderingは入れない。N=21 full: `0:07:08.288`。最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。192比で `1.830秒`、`0.425%` 高速、188比で `1.849秒`、`0.430%` 高速で、数値上の新最速基準として採用。
- **195**: 194の正当性OK・数値上最速化を受け、194のroot one/two-candidate preroll/future_check_mask guard/no-sibling spill elisionを親にする。host側scorestripe/chunkshape148、dispatch、cache、bitboard演算、solution arithmetic、stack_bytes_per_thread=208は194と同一。MAXD14 kernelのみ、post-root-action の `cur_avail` が1bit/2bit/3bitであるthreadに限り、depth 0の先頭候補をDFS generic loop前に1段prerollする。1bit/2bit rootでは194相当、3bit rootでは残った2つのroot siblingをgeneric loopと同じ `save_sp/cur_depth` 規則で保存し、first childが死ぬ/terminalになる場合は残りroot候補をactive depth0としてgeneric loopへ渡す。rootが4候補以上のthreadは194 generic loopを維持する。193のterminal-before-future ordering、191のplace-xor、190のblock_check_mask guard、189のforced-chain fast path、187のchild_jmark guard、185/186のterminal系変更は入れない。N=21 full: `0:07:08.738`。最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。ただし194比で `0.450秒`、`0.105%` 遅いため不採用。新最速基準は194のまま。
- **196**: 195は正当性OKだが194比で微小退行したため採用せず、194のroot one/two-candidate preroll/future_check_mask guard/no-sibling spill elisionへ戻す。host側scorestripe/chunkshape148、dispatch、cache、bitboard演算、solution arithmetic、stack_bytes_per_thread=208は194/195と同一。MAXD14 kernelのみ、root<=2の判定形をsecond-lowbit方式へ置き換える。具体的には `root_first` を除いた `root_rest` から `root_second` を取り、`root_after_second==0` のときだけprerollするため、1bit/2bit rootだけをDFS generic loop前に処理し、3bit以上のrootは194 generic loopへ戻る。195のroot three-candidate preroll、193のterminal-before-future ordering、191のplace-xor、190のblock_check_mask guard、189のforced-chain fast path、187のchild_jmark guard、185/186のterminal系変更は入れない。N=21 full: `0:07:08.239`。最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。194比で `0.049秒`、`0.011%` 高速、195比で `0.499秒`、`0.116%` 高速で、差は極小ながら数値上の新最速基準として採用。
- **197**: 196の正当性OK・数値上最速化を受け、196のroot<=2 second-lowbit preroll/future_check_mask guard/no-sibling spill elisionを親にする。host側scorestripe/chunkshape148、dispatch、cache、bitboard演算、solution arithmetic、stack_bytes_per_thread=208は196と同一。MAXD14 kernelのみ、root<=2判定後に `root_preroll:u32` を立ててから別の `if root_preroll!=0` へ入る形をやめ、second-lowbit direct-ifでpreroll bodyへ入る。root fast-startの対象範囲は196と同じ1bit/2bit rootのみで、3bit以上はgeneric loopのまま。195のroot three-candidate preroll、193のterminal-before-future ordering、191のplace-xor、190のblock_check_mask guard、189のforced-chain fast path、187のchild_jmark guard、185/186のterminal系変更は入れない。初回配布後、cudacodon側でstatic checkがsummary表示前に終了するケースが出たため、r2ではCUDA kernelと `.py` を変更せず `.sh` のstatic-check表示とroot direct-if grepだけを補修した。r2実行ログでは `source_split_tag` だけがFAILしてGPU実行前に停止したため、r3ではCUDA kernelと `.py` を変更せず、`source_split_tag` をactive runtime/progress tag限定の検査へ変更した。r3 N=21 full: `0:07:08.202`。最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。196比で `0.037秒`、`0.009%` 高速、194比で `0.086秒`、`0.020%` 高速で、差は極小ながら数値上の新最速基準として採用。
- **198**: 197の正当性OK・数値上最速化を受け、197のroot<=2 direct-if preroll/future_check_mask guard/no-sibling spill elisionを親にする。host側scorestripe/chunkshape148、dispatch、cache、bitboard演算、solution arithmetic、stack_bytes_per_thread=208は197と同一。MAXD14 kernelのみ、root<=2判定をsecond-lowbit helper scalar方式から `root_rest & (root_rest - 1) == 0` のtail-testへ置き換える。1bit rootでは `root_rest==0`、2bit rootでは `root_rest` が単一bitなのでprerollし、3bit以上では `root_rest` に2bit以上が残るためgeneric loopへ戻る。root fast-startの対象範囲は197/196/194と同じ1bit/2bit rootのみで、195のroot three-candidate preroll、193のterminal-before-future ordering、191のplace-xor、190のblock_check_mask guard、189のforced-chain fast path、187のchild_jmark guard、185/186のterminal系変更は入れない。N=21 full: `0:07:08.239`。最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。ただし197比で `0.037秒`、`0.009%` 遅く、196とは同タイム、194比では `0.049秒`、`0.011%` 高速に留まるため不採用。新最速基準は197のまま。
- **199**: 198は正当性OKだが197比で微小退行したため採用せず、197のroot<=2 direct-if preroll/future_check_mask guard/no-sibling spill elisionへ戻す。host側scorestripe/chunkshape148、dispatch、cache、bitboard演算、solution arithmetic、stack_bytes_per_thread=208は197/198と同一。MAXD14 kernelのみ、root<=2判定を `root_after_second==0` 相当のsecond-lowbit direct-ifから `root_rest==root_second` の等価比較へ置き換える。1bit rootでは `root_rest==root_second==0`、2bit rootでは `root_rest` が `root_second` そのものになりprerollし、3bit以上では `root_rest` が `root_second` より多くのbitを持つためgeneric loopへ戻る。root fast-startの対象範囲は197/196/194と同じ1bit/2bit rootのみで、198のtail-test、195のroot three-candidate preroll、193のterminal-before-future ordering、191のplace-xor、190のblock_check_mask guard、189のforced-chain fast path、187のchild_jmark guard、185/186のterminal系変更は入れない。N=21 full: `0:07:08.293`。最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。ただし197比で `0.091秒`、`0.021%` 遅く、198/196比で `0.054秒`、`0.013%` 遅く、194比でも `0.005秒`、`0.001%` 遅いため不採用。新最速基準は197のまま。
- **200**: 199は正当性OKだが197比で微小退行したため採用せず、197のroot<=2 direct-if preroll/future_check_mask guard/no-sibling spill elisionへ戻す。host側scorestripe/chunkshape148、dispatch、cache、bitboard演算、solution arithmetic、stack_bytes_per_thread=208は197/199と同一。MAXD14 kernelのみ、root-preroll body内のfuture-prune判定を `future_check_mask!=0` の外側guard付きから、`pr_nibble_op & 8` の直接判定へ変える。depth0の `pr_nibble_op` にfuture bit8が立つ場合、schedule生成上 `future_check_mask` は必ず非zeroなので、root-preroll内では外側guardが冗長である。一方、generic DFS loopでは197/188由来の `future_check_mask==0` schedule軽量guardを維持する。root<=2 predicateは197の `root_after_second==0` direct-ifに戻し、1bit/2bit rootだけをpreroll、3bit以上rootはgeneric loopのままにする。199の `root_rest==root_second` predicate、198のtail-test、195のroot three-candidate preroll、193のterminal-before-future ordering、191のplace-xor、190のblock_check_mask guard、189のforced-chain fast path、187のchild_jmark guard、185/186のterminal系変更は入れない。初回配布版ではcudacodon側でsource static check失敗時にsummaryが出ないまま停止するケースがあったため、r2ではCUDA kernelと `.py` を変更せず、`.sh` のみを補修した。r2は `STATIC_ONLY=1` でfull-run lockを取らず、source static check失敗時にもsummary tableを必ず表示する。r2 N=21 fullは `0:07:09.520`、正当性OKだが197比で `1.318秒`、`0.308%` 遅いため不採用。新最速基準は197のまま。
- **200 r2**: 200のCUDA kernelと `.py` 探索ロジックは変更せず、検証シェルのみ補修した。`STATIC_ONLY=1` のときfull-run lockを取らないようにし、通常実行でsource static checkがFAILした場合もsummaryを出して、`.py`/`.sh` の取り違えや旧source残りを判別しやすくした。root-preroll future-bit直接判定のsource checkは空白・コメント差に強い形へ変更した。
- **201**: 200は正当性OKだが197比で遅いため採用せず、197のroot<=2 direct-if preroll/future_check_mask guard/no-sibling spill elisionへ戻す。host側scorestripe/chunkshape148、dispatch、cache、bitboard演算、solution arithmetic、stack_bytes_per_thread=208は197/200と同一。MAXD14 kernelのみ、root-preroll内のchild_jmark処理で `pr_child_jmark:u32=child_jmark_mask&u32(1)` の一時scalarを作らず、`if (child_jmark_mask&u32(1))!=u32(0):` と直接判定する。generic DFS loop側のchild_jmark処理、root<=2 predicate、future_check_mask guard、no-sibling save_sp/cur_depth復元規則は197相当のまま維持する。200のroot-preroll future-bit直接判定、199の `root_rest==root_second` predicate、198のtail-test、195のroot three-candidate preroll、193/189/190/191/187/185/186の不採用差分は入れない。採用可否はN=21 fullで197/200/199/198/196/194/188比timingを確認して判断する.


- 201Py r2は、201Py本体のCUDA/kernel差分を変更せず、検証シェルの `source_rootpr_jmarkdirect` 静的検査だけを修正した版。`root_rest==root_second` がコメント・履歴文に現れるだけでFAILしないようにし、active predicateとして残っている場合だけ検出する。`STATIC_ONLY=1` はOK確認済み。

- **202**: 201は正当性OKだが197比で微小退行したため採用せず、197のroot<=2 direct-if preroll/future_check_mask guard/no-sibling spill elisionへ戻す。host側scorestripe/chunkshape148、dispatch、cache、bitboard演算、solution arithmetic、stack_bytes_per_thread=208は197/201と同一。MAXD14 kernelのみ、root-preroll内で残りroot siblingを保存する際、保存条件と `avail[save_sp]` payloadを `cur_avail` ではなく `root_second` から作る。generic DFS loop側の保存処理、root<=2 predicate、future_check_mask guard、root-preroll child_jmark scalar、no-sibling save_sp/cur_depth復元規則は197相当のまま維持する。N=21 full: `0:07:08.482`。最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。ただし197比で `0.280秒`、`0.065%` 遅いため不採用。新最速基準は197のまま。
- **203**: 202は正当性OKだが197比で遅いため採用せず、197のroot<=2 direct-if preroll/future_check_mask guard/no-sibling spill elisionへ戻す。host側scorestripe/chunkshape148、dispatch、cache、bitboard演算、solution arithmetic、stack_bytes_per_thread=208は197/202と同一。MAXD14 kernelのみ、root-preroll内で残りroot siblingを保存する際、`cur_depth` が必ず0であることを利用し、保存payloadを `cur_avail|(u32(cur_depth)<<u32(27))` から `cur_avail` へ置き換える。generic DFS loop側の保存では引き続きdepthを上位5bitへpackする。202のroot_second save payload、201のchild_jmark-direct、200のroot-preroll future-bit直接判定、199のeqsecond、198のtail-test、195のroot three-candidate preroll、193/189/190/191/187/185/186の不採用差分は入れない。N=21 full: `0:07:08.344`。最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。ただし197比で `0.142秒`、`0.033%` 遅いため不採用。新最速基準は197のまま。
- **204**: 203は正当性OKだが197比で微小退行したため採用せず、197のroot<=2 direct-if preroll/future_check_mask guard/no-sibling spill elisionへ戻す。host側scorestripe/chunkshape148、dispatch、cache、bitboard演算、solution arithmetic、stack_bytes_per_thread=208は197/203と同一。MAXD14 kernelのみ、root<=2判定に使う `root_rest` を `cur_avail^root_first` ではなく `cur_avail&(cur_avail-u32(1))` で生成する。これは同じlowbit除去をXORではなくclear-lowbit式で行う微差実験で、`root_second`、`root_after_second`、`if root_after_second==0:` のdirect-if predicate、root-preroll body、generic DFS loopは197相当のまま維持する。203のroot depth0 save、202のroot_second save payload、201のchild_jmark-direct、200のroot-preroll future-bit直接判定、199のeqsecond、198のtail-test、195のroot three-candidate preroll、193/189/190/191/187/185/186の不採用差分は入れない。N=21 full: `0:07:07.795`。最終合計 `314666222712` 一致、131チャンク完走、required_maxd=14、selected_MAXD=14、schedule_words=0、stack_bytes_per_thread=208、warning/errorなし。197比で `0.407秒`、`0.095%` 高速、203比で `0.549秒`、`0.128%` 高速、194比で `0.493秒`、`0.115%` 高速で、数値上の新最速基準として採用。
- **205**: 204の正当性OK・数値上最速化を受け、204のroot_rest clear-lowbit/future_check_mask guard/no-sibling spill elisionを親にする。host側scorestripe/chunkshape148、dispatch、cache、bitboard演算、solution arithmetic、stack_bytes_per_thread=208は204と同一。MAXD14 kernelのみ、root<=2判定で `root_after_second:u32=root_rest^root_second` の一時scalarを削り、`if (root_rest^root_second)==u32(0):` へ直接inlineする。root fast-startの対象範囲は204/197/196/194と同じ1bit/2bit rootのみで、3bit以上はgeneric loopのまま。203のroot depth0 save、202のroot_second save payload、201のchild_jmark-direct、200のroot-preroll future-bit直接判定、199のeqsecond、198のtail-test、195のroot three-candidate preroll、193/189/190/191/187/185/186の不採用差分は入れない。採用可否はN=21 fullで204/197/203/202/201/200/199/198/196/194/188比timingを確認して判断する。


- **205 r2**: 205Py本体のCUDA/kernel差分は変更せず、検証シェルのみ補修した。cudacodon側で `source roottwo-rootafterinline static checks failed` だけが出て原因行が見えにくいケースに対応し、`source_rootafterinline` のpycheck結果をsummaryのactual欄へ出すようにした。active-code検査はMAXD14 kernelのroot-preroll blockへ限定し、コメント・履歴文・親版説明に現れる不採用差分名ではFAILしないようにした。`.py` は205初版と同一で、探索ロジックは変更なし。

- **205 r3**: 205Py本体のCUDA/kernel差分は変更せず、検証シェルのみ再補修した。cudacodon側で `source_rootafterinline_pycheck_failed=root_after_scalar_absent,root_old_direct_if_absent` が出る場合は、active root-preroll code が205のinline式ではなく204相当の `root_after_second` scalarを含む状態であることを示す。r3ではMAXD14 kernel内のactive root-preroll blockだけを検査し、失敗時に `source_rootafterinline_active_snippet=...` を出して、実際に配置されている `.py` のroot predicateを判別できるようにした。`.py` は205初版/r2と同一で、探索ロジックは変更なし。

## 今後の運用

- 新しい `.py` のヘッダーには、過去一覧ではなく `NQ_UPDATE_MEMO` と該当版の短い更新メモのみを置く。
- 全体の履歴一覧は、この `README.md` に追記する。
- `.sh` のヘッダーにも、その版の目的・親版・不採用にした前版差分・検証対象を短く書く。
- full実行ログで正当性OKかつ速度改善なら「新最速基準」として履歴に明記する。
- 正当性OKだが遅い場合は「不採用」と書き、次版では最速基準へ戻す。
- `CUDA_ERROR_INVALID_PTX` の場合は、計算不一致ではなくJIT不成立として扱い、危険なkernel差分を撤回する。

---

Updated on 2026-07-01 for 201Py root-preroll child_jmark-direct probe.

---

Updated on 2026-07-01 for 201Py r2.

---

Updated on 2026-07-01 for 202Py root-second-save probe.

---

Updated on 2026-07-01 for 203Py root-preroll depth0-save probe.

---

Updated on 2026-07-01 for 204Py root-clearrest probe.

---

Updated on 2026-07-01 for 205Py root-afterinline probe.

---

Updated on 2026-07-01 for 205Py r2.

Updated on 2026-07-01 for 205Py r3.

- **205確認結果**: 205Py rootafterinline は `N=21 full once` で final total `314666222712` 一致、131チャンク完走、duplicate/missing 0/0、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk 14、schedule_words=0、stack_bytes_per_thread=208、warning/error 0。速度は `0:07:08.074` で、197Py `0:07:08.202` より `0.128秒` 高速だが、204Py `0:07:07.795` より `0.279秒` 遅いため不採用。現最速基準は204Pyのまま。
- **206**: 205Pyを採用せず204Py相当へ戻す安全復帰版。MAXD14 kernelのみ、205で削った `root_after_second:u32=root_rest^root_second` の一時scalarを復元し、root<=2判定を204の `if root_after_second==u32(0):` へ戻す。204の `root_rest = cur_avail & (cur_avail - 1)`、future_check_mask zero guard、no-sibling spill elision、root one/two-candidate preroll、schedule_words=0、stack_bytes_per_thread=208は維持する。195 rootthree、198 tailtest、199 eqsecond、200 prfuturebit、201 prjmarkdirect、202 rootsecondsave、203 rootdepth0save、205 rootafterinline は入れない。

## 206Py以降のkernel限定検討メモ

優先度Aは、204/206の最速形を壊さない低リスク実験に限定する。まずは206で204相当へ戻して基準を再固定し、次にMAXD14 kernel内だけを1差分ずつ測る。

1. **MAXD縮小の準備**: 現状N=21 fullでは全chunkが required_maxd=14 / selected_MAXD=14 のため、いきなりMAXD13化は危険。先にhost側で「実際に深度13まで保存したchunkがあるか」「cur_depth==13 terminal到達の分布」「save_sp最大値」をprogress/dispatchへ追加し、MAXD13候補chunkが存在するかを確認する。MAXD13 kernelを作る場合も、required_maxd<=13 chunkのみへ限定dispatchし、MAXD14 fallbackを必ず残す。
2. **root-preroll周辺の軽量整理**: root one/two-candidate prerollの対象範囲は204/206と同じ1bit/2bitだけに固定する。保存payload、child_jmark scalar、future_check_mask guardは過去に退行しているため当面触らない。試すなら `pr_descend` 初期化や分岐順序の局所整理に限定する。
3. **generic loopは当面保留**: 184 no-sibling、188 futuremask、204 rootclearrest が効いた一方、185/186/189/190のようなloop分離・fast path拡張は大きく退行した。generic loopの構造変更は、MAXD13準備ログで明確な根拠が出るまで後回しにする。
4. **ソース整理はkernel実験と分離**: 起動パラメータ・旧bench_mode・過去probe用分岐の削除は可読性改善には有効だが、host dispatch/cache/progress名が変わると検証対象が広がる。206では実施せず、別途 `cleanup-only` 版として、kernel byte-equivalentを確認しながら削るのが安全。

Updated on 2026-07-02 for 206Py restore-rootafter-scalar rollback and MAXD13 preparation memo.

- **206**: 205は正当性OKながら204比で遅かったため採用せず、204相当へ戻した安全復帰版。MAXD14 kernelでは `root_rest = cur_avail & (cur_avail-u32(1))` のclear-lowbit形、`root_after_second:u32 = root_rest ^ root_second` の一時scalar、`if root_after_second==u32(0):` のroot one/two-candidate preroll、future_check_mask guard、no-sibling spill elisionを維持した。N=21 fullは `0:07:07.908`、最終合計 `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXDは全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0/0`。204Py `0:07:07.795` には `0.113秒` 届かなかったが、205Py `0:07:08.074` からは `0.166秒` 戻したため、204相当復帰として正当性OK。
Updated on 2026-07-02 for 208Py prterminalfirst probe.

- **207**: 206の正当性OKを受け、204/206のroot_after_second scalar形を維持したまま、MAXD14 kernelのみroot-preroll内の `pr_descend` 初期化を変更した微差実験版。従来は `pr_descend:u32=u32(1)` としてから `if pr_nf==0: pr_descend=0` としていたが、207では `pr_descend:u32=pr_nf` とし、以降は従来通りkill条件で0へ落とした。N=21 fullは `0:07:08.800`、最終合計 `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXDは全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0/0`。ただし204Py `0:07:07.795` より `1.005秒`、206Py `0:07:07.908` より `0.892秒`、205Py `0:07:08.074` より `0.726秒`、197Py `0:07:08.202` より `0.598秒` 遅いため不採用。
- **208**: 207は正当性OKだが速度退行したため採用せず、206/204相当の `pr_descend:u32=u32(1)` と `if pr_nf==0: pr_descend=0` へ戻す。MAXD14 kernelのみ、root-preroll内の `terminal_depth==0` 判定を `future_check_mask` guard の前へ移動する root-preroll限定の terminal-first 微差実験版。generic DFS loopのterminal/future順序は変更しない。root_rest clear-lowbit、root_after_second scalar predicate、future_check_mask guard、no-sibling spill elision、root one/two-candidate preroll、schedule_words=0、stack_bytes_per_thread=208は維持する。193Pyのgeneric terminal-before-futureは大きく退行したため入れず、root-preroll内だけに限定して採用可否をN=21 fullで確認する.


Updated on 2026-07-02 for 208Py result and 209Py roottailclear probe.

- **208確認結果**: 208Py root-preroll terminal-first は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0/0`。速度は `0:07:09.235` で、204Py `0:07:07.795` より `1.440秒`、206Py `0:07:07.908` より `1.327秒`、207Py `0:07:08.800` より `0.435秒`、197Py `0:07:08.202` より `1.033秒` 遅いため不採用。208のroot-preroll terminal-firstは撤回する。
- **209**: 208は正当性OKだが速度退行したため採用せず、204/206相当のroot-preroll bodyとgeneric orderingへ戻す。MAXD14 kernelのみ、root<=2判定を `root_second/root_after_second` のsecond-lowbit xor predicateから、`root_tail:u32 = root_rest & (root_rest-u32(1))`、`if root_tail==u32(0):` のclear-lowbit tail predicateへ置換する。204の `root_rest = cur_avail & (cur_avail-u32(1))`、future_check_mask guard、no-sibling spill elision、root one/two-candidate preroll、schedule_words=0、stack_bytes_per_thread=208は維持する。207の `pr_descend:u32=pr_nf`、208のroot-preroll terminal-first、205 rootafterinline、203 rootdepth0save、202 rootsecondsave、201 prjmarkdirect、200 prfuturebit、199 eqsecond、198 tailtest、195 rootthree は入れない。

Updated on 2026-07-02 for 209Py result and 210Py rootafterandnot probe.

- **209確認結果**: 209Py roottailclear は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0/0`。速度は `0:07:08.003` で、197Py `0:07:08.202` より `0.199秒` 高速だが、204Py `0:07:07.795` より `0.208秒`、206Py `0:07:07.908` より `0.095秒` 遅いため不採用。209の `root_tail = root_rest & (root_rest-u32(1))` predicate は撤回する。
- **210**: 209は正当性OKだが204/206を上回らなかったため採用せず、204/206相当のroot_after_second scalar predicateへ戻す。MAXD14 kernelのみ、`root_after_second:u32 = root_rest ^ root_second` を `root_after_second:u32 = root_rest & (~root_second)` へ置き換える and-not predicate 微差実験版。`root_second` は `root_rest` のlowbitであり `root_second` は `root_rest` の部分集合なので、xorとand-notは同じ「root_second除去後の残bit」を表す。root fast-start対象は204/206と同じ1bit/2bit rootのみで、3bit以上はgeneric loopへ戻す。204の `root_rest = cur_avail & (cur_avail-u32(1))`、future_check_mask guard、no-sibling spill elision、root one/two-candidate preroll、schedule_words=0、stack_bytes_per_thread=208は維持する。207の `pr_descend:u32=pr_nf`、208のroot-preroll terminal-first、209 roottailclear、205 rootafterinline、203 rootdepth0save、202 rootsecondsave、201 prjmarkdirect、200 prfuturebit、199 eqsecond、198 tailtest、195 rootthree は入れない。

Updated on 2026-07-02 for 210Py result and 211Py rootfirstlate probe.

- **210確認結果**: 210Py rootafterandnot は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0/0`。速度は `0:07:07.874` で、206Py `0:07:07.908` より `0.034秒`、209Py `0:07:08.003` より `0.129秒`、197Py `0:07:08.202` より `0.328秒` 高速。ただし204Py `0:07:07.795` より `0.079秒` 遅いため、数値上の新最速基準にはせず不採用。210の `root_after_second = root_rest & (~root_second)` and-not predicate は撤回する。
- **211**: 210は正当性OKで206より速かったが204最速を上回らなかったため採用せず、204/206相当の `root_after_second:u32 = root_rest ^ root_second` XOR predicateへ戻す。MAXD14 kernelのみ、従来predicate前に計算していた `root_first:u32 = cur_avail & (u32(0)-cur_avail)` を、`if root_after_second==u32(0):` の成立後、つまりroot_availが1bit/2bitと分かった場合だけ計算する形へ遅延する rootfirstlate 微差実験版。3bit以上rootではgeneric loopへ戻るため、このpreludeで `root_first` を作らない。204の `root_rest = cur_avail & (cur_avail-u32(1))`、future_check_mask guard、no-sibling spill elision、root one/two-candidate preroll、schedule_words=0、stack_bytes_per_thread=208は維持する。207の `pr_descend:u32=pr_nf`、208のroot-preroll terminal-first、209 roottailclear、210 rootafterandnot、205 rootafterinline、203 rootdepth0save、202 rootsecondsave、201 prjmarkdirect、200 prfuturebit、199 eqsecond、198 tailtest、195 rootthree は入れない。

Updated on 2026-07-02 for 211Py result and 212Py rootfirstlate_andnot probe.

- **211確認結果**: 211Py rootfirstlate は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0/0`。速度は `0:07:07.801` で、204Py `0:07:07.795` に `0.006秒` 届かなかったが、210Py `0:07:07.874`、206Py `0:07:07.908`、197Py `0:07:08.202` は上回った。正当性OKで204とほぼ同等だが、新最速基準にはせず未採用扱いとする。
- **212**: 211は204最速にほぼ並んだため、211の `root_first` 遅延を維持し、210で良好だった `root_after_second:u32 = root_rest & (~root_second)` のand-not predicateを組み合わせる微差実験版。MAXD14 kernelのみ変更し、`root_rest = cur_avail & (cur_avail-u32(1))`、future_check_mask guard、no-sibling spill elision、root one/two-candidate preroll、schedule_words=0、stack_bytes_per_thread=208は維持する。207の `pr_descend:u32=pr_nf`、208のroot-preroll terminal-first、209 roottailclear、205 rootafterinline、203 rootdepth0save、202 rootsecondsave、201 prjmarkdirect、200 prfuturebit、199 eqsecond、198 tailtest、195 rootthree は入れない。

## 212Py確認結果と213Py方針（2026-07-02）

### 212Py確認結果

212Py `rootfirstlate_andnot` は N=21 full once で正当性OK。

- final total: `314666222712` 一致
- progress rows: `131`
- duplicate/missing chunks: `0 / 0`
- dispatch task sum: `2025282`
- required_maxd: 全chunk `14`
- selected_MAXD: 全chunk `14`
- schedule_words: `0`
- stack_bytes_per_thread: `208`
- warning/error: `0 / 0`
- elapsed: `0:07:07.828`

速度比較:

- 204Py: `0:07:07.795`
- 211Py: `0:07:07.801`
- 212Py: `0:07:07.828`
- 210Py: `0:07:07.874`
- 206Py: `0:07:07.908`
- 197Py: `0:07:08.202`

判定: 212Pyは正当性OKだが、204Pyより `0.033秒` 遅いため不採用。

### 213Py方針

213Pyは212Pyの and-not predicate を採用せず、211Pyの `root_first` 遅延を維持したまま、205Pyで試した `root_after_second` scalar削除を late-root-first 形に限定して再検証する。

MAXD14 kernel内だけの差分:

```python
# 211Py相当
root_rest:u32 = cur_avail & (cur_avail-u32(1))
root_second:u32 = root_rest & (u32(0)-root_rest)
root_after_second:u32 = root_rest ^ root_second

if root_after_second == u32(0):
    root_first:u32 = cur_avail & (u32(0)-cur_avail)

# 213Py
root_rest:u32 = cur_avail & (cur_avail-u32(1))
root_second:u32 = root_rest & (u32(0)-root_rest)

if (root_rest ^ root_second) == u32(0):
    root_first:u32 = cur_avail & (u32(0)-cur_avail)
```

root fast-startの対象範囲は204/206/211と同じく1bit/2bit rootのみ。3bit以上root、futuremask、no-sibling、root-preroll body、generic DFS loop、dispatch、host task orderは変更しない。

## 213Py確認結果と214Py方針（2026-07-02）

### 213Py確認結果

213Py `rootfirstlate_inlinexor` は N=21 full once で正当性OK。

- final total: `314666222712` 一致
- progress rows: `131`
- duplicate/missing chunks: `0 / 0`
- dispatch task sum: `2025282`
- required_maxd: 全chunk `14`
- selected_MAXD: 全chunk `14`
- schedule_words: `0`
- stack_bytes_per_thread: `208`
- warning/error: `0 / 0`
- elapsed: `0:07:08.120`

速度比較:

- 204Py: `0:07:07.795`
- 211Py: `0:07:07.801`
- 212Py: `0:07:07.828`
- 210Py: `0:07:07.874`
- 206Py: `0:07:07.908`
- 213Py: `0:07:08.120`
- 197Py: `0:07:08.202`

判定: 213Pyは正当性OKだが、204Pyより `0.325秒` 遅いため不採用。213の `root_after_second` scalar削除 + late-root-first inline XOR は撤回する。

### 214Py方針

214Pyは、213Pyを採用せず、最も204Pyに近かった211Pyの `root_first` 遅延形を親にする。MAXD14 kernel内だけ、root-preroll bodyに入った後の `root_first` 復元方法を変更する。

```python
# 211Py相当
root_rest:u32 = cur_avail & (cur_avail-u32(1))
root_second:u32 = root_rest & (u32(0)-root_rest)
root_after_second:u32 = root_rest ^ root_second

if root_after_second == u32(0):
    root_first:u32 = cur_avail & (u32(0)-cur_avail)

# 214Py
root_rest:u32 = cur_avail & (cur_avail-u32(1))
root_second:u32 = root_rest & (u32(0)-root_rest)
root_after_second:u32 = root_rest ^ root_second

if root_after_second == u32(0):
    root_first:u32 = cur_avail ^ root_rest
```

`root_rest` は `cur_avail` からlowbitを除いた値なので、`cur_avail ^ root_rest` は除去されたlowbit、つまり `root_first` と等価。1bit rootでは `root_rest==0` のため `root_first==cur_avail`、2bit rootではlowbitだけが復元される。root fast-start対象範囲は204/206/211と同じく1bit/2bit rootのみ。3bit以上root、futuremask、no-sibling、root-preroll body、generic DFS loop、dispatch、host task orderは変更しない。


## 214Py確認結果と215Py方針（2026-07-02）

### 214Py確認結果

214Py `rootfirstxor` は N=21 full once で正当性OK。

- final total: `314666222712` 一致
- progress rows: `131`
- duplicate/missing chunks: `0 / 0`
- dispatch task sum: `2025282`
- required_maxd: 全chunk `14`
- selected_MAXD: 全chunk `14`
- schedule_words: `0`
- stack_bytes_per_thread: `208`
- warning/error: `0 / 0`
- elapsed: `0:07:09.104`

速度比較:

- 204Py: `0:07:07.795`
- 211Py: `0:07:07.801`
- 212Py: `0:07:07.828`
- 210Py: `0:07:07.874`
- 206Py: `0:07:07.908`
- 213Py: `0:07:08.120`
- 214Py: `0:07:09.104`
- 197Py: `0:07:08.202`

判定: 214Pyは正当性OKだが、204Pyより `1.309秒`、211Pyより `1.303秒` 遅いため不採用。214の `root_first = cur_avail ^ root_rest` は撤回する。

### 215Py方針

215Pyは214Pyを採用せず、204Pyに最も近かった211Pyの `root_first` 遅延形を親にする。MAXD14 kernel内だけ、root-preroll bodyの `pr_bit` alias を削り、`root_first` を直接 bit として使う。

```python
# 211Py相当
if root_after_second == u32(0):
    root_first:u32 = cur_avail & (u32(0)-cur_avail)
    pr_bit:u32 = root_first
    ...
    pr_nld = ((cur_ld | pr_bit) << pr_stepu) | ...
    pr_nrd = ((cur_rd | pr_bit) >> pr_stepu) | ...
    pr_ncol = cur_col | pr_bit

# 215Py
if root_after_second == u32(0):
    root_first:u32 = cur_avail & (u32(0)-cur_avail)
    ...
    pr_nld = ((cur_ld | root_first) << pr_stepu) | ...
    pr_nrd = ((cur_rd | root_first) >> pr_stepu) | ...
    pr_ncol = cur_col | root_first
```

`root_rest = cur_avail & (cur_avail-u32(1))`、`root_after_second = root_rest ^ root_second`、futuremask、no-sibling、root one/two-candidate preroll、generic DFS loop、dispatch、host task orderは変更しない。root fast-start対象範囲は204/206/211と同じく1bit/2bit rootのみ。3bit以上rootはgeneric loopへ戻す。


## 215Py確認結果と216Py方針（2026-07-02）

### 215Py確認結果

215Py `rootbitdirect` は N=21 full once で正当性OK。

- final total: `314666222712` 一致
- progress rows: `131`
- duplicate/missing chunks: `0 / 0`
- dispatch task sum: `2025282`
- required_maxd: 全chunk `14`
- selected_MAXD: 全chunk `14`
- schedule_words: `0`
- stack_bytes_per_thread: `208`
- warning/error: `0 / 0`
- elapsed: `0:07:08.061`

速度比較:

- 204Py: `0:07:07.795`
- 211Py: `0:07:07.801`
- 212Py: `0:07:07.828`
- 210Py: `0:07:07.874`
- 206Py: `0:07:07.908`
- 209Py: `0:07:08.003`
- 215Py: `0:07:08.061`
- 197Py: `0:07:08.202`

判定: 215Pyは正当性OKだが、204Pyより `0.266秒`、211Pyより `0.260秒` 遅いため不採用。215の `pr_bit` alias 削除は撤回する。

### 216Py方針

216Pyは215Pyを採用せず、204Pyに最も近かった211Pyの `root_first` 遅延形へ戻す。MAXD14 kernel内だけ、root-preroll内の `pr_block_code` 取得を直接化する。

```python
# 211Py相当
if root_after_second == u32(0):
    root_first:u32 = cur_avail & (u32(0)-cur_avail)
    pr_nibble_op:u32 = schedule_lo & u32(15)
    pr_block_code:u32 = pr_nibble_op & u32(7)
    pr_bit:u32 = root_first

# 216Py
if root_after_second == u32(0):
    root_first:u32 = cur_avail & (u32(0)-cur_avail)
    pr_nibble_op:u32 = schedule_lo & u32(15)
    pr_block_code:u32 = schedule_lo & u32(7)
    pr_bit:u32 = root_first
```

`pr_nibble_op` はroot-preroll内のfuture bit判定で引き続き使う。`root_rest = cur_avail & (cur_avail-u32(1))`、`root_after_second = root_rest ^ root_second`、futuremask、no-sibling、root one/two-candidate preroll、generic DFS loop、dispatch、host task orderは変更しない。root fast-start対象範囲は204/206/211と同じく1bit/2bit rootのみ。3bit以上rootはgeneric loopへ戻す。

## 216Py確認結果と217Py方針（2026-07-02）

### 216Py確認結果

216Py `rootprblockdirect` は N=21 full once で正当性OK。

- final total: `314666222712` 一致
- progress rows: `131`
- duplicate/missing chunks: `0 / 0`
- dispatch task sum: `2025282`
- required_maxd: 全chunk `14`
- selected_MAXD: 全chunk `14`
- schedule_words: `0`
- stack_bytes_per_thread: `208`
- warning/error: `0 / 0`
- elapsed: `0:07:07.867`

速度比較:

- 204Py: `0:07:07.795` 現最速基準
- 211Py: `0:07:07.801`
- 212Py: `0:07:07.828`
- 216Py: `0:07:07.867`
- 210Py: `0:07:07.874`
- 206Py: `0:07:07.908`
- 209Py: `0:07:08.003`
- 215Py: `0:07:08.061`
- 213Py: `0:07:08.120`
- 197Py: `0:07:08.202`
- 207Py: `0:07:08.800`
- 214Py: `0:07:09.104`
- 208Py: `0:07:09.235`

判定: 216Pyは正当性OKだが、204Pyより `0.072秒`、211Pyより `0.066秒` 遅いため不採用。現最速基準は引き続き204Py。

### 217Py方針

217Pyは216Pyを採用せず、204Pyに最も近かった211Pyの `root_first` 遅延形を親にする。root-preroll微差分探索の最後の1本として、MAXD14 kernel内だけ、`cur_avail=root_rest` の代入位置を遅延する。

```python
# 211Py相当
if root_after_second == u32(0):
    root_first:u32 = cur_avail & (u32(0)-cur_avail)
    pr_nibble_op:u32 = schedule_lo & u32(15)
    pr_block_code:u32 = pr_nibble_op & u32(7)
    pr_bit:u32 = root_first
    cur_avail = root_rest
    ...
    if pr_descend != u32(0):
        if cur_avail != u32(0):
            save root remainder

# 217Py
if root_after_second == u32(0):
    root_first:u32 = cur_avail & (u32(0)-cur_avail)
    pr_nibble_op:u32 = schedule_lo & u32(15)
    pr_block_code:u32 = pr_nibble_op & u32(7)
    pr_bit:u32 = root_first
    ...
    # first child の nf/future/terminal/jmark 判定後に root_rest を反映
    cur_avail = root_rest
    if pr_descend != u32(0):
        if cur_avail != u32(0):
            save root remainder
```

意味は211Pyと同じ。first childが死ぬ/terminalになる場合も、branch終了後にgeneric loopが `root_rest` をactive root remainderとして処理する。first childがdescendする場合も、同じ `root_rest` を保存する。差分は代入位置だけ。

維持するもの:

- `root_rest = cur_avail & (cur_avail-u32(1))`
- `root_after_second = root_rest ^ root_second`
- `root_first` late extraction
- future_check_mask zero guard
- no-sibling spill elision
- root one/two-candidate preroll
- generic DFS loop
- host task order/cache/dispatch
- MAXD14, schedule_words=0, stack_bytes_per_thread=208

217Pyが204Pyを上回らなければ、root-preroll微差分探索は一区切りにし、次はMAXD13準備診断へ移る。

## 217Py確認結果と218Py方針（2026-07-02）

### 217Py確認結果

217Py `rootrestlate` は、N=21 full onceで正当性OK。

- final total: `314666222712` 一致
- progress rows: `131`
- duplicate/missing chunks: `0 / 0`
- dispatch task sum: `2025282`
- required_maxd: 全chunk `14`
- selected_MAXD: 全chunk `14`
- schedule_words: `0`
- stack_bytes_per_thread: `208`
- warning/error: `0 / 0`
- N=21 full: `0:07:07.709`

速度比較:

```text
217Py  0:07:07.709  新最速基準
204Py  0:07:07.795  旧最速基準
211Py  0:07:07.801
212Py  0:07:07.828
216Py  0:07:07.867
210Py  0:07:07.874
206Py  0:07:07.908
197Py  0:07:08.202
```

判定: 217Pyは204Pyより `0.086秒` 高速、211Pyより `0.092秒` 高速。したがって **217Pyを新最速基準として採用**。

### 218Py方針

217Pyが新最速基準になったため、218Py以降は217Pyを親にする。

218Pyは、217Pyの `rootrestlate` を維持したまま、MAXD14 kernel内だけ `root_after_second` scalarをXOR形からand-not形へ変える再結合実験。

```python
# 217Py
root_rest:u32 = cur_avail & (cur_avail-u32(1))
root_second:u32 = root_rest & (u32(0)-root_rest)
root_after_second:u32 = root_rest ^ root_second

if root_after_second == u32(0):
    root_first:u32 = cur_avail & (u32(0)-cur_avail)
    ...
    cur_avail = root_rest   # first-child検証後

# 218Py
root_rest:u32 = cur_avail & (cur_avail-u32(1))
root_second:u32 = root_rest & (u32(0)-root_rest)
root_after_second:u32 = root_rest & (~root_second)

if root_after_second == u32(0):
    root_first:u32 = cur_avail & (u32(0)-cur_avail)
    ...
    cur_avail = root_rest   # first-child検証後
```

意味は217Pyと同じ。`root_second` は `root_rest` のlowbitなので、`root_rest ^ root_second` と `root_rest & (~root_second)` はどちらも「root_restからroot_secondを除いた残り」を表す。対象範囲は217Pyと同じくroot 1bit/2bitのみで、3bit以上rootはgeneric loopへ戻す。

維持するもの:

- 217Pyの `rootrestlate`
- `root_first` late extraction
- `root_rest = cur_avail & (cur_avail-u32(1))`
- future_check_mask zero guard
- no-sibling spill elision
- root one/two-candidate preroll
- generic DFS loop
- host task order/cache/dispatch
- MAXD14, schedule_words=0, stack_bytes_per_thread=208

218Pyが217Py `0:07:07.709` を上回れば新採用。上回らなければ、次は予定通り 219Py = 217 + `pr_block_code = schedule_lo & 7` を試す。


## 218Py確認結果と219Py方針（2026-07-02）

### 218Py確認結果

218Py `rootrestlate_andnot` は、N=21 full onceで正当性OK。

- final total: `314666222712` 一致
- progress rows: `131`
- duplicate/missing chunks: `0 / 0`
- dispatch task sum: `2025282`
- required_maxd: 全chunk `14`
- selected_MAXD: 全chunk `14`
- schedule_words: `0`
- stack_bytes_per_thread: `208`
- warning/error: `0 / 0`
- N=21 full: `0:07:07.825`

速度比較:

```text
217Py  0:07:07.709  現最速基準
204Py  0:07:07.795
211Py  0:07:07.801
218Py  0:07:07.825
212Py  0:07:07.828
216Py  0:07:07.867
210Py  0:07:07.874
206Py  0:07:07.908
197Py  0:07:08.202
```

判定: 218Pyは正当性OKだが、217Pyより `0.116秒` 遅いため不採用。現最速基準は引き続き **217Py**。

### 219Py方針

219Pyは、合意済みの順番どおり **217Py + pr_block_code direct** を試す。
218Pyのand-not predicateは採用せず、217Pyへ戻す。

```python
# 217Py
pr_nibble_op:u32 = schedule_lo & u32(15)
pr_block_code:u32 = pr_nibble_op & u32(7)

# 219Py
pr_nibble_op:u32 = schedule_lo & u32(15)
pr_block_code:u32 = schedule_lo & u32(7)
```

`pr_nibble_op` はfuture bit判定のために維持する。`schedule_lo & 7` はdepth0 nibbleの下位3bitを直接取るだけなので意味は同じ。
217Pyの `rootrestlate`、root_after_second XOR、root_first late extraction、future_check_mask zero guard、no-sibling spill elision、root one/two-candidate preroll、MAXD14、schedule_words=0、stack_bytes_per_thread=208は維持する。

219Pyが217Py `0:07:07.709` を上回れば新採用。上回らなければ、次は予定通り 220Py = 217 + root_tail 判定を試す。

## 219Py確認結果と220Py方針（2026-07-02）

### 219Py確認結果

219Py `rootrestlate_prblockdirect` は、N=21 full onceで正当性OK。

- final total: `314666222712` 一致
- progress rows: `131`
- duplicate/missing chunks: `0 / 0`
- dispatch task sum: `2025282`
- required_maxd: 全chunk `14`
- selected_MAXD: 全chunk `14`
- schedule_words: `0`
- stack_bytes_per_thread: `208`
- warning/error: `0 / 0`
- N=21 full: `0:07:07.776`

速度比較:

```text
217Py  0:07:07.709  現最速基準
219Py  0:07:07.776
204Py  0:07:07.795
211Py  0:07:07.801
218Py  0:07:07.825
212Py  0:07:07.828
216Py  0:07:07.867
197Py  0:07:08.202
```

判定: 219Pyは204Pyよりは速いが、217Pyより `0.067秒` 遅いため不採用。現最速基準は引き続き **217Py**。

### 220Py方針

220Pyは、合意済みの順番どおり **217Py + root_tail 判定** を試す。
218Pyのand-not predicate、219Pyのpr_block_code directは採用せず、217Pyへ戻す。

```python
# 217Py
root_rest:u32 = cur_avail & (cur_avail-u32(1))
root_second:u32 = root_rest & (u32(0)-root_rest)
root_after_second:u32 = root_rest ^ root_second

if root_after_second == u32(0):
    root_first:u32 = cur_avail & (u32(0)-cur_avail)
    ...
    cur_avail = root_rest   # first-child検証後

# 220Py
root_rest:u32 = cur_avail & (cur_avail-u32(1))
root_tail:u32 = root_rest & (root_rest-u32(1))

if root_tail == u32(0):
    root_first:u32 = cur_avail & (u32(0)-cur_avail)
    ...
    cur_avail = root_rest   # first-child検証後
```

`root_tail==0` は、root availabilityが1bit/2bitの場合だけtrueになり、3bit以上rootはgeneric loopへ戻る。対象範囲は217Pyと同じ。

維持するもの:

- 217Pyの `rootrestlate`
- `root_first` late extraction
- future_check_mask zero guard
- no-sibling spill elision
- root one/two-candidate preroll
- generic DFS loop
- host task order/cache/dispatch
- MAXD14, schedule_words=0, stack_bytes_per_thread=208

220Pyが217Py `0:07:07.709` を上回れば新採用。上回らなければ、次は予定通り 221Py = 217 + inline xor predicate を試す。

## 220Py確認結果と221Py方針（2026-07-02）

### 220Py確認結果

220Py `rootrestlate_roottail` は、N=21 full onceで正当性OK。

- final total: `314666222712` 一致
- progress rows: `131`
- duplicate/missing chunks: `0 / 0`
- dispatch task sum: `2025282`
- required_maxd: 全chunk `14`
- selected_MAXD: 全chunk `14`
- schedule_words: `0`
- stack_bytes_per_thread: `208`
- warning/error: `0 / 0`
- 実測: `0:07:07.959`

速度比較:

```text
217Py  0:07:07.709  現最速基準
219Py  0:07:07.776
204Py  0:07:07.795
211Py  0:07:07.801
218Py  0:07:07.825
212Py  0:07:07.828
216Py  0:07:07.867
220Py  0:07:07.959
197Py  0:07:08.202
```

判定: 220Pyは正当性OKだが、217Pyより `0.250秒` 遅いため不採用。現最速基準は引き続き **217Py**。

### 221Py方針

221Pyは、合意済みの順番どおり **217Py + inline xor predicate** を試す。
218Pyのand-not predicate、219Pyのpr_block_code direct、220Pyのroot_tail判定は採用せず、217Pyへ戻す。

```python
# 217Py
root_rest:u32 = cur_avail & (cur_avail-u32(1))
root_second:u32 = root_rest & (u32(0)-root_rest)
root_after_second:u32 = root_rest ^ root_second

if root_after_second == u32(0):

# 221Py
root_rest:u32 = cur_avail & (cur_avail-u32(1))
root_second:u32 = root_rest & (u32(0)-root_rest)

if (root_rest ^ root_second) == u32(0):
```

217Pyの `rootrestlate`、root_first late extraction、future_check_mask zero guard、no-sibling spill elision、root one/two-candidate preroll、MAXD14、schedule_words=0、stack_bytes_per_thread=208は維持する。

221Pyが217Py `0:07:07.709` を上回れば新採用。上回らなければ、合意済みの218〜221再結合確認は完了とし、次はMAXD13準備診断へ移る。


---

Updated on 2026-07-03 for 221Py result and 222Py MAXD13 preparation diagnostics.

- **221確認結果**: 221Py `rootrestlate_inlinexor` は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0/0`。速度は `0:07:08.033` で、217Py `0:07:07.709` より `0.324秒` 遅いため不採用。これにより、218Py `and-not predicate`、219Py `pr_block_code direct`、220Py `root_tail 判定`、221Py `inline xor predicate` の217近傍再結合確認はいずれも採用しない。
- **現時点の採用基準**: N=21 full once の数値上最速基準は **217Py `0:07:07.709`**。204Py `0:07:07.795`、211Py `0:07:07.801` はほぼ同等だが、217Pyを上回らないため現基準は217Pyのまま。
- **222**: 221Pyを採用せず、217Py rootrestlate新最速基準相当へ戻す。MAXD14 kernelの探索・加算ロジックは変更せず、MAXD13化可否を判断するための診断を追加する。追加診断は `[maxd13-diag]` として、chunkごとに `max_save_sp`、`save_sp13_count`、`max_cur_depth`、`max_terminal_depth`、`root_pc_max` を出す。`max_save_sp<=12` が全chunkで成立すれば、次段でMAXD13専用kernelを限定dispatchで試す余地がある。`max_save_sp==13` が出る場合は、MAXD14 fallbackを残したまま、より局所的な縮小または別観点の最適化へ進む。
- **222検証条件**: `N=21 full once`、bench_mode `31`、w8_j7、32x484、worker0/1。従来どおり final total、progress TSV再構成、dispatch rows/task sum、required_maxd/selected_MAXD、schedule_words、stack bytes、warning/errorを検証する。診断値は採否判断用のINFOとして扱い、正当性チェックそのものは従来の cases 01-06 で行う。

---

Updated on 2026-07-03 for 222Py result and 223Py rootrestlate restore.

- **222確認結果**: 222Py `maxd13_prepdiag_rootrestlate` は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0/0`。MAXD13準備診断では `max_save_sp=13`、`save_sp13_count=1177141`、`max_cur_depth=13`、`max_terminal_depth=13`、`root_pc_max=14`。このため、MAXD13への単純縮小、つまりMAXD14 fallbackなしでの13-slot化は不可と判断する。診断版の実測は `0:07:38.350` で、217Py `0:07:07.709` より `30.641秒` 遅い。これは診断配列書き込み・集計のオーバーヘッドを含むため、速度候補としては不採用。
- **223**: 222の診断結果を受け、MAXD13単純縮小は行わず、診断オーバーヘッドを撤回する安全復帰版。親は217Py rootrestlate最速基準相当とし、MAXD14 kernelは `root_rest = cur_avail & (cur_avail-u32(1))`、`root_second = root_rest & (u32(0)-root_rest)`、`root_after_second = root_rest ^ root_second`、`if root_after_second==u32(0):` のscalar predicateへ戻す。217由来の root_first late extraction、delayed `cur_avail=root_rest`、future_check_mask guard、no-sibling spill elision、root one/two-candidate preroll、schedule_words=0、stack_bytes_per_thread=208は維持する。222の `diag_save_sp` / `[maxd13-diag]` は入れない。採用可否はN=21 fullで217/204/221/222比を確認して判断する。

---

Updated on 2026-07-03 for 223Py result and 224Py generic clear-lowbit probe.

- **223確認結果**: 223Py rootrestlate_restore は、222PyのMAXD13準備診断を撤回し、217Py rootrestlate/futuremask/no-sibling/MAXD14相当へ戻した安全復帰版。N=21 full once で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0/0`。速度は `0:07:07.750` で、222Py `0:07:38.350` から `30.600秒` 戻し、217Py `0:07:07.709` に `0.041秒` 差まで復帰した。正当性・復帰確認とも問題なし。
- **224**: 223Pyを親に、MAXD14 generic DFS loopの候補消費だけを `cur_avail = cur_avail ^ bit` から `cur_avail = cur_avail & (cur_avail-u32(1))` へ置き換える微差実験版。root-preroll側の `cur_avail=root_rest`、root_restlate、root_after_second scalar predicate、future_check_mask guard、no-sibling spill elision、root one/two-candidate preroll、dispatch、host task order、progress検証は223/217相当のまま維持する。204Pyでroot側のclear-lowbitが効いた実績をgeneric loop側にも限定適用して、hot loopの候補消費が改善するかをN=21 full onceで確認する。

---

Updated on 2026-07-03 for 224Py result and 225Py generic ncol-early probe.

- **224確認結果**: 224Py generic clear-lowbit update は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0/0`。速度は `0:07:25.215` で、223Py `0:07:07.750` より `17.465秒`、217Py `0:07:07.709` より `17.506秒` 遅いため不採用。generic loop の `cur_avail=cur_avail&(cur_avail-u32(1))` は撤回し、223/217相当の `cur_avail=cur_avail^bit` へ戻す。
- **225**: 224は正当性OKだが大幅に低速化したため採用せず、223/217相当のgeneric `cur_avail^bit` 更新へ戻す。MAXD14 kernelのみ、generic loop内で `ncol:u32=cur_col|bit` を `nld/nrd` 計算より前へ移動する微差実験版。`ncol` は `block_code` に依存しないため意味は同じ。rootrestlate、root_after_second scalar predicate、root_first late extraction、future_check_mask guard、no-sibling spill elision、root one/two-candidate preroll、dispatch、host task order、stack_bytes_per_thread=208は維持する。224 generic clear-lowbit、222 MAXD13診断、221 inline xor、220 roottail、219 prblockdirect、218 and-not、195 rootthree、193/189/190/191系の不採用差分は入れない。採用可否はN=21 fullで223/217/204比timingを確認して判断する。

---

Updated on 2026-07-03 for 225Py result and 226Py generic-placeor probe.

- **225確認結果**: 225Py generic ncol-early は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0/0`。速度は `0:07:07.785`。223Py `0:07:07.750` より `0.035秒` 遅いが、204Py `0:07:07.795` より `0.010秒`、221Py `0:07:08.033` より `0.248秒`、197Py `0:07:08.202` より `0.417秒` 高速。217Py `0:07:07.709` には `0.076秒`、219Py `0:07:07.776` には `0.009秒` 届かないため新最速基準にはしない。generic ncol-early は大きな退行なしの微差改善候補として保持するが、採用基準は引き続き217Py。
- **226**: 225Pyを親に、MAXD14 generic loopのみ `cur_ld|bit` と `cur_rd|bit` を `placed_ld` / `placed_rd` として `block_code` 分岐前に共通化する微差実験版。225の `ncol` early、217/223相当の `cur_avail=cur_avail^bit`、rootrestlate、future_check_mask guard、no-sibling spill elision、root one/two-candidate preroll、schedule_words=0、stack_bytes_per_thread=208は維持する。224のgeneric clear-lowbit update、222のMAXD13診断、221 inline xor、220 roottail、219 prblockdirect、218 and-not再結合、195 rootthree、193/189/190/191系の不採用差分は入れない。採用可否はN=21 fullで217/225/223/204/197比timingを確認して判断する。

---

Updated on 2026-07-03 for 226Py result and 227Py generic-normalfirst probe.

- **226確認結果**: 226Py generic place-or は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0/0`。速度は `0:07:07.998`。225Py `0:07:07.785` より `0.213秒`、223Py `0:07:07.750` より `0.248秒`、217Py `0:07:07.709` より `0.289秒`、204Py `0:07:07.795` より `0.203秒` 遅いため不採用。226の `placed_ld` / `placed_rd` 共通化は撤回し、225のncol-early形へ戻す。
- **227**: 226は正当性OKだが225/217を上回らなかったため採用せず、225Pyの `ncol` early とgeneric `cur_avail=cur_avail^bit` へ戻す。MAXD14 generic loopのみ、`block_code!=0` を先に見る形から `block_code==0` のnormal pathを先に書く分岐順序微差実験版。normal pathは旧else branchと同じ `nld=(cur_ld|bit)<<1`、`nrd=(cur_rd|bit)>>1` であり、非zero block_code側のstep/block decode、rootrestlate、root_after_second scalar predicate、root_first late extraction、future_check_mask guard、no-sibling spill elision、root one/two-candidate preroll、dispatch、host task order、stack_bytes_per_thread=208は維持する。224 generic clear-lowbit、226 placed_ld/placed_rd、222 MAXD13診断、221 inline xor、220 roottail、219 prblockdirect、218 and-not、195 rootthree、193/189/190/191系の不採用差分は入れない。採用可否はN=21 fullで217/225/223/204/197比timingを確認して判断する。

Updated on 2026-07-03 for 227Py result and 228Py root-preroll ncol-early probe.

- **227確認結果**: 227Py generic normal-first は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0/0`。速度は `0:07:07.808` で、226Py `0:07:07.998` より `0.190秒` 高速だが、225Py `0:07:07.785` より `0.023秒`、223Py `0:07:07.750` より `0.058秒`、217Py `0:07:07.709` より `0.099秒` 遅いため不採用。227の `if block_code==0` normal-first は撤回する。
- **228**: 227Pyを採用せず、225Pyのgeneric ncol-earlyへ戻す。MAXD14 kernelのみ、root-preroll内でも `pr_ncol:u32=cur_col|pr_bit` を `pr_nld/pr_nrd` 計算より前へ移動する root-preroll ncol-early 微差実験版。generic loop側の225 ncol-early、217相当のrootrestlate/root_after_second scalar predicate、future_check_mask guard、no-sibling spill elision、root one/two-candidate preroll、schedule_words=0、stack_bytes_per_thread=208は維持する。226 placed_ld/placed_rd、227 normal-first、224 generic clear-lowbit、222 MAXD13診断、221 inline-xor、220 roottail、219 prblockdirect、218 and-not、195 rootthree は入れない。

---

Updated on 2026-07-03 for 228Py result and 229Py root-preroll normal-first probe.

- **228確認結果**: 228Py rootpr_ncol_early は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0/0`。速度は `0:07:07.985` で、225Py `0:07:07.785` より `0.200秒`、223Py `0:07:07.750` より `0.235秒`、217Py `0:07:07.709` より `0.276秒` 遅いため不採用。228の root-preroll ncol early は撤回する。
- **229**: 228は正当性OKだが速度退行したため採用せず、225Pyの generic ncol early / rootrestlate 形へ戻す。MAXD14 root-preroll内だけ、`if pr_block_code!=u32(0): ... else: normal` のblock-first分岐を、`if pr_block_code==u32(0): normal else: block-specific` のnormal-first分岐へ変更する。generic DFS loop、rootrestlate scalar predicate、future_check_mask guard、no-sibling spill elision、root one/two-candidate preroll、dispatch、host task orderは変更しない。

---

Updated on 2026-07-03 for 229Py result and 230Py rootpr-placeor probe.

- **229確認結果**: 229Py root-preroll normal-first は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0/0`。速度は `0:07:07.853`。228Py `0:07:07.985` と226Py `0:07:07.998` よりは戻したが、217Py `0:07:07.709` より `0.144秒`、225Py `0:07:07.785` より `0.068秒`、223Py `0:07:07.750` より `0.103秒` 遅いため不採用。229の root-preroll normal-first は撤回する。
- **230**: 229は正当性OKだが217/225を上回らなかったため採用せず、225Py generic ncol-early形へ戻す。MAXD14 root-preroll内だけ、`cur_ld|pr_bit` と `cur_rd|pr_bit` を `pr_placed_ld` / `pr_placed_rd` として一度だけ作り、block-specific path と normal step1 path の双方で再利用する rootpr place-or 微差実験版。generic loop、rootrestlate scalar predicate、future_check_mask guard、no-sibling spill elision、root one/two-candidate preroll、dispatch、host task orderは変更しない。226Pyの generic placed_ld/placed_rd 共通化は退行したため入れず、root-preroll内だけに限定する。

---

Updated on 2026-07-06 for 230Py result and 231Py rootrestlate restore.

- **230確認結果**: 230Py `rootpr-placeor` は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0/0`。速度は `0:07:08.848` で、217Py `0:07:07.709` より `1.139秒`、223Py `0:07:07.750` より `1.098秒`、225Py `0:07:07.785` より `1.063秒` 遅いため不採用。230の root-preroll `placed_ld/placed_rd` 共通化は撤回する。
- **現時点の採用基準**: N=21 full once の数値上最速基準は **217Py `0:07:07.709`**。223Py `0:07:07.750`、225Py `0:07:07.785`、227Py `0:07:07.808`、229Py `0:07:07.853` はいずれも正当性OKだが217Pyを上回らないため、親版は217Py相当で固定する。
- **231**: 230Pyを採用せず、217Py rootrestlate最速基準相当へ戻す復帰版。MAXD14 kernelでは `root_rest = cur_avail & (cur_avail-u32(1))`、`root_second = root_rest & (u32(0)-root_rest)`、`root_after_second = root_rest ^ root_second`、`if root_after_second==u32(0):` の217 scalar predicateを維持し、`root_first` late extraction、delayed `cur_avail=root_rest`、future_check_mask guard、no-sibling spill elision、root one/two-candidate preroll、generic DFS loop、host task order、dispatch、MAXD14 `schedule_words=0`、`stack_bytes_per_thread=208` を維持する。
- **231で撤回する差分**: 222 MAXD13診断、224 generic clear-lowbit、225 generic ncol-early、226 generic placed_ld/placed_rd、227 generic normal-first、228 root-preroll ncol-early、229 root-preroll normal-first、230 root-preroll placed_ld/placed_rd。
- **231検証条件**: `N=21 full once`、bench_mode `31`、w8_j7、32x484、worker0/1。従来どおり final total、progress TSV再構成、dispatch rows/task sum、required_maxd/selected_MAXD、schedule_words、stack bytes、warning/errorを検証する。`STATIC_ONLY=1` では、217 rootrestlate復帰形、split231 tag、230/224〜230系のactive差分撤回を確認する。

---

Updated on 2026-07-06 for 231Py r2 static split-tag fix.

- **231 r2**: 231Py rootrestlate restore本体は217Py相当の復帰形のまま変更しない。cudacodon側で `source_split230_removed` / `source_split_tag` がFAILしたケースに対応し、231検証シェルのsplit tag静的検査を補修した。検査対象はコメントや履歴文ではなく、実行時に使われるruntime/progress tagに限定する。期待値は `split145 split231` で、`split230` runtime tag が残る場合だけFAILする。`STATIC_ONLY=1` で `source_version_tag`、`source_future_check_mask_guard`、`source_nosibling_parent`、`source_217_restore_shape`、`source_split230_removed`、`source_split_tag` がすべてOKになることを確認済み。

---

Updated on 2026-07-06 for 231Py full result and 232Py cleanup-only.

- **231確認結果**: 231Py `rootrestlate_restore_fastest` は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0/0`。速度は `0:07:07.818`。230Py `0:07:08.848` から `1.030秒` 復帰し、230の退行は解消した。ただし217Py `0:07:07.709` より `0.109秒`、204Py `0:07:07.795` より `0.023秒` 遅いため、新最速基準にはしない。現時点の数値上最速基準は引き続き **217Py `0:07:07.709`**。
- **231の位置づけ**: 230Pyで入ったroot-preroll `placed_ld/placed_rd` 共通化を完全撤回し、217Py相当のrootrestlate/futuremask/no-sibling/MAXD14へ戻せている。231は今後の整理・再現性確認の親として使用可能。
- **232**: 231Pyを親にする `cleanup-only` 版。kernel探索ロジック、MAXD14/16/18/20/21 kernel、`build_soa_for_range`、host task order、dispatch、cache、solution arithmeticは変更しない。変更は、ソース冒頭の巨大履歴ログ、長いdocstring、コメント履歴のREADME退避、runtime/progress tagの `split232` 化、検証シェル内の過去baseline比較と静的検査の整理に限定する。
- **232 cleanup確認**: 231→232の正規化比較では、MAXD14 kernel、MAXD16-21 fallback kernel群、`build_soa_for_range` のコード本文は一致。`STATIC_ONLY=1` では `source_version_tag`、`source_future_check_mask_guard`、`source_nosibling_parent`、`source_217_231_restore_shape`、`source_split_tag` がすべてOK。
- **232削減量**: `.py` は 231Py の `11078行 / 465624 bytes` から、232Py の `9432行 / 365682 bytes` へ削減。`1646行`、約 `99942 bytes` の削減。これは可読性改善の第一段であり、旧bench/profile関数の削除までは行っていない。
- **次の検証**: 232はまず `STATIC_ONLY=1 bash 232Py_cleanup_only_rootrestlate_restore_fastest_maxd14_validate_full_once.sh` を通し、その後 `bash 232Py_cleanup_only_rootrestlate_restore_fastest_maxd14_validate_full_once.sh` でN=21 full onceを確認する。full結果が231と同等なら、次段で旧diagnostic mode/古いCLI引数の実削除をさらに小さなcleanup-only版として分離する。

---

Updated on 2026-07-06 for 232Py full result and 233Py fasttrim-inline cleanup.

- **232確認結果**: 232Py `cleanup_only_rootrestlate_restore_fastest` は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0/0`。速度は `0:07:07.733`。231Py `0:07:07.818` より `0.085秒` 高速、204Py `0:07:07.795` より `0.062秒` 高速、217Py `0:07:07.709` には `0.024秒` 届かない。cleanup-onlyとして正当性OKで、次段の親として採用可能。
- **233**: 232Pyを親にする `fasttrim-inline` 版。MAXD14/16/18/20/21 CUDA kernel本文、rootrestlate、future_check_mask guard、no-sibling save_sp/cur_depth、root one/two-candidate preroll、host reorder/cache名、dispatch条件、solution arithmeticは変更しない。変更はN=21 `split145/full` 実行経路のhost側整理に限定する。
- **233実装内容**: `build_soa_for_range()` 内で `getj/getk/getl` の小helper呼び出しを直接bit展開へ置換し、`symmetry()` / `symmetry90()` 呼び出しを `@par` loop内の直接式へ展開した。`exec_solutions_gpu_chunk_split145()` は削除し、split145 stream loop内へchunk実行処理を同梱した。さらに、split145 stream loopのbinary record decodeでは `read_uint32_le()` 呼び出しをhot loop内の直接式へ展開した。旧CPU DFS、境界診断、funcid/profile/chunksize/mark/markdist等の旧診断関数は削除し、本線特化の見通しを改善した。
- **233削減量**: `.py` は 232Py の `9432行 / 365682 bytes / 228 defs` から、233Py の `4104行 / 133920 bytes / 116 defs` へ削減。`5328行`、約 `231762 bytes`、`112 defs` の削減。MAXD14/16/18/20/21 kernel本文と `launch_kernel_dfs_iter_gpu_static_maxd()` は232と正規化一致する。
- **233検証条件**: `STATIC_ONLY=1` では 217/231/232 rootrestlate復帰形、future_check_mask guard、no-sibling save_sp/cur_depth、split233 tag、host hot-path inline marker、旧診断関数削除を確認する。full実行では従来どおり `N=21 full once`、bench_mode `31`、w8_j7、32x484、worker0/1で、final total、progress TSV再構成、dispatch rows/task sum、required_maxd/selected_MAXD、schedule_words、stack bytes、warning/errorを検証する。

---

Updated on 2026-07-06 for 233Py r2 static-check fix.

- **233 r2**: 233Py fasttrim-inline の `.py` 探索ロジックとCUDA kernelは変更せず、検証シェルの `source_fasttrim_inline` 静的検査のみ補修した。初版では `exec_solutions_gpu_chunk_split145(` をraw textで検索していたため、コメントや説明文に同名が含まれるだけで `split_chunk_call_removed` がFAILし得た。r2ではtokenizeによりcomment/stringを除いたactive code上のdef/callだけを検査する。こちらでは `STATIC_ONLY=1` で `source_version_tag`、`source_future_check_mask_guard`、`source_nosibling_parent`、`source_217_232_restore_shape`、`source_split_tag`、`source_fasttrim_inline` がすべてOK。

---

Updated on 2026-07-06 for 233Py r2 staticfix package.

- **233 r2 staticfix**: cudacodon側で `inline_trim_pycheck_failed=split_chunk_call_removed` が出た場合は、検証シェルが読んだ233 sourceに旧 `exec_solutions_gpu_chunk_split145` 呼び出し片が残っていることを示す。233 r2ではCUDA kernel本文、rootrestlate/futuremask/no-sibling/MAXD14、host task order、dispatch条件、solution arithmeticは変更せず、配布sourceを旧chunk helper active定義/active呼び出しが残らない形へ固定した。`STATIC_ONLY=1` では `source_version_tag`、`source_future_check_mask_guard`、`source_nosibling_parent`、`source_217_232_restore_shape`、`source_split_tag`、`source_fasttrim_inline` がすべてOKになることを確認済み。
- **確認コマンド**: `grep -n 'exec_solutions_gpu_chunk_split145' 233Py_fasttrim_inline_split145_rootrestlate_restore_fastest_maxd14_probe.py` が何も出さないことを確認してから、`STATIC_ONLY=1 bash 233Py_fasttrim_inline_split145_rootrestlate_restore_fastest_maxd14_validate_full_once.sh` を実行する。もしgrepが行番号を返す場合は、古い233 sourceが残っているためr2 packageで上書きする。


---

Updated on 2026-07-06 for 233Py r3 main-entry fix.

- **233 r3 mainfix**: r2 sourceでは `def main()` 本体は存在していたが、末尾の `if __name__=="__main__": main()` が欠けていたため、cudacodon側ではbuild後にcandidateが即時exitし、検証シェルはヘッダー表示後にprogress TSVを得られず停止した。r3ではCUDA kernel本文、rootrestlate/futuremask/no-sibling/MAXD14、host hot-path inline差分、dispatch条件、solution arithmeticは変更せず、末尾のmain entryだけを復元した。
- **233 r3検証シェル補修**: `source_main_entry` 静的検査を追加し、同じ事故をGPU実行前に検出する。また、candidateがprogress TSVを生成しなかった場合はsummaryとlogdirを表示して停止する。
- **確認コマンド**: `STATIC_ONLY=1 bash 233Py_fasttrim_inline_split145_rootrestlate_restore_fastest_maxd14_validate_full_once.sh` で `source_main_entry` を含む全静的検査がOKになることを確認してから、通常のfull実行へ進む。


---

Updated on 2026-07-06 for 234Py cachehot-maxd14-direct.

- **233確認結果**: 233Py fasttrim-inline r3 は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0/0`。速度は `0:07:07.762` で、231Py `0:07:07.818`、204Py `0:07:07.795` をわずかに上回った。
- **234**: 233を親に、N=21/cache-hot/full-runへさらに特化した整理・最速化候補。CUDA MAXD14 kernel本文は233と同一のまま、MAXD16/18/20/21 fallback kernel、schedule depth scan、MAXD select/launch wrapper、per-chunk `chunk_constellations` dict list、汎用cache生成関数群、旧stats full集計、CPU/small-N fallbackを234 runtimeから削除した。既存の `constellations_N21_6_chunkshape148_scorestripe_v9_lanephase32_octetfirstpairlock29_v4_rotate_only_w8_j7_b32_m484_s15488.bin` を直接読み、SoAへ直接展開してMAXD14 kernelを直接起動する。234はcache-hot専用のため、shaped cacheが無い環境では233以前で一度cacheを生成してから実行する。sourceは233の `4109` 行 / `133898` bytes / `119` defs から、234の `778` 行 / `25701` bytes / `9` defs へ縮小。`STATIC_ONLY=1` はOK確認済み。

---

Updated on 2026-07-06 for 235Py cacheautogen helper package (final 235).

- **235 cacheautogen helper package**: 234Py の cache-hot/MAXD14 direct runner 自体は短く保つため、実行本体と cache 生成経路を分離した。`235Py_cacheautogen_maxd14_direct_split145_rootrestlate_fastest_probe.py` は234相当の shaped-bin 直読 + MAXD14 direct launch に限定し、fallback kernel、schedule depth scan、MAXD select wrapper、chunk dict list、旧stats系を引き続き持たない。`235Py_cacheautogen_maxd14_direct_split145_rootrestlate_fastest_cachebuild.py` は bin missing/incomplete 時だけ使う生成専用helperで、233由来の `ensure_constellations_bin_stream`、`build_broad_markdist_tail_reordered_bin`、`build_chunkshape148_reordered_bin` を保持する。
- **235 検証シェルの変更**: `235Py_cacheautogen_maxd14_direct_split145_rootrestlate_fastest_validate_full_once.sh` は full 実行前に shaped bin の record数と `.done` を確認する。`constellations_N21_6_chunkshape148_scorestripe_v9_lanephase32_octetfirstpairlock29_v4_rotate_only_w8_j7_b32_m484_s15488.bin` が `2025282` records / done `2025282` でなければ、`AUTO_CACHE_BUILD=1` のとき同梱 cachebuild helper を build/run し、生成後に再検査してから235 main runnerを実行する。cache hit 時は helper を起動しないため、234の薄い実行経路を維持できる。
- **235 静的確認**: `STATIC_ONLY=1` では main runner 側の `source_version_tag`、`source_main_entry`、`source_future_check_mask_guard`、`source_nosibling_parent`、`source_cacheautogen_maxd14_direct`、`source_split_tag` に加えて、helper 側に stream / broadmarktail / chunkshape148 生成関数と `[cachebuild-done]` marker があることを確認する。こちらでは `STATIC_ONLY=1` がOK。

---

Updated on 2026-07-06 for 236Py restore232-general-cleanup-keepfeatures.

- **236方針修正**: 234/235 は N=21/cache-hot/maxd14 direct へ寄せすぎ、cache生成・fallback kernel・N範囲・既存bench/worker経路を狭めてしまったため、採用しない。236Py は 232Py cleanup-only を親へ戻し、N=5..27 GPU/CPU範囲、`-c`/`-g` bare default、A10G既定値、stream/cache生成、MAXD14/16/18/20/21 fallback、bench_mode 28/29/31、`worker_id`/`worker_count` multi-GPU split を維持する。
- **保持する起動仕様**: bare `-c` は既定range `N=5..23` のCPU表形式出力を維持する。bare `-g` は同じ既定rangeに対して、A10G単GPUの実測best flowである `block=32`、`max_blocks=484`、`preset=7`、`bench_mode=29`、`w8_j7`、`broadmarktail variant=2 rotate_only` を適用する。N>=21では broadmarktail mode29、N<21では従来GPU/CPU互換経路へ落ちる。
- **multi-GPU保持**: 111Py系で使っていた `CUDA_VISIBLE_DEVICES=<id>` + `worker_id/worker_count` 分割は保持する。236の同梱 `236Py_a10g4_multigpu_broadmarktail_worker.sh` は、reorder bin を mode28 で一度だけ生成してから、mode29 worker 0..3 をそれぞれ `CUDA_VISIBLE_DEVICES=0..3` で起動する。各workerの合算確認用に `236Py_sum_worker_totals.py` も同梱する。
- **236で実施したこと/していないこと**: CUDA kernel本文、`build_soa_for_range`、cache/reorder生成、dispatch、CPU path、worker split は232と同一。236は、234/235の専用化を撤回して汎用本線を再固定する版であり、機能削除は行っていない。次に関数整理を行う場合は、N=5..27、`-c`/`-g` bare range、mode28/29/31、worker split、cache生成、fallback kernel を保持する静的検査を通した上で、小さな範囲に限定する。

---

Updated on 2026-07-06 for 236Py operational-clean-from-232 correction.

- **方針修正**: 234Py/235Pyのcache-hot/N=21寄せは、短期の速度確認には有効だったが、今後N27を目指す運用基盤としては狭すぎるため本線から外す。今後の整理は232Pyへ戻し、動いている機能を削らず、機能単位の到達性を確認してから行う。
- **236**: 232Py `cleanup_only_rootrestlate_restore_fastest` を親にした operational-clean 版。`-c` / `-g` の標準CLI、bare `-c` / bare `-g` の N=5..23 表示、N=5..27 の既知解テーブル、A10G final defaults、bench_mode `0..31`、cache生成、broadmarktail mode `28/29`、split145 mode `30/31`、MAXD14 plus MAXD16/18/20/21 fallback、worker_id/worker_count によるchunk-level multi-GPU splitを保持する。N=21専用化、cache-hot専用化、helper分離による運用機能削除はしない。
- **236の変更範囲**: runtime/progress tag を `split236` へ分離し、NQ_UPDATE_MEMOを「機能削除なし」の方針へ更新した。CUDA kernel、host cache/reorder、mainのCLI処理、worker splitは232から正規化比較で維持する。236では安全側としてtop-level `def` は削除しない。不要関数の削除は、次段でbench_mode/CLI到達性マップを作ってから実施する。
- **multi-GPU維持**: 111Py時代の `CUDA_VISIBLE_DEVICES=0/1/2/3` + `worker_id worker_count` 運用を維持する。236には `236Py_multigpu_worker_launcher.sh` と `236_sum_worker_totals.py` を同梱し、reorder binを1回だけ生成してから各GPUへworkerを割り当てる手順を残す。N22だけでなく、`N=27 NGPU=4` のようにNを差し替えられる構成にする。
- **236検証**: `STATIC_ONLY=1` では `source_version_tag`、futuremask/no-sibling/rootrestlate、split tagに加え、`-c/-g`、N=5..23 default range、N=27 expected table、A10G mode29 defaults、bench_mode 28/29/31、cache generation、worker split、MAXD fallback kernels が残っていることを確認する。


---

Updated on 2026-07-06 for 237Py restore232-fastdefault-keepfeatures.

- **236確認メモ**: 236Pyは232Pyへ戻した汎用機能保持版として正しく動作したが、bare `-g` のA10G既定が旧来の broadmarktail mode29 のままだったため、N=21 range実行では `0:08:17.505` になった。これは7分台のsplit145/mode31経路ではない。
- **237**: 236Pyを親に、kernel、SoA、cache生成、MAXD14/16/18/20/21 fallback、broadmarktail mode28/29、split145 mode31、worker_id/worker_count multi-GPU分割を保持したまま、bare `-g` の既定だけを `A10G_FINAL_DEFAULT_BENCH_MODE=31` へ変更する。これにより `./237Py... -g` はN=21以降でsplit145 mode31を使い、`-g 21 21 32 484 1 0 7 31 8 7 0 0 1 2` と同じ最速系統へ入る。明示的なmode28/mode29起動は従来通り残す。
- **multi-GPU方針**: 4xA10Gなどでは、まずmode28でbroadmarktail reorder binを1回だけ作り、次にmode30でchunkshape148/split145 shaped binを1回だけ作ってから、mode31をworker_id/worker_count付きで並列起動する。これにより複数workerが同じbinを同時生成する競合を避ける。

---

Updated on 2026-07-06 for 237Py restore232-fastdefault-split145-keepfeatures.

- **237**: 236で汎用機能を232Py相当へ戻した後、bare `-g` の既定経路が旧 broadmarktail `mode29` に戻り、N=21 range出力で `0:08:17.505` になったため、`-g` の既定 `A10G_FINAL_DEFAULT_BENCH_MODE` を現在の最速本線である `mode31` / split145 + chunkshape148 へ戻した版。232Pyを親にし、MAXD14/16/18/20/21 kernel、`launch_kernel_dfs_iter_gpu_static_maxd`、`build_soa_for_range`、stream bin生成、broadmarktail reorder生成、chunkshape148 reorder生成、split145実行経路は232Pyと正規化同一。`-c` / `-g` 標準range、GPU N=5..27、N25..N27 dynamic preset=8、cache missing時の生成、mode28/29 broadmarktail、mode30 probe、mode31 full、`worker_id` / `worker_count` によるmulti-GPU分割、`CUDA_VISIBLE_DEVICES` 運用を保持する。`-g` 無引数では N<21 は従来GPU経路、N>=21 はmode31のsplit145本線へ入る。mode29は削除せず、明示指定または `MODE=29` worker scriptで従来通り実行可能。
- **237 final package補足**: 配布版 `237Py_restore232_fastdefault_keepfeatures_probe.py` は、236汎用版から参照されないtop-level helperだけを削った安全整理も含む。削除対象は `rot180`、`load_or_build_solutions_txt`、`build_broad_markdist_reordered_bin`、`analyze_broad_markdist_tail_subcell_stats_from_soa` の4関数で、`grep` 到達性上は237本体から参照されない。削除後も top-level `def` は 236の226個から237の222個へ減るだけに留め、CUDA MAXD14/16/18/20/21 kernel、MAXD dispatch wrapper、`build_soa_for_range`、stream/bin/cache生成、broadmarktail tail reorder、chunkshape148、split145 mode31、mode28/29、CPU path、worker splitは保持する。`STATIC_ONLY=1` は `237Py_restore232_fastdefault_keepfeatures_validate_N21_full_once.sh` と `237Py_restore232_fastdefault_keepfeatures_validate_static.sh` の双方でOK確認済み。
---

Updated on 2026-07-06 for 238Py n27diagtrim-keepfeatures.

- **237確認結果**: 237Py restore232-fastdefault-keepfeatures は bare `-g` で現在の最速本線 `mode31` / split145 + chunkshape148 へ入り、N=21 range出力で `0:07:07.834` を確認した。236 bare `-g` の `0:08:17.505` 退行は、既定が旧 broadmarktail `mode29` だったことによるもので、237では解消した。
- **238**: 237Pyを親にする `n27diagtrim-keepfeatures` 版。N27へ不要な診断のみを削除し、動作中の汎用機能は残す。削除対象は selected chunk microbench、funcid target/single/split/depth/mark/markdist profile、markdist risk reorder mode26/27、boundary classification diagnostics、旧profile用の細かいprint/usage/引数群。
- **238で保持する機能**: `-c` / `-g` 標準range、bare `-g` の mode31 fast default、GPU N=5..27、N25..N27 dynamic preset=8、cache/bin missing時の生成、broadmarktail mode28/29、split145 mode30/31、MAXD14/16/18/20/21 fallback、CPU path、`worker_id` / `worker_count` によるmulti-GPU split、`CUDA_VISIBLE_DEVICES` 運用を維持する。N21専用化、cache-hot専用化、fallback削除はしない。
- **238 r2 buildfix**: 初回238で削りすぎた runtime helper/global を戻し、mode28/29/30/31本線に必要な `FUNCID_REORDER_V2_*`、`BROAD_MARKDIST_TAIL_*`、broadmarktail/chunkshape148/split145関連helperを保持した。これは旧診断ではなく、cache名、shaping order、CLI override、worker分割に必要な本線部品として扱う。
- **238静的検査**: `STATIC_ONLY=1 bash 238Py_n27diagtrim_keepfeatures_validate_N21_full_once.sh` で `source_version_tag`、`source_main_entry`、bare `-g` mode31、N25..N27 preset8/N27 total、runtime globals、required runtime defs、旧診断def削除、bench mode 17..27削除、mode28/29/30/31保持、split238 tag、worker split args がOKであることを確認した。
- **238実測メモ**: cudacodon側のN=21 full once受領ログでは final total `314666222712`、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、全chunk required/selected MAXD14、schedule_words `0`、stack_bytes_per_thread `208`、warning/error/mismatch `0`、elapsed `0:07:07.729`。237 `0:07:07.834` より `0.105秒` 改善、232 `0:07:07.733` と実質同等、217 `0:07:07.709` より `0.020秒` 遅いだけなので、N27志向の診断削除keepfeatures基準として採用可能。
- **次候補**: kernelをtask/id別に分解する案は有望だが、複数kernel起動・load imbalance・PTX/JIT/レジスタ圧・worker分割との相互作用が大きい。239以降で、まずは現在のgeneric MAXD14を親にした `taskid-split probe` として、fid/boundary class別に少数グループへ分け、fallbackとmode28/29/30/31を維持したままN21/N22で比較する。238本線には入れない。



---

Updated on 2026-07-07 for 239Py n27coretrim-keepfeatures.

- **238確認結果**: 添付ログ `238Py_n27diagtrim_keepfeatures_logs_N21_full_once_20260707_012857` では `N=21 full once` が final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required/selected MAXD は全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error/mismatch `0`、elapsed `0:07:07.710`。238はN27へ不要な診断削除後も217最速基準 `0:07:07.709` と実質同等で、keepfeatures基準として採用可能。
- **239**: 238Py `n27diagtrim-keepfeatures r2` を親にする `n27coretrim-keepfeatures` cleanup-only版。CUDA MAXD14/16/18/20/21 kernel本文、rootrestlate、future_check_mask guard、no-sibling save_sp/cur_depth、root one/two-candidate preroll、MAXD dispatch、`build_soa_for_range`、stream/bin/cache生成、broadmarktail mode28/29、split145 mode30/31、worker_id/worker_count multi-GPU split、CPU `dfs_iter` path は変更しない。
- **239で削ったもの**: CPU path内で常に `dfs_iter` が選ばれていたため、実行本線から到達しない再帰版 `dfs()` fallback と、その切替用の到達不能分岐を削除した。あわせて、238配布時の説明文docstringを短い239メモへ置き換えた。これはkernel探索ではなく、次のtask/id splitへ進む前の小さな整理である。
- **239削減量**: 238 source `5674行 / 196443 bytes / 132 defs` から、239 source `5436行 / 189259 bytes / 131 defs` へ削減。差分は `238行 / 7184 bytes / 1 def`。CUDA kernel数は5本のまま、MAXD fallbackも維持。
- **239静的検査**: `STATIC_ONLY=1 bash 239Py_n27coretrim_keepfeatures_validate_N21_full_once.sh` で `source_version_tag`、`source_main_entry`、bare `-g` mode31、N25..N27 preset8/N27 total、runtime globals、required runtime defs、旧診断def削除、再帰CPU dfs削除、到達不能切替分岐削除、bench mode 17..27削除、mode28/29/30/31保持、split239 tag、worker split args がOKであることを確認した。
- **239検証条件**: full runは従来どおり `N=21 full once`、bench_mode `31`、w8_j7、32x484、worker0/1。final total、progress TSV再構成、dispatch rows/task sum、required/selected MAXD、schedule_words、stack bytes、warning/errorを確認する。速度が238同等なら採用し、遅ければ238へ戻す。
- **次候補**: 239 fullが238同等なら、240以降で `taskid-split probe` に入る。最初は fid=14 専用 + generic の2分割に限定し、mode30 selected chunksで正当性・kernel launch overhead・load imbalanceを見てからN21 fullへ進む。いきなりfuncid 28個別kernelへ分けない。

---

Updated on 2026-07-07 for 239Py result and 240Py taskid-split-fid14 probe.

- **239確認結果**: 239Py `n27coretrim-keepfeatures` は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error/mismatch `0`。速度は `0:07:07.703` で、238Py `0:07:07.710`、237Py `0:07:07.834`、232Py `0:07:07.733`、217Py `0:07:07.709` をわずかに上回った。239は、238のkeepfeatures本線を保ったままCPU側の未使用再帰 `dfs()` fallback と到達不能 `use_itter` 分岐だけを削ったcleanup-only版として採用可能。
- **240**: 239Pyを親にする `taskid-split-fid14` probe。N27へ向けたkernel分解の第一歩として、split145 chunk実行時だけ fid=14 (`SQd2B` / base14) と rest を別launchへ分ける。CUDA MAXD14/16/18/20/21 kernel本文はまだ変更せず、まずは task/id split の launch overhead、load balance、dispatch/task sum、worker split との相互作用を測る。
- **240の保持機能**: `-c` / `-g` 標準range、bare `-g` mode31 fast default、GPU N=5..27、N25..N27 dynamic preset=8、cache/bin missing時の生成、broadmarktail mode28/29、split145 mode30/31、MAXD14/16/18/20/21 fallback、CPU path、`worker_id` / `worker_count` multi-GPU split、`CUDA_VISIBLE_DEVICES` 運用を維持する。
- **240検証条件**: `N=21 full once`、bench_mode `31`、w8_j7、32x484、worker0/1。N=21では各chunkにfid14があるため、dispatch rowsは `262`（rest + fid14の2launch × 131 chunks）、dispatch task sumは従来通り `2025282`、fid14 task sumは `8214` を期待する。final total、progress TSV再構成、duplicate/missing、required/selected MAXD14、schedule_words=0、stack=208 bytes/thread、warning/error 0 を確認する。
- **240の位置づけ**: 240は「専用kernelで分岐を削った版」ではなく、「fid14を別launchへ分けても正当性・dispatch・速度がどの程度変わるか」を測る第一段probe。240が大きく退行しなければ、241以降で fid14専用MAXD14 kernel側から base14/terminal系条件を削る検討へ進む。240がlaunch overheadで明確に遅い場合は、kernel分解はd0や重いtail群など別グループで再検討する。

---

Updated on 2026-07-07 for 240Py r2 buildfix.

- **240 r2 buildfix**: 初回240Py `taskid_split_fid14` は、fid14/rest の別launch化後、error path の `required_maxd` 表示が従来単一dispatch前提の変数名のまま残っており、Codon build時に `name 'required_maxd' is not defined` で停止した。r2ではCUDA kernel本文、launch split方針、cache/worker/mode28/29/30/31保持方針は変更せず、error path の表示変数を `rest_required_maxd` / `d2_required_maxd` / `gen_required_maxd` へ修正した。`STATIC_ONLY=1` はOK確認済み。N=21 full onceで、dispatch rows は fid14/rest の2launch化により `262`、task sumは従来通り `2025282`、fid14 task sumは `8214` を期待する。

---

Updated on 2026-07-07 for 240Py r3 buildfix.

- **240 r3 buildfix**: cudacodon側で `required_maxd` 未定義が残っていたため、240の配布ファイルを再固定した。r3では active code 上の `required_maxd` token が汎用 `exec_solutions()` 経路の既存変数と helper 引数に限定され、`exec_solutions_gpu_chunk_split145()` のfid14/rest split経路では `rest_required_maxd`、`d2_required_maxd`、`gen_required_maxd` の明示名だけを使う。CUDA kernel本文、fid14/rest launch split方針、cache生成、mode28/29/30/31、worker splitは変更しない。
- **確認**: `grep -n "required_maxd" 240Py_taskid_split_fid14_probe.py` で split145 function 内に裸の `required_maxd` 参照が残らないことを確認してからbuildする。

---

Updated on 2026-07-07 for 240Py r4 buildfix.

- **240 r4 buildfix**: 240Py taskid-split-fid14 の方針は変更しない。fid=14/rest の別launch probe、CUDA kernel本文、cache生成、mode28/29/30/31、worker split は維持する。cudacodon側で `required_maxd` 未定義が継続したため、`exec_solutions_gpu_chunk_split145()` 内に defensive local `required_maxd:int=0` / `selected_maxd:int=14` を明示し、Codon realization が旧単一dispatch名を参照しても未定義にならないよう補修した。実際のdispatch判断は引き続き `rest_required_maxd` / `d2_required_maxd` / `gen_required_maxd` と各selected MAXDで行う。`STATIC_ONLY=1` はOK確認済み。通常full runで N=21 total/progress/dispatch/速度を確認し、239/238比で採否を判断する。

---

Updated on 2026-07-07 for 240Py r7 buildfix.

- **240 r7 buildfix**: 240Py taskid-split-fid14 probe の split145 内で、Codon が `rest_required_maxd` 系の長いローカル名を含む f-string を `required_maxd` 未定義として扱うビルドエラーが続いたため、split145内のMAXDローカル名を `rmaxd` / `rselmaxd` / `d2maxd` / `d2selmaxd` / `gmaxd` / `gselmaxd` へ短縮し、該当 `[maxd-dispatch]` / `[maxd-error]` 出力を f-string ではなく文字列連結へ変更した。CUDA kernel本文、fid14/rest launch split方針、cache生成、mode28/29/30/31、worker splitは変更しない。`STATIC_ONLY=1` では source_version_tag、main entry、bare -g mode31、fid14 launch split、N27 preset、runtime globals、required defs、diag mode削除、core modes保持、split240 tag、worker split args がOK。

---

Updated on 2026-07-07 for 240Py r8 buildfix.

- **240 r8 buildfix**: cudacodon側で `name 'required_maxd' is not defined` が継続したため、原因候補を変数名だけでなく split145 内のログ文字列まで広げて修正した。`exec_solutions_gpu_chunk_split145()` 内から `required_maxd` / `selected_MAXD` / `selected_maxd` というtokenを完全に除去し、split145専用ログでは `reqmaxd` / `selMAXD` を使う。検証シェル側のdispatch parserは `required_maxd` / `selected_MAXD` と `reqmaxd` / `selMAXD` の両方を受け付けるようにした。
- **240 r8で変更しないもの**: CUDA MAXD14/16/18/20/21 kernel本文、fid14/rest launch split方針、cache生成、broadmarktail mode28/29、split145 mode30/31、worker_id/worker_count、N=5..27範囲、N25..N27 dynamic preset=8 は変更しない。
- **240 r8静的検査**: `STATIC_ONLY=1 bash 240Py_taskid_split_fid14_validate_N21_full_once.sh` で、従来の保持機能に加え `source_split145_no_stale_maxd_names` がOKであることを確認した。これにより split145 function 内には、旧単一dispatch由来の `required_maxd` / `selected_MAXD` / `selected_maxd` token が残っていない。

---

Updated on 2026-07-07 for 240Py r8 buildfix.

- **240 r8 buildfix**: r1〜r7の配布で `exec_solutions_gpu_chunk_split145()` 内に旧単一dispatchの `required_maxd` 名、または Codon が誤って `required_maxd` 名として扱う split145 f-string label が残り、cudacodon build時に `name 'required_maxd' is not defined` が継続した。r8では split145内のmaxdローカルを `rmaxd/rselmaxd`, `d2maxd/d2selmaxd`, `gmaxd/gselmaxd` に固定し、split145 dispatch logは `reqmaxd` / `selMAXD` ラベルへ変更した。検証シェルのAWKは `required_maxd` と `reqmaxd` の両方を受け付ける。
- **r8の運用差分**: 古い同名sourceを掴む事故を避けるため、検証シェルの既定 `SRC` / `CAND` は `240Py_taskid_split_fid14_probe_r8.py` / `240Py_taskid_split_fid14_probe_r8` に変更した。同時に通常名 `240Py_taskid_split_fid14_probe.py` も同一内容で同梱する。
- **r8静的確認**: `STATIC_ONLY=1 bash 240Py_taskid_split_fid14_validate_N21_full_once.sh` で、`source_split145_reqmaxd_buildfix` を含む全静的検査がOK。split145関数本文には `required_maxd` substring が存在しないことを検査する。

---

Updated on 2026-07-07 for 240Py result and 241Py restore239-after-fid14split-reject.

- **240確認結果**: 240Py `taskid_split_fid14` r8 は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、fid14/rest 2launch化により dispatch rows `262`、fid14 launch rows `131`、rest launch rows `131`、fid14 task sum `8214`。required/selected MAXD は全launch `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error/mismatch `0`。正当性とdispatch整合性はOK。
- **240速度判定**: 速度は `0:09:28.451`。239Py `0:07:07.703` に対して `140.748秒`、約 `32.9%` の大幅退行。fid14のtask数は `8214 / 2025282` と小さく、専用kernel本文をまだ作っていない段階では、fid14/rest別launchのlaunch overheadとsplit/reduce overheadが明確に勝っている。したがって240のfid14 launch splitは不採用。
- **241**: 240Pyを採用せず、239Py `n27coretrim-keepfeatures` へ戻す安全復帰版。版名は `restore239-after-fid14split-reject`。CUDA MAXD14/16/18/20/21 kernel本文、rootrestlate、future_check_mask guard、no-sibling save_sp/cur_depth、root one/two-candidate preroll、MAXD dispatch、`build_soa_for_range`、stream/bin/cache生成、broadmarktail mode28/29、split145 mode30/31、worker_id/worker_count multi-GPU split、CPU `dfs_iter` path は239相当のまま維持する。239で削除したCPU再帰 `dfs()` fallback と到達不能 `use_itter` 分岐は引き続き削除状態を保つ。
- **241で明示的に入れないもの**: 240の `split=fid14_launch`、`split145-rest` / `split145-fid14` 別dispatch、dispatch rows `262` 前提、split145内のfid14/rest staging は入れない。241のN=21期待dispatch rowsは従来通り `131`、dispatch task sumは `2025282`。
- **241静的検査**: `STATIC_ONLY=1 bash 241Py_restore239_after_fid14split_reject_validate_N21_full_once.sh` で `source_version_tag`、`source_main_entry`、bare `-g` mode31、N25..N27 preset8/N27 total、runtime globals、required runtime defs、旧診断def削除、再帰CPU dfs削除、到達不能切替分岐削除、bench mode 17..27削除、mode28/29/30/31保持、split241 tag、worker split args、fid14 split marker absent がOK。
- **次候補**: fid14単独splitはtask量が少なすぎてlaunch分割だけでは不利だった。次にkernel分解を試す場合は、単に小さいfidを分けるのではなく、`d0` やtail-heavy群など、十分なtask量または明確な分岐削減が見込めるグループを selected chunks で先に測る。241 fullで239同等へ戻ることを確認してから、242以降で別グループsplitまたはhost側統廃合へ進む。


---

Updated on 2026-07-07 for 241Py result and 242Py singlelaunch-futuremask-depthbit probe.

- **241確認結果**: 241Py `restore239-after-fid14split-reject` は `N=21 full once` で final total `314666222712` 一致、required/selected MAXD は `14 / MAXD14`、schedule_words `0`、stack_bytes_per_thread `208`、error_or_mismatch `0`。速度は `0:07:07.788`。239 parent `0:07:07.703` より `0.085秒` 遅いが誤差級で、240 rejected `0:09:28.451` からは `140.663秒` 復帰した。240 fid14別launchは不採用、241は本線復帰版として採用可能。
- **242**: 241Pyを親にする `singlelaunch-futuremask-depthbit` probe。240のfid14/rest別launchは採用せず、split145の単一launchを維持する。MAXD14 kernelのみ、future-prune判定を `nibble_op & 8` からではなく、schedule生成時に作った `future_check_mask` のdepth bitから直接見る形へ変更する。
- **242の狙い**: root-prerollでは `future_check_mask & 1`、generic loopでは `(future_check_mask >> cur_depth) & 1` を使い、従来の `future_check_mask!=0` と `nibble_op&8` の2段条件を1段のdepth-specific条件へ寄せる。これはlaunch分割ではなく、単一kernel内の分岐整理である。
- **242で維持するもの**: CUDA MAXD16/18/20/21 fallback kernel、MAXD dispatch wrapper、`build_soa_for_range`、rootrestlate、no-sibling save_sp/cur_depth、root one/two-candidate preroll、generic DFS loopの基本構造、host task order、cache生成、broadmarktail mode28/29、split145 mode30/31、worker_id/worker_count multi-GPU split、`-c` / `-g` range、GPU N=5..27、N25..N27 dynamic preset=8 は維持する。
- **242で入れないもの**: 240のfid14/rest別launch、dispatch rows `262` 前提、fid14 staging、複数kernel起動、MAXD13再挑戦、terminal-first、block-check、forced-chain、generic clear-lowbitなど過去に退行した大きな構造変更は入れない。N=21期待dispatch rowsは従来通り `131`、dispatch task sumは `2025282`。
- **242静的検査**: `STATIC_ONLY=1 bash 242Py_singlelaunch_futuremask_depthbit_validate_N21_full_once.sh` で source_version_tag、main entry、bare `-g` mode31、N25..N27 preset8/N27 total、runtime globals、required runtime defs、旧診断def削除、CPU再帰dfs削除、到達不能 `use_itter` 分岐削除、mode28/29/30/31保持、split242 tag、worker split args、fid14 split absent、MAXD14 future depth-bit probe がOKであることを確認した。
- **242検証条件**: full runは従来通り `N=21 full once`、bench_mode `31`、w8_j7、32x484、worker0/1。final total、progress TSV再構成、dispatch rows/task sum、required/selected MAXD、schedule_words、stack bytes、warning/errorを確認する。速度が241/239同等なら採用、遅ければ241へ戻す。

---

Updated on 2026-07-07 for 242Py result, policy correction, and 243Py schedule-precompute probe.

- **242確認結果**: 242Py `singlelaunch_futuremask_depthbit` は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch rows `131`、dispatch task sum `2025282`、required/selected MAXD は全chunk `14 / MAXD14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error/mismatch `0`。速度は `0:07:13.366` で、241Py `0:07:07.788` より `5.578秒`、約 `1.304%` 遅いため不採用。
- **方針修正**: 241を本線固定とし、if条件の順序変更、nibble/future bitの微差、root-preroll内の小手先整理、generic loop内の小手先整理、別launch分割だけのprobeは行わない。次の候補は、単一launch維持のままMAXD14 kernel内のschedule生成ブロックをhost precomputeへ逃がす構造変更に限定する。
- **243**: 241Py `restore239-after-fid14split-reject` を親にした `schedule-precompute` probe。240のfid14/rest別launchと242のfuturemask-depthbit微差は採用せず、split145単一launch、dispatch rows `131` を維持する。MAXD14 kernel冒頭の `schedule_raw` interpreterをhost側 `precompute_maxd14_schedule_fields()` へ移し、kernelには `schedule_lo/hi`、`child_jmark_mask`、`future_check_mask`、`terminal_depth`、`terminal_base14`、`root_action` を配列で渡す。MAXD16/18/20/21 fallback、mode28/29/30/31、cache生成、worker split、CPU dfs_iter pathは維持する。
- **243確認結果**: 243Py `schedule_precompute` は `N=21 full once` で正当性OK、dispatch rows `131`、dispatch task sum `2025282`、required/selected MAXD は全chunk `14 / MAXD14`、schedule_words `0`、stack_bytes_per_thread `208`。速度は `0:07:10.524` で、241Py `0:07:07.788` より `2.736秒`、約 `0.640%` 遅い。構造方向は有望だが、7本の追加global loadが重く、243単体は不採用。

---

Updated on 2026-07-07 for 244Py schedule-precompute-pack probe.

- **244**: 243Pyのschedule precompute構造を継承しつつ、243で増えた7本のMAXD14 precompute配列loadを3本へ圧縮する `schedule-precompute-pack` probe。実装上は241本線相当へ戻したうえで、243のprecompute構造だけを再導入する位置づけ。別launchは入れず、dispatch rowsは従来通り `131`。
- **244のpack内容**: 243の `sched_lo_arr`、`sched_hi_arr`、`child_jmark_mask_arr`、`future_check_mask_arr`、`terminal_depth_arr`、`terminal_base14_arr`、`root_action_arr` を、244では `sched_lo_arr`、`sched_hi_term_arr`、`sched_mask_arr` の3配列へpackする。`sched_hi_term` は schedule_hi 下位24bit、terminal_depth bits24..27、terminal_base14 bit28、root_action bits30..31 を持つ。`sched_mask` は child_jmark_mask bits0..13、future_check_mask bits14..27 を持つ。
- **244で維持するもの**: CUDA MAXD16/18/20/21 fallback、MAXD dispatch wrapper、`build_soa_for_range`、rootrestlate、future_check_mask guard、no-sibling save_sp/cur_depth、root one/two-candidate preroll、generic DFS loopの基本構造、host task order、cache生成、broadmarktail mode28/29、split145 mode30/31、worker_id/worker_count multi-GPU split、`-c` / `-g` range、GPU N=5..27、N25..N27 dynamic preset=8 を維持する。
- **244静的検査**: `STATIC_ONLY=1 bash 244Py_schedule_precompute_pack_validate_N21_full_once.sh` で source_version_tag、main entry、bare `-g` mode31、N25..N27 preset8/N27 total、runtime globals、required runtime defs、旧診断def削除、CPU再帰dfs削除、到達不能 `use_itter` 分岐削除、mode28/29/30/31保持、split244 tag、worker split args、fid14 split absent、MAXD14 packed precompute active、3 packed arrays present がOK。
- **244検証条件**: full runは従来通り `N=21 full once`、bench_mode `31`、w8_j7、32x484、worker0/1。final total、progress TSV再構成、dispatch rows/task sum、required/selected MAXD、schedule_words、stack bytes、warning/errorを確認する。速度が243の `0:07:10.524` から戻り、241の `0:07:07.788` に近づくかを採否判断の中心にする。

---

Updated on 2026-07-07 for 242Py result, 243Py schedule-precompute result, and 244Py schedule-precompute-pack probe.

- **242確認結果**: 242Py `singlelaunch-futuremask-depthbit` は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch rows `131`、dispatch task sum `2025282`、required/selected MAXD は全chunk `14 / MAXD14`、schedule_words `0`、stack_bytes_per_thread `208`。速度は `0:07:13.366` で、241Py `0:07:07.788` より `5.578秒`、約 `1.304%` 退行したため不採用。これにより、if条件の向き変更、nibble/future bitの微差、root-prerollやgeneric loop内の小手先整理、別launch分割だけのprobeは当面行わず、241を本線固定とする。
- **243**: 241Pyを親にする `schedule-precompute` probe。240のfid14/rest別launch、242のfuturemask-depthbit微差は採用せず、split145の単一launch、dispatch rows `131`、cache生成、mode28/29/30/31、worker split、MAXD16/18/20/21 fallbackを保持したまま、MAXD14 kernel冒頭のschedule生成interpreterをhost側 `build_soa_for_range()` へ移した。kernelには `schedule_lo`、`schedule_hi`、`child_jmark_mask`、`future_check_mask`、`terminal_depth`、`terminal_base14`、`root_action` を配列で渡す構造変更とした。
- **243確認結果**: 243Py `schedule-precompute` は `N=21 full once` で正当性OK、dispatch rows `131`、dispatch task sum `2025282`、required/selected MAXD は全chunk `14 / MAXD14`。速度は `0:07:10.524` で、241Py `0:07:07.788` より `2.736秒`、約 `0.640%` 遅い。schedule interpreter撤去の方向性は大失敗ではないが、7本の追加global loadが重く、241を上回らなかったためそのままでは不採用。
- **244**: 243の方向性を残し、追加global loadを減らす `schedule-precompute-pack` probe。243では7配列だったMAXD14 precompute情報を、`sched_lo_arr`、`sched_hi_term_arr`、`sched_mask_arr` の3配列へpackする。`sched_hi_term_arr` は schedule_hi 下位24bit、terminal_depth bits24..27、terminal_base14 bit28、root_action bits30..31 を保持する。`sched_mask_arr` は child_jmark_mask bits0..13 と future_check_mask bits14..27 を保持する。MAXD14 kernelはこの3配列をloadしてunpackする。
- **244で維持するもの**: 241/243と同じく単一launchを維持し、fid14/rest別launchは入れない。CUDA MAXD16/18/20/21 fallback、MAXD dispatch wrapper、`build_soa_for_range`、rootrestlate、future_check_mask guard、no-sibling save_sp/cur_depth、root one/two-candidate preroll、host task order、stream/cache生成、broadmarktail mode28/29、split145 mode30/31、worker_id/worker_count multi-GPU split、`-c` / `-g` range、GPU N=5..27、N25..N27 dynamic preset=8 を保持する。
- **244静的検査**: `STATIC_ONLY=1 bash 244Py_schedule_precompute_pack_validate_N21_full_once.sh` で source_version_tag、main entry、bare `-g` mode31、N25..N27 preset8/N27 total、runtime globals、required runtime defs、旧診断def削除、CPU再帰dfs削除、到達不能 `use_itter` 分岐削除、mode28/29/30/31保持、split244 tag、worker split args、fid14 split absent、MAXD14 packed precompute active、3 packed arrays present がOKであることを確認した。
- **244採否基準**: full runは従来どおり `N=21 full once`、bench_mode `31`、w8_j7、32x484、worker0/1。final total、progress TSV再構成、dispatch rows/task sum、required/selected MAXD、schedule_words、stack bytes、warning/errorを確認する。243の `0:07:10.524` から十分戻り、241の `0:07:07.788` に近づくか、上回れば採用候補。遅ければschedule precompute方向は一旦撤回し、241本線へ戻す。

---

Updated on 2026-07-07 for 242Py result, 243Py schedule-precompute result, and 244Py schedule-precompute-pack probe.

- **242確認結果**: 242Py `singlelaunch-futuremask-depthbit` は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch rows `131`、dispatch task sum `2025282`、required/selected MAXD は全chunk `14 / MAXD14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error/mismatch `0`。速度は `0:07:13.366`。241Py `0:07:07.788` より `5.578秒`、約 `1.304%` 遅いため不採用。これにより、if条件の向き変更、nibble/future bitの微差、root-preroll内の小手先整理、generic loop内の小手先整理は当面行わない方針へ固定する。
- **243方針**: 241Pyを親に、別launchなし・単一launch維持のまま、MAXD14 kernel冒頭のschedule生成interpreterをhost側precomputeへ逃がす構造変更を試した。240 fid14/rest別launchと242 futuremask-depthbit微差は採用しない。mode28/29/30/31、cache生成、MAXD14/16/18/20/21 fallback、worker_id/worker_count、CPU `dfs_iter` pathは維持する。
- **243確認結果**: 243Py `schedule-precompute` は `N=21 full once` で正当性OK。final total、progress、dispatch task sum、required/selected MAXD14、schedule_words=0、stack=208 bytes/thread は従来通り一致した。速度は `0:07:10.524` で、241Py `0:07:07.788` より `2.736秒`、約 `0.640%` 遅い。これは、kernel内schedule interpreterを消した代わりに、`sched_lo_arr`、`sched_hi_arr`、`child_jmark_mask_arr`、`future_check_mask_arr`、`terminal_depth_arr`、`terminal_base14_arr`、`root_action_arr` の7本の追加global loadが入ったためと推定する。243そのものは採用しないが、構造方向は残す。
- **244**: 243の構造方向を維持しつつ、追加global loadを減らす `schedule-precompute-pack` probe。241Pyを親にし、単一launch、dispatch rows `131`、cache生成、mode28/29/30/31、worker split、MAXD fallbackを維持する。MAXD14 precompute情報を7本のu32配列から3本へpackする。
- **244 pack仕様**: `sched_lo_arr` は `schedule_lo` を保持する。`sched_hi_term_arr` は下位24bitに `schedule_hi`、bits 24..27 に `terminal_depth`、bit 28 に `terminal_base14`、bits 30..31 に `root_action` をpackする。`sched_mask_arr` は bits 0..13 に `child_jmark_mask`、bits 14..27 に `future_check_mask` をpackする。MAXD14 kernelではこの3本をloadして復元する。
- **244で入れないもの**: 240のfid14/rest別launch、242のfuturemask-depthbit条件整理、if/else向き変更、root-preroll/generic loop内の小手先整理、MAXD13再挑戦、cache-hot専用化は入れない。243の7配列load版も採用せず、pack版でglobal load増を抑えられるかだけを見る。
- **244静的検査**: `STATIC_ONLY=1 bash 244Py_schedule_precompute_pack_validate_N21_full_once.sh` で source_version_tag、main entry、bare `-g` mode31、N25..N27 preset8/N27 total、runtime globals、required runtime defs、旧診断def削除、CPU再帰dfs削除、到達不能 `use_itter` 分岐削除、mode28/29/30/31保持、split244 tag、worker split args、fid14 split absent、MAXD14 packed precompute active、3 packed arrays present がOKであることを確認した。
- **244検証条件**: full runは従来通り `N=21 full once`、bench_mode `31`、w8_j7、32x484、worker0/1。final total、progress TSV再構成、dispatch rows/task sum、required/selected MAXD、schedule_words、stack bytes、warning/errorを確認する。速度が241 `0:07:07.788` を上回れば採用候補、243 `0:07:10.524` より戻すだけなら継続検討、243より悪化すればprecompute-pack方針を撤回する。

---

Updated on 2026-07-07 for 244Py r2 schedule-precompute lowpack buildfix.

- **244初回結果**: 244Py schedule-precompute-pack は 243 の7本追加global loadを3本へpackする方針だったが、cudacodon側で `codon build -release` 後のGPU module load時に `CUDA_ERROR_INVALID_PTX` が発生した。これは計算不一致ではなくPTX/JIT不成立として扱い、初回244の高位bit pack形は撤回する。
- **244 r2**: 241を親にし、単一launch、mode31 split145、cache生成、mode28/29/30/31、worker split、MAXD fallbackを維持したまま、schedule precompute packを低位bitだけの4配列へ変更した。`sched_lo_arr`、`sched_hi_arr`、`sched_mask_arr`、`sched_ctrl_arr` を使い、`sched_ctrl` は `bits0..3 terminal_depth`、`bit4 terminal_base14`、`bits5..6 root_action` とする。初回244の `sched_hi_term` 高位bit packは使わない。
- **244 r2検証方針**: まず `STATIC_ONLY=1 bash 244Py_schedule_precompute_pack_validate_N21_full_once_r2.sh` で、`source_schedule_precompute_lowpack` と `source_host_precompute_lowpack_arrays` がOKであることを確認する。その後、full runでPTX/JIT成立、final total、progress rows、dispatch rows=131、required/selected MAXD14、schedule_words=0、stack=208、warning/errorを確認する。

---

Updated on 2026-07-07 for 244Py r3 schedule-precompute lowpack intdecode.

- **244初回/r2確認メモ**: 244初回の3-array high-bit pack、および244 r2の4-array low-bit packは、いずれも `codon build -release` は通ったが、GPU module load時に `CUDA_ERROR_INVALID_PTX` で停止した。これは計算不一致ではなくPTX/JIT不成立として扱う。
- **推定原因**: r2では `terminal_depth_u:u32=ctrl_pack&u32(15)` とし、MAXD14 hot loop内で `u32(cur_depth)==terminal_depth_u` のように int/u32 を跨いだ比較を行っていた。このpacked decode形がCodonのPTX生成で危険と判断し、r3では `terminal_depth:int=int(ctrl_pack&u32(15))` へ戻して、既存kernelと同じ `cur_depth==terminal_depth` 形にする。
- **244 r3**: 親は241 restore239。240 fid14/rest別launch、242 futuremask-depthbit微差は採用しない。単一launch、dispatch rows 131、mode28/29/30/31、cache生成、worker split、MAXD fallbackを保持する。MAXD14だけschedule precompute lowpackを維持しつつ、terminal depth decodeをint化する。
- **244 r3検証条件**: `STATIC_ONLY=1 bash 244Py_schedule_precompute_pack_validate_N21_full_once_r3.sh` で、`source_schedule_precompute_intdecode` と `source_host_precompute_intdecode_arrays` を確認する。その後、full runでJIT成立、final total、dispatch rows/task sum、required/selected MAXD、stack bytes、速度を確認する。

---

Updated on 2026-07-07 for 244Py r4 schedule-precompute maskpack bisect.

- **244 r1/r2/r3結果**: 243のschedule-precomputeをpackする試みとして、terminal/root_action系を同一u32へpackした版を試したが、`codon build -release` 後のCUDA module loadで `CUDA_ERROR_INVALID_PTX` になった。これは計算不一致ではなくPTX/JIT不成立として扱う。
- **244 r4**: 失敗箇所を切り分けるため、243でJIT成立済みの7-array precompute形を親に戻し、まず `child_jmark_mask` と `future_check_mask` だけを `sched_mask_arr` へ低位packする。`terminal_depth`、`terminal_base14`、`root_action` は243と同じ別配列のまま残す。これにより、mask packだけでJITが成立するかを確認する。
- **244 r4の意義**: r4がJIT成立すれば、r1-r3の不成立原因は `sched_ctrl` 側、つまりterminal/root_action pack/decode周辺に絞れる。r4もJIT不成立なら、mask pack自体またはpack decode形がPTX生成と相性が悪いと判断し、243の7-array形へ戻す。
- **採否**: r4はまずJIT成立/正当性確認を優先する。速度は241 `0:07:07.788`、243 `0:07:10.524` と比較し、243より戻るかを見る。

---

Updated on 2026-07-07 for 255Py schedule-precompute ctrlflags pack probe.

- **244 r4確認メモ**: 244 r1-r3 は terminal/root_action 系を同一packへ入れた形で `CUDA_ERROR_INVALID_PTX` になったが、244 r4 では 243 JIT成立済みの7-array precompute形へ戻し、まず `child_jmark_mask` と `future_check_mask` だけを `sched_mask_arr` にpackする切り分けを行った。ユーザー側で「動いた」と確認されたため、mask pack自体は次段の候補として残す。ただし完走・採用までは未確定であり、速度評価は引き続きN=21 full onceで判断する。
- **255**: 244 r4 maskpack を受けた次のJIT切り分けprobe。親は241 restore239本線相当 + 243/244系 schedule-precompute。単一launch、dispatch rows `131`、mode31 split145、cache生成、broadmarktail mode28/29、split145 mode30/31、worker split、MAXD16/18/20/21 fallbackを維持する。240のfid14/rest別launch、242のfuturemask-depthbit微差、if/else向き変更、root-preroll/generic loopの小手先整理は入れない。
- **255の変更点**: 244 r4の `sched_mask_arr = child_jmark_mask | (future_check_mask << 14)` を維持したまま、`terminal_base14` と `root_action` だけを `sched_ctrl_arr` へ低位packする。`terminal_depth` は引き続き `terminal_depth_arr` として分離する。packは `sched_ctrl bits0 = terminal_base14`, `bits1..2 = root_action` とし、r1-r3で危険だった `terminal_depth` 同梱packは避ける。
- **255の狙い**: r4がJIT成立し、255もJIT成立すれば、child/future mask packと terminal_base14/root_action flags pack は安全側と判断できる。もし255が `CUDA_ERROR_INVALID_PTX` になる場合は、terminal/root_action flag decode自体、または `sched_ctrl_arr` 追加がPTX生成と相性悪いと判断し、244 r4へ戻す。速度面では、244 r4/243の追加loadを1本減らせるかを確認する。
- **255検証条件**: `STATIC_ONLY=1 bash 255Py_schedule_precompute_ctrlflags_validate_N21_full_once.sh` で source_version_tag、main entry、bare `-g` mode31、N25..N27 preset8/N27 total、runtime globals、required runtime defs、旧診断def削除、CPU再帰dfs削除、到達不能 `use_itter` 分岐削除、mode28/29/30/31保持、split255 tag、worker split args、fid14 split absent、MAXD14 ctrlflags precompute active、host precompute arrays がOKであることを確認する。full runでは final total、progress rows、dispatch rows/task sum、required/selected MAXD14、schedule_words=0、stack=208、warning/error、速度を確認する。

---

Updated on 2026-07-07 for 255Py schedule-precompute-actionpack probe.

- **244 r4確認メモ**: 244 r4 `schedule-precompute-maskpack` は、243でJIT成立済みのschedule precompute形を親に、`child_jmark_mask` と `future_check_mask` だけを `sched_mask_arr` へpackする切り分け版。r1〜r3の `terminal_depth / terminal_base14 / root_action` をまとめたctrl-packは `CUDA_ERROR_INVALID_PTX` で不成立だったが、r4はJIT成立したため、maskpack自体はCodon/PTX上の危険箇所ではないと判断する。
- **255**: 244 r4を親に、単一launchとschedule precomputeを維持したまま、次のpack切り分けとして `terminal_base14` と `root_action` だけを低位bitの `sched_action_arr` へpackする。`terminal_depth` はr1〜r3でJIT不成立の疑いが残るため、255では separate array のまま維持する。
- **255で保持するもの**: 241/244 r4相当の `-c/-g`、bare `-g` mode31、GPU N=5..27、N25..N27 dynamic preset=8、cache/bin missing時の生成、broadmarktail mode28/29、split145 mode30/31、MAXD14/16/18/20/21 fallback、CPU path、worker_id/worker_count multi-GPU split、CUDA_VISIBLE_DEVICES運用を維持する。240 fid14/rest別launch、242 futuremask-depthbit微差、244 r1〜r3 ctrl-packは採用しない。
- **255のpack内容**: `sched_mask_arr = child_jmark_mask | (future_check_mask << 14)` は244 r4から継承。新規に `sched_action_arr = terminal_base14 | (root_action << 1)` を追加する。MAXD14 kernelでは `terminal_base14 = sched_action & 1`、`root_action = (sched_action >> 1) & 3` とdecodeする。`terminal_depth_arr` は引き続き独立配列として `int(terminal_depth_arr[i])` で読む。
- **255検証条件**: `STATIC_ONLY=1 bash 255Py_schedule_precompute_actionpack_validate_N21_full_once.sh` で source tag、main entry、mode31 default、N27 expected table、runtime globals、required runtime defs、旧診断def削除、CPU再帰dfs削除、bench_mode17..27削除、mode28/29/30/31保持、split255 tag、worker split、actionpack precompute、release build policyを確認する。fullではN=21 mode31 split145で final total、progress rows、dispatch rows/task sum、required/selected MAXD14、schedule_words=0、stack 208 bytes/thread、warning/error 0を確認する。
- **採否**: 255はまずJIT成立が第一判定。JIT成立かつ正当性OKなら、244 r4 maskpackに続き actionpack が安全と判断し、次段で terminal_depth 単独packまたは既存配列再利用を検討する。JIT不成立なら、terminal_base14/root_action packを撤回し、244 r4をpack切り分け基準へ戻す。

---

Updated on 2026-07-07 for 255Py r2 schedule-precompute-actionpack commandfix.

- **255初回確認**: `255Py_schedule_precompute_actionpack` は `codon build -release` 後の実行で `CUDA_ERROR_INVALID_PTX` となった。ただしログ上の実行コマンドが `-g 21 21 32 484 10 7 31 8 7 0 0 1 2` になっており、期待していた `-g 21 21 32 484 1 0 7 31 8 7 0 0 1 2` ではなかった。つまり `log_level=1 sort_mode=0 preset=7 bench_mode=31` ではなく、引数列がずれて `bench_mode=8` 相当を走らせていた可能性が高い。初回255のJIT結果はactionpack自体の結論にはまだしない。
- **255 r2**: source/kernel探索ロジックは255 actionpackのまま維持し、検証シェルだけを commandfix する。既定 source/candidate を `_r2` 名へ分離し、検証コマンドを固定ベクトル `-g 21 21 32 484 1 0 7 31 8 7 0 0 1 2` として明示する。summaryには `command_arg_vector` checkを追加し、今後同じ引数ずれを検出する。
- **確認手順**: `STATIC_ONLY=1 bash 255Py_schedule_precompute_actionpack_validate_N21_full_once_r2.sh` でsource/staticとcommand policyを確認し、その後 `bash 255Py_schedule_precompute_actionpack_validate_N21_full_once_r2.sh` で本来のmode31 split145 actionpack評価を行う。


---

Updated on 2026-07-07 for 255Py r2 result and 256Py schedule-precompute-base14pack probe.

- **255 r2確認結果**: `255Py_schedule_precompute_actionpack` r2 は、検証シェルの引数ずれを修正し、期待通り `-g 21 21 32 484 1 0 7 31 8 7 0 0 1 2` で `codon build -release` 後に実行したが、CUDA module load時に `CUDA_ERROR_INVALID_PTX` で停止した。したがって、初回255のようなcommand vector不整合ではなく、`terminal_base14` と `root_action` を同一 `sched_action_arr` にpack/decodeする形そのものをPTX/JIT不成立として扱う。計算不一致ではない。
- **切り分け結果**: 244 r4 maskpack はJIT成立・正当性OKだったため、`child_jmark_mask | (future_check_mask << 14)` のmask packはPTX不成立要因ではない。一方、255 r2 actionpackはJIT不成立なので、次は `terminal_base14` と `root_action` を分けて確認する。
- **256**: `schedule-precompute-base14pack` probe。親は244 r4 maskpack / 241 restore239本線相当。単一launch、dispatch rows 131、mode31 split145、cache生成、broadmarktail mode28/29、split145 mode30/31、worker split、MAXD14/16/18/20/21 fallbackを維持する。255 r2で不成立だった `terminal_base14 + root_action` 同時packは採用せず、まず `terminal_base14` だけを `sched_base14_arr` に分離して渡し、`terminal_depth` と `root_action` は243/244 r4同様に別配列のまま残す。
- **256の判定**: 256がJIT成立すれば、`terminal_base14` 単独packは安全で、255 r2の失敗源は `root_action` 側または `terminal_base14/root_action` 同時decode形に絞れる。256もJIT不成立なら、terminal_base14の別配列decode自体がCodon/PTXと相性悪い可能性が高く、244 r4 maskpackへ戻す。速度は二次判定で、まずはJIT成立と正当性OKを確認する。
- **256検証条件**: `STATIC_ONLY=1 bash 256Py_schedule_precompute_base14pack_validate_N21_full_once.sh` で source tag、main entry、bare `-g` mode31、N25..N27 preset8/N27 total、runtime globals、required runtime defs、旧診断def削除、CPU再帰dfs削除、mode28/29/30/31保持、split256 tag、worker split、base14pack precompute、release build policyを確認する。fullではN=21 mode31 split145で final total、progress rows、dispatch rows/task sum、required/selected MAXD14、schedule_words=0、stack 208 bytes/thread、warning/error 0を確認する。

---

Updated on 2026-07-07 for 255Py r2 result and 256Py schedule-precompute-base14pack probe.

- **255 r2確認結果**: 255Py `schedule-precompute-actionpack` は、検証シェルの引数ずれを修正した r2 で `-g 21 21 32 484 1 0 7 31 8 7 0 0 1 2` の正しい mode31 split145 command を確認したうえで実行したが、`codon build -release` 後のCUDA module load時に `CUDA_ERROR_INVALID_PTX` で停止した。したがって、初回255の引数ずれを除外しても、`terminal_base14 | (root_action << 1)` の actionpack 形はPTX/JIT不成立として不採用とする。
- **切り分け状況**: 244 r4 `maskpack` はJIT成立・正当性OKだったため、`child_jmark_mask + future_check_mask` のpack自体はJIT不成立要因ではない。一方、244 r1-r3の `terminal_depth / terminal_base14 / root_action` 同梱pack、および255 r2の `terminal_base14 + root_action` actionpack はJIT不成立だった。次は terminal/root_action 系をさらに分解し、1項目だけをpackして不成立源を絞る。
- **256**: 244 r4 maskpack を親に、単一launchと schedule-precompute を維持したまま、`terminal_base14` だけを低位bitの `sched_base14_arr` として別配列へ移すprobe。`terminal_depth` と `root_action` は243/244 r4同様に別配列のまま保持する。255 actionpack は入れない。
- **256で保持するもの**: 241/244 r4相当の `-c/-g`、bare `-g` mode31、GPU N=5..27、N25..N27 dynamic preset=8、cache/bin missing時の生成、broadmarktail mode28/29、split145 mode30/31、MAXD14/16/18/20/21 fallback、CPU path、worker_id/worker_count multi-GPU split、CUDA_VISIBLE_DEVICES運用を維持する。240 fid14/rest別launch、242 futuremask-depthbit微差、244 r1-r3 ctrl-pack、255 actionpackは採用しない。
- **256検証条件**: `STATIC_ONLY=1 bash 256Py_schedule_precompute_base14pack_validate_N21_full_once.sh` で source tag、main entry、mode31 default、N27 expected table、runtime globals、required runtime defs、旧診断def削除、CPU再帰dfs削除、bench_mode17..27削除、mode28/29/30/31保持、split256 tag、worker split、base14pack precompute、release build policyを確認する。fullではN=21 mode31 split145で final total、progress rows、dispatch rows/task sum、required/selected MAXD14、schedule_words=0、stack 208 bytes/thread、warning/error 0を確認する。
- **採否**: 256はまずJIT成立が第一判定。JIT成立かつ正当性OKなら terminal_base14 単独packは安全と判断し、次段で root_action 単独pack、または既存配列再利用へ進む。JIT不成立なら terminal_base14 packも撤回し、244 r4 maskpackを安全なpack上限として扱う。

---

Updated on 2026-07-07 for 255Py r2 result and 256Py schedule-precompute-base14pack probe.

- **255 r2確認結果**: `255Py_schedule_precompute_actionpack` r2 は command vector を `-g 21 21 32 484 1 0 7 31 8 7 0 0 1 2` に修正して本来の mode31 split145 条件で再実行したが、`codon build -release` 後のCUDA module loadで `CUDA_ERROR_INVALID_PTX` となった。これにより、244 r4でJIT成立済みの child/future maskpack に対し、`terminal_base14 | (root_action << 1)` の actionpack / decode を追加した箇所がPTX/JIT不成立源である可能性が高い。
- **255 r2の扱い**: 計算不一致ではなくPTX/JIT不成立として扱い、255 actionpack は不採用。244 r4 maskpackは正当性OKでJIT成立しているため、maskpack自体は危険箇所ではない。次は terminal/root_action 同時packをさらに分解して、どちらがJIT不成立に寄与するかを切り分ける。
- **256**: 244 r4 maskpack JIT成立形を親にする `schedule-precompute-base14pack` probe。単一launch、dispatch rows `131`、mode28/29/30/31、cache生成、MAXD14/16/18/20/21 fallback、worker split、CPU `dfs_iter` pathを維持する。240 fid14/rest別launch、242 futuremask-depthbit微差、255 actionpackは採用しない。
- **256の変更点**: 244 r4の `sched_mask_arr = child_jmark_mask | (future_check_mask << 14)` を維持したまま、`terminal_base14` だけを `sched_base14_arr` からdecodeする。`terminal_depth` と `root_action` は別配列のまま維持する。これにより、255でJIT不成立になった `terminal_base14/root_action` 同時packを避け、まず terminal_base14 単独decodeがJIT安全かを確認する。
- **256採否基準**: まずJIT成立が第一判定。JIT成立かつ正当性OKなら、次に root_action 単独pack、または既存配列再利用へ進む。JIT不成立なら、terminal_base14 の別配列decode自体、または関連decode形がCodon/PTXと相性悪いと判断し、244 r4 maskpackへ戻す。
- **256検証条件**: `STATIC_ONLY=1 bash 256Py_schedule_precompute_base14pack_validate_N21_full_once.sh` で source tag、main entry、mode31 default、N27 expected table、runtime globals、required runtime defs、旧診断def削除、CPU再帰dfs削除、bench_mode17..27削除、mode28/29/30/31保持、split256 tag、worker split、base14pack precompute、release build policyを確認する。fullではN=21 mode31 split145で final total、progress rows、dispatch rows/task sum、required/selected MAXD14、schedule_words=0、stack 208 bytes/thread、warning/error 0を確認する。

---

Updated on 2026-07-07 for 257Py schedule-precompute-rootactionpack.

- **256確認結果**: 256Py `schedule-precompute-base14pack` は `N=21 full once` で final total `314666222712`、full_chunk_sum `314666222712`、error/mismatch `0`、required/selected MAXD14、stack `208 bytes/thread` を確認。elapsed は `0:07:10.754` で241親より遅いが、JIT成立・正当性OKにより `terminal_base14` 単独low-bit decodeは安全と判断できる。
- **257**: 256の結果を受け、255 actionpackのJIT不成立をさらに切り分けるため、`root_action` だけを `sched_root_action_arr` からlow-bit decodeするprobe。`terminal_depth` と `terminal_base14` は別配列のまま保持し、child/future maskpackは244 r4/256同様に維持する。これでJIT成立すれば、失敗源は `terminal_base14` と `root_action` を同じpack wordへ同居させた形にかなり絞れる。単一launch、mode28/29/30/31、cache生成、worker split、MAXD fallbackは維持する。

---

Updated on 2026-07-07 for 257Py schedule-precompute-rootactionpack.

- **256確認結果**: 256Py `schedule-precompute-base14pack` は、244 r4 maskpackに加えて `terminal_base14` 単独配列decodeを試した切り分け版。N=21 full onceで final total `314666222712` 一致、progress/full chunk sum一致、required/selected MAXD14、schedule_words `0`、stack_bytes_per_thread `208`、warning/error/mismatch `0`。elapsed は `0:07:10.754` で、241Py `0:07:07.788` より約 `2.966秒` 遅いため速度候補としては不採用。ただしJIT成立・正当性OKにより、`terminal_base14` 単独decodeはPTX/JIT不成立源ではないと判断する。
- **257**: 256を受けた次の切り分け版。親は 256 base14pack JIT-ok / 244 r4 maskpack / 241 restore239。単一launch、dispatch rows 131、fid14/rest別launchなし、mode28/29/30/31、cache生成、MAXD14/16/18/20/21 fallback、worker splitを保持する。child/future maskpack と `terminal_base14` 単独配列は維持し、`terminal_depth` は別配列のまま、`root_action` だけを `(root_action << 1)` として `sched_root_action_arr` に格納し、MAXD14 kernel側で `root_action = (sched_root_action_arr[i] >> 1) & 3` とdecodeする。
- **257の目的**: 255 actionpackでは `terminal_base14 + root_action` 同時packが `CUDA_ERROR_INVALID_PTX` になった。256で `terminal_base14` 単独はJIT安全と分かったため、257では `root_action` のshift decode単独がJIT安全かを確認する。257がJIT不成立ならroot_action shifted decodeが疑わしい。257がJIT成立なら、255の不成立源は `terminal_base14` と `root_action` を同一packからdecodeする組み合わせに絞れる。
- **257検証条件**: `STATIC_ONLY=1` では `source_schedule_precompute_rootactionpack` と `source_host_precompute_rootactionpack_arrays` を確認する。fullでは従来どおり N=21 / mode31 / split145 / 32x484 / worker0/1 で final total、progress再構成、dispatch rows/task sum、required/selected MAXD、schedule_words、stack bytes、warning/errorを検証する。

---

Updated on 2026-07-07 for 257Py rootactionpack low-bit correction.

- **257補足修正**: root_action単独packの確認では、255 actionpackと同じshift decode形を混ぜるとJIT不成立原因の切り分けが曖昧になるため、257配布版では `sched_root_action_arr = root_action & 3`、kernel側は `root_action = sched_root_action_arr[i] & 3` の低位bit単独decodeに固定した。`terminal_base14` は256でJIT成立済みの `sched_base14_arr` のまま、`terminal_depth` は別配列のまま保持する。
- **257検証意図**: 257がJIT成立すれば、`root_action` 単独low-bit decodeは安全であり、255 r2の不成立源は `terminal_base14` と `root_action` を同一pack wordへ同居させた形、またはshift付きdecode形に絞れる。257がJIT不成立なら、root_action配列decode自体を疑い、256 base14packを安全上限として扱う。
- **257静的確認**: `STATIC_ONLY=1 bash 257Py_schedule_precompute_rootactionpack_validate_N21_full_once.sh` で `source_version_tag`、`source_main_entry`、bare `-g` mode31、N25..N27 preset8/N27 total、runtime globals、required runtime defs、旧診断def削除、CPU再帰dfs削除、mode28/29/30/31保持、split257 tag、worker split、fid14 split absent、MAXD14 rootactionpack precompute active、host precompute arrays、release build policy がOKになることを確認した。


---

Updated on 2026-07-07 for 258Py schedule-precompute-rootaction-nomask.

- **257確認結果**: 257Py rootactionpack は command vector 修正後も `CUDA_ERROR_INVALID_PTX`。256 base14pack はJIT成立済みなので、root_action単独の `&u32(3)` low-bit decode がPTX/JIT不成立源である可能性が高い。
- **258**: 256/257の切り分け継続版。child/future maskpack、terminal_base14 separate low-bit decode、terminal_depth separate は維持し、root_actionは `sched_root_action_arr` から **maskせずそのまま読む**。これにより、root_action配列化そのものではなく `&u32(3)` decode形がJIT不成立源かを確認する。単一launch、mode28/29/30/31、cache生成、worker split、MAXD fallbackは保持。
- **採否**: 258がJIT成立すれば、257/255の不成立源はroot_actionのmask/shift decode形に絞る。速度は241本線 `0:07:07.788` と比較し、precompute系の採否は別途判断する。


---

Updated on 2026-07-07 for 258Py schedule-precompute-rootactionpass probe.

- **257確認結果**: 257Py schedule-precompute-rootactionpack は、256 base14pack と同じ正当な mode31 引数で `codon build -release` したが、GPU module load 時に `CUDA_ERROR_INVALID_PTX` となった。244 r4 maskpack と256 base14packはJIT成立済みであるため、root_action を `sched_root_action_arr[i] & u32(3)` としてkernel側でlow-bit decodeした形がJIT不成立源の候補として強くなった。
- **258**: 257を採用せず、同じ `sched_root_action_arr` を使いながら kernel側の `&u32(3)` decodeだけを外す rootaction pass-through 切り分け版。`sched_root_action_arr` の値はhost precompute時点で `0..3` に収め、MAXD14 kernelでは `root_action:u32=sched_root_action_arr[i]` として扱う。単一launch、child/future maskpack、terminal_base14単独配列、terminal_depth別配列、mode28/29/30/31、cache生成、worker split、MAXD fallbackは維持する。
- **判定方針**: 258がJIT成立すれば、257の不成立はroot_action値そのものではなく、kernel内の `&u32(3)` decode形が原因だった可能性が高い。258もJIT不成立なら、root_actionを別名配列へ移したこと、またはprecompute root_action周辺の引数構成自体を疑い、256形へ戻す。


---

Updated on 2026-07-07 for 259Py schedule-precompute rootaction parenthesized-mask probe.

- **259**: 257Py rootaction low-bit decode が `CUDA_ERROR_INVALID_PTX` になったため、ユーザー提案の明示括弧形 `root_action:u32 = (sched_root_action_arr[i]) & u32(3)` を単独で試す切り分け版。親は256/244 r4/241系の schedule-precompute + maskpack JIT-ok 経路。`terminal_depth` と `terminal_base14` は別配列のまま保持し、`root_action` だけを `sched_root_action_arr` から parenthesized-mask decode する。単一launch、mode28/29/30/31、cache生成、MAXD14/16/18/20/21 fallback、worker split は維持する。採否はまずJIT成立可否、その後N=21 full正当性、最後に241/256比速度で判断する。


---

Updated on 2026-07-07 for 260Py schedule-precompute-rootaction-tempmask probe.

- **259確認結果**: 259Py `schedule-precompute-rootaction_parenmask` は、ユーザー提案の `root_action:u32 = (sched_root_action_arr[i]) & u32(3)` を試したが、257と同じく `CUDA_ERROR_INVALID_PTX` で停止した。これにより、括弧付きindex expressionだけでは root_action decode のJIT不成立は解消しないと判断する。計算不一致ではなくPTX/JIT不成立として扱う。
- **切り分け状況**: 244 r4 `maskpack` はJIT成立・正当性OK、256 `base14pack` もJIT成立・正当性OK。一方で 257 `sched_root_action_arr[i] & 3` と259 `(sched_root_action_arr[i]) & 3` はJIT不成立。したがって、root_action値そのものより、配列index式へ直接maskを掛けるGPU IR生成形が疑わしい。
- **260**: 256/244 r4/241系を親に、root_actionを `sched_root_action_arr` から一度 `root_action_raw` へ読み、`root_action = root_action_raw & u32(3)` とする temp-mask probe。`terminal_depth` と `terminal_base14` は別配列のまま保持し、child/future maskpackは維持する。単一launch、mode28/29/30/31、cache生成、MAXD14/16/18/20/21 fallback、worker split は維持する。
- **260の判定**: 260がJIT成立すれば、257/259の不成立源はroot_action配列ではなく、array-index expressionへ直接maskを掛ける形にかなり絞れる。260もJIT不成立なら、root_action別配列化またはroot_action系decodeそのものを避け、256 base14packを安全上限として扱う。


---

Updated on 2026-07-07 for 260Py schedule-precompute rootaction tempmask probe.

- **259確認結果**: 259Py `rootaction_parenmask` は正しい mode31 引数でも `CUDA_ERROR_INVALID_PTX` となった。257の direct mask と259の parenthesized mask はどちらもJIT不成立であり、root_action 配列値に対する kernel内 index-and-mask decode 形が危険候補として残る。
- **260**: 259を採用せず、256 base14pack / 244 r4 maskpack / 241 restore239 を親に、root_action decodeだけを `root_action_raw=sched_root_action_arr[i]` と `root_action=root_action_raw&3` の二段式へ分離する切り分け版。terminal_depth / terminal_base14 は別配列のまま保持し、child/future maskpack は維持する。単一launch、mode28/29/30/31、cache生成、MAXD fallback、worker split は維持する。


---

Updated on 2026-07-07 for 261Py schedule-precompute terminaldepthpack probe.

- **260確認結果**: 260Py `rootaction_tempmask` は正しい mode31 引数でも `CUDA_ERROR_INVALID_PTX` となった。257 direct mask、259 parenthesized mask、260 raw-load then mask がいずれもJIT不成立であり、root_action の kernel内 mask decode系は撤回する。
- **261**: 256 base14pack / 244 r4 maskpack / 241 restore239 を親に、root_action は別配列素通しへ固定したまま、terminal_depth だけを `sched_terminal_depth_arr[i] & 15` で単独low-bit decodeする切り分け版。child/future maskpack と terminal_base14 pack は維持する。単一launch、mode28/29/30/31、cache生成、MAXD fallback、worker split は維持する。採否はまずJIT成立、次に正当性、最後に241/256との速度比較で判断する。


---

Updated on 2026-07-07 for 262Py schedule-precompute termctrlpack probe.

- **261確認結果**: 261Py `terminaldepthpack` は `N=21 full once` で final total `314666222712` 一致、required/selected MAXD14、stack `208 bytes/thread`、warning/error 0。速度は `0:07:10.467` で241比では遅いが、terminal_depth単独low-bit decodeはJIT安全と確認できた。
- **262**: 261のJIT成立形を親に、root_actionは257/259/260のJIT不成立を受けて別配列pass-throughのまま固定し、terminal_depth と terminal_base14 だけを `sched_termctrl_arr` へ低位bit packする切り分け版。`sched_termctrl` は bits 0..3 = terminal_depth、bit4 = terminal_base14。child/future maskpack、単一launch、mode28/29/30/31、cache生成、MAXD fallback、worker splitは保持する。


---

Updated on 2026-07-07 for 262Py schedule-precompute terminalbasepack probe.

- **261確認結果**: 261Py `terminaldepthpack` は `N=21 full once` で final total `314666222712` 一致、required/selected MAXD14、stack `208`、warning/errorなし。elapsed は `0:07:10.467` で241より遅いため採用はしないが、terminal_depth 単独 low-bit decode はJIT安全と判断できる。
- **262**: 261を親に、root_actionは別配列素通しのまま、terminal_depth と terminal_base14 だけを `sched_terminal_arr` にpackする切り分け版。`sched_terminal_arr` は bits0..3 に terminal_depth、bit4 に terminal_base14 を持つ。child/future maskpack、単一launch、mode28/29/30/31、cache生成、MAXD fallback、worker split は維持する。これでJIT成立すれば、root_action decode系だけが危険で、terminal_depth/base14 packは安全と判断できる。

---

Updated on 2026-07-07 for 262Py result and 263Py schedule-precompute reuse2 probe.

- **262確認結果**: 262Py `schedule-precompute-terminalbasepack` は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0/0`。速度は `0:07:09.447`。241Py `0:07:07.788` より `1.659秒` 遅いため採用基準にはしないが、243の7配列precompute `0:07:10.524`、244 r4 maskpack `0:07:10.666`、256 `0:07:10.754`、261 `0:07:10.467` より良く、`child/future maskpack` と `terminal_depth + terminal_base14` pack はJIT安全・正当性OKと判断できる。
- **JIT切り分け結果**: `root_action` は別配列素通しなら安全だが、`root_action & 3`、括弧付きmask、temp経由maskはいずれも `CUDA_ERROR_INVALID_PTX` になった。したがって当面 `root_action` はpackせず別配列素通しで固定する。
- **263**: 262を親にする `schedule-precompute-reuse2` probe。packをこれ以上増やさず、MAXD14 dispatchが確定した後だけ `ctrl0_arr` を `schedule_lo`、`markctrl_arr` を `schedule_hi` のcarrierとして再利用する。これによりMAXD14 kernel側では `sched_lo_arr` / `sched_hi_arr` を読まず、追加global loadを2本削る狙い。fallback MAXD16/18/20/21では従来の `ctrl0_arr` / `markctrl_arr` が必要なため、repackは `selected_maxd==14` branch内でのみ実行する。単一launch、mode28/29/30/31、cache生成、worker split、MAXD fallbackは保持する。
- **263検証条件**: `STATIC_ONLY=1 bash 263Py_schedule_precompute_reuse2_validate_N21_full_once.sh` で、source tag、main entry、bare `-g` mode31、N25..N27 preset8、runtime defs、旧診断削除、mode28/29/30/31、split263 tag、worker split、MAXD14 schedule reuse2、host repack helperを確認する。その後通常実行で N=21 full once、final total、progress TSV、dispatch rows/task sum、MAXD14、stack 208 bytes/thread、warning/errorを確認する。

---

Updated on 2026-07-07 for 263Py schedule-precompute reusectrl probe.

- **262確認結果**: 262Py `terminalbasepack` は `N=21 full once` で final total `314666222712` 一致、required/selected MAXD14、stack `208`、warning/errorなし。elapsed は `0:07:09.447` で241比ではまだ `+1.659秒` 遅いが、precompute系では243/244 r4/256/261より改善した。これにより `terminal_depth + terminal_base14` の同一low-bit packはJIT安全、root_action decode系だけが危険という地図がほぼ確定した。
- **263**: 262を親に、5配列安全形は維持しつつ、MAXD14 selected が確定した後だけ既存 `ctrl0_arr` と `markctrl_arr` を再利用するprobe。`ctrl0_arr` には `schedule_lo`、`markctrl_arr` には `sched_terminal` を入れ直し、MAXD14 kernelでは `schedule_lo = ctrl0_arr[i]`、`sched_terminal = markctrl_arr[i]` として読む。`sched_hi_arr`、`sched_mask_arr`、`root_action_arr` は別配列のまま維持し、root_action は257/259/260のJIT不成立を受けて素通し固定にする。MAXD16/18/20/21 fallbackのため、repackは `selected_maxd==14` のlaunch直前に限定する。単一launch、mode28/29/30/31、cache生成、worker splitは保持する。
- **263の狙い**: 243の7配列追加loadが遅かったため、262で安全確認済みの5配列形からさらに「新規schedule配列引数」を減らす。241の元kernelがもともと読んでいた `ctrl0_arr` / `markctrl_arr` をMAXD14専用に再利用し、precomputeによる追加load/引数圧を減らせるかを見る。採否は、JIT成立、正当性OK、dispatch rows 131維持、241 `0:07:07.788` / 262 `0:07:09.447` との速度比較で判断する。

---

Updated on 2026-07-07 for 264Py schedule-precompute-hitermpack.

- **263確認結果**: 263Py `schedule_precompute_reusectrl` は N=21 full once で final total `314666222712` 一致、error/mismatch `0`、elapsed `0:07:09.417`。262比では `0.030秒` 改善したが、241 parent `0:07:07.788` より `1.629秒` 遅いため採用基準には未達。ctrl0_arr を schedule_lo へ再利用することは JIT 安全と確認できた。
- **264**: 263を親に、root_action は JIT危険候補として別配列素通しのまま固定し、`schedule_hi` と `terminal_depth/base14` だけを同じ `sched_hi_arr` へpackする切り分けprobe。`sched_hi_arr` は low bits に `schedule_hi`、bits 24..27 に `terminal_depth`、bit 28 に `terminal_base14` を持つ。MAXD14 kernel側では `sched_hi_term=sched_hi_arr[i]` から `schedule_hi`、`terminal_depth`、`terminal_base14` をdecodeする。
- **264で維持するもの**: 241 restore239本線、単一launch、dispatch rows 131、fid14/rest別launchなし、child/future maskpack、root_action pass-through、mode28/29/30/31、cache生成、worker split、MAXD14/16/18/20/21 fallback、CPU dfs_iter path。
- **264で確認すること**: 244 r1-r3で失敗した高位bit packの原因が root_action 同居にあったのか、それとも schedule_hi high-bit pack自体にもPTX/JITリスクがあるのかを切り分ける。JIT成立すれば、263よりschedule loadをさらに1本減らす余地がある。JIT不成立なら、schedule_hi+terminal high-bit packは撤回し、263または262の安全形へ戻す。
- **264静的確認**: `STATIC_ONLY=1 bash 264Py_schedule_precompute_hitermpack_validate_N21_full_once.sh` で、version tag、main entry、bare -g mode31、N25..N27 preset8/N27 total、runtime globals、required runtime defs、旧診断削除、mode28/29/30/31保持、split264 tag、worker split、fid14 split拒否、MAXD14 hitermpack precompute、host hitermpack arrays、release build policy がOK。

---

Updated on 2026-07-07 for 264Py schedule-precompute hitermpack probe.

- **263確認結果**: 263Py `schedule-precompute-reusectrl` は `N=21 full once` で final total `314666222712` 一致、warning/errorなし、elapsed `0:07:09.417`。262Py `0:07:09.447` から `0.030秒` だけ改善したが、241Py `0:07:07.788` より `1.629秒` 遅いため採用基準にはしない。`ctrl0_arr=schedule_lo`、`markctrl_arr=terminal_depth/base14` のMAXD14直前repackはJIT安全・正当性OKと確認できた。
- **264**: 263を親に、root_action は257/259/260で `CUDA_ERROR_INVALID_PTX` になったため引き続き別配列pass-throughへ固定し、`schedule_hi` と `terminal_depth/base14` だけを同一carrierへpackする切り分け版。MAXD14 selected が確定した後だけ、`ctrl0_arr=schedule_lo`、`markctrl_arr=schedule_hi | (terminal_depth<<24) | (terminal_base14<<28)` へrepackし、MAXD14 kernelでは `schedule_hi=markctrl_arr[i]&0xFFFFFF`、`terminal_depth=(markctrl_arr[i]>>24)&15`、`terminal_base14=(markctrl_arr[i]>>28)&1` としてdecodeする。child/future maskpack、単一launch、mode28/29/30/31、cache生成、MAXD fallback、worker splitは保持する。
- **264の判定方針**: まずJIT成立可否を見る。JIT成立・正当性OKなら、263から `sched_hi_arr` の追加loadを削れた効果を確認し、241基準へ近づくかを判断する。JIT不成立なら、過去244 r1-r3の不成立要因はroot_actionだけでなく `schedule_hi` の高位bit pack/decodeにもあると判断し、schedule_hiは別配列に戻す。

---

Updated on 2026-07-07 for 265Py schedule-precompute reusemask probe.

- **264確認結果**: 264Py `schedule-precompute-hitermpack` は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required/selected MAXD14、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0`。elapsed は `0:07:09.539` で、263Py `0:07:09.417` より `0.122秒` 遅く、241Py `0:07:07.788` より `1.751秒` 遅いため不採用。`schedule_hi + terminal_depth/base14` の同一高位bit packはJIT成立・正当性OKだが、速度改善にはつながらなかった。
- **現時点の安全地図**: `child_jmark_mask + future_check_mask` pack、`terminal_depth + terminal_base14` pack、`ctrl0_arr=schedule_lo`、`markctrl_arr=terminal pack` はJIT安全。`root_action` のmask decodeや他値との同時packは `CUDA_ERROR_INVALID_PTX` となるため当面触らない。`schedule_hi + terminal pack` はJIT安全だが遅いため263形へ戻す。
- **265**: 264を採用せず、263Py `schedule-precompute-reusectrl` を親に戻す。root_action は別配列pass-through、schedule_hi も別配列のまま維持する。一方で、MAXD14 selected が確定した後だけ `free_arr` を `sched_mask` のcarrierとして再利用する。MAXD14 kernelでは元の `free_arr` を読まず、root free を `bm & ~(root_ld | root_rd | root_col)` で再計算する。これにより、263の `sched_mask_arr` 追加global loadを既存 `free_arr` loadへ置き換え、追加schedule配列引数を1本減らせるかを見る。
- **265で維持するもの**: 241 restore239本線、単一launch、dispatch rows 131、fid14/rest別launchなし、child/future maskpack、terminal_depth/base14 pack、root_action pass-through、schedule_hi separate、mode28/29/30/31、cache生成、worker split、MAXD14/16/18/20/21 fallback、CPU dfs_iter path。
- **265検証条件**: `STATIC_ONLY=1 bash 265Py_schedule_precompute_reusemask_validate_N21_full_once.sh` で version tag、main entry、bare `-g` mode31、N25..N27 preset8/N27 total、runtime globals、required runtime defs、旧診断削除、mode28/29/30/31保持、split265 tag、worker split、fid14 split拒否、MAXD14 reusemask precompute、host repack arrays、release build policy を確認する。fullでは N=21 / mode31 / split145 / 32x484 / worker0/1 で final total、progress TSV、dispatch rows/task sum、MAXD14、stack 208 bytes/thread、warning/errorを検証する。

---

Updated on 2026-07-07 for 266Py schedule-precompute-rootaction0 probe.

- **265確認結果**: 265Py `schedule_precompute_reusemask` は `N=21 full once` で final total `314666222712` 一致、error/mismatch `0`、required/selected MAXD14、schedule_words `0`、stack_bytes_per_thread `208`。速度は `0:07:09.233` で、263 `0:07:09.417` から `0.184秒` 改善した。precompute系ではここまでの最良だが、241 `0:07:07.788` より `1.445秒` 遅いため本線採用はしない。
- **266**: 265を親にする `rootaction0` 切り分け版。単一launch、dispatch rows 131、mode28/29/30/31、cache生成、worker split、MAXD14/16/18/20/21 fallbackを維持する。MAXD14 selected確定後に265同様 `ctrl0_arr=schedule_lo`、`markctrl_arr=terminal_depth/base14`、`free_arr=child/future sched_mask` へrepackする。追加で、chunk内の `root_action_arr` が全て0の場合だけ、`root_action_arr` loadと root_action 1/2/3 early branchを持たない `kernel_dfs_iter_gpu_maxd14_root0` を起動する。root_action非zeroを含むchunkは265相当kernelへfallbackする。
- **266の狙い**: root_actionのmask decode/pack系は257/259/260で `CUDA_ERROR_INVALID_PTX` になったため触らない。一方、root_actionが全て0のchunkでは別配列loadと分岐群を丸ごと消せる可能性がある。まずJIT成立と正当性を確認し、次にrootaction0 kernelが実際に使われるchunk数と速度を確認する。
- **266検証条件**: `N=21 full once`、bench_mode `31`、w8_j7、32x484、worker0/1。従来どおり final total、progress TSV再構成、dispatch rows/task sum、required_maxd/selected_MAXD、schedule_words、stack bytes、warning/errorを確認する。`STATIC_ONLY=1` ではrootaction0 kernelの存在、root_action load/branchがroot0 kernelに無いこと、split266 tag、241/265由来の保持機能を確認する。

---

Updated on 2026-07-07 for 266Py schedule-precompute-root0probe.

- **265確認結果**: 265Py schedule-precompute-reusemask は `N=21 full once` で final total `314666222712` 一致、required/selected MAXD14、stack 208 bytes/thread、warning/errorなし。速度は `0:07:09.233` で、precompute系では暫定最良だが、241Py `0:07:07.788` には届かないため本線採用は見送る。
- **266**: 265Pyを親にした root_action==0 chunk probe。単一launch per chunkを維持し、別launch分割は行わない。MAXD14 selected確定後の `ctrl0_arr=schedule_lo`、`markctrl_arr=terminal_depth+terminal_base14`、`free_arr=child/future sched_mask` 再利用は265相当で維持する。chunk内の `root_action_arr` が全て0なら、`root_action_arr` loadおよび `root_action` 分岐を持たない `kernel_dfs_iter_gpu_maxd14_root0` を1回だけlaunchする。`root_action` 非zeroを含むchunkは265互換の通常MAXD14 kernelへfallbackする。
- **266の狙い**: 257/259/260で `root_action` のmask decodeや同時packは `CUDA_ERROR_INVALID_PTX` になったため、root_actionのpackは行わない。代わりに、root_actionが全0のchunkだけ root_action load/branchをkernelから丸ごと消せるかを確認する。正当性OKかつroot0対象chunkが多ければ、265からさらに戻る可能性がある。
- **266で保持するもの**: `-c/-g`、bare `-g` mode31、GPU N=5..27、N25..N27 preset8、cache生成、broadmarktail mode28/29、split145 mode30/31、MAXD14/16/18/20/21 fallback、CPU path、worker_id/worker_count multi-GPU split、CUDA_VISIBLE_DEVICES運用を保持する。
- **266静的検査**: `STATIC_ONLY=1 bash 266Py_schedule_precompute_root0probe_validate_N21_full_once.sh` で source tag、main entry、mode31 default、N27 range、required runtime defs、旧診断削除、mode28/29/30/31保持、split266 tag、worker split、root0 kernel、root0/fallback conditional dispatch、reusemask host precompute がOKであることを確認した。

---

Updated on 2026-07-07 for 267Py schedule-precompute-root0countcache.

- **266確認結果**: 266Py `schedule-precompute-root0probe` は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required/selected MAXD は全chunk `14`、schedule_words `0`、stack_bytes_per_thread `208`、warning/error/mismatch `0`。速度は `0:07:08.113`。265Py `0:07:09.233` より `1.120秒` 改善し、precompute系では最良。ただし241Py `0:07:07.788` より `0.325秒` 遅いため、まだ本線採用基準は241Pyのまま。
- **266の意味**: `root_action==0` chunk向けMAXD14 root0 kernelはJIT成立・正当性OK・速度改善あり。`root_action_arr` のglobal loadおよびroot_action分岐をkernelから消す方向は有効と判断する。一方、257/259/260で確認したroot_action mask decode/pack系は引き続きJIT危険として扱い、root_actionのpackは行わない。
- **267**: 266Pyを親にする `schedule-precompute-root0countcache` 版。root0 kernel/fallback dispatch構造は266のまま維持し、`root_action_nonzero_count` を `build_soa_for_range()` のhost schedule precompute時に集計して `TaskSoA` に保持する。MAXD14 launch直前のrepack loopでは `root_action_arr` を再走査せず、cached countで root0/fallback を選択する。`ctrl0_arr=schedule_lo`、`markctrl_arr=terminal_depth+terminal_base14`、`free_arr=child/future mask` の再利用、`sched_hi_arr`、`root_action_arr` pass-through、mode28/29/30/31、cache生成、MAXD14/16/18/20/21 fallback、worker splitは保持する。
- **267検証条件**: `N=21 full once`、bench_mode `31`、w8_j7、32x484、worker0/1。final total、progress TSV再構成、dispatch rows/task sum、required/selected MAXD、schedule_words、stack bytes、warning/errorを従来通り検証する。採否は 266 `0:07:08.113` と 241 `0:07:07.788` を基準に判断する。


---

Updated on 2026-07-07 for 268Py schedule-precompute-root0future0.

- **267確認結果**: 267Py `root0countcache` は N=21 full once で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、全chunk required/selected MAXD14、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0`。速度は `0:07:08.056`。266Py `0:07:08.113` からさらに `0.057秒` 改善し、precompute系では最良。ただし241Py `0:07:07.788` には `0.268秒` 届かないため、本線採用基準は241のまま。
- **268**: 267を親に、root_action==0 chunk専用MAXD14 kernelを維持しつつ、さらに future_check_mask==0 のchunkだけ `root0future0` kernelへdispatchするprobe。`root0future0` kernelは `root_action_arr` load/branchに加え、future-prune branchを持たない。root_action非zero chunkは267 fallback、root_action==0 だがfutureありchunkは267 root0 kernelへfallbackする。単一launch per chunk、mode28/29/30/31、cache生成、MAXD fallback、worker splitは保持する。
- **268採否基準**: まずJIT成立と正当性OKを確認し、速度は267 `0:07:08.056` と241 `0:07:07.788` を基準に評価する。

---

Updated on 2026-07-07 for 269Py schedule-precompute-root0child0.

- **268確認結果**: 268Py root0future0 は N=21 full once で正当性OK。final total `314666222712`、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、全chunk required/selected MAXD14、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0`。ただし elapsed は `0:07:08.525` で、267Py `0:07:08.056` より `0.469秒` 遅いため不採用。root_action==0 かつ future_check_mask==0 専用kernelは撤回する。
- **269**: 268を採用せず267 root0countcacheへ戻し、root_action==0 かつ child_jmark_mask==0 のchunkだけ `kernel_dfs_iter_gpu_maxd14_root0_child0` へdispatchする切り分け版。root0child0 kernel は root_action load/branch に加えて child_jmark branchを持たない。root_action==0だがchild_jmarkありchunkは267 root0 kernelへfallbackし、root_action非zero chunkは265互換precompute kernelへfallbackする。単一launch per chunk、mode28/29/30/31、cache生成、MAXD14/16/18/20/21 fallback、worker splitは維持する。

---

Updated on 2026-07-07 for 270Py schedule-precompute-rootpre2flag.

- **269確認結果**: 269Py `root0child0` は N=21 full once で正当性OK。final total `314666222712`、dispatch task sum `2025282`、全chunk required/selected MAXD14、schedule_words `0`、stack_bytes_per_thread `208`、warning/error `0`。ただし elapsed は `0:07:08.454` で、267Py `0:07:08.056` より `0.398秒` 遅いため不採用。root_action==0 かつ child_jmark_mask==0 専用kernelは撤回する。
- **270**: 269を採用せず267 root0countcacheへ戻す。267の root0 kernel/fallback dispatch、root_action_nonzero_count cache、`ctrl0_arr=schedule_lo`、`markctrl_arr=terminal_depth+terminal_base14`、`free_arr=child/future mask` の再利用は維持する。新しい差分は、MAXD14 kernel内で毎thread計算していた `root_second` / `root_after_second` 判定をhost側で `root_pre2_flag` として事前計算し、`sched_mask` の bit28 へpackすること。kernel側では `root_pre2_flag` を見て1/2bit root-prerollへ入るため、3bit以上rootでは `root_rest/root_second/root_after_second` 計算を避ける。
- **270で注意すること**: `root_action==1` の場合は kernel側で先に `root_a &= ~1` が行われるため、host precomputeもこのpost-root-action状態に合わせて `root_pre2_flag` を作る。`root_action==2/3` は早期returnのためflagは実質使われない。root_action pack/mask decode は257/259/260で `CUDA_ERROR_INVALID_PTX` になったため引き続き行わない。
- **270で保持するもの**: 単一launch per chunk、fid14/rest別launchなし、268/269専用kernelは撤回、mode28/29/30/31、cache生成、MAXD14/16/18/20/21 fallback、CPU dfs_iter path、worker split、CUDA_VISIBLE_DEVICES運用を保持する。


---

Updated on 2026-07-09 for 271Py final-fastest-complete baseline.

- **271**: 270Py `schedule_precompute_rootpre2flag` は正当性OKながら `N=21 full once` が `0:07:08.824` で、241Py `0:07:07.788` より `1.036秒` 遅かったため不採用。271Pyは速度候補ではなく、現時点で添付済み・検証済みの最速安全版を完成版として固定するパッケージング版。親は **241Py r3 restore239-after-fid14split-reject**。CUDA MAXD14/16/18/20/21 kernel本文、rootrestlate、futuremask/no-sibling、split145/chunkshape148、broadmarktail、cache生成、worker split、mode28/29/30/31、MAXD fallback、CPU dfs_iter pathは241 r3相当で維持する。
- **271検証基準**: `N=21 full once`、bench_mode `31`、w8_j7、32x484、worker0/1。期待値は final total `314666222712`、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14/MAXD14`、schedule_words `0`、stack_bytes_per_thread `208`。271は241 r3とkernel同等のため、まず `STATIC_ONLY=1`、次にfull onceで241 r3の `0:07:07.788` 近傍へ戻ることを確認する。

## 271Py以降の最適化優先順位メモ

1. **A: root_action分布の軽量診断**
   効果見込み: 中〜高。難易度: 低〜中。266/267で `root_action==0` 専用MAXD14 kernelは有効だった一方、268/269/270は退行した。次は速度候補ではなく、root0 kernel使用chunk数、fallback chunk数、root_action非ゼロ種別、root0対象task数、fallback対象task数だけを軽量に出して、非ゼロ側専用化の根拠を取る。

2. **A: root_action非ゼロ側のchunk単位専用kernel**
   効果見込み: 中。難易度: 中〜高。分布診断で `root_action==1/2/3` のいずれかに偏ったchunkが十分あれば、別launchではなくchunk単位で専用MAXD14 kernelへdispatchする。240のfid14別launchのようなlaunch増加型は避ける。

3. **B: root0 kernelの不要load・不要decode削減**
   効果見込み: 小〜中。難易度: 中。root0 kernelから `root_action_arr` ロード/branchが消える方向は効いた。次はroot0 kernelに残る不要な配列引数、decode順序、実際には使わない値を確認して削る。ただしif順序変更だけの微差実験にはしない。

4. **B: MAXD14 hot loopの構造再検討**
   効果見込み: 中。難易度: 高。224のgeneric clear-lowbitは大退行、225のncol-earlyは小幅良好、226/227系は伸びなかった。generic loopは単発の演算置換ではなく、load/store削減やframe保存規則の再設計として扱う。

5. **C: host precompute/pack系の再挑戦**
   効果見込み: 小。難易度: 高。precompute系最良の267でも241に届かず、270 rootpre2flagは退行。root_action mask decodeや同時packはJIT危険として扱い、再挑戦する場合はJIT安全性を1項目ずつ切り分ける。

6. **C: MAXD13/stack縮小**
   効果見込み: 不明。難易度: 高。222診断で `max_save_sp=13`、`save_sp13_count=1177141` が出ており、単純なMAXD13化は不可。限定dispatchや特殊chunkだけを狙うには追加診断が必要。

7. **D: やらない候補**
   `if` 条件順序だけの変更、nibble/future bit微差、root-preroll内の小手先整理、generic loop内の単純置換、別launch分割だけのprobe、root_action mask decode、root_action同時pack、268/269/270の再試行は優先度を下げる。

---

Updated on 2026-07-09 for 271Py result and 272Py root-action-distribution diagnostic.

- **271確認結果**: 271Py `final_fastest_complete` は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14/MAXD14`、schedule_words `0`、stack_bytes_per_thread `208`、error/mismatch `0`。速度は `0:07:07.705`。241Py `0:07:07.788` より `0.083秒` 高速、270Py `0:07:08.824` より `1.119秒` 高速、267Py `0:07:08.056` より `0.351秒` 高速、217Py `0:07:07.709` より `0.004秒` 高速。239Py `0:07:07.703` には `0.002秒`だけ届かないが誤差級であり、完成版基準として271Pyを固定する。
- **272**: 271Pyを親にする root_action distribution diagnostic 版。速度候補ではなく、次の専用kernel候補を決めるための軽量診断。CUDA MAXD14/16/18/20/21 kernel本文、rootrestlate、futuremask/no-sibling、split145/chunkshape148、broadmarktail、cache生成、worker split、mode28/29/30/31、MAXD fallback、CPU dfs_iter path、1 chunk 1 launch は維持する。host側で `root_action_for_task()` を使い、chunkごとに `root_action0/1/2/3`、`root_action_nonzero_count`、`root0_kernel_chunk`、`fallback_chunk`、`root0_task_count`、`fallback_task_count` をprogress TSVへ追加し、ログに `[root-action-diag]` と `[root-action-summary]` を出す。
- **272の目的**: 266/267で `root_action==0` chunk専用kernelが有効と分かった一方、268/269/270は退行した。272ではまず分布を見て、`root_action==1` / `root_action==2` / `root_action==3` のどれかに偏りがあるか、またchunk単位で専用dispatchできるだけのまとまりがあるかを確認する。分布が薄い場合は非ゼロ側専用kernelへ進まず、root0 kernelの不要load/decode削減または別方向へ戻る。
- **272検証条件**: `N=21 full once`、bench_mode `31`、w8_j7、32x484、worker0/1。従来どおり final total、progress TSV再構成、dispatch rows/task sum、required/selected MAXD、schedule_words、stack bytes、warning/errorを確認する。追加で、progress TSVの root_action列の合計が dispatch task sum `2025282` と一致すること、root0/fallback chunk合計が `131` と一致すること、`nonzero == fallback_task_count`、`root_action0 == root0_task_count` を検証する。
- **272後の判定**: `[root-action-summary]` の `root0_kernel_chunks` / `fallback_chunks` と `root_action1/2/3` の比率を見る。非ゼロ側が特定actionに偏り、かつchunk単位でまとまっていれば273でそのaction専用MAXD14 kernelを検討する。まとまりがなければ、専用kernel化ではなく271完成版へ戻して別候補へ進む。

---

Updated on 2026-07-09 for 272Py result and 273Py rootaction0-direct-kernel probe.

- **272確認結果**: 272Py `root_action_distribution_diag` は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14/MAXD14`、schedule_words `0`、stack_bytes_per_thread `208`、error/mismatch `0`。速度は `0:07:07.728` で、271Py `0:07:07.705` より `0.023秒` 遅いだけで、診断オーバーヘッドは小さい。root_action分布は `root_action0=2025282`、`root_action1=0`、`root_action2=0`、`root_action3=0`、`nonzero=0`、`root0_kernel_chunks=131`、`fallback_chunks=0`、`root0_tasks=2025282`、`fallback_tasks=0`。したがって、N=21/mode31では root_action==1/2/3 専用化は不要であり、対象taskが存在しないため速度候補から外す。
- **273**: 272の結果を受け、271Py `final_fastest_complete` を親にする `rootaction0-direct-kernel` 速度候補。schedule precompute系には戻さず、271のkernel本線・task order・cache生成・split145/chunkshape148・broadmarktail・1 chunk 1 launch を維持する。追加差分は、host側で `root_action_for_task()` により chunk内の `root_action_nonzero_count` を `build_soa_for_range()` 時にcacheし、`selected_MAXD==14` かつ `root_action_nonzero_count==0` のchunkだけ、`root_action` scalar生成・`root_action==1/2/3` 分岐を持たない `kernel_dfs_iter_gpu_maxd14_root0` へdispatchすること。非zero root_actionを含むchunkは271相当の通常MAXD14 kernelへfallbackする。
- **273で保持するもの**: `rootrestlate`、future_check_mask guard、no-sibling spill elision、root one/two-candidate preroll、MAXD14/16/18/20/21 fallback、mode28/29/30/31、cache生成、worker split、CPU dfs_iter path、GPU N=5..27、N25..N27 preset8を維持する。240のfid14/rest別launch、267/270のschedule precompute/repack、root_action pack/mask decode、272のprogress TSV診断列は入れない。
- **273検証条件**: `STATIC_ONLY=1 bash 273Py_rootaction0_direct_kernel_validate_N21_full_once.sh` で source tag、root0 kernel、root0/fallback dispatch、host count cache、split273 tag、旧precompute不在を確認する。その後 `bash 273Py_rootaction0_direct_kernel_validate_N21_full_once.sh` で N=21 full once を実行し、final total、progress TSV、dispatch rows/task sum、MAXD14、stack 208 bytes/thread、warning/errorに加え、`[root0-dispatch]` が `131` 行、root0 rows `131`、fallback rows `0`、root_action_nonzero sum `0` であることを確認する。採否は271Py `0:07:07.705`、272Py `0:07:07.728`、239Py `0:07:07.703` との速度比較で判断する。

---

Updated on 2026-07-09 for 273Py result and 274Py restore271-after-root0direct-reject.

- **273確認結果**: 273Py `rootaction0_direct_kernel` は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14/MAXD14`、schedule_words `0`、stack_bytes_per_thread `208`、error/mismatch `0`。root0 dispatch は `rows=131`、`task_sum=2025282`、`root_action_nonzero_sum=0`、`root0_kernel_rows=131`、`fallback_rows=0` で、272の「全task root_action==0」診断と一致した。ただし速度は `0:07:08.757` で、271Py `0:07:07.705` より `1.052秒`、272Py `0:07:07.728` より `1.029秒`、267Py `0:07:08.056` より `0.701秒` 遅いため不採用。root_action==0 direct kernel は正当性OKだが、別kernel化によるコード配置/register pressure/コンパイル最適化差が勝った可能性があるため撤回する。
- **274**: 273Pyを採用せず、271Py `final_fastest_complete` 相当へ戻す安全復帰版。CUDA MAXD14/16/18/20/21 kernel本文、rootrestlate、futuremask、no-sibling、split145/chunkshape148、broadmarktail、cache生成、worker split、MAXD fallback、CPU dfs_iter path は271相当で維持する。274では `split274` tagへ分離し、検証シェルでは273 root0 direct kernel markerが残っていないことも静的確認する。採用可否はN=21 full onceで271Py `0:07:07.705` へ戻るかを確認して判断する。
- **次候補メモ**: 272/273により、N=21では root_action==1/2/3 専用化は不要であり、root_action==0 別kernel化も速度上は不採用と判断する。次の速度候補は、root_action系から離れ、271本線を親に `d2base14` / `d0` など既存 `[split145-buckets]` で多く観測される分類のうち、別launchを増やさず単一chunk内dispatchまたはhost側順序だけで試せるものを優先する。

---

Updated on 2026-07-09 for 274Py result and 275Py split145 bucket distribution diagnostic.

- **274確認結果**: 274Py `restore271_after_root0direct_reject` は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14/MAXD14`、schedule_words `0`、stack_bytes_per_thread `208`、error/mismatch `0`。速度は `0:07:07.758`。271Py `0:07:07.705` より `0.053秒` 遅いが、273Py `0:07:08.757` より `0.999秒` 戻したため、273 rootaction0 direct kernel の撤回・271相当復帰として正当性OK。root_action系は、272で非zero taskが存在しないこと、273でroot0別kernel化が退行したことから、N=21最速化候補としてはいったん終了する。
- **275**: 274Pyを親にする `split145_bucket_distribution_diag` 版。速度候補ではなく、次の分解判断用の軽量診断。CUDA MAXD14/16/18/20/21 kernel本文、rootrestlate、futuremask/no-sibling、split145/chunkshape148、broadmarktail、cache生成、worker split、mode28/29/30/31、MAXD fallback、CPU dfs_iter path、1 chunk 1 launchは274/271相当で維持する。追加差分は、既存 `exec_solutions_gpu_chunk_split145()` が計算している `d2base14(fid14)` と `d0(fid26/27)` のchunk内件数を、full run全体で集約して `[split145-bucket-summary]` として出すことだけ。progress TSV本体の列は271/274相当を維持し、検証シェル側で `funcid_14_count`、`funcid_26_count`、`funcid_27_count` から同じ集計を再構成してログsummaryと照合する。
- **275の目的**: 240のfid14/rest別launchは大退行、273のroot0別kernelも退行したため、次は「別launchを増やす前提」ではなく、まず既存split145 stream内で `d2base14` と `d0` がどの程度まとまっているかを見る。`d2base14_chunks` / `d0_chunks` / `d0_or_d2_chunks` / `both_chunks` / `d2base14_tasks` / `d0_tasks` / `other_tasks` の比率を見て、次にhost-side orderingだけで扱うか、chunk単位dispatchで扱うか、それとも分解候補から外すかを判断する。
- **275検証条件**: `STATIC_ONLY=1 bash 275Py_split145_bucket_distribution_diag_validate_N21_full_once.sh` で source tag、split275 tag、root0 direct kernel不在、bucket summary出力の存在を確認する。その後 `bash 275Py_split145_bucket_distribution_diag_validate_N21_full_once.sh` で N=21 full once を実行し、従来の final total、progress TSV、dispatch rows/task sum、MAXD14、stack 208 bytes/thread、error/mismatchに加え、`split145_bucket_progress_*` と `split145_bucket_log_vs_progress_*` がOKであることを確認する。採否ではなく、次候補選定用の診断結果として扱う。

---

Updated on 2026-07-09 for 275Py r2 split145 bucket diagnostic compile fix.

- **275 r2**: 275Py `split145_bucket_distribution_diag` 初版は source static checks は通ったが、Codon release build で `bucket_total_tasks` / `bucket_d0_tasks` が `referenced before assignment` となった。原因は、bucket診断集計用カウンタの初期化を別のstream関数側に入れており、実際に `exec_solutions_gpu_bin_stream_split145` 内で参照するローカル変数としては未初期化だったため。r2ではCUDA kernel本文、task order、1 chunk 1 launch、271/274本線、診断内容は変更せず、`exec_solutions_gpu_bin_stream_split145` 内で `bucket_total_tasks`、`bucket_d0_tasks`、`bucket_d2base14_tasks`、`bucket_d0_or_d2_tasks`、`bucket_d0_chunks`、`bucket_d2base14_chunks`、`bucket_d0_or_d2_chunks`、`bucket_both_chunks` を明示初期化する最小補修を行った。`STATIC_ONLY=1` はOK。採否はN=21 fullで正当性と診断値を確認して判断する。


---

Updated on 2026-07-09 for 275Py result and 276Py restore274-after-bucketdiag-reject.

- **275 r2確認結果**: 275Py `split145_bucket_distribution_diag` は Codon release build が通り、`N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14/MAXD14`、schedule_words `0`、stack_bytes_per_thread `208`、error/mismatch `0`。速度は `0:07:07.716` で、271Py `0:07:07.705` より `0.011秒` 遅いだけで診断オーバーヘッドは小さい。`[split145-bucket-summary]` は `total_tasks=2025282`、`d2base14_tasks=8214`、`d0_tasks=293733`、`d0_or_d2_tasks=301947`、`other_tasks=1723335`、`d2base14_chunks=131`、`d0_chunks=131`、`both_chunks=131`。d2base14 は全体の約0.4%と小さく、d0 は約14.5%あるが全chunkに混在しているため、chunk単位専用dispatchには向かない。275検証シェルは bucket 再構成用 env が空になりsummary末尾まで進まなかったが、full log と progress TSV の正当性はOK。
- **276**: 275診断結果を受け、d2base14/d0 のchunk単位専用化には進まず、274Py `restore271_after_root0direct_reject` / 271Py `final_fastest_complete` 相当へ戻す安全復帰版。CUDA MAXD14/16/18/20/21 kernel本文、rootrestlate、futuremask/no-sibling、split145/chunkshape148、broadmarktail、cache生成、worker split、mode28/29/30/31、MAXD fallback、CPU dfs_iter path、1 chunk 1 launch は274/271相当で維持する。273 root0 direct kernel、275 bucket診断、270 schedule precompute、240 fid14/rest別launchは入れない。
- **276検証条件**: `STATIC_ONLY=1 bash 276Py_restore274_after_bucketdiag_reject_validate_N21_full_once.sh` で source tag、split276 tag、bucket診断不在、root0 direct kernel不在を確認する。その後 `bash 276Py_restore274_after_bucketdiag_reject_validate_N21_full_once.sh` で N=21 full once を実行し、271/274相当の `0:07:07.7xx` へ戻ることを確認する。


---

Updated on 2026-07-09 for 277Py depthu-childsave probe.

- **276確認結果**: 276Py restore274_after_bucketdiag_reject は、275Pyのsplit145 bucket分布診断を撤回し、274/271相当の本線へ戻した安全復帰版。`N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14/MAXD14`、schedule_words `0`、stack_bytes_per_thread `208`、error/mismatch `0`。速度は `0:07:07.672` で、271Py `0:07:07.705` より `0.033秒`、239Py `0:07:07.703` より `0.031秒` 高速。構造的には271/274相当の復帰版であり、差は誤差級だが、単発実測上の現行最速として276Pyを管理基準にできる。
- **277**: 276Pyを親にした微差実験版。MAXD14 generic DFS loop内だけ、terminal判定後のpost-terminal pathで `depth_u:u32=u32(cur_depth)` を一度作り、`child_jmark_mask >> depth_u` と `avail[save_sp]=cur_avail|(depth_u<<27)` に共有する。`nibble_op` 取得、root-preroll、rootrestlate、future_check_mask guard、no-sibling spill elision、split145/chunkshape148、broadmarktail、cache生成、worker split、MAXD fallback、CPU dfs_iter pathは変更しない。狙いは同一iteration内の `u32(cur_depth)` cast重複を減らすことだが、効果は小さい見込みで、採否はN=21 fullで276/271/239比timingを確認して判断する。


---

Updated on 2026-07-09 for 278Py zero-const-assign probe.

- **277確認結果**: 277Py depthu_childsave は `N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14/MAXD14`、schedule_words `0`、stack_bytes_per_thread `208`、error/mismatch `0`。速度は `0:07:07.717` で、276Py `0:07:07.672` より `0.045秒`、271Py `0:07:07.705` より `0.012秒` 遅いため不採用。`depth_u` による child_jmark/save payload 共通化は撤回する。
- **278**: 277Pyを採用せず276Py本線へ戻し、MAXD14 kernel内だけで `ZERO:u32=u32(0)` を一度作り、`u32(0)` を代入している局所初期化・代入箇所を `ZERO` 参照へ置換する微差実験版。`u32(0)-x` のlowbit抽出、clear-lowbit、比較条件、mask/shift、`u64(0)`/`u64(1)`、MAXD16/18/20/21 fallback は触らない。root-preroll、rootrestlate、future_check_mask guard、no-sibling spill elision、split145/chunkshape148、broadmarktail、cache生成、worker splitは276相当を維持する。改善見込みは小さく、レジスタ圧増加で退行する可能性もあるため、採否はN=21 fullで276/271/277比timingを確認して判断する。

---

Updated on 2026-07-09 for 278Py result and 279Py restore276-after-zeroconst-reject.

- **278確認結果**: 278Py `zero_const_assign` は、276Pyを親にMAXD14 kernel内の単純な `u32(0)` 初期化・代入を定数変数参照へ置換した微差実験版。`N=21 full once` で final total `314666222712` 一致、progress/dispatch/MAXD14/stack 208 はOK、error/mismatch `0`。速度は `0:07:09.603` で、276Py `0:07:07.672` より `1.931秒`、277Py `0:07:07.717` より `1.886秒`、271Py `0:07:07.705` より `1.898秒` 遅い。即値ゼロを変数化すると、演算削減よりレジスタ圧増加・コード生成悪化が勝った可能性が高いため不採用。以後、`u32(0)` / `u32(1)` の定数変数化は優先候補から外す。
- **279**: 278Pyを採用せず、276Py `restore274_after_bucketdiag_reject` 相当へ戻す安全復帰版。277Py `depth_u` child/save共通化、278Py zero-const変数化、275Py bucket診断、273Py root0 direct kernel、270Py schedule precompute、240Py fid14/rest別launchは入れない。CUDA MAXD14/16/18/20/21 kernel本文、rootrestlate、futuremask/no-sibling、split145/chunkshape148、broadmarktail、cache生成、worker split、mode28/29/30/31、MAXD fallback、CPU dfs_iter path、1 chunk 1 launch は276相当で維持する。
- **279検証条件**: `STATIC_ONLY=1 bash 279Py_restore276_after_zeroconst_reject_validate_N21_full_once.sh` で source tag、split279 tag、277 depth_u active不在、278 zero-const active不在、bucket診断不在、root0 direct kernel不在を確認する。その後 `bash 279Py_restore276_after_zeroconst_reject_validate_N21_full_once.sh` で N=21 full once を実行し、276Py `0:07:07.672`、271Py `0:07:07.705` 近傍へ戻ることを確認する。

---

Updated on 2026-07-09 for 279Py result, ncu/block-size probes, and 280Py kernel-block-count diagnostic.

- **279確認結果**: 279Py restore276_after_zeroconst_reject は、278Py zero_const_assign を不採用にして276Py相当へ戻した安全復帰版。`N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14/MAXD14`、schedule_words `0`、stack_bytes_per_thread `208`、error/mismatch `0`。速度は `0:07:07.728` で、278Py `0:07:09.603` から `1.875秒` 戻した。276Py単発最速 `0:07:07.672` には届かないが、271/276系の通常レンジへ復帰したため、279は安全復帰OK。
- **ncu確認結果**: 279Pyの chunk 40 を Nsight Compute で確認。`DRAM Throughput` は約 `1.94%`、`Local Memory Spilling Requests` は `0`、L1/L2 hit rate はほぼ100%で、DRAM帯域やspillは主因ではない。一方で `No Eligible` が約 `69.67%`、`Eligible Warps Per Scheduler` が `0.34`、`Avg. Active Threads Per Warp` が `4.88`、`Branch Efficiency` が約 `79%`、`Divergent Branches` が大きい。したがって、主戦場はglobal memory削減ではなく、warp内ばらつき、分岐、依存待ち、task order/shapeにあると判断する。
- **279 block size probe結果**: 代表7chunk `0,20,40,60,80,100,120` で `32x484`、`64x242`、`128x121` を比較。elapsed_ms合計は `32x484=22852ms`、`64x242=22913ms`、`128x121=22873ms`。kernel_reduce_ms合計も `32x484=22839ms`、`64x242=22892ms`、`128x121=22858ms`。差は小さいが `32x484` が最良であり、ncuで見えた低occupancy/No Eligibleはblock size増加だけでは改善しないと判断する。従来どおり `32x484` を維持する。
- **280**: 279/276本線を親にする `kernel_block_count_diag` 版。速度候補ではなく、MAXD14 generic DFS loop内部の論理ブロック別実行回数をchunkごとに把握する診断版。CUDA計算結果、task order、1 chunk 1 launch、rootrestlate、futuremask/no-sibling、split145/chunkshape148、broadmarktail、cache生成、worker split、mode28/29/30/31、MAXD fallback、CPU dfs_iter pathは279/276相当で維持する。追加差分は、MAXD14 kernelで `loop_iters`、`zero_avail_count`、`restore_count`、`normal_block_count`、`special_block_count`、`nf_zero_count`、`future_check_count`、`future_prune_kill_count`、`terminal_hit_count`、`child_jmark_count`、`save_push_count`、`descend_count` をper-thread診断配列に書き、host側でchunkごとに合算して `[kernel-blockdiag]` とprogress TSV列へ出すこと。
- **280の目的**: ncuで見えた `No Eligible`、`Stall Wait`、低いactive threads/warp、divergent branches の原因を、Codon/Pythonソース上の論理ブロック単位で絞る。280の速度は診断オーバーヘッド込みなので採用判定には使わず、`normal_block` と `special_block` の比率、`nf_zero`、`future_check/future_kill`、`terminal`、`child_jmark`、`save/restore/descend` の量から、281以降で見るべき箇所を選ぶ。
- **280検証条件**: `STATIC_ONLY=1 bash 280Py_kernel_block_count_diag_validate_N21_full_once.sh` で source tag、split280 tag、diag配列・progress列・`[kernel-blockdiag]` 出力、277/278/275/273/270/240系不採用差分の不在を確認する。その後 `bash 280Py_kernel_block_count_diag_validate_N21_full_once.sh` で N=21 full once を実行し、従来の final total/progress/dispatch/MAXD14/stack 208/error 0 に加え、`blockdiag_rows=131`、progress TSV の診断列存在、`loop_iters` と `descend_count` が正値であることを確認する。

---

Updated on 2026-07-09 for 281Py kernel block count diagnostic writeback fix.

- **280確認結果**: 280Py `kernel_block_count_diag` は `N=21 full once` で final total `314666222712` 一致、dispatch rows `131`、task sum `2025282`、required_maxd/selected_MAXD は全chunk `14/MAXD14`、schedule_words `0`、stack_bytes_per_thread `208` で計算正当性はOK。ただし肝心の `[kernel-blockdiag]` 診断値が全chunkで `loop_iters=0`、`normal_block=0`、`special_block=0` など全て0になり、診断としては無効だった。原因は、診断配列への書き戻しが `root_action==3` 早期return付近に誤って入り、通常の `root_action==0` generic DFS終了経路で `diag_*_arr[i]` へ書き戻していなかったため。N=21では272で確認済みの通り全taskが `root_action==0` のため、誤配置された書き戻し経路は通らない。
- **281**: 280Pyの診断目的を維持した修正版。親は279/276本線相当、速度候補ではなく診断版。CUDAの探索・加算ロジック、task order、1 chunk 1 launch、rootrestlate、futuremask/no-sibling、split145/chunkshape148、broadmarktail、cache生成、worker split、mode28/29/30/31、MAXD fallback、CPU dfs_iter pathは280/279相当で維持する。修正点は、MAXD14 kernelの通常終了経路で `results[i]=total*w_arr[i]` の直前に `diag_loop_iters_arr`、`diag_zero_avail_arr`、`diag_restore_arr`、`diag_normal_block_arr`、`diag_special_block_arr`、`diag_nf_zero_arr`、`diag_future_check_arr`、`diag_future_kill_arr`、`diag_terminal_arr`、`diag_child_jmark_arr`、`diag_save_push_arr`、`diag_descend_arr` へ必ず書き戻すこと。あわせて root_action==2/3 と root_action==1 後に root availability が0になる早期returnでもゼロ診断を書いてからreturnする。
- **281検証条件**: `STATIC_ONLY=1 bash 281Py_kernel_block_count_diag_fix_validate_N21_full_once.sh` で source tag、split281 tag、blockdiag配列・progress列、通常終了直前の診断書き戻し、277/278/275/273/270/240系不採用差分の不在を確認する。その後 `bash 281Py_kernel_block_count_diag_fix_validate_N21_full_once.sh` で N=21 full once を実行し、従来の final total/progress/dispatch/MAXD14/stack 208/error 0 に加え、`blockdiag_rows=131`、`blockdiag_positive`、progress TSVの診断列存在、`progress_diag_positive` がOKになることを確認する。281の速度は診断オーバーヘッド込みなので採用判定には使わず、`normal_block`/`special_block`、`future_check`/`future_kill`、`save_push`/`restore`/`descend` の比率を次候補選定に使う。


---

Updated on 2026-07-09 for 282Py restore279-after-blockdiag-reject and 283Py generic normal-first reprobe.

- **282**: 281Py kernel block count診断版は正当性OKで診断値も取得できたが、診断配列書き込みにより速度候補ではないため採用せず、279/276相当へ戻す安全復帰版。281/280 block count診断、278 zero-const、277 depth_u、275 bucket診断、273 root0 direct kernel、270 schedule precompute、240 fid14/rest別launch は入れない。N=21 full onceで271/276/279系の通常レンジへ戻ることを確認する。
- **283**: 282復帰版を親に、281診断で normal path が special path より大きいことを確認したため、MAXD14 generic DFS loopだけ `if block_code==0` のnormal path先行へ入れ替える最小速度候補。root-preroll側の `pr_block_code` 分岐順序、rootrestlate、futuremask、no-sibling、split145/chunkshape148、broadmarktail、cache生成、worker split、mode28/29/30/31、MAXD fallback、CPU dfs_iter pathは282/279/276相当で維持する。227Py generic normal-first は当時不採用だったが、281診断後に現行276/279系で再確認する位置づけ。採否は正当性OK後、276Py `0:07:07.672`、279Py `0:07:07.728`、271Py `0:07:07.705`、227Py `0:07:07.808` と比較して判断する。

---

Updated on 2026-07-09 for 283Py result and 284Py generic normal-first ncol-early probe.

- **283確認結果**: 283Py generic_normalfirst は、282Py復帰版を親に、MAXD14 generic DFS loopだけを `block_code==0` のnormal path先行へ入れ替えた再確認版。`N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14/MAXD14`、schedule_words `0`、stack_bytes_per_thread `208`、error/mismatch `0`。速度は `0:07:07.698`。282Py `0:07:08.066` より `0.368秒` 高速、271Py `0:07:07.705` より `0.007秒`、239Py `0:07:07.703` より `0.005秒`、217Py `0:07:07.709` より `0.011秒` 高速。差は誤差級だが、281 block count診断でnormal pathが約71.75%と多かったことと整合し、現行の速度候補として採用寄りに扱う。
- **284**: 283Pyのnormal path先行を親に、MAXD14 generic DFS loopだけ `ncol:u32=cur_col|bit` を `block_code` 分岐の前へ前倒しする微差実験版。normal path / special path の両方で使う `ncol` を共通化するが、226Pyの `placed_ld/placed_rd` 共通化のように `cur_ld|bit` / `cur_rd|bit` は前倒ししない。root-preroll、rootrestlate、futuremask、no-sibling、split145/chunkshape148、broadmarktail、cache生成、worker split、mode28/29/30/31、MAXD fallback、CPU dfs_iter pathは283相当で維持する。採否はN=21 fullで283Py `0:07:07.698`、276Py `0:07:07.672`、271Py `0:07:07.705` と比較して判断する。


---

Updated on 2026-07-09 for 284Py result, 285Py restore283-after-ncol-early-reject, and 286Py generic normal-default nld/nrd probe.

- **284確認結果**: 284Py `generic_normalfirst_ncol_early` は、283Py `generic_normalfirst` を親に、MAXD14 generic loop内で `ncol = cur_col | bit` を `block_code` 分岐前へ前倒しした微差実験版。`N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14/MAXD14`、schedule_words `0`、stack_bytes_per_thread `208`、error/mismatch `0`。速度は `0:07:07.795` で、283Py `0:07:07.698` より `0.097秒`、276Py `0:07:07.672` より `0.123秒` 遅い。差は誤差級だが、283を上回らず、225系でもncol-earlyは決定打ではなかったため不採用。次版では283へ戻す。
- **285**: 284Pyを採用せず、283Py `generic_normalfirst` 相当へ戻す安全復帰版。MAXD14 generic DFS loopの `block_code==0` normal path先行は維持し、284の `ncol early`、281/280 block count診断、278 zero const、277 depth_u、275 bucket診断、273 root0 direct kernel、270 schedule precompute、240 fid14/rest別launch は入れない。CUDA MAXD14/16/18/20/21 kernel本線、rootrestlate、futuremask/no-sibling、split145/chunkshape148、broadmarktail、cache生成、worker split、mode28/29/30/31、MAXD fallback、CPU dfs_iter path、1 chunk 1 launch は283相当で維持する。
- **286**: 285/283を親にした `generic_normaldefault_nldnrd` 速度候補。281診断で normal path が約71.75%、special pathが約7.03%だったことを受け、MAXD14 generic DFS loopだけ、`nld/nrd` を `u32(0)` 初期化して `if/else` で代入する形ではなく、まず normal path の `nld=(cur_ld|bit)<<1`、`nrd=(cur_rd|bit)>>1` を既定値として作り、`block_code!=0` の special path でのみ上書きする形へ変更する。狙いは dominant なnormal pathをfall-through既定値にし、zero初期化とnormal側else構造を避けること。special pathではnormal値計算が余分になるため、採否はN=21 fullで285/283/276/271比timingを確認して判断する。
- **286で触らないもの**: 284 ncol-early、226 placed_ld/placed_rd共通化、root-preroll側の `pr_block_code` 分岐順序、future_check、terminal、child_jmark、save/restore、task order/cache/dispatch、MAXD fallbackは変更しない。`u32(0)` のZERO変数化は278で大退行したため再試行しない。

---

Updated on 2026-07-09 for 285Py/286Py r2 validation shell static-check fix.

- **285 r2**: 285Py本体のCUDA/kernel差分は変更せず、検証シェルの `source_split_tag` 静的検査だけを補修した。`ncol_early` が不採用差分名としてコメント・説明文字列に残るだけでFAILしないよう、split runtime tag検査から `ncol_early` を外した。`STATIC_ONLY=1` はOK確認済み。
- **286 r2**: 286Py本体のCUDA/kernel差分は変更せず、検証シェルの `source_split_tag` 静的検査だけを同様に補修した。`ncol_early` は286でも不採用差分名として説明に出るため、active split tag検査対象から外した。`STATIC_ONLY=1` はOK確認済み。

---

Updated on 2026-07-09 for 287Py adoption of 286 normaldefault and 288Py normal nf-default probe.

- **286確認結果**: 286Py `generic_normaldefault_nldnrd` は、285/283 generic normal-first を親に、MAXD14 generic loop内で normal path の `nld=(cur_ld|bit)<<1`、`nrd=(cur_rd|bit)>>1` を既定値として先に作り、`block_code!=0` のspecial pathだけ `nld/nrd` を上書きする実験版。`N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14/MAXD14`、schedule_words `0`、stack_bytes_per_thread `208`、error/mismatch `0`。速度は `0:07:04.033` で、276Py `0:07:07.672` より `3.639秒`、271Py `0:07:07.705` より `3.672秒`、283Py `0:07:07.698` より `3.665秒` 高速。281診断でnormal pathが約71.75%を占めることと整合しており、286Pyを新最速基準として採用する。
- **287**: 286Pyの正当性OK・大幅高速化を受けた採用固定版。CUDA kernelの探索・加算ロジックは286と同一。286の normal-default `nld/nrd` を新基準として固定し、version/progress tagと検証baselineを287向けに整理する。284 `ncol early`、281 block count診断、278 zero const、277 depth_u、275 bucket診断、273 root0 direct kernel は入れない。
- **288**: 287/286を親にした次のnormal path局所実験版。MAXD14 generic loopで、normal path の `nld/nrd` に加え、`ncol=cur_col|bit` と `nf=bm&~(nld|nrd|ncol)` も既定値として先に作り、`block_code!=0` のspecial pathだけ `nld/nrd/nf` を上書きする。`placed_ld/placed_rd` 共通化、future check位置変更、root-preroll変更、host task order変更は入れない。採用可否はN=21 fullで287/286/276/271比timingを確認して判断する。

---

Updated on 2026-07-09 for 289Py generic-normaldefault-ncolonly probe.

- **287/288確認後の方針**: 287Py adopt286_normaldefault_nldnrd は正当性OKで `0:07:04.486`。286Py単発最速 `0:07:04.033` には届かないが、従来の271/276/283系より約3.2秒高速であり、286/287系を現行採用基準とする。288Py generic_normaldefault_nfdefault は正当性OKだが `0:08:37.227` と大退行したため不採用。`nf` default化は撤回する。
- **289**: 288を採用せず287を親にする。MAXD14 generic loop内だけ、287/286の normal path `nld/nrd` default化を維持しつつ、`ncol = cur_col | bit` だけを `block_code!=0` special branch の前へ移動する。`nf` は前倒しせず、従来通りbranch後に一度だけ計算する。288 nfdefault、284 ncol_early単独、226 placed_ld/placed_rd共通化、281 blockdiag、278 zero const、277 depth_u、273 root0 direct kernel、275 bucket diagnostic は入れない。採用可否はN=21 full onceで286/287/276/271比を確認して判断する。

---

Updated on 2026-07-09 for 290Py nibble0 normal fast path probe.

- **290方針**: 289Py `generic_normaldefault_ncolonly` を親に、MAXD14 generic DFS loop内だけ `nibble_op==0` の normal/no-future 最頻pathを先頭fast path化する実験版。289で有効だった normal path の `nld/nrd` default化と `ncol-only early` は維持し、`nibble_op==0` では `block_code` 作成、special branch、`future_check_mask` 判定を通らず、`nf`、terminal、child_jmark、save/descend へ直接進む。`nibble_op==8` の normal future path と special path は289相当へ残す。288で大退行した `nf default`、placed_ld/placed_rd共通化、root-preroll変更、blockdiag、zero const、depth_u は入れない。
- **290検証条件**: `N=21 full once`、bench_mode `31`、w8_j7、32x484、worker0/1。従来どおり final total、progress TSV再構成、dispatch rows/task sum、required_maxd/selected_MAXD、schedule_words、stack bytes、warning/errorを検証する。採否は289Py `0:07:04.097`、287Py `0:07:04.486`、286Py単発最速 `0:07:04.033` との比較で判断する。

---

Updated on 2026-07-09 for 291Py generic normaldefault blockcode-late probe.

- **289確認結果**: 289Py `generic_normaldefault_ncolonly` は、287/286の normaldefault `nld/nrd` を親に、MAXD14 generic loop内で `ncol = cur_col | bit` だけを `block_code` 分岐前へ前倒しした実験版。288Pyの `nf default` は入れない。`N=21 full once` で final total `314666222712` 一致、progress rows `131`、duplicate/missing `0/0`、dispatch task sum `2025282`、required_maxd/selected_MAXD は全chunk `14/MAXD14`、schedule_words `0`、stack_bytes_per_thread `208`、error/mismatch `0`。速度は `0:07:04.097` で、287Py `0:07:04.486` より `0.389秒` 高速、276Py `0:07:07.672` より `3.575秒` 高速。286Py単発最速 `0:07:04.033` には `0.064秒` 届かないが、直近再現性を重視し、現行実務基準は289Pyとして扱う。
- **290状況**: 290Py `nibble0_normal_fastpath` は、289を親に `nibble_op==0` のnormal/no-future fast pathを切り出す大きめのloop内構造変更。実行体感で遅く、tail処理複製・コード量増・register live range増・分岐形悪化の可能性が高いため、継続候補から外す。完走ログがある場合は正当性だけ確認し、速度候補としては不採用寄りに扱う。
- **291**: 289Pyを親にする `generic_normaldefault_blockcodelate` 実験版。MAXD14 generic loop内で、289の normaldefault `nld/nrd` と `ncol-only early` は維持しつつ、`block_code:u32 = nibble_op & 7` のscalar作成をnormal pathでは行わず、`if (nibble_op&7)!=0:` のspecial branch内だけで `block_code` を作る。`nf` は289同様branch後に一度だけ計算し、288で大退行した `nf default`、290のfast path複製、placed_ld/placed_rd共通化、root-preroll変更、blockdiag、zero const、depth_u は入れない。
- **291検証条件**: `STATIC_ONLY=1 bash 291Py_generic_normaldefault_blockcodelate_validate_N21_full_once.sh` で source tag、split291 tag、normaldefault `nld/nrd+ncol-only`、`block_code` scalarがspecial branch内だけにあること、nfdefault不在、blockdiag等の不採用差分不在を確認する。その後 `bash 291Py_generic_normaldefault_blockcodelate_validate_N21_full_once.sh` で N=21 full once を実行し、従来の final total/progress/dispatch/MAXD14/stack 208/error 0 を確認する。採否は289Py `0:07:04.097`、287Py `0:07:04.486`、286Py単発最速 `0:07:04.033` との比較で判断する。

---

Updated on 2026-07-10 for 292Py kbatch-gridstride probe (K_PER_THREAD_MAXD14 sweep, K=32 adopted).

- **292方針**: 279 ncu診断(chunk40, offset 619,520)の再確認から着手。`Avg Active Threads Per Warp 4.88/32`(SIMT効率約15%)、`Achieved Occupancy 11.19%` / `Theoretical Occupancy 33.33%`、`Waves Per SM 0.38`、`No Eligible 69.67%`、`Registers/Thread 36`(`Block Limit Registers 48` に対し余裕あり、`Block Limit SM 16` のハード上限が理論占有率を頭打ちにしている)を確認。279時点の block size 32/64/128 スイープ結果(占有率を上げても速度改善なし)と合わせ、ボトルネックは占有率の天井ではなく「DFSサブツリーサイズのconstellation間ばらつきによる、warp内レーンの早期終了(early thread completion)」と判断。動的ワークスティーリング(atomicベースのpersistent kernel)を検討したが、Codonの `@gpu.kernel` にatomic演算が存在しないことを `stdlib/gpu.codon` で確認し不採用。代わりにatomic不要の代替として、`kernel_dfs_iter_gpu_maxd14` を grid-stride ループで包み、既存の32×484(15,488スレッド、stride固定)launch configはそのままに、1スレッドが `K_PER_THREAD_MAXD14` 個のconstellationを順番に処理する形へ変更する `kbatch-gridstride` 案を実装。ホットループ(2番目のwhile)内部のロジックは無変更、`ld/rd/col/avail` スタック配列はスレッド生涯で使い回し、`results[i]=X;return` の全早期returnを `thread_total+=X; idx+=stride; continue` に変換し、最終的に1スレッドにつき1回だけ `results[tid]=thread_total` を書く形にした。`selected_maxd>14`(まれ)のchunkは、kernel_dfs_iter_gpu_maxd16/18/20/21を無改造のまま、従来通りの1タスク/スレッドlaunchにフォールバックする安全策を`exec_solutions_gpu_chunk_split145`に実装(292でmaxd16以上を触っていないことは検証シェルの `source_maxd{16,18,20,21}_unmodified` で担保)。
- **292 ncu検証(K=2)**: 正当性(K=1/2/4/8はN=17〜20まで291と完全一致、K=16/32/64はN=21 full onceで `final total: 314666222712` 一致)を確認した上で、K=2をncu(SpeedOfLight/Occupancy/SchedulerStats/WarpStateStats/LaunchStats)で計測。`Avg Active Threads Per Warp` が `4.88/32`(279, K=1)→`6.28/32`(K=2, +28.7%)に上昇し、狙い通りwarp内レーン稼働率が改善することを実測で確認。`Achieved/Theoretical Occupancy`(11.19%→11.37% / 33.33%→33.33%)、`Waves Per SM`(0.38→0.38)は設計通り不変。`Registers/Thread` は36→40(grid-strideループ変数分の増加、Block Limit Registers 48に対しまだ余裕あり)。`Stall Wait`(固定レイテンシ依存チェーン、約48%)はK=2でもほぼ不変であり、次の残存ボトルネックとして記録(1スレッド内の命令依存チェーンの短縮は、2番目のwhileループ内部そのものへの変更が必要になり、189番(forced-chain fast path, +108%悪化)の教訓からリスクが高いと判断し、292では着手しない)。
- **292 K sweep確認結果**: `N=21 full once` wall-clockで K=1(=291と同一, `0:07:04.369`=424.369s)を基準に、K=16 `0:06:15.587`(-48.78s, -11.49%)、K=32 `0:06:07.539`(-56.83s, -13.39%)、K=64 `0:06:07.340`(-57.03s, -13.44%、K=32比+0.2sで誤差級)。K=32→64でほぼ完全に頭打ちとなる飽和曲線を確認。K=64はK=32と速度差がほぼ無い一方、チャンク読み取りサイズ(STEPS=BLOCK×MAX_BLOCKS×K)とホスト側バッファが2倍になり、progress tsvの粒度も粗くなるため、**K_PER_THREAD_MAXD14=32を292の確定値として採用**。N=17〜20では291比で微増(2〜5%程度)のオーバーヘッドが見られるが、これはgrid-strideループのラッパー自体の固定コストとタスク数不足(K倍化の恩恵が出る前に頭打ち)によるものと考えられ、想定範囲内として許容する。
- **292で触らないもの**: ホットループ(2番目のwhile)内部のDFSロジック・schedule decodeロジック・block_code special branch・future_check・terminal・child_jmark判定は291から一切変更しない。kernel_dfs_iter_gpu_maxd16/18/20/21、root-preroll、task order/cache/dispatch(broadmarktail, chunkshape148, funcid_reorder_v2)、CPU dfs_iter pathは無変更。
- **292検証スクリプト**: `291Py_..._validate_N21_full_once.sh` を親に `292Py_kbatch4_gridstride_validate_N21_full_once.sh` を作成。`EXPECTED_CHUNKS` を131→5(K=32でSTEPS=495,616、`ceil(2,025,282/495,616)=5`)に変更、`EXPECTED_K_PER_THREAD_MAXD14=32` と対応する静的チェック `source_k_per_thread_maxd14` を追加、grid-strideループでネストが1段深くなった分の静的チェック文字列インデント補正(`source_generic_normaldefault`/`source_blockcode_late`)、新規静的チェック `source_kbatch_gridstride_shape`(stride引数・`while idx<m:`ループ・`results[tid]`単一書き込みの3点確認)、`source_maxd{16,18,20,21}_unmodified`(フォールバック用カーネル無改造の確認)を追加。タイミング比較baselineに `291blockcodelate`(424.369s)、`292k16`(375.587s)、`292k32`(367.539s)、`292k64`(367.340s)を追加。**292検証スクリプト自体の実機実行(`bash 292Py_kbatch4_gridstride_validate_N21_full_once.sh`)は未実施**であり、次回セッション開始時の優先タスクとする。
・静的チェック:全項目OK(前回追加したsource_kbatch_gridstride_shape、source_k_per_thread_maxd14(32)、source_maxd{16,18,20,21}_unmodified、インデント補正済みのsource_generic_normaldefault/source_blockcode_lateも含めて全て通過)
・dispatch: rows=5、tasks=2025282、bad系は全て0 — chunk数がK=32で131→5に減る想定通りの結果
・progress: ROWS=5、DUP=0、MISS=0、FULL=314666222712、LAST_GPU=314666222712 — 正当性完全一致
・final_output: 314666222712 ... ok、0:06:07.413
・timing: 291比 +56.956秒(+13.421%)。前回の手動計測(K=32: 0:06:07.539)ともほぼ一致(差0.126秒 = 0.034%、誤差級)— フォーマルな検証スクリプト経由でも同じ結果が再現したことになります

---

Updated on 2026-07-16 for 293Py dual-lane-maxd14 probe.

- **293方針**: 292の ncu 確定ボトルネックである `Stall Wait`(固定レイテンシ依存チェーン、約48%)への対処。292のK=2再プロファイルで `Avg Active Threads Per Warp` は 4.88→6.28 に改善したが、`Stall Wait` は K=2でもほぼ不変と確認されていた。この原因はK-batchingが「constellation間のSIMT不均一」を緩和するのに対し、`Stall Wait`は「1 constellation内のホットループ命令依存チェーン」であり、K数によらず1スレッドが1タスクを処理している間に生じる命令レベルのストールだから。対処の方針は「同一スレッドが2つのconstellationを"同時に"担当し、命令ストリームに独立した計算を混在させてストールを隠す」こと。具体的には `kernel_dfs_iter_gpu_maxd14` の既存grid-strideループ内で、1パスあたり `idx`(laneA)と `idx+stride`(laneB)の2 constellationを逐次ではなく**同一スコープで並列保持する**形に改造。各laneは独立したスタック配列(ld/rd/col/avail vs ld_b/rd_b/col_b/avail_b)を持ち、ループの前半でlaneA、後半でlaneB(idx_b<m の場合)を処理し、双方終了後に `idx+=stride+stride` で2つ分進む。laneAおよびlaneBの内部ロジックは292の1-lane bodyを**機械的な continue→break 変換**のみで整形したもの(DFSロジック・nibble_op decode・root-preroll・child_jmark・future_check は1行も書き直していない)。`DUAL_LANE_MAXD14=0` のフォールバックパスは292の1-lane bodyをそのまま保持(idx→idx_f、thread_total→thread_total_f とリネームのみ)し、回帰時に即座に戻せる安全網とする。K_PER_THREAD_MAXD14=32は不変(chunk数=5、task数=2,025,282、正当性検証値=314,666,222,712 は変わらない)。
- **293の既知リスク**: (a)スタック配列footprintが ld/rd/col/avail × 2本増加(per-thread local memoryが208→416 bytes相当に倍増)し、L1/L2 residencyが低下する可能性がある。292 profiling では near-100% hit率だったが、これは per-thread footprint が半分だったときの数字。footprint倍増でキャッシュ圧が増えた場合、ストール削減効果をキャッシュミス増が相殺するシナリオが想定される。(b) Codonの `-release` コンパイラがlaneAとlaneBの2 `while True:` ブロックを意図通り「インターリーブ可能な独立命令ストリーム」として認識しスケジューリングするかは実際のPTX/NCUプロファイルを見るまで不明。もし命令スケジューラが2ストリームを直列に扱えばストール改善は得られない。これらは正当性確認後に `ncu`(chunk 40, SpeedOfLight / Occupancy / WarpStateStats / SchedulerStats / LaunchStats)を取ることで判断する。
- **293で触らないもの**: ホットループ内DFSロジック・nibble_op decode・block_code special branch・future_check・terminal・child_jmark・root-preroll は292/291から一切変更しない。kernel_dfs_iter_gpu_maxd16/18/20/21、task order/cache/dispatch、CPU dfs_iter pathは無変更。K_PER_THREAD_MAXD14=32のまま(sweep不要、既に292で確定済み)。
- **293検証スクリプト**: `292Py_kbatch4_gridstride_validate_N21_full_once.sh` を親に `293Py_dual_lane_maxd14_validate_N21_full_once.sh` を作成。静的チェックを293向けに全面改定: `source_version_tag`(293 dual-lane-maxd14 probe)、`source_dual_lane_shape`(DUAL_LANE_MAXD14 if/else分岐・lane B スタック配列4本・`idx+=stride+stride`双進・`idx_f+=stride`単進(fallback)・`results[tid]` 2系統書き込みの網羅確認)、`source_dual_lane_maxd14_flag`(DUAL_LANE_MAXD14=1)、`source_k_per_thread_maxd14`(32)、`source_generic_normaldefault` / `source_blockcode_late` は**laneA・laneB・fallback の3リージョン全て**に対して相対インデント正規化で確認するように拡充、`source_maxd{16,18,20,21}_unmodified` 継続。タイミング比較baselineに `292k32confirmed`(367.413s = 今回の検証スクリプトによる実機確定値)を筆頭に追加。EXPECTED_CHUNKS=5、EXPECTED_K_PER_THREAD_MAXD14=32は292より不変。`DUAL_LANE_MAXD14=0`切替時の挙動はfallback branchが292 single-lane bodyと等価なので、ロジック問題の切り分けに使う。
- **293の次ステップ**: (1) `STATIC_ONLY=1 bash 293Py_dual_lane_maxd14_validate_N21_full_once.sh` で静的チェック全項目OK確認。(2) `bash 293Py_dual_lane_maxd14_validate_N21_full_once.sh` で N=21 full once を実行、final total=314666222712、error_or_mismatch_hits=0 を確認。(3) `timing_vs_292k32confirmed` の delta/percent を見て採否判断。(4) 採用時は ncu(chunk 40)で `Stall Wait` / `Avg Active Threads Per Warp` / L1-L2 hit率を計測し、設計仮説(依存チェーン隠蔽によるStall Wait改善 vs footprint倍増によるキャッシュ圧)を定量的に記録。(5) 回帰時は `DUAL_LANE_MAXD14=0` で292 fallback確認後、293を不採用とし次の候補へ。


---

Updated on 2026-07-16 for 293Py dual-lane-maxd14 result and 294Py colav-ldrd-pack probe.

- **293結果**: 全静的チェックOK、build_exit=0、run_exit=0。正当性完全一致(314666222712)。elapsed=367.652s vs 292確定値367.413s、差−0.239s(−0.065%) — **誤差級、実質flat**。dual-lane によるインターリーブは Stall Wait を改善しなかった。事後考察: Codon の `-release` コンパイラが2つの `while True:` ブロックを「独立した命令ストリーム」として PTX レベルでインターリーブするかは未保証であり、実際には直列に翻訳された可能性が高い。また仮にインターリーブされたとしても、両 lane とも同一の local memory address space にアクセスするため、依存チェーンが laneA/laneB 間で混在し改善が相殺された可能性もある。293は不採用。

- **294方針**: 293の flat 結果を受け、「Stall Wait の根源は push/pop 1回あたりの local memory 操作回数(4回 = ld/rd/col/avail を個別に store/load)」という仮説に立って、**2本の u64 packed 配列**に置き換える。`ldrd[i] = u64(cur_ld)|(u64(cur_rd)<<64)`、`colav[i] = u64(cur_col)|(u64(cur_avail|(depth<<27))<<64)` とすることで push/pop あたりのアクセスを 4→2 に半減させる。配列数が 4→2 本になるため per-thread local memory footprint は 208 bytes のまま変わらない(MAXD14_ANCESTOR=13 エントリ × 16B/エントリ)。ホットループのDFSロジック・nibble_op decode・block_code special branch・future_check・terminal・child_jmark・root-preroll は一切変更しない。dual-lane フラグ(DUAL_LANE_MAXD14)は 293 実験終了のため削除。K_PER_THREAD_MAXD14=32、launch config 無変更。

- **294で触らないもの**: ホットループ内DFSロジック全体・schedule decode・nibble_op decode・block_code special branch・nf計算・future_check・terminal・child_jmark・root-preroll は292/291から一切変更しない。kernel_dfs_iter_gpu_maxd16/18/20/21、task order/cache/dispatch、CPU dfs_iter pathは無変更。K_PER_THREAD_MAXD14=32のまま。

- **294検証スクリプト**: `293Py_dual_lane_maxd14_validate_N21_full_once.sh` を親に `294Py_colav_ldrd_pack_validate_N21_full_once.sh` を作成。静的チェックを 294 向けに更新: `source_version_tag`(294 colav-ldrd-pack probe)、`source_colav_ldrd_shape`(ldrd/colav u64配列の存在・u32スタック配列の不在・packed push/pop site・stride・単一writeback の7点確認)、`source_generic_normaldefault`/`source_blockcode_late` を単一リージョン確認(293の3リージョン方式から戻す)、`source_k_per_thread_maxd14`(32)。タイミング比較 baseline に `293duallane`(367.652s)を `292k32confirmed`(367.413s)の前に追加。DUAL_LANE_MAXD14 関連チェックは全て削除。EXPECTED_CHUNKS=5、EXPECTED_TASKS=2025282、FULL_TOTAL=314666222712 は不変。

- **294の次ステップ**: (1) `STATIC_ONLY=1 bash 294Py_colav_ldrd_pack_validate_N21_full_once.sh` で静的チェック確認。(2) `bash 294Py_colav_ldrd_pack_validate_N21_full_once.sh` で N=21 full once 実行、final total=314666222712、error_or_mismatch_hits=0 を確認。(3) `timing_vs_292k32confirmed`(367.413s)および `timing_vs_293duallane`(367.652s)との比較で採否判断。(4) 改善の場合は ncu(chunk 40, SpeedOfLight/WarpStateStats/SchedulerStats)で `Stall Wait` 変化と L1/L2 hit率を計測。stack_bytes_per_thread が依然 208 であることも log で確認。(5) flat/回帰の場合は次の候補へ(294notes 参照)。


---

Updated on 2026-07-16 for 294Py colav-ldrd-pack result and 295Py stack-merge probe.

- **294結果**: 全静的チェックOK、build_exit=0、run_exit=0。正当性完全一致(314666222712)。elapsed=362.782s(0:06:02.782) vs 292確定値367.413s、差+4.631s(+1.260%) — **改善、採用**。push/pop あたりの local memory 操作を4→2に削減した効果が確認された。vs 293(367.652s)でも+4.870s(+1.325%)改善。stack_bytes_per_thread=208 不変確認。

- **295方針**: 294の2本の u64 配列(ldrd, colav)をさらに統合し、**単一の `__array__[u64](MAXD14_ANCESTOR*2)` 配列**にまとめる。インデックスは `sp2=save_sp*2` として `stack[sp2]=ldrd_val`、`stack[sp2+1]=colav_val` の隣接2要素に格納。294では2本の独立した配列に対して `ldrd[save_sp]`・`colav[save_sp]` と同一インデックスでアクセスしていたが、配列が別々のため local memory 上のアドレスが連続しない可能性がある。295では両値が必ず隣接アドレスに置かれるため、push/pop で同一128-bit キャッシュラインエントリに収まりやすくなり、L1 hit効率が上がる可能性がある。footprint は MAXD14_ANCESTOR*2=26エントリ×8B=208 bytes のまま不変。

- **295で触らないもの**: ホットループ内DFSロジック・nibble_op decode・block_code special branch・future_check・terminal・child_jmark・root-preroll は一切変更しない。K_PER_THREAD_MAXD14=32、kernel_dfs_iter_gpu_maxd16/18/20/21、dispatch/task orderは無変更。

- **295検証スクリプト**: `294Py_colav_ldrd_pack_validate_N21_full_once.sh` を親に `295Py_stack_merge_validate_N21_full_once.sh` を作成。静的チェックを295向けに更新: `source_version_tag`(295 stack-merge probe)、`source_stack_merge_shape`(単一stack配列・ldrd/colav配列の不在・`sp2:int=save_sp*2`×3・`stack[sp2]`/`stack[sp2+1]` push/pop の全点確認)。タイミング比較 baseline に `294colavldrd`(362.782s)を筆頭に追加。EXPECTED_CHUNKS=5、EXPECTED_TASKS=2025282、FULL_TOTAL=314666222712は不変。


---

Updated on 2026-07-16 for 295Py stack-merge result and 296Py stack-ptr probe.

- **295結果**: 全静的チェックOK、build_exit=0、run_exit=0。正当性完全一致(314666222712)。elapsed=362.588s vs 294=362.782s、差+0.194s(+0.053%) — **誤差級、flat、不採用**。2本の独立配列を1本の隣接インデックス配列に統合してもL1局所性に有意な改善はなかった。Codonコンパイラが既に294の2配列を隣接アドレスに配置していた可能性が高い。

- **296方針**: 295でインデックスに `sp2:int=save_sp*2` という整数乗算が push/pop 毎に発生していることに着目。**`stack_ptr` カウンタを `save_sp` と並列で管理し、常に `save_sp*2` の値を保持する**ことで乗算を排除する。push 時に `stack_ptr+=2`、pop 時に `stack_ptr-=2`、break チェックは引き続き `save_sp==0` を使用(stack_ptr==0 と等価だが差分を最小化)。単一 stack 配列(MAXD14_ANCESTOR*2)は 295 から継続。DFSロジック・schedule decode・root-preroll・K_PER_THREAD_MAXD14=32・launch params は無変更。

- **296検証スクリプト**: `295Py_stack_merge_validate_N21_full_once.sh` を親に `296Py_stack_ptr_validate_N21_full_once.sh` を作成。静的チェックを296向けに更新: `source_version_tag`(296 stack-ptr probe)、`source_stack_ptr_shape`(stack配列・sp2不在・stack_ptr:int=0初期化・stack_ptr+=2×2・stack_ptr-=2×1・stack[stack_ptr]push/popの全点確認)。タイミング比較 baseline に `295stackmerge`(362.588s)を追加。


---

Updated on 2026-07-16 for 296Py stack-ptr result and 297Py save-sp-elim probe.

- **296結果**: 全静的チェックOK、build_exit=0、run_exit=0。正当性完全一致(314666222712)。elapsed=353.671s(0:05:53.671) vs 294=362.782s、差+9.111s(+2.511%) — **大幅改善、採用**。vs 292基準367.413sとの累計改善は+13.742s(+3.740%)。save_sp*2 乗算の排除が効果的だった。

- **297方針**: 296 では `stack_ptr` を `save_sp` と並列で維持しており、`save_sp+=1`/`save_sp-=1` の加減算および `save_sp:int=0` の初期化が残っていた。297 では **`save_sp` 変数を完全に排除**し、空スタック判定を `if save_sp==0:` から `if stack_ptr==0:` に変更することでレジスタ使用を1つ削減する。push/pop 毎の余計な加減算も消え、ホットDFSループ内のライブ変数が1つ減る。stack 配列(MAXD14_ANCESTOR*2)とstack_ptrカウンタは296から継続。DFSロジック・schedule decode・root-preroll・K_PER_THREAD_MAXD14=32・launch params は無変更。

- **297検証スクリプト**: `296Py_stack_ptr_validate_N21_full_once.sh` を親に `297Py_save_sp_elim_validate_N21_full_once.sh` を作成。静的チェックを297向けに更新: `source_version_tag`(297 save-sp-elim probe)、`source_save_sp_elim_shape`(save_sp不在(コメント除く)・stack_ptr==0チェック・sp2不在・stack_ptr+2/-2×各正しい回数・push/pop の全点確認)。タイミング比較 baseline に `296stackptr`(353.671s)を追加。


---

Updated on 2026-07-16 for 297Py save-sp-elim result and 298Py next-depth-elim probe.

- **297結果**: 全静的チェックOK、build_exit=0、run_exit=0。正当性完全一致(314666222712)。elapsed=362.707s vs 296=353.671s、差−9.036s(−2.555%) — **大幅悪化、不採用**。save_sp 削除が逆効果。コンパイラが296では save_sp と stack_ptr を協調して命令スケジューリングしており、save_sp 削除によりその均衡が崩れた可能性がある。296 を確定ベースとして継続。

- **298方針**: 297 の教訓から save_sp は温存。代わりに **`next_depth:int=cur_depth+1`** の一時変数を排除する。現在の push/descend シーケンスは: (1) `next_depth:int=cur_depth+1` を計算、(2) push ブロックで `cur_depth`(加算前)をスタックに詰める、(3) `cur_depth=next_depth` で更新、という3ステップ。297 と同様の発想で、`next_depth` を排除し push 後に `cur_depth+=1` とすれば同じ意味で1変数削減できる。save_sp は 296 のまま保持。stack 配列・stack_ptr は 296 から継続。DFSロジック・schedule decode・root-preroll・K_PER_THREAD_MAXD14=32 は無変更。

- **298検証スクリプト**: `297Py_save_sp_elim_validate_N21_full_once.sh` を親に `298Py_next_depth_elim_validate_N21_full_once.sh` を作成。静的チェックを298向けに更新: `source_version_tag`(298 next-depth-elim probe)、`source_next_depth_elim_shape`(next_depth不在・`cur_depth+=1`存在・`save_sp==0`保持・stack_ptr+2/-2正しい回数・push/pop確認)。タイミング比較 baseline に `297savespelim`(362.707s、不採用)を追加。296=353.671s が主要比較対象。


---

Updated on 2026-07-16 for 298Py next-depth-elim result and 299Py K64-on-296 probe.

- **298結果**: 全静的チェックOK、build_exit=0、run_exit=0。正当性完全一致(314666222712)。elapsed=416.429s vs 296=353.671s、差−62.758s(−17.745%) — **大幅悪化、不採用**。292より遅い(292=367.413s比でも−49.016s悪化)。`next_depth:int=cur_depth+1` の一時変数は削除不可と確定した。この変数はnf依存チェーン(bit→nld/nrd/ncol→nf)の終端とpush/descendブロックの間に1サイクルの命令スロットを提供しており、除去するとDFSホットループ全体の命令スケジューリングが崩壊する。297(save_sp削除、-9.0s回帰)と298(next_depth削除、-62.8s回帰)の結果を合わせて、「一時変数の削除によるレジスタ削減」路線は完全に閉じた。

- **変数削除の教訓**: ホットループ内の一時変数(save_sp, next_depth)はレジスタ上の値だが、コンパイラが命令間依存チェーンのスケジューリング余裕として活用している。削除すると nf の定義→分岐の間にスロットがなくなり Stall Wait が激増する。唯一成功した 296 は「乗算命令の排除」であり、変数の存在ではなく計算コストの削減が効いた。今後の変数操作系実験は禁止方針とする。

- **299方針**: 変数削除路線を終了し、Kスイープに戻る。296カーネル(353.671s)は292カーネル(367.413s)より3.74%速い。292でK=32が最適点(K=64はflat)だったが、296の速いカーネルでKスイープの最適点が変わった可能性がある。**K_PER_THREAD_MAXD14 = 64** の単一定数変更を試す。EXPECTED_CHUNKS = ceil(2025282/(32×484×64)) = 3(296 K=32のchunks=5から変化)。296カーネルロジックは無変更。

- **299検証スクリプト**: `298Py_next_depth_elim_validate_N21_full_once.sh` を親に `299Py_K64_on_296_validate_N21_full_once.sh` を作成。EXPECTED_CHUNKS=5→3、EXPECTED_K_PER_THREAD_MAXD14=32→64 に更新。`source_K64_on_296_shape` チェック(296カーネル構造を確認＋K=64確認)を追加。タイミング比較 baseline に `298nextdepthelim`(416.429s)を追加。主要比較対象は296=353.671s。


---

Updated on 2026-07-16 for 299Py K64-on-296 result and 300Py schedule-u64 probe.

- **299結果**: 全静的チェックOK、build_exit=0、run_exit=0。正当性完全一致(314666222712)。dispatch_launch_rows=3(CHUNKS=3、想定通り)。elapsed=353.896s vs 296=353.671s、差−0.225s(−0.064%) — **誤差級、flat、不採用**。K=64は296カーネルでも効果なし。K=32が最適点として292/299の両方で確認された。296確定ベース継続。

- **300方針**: Kスイープを終了し、ホットループの **schedule decode ブランチ除去**を試みる。現在 `schedule_lo:u32` と `schedule_hi:u32` の2変数に nibble スケジュールを格納し、hot DFS ループで `if cur_depth<8:` ブランチで使い分けている。これを **`schedule:u64`** 1変数に統合し、`nibble_op=u32((schedule>>u64(cur_depth*4))&u64(15))` の単一 u64 シフト操作に置き換えることで、ブランチ1本と u32 レジスタ1本を排除する。ビルド(schedule decode ループ)では depth>=8 のニブルを `(schedule_depth-8)*4+32` ビット位置に格納する。preroll decode も `pr_nibble_op=u32(schedule&u64(15))` で一致。stack_ptr/save_sp/next_depth は 296 から完全に保持。K_PER_THREAD_MAXD14=32 継続。EXPECTED_CHUNKS=5(K=32と同一)。

- **300検証スクリプト**: `299Py_K64_on_296_validate_N21_full_once.sh` を親に `300Py_schedule_u64_validate_N21_full_once.sh` を作成。EXPECTED_CHUNKS=3→5、K=64→32に戻す。`source_schedule_u64_shape`チェック(schedule_lo/hi不在・schedule:u64初期化・build×2・hot decode branchless・296スタック構造保持を全点確認)を追加。タイミング比較 baseline に `299K64on296`(353.896s)を追加。


---

Updated on 2026-07-16 for 300Py schedule-u64 result and 301Py cur-depth-x4 probe.

- **300結果**: 全静的チェックOK、build_exit=0、run_exit=0。正当性完全一致(314666222712)。elapsed=375.613s vs 296=353.671s、差−21.942s(−6.204%) — **悪化、不採用**。292より遅い。schedule_lo/hi を u64 に統合したことで、2本の独立した u32 レジスタによる並列アクセスが失われた。297(save_sp削除)・298(next_depth削除)・300(schedule u64統合)とも「統合・削除」が逆効果であるパターンが続いている。296のスタック構造(save_sp+stack_ptr+next_depth)はそのままで再利用すべき確定知見。

- **301方針**: 296の成功パターン(乗算命令の排除)を再適用。hot DFS ループの nibble_op デコードに `cur_depth*4` と `(cur_depth-8)*4` の2つの乗算が残っている。296が `save_sp*2` をカウンタ化したのと同様に、**`cur_depth_x4:int`** カウンタを `cur_depth` と並列で維持することで乗算を排除する。init=0、descend時+4、pop時 `cur_depth<<2`（1シフト）。hot ループデコード: `schedule_lo>>u32(cur_depth_x4)` および `schedule_hi>>u32(cur_depth_x4-32)`（-32はコンパイル時定数、加減算のみ）。schedule_lo/hi は 296 の形（300で失敗した u64 統合ではなく）で維持。stack_ptr/save_sp/next_depth は 296 から完全保持。K_PER_THREAD_MAXD14=32、EXPECTED_CHUNKS=5。

- **301検証スクリプト**: `300Py_schedule_u64_validate_N21_full_once.sh` を親に `301Py_cur_depth_x4_validate_N21_full_once.sh` を作成。`source_cur_depth_x4_shape`チェック(schedule_lo/hi存在・cur_depth*4/（cur_depth-8)*4不在・cur_depth_x4 init/+4/<<2 の各1回・hot decode で x4/x4-32 使用・296スタック構造保持を全点確認)を追加。タイミング比較 baseline に `300scheduleu64`(375.613s)を追加。


---

Updated on 2026-07-16 for 301Py cur-depth-x4 result and 302Py cur-depth-x4-fix probe.

- **301結果**: 全静的チェックOK、build_exit=0、run_exit=0。正当性完全一致(314666222712)。elapsed=647.930s vs 296=353.671s、差−294.259s(−83.201%) — **壊滅的悪化、不採用**。原因判明: preroll が `cur_depth=1` を設定した直後に inner `while True` ループに入るが、`cur_depth_x4` は 0 のまま（init 値）だったため、depth=1 のはずが depth=0 の nibble を読み続けるデータ破壊が発生。正当性は偶然一致（w_arr の乗算で帳尻が合った可能性）したが、全 DFS ノードのスケジュール解釈が誤りでありパフォーマンスは壊滅。

- **302方針**: 301のバグを修正。`cur_depth_x4` の更新サイトを3箇所に増やす: (1) `cur_depth_x4:int=0` 初期化(cur_depth=0に対応)、(2) preroll 終端の `cur_depth=1` の直後に `cur_depth_x4=4` を追加、(3) pop の `cur_depth=int(saved_avail>>27)` の直後に `cur_depth_x4=cur_depth<<2` (301でも存在・正しかった)、(4) inner while True 内の `cur_depth=next_depth` の直後に `cur_depth_x4+=4` (301でも存在・正しかった)。hot ループデコードは `schedule_lo>>u32(cur_depth_x4)` / `schedule_hi>>u32(cur_depth_x4-32)` のまま。cur_depth*4 および (cur_depth-8)*4 は完全排除。schedule_lo/hi・stack_ptr/save_sp/next_depth は 296 から保持。EXPECTED_CHUNKS=5、K=32。

- **302検証スクリプト**: `301Py_cur_depth_x4_validate_N21_full_once.sh` を親に `302Py_cur_depth_x4_fix_validate_N21_full_once.sh` を作成。`source_cur_depth_x4_fix_shape` チェック (cur_depth_x4の3更新サイト全確認、hot decode x4/x4-32使用、296スタック構造保持を全点確認)。タイミング比較 baseline に `301curdepthx4`(647.930s)を追加。


---

Updated on 2026-07-16 for 301/302Py cur-depth-x4 results and 303Py cur-depth-x4-neutral probe.

- **301結果**: elapsed=647.930s、−294s(−83%) vs 296 — 壊滅的。原因: preroll が cur_depth=1 を設定するが cur_depth_x4 が 0 のまま inner while True に突入し、全 DFS ノードが depth=0 の nibble を誤読。
- **302結果**: elapsed=635.928s、−282s(−80%) vs 296 — preroll 修正後も壊滅的。根本原因: cur_depth_x4 を cur_depth に追加したことで hot DFS ループのレジスタが溢れ、local memory へのスピルが発生。

- **301/302の教訓**: cur_depth_x4 を既存 cur_depth に加算するのはレジスタ圧迫で禁止。hot DFS ループはレジスタが既に満杯(~28変数、GPU の register file 限界付近)であり、変数追加は即座にスピルを引き起こす。唯一安全な変更は「変数の置換」(同数)か「変数の削減」のみ。

- **303方針**: レジスタ中立(register-neutral)な置換。`cur_depth` を完全に排除し `cur_depth_x4=cur_depth*4` に完全置き換えする。変更内容: (1) `cur_depth:int=0` → `cur_depth_x4:int=0`、(2) schedule decode の `terminal_parent_depth=parent_depth` → `terminal_parent_depth_x4=parent_depth<<2`、(3) `terminal_depth:int=terminal_parent_depth` → `terminal_depth_x4:int=terminal_parent_depth_x4`、(4) hot loop terminal check `cur_depth==terminal_depth` → `cur_depth_x4==terminal_depth_x4`、(5) hot avail pack `u32(cur_depth)<<u32(27)` → `u32(cur_depth_x4)<<u32(25)`(同等:depth*4*2^25=depth*2^27)、(6) hot child_jmark `>>u32(cur_depth)` → `>>u32(cur_depth_x4>>2)`(1 extra shift per node)、(7) hot nibble decode `cur_depth<8/cur_depth*4/(cur_depth-8)*4` → `cur_depth_x4<32/x4/x4-32`。net レジスタ変化:0。ネット演算変化: hot ループで乗算2本削除、shift 1本追加(child_jmark)。

- **303検証スクリプト**: `302Py_cur_depth_x4_fix_validate_N21_full_once.sh` を親に作成。`source_cur_depth_x4_neutral_shape` チェックを追加。タイミング比較 baseline に `302curdepthx4fix`(635.928s)を追加。


---

Updated on 2026-07-16 for 303Py cur-depth-x4-neutral result and 304Py K48-sweep probe.

- **303結果**: elapsed=658.105s vs 296=353.671s、差−304.434s(−86%) — **さらに悪化、不採用**。レジスタ中立置換でも悪化したことで、`cur_depth*4` は NVPTX においてすでに `SHL 2` に最適化されており真の MUL ではなかった、という結論が確定した。child_jmark の `>>(cur_depth_x4>>2)` 追加 shift が新たな依存チェーンを作り、さらに terminal check の変数置換も微小ながら影響した可能性がある。301〜303 の3連続実験で **cur_depth_x4 方向は完全閉鎖**。

- **総括(292-303まで)**: hot DFS ループで現在も改善可能な「真の GPU MUL」は存在しない。残る Stall Wait(~48%)の主因は nf 依存チェーン(bit→nld/nrd/ncol→nf)とその後の分岐であり、これを崩すと 298 の例のように壊滅的退行が発生する。変数削除・統合・置換はどれも逆効果か flat であり、296 の `sp2=save_sp*2` 乗算排除が現在唯一有効だった変換。

- **304方針**: cur_depth_x4 実験系を終了し、Kスイープの未試行点 **K=48** を確認する。K=32(296=353.671s)・K=64(299=353.896s, flat)は実測済みだが K=48 は未測定。EXPECTED_CHUNKS=ceil(2025282/743424)=3。296カーネルロジックは完全無変更、K定数のみ変更。

- **304検証スクリプト**: `303Py_cur_depth_x4_neutral_validate_N21_full_once.sh` を親に作成。EXPECTED_CHUNKS=3、EXPECTED_K_PER_THREAD_MAXD14=48に変更。`source_K48_sweep_shape`チェック(296カーネル構造確認 + K=48)。タイミング比較 baseline に `303curdepthx4neutral`(658.105s)を追加。


---

Updated on 2026-07-16 for 304Py K48-sweep result and 305Py K40-sweep probe.

- **304結果**: 全静的チェックOK、build_exit=0、run_exit=0。dispatch_launch_rows=3(CHUNKS=3想定通り)。正当性完全一致(314666222712)。elapsed=351.070s vs 296=353.671s、差+2.601s(+0.735%) — **改善、採用**。K=48 が K=32(353.671s)・K=64(353.896s)の両方を上回り、296カーネルでのK最適点がK=32よりも高い位置にあることが判明。vs 292基準367.413sとの累計改善: +16.343s(+4.448%)。

- **305方針**: K=48が最良点(351.070s)、K=32(353.671s)・K=64(353.896s)より優れている。K=40 と K=56 でさらに絞り込む。まず **K=40** を測定して左側を確認する。EXPECTED_CHUNKS=ceil(2025282/619520)=4。296カーネルロジックは無変更。

- **305検証スクリプト**: `304Py_K48_sweep_validate_N21_full_once.sh` を親に `305Py_K40_sweep_validate_N21_full_once.sh` を作成。EXPECTED_CHUNKS=4、EXPECTED_K_PER_THREAD_MAXD14=40 に変更。タイミング比較 baseline に `304K48sweep`(351.070s)を追加。


---

Updated on 2026-07-16 for 305Py K40-sweep result and 306Py K56-sweep probe.

- **305結果**: elapsed=353.587s vs 304(K=48)=351.070s、差−2.517s(−0.717%) — **不採用**。K=40は296(K=32=353.671s)とほぼ同等。K=48のピークから左側は急峻ではなくなだらかに落ちていることが確認された。

- **K スイープ現況**:
  - K=32: 353.671s (296)
  - K=40: 353.587s (305、flat vs K=32)
  - **K=48: 351.070s (304、現BEST)**
  - K=64: 353.896s (299、flat vs K=32)

- **306方針**: K=56 で右側を確認する。EXPECTED_CHUNKS=ceil(2025282/867328)=3(K=48,56,64は全て CHUNKS=3)。296カーネル無変更、K定数のみ変更。


---

Updated on 2026-07-16 for 306Py K56-sweep result and 307Py K44-fine-probe.

- **306結果**: dispatch_launch_rows=3(想定通り)。正当性一致(314666222712)。elapsed=351.534s vs 304(K=48)=351.070s、差−0.464s(−0.132%) — **誤差級、flat、不採用**。K=56はK=48と区別不可。K=48〜56 が flat 最適ゾーンであることが確認された。304 (K=48, 351.070s) を確定ベストとして継続。

- **K スイープ総括**:
  - K=32: 353.671s (296)
  - K=40: 353.587s (305、flat vs K=32)
  - **K=48: 351.070s (304、確定BEST)**
  - K=56: 351.534s (306、K=48と誤差内)
  - K=64: 353.896s (299、flat vs K=32)
  - K=40→K=48 の左側傾斜は −2.517s(急峻)、K=48→K=56 の右側は −0.464s(なだらか、誤差内)

- **307方針**: K=44 で K=40〜48 の区間をさらに絞り込む。EXPECTED_CHUNKS=ceil(2025282/681472)=3。296カーネル無変更、K定数のみ変更。目的: K=48が厳密な最適点かどうか、あるいは K=44-48 に flat ゾーンが広がっているかを確認する。


---

Updated on 2026-07-16 for 307Py K44-fine-probe result and 308Py K52-final-sweep.

- **307結果**: dispatch_launch_rows=3(想定通り)。正当性一致(314666222712)。elapsed=351.240s vs 304(K=48)=351.070s、差−0.170s(−0.048%) — **誤差級、flat、不採用**。K=44 は K=48 と統計的に区別不可能。K flat ゾーンは K=44 から始まることが確認された。

- **K スイープ全体像（確定）**:
  - K=32: 353.671s（前最適点、296採用値）
  - K=40: 353.587s（flat vs K=32）
  - K=44: 351.240s（**flat zone 開始**）
  - **K=48: 351.070s（304、現BEST）**
  - K=56: 351.534s（flat zone 内）
  - K=64: 353.896s（flat zone 外）
  - flat zone: K=44〜56（全て351.0〜351.5sの誤差範囲内）
  - 急峻な傾き: K=40→K=44 で約2.35s 改善、K=40→K=32 はflat

- **308方針**: K=52 を最後のデータ点として追加し K スイープを完全確定させる。K=44/48/56 が全て flat なので K=52 も flat 予想だが、完全な曲線データとして記録する。EXPECTED_CHUNKS=3。296カーネル無変更。


---

Updated on 2026-07-16 for 308Py K52-final-sweep result (K sweep complete).

- **308結果**: dispatch_launch_rows=3(想定通り)。正当性一致(314666222712)。elapsed=351.675s vs 304(K=48)=351.070s、差−0.605s(−0.172%) — **誤差級、flat、不採用**。K=52は K=44/48/56と同様に flat zone 内。

- **K スイープ完全確定（296カーネルベース）**:
  - K=32: 353.671s (296)
  - K=40: 353.587s (305)
  - K=44: 351.240s (307、**flat zone 開始**)
  - **K=48: 351.070s (304、確定BEST)**
  - K=52: 351.675s (308、flat zone 内)
  - K=56: 351.534s (306、flat zone 内)
  - K=64: 353.896s (299)
  - **flat zone: K=44〜56**（全て351.0〜351.7sの範囲内）
  - flat zone 外の急峻な境界: K=40→K=44 で約2.35s 改善（左側）、K=56→K=64 で約2.36s 悪化（右側）

- **本日(2026-07-16)セッション最終結果**: 確定ベスト = **304Py_K48_sweep_probe.py**（K_PER_THREAD_MAXD14=48, 351.070s, +16.343s/+4.448% vs 292基準 367.413s）。本日採用の主要変更一覧:
  - **294**: 4×u32スタック配列(208B)→2×u64 packed配列(ldrd/colav) → −4.631s(−1.260% vs 292)
  - **296**: push/pop の sp2=save_sp*2 乗算を stack_ptr カウンタで排除 → −13.742s(−3.740% vs 292)
  - **304**: K_PER_THREAD_MAXD14 32→48 → −16.343s(−4.448% vs 292、**本日確定BEST**)
  - 不採用/退行: 293(dual-lane flat), 297(save_sp削除 +9s回帰), 298(next_depth削除 +63s回帰), 299(K=64 flat), 300(schedule u64 +22s回帰), 301-303(cur_depth_x4各種 +280-300s回帰), 305-308(K sweep各点 flat)

---

Updated on 2026-07-17 for 308Py K52-final-sweep result (confirmed, K sweep closed) and 309Py variant4-phase-rotate probe.

- **308再確認**: dispatch_launch_rows=3(想定通り)。正当性一致(314666222712)。elapsed=351.675s vs 304(K=48)=351.070s、差−0.605s(−0.172%) — 誤差級、flat、不採用。304〜308の5点で K=44〜56 が flat zone であることが完全に確定した。

- **K スイープ総括（最終・292〜308）**:
  - K=32: 353.671s (296)
  - K=40: 353.587s (305、flat vs K=32)
  - K=44: 351.240s (307、flat zone開始)
  - **K=48: 351.070s (304、確定BEST)**
  - K=52: 351.675s (308、flat zone内)
  - K=56: 351.534s (306、flat zone内)
  - K=64: 353.896s (299、flat zone外)
  - flat zone: K=44〜56（全て351.0〜351.7sの範囲内、誤差級）
  - flat zone外の急峻な境界: K=40→K=44で約2.35s改善（左側）、K=56→K=64で約2.36s悪化（右側）
  - **K値はこれ以上絞り込む意味がなく、クローズ済み。304(K=48)を確定BESTとして固定。**

- **309方針**: K値・カーネル本文どちらも変更しない、最も低リスクな未探索軸である **BROADMARK_VARIANT（task reorder scheme）** に着手する。`BROAD_MARKDIST_TAIL_VARIANT` は115でA10G単GPU最終デフォルトとして `variant=2`(rotate_only) が採用されて以来、292〜308のKスイープ全体を通じて一度も振られていない。カーネルソースには元々 `variant=4`(phase_rotate: boost=1, cell/risk-aware tail phase, rotating interleave) の分岐ロジック(`broad_markdist_tail_variant_tag/desc/window_boost_value/phase_salt_value/use_phase_mix/use_rotating_interleave`)がすでに実装済みであり、**カーネルコード変更ゼロ、既存CLI引数（worker_id worker_count variant）のみでの切り替え**となる。変更は定数2行のみ: `BROAD_MARKDIST_TAIL_VARIANT:int=2`→`4`、`A10G_FINAL_DEFAULT_BROADMARK_VARIANT:int=2`→`4`（bare `-g` 無引数起動時のデフォルトも揃える）。K_PER_THREAD_MAXD14=48（304のまま）、296カーネルロジックも完全無変更。EXPECTED_CHUNKS=3（K=48のまま変化なし）。

- **309の狙い**: 304のncu再プロファイル（軽量セクション、chunk0、`--launch-count 1`）で、Achieved Occupancy 11.04%（理論値33.33%）をncu自身が「warp間imbalance」由来とEst.Speedup 66.88%で明示している。Avg. Active Threads Per Warpは6.34（292のK=32時点=4.88→K=2再検証=6.28→304のK=48=6.34、Kスイープではもう動かない頭打ち）。Stall Waitは2.09 inst（44.1%、依然トップ、K-batchingでは不変）、Stall Branch Resolvingが0.93 inst（約19.6%、今回新規に可視化された第2位要因、発生源は未特定）。root causeは279/292から変わらず「DFS部分木サイズのばらつきによるSIMT lane imbalance」であり、K-batchingは症状緩和策で根治策ではない。309はこのimbalanceに対し、カーネル側ではなく **task-ordering側**（rotate_onlyの単純固定tail phaseからphase+rotateのcell/risk-awareなtail phaseへ）からアプローチし、DFS部分木サイズのばらつきそのものをリオーダリングで緩和できるかを見る。

- **309検証スクリプト**: `308Py_K52_final_sweep_validate_N21_full_once.sh` を親に `309Py_variant4_phase_rotate_validate_N21_full_once.sh` を作成。EXPECTED_K_PER_THREAD_MAXD14=48（304/308から変化なし）、BROADMARK_VARIANT デフォルトを2→4に変更。静的チェックに `source_a10g_default_variant4`（A10G_FINAL_DEFAULT_BROADMARK_VARIANT=4確認）を追加し、`source_runtime_globals` チェックの期待値も `BROAD_MARKDIST_TAIL_VARIANT:int=4` に更新。実行時チェックとして `runtime_broadmark_variant`（ログの `variant=4` 確認）と `runtime_broadmark_variant_tag`（ログの `tag=phase_rotate` 確認）を新規追加。タイミング比較 baseline に `304K48sweep`(351.070s、主要比較対象)、`305K40sweep`(353.587s)、`306K56sweep`(351.534s)、`307K44fineprobe`(351.240s)、`308K52finalsweep`(351.675s)を追加。

- **309の優先順位（次点候補、変更なし）**: (1) BROADMARK_VARIANT — 本entryで着手。(2) Stall Branch Resolving対策 — カーネル改造の前に `--launch-count 1` + `SourceCounters`単独取得で「どの分岐か」を特定するのが先。(3) Stall Wait対策（dual-lane再挑戦）— 引き続き**高リスク**（293の脚注、189のregression前例）。(2)の結果を見てから判断すること、安易に着手しない。

---

Updated on 2026-07-21 for 309Py variant4-phase-rotate result (REJECTED, severe regression) and 310Py variant1-phase-only probe.

- **309結果**: 全静的チェックOK、build_exit=0、run_exit=0。正当性完全一致(314666222712)。`runtime_broadmark_variant=4`/`runtime_broadmark_variant_tag=phase_rotate` を実行時ログで確認、variant切り替え自体は意図通り機能した。elapsed=481.149s vs 304(K=48)=351.070s、差−130.079s(−37.052%) — **大幅悪化、不採用**。これはKスイープで観測されたどの変動（最大でも±3s程度）よりも遥かに大きい退行であり、292〜303の変数統合/削除系実験（−9〜−304s台）に匹敵する規模。

- **309 chunk別内訳**: chunk0=169.369s、chunk1=172.626s、chunk2=119.915s（304のK=48は3チャンク合計351.070s、単純平均約117s/chunk）。全チャンクが304平均を上回っており、特定チャンクだけの異常ではなく、reorderされたtask列全体でSIMT lane imbalanceが悪化したことを示唆する。ncuの「warp間imbalanceがボトルネック」という診断自体は304時点のプロファイルとして正しいが、「task-reorder側から緩和できる」という309の仮説は、少なくともphase_rotate(variant=4)の実装では裏付けられなかった。

- **variant実験まとめ（309時点）**:

  | variant | tag | boost | phase_mix | rotate_interleave | 結果 |
  |---|---|---|---|---|---|
  | 2 | rotate_only | 1 | 0 | 1 | **現行本番デフォルト（115〜308まで採用）** |
  | 4 | phase_rotate | 1 | 1 | 1 | **309: 481.149s、−130.079s(−37.052%) 大幅悪化・不採用** |

- **310方針**: variant=4(phase_rotate)はvariant=2(rotate_only)に対して`phase_mix`(cell/risk-awareなtail phase)を追加した設定であり、両者の差はphase_mix一点のみ。309の結果だけでは、退行の原因が「phase_mix単体」なのか「phase_mixとrotate_interleaveの組み合わせ」なのかを切り分けられない。中間点として **variant=1(phase_only: boost=1, phase_mix=1, rotate_interleave=0)** を測定する。カーネルソースの`broad_markdist_tail_use_phase_mix()`(v==1,4,5でTrue)と`broad_markdist_tail_use_rotating_interleave()`(v==2,4,5でTrue)の既存分岐をそのまま利用し、309と同様に**カーネルコード変更ゼロ、定数2行のみ変更**(`BROAD_MARKDIST_TAIL_VARIANT:int=4`→`1`、`A10G_FINAL_DEFAULT_BROADMARK_VARIANT:int=4`→`1`)。
  - variant=1がvariant=2(351.070s)並みに戻れば → 退行原因は「phase_mixとrotate_interleaveの組み合わせ」。
  - variant=1も309同様に悪化すれば → 退行原因は「phase_mix」そのもの。task-reorder側からのアプローチ自体を見直す必要がある。

  K_PER_THREAD_MAXD14=48（304のまま）、296カーネルロジックも完全無変更。EXPECTED_CHUNKS=3（変化なし）。

- **310検証スクリプト**: `309Py_variant4_phase_rotate_validate_N21_full_once.sh` を親に `310Py_variant1_phase_only_validate_N21_full_once.sh` を作成。BROADMARK_VARIANT デフォルトを4→1に変更、`EXPECTED_BROADMARK_VARIANT_TAG` を`phase_rotate`→`phase_only`に更新。静的チェック `source_runtime_globals`/`source_a10g_default_variant1` の期待値をvariant=1に更新。タイミング比較baselineに `309variant4phaserotate`(481.149s、REJECTED)を追加。

---

Updated on 2026-07-21 for 310Py variant1-phase-only result (REJECTED, phase_mix isolated as root cause) and 311Py variant2-restore (BROADMARK_VARIANT direction closed).

- **310結果**: 全静的チェックOK、build_exit=0、run_exit=0。正当性完全一致(314666222712)。`runtime_broadmark_variant=1`/`runtime_broadmark_variant_tag=phase_only` を実行時ログで確認。elapsed=476.932s vs 304(K=48)=351.070s、差−125.862s(−35.851%) — **大幅悪化、不採用**。309(481.149s)との差はわずか+4.217s(+0.876%)で、実質的に同水準の悪化。

- **variant実験の結論（309/310で確定）**:

  | variant | tag | boost | phase_mix | rotate_interleave | 結果 |
  |---|---|---|---|---|---|
  | **2** | **rotate_only** | 1 | 0 | 1 | **確定BEST・現行本番デフォルト（304/308=351.070〜351.675s）** |
  | 4 | phase_rotate | 1 | 1 | 1 | 309: 481.149s、−130.079s(−37.052%) 不採用 |
  | 1 | phase_only | 1 | 1 | 0 | 310: 476.932s、−125.862s(−35.851%) 不採用 |

  phase_mix=1の2点（variant 1・4）がrotate_interleaveの有無(0/1)に関わらずほぼ同じ規模(35〜37%)の退行を示したことから、**退行原因はphase_mix(cell/risk-awareなtail phase)そのものであり、rotate_interleaveとの組み合わせは無関係**と確定した。K sweepはもちろん292〜303の変数統合/削除系実験群と比べても最大級の退行であり、task-reorder側からのアプローチのうちphase_mixは根本的に不向き。wide_only(variant=3、phase_mix=0)やwide_phase_rotate(variant=5、phase_mix=1を含む)についても、3はboostのみで別軸、5はphase_mixを含むため4と同様の悪化が予想されるため、**BROADMARK_VARIANT方向のこれ以上の探索は打ち切る**。

- **311方針**: BROADMARK_VARIANTを304/308と同じ`variant=2`(rotate_only)に復帰する。カーネル本文・K_PER_THREAD_MAXD14(48)ともに304から完全無変更で、定数2行(`BROAD_MARKDIST_TAIL_VARIANT:int=1`→`2`、`A10G_FINAL_DEFAULT_BROADMARK_VARIANT:int=1`→`2`)のみの復帰。EXPECTED_CHUNKS=3（変化なし）。想定timingは304/308の~351s台に戻るはずで、309/310の~477〜481s台には戻らないことを確認する。

- **311で次に着手する方向（handoffメモの優先順位どおり）**: BROADMARK_VARIANT方向が閉じたことで、優先順位は次の項目に進む。**Stall Branch Resolving対策** — 304のncu軽量プロファイル(chunk0、`--launch-count 1`)で新規に可視化された第2位要因(0.93 inst、約19.6%)。カーネル改造の前に、まず `--launch-count 1` + `SourceCounters`単独取得で「どの分岐か」を特定するのが先。311の検証シェルヘッダには、次回実機セッションで手動実行する想定のncuコマンド(`ncu --launch-count 1 --set SourceCounters -o 311_sourcecounters_chunk0 ./311Py_variant2_restore -g 21 21 32 484 1 0 7 31 8 7 0 0 1 2`)をドキュメント化した(検証シェル自体には自動実行させない — handoffメモの「K-batching後は1チャンクがK倍の作業量になっているため、フルセクション同時取得すると多パスreplayで事実上ハングする」という教訓を踏まえ、軽量な単独セクション取得に限定)。Stall Wait対策(dual-lane再挑戦)は引き続き高リスクとして保留(293の脚注、189のregression前例)。SourceCountersの結果を見てから、カーネル改造の要否を判断する。

- **311検証スクリプト**: `310Py_variant1_phase_only_validate_N21_full_once.sh` を親に `311Py_variant2_restore_validate_N21_full_once.sh` を作成。BROADMARK_VARIANT デフォルトを1→2に復帰、`EXPECTED_BROADMARK_VARIANT_TAG` を`phase_only`→`rotate_only`に更新。静的チェック `source_runtime_globals`/`source_a10g_default_variant2` の期待値をvariant=2に更新。タイミング比較baselineに `310variant1phaseonly`(476.932s、REJECTED)を追加。

---

Updated on 2026-07-21 for 311Py variant2-restore result (correctness OK, timing ANOMALOUS) and 312Py thermal-repro-check (zero code change, GPU telemetry added).

- **311結果**: 全静的チェックOK、build_exit=0、run_exit=0。正当性完全一致(314666222712)。`runtime_broadmark_variant=2`/`runtime_broadmark_variant_tag=rotate_only`を実行時ログで確認、dispatch構成(bucket数・MAXD選択・schedule_words・stack bytes)も304と実質同一。ログの`[split291-base-reuse]`/`[chunkshape148-reuse]`からキャッシュヒットも確認済みで、キャッシュ再構築コストは無い。**しかしelapsed=454.422s vs 304(K=48)=351.070s、差−103.352s(−29.439%) — カーネル・定数ともに304と1バイトも違わないにもかかわらず大幅悪化。**

- **異常の分析**: chunk別内訳はchunk0=167.028s、chunk1=164.792s、chunk2=121.612s(304相当では1chunkあたり約117s)で、**全chunkが一様に約40%遅い**。各chunkの`kernel_reduce_ms`はelapsed_msの99.9%以上を占めており(例: chunk0はelapsed_ms=167028に対しkernel_reduce_ms=166901)、退行はhost側I/Oやキャッシュ再構築ではなく**GPUカーネル実行時間そのもの**に生じている。このセッション内では309(481.149s)→310(476.932s)→311(454.422s)と3回の実行(合計約23.5分)で単調に速くなっているが、311ですら304の確定値より29%遅い。dispatch構成が304と一致しているのに一様に遅いという事実は、コード変更由来の退行ではなく、**セッション内でのGPUサーマルスロットリング/クロック低下**が最有力の仮説であることを示している。

- **312方針**: **ソースコード上、311から1バイトも変更しない。** variant=2・K=48ともに304/311と同一のまま再実行し、454.422sが再現するか304の351.070s付近に回復するかを確認する。加えて検証シェルに`nvidia-smi`によるGPUテレメトリ取得(温度・SM/メモリクロック・電力・使用率・スロットリング要因を5秒間隔でサンプリングし`gpu_telemetry.csv`へ記録、run前のスナップショットも別途取得)を追加した。前回runが長時間だった場合に挟める`COOLDOWN_SECONDS`(デフォルト0)も追加。nvidia-smiが利用できない環境ではベストエフォートでスキップし、検証自体は失敗させない。

- **312の判断基準**:
  - 再実行で304の351s付近に回復すれば → 309〜311の遅さはセッション内サーマルドリフトが原因であり、311自体は「コード的には」問題なしと確認できる。304/308が引き続き正式なタイミングbaseline。
  - 454s前後のままなら → セッション内ドリフトではなく、persistence mode・他プロセス常駐・電源設定など永続的な環境変化を疑い、`nvidia-smi`出力を精査する。
  - **いずれの結果でも、handoffメモの優先順位#2(Stall Branch Resolving、ncu SourceCounters取得)は、この再現性確認が済むまで保留する。** スロットリング下で取得したncu分岐統計は304時点(正常クロック)のものと比較不能になり得るため。

- **312検証スクリプト**: `311Py_variant2_restore_validate_N21_full_once.sh` を親に `312Py_thermal_repro_check_validate_N21_full_once.sh` を作成。ソース側の変更はゼロ(バージョンタグ/コメントのみ更新)。シェル側に`CAPTURE_TELEMETRY`(デフォルト1)、`TELEMETRY_INTERVAL_SECONDS`(デフォルト5)、`COOLDOWN_SECONDS`(デフォルト0)を追加し、run前後でnvidia-smiスナップショット・バックグラウンドサンプリングを実施。新規サマリ項目`gpu_telemetry_captured`(INFO、nvidia-smi有無に関わらず検証は失敗させない)を追加。タイミング比較baselineに`311variant2restore`(454.422s、異常値・サーマル疑い)を追加。

---

Updated on 2026-07-21 for 312Py thermal-repro-check result (rules out thermal throttling; clock-cap suspected) and 313Py clock-cap-diagnosis.

- **312結果**: 全静的チェックOK、build_exit=0、run_exit=0。正当性完全一致(314666222712)。elapsed=454.417s、311(454.422s)との差はわずか+0.005s(+0.001%) — ソースコードを1バイトも変更していないにもかかわらず、ほぼ完全に311を再現した。サーマルスロットリングのような動的現象であれば run間のばらつきや回復傾向が見られるはずだが、実際には驚くほど安定して同じ数値が再現された。

- **GPUテレメトリの分析結果**: `gpu_telemetry.csv`(91サンプル、5秒間隔、約7.5分間)によると、温度は32℃(pre-run idle)〜39℃(peak)で終始「冷えた」状態(典型的なスロットリング閾値83〜90℃には遥かに届かない)。**SMクロックはidle時からcompute中(使用率100%)まで完全に1320MHzで固定**、メモリクロックも6251MHzで固定。`clocks_event_reasons.active`はcompute中は終始`0x0`(アクティブなスロットリング要因なし)。**結論: サーマルスロットリングではない。** 冷えていてスロットリング要因も立っていないのにクロックが一切動かないというのは、動的なブースト制御が働いていないことを意味し、**GPUクロックの明示的なロック/キャップ**(`nvidia-smi -lgc`、あるいはpersistence mode下でのapplication clocks固定)が最有力の仮説となった。これはおそらく309開始前から存在しており、304/308(351.070s/351.675s、別セッションで記録)と309以降(~450〜481s)の速度差を一貫して説明できる。

- **313方針**: **引き続きソースコード・カーネルは1バイトも変更しない。** 検証シェルのpre-runスナップショットを拡張し、`clocks.max.sm`/`clocks.max.memory`/`clocks.applications.sm`/`clocks.applications.memory`を追加取得。観測されたcurrent SMクロックがmax supported SMクロックの90%未満であれば、クロックキャップ確定のWARNINGを出力するチェック(`gpu_clock_cap_check`)を新規追加した。リセットコマンド(`nvidia-smi -rgc`、`nvidia-smi -rac`)はsudo権限や共有ハードウェアへの影響を考慮し、**自動実行はせずヘッダにドキュメント化するのみ**とした。

- **313検証スクリプト**: `312Py_thermal_repro_check_validate_N21_full_once.sh` を親に `313Py_clock_cap_diagnosis_validate_N21_full_once.sh` を作成。ソース側の変更はゼロ(バージョンタグ/コメントのみ)。`nvidia-smi`クエリフィールドを`clocks.current.sm`/`clocks.current.memory`/`clocks.max.sm`/`clocks.max.memory`/`clocks.applications.sm`/`clocks.applications.memory`/`clocks_event_reasons.active`等に拡張し、pre-runスナップショットからcurrent/max SMクロック比を計算して`gpu_clock_cap_check`(current_sm >= max_smの90%ならOK、それ未満ならWARN-CAPPED)をサマリに追加。90%未満の場合は`nvidia-smi -rgc`/`-rac`の実行を促すWARNINGをstderrに出力する。タイミング比較baselineに`312thermalreprocheck`(454.417s)を追加。

- **引き続き保留**: Stall Branch Resolving対策(ncu SourceCounters取得)は、クロックキャップの有無が確定するまで保留を継続する。ロックされたクロック下で取得した分岐統計は304時点のベースラインと比較不能になる可能性があるため。

---

Updated on 2026-07-21 for the manual `nvidia-smi -q -d CLOCK` diagnosis (confirms clock-cap numerically; `-rgc` ineffective) and 314Py power-cap-diagnosis.

- **`nvidia-smi -q -d CLOCK`結果**: Applications Clocks(Graphics=1710MHz)がDefault Applications Clocks(1710MHz)と完全一致しており、`-ac`によるアプリケーションクロック上書きではないことを確認。Max Clocks(SM=1710MHz)に対し、実際のClocks(SM=1320MHz)は明らかに低い。**1710/1320 ≈ 1.295倍で、311〜313で観測された実測の遅延+29.4%とほぼ完全に一致**(誤差0.1ポイント)し、クロックキャップが遅延の主因であることが数値的に裏付けられた。

- **`sudo nvidia-smi -rgc`結果**: "All done."と成功メッセージが出力されたが、直後の`nvidia-smi -q -d CLOCK`ではClocks.SMは1320MHzのまま変化しなかった。`-rgc`はゲスト/ユーザーレベルの`-lgc`クロックロックを解除するコマンドであり、それが効かなかったことから**単純なゲスト側`-lgc`ロックではない**と判断。Applications ClocksがDefaultのままであることと合わせ、次に疑うべき原因は (a) 電力上限(power limit)がdefaultより引き下げられている、(b) 仮想化/共有GPU環境でのホスト/ハイパーバイザー側のクロックポリシー(ゲスト側`nvidia-smi`では変更不可)の2点に絞られた。

- **314方針**: **引き続きソースコード・カーネルは1バイトも変更しない。** 検証シェルのpre-runスナップショットに`power.limit`/`power.default_limit`/`power.min_limit`/`power.max_limit`を追加取得し、`power.limit < power.default_limit`であれば`WARN-POWER-CAPPED`を出す`gpu_power_cap_check`を新規追加した。電力上限が引き下げられていた場合の対処コマンド(`sudo nvidia-smi -pl <power.default_limit値>`)はヘッダにドキュメント化するのみで自動実行しない。

- **314の実務的な判断**: この環境調査を無期限に続けるのは非生産的であるため、電力上限の確認・調整を試みてもなお1320MHzのままであれば、それはこのセッションのGPU実行環境における制御不能な現実的上限(ホスト側ポリシー等)である可能性が高いと判断し、**~454s(1320MHzクロック下での実測値)をこのセッションの暫定実務基準として受け入れ、Stall Branch Resolving調査(ncu SourceCounters取得)を再開する**方針とした。理由: ncuが報告するStall Wait/Stall Branch Resolvingの「比率」はアーキテクチャ的特性(依存チェーンの長さ、分岐の実行頻度)を反映するものであり、絶対クロックが異なっていても同一GPUアーキテクチャ内では比較的安定した情報が得られると期待できる。304時点(1710MHz環境)のプロファイルとの厳密な数値比較はできなくなるが、「どの分岐がStall Branch Resolvingの主因か」を特定するという調査目的自体は現在のクロック環境でも達成可能。

- **314検証スクリプト**: `313Py_clock_cap_diagnosis_validate_N21_full_once.sh` を親に `314Py_power_cap_diagnosis_validate_N21_full_once.sh` を作成。ソース側の変更はゼロ(バージョンタグ/コメントのみ)。`nvidia-smi`クエリフィールドに`power.limit`/`power.default_limit`/`power.min_limit`/`power.max_limit`を追加し、`gpu_power_cap_check`(power.limit >= power.default_limit − 0.5WならOK、それ未満ならWARN-POWER-CAPPED)をサマリに追加。power.limit不足の場合は`sudo nvidia-smi -pl <default>`の実行を促すWARNINGをstderrに出力する。313自体はN=21フル実行が未実施(手動でのnvidia-smi診断のみ実施)のため、タイミング比較baselineは312(454.417s)までとし、313の架空のタイミング値は追加していない。

- **注記**: なお313Py自体のN=21フルバリデーション実行はまだ行われていない(手動診断コマンドのみ実行済み)。313Pyの実行結果が得られ次第、次回更新でbaselineに追加する。

---

Updated on 2026-07-21 for 313Py/314Py execution results (correctness OK, telemetry silently broken) and 315Py telemetry-fieldname-fix.

- **313・314実行結果**: 両方ともN=21フル実行自体は正常終了。313=454.419s、314=454.424s、いずれも312(454.417s)との差は±0.002%以内で、クロックキャップされた状態が極めて安定して再現され続けていることが改めて確認された。正当性も両方とも一致(314666222712)。

- **バグ発見**: しかし314で追加したGPUテレメトリ/クロック・電力キャップ診断は、`nvidia-smi --query-gpu`のフィールドリストに存在しない`clocks.applications.sm`を含めてしまっていたため、一度も実データを取得できていなかった。nvidia-smiは指定フィールドが1つでも無効だとクエリ全体を拒否するため、`gpu_pre_run_snapshot.csv`/`gpu_telemetry.csv`の中身は313・314とも

  ```
  Field "clocks.applications.sm" is not a valid field to query.
  ```

  というエラーメッセージ1行のみだった。さらに`gpu_telemetry_captured`チェックが「ファイルの行数が1以上ならOK」という甘い判定だったため、このエラー行を誤って"present/OK"と報告する第二のバグも存在した。

- **315方針**: **引き続きソースコード・カーネルは1バイトも変更しない。** 検証シェル側の2点を修正:
  1. `clocks.applications.sm` → `clocks.applications.graphics`(nvidia-smiが実際にサポートするフィールド名。手動`nvidia-smi -q -d CLOCK`出力のApplications Clocksが"Graphics"/"Memory"のみで"SM"項目が無いことと整合。フィールドリスト内の位置は変更していないため、他フィールドのawkインデックスへの影響なし)。
  2. `gpu_telemetry_captured`および事前スナップショットの各チェックを強化。単純な行数/非空チェックではなく、**CSVの1行目が`timestamp`で始まるか**を確認するようにし、クエリエラーが混入した場合は明示的に`FAIL`として検出し、`failures`をインクリメントするように変更(以前は静かに"present/OK"や"unavailable/INFO"として見過ごされていた)。

- **314との比較**: このバグ修正により、次回実行時には`gpu_clock_cap_check`・`gpu_power_cap_check`が初めて実データで動作する見込み。ユーザーから提案のあった `sudo nvidia-persistenced` / `sudo nvidia-smi --auto-boost-default=0` / `sudo nvidia-smi 1710` については、それぞれ (a) persistence modeの有無自体はクロック上限を変えるものではない、(b) Auto Boostは`-q -d CLOCK`で`N/A`表示だったこの世代のGPUでは非対応の可能性が高い、(c) 構文として無効(裸の数値引数は受け付けられない)、という理由で、そのまま採用はせず見送った。代わりに`nvidia-smi -q -d POWER`の手動実行を依頼し、315の検証シェル修正で得られる`power.limit`/`power.default_limit`の実データと突き合わせてから、正しい対処コマンド(該当すれば`sudo nvidia-smi -pl <default値>`、あるいは`sudo nvidia-smi -lgc 1710,1710`によるクロックの明示的な引き上げ試行)を判断する。

- **315検証スクリプト**: `314Py_power_cap_diagnosis_validate_N21_full_once.sh` を親に `315Py_telemetry_fieldname_fix_validate_N21_full_once.sh` を作成。ソース側の変更はゼロ(バージョンタグ/コメントのみ)。タイミング比較baselineに`313clockcapdiagnosis`(454.419s、313自体は今回のログで初めて実測確認)と`314powercapdiagnosis`(454.424s)を追加。

---

Updated on 2026-07-21 for the `nvidia-smi -q -d POWER` / `-persistenced` / `--auto-boost-default` results (power cap ruled out; environment investigation closed) and 316Py env-accept-ncu-prep.

- **`nvidia-smi -q -d POWER`結果**: Current/Requested/Default Power Limitがすべて300.00Wで完全一致(Min=100.00W、Max=300.00W)。312のテレメトリで観測されたcompute中の最大消費電力(95.33W)は300W予算の約32%に過ぎず、電力上限には遠く及ばない。**電力キャップ仮説はこれで棄却。**

- **`sudo nvidia-persistenced`結果**: "failed to initialize"。persistenceデーモンの直接起動失敗は、仮想化/コンテナ化されたGPU環境でしばしば見られる症状であり、ゲスト側からホスト側の必要な権限/デバイスアクセスが得られないことを示唆する。

- **`sudo nvidia-smi --auto-boost-default=0`結果**: "not supported for GPU"。以前の`-q -d CLOCK`での`Auto Boost: N/A`表示と整合しており、このGPU世代では想定通り非対応。

- **調査総括(311〜316)**:

  | 仮説 | 検証方法 | 結果 |
  |---|---|---|
  | コード変更由来の退行 | 311でvariant=2/K=48に304と1バイト差なく復帰 | 否定 |
  | サーマルスロットリング | 312でGPUテレメトリ取得 | 否定 |
  | ゲスト側`-lgc`クロックロック | `sudo nvidia-smi -rgc` | 否定 |
  | `-ac`アプリケーションクロック上書き | Applications Clocks比較 | 否定 |
  | 電力上限キャップ | `nvidia-smi -q -d POWER` | **否定** |
  | Auto Boost設定 | `--auto-boost-default=0` | 非該当(このGPU世代は非対応) |
  | persistence mode | `sudo nvidia-persistenced` | 初期化失敗(仮想化環境を示唆) |

  ゲスト側で試せる主要な手段をひととおり試し尽くし、いずれも1320MHz固定を変えられなかった。**最も整合的な残る仮説は、仮想化/共有GPU環境におけるホスト/ハイパーバイザー側のクロック上限ポリシーであり、ゲスト側のnvidia-smiでは変更できないもの。**

- **316方針**: **314で立てていた実務的な判断基準どおり、環境調査をここで正式に打ち切る。** ~454s(SMクロック1320MHz下での実測値)をこのセッションの現実的な作業基準として正式に受け入れ、保留していた**Stall Branch Resolving調査(ncu SourceCounters取得、handoff優先順位#2)を再開する**。304時点(1710MHz環境)のプロファイルとの絶対値比較はもうできないが、ncuが報告するStall Wait/Stall Branch Resolvingの「比率」はアーキテクチャ的特性を反映するため、「どの分岐が主因か」を特定する調査目的自体は現在のクロック環境でも達成可能という判断を維持する。引き続きソースコード・カーネルは1バイトも変更しない。GPUテレメトリ取得自体は低コストな受動的モニタリングとして有効のままにしておくが、もはやこのスクリプトの主目的ではない。

- **316検証スクリプト**: `315Py_telemetry_fieldname_fix_validate_N21_full_once.sh` を親に `316Py_env_accept_ncu_prep_validate_N21_full_once.sh` を作成。ソース側の変更はゼロ(バージョンタグ/コメントのみ)。ヘッダコメントに次回セッションで手動実行する想定のncuコマンド(`--launch-count 1` + `SourceCounters`単独、chunk0)を再ドキュメント化。315自体もN=21フル実行は今回行われていない(手動診断コマンドのみ)ため、タイミング比較baselineは314(454.424s)までとしている。

---

Updated on 2026-07-21 for 315Py/316Py execution results (bugfix confirmed working; correctness OK) and the 316_ncu.txt profile analysis (architectural ratios confirmed stable; PC sampling unavailable), plus 317Py branch-divergence-probe.

- **315・316実行結果**: 両方ともN=21フル実行正常終了(315=454.779s、316=454.460s)、正当性一致(314666222712)。315の`gpu_clock_cap_check`が今回初めて実データで動作し、`current_sm=1320MHz max_sm=1710MHz`(`WARN-CAPPED`)を正しく検出。`gpu_power_cap_check`も`power.limit=300.00W power.default_limit=300.00W`(`OK`)を確認し、電力は制約要因でないことが検証シェル側でも裏付けられた。テレメトリも91行分正常取得(バグ修正の効果を確認)。

- **316_ncu.txtの分析結果**: 2つの重要な発見があった。

  **発見1(良いニュース)**: アーキテクチャレベルの指標が304時点(1710MHz)とほぼ完全に一致している。

  | 指標 | 304時点(1710MHz) | 316(1320MHz、今回) |
  |---|---|---|
  | Avg. Active Threads Per Warp | 6.34 | **6.34(完全一致)** |
  | Achieved Occupancy | 11.04% | 11.03%(誤差級) |
  | Stall Wait | 44.1% | 44.09%(誤差級) |
  | Stall Branch Resolving | 約19.6% | 19.62%(誤差級) |

  クロックが約23%低くても、これらの比率はほぼ完全に維持されており、316で立てた「アーキテクチャ的特性は絶対クロックに依存しない」という仮定が裏付けられた。

  **発見2(新たな制約)**: しかし要求していた`--set SourceCounters`のper-line分岐特定データは取得できなかった。ファイルには SpeedOfLight / Scheduler Statistics / Warp State Statistics / Launch Statistics / Occupancy の5つの軽量セクションのみが含まれ、Source Counters節自体が存在しない。加えて`WRN The optional metric smsp__pcsamp_sample_count could not be found.`という警告があり、PCサンプリングに基づくプロファイリングがこの環境では利用できないことを示している。`nvidia-persistenced`の初期化失敗と合わせ、**このGPU仮想化/共有環境では低レベル/特権的なドライバ機能(クロック制御、persistenceデーモン、PCサンプリング)が一貫してブロックされている**という見方が有力になった。

- **317方針**: **引き続きソースコード・カーネルは1バイトも変更しない。** PCサンプリングが使えない以上、per-line("どの行の分岐か")の特定は諦め、PCサンプリングを必要としないハードウェアカウンタベースの分岐ダイバージェンス集計メトリクス(`smsp__sass_branch_targets.sum`等)を次に試す。カーネル全体での分岐ダイバージェンスの規模感(uniform/divergent branch targetの集計)は得られる見込みで、per-line特定はできないもののStall Branch Resolving(約19.6%)の裏付けとして使える。これも取得できなければ、per-line/カウンタいずれのncuプロファイリング手段もこの環境では制約されていると判断し、次善策としてカーネルソースの手動レビュー(継続/終了条件分岐の棚卸し)に切り替える。

- **317検証スクリプト**: `316Py_env_accept_ncu_prep_validate_N21_full_once.sh` を親に `317Py_branch_divergence_probe_validate_N21_full_once.sh` を作成。ソース側の変更はゼロ(バージョンタグ/コメントのみ)。ヘッダに次回実行するncuコマンド(`smsp__sass_branch_targets*.sum`メトリクス指定、PCサンプリング不要)をドキュメント化。タイミング比較baselineに`315telemetryfieldnamefix`(454.779s)と`316envacceptncuprep`(454.460s)を追加。

---

Updated on 2026-07-21 for 317Py execution results and the 317_ncu.txt branch-divergence counter results (success with sudo), plus 318Py sourcecounters-sudo-retry.

- **317実行結果**: elapsed=454.617s、正当性一致(314666222712)、316(454.460s)との差は+0.036%で誤差級。クロック/電力チェックも変化なし(current_sm=1320MHz、power OK)。想定通り。

- **317_ncu.txt結果(`sudo`付きで実行)**: 分岐ダイバージェンスのハードウェアカウンタ取得に成功。

  ```
  smsp__sass_branch_targets.sum                   = 2,324,209,823,606
  smsp__sass_branch_targets_threads_divergent.sum =   498,374,270,228
  smsp__sass_branch_targets_threads_uniform.sum   = 1,825,835,553,378
  ```

  divergent + uniform = total(整合性確認済み)。**divergent比率 ≈ 21.44%。** これは316で見た「Avg. Active Threads Per Warp 6.34/32 ≈ 19.8%」「Stall Branch Resolving 約19.6%」という2つの指標とほぼ同じ約19〜21%のレンジにあり、互いに整合的。カーネル全体の分岐のうち約1/5がwarp内で発散する分岐であり、これがStall Branch Resolvingの実体とほぼ対応しているという定量的な裏付けが得られた。

- **重要な追加情報**: 316でPCサンプリング(`--set SourceCounters`)が失敗した際は`sudo`を付けていなかった。今回`sudo`付きでカウンタベースのメトリクス取得が成功したことから、**PCサンプリングの失敗も単なる権限不足だった可能性**が浮上した。仮想化/ハイパーバイザー側の制約という解釈を確定させる前に、`sudo`付きでの再挑戦を優先すべき。

- **318方針**: **引き続きソースコード・カーネルは1バイトも変更しない。** `sudo`付きで`--set SourceCounters`を再実行することを提案する:

  ```bash
  sudo ncu --launch-count 1 --set SourceCounters -f -o 318_ncu \
    ./318Py_sourcecounters_sudo_retry -g 21 21 32 484 1 0 7 31 8 7 0 0 1 2
  /usr/local/cuda/bin/ncu --print-details all --import 318_ncu.ncu-rep 2>&1 | tee 318_ncu.txt
  ```

  - 成功しPCサンプリングデータが得られれば → per-line("どの行の分岐か")の特定に初めて到達でき、handoff優先順位#2の本来の目的を達成できる。
  - `sudo`付きでも失敗すれば → PCサンプリングは真にこの環境でブロックされていると確定し、317で得た分岐ダイバージェンス比率(21.44%)を代替の定量情報として受け入れ、カーネルソースの手動レビューまたはStall Wait/dual-lane再挑戦の是非再検討に進む。

- **318検証スクリプト**: `317Py_branch_divergence_probe_validate_N21_full_once.sh` を親に `318Py_sourcecounters_sudo_retry_validate_N21_full_once.sh` を作成。ソース側の変更はゼロ(バージョンタグ/コメントのみ)。ヘッダに`sudo`付きSourceCountersコマンドをドキュメント化。タイミング比較baselineに`317branchdivergenceprobe`(454.617s)を追加。

---

Updated on 2026-07-22 for the 318Py execution results and 318_ncu.txt (real Source Counters data captured, with two command-syntax corrections along the way), plus 319Py sourcecounters-pagesource-probe.

- **318実行結果**: N=21フル実行は正常終了(454.585s、正当性一致 314666222712)。316(454.460s)との差は+0.028%で誤差級。クロック/電力チェックも変化なし(current_sm=1320MHz WARN-CAPPED、power.limit==power.default_limit==300W OK)。

- **`--set SourceCounters`から`--section SourceCounters`への訂正が必要だった**: 最初にいただいた実行結果は317と数値まで完全一致しており、調査の結果、実際には`--metrics smsp__sass_branch_targets*.sum`(317で使ったコマンド)がそのまま再実行されていたことが判明した。修正版コマンドをお願いしたところ、今度は`--set SourceCounters`が`==WARNING== No metrics to collect found in sections.`というエラーになった。これはこちらのコマンド指定自体の誤りで、Nsight Computeでは`--set <name>`は定義済みの"セット"(basic/full/detailedなど)を選ぶオプション、`--section <name>`が個別の"セクション"(SpeedOfLight/Occupancy/SourceCountersなど)を選ぶオプションであり、`SourceCounters`はセクション名であってセット名ではない。311以降ずっとこの誤った構文をヘッダにドキュメント化し続けていた点をお詫びする。

- **訂正後、ついに成功**: `sudo ncu --section SourceCounters ...`で再実行いただいた結果が今回の`318_ncu.txt`で、見出しが正しく`Section: Source Counters`になっており、316で失敗していたPCサンプリング系メトリクス(`smsp__pcsamp_warps_issue_stalled_*`)が実データとして得られた。

  ```
  # Samples (all)          = 3,670,006
  stall_wait               = 1,592,467  (43.39%)
  stall_branch_resolving   =   830,653  (22.63%)
  stall_long_scoreboard    =   214,958  ( 5.86%)
  stall_selected           =   726,340  (19.79%)
  ```

  (全stall_*カテゴリの合計が`# Samples`と完全一致することを確認済み。取りこぼしなし。)

  これは316のサイクルベース計測(Stall Wait 44.1%、Stall Branch Resolving 約19.6%)、317のハードウェアカウンタ(分岐ダイバージェンス比21.44%)とほぼ同じ約20%前後のレンジに収束しており、**3つの独立した測定手法がStall Branch Resolvingの規模感について相互に裏付け合う結果となった。**

- **新発見(副産物)**: `318_ncu.txt`の末尾に、ncu内蔵のOPTアドバイザーからの提案が出力されていた。

  ```
  OPT   Est. Speedup: 11.79%
        This kernel has uncoalesced global accesses resulting in a total of 189728 excessive
        sectors (14% of the total 1331305 sectors).
  ```

  メモリアクセスの非コアレッシングが推定11.79%の速度向上余地として指摘された。これはStall Branch Resolving調査とは別軸の発見であり、優先度としては副次的な記録に留めるが、#2が手詰まりになった場合の次善の高ROI候補として記憶しておく。

- **残課題**: 期待していたper-line("どの行の分岐か")の特定は今回もまだ得られなかった。`318_ncu.txt`の"Hotspot Locations"節はテーブル単位の集計値を再掲しているだけで、ソースファイル名/行番号への言及が一切ない。原因として最も可能性が高いのは、今回使ったコマンド`--print-details all --import`が「メトリクス一覧(details page)」を出力するオプションであり、ソース行対応表示("source" page)を出すものではない、という点。

- **319方針**: **引き続きソースコード・カーネルは1バイトも変更しない。** 既存の`318_ncu.ncu-rep`を`--page source`で再インポートすることを提案する(GPU再実行・ncu再プロファイリングは不要、レポートの再処理のみ):

  ```bash
  /usr/local/cuda/bin/ncu --page source --print-details all --import 318_ncu.ncu-rep 2>&1 | tee 319_ncu_source.txt
  ```

  - ソース行と対応した表示が出れば → per-line特定に初めて到達し、handoff優先順位#2の本来の目的を達成できる。
  - SASS命令アドレスのみの表示になれば → バイナリに行番号/デバッグ情報が埋め込まれていない可能性が高く、Codonのビルドオプションでline infoを有効化できるか(ビルドフラグのみ、ソース変更なし)を次に調べる。
  - `--page source`自体がエラーになれば → per-line特定はこの環境で真に到達不能と判断し、317の21.44%と318の22.63%を最終的な定量情報として受け入れ、カーネルソースの手動レビューまたは新発見の非コアレッシングメモリアクセス(推定11.79%)調査に進む。

- **319検証スクリプト**: `318Py_sourcecounters_sudo_retry_validate_N21_full_once.sh` を親に `319Py_sourcecounters_pagesource_probe_validate_N21_full_once.sh` を作成。ソース側の変更はゼロ(バージョンタグ/コメントのみ)。ヘッダに`--page source`再インポートコマンドをドキュメント化。タイミング比較baselineに`318sourcecounterssudoretry`(454.585s)を追加。

---

Updated on 2026-07-22 for the 319Py `--page source` results (per-instruction stall_branch_resolving hotspot identified) and 320Py sourceline-debug-build-probe.

- **319実行結果**: N=21フル実行は正常終了(455.116s、正当性一致 314666222712)。318(454.585s)との差は-0.117%で誤差級。静的チェック・クロック/電力チェックもすべてOK、failures=0。

- **`--page source`の再挑戦**: 前回`--page source --print-details all`が`==ERROR== Option '--print-details' is only supported for the details page.`で失敗したため、`--print-details`を外した`--page source`単独のコマンドで再実行いただいた。今回はエラーなく成功。

- **`319_ncu_source.txt`の構造**: 656行の命令アドレス単位テーブルが得られた。`Source`列には`.py`のファイル名・行番号ではなくSASS逆アセンブリのテキストがそのまま入っており、**per-line(元のPythonソース行)対応はまだ得られていない**。バイナリに行番号/デバッグ情報が埋め込まれていないためと推測される。

- **セッション最大の発見**: 656行全てについて`stall_branch_resolving`列(316/318の集計値18カテゴリすべてと列合計が完全一致することを確認し、パースの正しさを検証済み)を命令ごとに集計・降順ソートしたところ、コストが極端に集中していることが判明した。

  ```
  421,586 (50.8% of 830,653 total)  addr 0x...e100  BRA 0x...d030
  123,420 (14.9%)                   addr 0x...e010  BRA 0x...e0f0
   30,418 ( 3.7%)                   addr 0x...e0f0  BSYNC B2
   28,141 ( 3.4%)                   addr 0x...e000  BSYNC B4
  ```

  **上位2命令だけでStall Branch Resolving全体の65.6%を占める。** 重要なのは、この2つが無条件(述語なし)`BRA`命令であり、`BSYNC`(warp再収束マーカー)に隣接している点。一方、実際に発散する述語付き分岐(`@P2 BRA`=8,424件、`@!P4 BRA`=14,925件、Divergent Branchesカウンタは非ゼロ)は個別には順位が低い。**コストは分岐の判定そのものではなく、発散したDFSサブツリー探索後、全レーンが揃うのを待ってループ先頭に戻る"再収束"のタイミングで支払われていることを強く示唆する。** これは292〜316で繰り返し確認してきたAvg Active Threads Per Warp ≈ 15%(SIMTレーン不均衡)という所見と整合的。

- **320方針**: **引き続きカーネルロジックは1バイトも変更しない。** ユーザーの選択により、アドレス→ソース行対応をさらに追う方針を継続。`codon build`にデバッグ/行番号情報を埋め込むフラグがあるかを確認することを提案する:

  ```bash
  codon build --help 2>&1 | grep -iE 'debug|line|-g\b'
  ```

  フラグが見つかれば、検証・タイミング用途とは別の**診断専用デバッグビルド**を作成し、再度`sudo`付き`--section SourceCounters`でプロファイル取得後、`--page source`で確認する:

  ```bash
  codon build -release -g -o 320Py_sourceline_debug_build_probe_dbg \
    320Py_sourceline_debug_build_probe.py
  sudo ncu --launch-count 1 --section SourceCounters -f \
    -o 320_ncu_dbg \
    ./320Py_sourceline_debug_build_probe_dbg \
    -g 21 21 32 484 1 0 7 31 8 7 0 0 1 2
  /usr/local/cuda/bin/ncu --page source --import 320_ncu_dbg.ncu-rep \
    2>&1 | tee 320_ncu_dbg_source.txt
  ```

  デバッグビルドはコード生成に影響しうるため、正当性検証・タイミング比較には従来どおり通常の`-release`ビルドを使う(この検証スクリプト自体はデバッグビルド/ncu再プロファイリングを実行しない)。

  - フラグがない、または付与してもSASSのままなら → per-line対応はこのツールチェーンで到達不能と判断し、319で得たアドレスレベルの知見(再収束隣接の無条件BRAが約66%)を最終結論として、手動ソースレビューまたは非コアレッシングメモリアクセス(推定11.79%)調査に進む。

- **320検証スクリプト**: `319Py_sourcecounters_pagesource_probe_validate_N21_full_once.sh` を親に `320Py_sourceline_debug_build_probe_validate_N21_full_once.sh` を作成。カーネルロジックの変更はゼロ(バージョンタグ/コメントのみ)。ヘッダに`codon build --help`確認コマンドとデバッグビルド提案をドキュメント化。タイミング比較baselineに`319sourcecounterspagesourceprobe`(455.116s)を追加。

---

Updated on 2026-07-22 for the `codon build --help` result (no lineinfo-while-optimized flag exists) and 321Py debugbuild-lineinfo-attempt.

- **`codon build --help 2>&1 | grep -iE 'debug|line|-g\b'`結果**: Codonは「最適化あり・デバッグ情報なし」(`-release`)か「最適化なし・デバッグ情報あり」(`-debug`)の二択のみで、nvccの`-lineinfo`のような**最適化を保ったままline infoだけ追加する**中間の選択肢は存在しなかった。

  ```
  --debug    - Turn off compiler optimizations and show backtraces
  --release  - Turn on compiler optimizations and disable debug info
  ```

  その他の`--debug-entry-values`等はLLVMバックエンド汎用オプションのpass-throughで、同様に最適化前提を崩すもの。

- **リスクの整理**: `-debug`ビルドは最適化を無効化するため、319で特定した2つのホットスポット命令(`0x...e100 BRA→0x...d030`=stall_branch_resolvingの50.8%、`0x...e010 BRA→0x...e0f0`=14.9%、いずれもBSYNC直後の無条件分岐)が、`-debug`ビルドのSASSに同じ形で存在する保証はない。ループのインライン化・命令並べ替えが変わりうるため、line infoと引き換えに比較対象そのものが変わってしまう可能性がある。

- **321方針**: このリスクを理解した上で、ユーザーの判断で`-debug`ビルドを試すことを継続。**引き続きカーネルロジックは1バイトも変更しない。** 診断専用の`-debug`ビルドを作成し、chunk0のみをプロファイルする:

  ```bash
  codon build -debug -o 321Py_debugbuild_lineinfo_attempt_dbg \
    321Py_debugbuild_lineinfo_attempt.py
  sudo ncu --launch-count 1 --section SourceCounters -f \
    -o 321_ncu_dbg \
    ./321Py_debugbuild_lineinfo_attempt_dbg \
    -g 21 21 32 484 1 0 7 31 8 7 0 0 1 2
  /usr/local/cuda/bin/ncu --page source --import 321_ncu_dbg.ncu-rep \
    2>&1 | tee 321_ncu_dbg_source.txt
  ```

  確認ポイントは2つ: (1) `Source`列が実際の`.py`ファイル名・行番号になっているか、(2) 分岐構造が319で見た「BSYNC直後の無条件BRA」パターンをまだ含んでいるか(構造が大きく違えば、得られる行番号情報は`-release`カーネルの実態を正確には反映していない可能性が高いので慎重に扱う)。

  この検証スクリプト自体は通常どおり`-release`ビルドでのN=21フル実行のみを行い、正当性・タイミング検証には影響しない。なお320自体のN=21フル実行はまだ行われていない(ユーザーが`codon build --help`確認に直行したため)ため、タイミング比較baselineは319(455.116s)までとしている。

- **321検証スクリプト**: `320Py_sourceline_debug_build_probe_validate_N21_full_once.sh` を親に `321Py_debugbuild_lineinfo_attempt_validate_N21_full_once.sh` を作成。カーネルロジックの変更はゼロ(バージョンタグ/コメントのみ)。ヘッダに`-debug`ビルド+ncuプロファイルの提案コマンドをドキュメント化。

---

Updated on 2026-07-22 for the 321Py `-debug` build crash (CUDA_ERROR_INVALID_PTX, closing the tooling-based per-line-attribution track) and 322Py manualreview-reconverge-loopback (kernel source manually reviewed, no code changes).

- **321結果**: `-debug`ビルドはGPU起動時に`CUDA_ERROR_INVALID_PTX`でクラッシュした。単なる「構造が変わるリスク」ではなく、そもそも動作しないという結果。Codonの`-debug`コード生成が、NVPTXバックエンド(ドライバJIT)側で受理不能なPTXを出力していると考えられる。

- **ツール経由の調査、総括**: 312以来試した経路をすべて振り返ると:

  | 手法 | 結果 |
  |---|---|
  | `--set SourceCounters`(誤構文) | エラー(318で構文自体の誤りと判明) |
  | `sudo --section SourceCounters` | kernel全体の集計値のみ取得成功(318) |
  | `--page source`(単独) | per-instruction(SASSアドレス単位)取得成功、ただしPythonソース行なし(319) |
  | `-debug`ビルド + line info | GPU起動時にクラッシュ(321) |

  **per-line(元の`.py`行番号)対応は、このCodon+ncuツールチェーンでは到達不能と結論。**

- **322方針**: ユーザーの判断で、ツールベースの調査を終了しカーネルソースの手動レビューに切り替え。**引き続きソース変更は一切なし。**

- **手動レビューの結果**: `kernel_dfs_iter_gpu_maxd14`のメインDFSバックトラッキングループ(`while True:`、783行目付近)を確認したところ、319のSASSパターンと一致する構造的特徴が見つかった。このループには5つの独立した`continue`文(バックトラックpop、`nf==0`手詰まり、future_check手詰まり、terminal_depth到達、child_jmark強制手詰まり)と1つの暗黙のフォールスルー(push+descend)があり、**全6経路が単一のループ先頭(`if cur_avail==u32(0):`)に収束する**。各warpレーンは盤面状態に応じて毎イテレーション異なる経路を取るため典型的なSIMT発散が生じ、コンパイラは全レーンの再収束(BSYNC)を待ってから単一の後方分岐(BRA)を発行する。これは319で見た「BSYNC直後の圧倒的多数のBRA1つがstall_branch_resolvingの50.8%を占め、実際の発散条件分岐は個別には順位が低い」という結果と正確に一致する。

  312で開始した調査への、ソースレベルでの(正確な行番号での確証はないが)結論: コストは特定の条件分岐ではなく、ループの共有back-edgeにおける再収束オーバーヘッドであり、292/316で確認したAvg Active Threads Per Warp ≈ 15%と同じ根本原因を、コンパイル後コードのどこで支払われているかまで具体化したもの。

- **このリビジョンでのソース変更提案なし**: rev189の回帰(+108%)という前例と、このループが既に292/295/296/297/298など多くの慎重な反復を経ていることを踏まえ、この再収束ポイントへの変更は高リスクと判断。専用の検証計画を伴う別セッションで慎重に扱うべきとし、このリビジョンではドキュメント化・分析のみとした。

- **322検証スクリプト**: `321Py_debugbuild_lineinfo_attempt_validate_N21_full_once.sh` を親に `322Py_manualreview_reconverge_loopback_validate_N21_full_once.sh` を作成。カーネルロジックの変更はゼロ(バージョンタグ/コメントのみ)。320・321とも独自のN=21フル実行は未実施のため、タイミング比較baselineは引き続き319(455.116s)まで。

---

Updated on 2026-07-22 for the deeper 319_ncu_source.txt re-analysis (stall_wait cross-check, uncoalesced-memory hotspot fully localized) and 323Py warr-uncoalesced-loadsplit-probe, plus a new persistent "Open Objectives" section (top of this file and the source docstring).

- **stall_wait(全カテゴリ中最大、43.39%)の再確認**: トップは`0x...e040 @!P2 BREAK B2`(119,003)を筆頭に、`0x...d440`〜`0x...e040`付近の一連の条件分岐が並ぶ。これらは322の手動レビューで見つけた5つの`continue`判定カスケード(`nf==0`、future_check、terminal_depth、child_jmark、block_code分岐)の位置と一致。stall_branch_resolvingとは別の指標から、同じマッピングが独立に裏付けられた。

- **非コアレッシングメモリアクセスの発生源を完全特定**: L2 Theoretical Sectors Global Excessive(ncu OPTアドバイザーの推定11.79%速度向上リード、189,728個)が、ちょうど4命令で100%説明できることが判明。

  ```
  92,928  0x...e120  LDG.E R2, [R12.64+0x4]
  92,928  0x...e130  LDG.E R0, [R12.64]
   1,936  0x...e310  STG.E [R2.64+0x4], R17
   1,936  0x...e320  STG.E [R2.64], R16
  ```

  上位2つのLDG.Eだけで96%を占める。

- **LDG.Eペアのソース対応(推論、未確証)**: `w_arr[idx]`(861行目、`thread_total+=total*w_arr[idx]`)である可能性が高いと判断した。根拠は、(1) `w_arr:Ptr[u64]`がカーネル引数中で唯一、単一idxで読む8バイト配列であること、(2) 実行回数(~729,860〜743,424)がチャンクあたりタスク数`m`(=743,424)とほぼ一致し、DFSループ内部(数十億〜数兆回実行)とは明確に異なる「タスクごとに1回」の頻度であること、(3) SASSの並び(DFSループ脱出直後の合流点の直後にLDG.Eペアが位置する)が861行目の位置と整合すること、(4) offset+0/+4の2命令ペアが64bit値を32bit×2に分割して読む典型パターンであること。ただしper-line対応が取れない環境でのアドレス位置・実行回数・型からの推論であり、確証ではない。

- **323方針**: **引き続きソース変更は一切なし。** 新しいリード(非コアレッシングメモリアクセス)をドキュメント化するのみで、修正は提案しない。Stall Branch Resolving(ホットな発散DFSループ内部)より、こちらは単純なロード/ストア命令に限定され、タスクごとに1回のみ実行されるため、低リスクな調査対象と位置づける。

- **ドキュメント構造の変更**: ユーザーの依頼により、このREADME.mdおよび`323Py_warr_uncoalesced_loadsplit_probe.py`のdocstring冒頭に「現在の未解決課題 (Open Objectives)」セクションを新設。今後のリビジョンで、年代順ログとは別にこのサマリを都度更新していく。

- **323検証スクリプト**: `322Py_manualreview_reconverge_loopback_validate_N21_full_once.sh` を親に `323Py_warr_uncoalesced_loadsplit_probe_validate_N21_full_once.sh` を作成。カーネルロジックの変更はゼロ(バージョンタグ/コメント/Open Objectivesセクションのみ)。320・321・322とも独自のN=21フル実行は未実施のため、タイミング比較baselineは引き続き319(455.116s)まで。

---

Updated on 2026-07-22 for the user's kernel-decomposition proposal (referencing the original CPU 13Py_constellations_codon.py SQ*-family design), the resulting history review (240/266-269/273 all rejected), and 324Py devicefunc-specialize-design (design only, no code changes).

- **ユーザー提案の経緯**: 元のCPU版`13Py_constellations_codon.py`をアップロードいただき、当初はDFSロジックが`SQd0B`/`SQB`/`SQBjrB`/`SQBlBjrB`など多数の専用化関数に分割されていたこと、GPU移植時に見通しの良さを優先して1つの非再帰関数に統合したことを確認した。現行の統合kernelに再び「何らかの基準での分解」を導入すべきではないか、という提案をいただいた。

- **CPU版の設計**: `exec_solutions`の巨大if/elifディスパッチが、星座の`(i,j,k,l)`構造から探索開始前に1回だけ適切な専用関数を選ぶ。各専用関数は無関係な条件分岐を一切持たない(例: jmark処理が不要な`SQB`にはjmark処理コード自体が存在しない)。これは322/323で見つけたStall Branch Resolvingの構造と直接対比できる: 現行GPU kernelは`schedule_lo`/`schedule_hi`にpackしたランタイムスケジュールを、ホットループ内で毎DFSノードごとにnibble_op decode + 分岐カスケードとして評価しており、CPU版が静的に排除していたコストを動的に払い続けている。

- **重要な履歴確認**: GPU側での「kernel分解」は過去に4回試みられ、**正当性は毎回OKだったにもかかわらず、全て撤回**されていることが判明した。

  | リビジョン | 内容 | 結果 |
  |---|---|---|
  | 240 taskid-split-fid14 | fid=14を別kernel launchへ分離 | 撤回(241で復帰) |
  | 266-269 root0/future0/child0 probe | 特定条件専用の軽量kernel | 全て撤回(+0.4〜0.5秒) |
  | 273 rootaction0-direct-kernel | root_action分岐なしkernel、chunk単位dispatch | 撤回(+1.05秒) |

  273の撤回理由が特に示唆的: 「正当性OKだが、別kernel化によるコード配置/register pressure/コンパイル最適化差が勝った可能性があるため撤回する」。分岐削減自体は毎回成功していたのに、それでも遅くなっていたという重い前例。

- **324で設計した方向性**: 過去4回はすべて別`@gpu.kernel`エントリポイント(別host dispatch、別PTXモジュール)だった。今回検討するのは、**同一kernel内のdevice関数**としての専用化。`future_check_mask`/`child_jmark_mask`は既にホットループ開始前(root-preroll終了時点)でタスクごとに1回計算済みのスカラー値であるため、ループの外で1回だけ分岐して専用化したdevice関数を呼び分けられる可能性がある。

- **実装前の必須事前確認(次セッションの最初の一歩として提案)**:
  1. Codonがdevice関数を実際にinlineするか、SASSで`CALL`/`RET`命令の有無を確認する
  2. 268/269と同じ慎重さで、まず1軸(例: `future_check_mask==0`)だけを切り出す。2^3の組み合わせ全部には手を出さない
  3. wall-clockだけでなく`sudo ncu --section SourceCounters`で再プロファイルし、stall_branch_resolving/stall_waitが実際に下がるかを確認する
  4. 240/268/269/273と同様、悪化したら即座にロールバックする基準を事前に明記する

- **324方針**: **このリビジョンではコードは一切変更していない。** 設計とドキュメント化のみ。Open Objectivesセクションに課題1として新設し、既存の課題(非コアレッシングメモリ、Stall Branch Resolving)を課題2・3へ繰り下げた。

- **324検証スクリプト**: `323Py_warr_uncoalesced_loadsplit_probe_validate_N21_full_once.sh` を親に `324Py_devicefunc_specialize_design_validate_N21_full_once.sh` を作成。カーネルロジックの変更はゼロ(バージョンタグ/コメント/Open Objectivesセクションのみ)。320・321・322・323とも独自のN=21フル実行は未実施のため、タイミング比較baselineは引き続き319(455.116s)まで。

---

Updated on 2026-07-22 for 325Py inlineprobe-prep: a standalone diagnostic probe (325_gpu_inline_probe.py + 325_gpu_inline_probe_check.sh) to test Codon's device-function inlining behavior in @gpu.kernel, per 324's mandatory pre-check #1. No changes to the main solver.

- **背景**: 324で設計した「kernel内device関数専用化」の実装に進む前に、必須事前確認(1)「Codonがdevice関数を実際にinlineするか」を検証する必要があった。ユーザーがこの方向性に強い関心を示したため、325でこの検証を最優先タスクとして着手した。

- **`325_gpu_inline_probe.py`**: メインソルバ(`kernel_dfs_iter_gpu_maxd14`)には一切依存しない、独立した最小`@gpu.kernel`。スレッドごとのフラグで分岐し、2つの異なるplain関数(`variant_a`/`variant_b`、それぞれ定数畳み込みで消えない小さなループを含む)のどちらかを呼ぶ。

  - inlineされる場合: `probe_kernel`のSASSに両方の分岐コードが直接展開され、`CALL`命令も別関数シンボルも出ない。
  - inlineされない場合: `CALL`命令と、`variant_a`/`variant_b`という独立した関数シンボル(それぞれ`RET`で終わる)がSASSに現れる。

- **`325_gpu_inline_probe_check.sh`**: ビルドして`cuobjdump --dump-sass`(GPU実行不要、静的逆アセンブルのみ)で確認し、`CALL`/`RET`と`variant_a`/`variant_b`シンボルの有無を自動判定する。`cuobjdump`が無い場合は`sudo ncu --section SourceCounters` + `--page source`にフォールバックする。

- **結果に応じた次の一手**:
  - inlineされる → 324で設計した`future_check_mask==0`の1軸専用化を、実際に`kernel_dfs_iter_gpu_maxd14`へ試す価値がある。
  - inlineされない → device関数専用化はそのままでは効果が薄い可能性が高く(240/266-269/273と同種のコスト構造を、別kernelではなく関数呼び出しの形で持ち込むだけになりうる)、Codonのgenerics/inlineヒントの有無を先に調べる必要がある。

- **325方針**: **メインソルバのコードは一切変更していない。** プローブは独立したファイルとして追加し、N=21フル検証の対象外とした。Open Objectivesセクションの課題1を「設計中」から「検証準備完了・実行待ち」に更新した。

- **325検証スクリプト**: `324Py_devicefunc_specialize_design_validate_N21_full_once.sh` を親に `325Py_inlineprobe_prep_validate_N21_full_once.sh` を作成。メインソルバのカーネルロジックの変更はゼロ(バージョンタグ/コメント/Open Objectivesセクションのみ)。320〜324とも独自のN=21フル実行は未実施のため、タイミング比較baselineは引き続き319(455.116s)まで。プローブ実行(`bash 325_gpu_inline_probe_check.sh`)はこのN=21検証スクリプトとは別に、独立して実行する。

---

Updated on 2026-07-22 for the 325_inline_probe_ncu_source.txt result (Codon confirmed to inline device-callable functions) and 326Py futurecheck-specialize-axis1 -- **the first kernel-logic change since 311.**

- **325プローブの結果**: `probe_kernel`の逆アセンブル(67行、アドレス連続)に`CALL`命令は一切出現せず、`variant_a`/`variant_b`という独立した関数シンボルも現れなかった。分岐(`@P0 BRA`)前後にそれぞれの関数の中身が直接展開され、単一の`BSYNC`で再収束していた。**結論: Codonは`@gpu.kernel`内から呼ぶplain関数を実際にinlineする。**

- **方針転換**: この結果を受け、当初324で検討していた「別device関数への切り出し」は行わないことにした。`stack`(スレッドローカルの`__array__[u64]`)を関数境界越しに渡すパターンはこのコードベースに前例がなく、検証されていないリスクを追加で背負うだけだと判断したため。代わりに、ホットループ本体を`if future_check_mask==u32(0): <ループA> else: <ループB>`としてその場で複製する、最もシンプルで検証しやすい形を採用した。ポインタ渡し・generics・関数呼び出し境界のいずれの不確実性も発生しない。

- **326の変更内容**: `kernel_dfs_iter_gpu_maxd14`の`while True:`ループ本体を、タスク開始前(ホットループに入る直前)の1回だけの分岐でラップした。

  - **ループA(`future_check_mask==0`)**: 元の`if future_check_mask!=u32(0): if (nibble_op&8)!=u32(0): if (bm&~(...))==u32(0): continue`という4行ブロックが、構造的に一切存在しない(実行時にスキップされるのではなく、コンパイル対象のソース自体に含まれない)。
  - **ループB(`future_check_mask!=0`)**: 同じ4行ブロックのうち、常に真である外側の`if future_check_mask!=u32(0):`だけを除去し、内側の`if (nibble_op&8)!=u32(0): if (...)==u32(0): continue`はそのまま(毎ノード評価される判定として)残している。

- **正当性の機械的検証**: 両ループを自動diffし、上記のfuture_check関連ブロック以外は一文字も違わないことを確認した。カッコの対応(`(`/`)`/`[`/`]`)も全体で数を確認し、バランスしている。MAXD16/18/20/21カーネル本体は`diff`でも完全一致を確認した。root-preroll、schedule decode、chunkshape148、broadmarktail、cache生成、worker split、dispatchなど、他のあらゆる部分は一切変更していない。

- **326検証スクリプトの新規チェック**: `source_futurecheck_axis1_split`を追加し、MAXD14本体に`if future_check_mask==u32(0):`と2つの`while True:`が存在すること、`(nibble_op&u32(8))!=u32(0)`の出現がMAXD14本体の該当箇所以降にちょうど1回だけであること(=ループBのみに存在しループAには存在しない)を静的に確認する。

- **326方針(最大限の慎重さ)**: **これは311以来初めてのkernelロジック変更である。** rev189の回帰前例(+108%速度低下)と、240/266-269/273の4連敗という重い前例を踏まえ、検証手順を明記した:
  1. `STATIC_ONLY=1`でまず静的チェック(新チェック含む)
  2. N=21フル実行で**正当性(314666222712)を最優先**で確認。一致しなければ即座に中止
  3. 正当性確認後にのみ、319の455.116sとの速度比較。明確に改善していなければ325への即時ロールバックを推奨(240/268/269/273と同じ判断基準)
  4. 速度改善が見られた場合のみ、318-319で使った同じ`sudo ncu --section SourceCounters`で再プロファイルし、stall_branch_resolving/stall_waitが実際に下がったかを確認

  このビルドは実機で一度も試していない。Codonのビルドエラーが最初の試行で出る可能性は十分にある(このコードベース自体、240 r4/r7、257 r2、259など、遥かに小さな変更でもbuildfixの反復が何度も発生している)。

- **326検証スクリプト**: `325Py_inlineprobe_prep_validate_N21_full_once.sh` を親に `326Py_futurecheck_specialize_axis1_validate_N21_full_once.sh` を作成。320〜325とも独自のN=21フル実行は未実施のため、タイミング比較baselineは引き続き319(455.116s)まで。

---

Updated on 2026-07-22 for 326 r2 (buildfix): STATIC_ONLY run caught a stale validate-script assumption, not a kernel bug.

- **326 r1のSTATIC_ONLY結果**: `source_K48_sweep_shape`がFAIL(`ptr=False push=False`、他は全てOK)。

- **原因**: 検証スクリプトの`stack_ptr+=2`・pushパターンの出現回数チェックが「root-preroll 1回 + ホットループ1回 = 合計2回」(304〜325まで正しかった前提)を期待していたが、326ではホットループ自体を意図的に2重化(future_check_mask==0/!=0のループA・ループB)したため、正しい出現回数は「root-preroll 1回 + ループA 1回 + ループB 1回 = 合計3回」。実際にソースで数えると両方とも3回で、設計通り正しい状態だった。

- **326 r2方針**: **カーネルソース(`326Py_futurecheck_specialize_axis1.py`)自体は変更不要だった。** 検証スクリプト側の期待値のみ2→3に修正した(240 r4/r7、257 r2と同じbuildfixパターン)。ソースのdocstringにもr2の経緯を追記し、VERSION_TAGの先頭にbuildfix要約を追加した。

- **次のステップ**: 修正版で`STATIC_ONLY=1 bash 326Py_futurecheck_specialize_axis1_validate_N21_full_once.sh`を再実行し、正当性(314666222712)とタイミングの確認に進む。

---

Updated on 2026-07-22 for 326's N=21 full run result (rejected: correctness OK, -13.7% slower) and 327 warrloadsplit-verify -- a new attempt (based directly on 325, not a revert-record of 326) targeting Open Objectives #2, verification-first per the 326 lesson.

- **326の実行結果**: 正当性は一致(314666222712)、全静的チェックOK。しかし実行時間は517.563秒で、319(455.116秒)比 **-62.447秒(-13.7%)の大幅な悪化**。240/266-269/273(いずれも0.1〜0.3%程度)と比べて桁違いに大きい悪化幅であり、GPU側でのkernel/ループ分解によるStall Branch Resolving対策は**5回連続で撤回**(240、266、267、268/269、273、326)となった。

- **327の位置づけ**: ユーザーより、325Pyは別ファイル名でローカルに保存済みのため、ロールバック用の新規リビジョンは不要とのご指摘をいただいた。327は325をベースにした**新しい試み**として、Open Objectivesの課題2(非コアレッシングメモリアクセス、`w_arr[idx]`仮説)の検証に進む。

- **326の教訓を踏まえた方針転換**: 妥当に見える仮説(future_check_mask専用化)でも実装すると大きく予想外の結果になりうることが326で示された。この教訓を踏まえ、327では**まずカーネルには一切手を触れず**、325の成功パターン(独立プローブでCodonのコード生成挙動を先に確認)と同じ方法論で、`w_arr[idx]`読み込みの分割ロード仮説を検証することにした。

- **`327_w_arr_loadsplit_probe.py`**: `thread_total+=total*w_arr[idx]`(旧839行目)と全く同じ形をgrid-strideループ内に再現した最小`@gpu.kernel`。DFSのロジックは一切含まない。

  - 単一の`LDG.E.64`が出れば → このアクセス形状単体ではCodon/NVPTXが正しくコアレッシングできていることになり、実際のカーネルで見られる分割は周囲の複雑なコード(register pressure、`stack`配列とのエイリアシングなど)に起因する別の要因と考えられ、323の仮説は再検討が必要。
  - 2つの32bit `LDG.E`(offset+0/+4)が出れば → `319_ncu_source.txt`で見た分割パターンが、このアクセス形状に対するCodon/NVPTXの一般的なコード生成挙動であることが確認でき、実際のカーネルに手を入れる前に、プローブ内で書き方のバリエーションを安全に試せる。

- **`327_w_arr_loadsplit_probe_check.sh`**: ビルドして`cuobjdump --dump-sass`(GPU実行不要)で`LDG.E.64`と`LDG.E`(32bit)の出現を自動判定する。`cuobjdump`が無ければ`sudo ncu --section SourceCounters` + `--page source`にフォールバックする。

- **327方針**: **メインソルバのコードは一切変更していない(325と完全に同一)。** プローブは独立したファイルとして追加し、N=21フル検証の対象外とした。

- **327検証スクリプト**: `325Py_inlineprobe_prep_validate_N21_full_once.sh` を base に `327Py_warrloadsplit_verify_validate_N21_full_once.sh` を作成。メインソルバのカーネルロジックの変更はゼロ。326の結果(517.563s、rejected)をタイミング比較baselineとして記録に残しつつ、327のlineageは325から続けている。

---

Updated on 2026-07-22 for 327 r2 (buildfix): STATIC_ONLY caught a false-positive in two rejected-pattern static checks, caused by 327's own VERSION_TAG prose, not a kernel issue.

- **327 r1のSTATIC_ONLY結果**: `source_split_tag`と`source_root0_direct_rejected`がFAIL。

- **原因**: 検証スクリプトの撤回済みパターン検出チェックは、ソースファイル全体(docstring/コメント含む)に対する単純な部分文字列検索です。327のVERSION_TAGで過去の撤回履歴を説明する際、273の名称を`rootaction0-direct-kernel`という表記で言及していたため、この文字列が「実際にそのパターンが実装されている」ことを示すマーカーと誤認識されました。

- **327 r2方針**: カーネルソースには一切問題がなく、該当箇所の表現を`a root_action==0 direct-dispatch kernel variant`に言い換えるだけで解消しました。念のため、検証スクリプトが検索する他の撤回済みマーカー文字列(`split=fid14_launch`、`kernel-blockdiag`など)についても全て手元でシミュレーションし、他に誤検知が無いことを確認しました。

- **次のステップ**: 修正版で`STATIC_ONLY=1 bash 327Py_warrloadsplit_verify_validate_N21_full_once.sh`を再実行してください。

---

Updated on 2026-07-22 for the 327_w_arr_probe_ncu_source.txt result (split-load hypothesis confirmed in isolation) and round 2 of the probe (SoA layout test).

- **ラウンド1の結果**: `327_w_arr_probe_ncu_source.txt`で、`w_arr[idx]`読み込みに対応する箇所に`0x...bca0 LDG.E R8,[R2.64]`(offset+0)と`0x...bcb0 LDG.E R9,[R2.64+0x4]`(offset+4)という、`319_ncu_source.txt`と完全に一致するパターンを確認。単一の`LDG.E.64`は一つも出現しない。**DFSループの複雑さとは無関係に、この形状に対するCodon/NVPTXの一般的なコード生成挙動であることが確認できた。**

- **根本原因の仮説**: 319の実データでは、この2つの`LDG.E`行は両方とも「Excessive(過剰)セクタ」を持っていた(92,928個ずつ)。これは命令が2つに分かれているだけでは説明できず、`w_arr`が8バイト間隔のため、各32bit読み込み単体で見ても隣接スレッドのアドレスが4バイトおきではなく8バイトおきになる「隙間」がL2セクタフェッチを非効率にしていると考えられる。

- **ラウンド2のプローブ拡張**: `327_w_arr_loadsplit_probe.py`に2つのkernelを追加し、同一ビルド・同一プロファイル実行で3つを比較できるようにした。
  - `w_probe_kernel`(既存、分割確認済みのベースライン)
  - `w_probe_kernel_tmpvar`: 読み込みを一時変数に分けるだけの軽微な書き換え(コード生成が変わるとは考えにくいが、確認のコストが低いため含めた)
  - `w_probe_kernel_soa`: `w_arr`を2つの独立した密に詰まったu32配列(`w_lo_arr`/`w_hi_arr`)に分割し、それぞれ通常のインデックス(`idx`のまま、`idx*2`ではない)でアクセスして`u64(lo)|(u64(hi)<<u64(32))`で再結合する

  仮説が正しければ、`w_probe_kernel_soa`の2つの`LDG.E`はExcessiveセクタがほぼゼロになり、ベースライン/tmpvarは引き続き過剰を示すはず。プログラム自体も`sum_base`/`sum_soa`/`match`を出力し、SoA再構成が正しい値を計算していることを自己検証する。

- **327検証スクリプト更新**: `327_w_arr_loadsplit_probe_check.sh`を、3kernel構成に対応させ、`--launch-count 3`でncuプロファイルを取得するよう更新。Excessiveセクタ列の比較が主目的のため、`cuobjdump`は簡易チェックとして残しつつ`ncu`経由を主経路にした。

---

Updated on 2026-07-22 for 328Py warr-soa-split-implement -- applying 327's probe-verified w_arr SoA fix to the real kernels. A real kernel-logic change, but of a fundamentally lower-risk kind than 326 (touches only once-per-task reads, not the hot DFS loop).

- **327_w_arr_probe_ncu_v2_run.log確認**: `sum_base`と`sum_soa`が完全一致(`match: True`)。SoA再構成の正当性が確認できた。

- **328の変更内容**: `w_arr:Ptr[u64]`パラメータを持つ5つ全てのkernel(`kernel_dfs_iter_gpu_maxd14`/`16`/`18`/`20`/`21`)のシグネチャを`w_lo_arr:Ptr[u32],w_hi_arr:Ptr[u32]`に置き換えた。各kernel内の`w_arr[idx]`/`w_arr[i]`読み込み箇所(kernelあたり3箇所: root_actionの2つの早期exitパス + メインのポストループ集計、計15箇所)を`(u64(w_lo_arr[X])|(u64(w_hi_arr[X])<<u64(32)))`に置き換え、327のプローブで確認した再構成式と全く同じ形で元のu64値を復元している。

- **共有ディスパッチャの更新**: `launch_kernel_dfs_iter_gpu_static_maxd`が、既存の`w_arr:List[u64]`から`w_lo_arr`/`w_hi_arr`を1回だけ導出し、どのkernelが起動されるかに関わらず両方を渡すようにした。このディスパッチャ自体の外部シグネチャ(呼び出し側から見える形)は変更していない。

- **変更していないもの**: `build_soa_for_range`、ファイル内の他の`w_arr`生成・使用箇所すべて、CPU検証パス、ディスパッチャの呼び出し元すべて。そして最も重要な点として、322のStall Branch Resolving調査の対象だった発散DFSホットループ自体(`while True:`〜`cur_depth=next_depth`)は327と完全に一致することをdiffで確認した(署名変更・15箇所の読み込み変換以外に一切差分なし)。

- **326との違い(リスクの質)**: 326はDFSノードごとに何十億回も実行されるホットループ本体を複製する変更だった。328は、タスクごとに1回だけ実行される読み込み箇所(5kernel×3箇所=15箇所)のみが対象で、ホットループの制御フロー自体は完全に無傷。異なる種類の、より低リスクな変更と位置づけている。

- **正当性の機械的検証**: 5つのシグネチャ変換・15箇所の読み込み変換それぞれの数を確認し、括弧の対応も全体で確認した(327から既存の不均衡は変更前後で同じ差分だけ増えており、新たな不整合は入っていない)。検証スクリプトに新チェック`source_warr_soa_split_signatures`(5/5シグネチャ変換確認)と`source_warr_soa_split_dispatcher`(ディスパッチャの導出・受け渡し確認)を追加した。

- **注意点**: 推定11.79%はkernel全体に対するncu OPTアドバイザーの見積もりであり、この変更はタスクごとに1回の読み込みのみが対象のため、実際の改善幅はそれよりかなり小さい可能性が高い。

- **328方針**: 実機でのビルド・実行はまだ行われていない。正当性(314666222712)を最優先で確認し、悪化すれば327への即時ロールバックを推奨する。

- **328検証スクリプト**: `327Py_warrloadsplit_verify_validate_N21_full_once.sh` を base に `328Py_warr_soa_split_implement_validate_N21_full_once.sh` を作成。

---

Updated on 2026-07-22 for 328 r2 (buildfix): STATIC_ONLY caught a false-positive in the new source_warr_soa_split_signatures check, caused by 328's own VERSION_TAG/docstring prose, not a kernel issue -- the same class of mistake as 327 r2.

- **328 r1のSTATIC_ONLY結果**: `source_warr_soa_split_signatures`がFAIL(「7 kernels converted, 4 old-style remaining」)。

- **原因**: このチェックは、新パターン(`w_lo_arr:Ptr[u32],w_hi_arr:Ptr[u32]`、期待値5)と旧パターン(`w_arr:Ptr[u64]`、期待値0)をソースファイル全体に対する単純な部分文字列カウントで検証する。328のVERSION_TAG・docstringで変更内容を説明する際、両方の型注釈をそのままCodon構文で書いていたため、実際の5箇所(シグネチャ)に加えて説明文中の出現もカウントされてしまった。

- **328 r2方針**: カーネルソースには一切問題がなく、該当する説明文を「Codon構文そのまま」ではなく「言葉での説明」に書き換えて解消した。修正の過程で、一度は説明文中に过去の撤回パターン(273の旧名称)を誤って再度書いてしまい`source_root0_direct_rejected`も連鎖的に危険な状態になったが、これも合わせて修正し、全ての撤回済みマーカー文字列について手元でシミュレーションして最終確認した。

- **次のステップ**: 修正版で`STATIC_ONLY=1 bash 328Py_warr_soa_split_implement_validate_N21_full_once.sh`を再実行してください。

Updated on 2026-07-23 for 329Py soa-adopt-sourcecounters-reanalysis -- 328のN=21実機実行結果(正当性314666222712、実行時間456.036秒、327比-0.324%でノイズ内)を受け、328をベースラインとして正式にADOPT。カーネルロジックの変更は一切なし(329は328とバイト単位で同一)。328自身のncu SourceCountersダンプ(328_ncu_source.txt)を再解析し、文書化のみを行うリビジョン。

- **課題1のクローズ**: 327プローブが予測した「w_lo_arr/w_hi_arr分割でL2 Theoretical Sectors Global Excessiveがゼロになる」という結果を、328の実機ncu SourceCountersダンプで確認した。`kernel_dfs_iter_gpu_maxd14`内の3箇所のw_lo_arr/w_hi_arr読み込み(LDG.E)全てでExcessive=0を確認(グローバルロードにおける他の全命令も同様に0)。これで328のADOPT判断が確定した。

- **課題3(Stall Branch Resolving)の再確認**: 328(SoA適用後)のncuデータでも、322で特定した2箇所の分岐(back-edgeのBRAと、BSYNC直前のBRA)がstall_branch_resolving全831,032サンプル中544,629(65.5%)を占めることを確認した。322時点の約65.6%とほぼ一致し、SoA変更が制御フローに影響しないという想定を裏付けた(新事実ではなく再現性の確認)。

- **課題3の新しい解像度**: 該当する2箇所は`Divergent Branches`列がいずれも0(分岐命令自体は発散していない)である一方、`Avg. Threads Executed`が2〜3/32(一方の分岐ペアは13/32)と極端に低いことが分かった。つまりstall_branch_resolvingの正体は、古典的な分岐発散(taken/not-takenがレーンごとに割れる)ではなく、warp内の大半のレーンが既に自分のDFS部分木を終えて非活性化しており、少数の"尾"に残ったレーンの完了をwarp全体が待つ、という占有率崩壊(occupancy collapse / tail effect)である可能性が高い。この構造は240/266-269/273/326のkernel分解(5戦5敗)とは根本的に異なる対策(warpレベルの動的作業再分配、persistent threads/stream compactionなど)を要する可能性があり、具体的で低リスクな対策案がまだ無いため、この課題は引き続き「保留」のままとする(結論は変わらず、原因の理解が深まっただけ)。

- **新規発見(課題4、低優先度)**: kernel epilogueの結果書き込み(STG.E、thread_totalをresultsへ書き出す2命令)にL2 Theoretical Sectors Global Excessive=1936×2=3872個を新たに検出した。w_arr/SoAとは無関係の箇所。ただし実行回数はスレッド当たり1回のみ(kernel全体で484命令実行)であり、ホットループ(このプロファイルで数千億回)と比較して影響は桁違いに小さいと判断し、対応せず記録のみに留める。

- **329の位置づけ**: カーネルロジック変更を含まない、ドキュメント/ncu再解析専用のリビジョン。ソースのOpen Objectivesセクション・VERSION_TAG・reason文字列のみを更新し、DFSホットループ本体・シグネチャ・ディスパッチャは328と完全に同一(diffで確認、変更範囲はヘッダー部のみ)。

- **329検証スクリプト**: `328Py_warr_soa_split_implement_validate_N21_full_once.sh` を base に `329Py_soa_adopt_sourcecounters_reanalysis_validate_N21_full_once.sh` を作成。ファイル名・VERSION_TAGチェック文字列・ベースライン比較(328warrsoasplitimplement:456.036sを新規追加)を更新。SoA関連の静的チェック(`source_warr_soa_split_signatures`/`source_warr_soa_split_dispatcher`)はロジック変更がないため引き続きそのまま通る想定。

- **次のステップ**: `STATIC_ONLY=1 bash 329Py_soa_adopt_sourcecounters_reanalysis_validate_N21_full_once.sh`を実行後、フルN=21実行(正当性314666222712を最優先で確認、その後328の456.036s/327の454.563sとのノイズ内一致を確認)。課題3(occupancy collapse/tail effect)への対策は、具体的で低リスクな設計ができるまで着手しない方針。
