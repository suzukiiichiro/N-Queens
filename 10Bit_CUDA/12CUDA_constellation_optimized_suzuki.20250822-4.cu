/**
  10CUDA_constellation_warp.cu複写
  21Py_constellations_optimized_codon.pyを移植

これまで実装した最適化
✅[Opt-01]    ビット演算枝刈り        全探索・部分盤面生成のすべてでbit演算徹底
✅[Opt-02]    左右対称性除去（左半分探索）        初手左半分/コーナー分岐で重複生成排除
✅[Opt-03]    中央列特別処理（奇数N）        奇数N中央列を専用内包表記で排除
✅[Opt-04]    180°対称除去        rot180_in_set で内包時点で重複除去
✅[Opt-05]    角位置分岐・COUNT分類        コーナー分岐/symmetryでCOUNT2/4/8分類
✅[Opt-06]    並列処理（初手分割）    未（現状は未実装）    これは現状は未実装 27Py_で実装

board変数にrowの情報を格納していないので対応不可
[Opt-07]    1行目以外の部分対称除去        jasmin/is_partial_canonicalで排除
[Opt-08]    軽量is_canonical・キャッシュ        Zobrist/jasmin/hash系でメモ化
[Opt-09]    Zobrist Hash        Zobrist導入済
[Opt-10]    マクロチェス（局所パターン）        violate_macro_patterns関数（導入済ならOK）

✅[Opt-11]    ミラー+90°回転重複排除 原則不要「あえてやらない」設計。必要ならis_canonicalで激重に
✅[Opt-12]ビット演算のインライン化
[Opt-13]    キャッシュ構造設計
  1. Jasmin変換キャッシュ
  2. 星座生成（サブコンステレーション）キャッシュ
  3. 星座（盤面）一意性管理
✅[Opt-14]    バックトラック関数の修正

✅ カーネル内の return によるデッドロック対処：早期 return を廃止して「cnt=0 をセットして最後の縮約だけ一緒に通過」
✅ free の初期化で mask を当てていない対処：free = mask & ~(ld | rd | col);
✅符号付き int でのビット操作 ✅ 対処（最小）：ビット系は unsigned int
✅LD/RDのビット式の括弧 ✅ 対処：意図を明示して 1u << (N-1-j) の形に。以後すべて 1u を使う。
✅デバイス再帰のスタック不足 size_t bytes = 1<<20; // 1MB など様子見 cudaDeviceSetLimit(cudaLimitStackSize, bytes);





=========================================
[Opt-13]  キャッシュ構造設計
=========================================
1. Jasmin変換キャッシュ
効果： 盤面の標準形（jasmin変換後）を使う場面で「同じ盤面を何度も正規化しない」ので、計算量を1/数倍〜10倍に削減。
現場ポイント：
正規化変換は枝刈り・重複除去の基盤なので、ここでのキャッシュは盤面生成・探索全体の高速化に直結。
辞書キャッシュはヒット率が高く、Nが大きくなるほど効果増大。

2. 星座生成（サブコンステレーション）キャッシュ
効果： 部分盤面（プリセットクイーン配置）など、同じ状態に再帰的到達した時に再生成・再計算を防げるので計算量を劇的に圧縮できる。
現場ポイント：
tuple key管理により「全く同じ状態は1度しか分岐しない」＝枝刈り×メモ化の強力ハイブリッド。
再帰呼び出し数が指数的に減るケースもあり、特に大Nや部分盤面生成で効果絶大。

3. 星座（盤面）一意性管理
効果： 星座（コンステレーション）の重複追加・多重登録を完全に防ぎ、リスト管理が爆発的に大きくなるのを防げる。
現場ポイント：
盤面「signature（ハッシュ）」でセット管理し、リスト追加前に存在チェック。
コンステレーション単位でのキャッシュ・集合管理は「メモリ節約・ダブり計算ゼロ・uniqueな探索」に直結。
応用： CUDA化/並列化のときも「unique集合化」は別スレッド間で衝突なしで設計できる


=========================================
[Opt-14]  バックトラック関数の修正
=========================================
◆ CUDA N-Queens さらなる高速化テクニック集
1. バックトラック関数の「ループと再帰」徹底最適化
不要な変数の排除
bit, next_ld, next_rd, next_col, next_free などをローカル変数で1行ずつ直書き。
不要な一時変数、複数回使わない変数は都度インラインで。
whileループの中で「free -= bit = free & -free」ワンライナー化。
再帰呼び出しの「return値」や「int型の加算」も極力インライン
合計カウントはintの加算のみでシンプルに

2. 変数に「register」修飾子を付ける
CUDA（特に古い環境やsm_アーキテクチャ）では、計算中に頻繁にアクセスされる変数はregisterを明示するとレジスタ利用が最適化されやすい
register register int bit;
register int free;
注意：現代のCUDAではregister指定が自動最適化されることも多いですが、明示でヒントを与えるのは有効な場合もある（最適化でレジスタ溢れしない場合）

3. __forceinline__指定でインライン展開を強制
再帰関数や小さな関数は__forceinline__指定

c
コピーする
編集する
__device__ __forceinline__ int SQd1B(...) { ... }
これで関数呼び出しオーバーヘッドをカットできる
特に深い再帰・頻繁な小関数コールの箇所は効果大。

4. if分岐を“条件式”でまとめる（条件分岐の予測ミス削減）
if(...) return ...; より res += (条件) * (値); で「分岐無し」の合成型にする

ただし、複雑になりすぎると可読性は低下するのでバランスをみて。

5. 定数やマスク値は事前計算・共有メモリ（shared）活用
たとえばmaskや、頻繁に使う(1 << N) - 1は__shared__変数でスレッド間共有

start, end, symmetryの決定ルールも先に配列にまとめる

6. カーネル内ループのアンローリング（loop unrolling）
再帰をループ化 or 明示的にアンローリング

c
コピーする
編集する
#pragma unroll 4
for(...) { ... }
whileループが短い（回数が小さい）場合は有効

7. bit演算を“無駄なく”徹底する
(free & -free) のような「最下位ビット取得」は超速い

再計算の重複を無くす

たとえば(ld|bit)<<1をnext_ldにだけ使い、他では再利用

8. 「分岐の無駄」カット
「再帰先がゼロ（next_free==0）」の判定はif(!next_free) continue;ではなく
while(avail) {...}の中でif(next_free){...}で1段だけ分岐する

9. shared memoryでwarpごとsumを高速集計
既にやっているが、reduceパターンの同期をできるだけ「warp内同期」に

CUDA11以降なら__syncwarp()なども使える

sum[tid]のreduceは最短経路でまとめる

10. CUDA Occupancy最大化（スレッド/ブロック数チューニング）
THREAD_NUM（blockDim.x）の調整

sm_xx世代で「最適なwarp数（32, 64, 128）」やblock数でGPU利用率最大化を

cudaOccupancyMaxPotentialBlockSize()で計測も

11. その他マイクロ最適化
LD, RD, COLの左シフト・右シフトの組み合わせをできるだけまとめて計算

symmetry分類（COUNT2/4/8）や、rot90/rot180計算も#defineマクロでインライン化

cudaMemcpyの回数最小化、カーネル実行単位でできるだけ大きなbatchで投げる

◆ 最後の一歩
ベンチマークを取って、1つ1つを順に導入 → 差分計測 → 効果を測るのが王道です！
「明らかに効果が見えない箇所」「registerを付けても効果が0な場合」は潔く消すのが現代の作法です。


NQueens_suzuki$ nvcc -O3 -arch=sm_61 -m64 -prec-div=false 12CUDA_constellation_optimized.cu && POCL_DEBUG=all && ./a.out -g
ptxas warning : Stack size for entry function '_Z19execSolutionsKernelP13ConstellationPjii' cannot be statically determined
GPU Constellations
 N:        Total      Unique      dd:hh:mm:ss.ms
 4:                0               0     000:00:00:00.40
 5:               18               0     000:00:00:00.00
 6:                4               0     000:00:00:00.00
 7:               40               0     000:00:00:00.00
 8:               92               0     000:00:00:00.00
 9:              352               0     000:00:00:00.00
10:              724               0     000:00:00:00.00
11:             2680               0     000:00:00:00.00
12:            14200               0     000:00:00:00.00
13:            73712               0     000:00:00:00.00
14:           365596               0     000:00:00:00.02
15:          2279184               0     000:00:00:00.09
16:         14772512               0     000:00:00:00.44
17:         95815104               0     000:00:00:03.93
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define INITIAL_CAPACITY 1000
#define presetQueens 4
#define THREAD_NUM		96
/**
  Constellation構造体の定義
*/
typedef struct{
  int id;
  int ld;
  int rd;
  int col;
  int startijkl;
  long solutions;
}Constellation;
/**
  IntHashSet構造体の定義
*/
typedef struct{
  int* data;
  int size;
  int capacity;
}IntHashSet;
/**
  ConstellationArrayList構造体の定義
*/
typedef struct{
  Constellation* data;
  int size;
  int capacity;
}ConstellationArrayList;
/**
 * 関数プロトタイプ
 */
__host__ __device__ unsigned int SQBkBlBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQBklBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQBlBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQBlBkBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQBlkBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQBlkBjrB_iter(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row, unsigned int free,unsigned int jmark,unsigned int endmark, unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQBkBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQBjlBkBlBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQBjlBklBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQBjlBlBkBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQBjlBlkBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQd2BkBlB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQd2BklB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQd2BlB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQd2B(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQd2BlBkB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQd2BlkB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQd1BkBlB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQd1BklB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQd1BlB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQd1B(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQd1BlBkB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQd1BlkB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQd0B(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQd0BkB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQd2BkB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
__host__ __device__ unsigned int SQd1BkB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int start,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);
/**
 * 盤面ユーティリティ群（ビットパック式盤面インデックス変換）
 *
 * Python実装のgeti/getj/getk/getl/toijklに対応。
 *
 * [i, j, k, l] 各クイーンの位置情報を5ビットずつ整数値（ijkl）にパック／アンパックするためのマクロ。
 * 15ビット～0ビットまでに [i|j|k|l] を格納する設計で、constellationのsignatureや
 * 回転・ミラー等の盤面操作を高速化する。
 *
 * 例：
 *   - geti(ijkl): 上位5ビット（15-19）からiインデックスを取り出す
 *   - toijkl(i, j, k, l): 各値を5ビット単位で連結し一意な整数値（signature）に変換
 *
 * [注意] N≦32 まで対応可能
 */
#define geti(ijkl) ( (ijkl>>15)&0x1F )
#define getj(ijkl) ( (ijkl>>10) &0x1F )
#define getk(ijkl) ( (ijkl>>5) &0x1F )
#define getl(ijkl) ( ijkl &0x1F )
#define toijkl(i,j,k,l) ( ((i)<<15)|((j)<<10)|((k)<<5)|(l) )
/**
  時計回りに90度回転
  rot90 メソッドは、90度の右回転（時計回り）を行います
  元の位置 (row,col) が、回転後の位置 (col,N-1-row) になります。
*/
#define rot90(ijkl,N) ( ((N-1-getk(ijkl))<<15) | ((N-1-getl(ijkl))<<10) | (getj(ijkl)<<5) | geti(ijkl) )
/**
  対称性のための計算と、ijklを扱うためのヘルパー関数。
  開始コンステレーションが回転90に対して対称である場合
*/
#define rot180(ijkl,N) ( ((N-1-getj(ijkl))<<15) | ((N-1-geti(ijkl))<<10) | ((N-1-getl(ijkl))<<5) | (N-1-getk(ijkl)) )
#define symmetry90(ijkl,N)( ((geti(ijkl)<<15) | (getj(ijkl)<<10) | (getk(ijkl)<<5) | getl(ijkl)) == ((N-1-getk(ijkl))<<15 | (N-1-getl(ijkl))<<10 | (getj(ijkl)<<5) | geti(ijkl)) )
/**
  symmetry: 回転・ミラー対称性ごとの重複補正 (90度:2, 180度:4, その他:8)
*/
#define symmetry(ijkl,N) ( (geti(ijkl)==N-1-getj(ijkl) && getk(ijkl)==N-1-getl(ijkl)) ? (symmetry90(ijkl,N) ? 2 : 4 ) : 8 )
/**
  左右のミラー 与えられたクイーンの配置を左右ミラーリングします。
  各クイーンの位置を取得し、列インデックスを N-1 から引いた位置に変更します（左右反転）。
  行インデックスはそのままにします。
*/
#define mirvert(ijkl,N) ( toijkl(N-1-geti(ijkl),N-1-getj(ijkl),getl(ijkl),getk(ijkl)) )
/**
 * 大小を比較して小さい最値を返却
 */
#define ffmin(a,b)(a<b ? a : b)
/**
  i,j,k,lをijklに変換し、特定のエントリーを取得する関数
  各クイーンの位置を取得し、最も左上に近い位置を見つけます
  最小の値を持つクイーンを基準に回転とミラーリングを行い、配置を最も左上に近い標準形に変換します。
  最小値を持つクイーンの位置を最下行に移動させる
  i は最初の行（上端） 90度回転2回
  j は最後の行（下端） 90度回転0回
  k は最初の列（左端） 90度回転3回
  l は最後の列（右端） 90度回転1回
  優先順位が l>k>i>j の理由は？
  l は右端の列に位置するため、その位置を基準に回転させることで、配置を最も標準形に近づけることができます。
  k は左端の列に位置しますが、l ほど標準形に寄せる影響が大きくないため、次に優先されます。
  i は上端の行に位置するため、行の位置を基準にするよりも列の位置を基準にする方が配置の標準化に効果的です。
  j は下端の行に位置するため、優先順位が最も低くなります。
*/

__host__ __device__ uint32_t jasmin(uint32_t ijkl, int N) 
{
  int arg=0;
  int min_val=ffmin(getj(ijkl), N-1-getj(ijkl));
  if (ffmin(geti(ijkl), N-1-geti(ijkl)) < min_val) {
    arg=2; min_val=ffmin(geti(ijkl), N-1-geti(ijkl));
  }
  if (ffmin(getk(ijkl), N-1-getk(ijkl)) < min_val) {
    arg=3; min_val=ffmin(getk(ijkl), N-1-getk(ijkl));
  }
  if (ffmin(getl(ijkl), N-1-getl(ijkl)) < min_val) {
    arg=1; min_val=ffmin(getl(ijkl), N-1-getl(ijkl));
  }
  for(unsigned int i=0; i < arg; ++i) ijkl=rot90(ijkl, N);
  if (getj(ijkl) < N-1-getj(ijkl)) ijkl=mirvert(ijkl, N);
  return ijkl;
}
/**
  CUDA 初期化
  */
bool InitCUDA()
{
  int count;
  cudaGetDeviceCount(&count);
  if(count==0){fprintf(stderr,"There is no device.\n");return false;}
  int i;
  for(i=0;i<count;i++){
    struct cudaDeviceProp prop;
    if(cudaGetDeviceProperties(&prop,i)==cudaSuccess){if(prop.major>=1){break;} }
  }
  if(i==count){fprintf(stderr,"There is no device supporting CUDA 1.x.\n");return false;}
  cudaSetDevice(i);
  return true;
}
/**
 * IntHashSet構造体のインスタンスを生成し、初期化する関数
 * @return 初期化済みのIntHashSetへのポインタ
 *         （data配列はINITIAL_CAPACITYで確保、size=0, capacity=INITIAL_CAPACITY）
 *         使用後は free_int_hashset() で解放すること
 */
IntHashSet* create_int_hashset()
{
  IntHashSet* set=(IntHashSet*)malloc(sizeof(IntHashSet));
  set->data=(int*)malloc(INITIAL_CAPACITY * sizeof(int));
  set->size=0;
  set->capacity=INITIAL_CAPACITY;
  return set;
}
/**
 * IntHashSet構造体が確保したメモリ領域を解放する関数
 * @param set 解放対象のIntHashSetポインタ
 *        （内部のdata配列と構造体本体をfreeする。多重freeに注意）
 */
void free_int_hashset(IntHashSet* set)
{
  free(set->data);
  free(set);
}
/**
 * IntHashSet内に指定した値が含まれているかを線形探索で判定する関数
 * @param set 探索対象のIntHashSetポインタ
 * @param value 判定したい整数値
 * @return 1: 含まれる / 0: 含まれない
 */
int int_hashset_contains(IntHashSet* set,unsigned int value)
{
  for(unsigned int i=0;i<set->size;i++){
    if(set->data[i]==value){ return 1; }
  }
  return 0;
}
/**
 * IntHashSetに指定した値を追加する関数
 * @param set 追加先のIntHashSetポインタ
 * @param value 追加したい整数値
 * @details
 *   既に同じ値が含まれている場合は何もしない（重複不可）。
 *   data配列が満杯の場合は容量を2倍に拡張（realloc）。
 */
void int_hashset_add(IntHashSet* set,int value)
{
  if(!int_hashset_contains(set,value)){
    if(set->size==set->capacity){
      set->capacity *= 2;
      set->data=(int*)realloc(set->data,set->capacity * sizeof(int));
    }
    set->data[set->size++]=value;
  }
}
/**
 * ConstellationArrayList構造体のインスタンスを生成し、初期化する関数
 * @return 初期化済みのConstellationArrayListへのポインタ
 *         （data配列はINITIAL_CAPACITYで確保、size=0, capacity=INITIAL_CAPACITY）
 *         使用後は free_constellation_arraylist() で解放すること
 */
ConstellationArrayList* create_constellation_arraylist()
{
  ConstellationArrayList* list=(ConstellationArrayList*)malloc(sizeof(ConstellationArrayList));
  list->data=(Constellation*)malloc(INITIAL_CAPACITY * sizeof(Constellation));
  list->size=0;
  list->capacity=INITIAL_CAPACITY;
  return list;
}
/**
 * ConstellationArrayList構造体と、その内部のdata配列を解放する関数
 * @param list 解放対象のConstellationArrayListポインタ
 * @note
 *   内部配列も本体もfreeされるので、多重解放に注意。
 */
void free_constellation_arraylist(ConstellationArrayList* list)
{
  free(list->data);
  free(list);
}
/**
 * ConstellationArrayListに要素を追加する関数
 * @param list 追加先のConstellationArrayListポインタ
 * @param value 追加するConstellation構造体
 * @details
 *   配列が満杯のときは容量を2倍に拡張（realloc）してから追加。
 */
void constellation_arraylist_add(ConstellationArrayList* list,Constellation value)
{
  if(list->size==list->capacity){
    list->capacity *= 2;
    list->data=(Constellation*)realloc(list->data,list->capacity * sizeof(Constellation));
  }
  list->data[list->size++]=value;
}
/**
 * すべてのフィールドを初期化したConstellation構造体のインスタンスを生成する関数
 * @return 初期化済みのConstellationへのポインタ（id,ld,rd,col,startijkl=0, solutions=-1）
 *         使用後はfree()でメモリ解放が必要
 */
Constellation* create_constellation()
{
  Constellation* new_constellation=(Constellation*)malloc(sizeof(Constellation));
  if(new_constellation){
    new_constellation->id=0;
    new_constellation->ld=0;
    new_constellation->rd=0;
    new_constellation->col=0;
    new_constellation->startijkl=0;
    new_constellation->solutions=-1;
  }
  return new_constellation;
}
/**
 * 引数で指定した値でConstellation構造体を生成・初期化する関数
 * @param id, ld, rd, col, startijkl, solutions 各フィールドにセットする値
 * @return フィールドがセット済みのConstellationへのポインタ
 *         使用後はfree()でメモリ解放が必要
 */
Constellation* create_constellation_with_values(int id,int ld,int rd,int col,int startijkl,long solutions)
{
  Constellation* new_constellation=(Constellation*)malloc(sizeof(Constellation));
  if(new_constellation){
    new_constellation->id=id;
    new_constellation->ld=ld;
    new_constellation->rd=rd;
    new_constellation->col=col;
    new_constellation->startijkl=startijkl;
    new_constellation->solutions=solutions;
  }
  return new_constellation;
}
/**
 * 指定したビットマスク・signatureからConstellationを生成し、リストに追加する関数
 * @param ld   クイーン配置の左斜め方向のビットマスク
 * @param rd   クイーン配置の右斜め方向のビットマスク
 * @param col  クイーン配置の縦方向のビットマスク
 * @param startijkl  盤面のsignature値
 * @param constellations  追加先のConstellationArrayListポインタ
 * @details
 *   solutionsフィールドは-1で初期化される。値はコピーされて配列に追加される。
 */
void add_constellation(int ld,int rd,int col,int startijkl,ConstellationArrayList* constellations)
{
  Constellation new_constellation={0,ld,rd,col,startijkl,-1};
  constellation_arraylist_add(constellations,new_constellation);
}
/**
 * Constellation構造体のstartijklの下位15ビット（jkl値）で昇順ソートするための比較関数
 * @param a 比較対象1（Constellation*へのvoid*）
 * @param b 比較対象2（Constellation*へのvoid*）
 * @return -1: a < b / 1: a > b / 0: 等しい
 * @details
 *   qsort等で使うことを想定。jkl値のみで比較する。
 */
int compareConstellations(const void* a, const void* b)
{
  Constellation* const1 = (Constellation*)a;
  Constellation* const2 = (Constellation*)b;
  // startijkl の最初の 15 ビットを取得
  unsigned int jkl1 = const1->startijkl & ((1 << 15) - 1);
  unsigned int jkl2 = const2->startijkl & ((1 << 15) - 1);
  // jkl に基づいてソート
  if (jkl1 < jkl2) {
      return -1;
  } else if (jkl1 > jkl2) {
      return 1;
  } else {
      return 0;
  }
}
/**
 * ConstellationArrayListのデータを、startijklの下位15ビット（jkl値）で昇順ソートする関数
 * @param constellations ソート対象のConstellationArrayListポインタ
 * @details
 *   比較関数 compareConstellations() を用いてqsortでソートされる。
 *   盤面のsignature（jkl値）でグルーピングや重複排除等を行う前処理にも使える。
 */
void sortConstellations(ConstellationArrayList* constellations) 
{
    // qsort を使ってソート
    qsort(constellations->data, constellations->size, sizeof(Constellation), compareConstellations);
}
/**
 * トラッシュ（無効・削除予定）用のダミーConstellationをリストに追加する関数
 * @param list 追加先のConstellationArrayListポインタ
 * @param ijkl トラッシュマーク対象の盤面signature値（下位ビット）
 * @details
 *   ld/rd/colを-1で埋め、startijklは(69<<20)|ijklとすることで
 *   「本来の探索対象ではない」ことを明示。探索・計数から除外したい時の管理用に利用。
 */
void addTrashConstellation(ConstellationArrayList* list,unsigned int ijkl) 
{
  // トラッシュ用のダミーコンステレーションを作成
  int ld = -1;
  int rd = -1;
  int col = -1;
  // 「69<<20」は“magic number”であり、通常のstartijklとは重複しない特殊値として扱う
  unsigned int startijkl = (69 << 20) | ijkl;
  // トラッシュコンステレーションをリストに追加
  add_constellation(ld, rd, col, startijkl, list);
}
/**
 * ConstellationArrayListをworkgroupSizeの倍数で区切るため、各グループ末尾に
 * トラッシュ（無効ダミー）コンステレーションを追加してリスト長を調整する関数
 *
 * @param constellations 入力となるConstellationArrayList（ソート済みを期待）
 * @param workgroupSize   1グループのスレッド数（CUDAのblockDimなど）
 * @return workgroupSizeの倍数にパディング済みの新ConstellationArrayList
 *
 * @details
 *   - 各startijkl（下位15ビット単位）のグループごとに、リスト長がworkgroupSizeで割り切れるまで
 *     addTrashConstellation() でダミーを追加。
 *   - 最後のグループも同様にパディング。
 *   - すでにsolutions>=0（解が既知）の要素は追加しない。
 *   - CUDAカーネルで「warp/block単位での等分散」に必須の前処理。
 */
ConstellationArrayList* fillWithTrash(ConstellationArrayList* constellations,unsigned int workgroupSize) 
{
  sortConstellations(constellations); // コンステレーションのリストをソート
  ConstellationArrayList* newConstellations = create_constellation_arraylist();// 新しいリストを作成
  unsigned int currentJkl = constellations->data[0].startijkl & ((1 << 15) - 1); // 最初のコンステレーションの currentJkl を取得
  for(unsigned int i = 0; i < constellations->size; i++) { // 各コンステレーションに対してループ
    Constellation c = constellations->data[i];
    if (c.solutions >= 0) continue;// 既にソリューションがあるものは無視
    if ((c.startijkl & ((1 << 15) - 1)) != currentJkl) { // 新しい ijkl グループの開始を確認
      while (newConstellations->size % workgroupSize != 0) { // workgroupSize の倍数になるまでトラッシュを追加
        addTrashConstellation(newConstellations, currentJkl);
      }
      currentJkl = c.startijkl & ((1 << 15) - 1);
    }
    add_constellation(c.ld, c.rd, c.col, c.startijkl, newConstellations);// コンステレーションを追加
  }
  while (newConstellations->size % workgroupSize != 0) { // 最後に残った分を埋める
    addTrashConstellation(newConstellations, currentJkl);
  }
  return newConstellations;
}
/**
 * 開始コンステレーション（部分盤面）の生成関数
 *
 * N-Queens探索の初期状態を最適化するため、3つまたは4つのクイーン（presetQueens）を
 * あらかじめ盤面に配置した全ての部分盤面（サブコンステレーション）を列挙・生成する。
 * 再帰的に呼び出され、各行ごとに可能な配置をすべて検証。
 *
 * @param ld   左対角線のビットマスク（既にクイーンがある位置は1）
 * @param rd   右対角線のビットマスク
 * @param col  縦方向（列）のビットマスク
 * @param k    事前にクイーンを必ず置く行のインデックス1
 * @param l    事前にクイーンを必ず置く行のインデックス2
 * @param row  現在の再帰探索行
 * @param queens 現在までに盤面に配置済みのクイーン数
 * @param LD/RD 探索初期状態用のマスク（使用例次第で追記）
 * @param counter 生成されたコンステレーション数を書き込むカウンタ
 * @param constellations 生成したコンステレーション（部分盤面配置）のリスト
 * @param N     盤面サイズ
 * @details
 *   - row==k/lの場合は必ずクイーンを配置し次の行へ進む
 *   - queens==presetQueensに到達したら、現時点の盤面状態をコンステレーションとして記録
 *   - その他の行では、空いている位置すべてにクイーンを順次試し、再帰的に全列挙
 *   - 生成された部分盤面は、対称性除去・探索分割等の高速化に用いる
 */
void setPreQueens(unsigned int ld,unsigned int rd,unsigned int col,unsigned int k,unsigned int l,unsigned int row,unsigned int queens,unsigned int LD,unsigned int RD,int *counter,ConstellationArrayList* constellations,unsigned int N)
{
  const unsigned int mask=(1<<N)-1;//setPreQueensで使用
  if(row==k || row==l){ // k行とl行はさらに進む
    setPreQueens(ld<<1,rd>>1,col,k,l,row+1,queens,LD,RD,counter,constellations,N);
    return;
  }
  // preQueensのクイーンが揃うまでクイーンを追加する。現在のクイーンの数が presetQueens に達した場合、現在の状態を新しいコンステレーションとして追加し、カウンターを増加させる。
  if(queens==presetQueens){
    // リストに４個クイーンを置いたセットを追加する
    add_constellation(ld,rd,col,row<<20,constellations);
    (*counter)++;
    return;
  }
  // k列かl列が終わっていなければ、クイーンを置いてボードを占領し、さらに先に進む。
  else{
    // 現在の行にクイーンを配置できる位置（自由な位置）を計算
    unsigned int free=mask&~(ld | rd | col | (LD>>(N-1-row)) | (RD<<(N-1-row)));
    register int bit;
    while(free){
      bit=free & (-free);
      free -= bit;
      // クイーンをおける場所があれば、その位置にクイーンを配置し、再帰的に次の行に進む
      setPreQueens((ld | bit)<<1,(rd | bit)>>1,col | bit,k,l,row+1,queens+1,LD,RD,counter,constellations,N);
    }
  }
}
/**
 * 指定した盤面 (i, j, k, l) を90度・180度・270度回転したいずれかの盤面が
 * すでにIntHashSetに存在しているかをチェックする関数
 *
 * @param ijklList 既出盤面signature（ijkl値）の集合（HashSet）
 * @param i,j,k,l  チェック対象の盤面インデックス
 * @param N        盤面サイズ
 * @return         いずれかの回転済み盤面が登録済みなら1、なければ0
 * @details
 *   - N-Queens探索で、既存盤面の90/180/270度回転形と重複する配置を高速に排除する。
 *   - 回転後のijklをそれぞれ計算し、HashSetに含まれていれば即1を返す（重複扱い）。
 *   - 真の“unique配置”のみ探索・カウントしたい場合の前処理として必須。
 */
unsigned int checkRotations(IntHashSet* ijklList,unsigned int i,unsigned int j,unsigned int k,unsigned int l,unsigned int N)
{
  unsigned int rot90=((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i;
  unsigned int rot180=((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+(N-1-k);
  unsigned int rot270=(l<<15)+(k<<10)+((N-1-i)<<5)+(N-1-j);
  if(int_hashset_contains(ijklList,rot90)){ return 1; }
  if(int_hashset_contains(ijklList,rot180)){ return 1; }
  if(int_hashset_contains(ijklList,rot270)){ return 1; }
  return 0;
}
/**
 * ConstellationArrayList内の全Constellationのsolutionsフィールド値を合計する関数
 * 
 * @param constellations 合計対象のConstellationArrayListポインタ
 * @param solutions      合計値の初期値（0を渡すのが標準だが累積加算も可）
 * @return 全要素のsolutions値（0より大きいもののみ）の合計
 * 
 * @details
 *   - 各Constellationのsolutions > 0 のものだけを加算（未計算=-1はスキップ）
 *   - N-Queens探索で全グループ/分割探索の解数を集約する用途
 *   - 戻り値は累積加算値なので、通常は0で初期化して使う
 */
long calcSolutions(ConstellationArrayList* constellations,long solutions)
{
  Constellation* c;
  for(unsigned int i=0;i<constellations->size;i++){
    c=&constellations->data[i];
    if(c->solutions > 0){
      solutions += c->solutions;
    }
  }
  return solutions;
}
/**
 * CUDAカーネル：各Constellation（部分盤面）ごとにN-Queens解数を並列探索し、block単位で合計値を出力する関数
 *
 * @param constellations 入力となるConstellation配列（部分盤面群）
 * @param _total         各blockごとの解数合計を書き込む配列（block数分）
 * @param N              盤面サイズ
 * @param totalSize      探索対象Constellationの総数
 *
 * @details
 *   - 各threadは自身のidxに対応するConstellationに対し、該当ソルバ（SQ...関数群）で解数を探索
 *   - dummy data（start==69）はスキップ（トラッシュ処理）
 *   - 盤面の対称性補正（symmetry(ijkl,N)）も適用し、正確なユニーク解数を求める
 *   - block内のthreadで部分和（sum[tid]）を計算し、warp・block内で段階的に加算・集約
 *   - 最終的にblockごとの合計値を_total[bid]に格納（CPU側で全block合計すれば総解数）
 *   - スレッド間同期（__syncthreads, __syncwarp）により正確な集約処理を実装
 *   - 大規模並列GPU探索時でも「高速・正確・スケーラブル」なN-Queens全解数計算を実現
 */
__global__ void execSolutionsKernel(Constellation* constellations,unsigned int* _total,int N, unsigned int totalSize)
{
    const unsigned int tid=threadIdx.x;
    const unsigned int bid=blockIdx.x;
    const unsigned int idx = bid*blockDim.x+tid;
    // 範囲外アクセスのチェック
    __shared__ unsigned int sum[THREAD_NUM];
    unsigned int cnt=0;
    sum[tid]=0;
    /**
    if (idx >= totalSize){
       sum[tid]=0;
       return;
    }
    */
    if(idx<totalSize){
      Constellation* constellation = &constellations[idx];
      int start = constellation->startijkl >> 20;
      //dummy dataはスキップする
      if(start==69){
        sum[tid]=0;
        return ;
      }else{
        //1 << … は 1u << … にします（1 << 31 付近で符号拡張回避）。
        const unsigned int ijkl = constellation->startijkl & ((1u<< 20) - 1u);
        const int j = getj(ijkl);//unsignedで値が崩れます
        const int k = getk(ijkl);//unsignedで値が崩れます
        const int l = getl(ijkl);//unsignedで値が崩れます
        /**
        const int j = getj(constellation->startijkl);
        const int k = getk(constellation->startijkl);
        const int l = getl(constellation->startijkl);
        const int ijkl = constellation->startijkl & ((1u<< 20) - 1u);
        */
        // ---- ここから型は unsigned に統一 ----
        //ld/rd/col/free/mask を unsigned（または uint32_t/uint64_t） に統一。
        const unsigned int mask = (N == 32) ? 0xFFFFFFFFu : ((1u << N) - 1u);

        unsigned int ld = (unsigned)constellation->ld >> 1;
        unsigned int rd = (unsigned)constellation->rd >> 1;
        unsigned int col = ((unsigned)constellation->col>>1)|(~((1u<< (N-2))-1u));

        // 最初のfreeは必ずNビット幅に丸める
        /* int free = mask & ~(ld | rd | col); */

        // ここは意図を明確化： 1u << (N-1-j/l)
        /* int LD = (1<<(N-1)>>j)|(1<<(N-1)>>l); */
        unsigned int LD=(1u<<(N-1-j))|(1u<<(N-1-l));
        ld |= (LD>>(N-start));
        if(start>(int)k){
          rd|=(1u<<(N-1-(start-k+1)));// (start - k + 1) 行だけ右斜めへ
        }
        if(j>=(2*N-33-start)){// クイーンjからのrdがない場合のみ追加する
          // ※ 元コードの「符号ビット占有」は unsigned なら不要だが、
          // 既存ロジックを踏襲しておく
          rd|=(1u<<(N-1-j))<<(N-2-start);// 符号ビットを占有する！
        }
        // … jmark/endmark/mark1/mark2 の決定と SQ* の分岐 …
        // 例：
        // const unsigned jmark = j + 1;
        // unsigned endmark = N - 2;
        // cnt = SQd1B(ld, rd, col, start, free, jmark, endmark, mark1, mark2, mask, N);
        // （※ SQ* 側も引数の型を unsigned に寄せていくと安全）
        unsigned int jmark = j + 1;
        unsigned int endmark = N - 2;
        unsigned int mark1, mark2;
        //ld/rd/col/free/mask を unsigned（または uint32_t/uint64_t） に統一。
        // free=mask & ~(ld | rd | col); に統一してください。
        unsigned int free=mask&~(ld|rd|col);
        // int cnt=0;
        /* int free=~(ld | rd | col); */
        /* int mask=(1<<N)-1; */
        // __shared__ int mask;
        // if (threadIdx.x == 0) {
        //     mask=(1<<N)-1;  // 最初のスレッドだけが計算
        // }
        // __syncthreads(); // 全スレッドで同期。board_maskの値が全員に伝わる
        // int free=mask&~(ld|rd|col);
        /**
          どのソリングアルゴリズムを使うかを決めるための大きなケースの区別
          クイーンjがコーナーから2列以上離れている場合
        */
        if(j<(N-3)){
          jmark=j+1;
          endmark=N-2;
          /**
            クイーンjがコーナーから2列以上離れているが、jクイーンからのrdが開始時
            に正しく設定できる場合。
          */
          if(j>2 * N-34-start){
            if(k<l){
              mark1=k-1;
              mark2=l-1;
              if(start<l){// 少なくともlがまだ来ていない場合
                if(start<k){// もしkがまだ来ていないなら
                  if(l != k+1){ // kとlの間に空行がある場合
                    cnt=SQBkBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
                  }else{// kとlの間に空行がない場合
                    cnt=SQBklBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
                  }
                }else{// もしkがすでに開始前に来ていて、lだけが残っている場合
                  cnt=SQBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
                }
              }else{// kとlの両方が開始前にすでに来ていた場合
                cnt=SQBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
              }
            }else{// l<k
              mark1=l-1;
              mark2=k-1;
              if(start<k){// 少なくともkがまだ来ていない場合
                if(start<l){// lがまだ来ていない場合
                  if(k != l+1){// lとkの間に少なくとも1つの自由行がある場合
                    cnt=SQBlBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
                  }else{// lとkの間に自由行がない場合
                    cnt=SQBlkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
                    // cnt=SQBlkBjrB_iter(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
                  }
                }else{ // lがすでに来ていて、kだけがまだ来ていない場合
                  cnt=SQBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
                }
              }else{// lとkの両方が開始前にすでに来ていた場合
                cnt=SQBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
              }
            }
          }else{
            /**
              クイーンjのrdをセットできる行N-1-jmarkに到達するために、
              最初にいくつかのクイーンをセットしなければならない場合。
            */
            if(k<l){
              mark1=k-1;
              mark2=l-1;

              if(l != k+1){// k行とl行の間に少なくとも1つの空行がある。
                cnt=SQBjlBkBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
              }else{// lがkの直後に来る場合
                cnt=SQBjlBklBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
              }
            }else{  // l<k
              mark1=l-1;
              mark2=k-1;
              if(k != l+1){// l行とk行の間には、少なくともefree行が存在する。
                cnt=SQBjlBlBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
              }else{// kがlの直後に来る場合
                cnt=SQBjlBlkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
              }
            }
          }
        }else if(j==(N-3)){// クイーンjがコーナーからちょうど2列離れている場合。
         // これは、最終行が常にN-2行になることを意味する。
          endmark=N-2;
          if(k<l){
            mark1=k-1;
            mark2=l-1;
            if(start<l){// 少なくともlがまだ来ていない場合
              if(start<k){// もしkもまだ来ていないなら
                if(l != k+1){// kとlの間に空行がある場合
                  cnt=SQd2BkBlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
                }else{
                  cnt=SQd2BklB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
                }
              }else{// k が開始前に設定されていた場合
                mark2=l-1;
                cnt=SQd2BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
              }
            }else{ // もしkとlが開始前にすでに来ていた場合
              cnt=SQd2B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
            }
          }else{// l<k
            mark1=l-1;
            mark2=k-1;
            endmark=N-2;
            if(start<k){// 少なくともkがまだ来ていない場合
              if(start<l){// lがまだ来ていない場合
                if(k != l+1){// lとkの間に空行がある場合
                  cnt=SQd2BlBkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
                }else{// lとkの間に空行がない場合
                  cnt=SQd2BlkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
                }
              }else{ // l が開始前に来た場合
                mark2=k-1;
                cnt=SQd2BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
              }
            }else{ // lとkの両方が開始前にすでに来ていた場合
              cnt=SQd2B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
            }
          }
        }else if(j==N-2){ // クイーンjがコーナーからちょうど1列離れている場合
          if(k<l){// kが最初になることはない、lはクイーンの配置の関係で
                      // 最後尾にはなれないので、常にN-2行目で終わる。
            endmark=N-2;

            if(start<l){// 少なくともlがまだ来ていない場合
              if(start<k){// もしkもまだ来ていないなら
                mark1=k-1;

                if(l != k+1){// kとlが隣り合っている場合
                  mark2=l-1;
                  cnt=SQd1BkBlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
                }else{
                  cnt=SQd1BklB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
                }
              }else{// lがまだ来ていないなら
                mark2=l-1;
                cnt=SQd1BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
              }
            }else{// すでにkとlが来ている場合
              cnt=SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
            }
          }else{ // l<k
            if(start<k){// 少なくともkがまだ来ていない場合
              if(start<l){ // lがまだ来ていない場合
                if(k<N-2){// kが末尾にない場合
                  mark1=l-1;
                  endmark=N-2;

                  if(k != l+1){// lとkの間に空行がある場合
                    mark2=k-1;
                    cnt=SQd1BlBkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
                  }else{// lとkの間に空行がない場合
                    cnt=SQd1BlkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
                  }
                }else{// kが末尾の場合
                  if(l != (N-3)){// lがkの直前でない場合
                    mark2=l-1;
                    endmark=(N-3);
                    cnt=SQd1BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
                  }else{// lがkの直前にある場合
                    endmark=(N-4);
                    cnt=SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
                  }
                }
              }else{ // もしkがまだ来ていないなら
                if(k != N-2){// kが末尾にない場合
                  mark2=k-1;
                  endmark=N-2;
                  cnt=SQd1BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
                }else{// kが末尾の場合
                  endmark=(N-3);
                  cnt=SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
                }
              }
            }else{// kとlはスタートの前
              endmark=N-2;
              cnt=SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
            }
          }
        }else{// クイーンjがコーナーに置かれている場合
          endmark=N-2;
          if(start>k){
            cnt=SQd0B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
          }else{
            /**
              クイーンをコーナーに置いて星座を組み立てる方法と、ジャスミンを適用
              する方法によって、Kは最後列に入ることはできない。
            */
            mark1=k-1;
            cnt=SQd0BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
          }
        }
        cnt*=symmetry(ijkl,N);
      }
    }
    // 完成した開始コンステレーションを削除する。
    /* sum[tid]=cnt * symmetry(ijkl,N); */
    sum[tid]=cnt;
    __syncthreads();

    if(tid<64&&tid+64<THREAD_NUM){ sum[tid]+=sum[tid+64]; }

    __syncwarp();if(tid<32){ sum[tid]+=sum[tid+32]; } 
    __syncwarp();if(tid<16){ sum[tid]+=sum[tid+16]; } 
    __syncwarp();if(tid<8){ sum[tid]+=sum[tid+8]; } 
    __syncwarp();if(tid<4){ sum[tid]+=sum[tid+4]; } 
    __syncwarp();if(tid<2){ sum[tid]+=sum[tid+2]; } 
    __syncwarp();if(tid<1){ sum[tid]+=sum[tid+1]; } 
    if(tid==0){ _total[bid]=sum[0]; }
}
/**
 * ConstellationArrayListの各Constellation（部分盤面）ごとに
 * N-Queens探索を分岐し、そのユニーク解数をsolutionsフィールドに記録する関数（CPU版）
 *
 * @param constellations 解探索対象のConstellationArrayListポインタ
 * @param N              盤面サイズ
 *
 * @details
 *   - 各Constellation（部分盤面）ごとにj, k, l, 各マスク値を展開し、
 *     複雑な分岐で最適な再帰ソルバー（SQ...関数群）を呼び出して解数を計算
 *   - 分岐ロジックは、部分盤面・クイーンの位置・コーナーからの距離などで高速化
 *   - 解数はtemp_counterに集約し、各Constellationのsolutionsフィールドに記録
 *   - symmetry(ijkl, N)で回転・ミラー重複解を補正
 *   - GPUバージョン(execSolutionsKernel)のCPU移植版（デバッグ・逐次確認にも活用）
 *
 * @note
 *   - N-Queens最適化アルゴリズムの核心部
 *   - temp_counterは再帰呼び出しで合計を受け渡し
 *   - 実運用時は、より多くの分岐パターンを組み合わせることで最大速度を発揮
 */
void execSolutions(ConstellationArrayList* constellations,int N)
{
  int j=0;
  int k=0;
  int l=0;
  unsigned int ijkl=0;
  int ld=0;
  int rd=0;
  int col=0;
  unsigned int startIjkl=0;
   int start=0;
  unsigned  int free=0;
   int LD=0;
  unsigned  int jmark=0;
  unsigned  int endmark=0;
  unsigned  int mark1=0;
  unsigned  int mark2=0;
  unsigned  int smallmask=(1<<(N-2))-1;
  unsigned  int cnt=0;
  unsigned  int mask=(1<<N-1)-1;
  for(unsigned int i=0;i<constellations->size;i++){
    Constellation* constellation=&constellations->data[i];
    startIjkl=constellation->startijkl;
    start=startIjkl>>20;
    ijkl=startIjkl & ((1<<20)-1);
    j=getj(ijkl);
    k=getk(ijkl);
    l=getl(ijkl);
    /**
      重要な注意：ldとrdを1つずつ右にずらすが、これは右列は重要ではないから
      （常に女王lが占有している）。
    */
    // 最下段から上に、jとlのクイーンによるldの占有を追加する。
    // LDとrdを1つずつ右にずらすが、これは右列は重要ではないから（常に女王lが占有している）。
    LD=(1<<(N-1)>>j) | (1<<(N-1)>>l);
    ld=constellation->ld>>1;
    ld |= LD>>(N-start);
    rd=constellation->rd>>1;// クイーンjとkのrdの占有率を下段から上に加算する。
    if(start>k){
      rd |= (1<<(N-1)>>(start-k+1));
    }
    if(j >= 2 * N-33-start){// クイーンjからのrdがない場合のみ追加する
      rd |= (1<<(N-1)>>j)<<(N-2-start);// 符号ビットを占有する！
    }
    // また、colを占有し、次にフリーを計算する
    col=(constellation->col>>1) | (~smallmask);
    free=mask&~(ld | rd | col);
    /**
      どのソリングアルゴリズムを使うかを決めるための大きなケースの区別
      クイーンjがコーナーから2列以上離れている場合
    */
    if(j<(N-3)){
      jmark=j+1;
      endmark=N-2;
      /**
        クイーンjがコーナーから2列以上離れているが、jクイーンからのrdが開始時
        に正しく設定できる場合。
      */
      if(j>2 * N-34-start){
        if(k<l){
          mark1=k-1;
          mark2=l-1;
          if(start<l){// 少なくともlがまだ来ていない場合
            if(start<k){// もしkがまだ来ていないなら
              if(l != k+1){ // kとlの間に空行がある場合
                cnt=SQBkBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
              }else{// kとlの間に空行がない場合
                cnt=SQBklBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
              }
            }else{// もしkがすでに開始前に来ていて、lだけが残っている場合
              cnt=SQBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
            }
          }else{// kとlの両方が開始前にすでに来ていた場合
            cnt=SQBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
          }
        }else{// l<k 
          mark1=l-1;
          mark2=k-1;
          if(start<k){// 少なくともkがまだ来ていない場合
            if(start<l){// lがまだ来ていない場合
              if(k != l+1){// lとkの間に少なくとも1つの自由行がある場合
                cnt=SQBlBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
              }else{// lとkの間に自由行がない場合
                cnt=SQBlkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
                // cnt=SQBlkBjrB_iter(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
              }
            }else{ // lがすでに来ていて、kだけがまだ来ていない場合
              cnt=SQBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
            }
          }else{// lとkの両方が開始前にすでに来ていた場合
            cnt=SQBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
          }
        }
      }else{
        /**
          クイーンjのrdをセットできる行N-1-jmarkに到達するために、
          最初にいくつかのクイーンをセットしなければならない場合。
        */
        if(k<l){
          mark1=k-1;
          mark2=l-1;

          if(l != k+1){// k行とl行の間に少なくとも1つの空行がある。
            cnt=SQBjlBkBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
          }else{// lがkの直後に来る場合
            cnt=SQBjlBklBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
          }
        }else{  // l<k
          mark1=l-1;
          mark2=k-1;
          if(k != l+1){// l行とk行の間には、少なくともefree行が存在する。
            cnt=SQBjlBlBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
          }else{// kがlの直後に来る場合 
            cnt=SQBjlBlkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
          }
        }
      }
    }else if(j==(N-3)){// クイーンjがコーナーからちょうど2列離れている場合。
     // これは、最終行が常にN-2行になることを意味する。
      endmark=N-2;
      if(k<l){
        mark1=k-1;
        mark2=l-1;
        if(start<l){// 少なくともlがまだ来ていない場合
          if(start<k){// もしkもまだ来ていないなら
            if(l != k+1){// kとlの間に空行がある場合
              cnt=SQd2BkBlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
            }else{
              cnt=SQd2BklB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
            }
          }else{// k が開始前に設定されていた場合
            mark2=l-1;
            cnt=SQd2BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
          }
        }else{ // もしkとlが開始前にすでに来ていた場合
          cnt=SQd2B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
        }
      }else{// l<k
        mark1=l-1;
        mark2=k-1;
        endmark=N-2;
        if(start<k){// 少なくともkがまだ来ていない場合
          if(start<l){// lがまだ来ていない場合
            if(k != l+1){// lとkの間に空行がある場合
              cnt=SQd2BlBkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
            }else{// lとkの間に空行がない場合
              cnt=SQd2BlkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
            }
          }else{ // l が開始前に来た場合
            mark2=k-1;
            cnt=SQd0BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
          }
        }else{ // lとkの両方が開始前にすでに来ていた場合
          cnt=SQd2B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
        }
      }
    }else if(j==N-2){ // クイーンjがコーナーからちょうど1列離れている場合
      if(k<l){// kが最初になることはない、lはクイーンの配置の関係で
                  // 最後尾にはなれないので、常にN-2行目で終わる。
        endmark=N-2;

        if(start<l){// 少なくともlがまだ来ていない場合
          if(start<k){// もしkもまだ来ていないなら
            mark1=k-1;

            if(l != k+1){// kとlが隣り合っている場合
              mark2=l-1;
              cnt=SQd1BkBlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
            }else{
              cnt=SQd1BklB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
            }
          }else{// lがまだ来ていないなら
            mark2=l-1;
            cnt=SQd1BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
          }
        }else{// すでにkとlが来ている場合
          cnt=SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
        }
      }else{ // l<k
        if(start<k){// 少なくともkがまだ来ていない場合
          if(start<l){ // lがまだ来ていない場合
            if(k<N-2){// kが末尾にない場合
              mark1=l-1;
              endmark=N-2;

              if(k != l+1){// lとkの間に空行がある場合
                mark2=k-1;
                cnt=SQd1BlBkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
              }else{// lとkの間に空行がない場合
                cnt=SQd1BlkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
              }
            }else{// kが末尾の場合
              if(l != (N-3)){// lがkの直前でない場合
                mark2=l-1;
                endmark=(N-3);
                cnt=SQd1BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
              }else{// lがkの直前にある場合
                endmark=(N-4);
                cnt=SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
              }
            }
          }else{ // もしkがまだ来ていないなら
            if(k != N-2){// kが末尾にない場合
              mark2=k-1;
              endmark=N-2;
              cnt=SQd1BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
            }else{// kが末尾の場合
              endmark=(N-3);
              cnt=SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
            }
          }
        }else{// kとlはスタートの前
          endmark=N-2;
          cnt=SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
        }
      }
    }else{// クイーンjがコーナーに置かれている場合
      endmark=N-2;
      if(start>k){
        cnt=SQd0B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
      }else{
        /**
          クイーンをコーナーに置いて星座を組み立てる方法と、ジャスミンを適用
          する方法によって、Kは最後列に入ることはできない。
        */
        mark1=k-1;
        cnt=SQd0BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,mask,N);
      }
    }
    // 完成した開始コンステレーションを削除する。
    constellation->solutions=cnt * symmetry(ijkl,N);
  }
}
/**
 * 開始コンステレーション（部分盤面配置パターン）の列挙・重複排除を行う関数
 *
 * @param ijklList        uniqueな部分盤面signature（ijkl値）の格納先HashSet
 * @param constellations  Constellation本体リスト（実際の盤面は後続で生成）
 * @param N               盤面サイズ
 *
 * @details
 *   - コーナー・エッジ・対角・回転対称性を考慮し、「代表解」となるuniqueな開始盤面のみ抽出する。
 *   - forループの入れ子により、N-Queens盤面の「最小単位部分盤面」を厳密な順序で列挙。
 *   - k, l, i, j 各インデックスの取り方・範囲・重複排除のための判定ロジックが最適化されている。
 *   - checkRotations()で既出盤面（回転対称）を排除、必要なものだけをijklListに追加。
 *   - このunique setをもとに、後段でConstellation構造体の生成・分割探索を展開可能。
 *
 * @note
 *   - 「部分盤面分割＋代表解のみ探索」戦略は大規模Nの高速化の要！
 *   - このループ構造・排除ロジックがN-Queensソルバの根幹。
 */
unsigned int rot180_in_set(IntHashSet* ijklList,unsigned int i,unsigned int j,unsigned int k,unsigned int l,unsigned int N)
{
  // toijkl: ((i<<15)|(j<<10)|(k<<5)|l)
  // 180°回転: (i,j,k,l) -> (N-1-j, N-1-i, N-1-l, N-1-k)
  unsigned int val = ((N-1-j)<<15) | ((N-1-i)<<10) | ((N-1-l)<<5) | (N-1-k);
  return int_hashset_contains(ijklList, val);
}
/**
 */
void genConstellations(IntHashSet* ijklList,ConstellationArrayList* constellations,int N)
{
  unsigned int halfN=(N+1)/2; // N の半分を切り上げる
  unsigned int L=1<<(N-1);    // Lは左端に1を立てる
  // --- [Opt-03] 中央列特別処理（奇数Nの場合のみ） ---
  if (N % 2 == 1) {
    int center = N / 2;
    for(unsigned int l = center + 1; l < N - 1; ++l) {
      for(unsigned int i = center + 1; i < N - 1; ++i) {
        if (i == (N - 1) - l) continue;
        for(unsigned int j = N - center - 2; j > 0; --j) {
          if (j == i || j == l) continue;
          if (!checkRotations(ijklList, i, j, center, l, N)
              && !rot180_in_set(ijklList, i, j, center, l, N)) {
            int_hashset_add(ijklList, toijkl(i, j, center, l));
          }
        }
      }
    }
  }
  // --- [Opt-03] ここまで ---

  /**
    コーナーにクイーンがいない場合の開始コンステレーションを計算する
    最初のcolを通過する
    k: 最初の列（左端）に配置されるクイーンの行のインデックス。
  */
  for(unsigned int k = 1; k < halfN; ++k) {
    // 奇数Nのとき中央列kをスキップ（上で個別処理済み）
    if ((N % 2 == 1) && (k == N / 2)) continue;
    /**
      l: 最後の列（右端）に配置されるクイーンの行のインデックス。
      l を k より後の行に配置する理由は、回転対称性を考慮して配置の重複を避けるためです。
    */
    for(unsigned int l = k + 1; l < N - 1; ++l) {
      /**
        i: 最初の行（上端）に配置されるクイーンの列のインデックス。
        最初の行を通過する
        k よりも下の行に配置することで、ボード上の対称性や回転対称性を考慮して、重複した解を避けるための配慮がされています。
      */
      for(unsigned int i = k + 1; i < (N - 1); ++i) {
        // i==N-1-l は、行iが列lの「対角線上」にあるかどうかをチェックしています。
        if (i == (N - 1) - l) continue;
        /**
          j: 最後の行（下端）に配置されるクイーンの列のインデックス。
          最後の行を通過する
        */
        for(unsigned int j = N - k - 2; j > 0; --j) {
          /**
            同じ列や行にクイーンが配置されている場合は、その配置が有効でないためスキップ
          */
          if (j == i || l == j) continue;
          /**
            回転対称でスタートしない場合
            checkRotationsで回転対称性をチェックし、対称でない場合に ijklList に配置を追加します。
          */
          if (!checkRotations(ijklList, i, j, k, l, N)) {
            int_hashset_add(ijklList, toijkl(i, j, k, l));
          }
        }
      }
    }
  }

  /**
    角の位置（コーナー）にクイーンがある場合の開始コンステレーションを計算する
  */
  for(unsigned int j = 1; j < N - 2; ++j) {
    for(unsigned int l = j + 1; l < N - 1; ++l) {
      int_hashset_add(ijklList, toijkl(0, j, 0, l));
    }
  }

IntHashSet* ijklListJasmin=create_int_hashset();
  unsigned int startConstellation;
  for(unsigned int i=0;i<ijklList->size;i++){
    startConstellation=ijklList->data[i];
    int_hashset_add(ijklListJasmin,jasmin(startConstellation,N));
  }
  //free_int_hashset(ijklList);
  ijklList=ijklListJasmin;
  /**
    jasmin関数を使用して、クイーンの配置を回転およびミラーリングさせて、最
    も左上に近い標準形に変換します。
    同じクイーンの配置が標準形に変換された場合、同じ整数値が返されます。
    ijkListJasmin は HashSet です。
    jasmin メソッドを使用して変換された同じ値のクイーンの配置は、HashSet に
    一度しか追加されません。
    したがって、同じ値を持つクイーンの配置が複数回追加されても、HashSet の
    サイズは増えません。
  */
  //int i,j,k,l,ld,rd,col,currentSize=0;
  int sc=0;
  int i=0;
  int j=0;
  int k=0;
  int l=0;
  int ld=0;
  int rd=0;
  int col=0;
  int LD=0;
  int RD=0;
  int counter=0;
  int currentSize=0;
  for(unsigned int s=0;s<ijklList->size;s++){
    sc=ijklList->data[s];
    i=geti(sc);
    j=getj(sc);
    k=getk(sc);
    l=getl(sc);
    /**
      プレクイーンでボードを埋め、対応する変数を生成する。
      各星座に対して ld,rd,col,start_queens_ijkl を設定する。
      碁盤の境界線上のクイーンに対応する碁盤を占有する。
      空いている最初の行、すなわち1行目から開始する。
      クイーンの左対角線上の攻撃範囲を設定する。
      L>>>(i-1) は、Lを (i-1) ビット右にシフトします。これにより、クイーンの
      位置 i に対応するビットが右に移動します。
      1<<(N-k) は、1を (N-k) ビット左にシフトします。これにより、位置 k に対
      応するビットが左に移動します。
      両者をビットOR (|) することで、クイーンの位置 i と k に対応するビットが
      1となり、これが左対角線の攻撃範囲を表します。
    */
    ld=(L>>(i-1)) | (1<<(N-k));
    /**
      クイーンの右対角線上の攻撃範囲を設定する。
      L>>>(i+1) は、Lを (i+1) ビット右にシフトします。これにより、クイーンの
      位置 i に対応するビットが右に移動します。
      1<<(l-1) は、1を (l-1) ビット左にシフトします。これにより、位置 l に対
      応するビットが左に移動します。
      両者をビットOR (|) することで、クイーンの位置 i と l に対応するビットが
      1となり、これが右対角線の攻撃範囲を表します。
    */
    rd=(L>>(i+1)) | (1<<(l-1));
    /**
      クイーンの列の攻撃範囲を設定する。
      1 は、最初の列（左端）にクイーンがいることを示します。
      L は、最上位ビットが1であるため、最初の行にクイーンがいることを示します。
      L>>>i は、Lを i ビット右にシフトし、クイーンの位置 i に対応する列を占有します
      L>>>j は、Lを j ビット右にシフトし、クイーンの位置 j に対応する列を占有します。
      これらをビットOR (|) することで、クイーンの位置 i と j に対応する列が1
      となり、これが列の攻撃範囲を表します。
    */
    col=1 | L | (L>>i) | (L>>j);
    /**
      最後の列のクイーンj、k、lの対角線を占領しボード上方に移動させる
      L>>>j は、Lを j ビット右にシフトし、クイーンの位置 j に対応する左対角線を占有します。
      L>>>l は、Lを l ビット右にシフトし、クイーンの位置 l に対応する左対角線を占有します。
      両者をビットOR (|) することで、クイーンの位置 j と l に対応する左対角線
      が1となり、これが左対角線の攻撃範囲を表します。
    */
    LD=(L>>j) | (L>>l);
    /**
      最後の列の右対角線上の攻撃範囲を設定する。
      L>>>j は、Lを j ビット右にシフトし、クイーンの位置 j に対応する右対角線を占有します。
      1<<k は、1を k ビット左にシフトし、クイーンの位置 k に対応する右対角線を占有します。
      両者をビットOR (|) することで、クイーンの位置 j と k に対応する右対角線
      が1となり、これが右対角線の攻撃範囲を表します。
    */
    RD=(L>>j) | (1<<k);
    // すべてのサブコンステレーションを数える
    counter=0;
    // すべてのサブコンステレーションを生成する
    setPreQueens(ld,rd,col,k,l,1,j==N-1 ? 3 : 4,LD,RD,&counter,constellations,N);
    currentSize=constellations->size;
     // jklとsymとstartはすべてのサブコンステレーションで同じである
    for(unsigned int a=0;a<counter;a++){
      constellations->data[currentSize-a-1].startijkl |= toijkl(i,j,k,l);
    }
  }
}
/**
 * 未使用変数警告（-Wunused-parameter等）を抑制するためのダミー関数
 *
 * @param unuse 未使用int変数（何でもOK、実際には使わない）
 * @param argv  未使用char*配列（何でもOK、実際には使わない）
 * @details
 *   - コンパイラの「未使用引数」警告抑制のため、型に応じて何らかの操作（printf等）を
 *     実装しておくのが定番。
 *   - 本番コードで未使用変数が残る場合や、必須関数のダミー実装等で用いる。
 *   - 最適化時に消される可能性もあるので、「本当に必要な値」には使わないこと。
 */
void f(int unuse,char* argv[]){
  printf("%d%s\n",unuse,argv[0]);
}
/** */
__host__ __device__ unsigned int SQd0B(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  if(row==endmark){ return 1; }
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){ if(row<endmark-1 || ~next_free){ total+=SQd0B((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); } }
  }
  return total;
}
__host__ __device__ unsigned int SQd0BkB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==mark1){
    while(free){
      free^=bit=free&-free;
      next_free=mask&~((ld|bit)<<2|(rd|bit)>>2|col|bit|1<<N-3);
      if(next_free){ total+=SQd0B((ld|bit)<<2,(rd|bit)>>2|1<<N-3,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,mask,N); }
    }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){ total+=SQd0BkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}
__host__ __device__ unsigned int SQd1BklB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==mark1){
    while(free){
      free^=bit=free&-free;
      next_free=mask&~((ld|bit)<<3|(rd|bit)>>3|col|bit|1|1<<N-4);
      if(next_free){ total+=SQd1B((ld|bit)<<3|1,(rd|bit)>>3|1<<N-4,col|bit,row+3,next_free,jmark,endmark,mark1,mark2,mask,N);
      }
    }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){ total+=SQd1BklB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}
__host__ __device__ unsigned int SQd1B(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  if(row==endmark){ return 1; }
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){
      if(row+1>=endmark||~next_free){ total+=SQd1B((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
    }
  }
  return total;
}
__host__ __device__ unsigned int SQd1BkBlB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==mark1){
    while(free){
      free^=bit=free&-free;
      next_free=mask&~((ld|bit)<<2|(rd|bit)>>2|col|bit|1<<N-3);
      if(next_free){ total+=SQd1BlB((ld|bit)<<2,(rd|bit)>>2|1<<N-3,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,mask,N); }
    }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){ total+=SQd1BkBlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}
__host__ __device__ unsigned int SQd1BlB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==mark2){
    while(free){
      free^=bit=free&-free;
      next_free=mask&~((ld|bit)<<2|1|(rd|bit)>>2|col|bit);
      if(next_free){ if(row+2>=endmark || ~next_free){ total+=SQd1B((ld|bit)<<2|1,(rd|bit)>>2,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,mask,N); } }
    }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){ total+=SQd1BlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}
__host__ __device__ unsigned int SQd1BlkB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==mark1){
    while(free){
      free^=bit=free&-free;
      next_free=mask&~((ld|bit)<<3|(rd|bit)>>3|col|bit|2|1<<N-3);
      if(next_free){ total+=SQd1B((ld|bit)<<3|2,(rd|bit)>>3|1<<N-3,col|bit,row+3,next_free,jmark,endmark,mark1,mark2,mask,N); }
    }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){ total+=SQd1BlkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}
__host__ __device__ unsigned int SQd1BlBkB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==mark1){
    while(free){
      free^=bit=free&-free;
      next_free=mask&~((ld|bit)<<2|(rd|bit)>>2|col|bit|1);
      if(next_free){ total+=SQd1BkB((ld|bit)<<2|1,(rd|bit)>>2,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,mask,N); }
    }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){ total+=SQd1BlBkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}
__host__ __device__ unsigned int SQd1BkB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==mark2){
    while(free){
      free^=bit=free&-free;
      next_free=mask&~((ld|bit)<<2|(rd|bit)>>2|col|bit|1<<N-3);
      if(next_free){ total+=SQd1B((ld|bit)<<2,(rd|bit)>>2|1<<N-3,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,mask,N); }
    }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){ total+=SQd1BkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}
__host__ __device__ unsigned int SQd2BlkB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==mark1){
    while(free){
      free^=bit=free&-free;
      next_free=mask&~((ld|bit)<<3|(rd|bit)>>3|col|bit|1<<N-3|2);
      if(next_free){ total+=SQd2B((ld|bit)<<3|2,(rd|bit)>>3|1<<N-3,col|bit,row+3,next_free,jmark,endmark,mark1,mark2,mask,N); }
    }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){ total+=SQd2BlkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}
__host__ __device__ unsigned int SQd2BklB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==mark1){
    while(free){
      free^=bit=free&-free;
      next_free=mask&~((ld|bit)<<3|(rd|bit)>>3|col|bit|1<<N-4|1);
      if(next_free){ total+=SQd2B((ld|bit)<<3|1,(rd|bit)>>3|1<<N-4,col|bit,row+3,next_free,jmark,endmark,mark1,mark2,mask,N); }
    }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){ total+=SQd2BklB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}
__host__ __device__ unsigned int SQd2BkB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==mark2){
    while(free){
      free^=bit=free&-free;
      next_free=~((ld|bit)<<2|(rd|bit)>>2|col|bit|1<<N-3);
      if(next_free){ total+=SQd2B((ld|bit)<<2,(rd|bit)>>2|1<<N-3,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,mask,N); }
    }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){ total+=SQd2BkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}
__host__ __device__ unsigned int SQd2BlBkB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==mark1){
    while(free){
      free^=bit=free&-free;
      next_free=mask&~((ld|bit)<<2|(rd|bit)>>2|col|bit|1);
      if(next_free){ total+=SQd2BkB((ld|bit)<<2|1,(rd|bit)>>2,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,mask,N); }
    }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){ total+=SQd2BlBkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}
__host__ __device__ unsigned int SQd2BlB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==mark2){
    while(free){
      free^=bit=free&-free;
      next_free=mask&~((ld|bit)<<2|(rd|bit)>>2|col|bit|1);
      if(next_free){ total+=SQd2B((ld|bit)<<2|1,(rd|bit)>>2,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,mask,N); }
    }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){ total+=SQd2BlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}
__host__ __device__ unsigned int SQd2BkBlB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==mark1){
    while(free){
      free^=bit=free&-free;
      next_free=mask&~((ld|bit)<<2|(rd|bit)>>2|col|bit|1<<N-3);
      if(next_free){ total+=SQd2BlB((ld|bit)<<2,(rd|bit)>>2|1<<N-3,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,mask,N); }
    }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){ total+=SQd2BkBlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}
__host__ __device__ unsigned int SQd2B(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  if(row==endmark){ if( (free&(~1))>0){ return 1; } }
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){
      if(row>=endmark-1||~((ld|bit)<<1|(rd|bit)>>1|col|bit)>0){
          total+=SQd2B((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N);
      }
    }
  }
  return total;
}
__host__ __device__ unsigned int SQBlBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==mark2){
    while(free){
      free^=bit=free&-free;
      next_free=mask&~((ld|bit)<<2|(rd|bit)>>2|col|bit|1);
      if(next_free){ total+=SQBjrB((ld|bit)<<2|1,(rd|bit)>>2,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,mask,N); }
    }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){ total+=SQBlBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}
__host__ __device__ unsigned int SQBkBlBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==mark1){
    while(free){
      free^=bit=free&-free;
      next_free=mask&~((ld|bit)<<2|(rd|bit)>>2|col|bit|1<<N-3);
      if(next_free){ total+=SQBlBjrB((ld|bit)<<2,(rd|bit)>>2|1<<N-3,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,mask,N); }
    }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){ total+=SQBkBlBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}
__host__ __device__ unsigned int SQBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==jmark){
    free&=~1;
    ld|=1;
    while(free){
      free^=bit=free&-free;
      next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
      if(next_free){ total+=SQB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
    }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){ total+=SQBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}
__host__ __device__ unsigned int SQB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  if(row==endmark){ return 1; }
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){ if(row>=endmark-1||~((ld|bit)<<1|(rd|bit)>>1|col|bit)>0){ total+=SQB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); } }
  }
  return total;
}
__host__ __device__ unsigned int SQBlBkBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==mark1){
    while(free){
      free^=bit=free&-free;
      next_free=~((ld|bit)<<2|(rd|bit)>>2|col|bit|1);
      if(next_free){ total+=SQBkBjrB((ld|bit)<<2|1,(rd|bit)>>2,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,mask,N); }
    }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(next_free){ total+=SQBlBkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}
__host__ __device__ unsigned int SQBkBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==mark2){
    while(free){
      free^=bit=free&-free;
      next_free=mask&~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1<<(N-3));
      if(next_free){ total+=SQBjrB((ld|bit)<<2,(rd|bit)>>2|1<<N-3,col|bit,row+2,next_free,jmark,endmark,mark1,mark2,mask,N); }
    }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){ total+=SQBkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}
__host__ __device__ unsigned int SQBklBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==mark1){
    while(free){
      free^=bit=free&-free;
      next_free=mask&~((ld|bit)<<3|(rd|bit)>>3|col|bit|1<<N-4|1);
      if(next_free){ total+=SQBjrB((ld|bit)<<3|1,(rd|bit)>>3|1<<N-4,col|bit,row+3,next_free,jmark,endmark,mark1,mark2,mask,N); }
    }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){ total+=SQBklBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}
__host__ __device__ unsigned int SQBlkBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==mark1){
    while(free){
      free^=bit=free&-free;
      next_free=mask&~((ld|bit)<<3|(rd|bit)>>3|col|bit|1<<N-3|2);
      if(next_free){ total+=SQBjrB((ld|bit)<<3|2,(rd|bit)>>3|1<<N-3,col|bit,row+3,next_free,jmark,endmark,mark1,mark2,mask,N); }
    }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(next_free){ total+=SQBlkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}

// N<=32 前提（N=18ならOK）
struct BlkFrame {
  unsigned int ld, rd, col;
  unsigned int row;
  unsigned int free;
  unsigned int jmark, endmark, mark1, mark2, mask, N;
  bool in_mark1; // row==mark1 の分岐中かどうか
};

// 既存の SQBjrB を使う（次段でこれもiter化推奨）
__host__ __device__
unsigned int SQBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row, unsigned int free,unsigned int jmark,unsigned int endmark, unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N);

__host__ __device__
unsigned int SQBlkBjrB_iter(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row, unsigned int free,unsigned int jmark,unsigned int endmark, unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total = 0;

  // 深さの上限：最悪でも N 程度 + α（rowが+3もあるが深さは減る）
  // 余裕を持って 64 くらい確保（N=18なら十分）
  BlkFrame st[64];
  int sp = 0;

  st[0] = {ld, rd, col, row, free, jmark, endmark, mark1, mark2, mask, N, (row == mark1)};

  while (sp >= 0) {
    BlkFrame &f = st[sp];

    // このフレームで試す候補が尽きたら戻る
    if (f.free == 0) { sp--; continue; }

    // free から1bit取り出す
    unsigned int bit = f.free & (~f.free + 1u);  // free & -free のunsigned安全版
    f.free ^= bit;

    if (f.in_mark1) {
      // row==mark1 の特殊ルール
      unsigned int next_free = f.mask & ~(
          ((f.ld | bit) << 3) |
          ((f.rd | bit) >> 3) |
          (f.col | bit) |
          (1u << (f.N - 3)) |
          2u
      );

      if (next_free) {
        total += SQBjrB(((f.ld | bit) << 3) | 2u,
                        ((f.rd | bit) >> 3) | (1u << (f.N - 3)),
                        (f.col | bit),
                        f.row + 3,
                        next_free,
                        f.jmark, f.endmark, f.mark1, f.mark2, f.mask, f.N);
      }
      // row==mark1 分岐は再帰しない（元もそう）
      // 次のbitへ
      continue;
    } else {
      // 通常ルール：再帰で SQBlkBjrB を呼んでいた部分
      unsigned int next_free = f.mask & ~(
          ((f.ld | bit) << 1) |
          ((f.rd | bit) >> 1) |
          (f.col | bit)
      );

      if (next_free) {
        // 再帰呼び出しを push に置換
        sp++;
        // 念のため溢れチェック（デバッグ時）
        if (sp >= 64) {
          // ここに来るならスタックサイズ不足。まずは増やす。
          // 本番は assert/return などお好みで
          sp = 63;
        }
        st[sp] = {
          (f.ld | bit) << 1,
          (f.rd | bit) >> 1,
          (f.col | bit),
          f.row + 1,
          next_free,
          f.jmark, f.endmark, f.mark1, f.mark2, f.mask, f.N,
          ((f.row + 1) == f.mark1)
        };
      }
      continue;
    }
  }

  return total;
}





__host__ __device__ unsigned int SQBjlBkBlBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==N-1-jmark){
    rd|=1<<N-1;
    next_free=mask&~(ld<<1|rd>>1|col);
    if(next_free){ total+=SQBkBlBjrB(ld<<1,rd>>1,col,row,next_free,jmark,endmark,mark1,mark2,mask,N); }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){
      total+=SQBjlBkBlBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N);
    }
  }
  return total;
}
__host__ __device__ unsigned int SQBjlBlBkBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==N-1-jmark){
    rd|=1<<N-1;
    next_free=mask&~ld<<1|rd>>1|col;
    if(next_free){ total+=SQBlBkBjrB(ld<<1,rd>>1,col,row,next_free,jmark,endmark,mark1,mark2,mask,N); }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){ total+=SQBjlBlBkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}
__host__ __device__ unsigned int SQBjlBklBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==N-1-jmark){
    rd|=1<<N-1;
    next_free=mask&~(ld<<1|rd>>1|col);
    if(next_free){ total+=SQBklBjrB(ld<<1,rd>>1,col,row,next_free,jmark,endmark,mark1,mark2,mask,N); }
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~((ld|bit)<<1|(rd|bit)>>1|col|bit);
    if(next_free){ total+=SQBjlBklBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}
__host__ __device__ unsigned int SQBjlBlkBjrB(unsigned int ld,unsigned int rd,unsigned int col,unsigned int row,register unsigned int free,unsigned int jmark,unsigned int endmark,unsigned int mark1,unsigned int mark2,unsigned int mask,unsigned int N)
{
  unsigned int total=0;
  register int bit;
  unsigned int next_free;
  if(row==N-1-jmark){
    rd|=1<<N-1;
    next_free=mask&~(ld<<1|rd>>1|col);
    total+=SQBlkBjrB(ld<<1,rd>>1,col,row,next_free,jmark,endmark,mark1,mark2,mask,N);
    return total;
  }
  while(free){
    free^=bit=free&-free;
    next_free=mask&~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(next_free){ total+=SQBjlBlkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,next_free,jmark,endmark,mark1,mark2,mask,N); }
  }
  return total;
}
/**
 * メインエントリポイント
 *
 * N-Queensの解探索をCPU/GPU両モードでベンチマーク・集計する。
 * コマンドラインで -c:CPU, -g:GPU の切り替え可（デフォルトGPU）
 *
 * - 各サイズ(size)ごとに：
 *   1. unique部分盤面(ijklList)を生成（genConstellations）
 *   2. パディング＆ソート（fillWithTrash）
 *   3. CPUならexecSolutions()で逐次計算、GPUならexecSolutionsKernelで並列集計
 *   4. 合計値を出力
 *   5. リソースを後処理
 *
 * タイミング計測はgettimeofdayで行い、1行で解数・所要時間等を出力。
 * CUDAメモリ確保/解放、CPU/GPU切り替え、コマンドライン引数の柔軟処理など、現場で求められる機能が一通り揃う。
 *
 * @param argc コマンドライン引数数
 * @param argv コマンドライン引数配列
 * @return 終了コード（正常終了で0）
 */
int main(int argc,char** argv)
{
  bool cpu=false,gpu=false;
  int argstart=2;
  {
    size_t bytes = 1 << 20; // 1MB
    cudaError_t e = cudaDeviceSetLimit(cudaLimitStackSize, bytes);
    if (e != cudaSuccess) {
      fprintf(stderr, "[warn] cudaDeviceSetLimit(StackSize,%zu) failed: %s\n",bytes, cudaGetErrorString(e));
      // 失敗しても続行（この後のクラッシュを防ぐため）
      cudaGetLastError();
    }
  }


  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='g'||argv[1][1]=='G'){gpu=true;}
    else{ gpu=true; } //デフォルトをgpuとする
    argstart=2;
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g] n steps\n",argv[0]);
    printf("  -c: CPU\n");
    printf("  -g: GPU\n");
  }
  if(cpu){ printf("\n\nCPU Constellations\n"); }
  else if(gpu){ printf("\n\nGPU Constellations\n");
   if(!InitCUDA()){return 0;}
  }
    int min=4; 
    int targetN=20;
    struct timeval t0;
    struct timeval t1;
    printf("%s\n"," N:            Total          Unique      dd:hh:mm:ss.ms");
    IntHashSet* ijklList;
    ConstellationArrayList* constellations;
    long TOTAL;
    long UNIQUE;
    int ss;
    int ms;
    int dd;
    int hh;
    int mm;
    for(unsigned int size=min;size<=targetN;++size){
      ijklList=create_int_hashset();
      constellations=create_constellation_arraylist();
      TOTAL=0;
      UNIQUE=0;
      gettimeofday(&t0,NULL);
      genConstellations(ijklList,constellations,size);
      // ソート
      ConstellationArrayList* fillconstellations = fillWithTrash(constellations, THREAD_NUM);	
      if(cpu){    
    	execSolutions(fillconstellations,size);
    	TOTAL=calcSolutions(fillconstellations,TOTAL);
      }
      if(gpu){
        int steps=24576;
	      int totalSize = fillconstellations->size;
        for(unsigned int offset = 0; offset < totalSize; offset += steps) {
      	  unsigned int currentSize = fmin(steps, totalSize - offset);
          int gridSize = (currentSize + THREAD_NUM - 1) / THREAD_NUM;  // グリッドサイズ
          unsigned int* hostTotal;
          cudaMallocHost((void**) &hostTotal,sizeof(int)*gridSize);
          unsigned int* deviceTotal;
          cudaMalloc((void**) &deviceTotal,sizeof(int)*gridSize);

          Constellation* deviceMemory;
          cudaMalloc((void**)&deviceMemory, currentSize * sizeof(Constellation));
          // デバイスにコピー
          cudaMemcpy(deviceMemory, &fillconstellations->data[offset], currentSize * sizeof(Constellation), cudaMemcpyHostToDevice);
          // カーネルを実行
          // segmantation falt対応
          // size_t bytes=1<<20;
          // cudaDeviceSetLimit(cudaLimitStackSize,bytes); // 1MB 目安。足りなければ増やす
          execSolutionsKernel<<<gridSize, THREAD_NUM>>>(deviceMemory,deviceTotal, size, currentSize);
          cudaError_t e = cudaGetLastError();
          if (e != cudaSuccess) {
            fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(e));
            exit(1);
          }
          e = cudaDeviceSynchronize();
          if (e != cudaSuccess) {
            fprintf(stderr, "kernel runtime error: %s\n", cudaGetErrorString(e));
            exit(1);
          }
          cudaDeviceSynchronize(); // 完了待ち
          // カーネル実行後にデバイスメモリからホストにコピー
          cudaMemcpy(hostTotal, deviceTotal, sizeof(int)*gridSize,cudaMemcpyDeviceToHost);
          // 取得したsolutionsをホスト側で集計
          // 取得したsolutionsをホスト側で集計
          for(unsigned int i = 0; i < gridSize; i++) {
            TOTAL += hostTotal[i];
          }
          //cudaFreeを追加
          cudaFree(deviceMemory);
          cudaFree(deviceTotal);
          cudaFreeHost(hostTotal);
        }
     }
     gettimeofday(&t1,NULL);
     if(t1.tv_usec<t0.tv_usec){
       dd=(t1.tv_sec-t0.tv_sec-1)/86400;
       ss=(t1.tv_sec-t0.tv_sec-1)%86400;
       ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
     }else{
       dd=(t1.tv_sec-t0.tv_sec)/86400;
       ss=(t1.tv_sec-t0.tv_sec)%86400;
       ms=(t1.tv_usec-t0.tv_usec+500)/10000;
     }
     hh=ss/3600;
     mm=(ss-hh*3600)/60;
     ss%=60;
     printf("%2d:%17ld%16ld%8.3d:%02d:%02d:%02d.%02d\n",size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
     // 後処理
     free_int_hashset(ijklList);
     free_constellation_arraylist(constellations);
  } 
  return 0;
}

