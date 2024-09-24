/**
 N:            Total          Unique      dd:hh:mm:ss.ms
 4:                0               0     000:00:00:00.31
 5:               18               0     000:00:00:00.00
 6:                4               0     000:00:00:00.00
 7:               40               0     000:00:00:00.00
 8:               92               0     000:00:00:00.00
 9:              352               0     000:00:00:00.00
10:              724               0     000:00:00:00.00
11:             2680               0     000:00:00:00.00
12:            14200               0     000:00:00:00.00
13:            73712               0     000:00:00:00.00
14:           365596               0     000:00:00:00.01
15:          2279184               0     000:00:00:00.09
16:         14772512               0     000:00:00:00.64
17:         95815104               0     000:00:00:03.41
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

// CUDA 初期化
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
 * 大小を比較して小さい最値を返却
 */
// #define fmin(a,b) (((a)<(b)) ? (a) : (b))
//#include <math.h>
int ffmin(int a,int b)
{
  if(a<b){
    return a;
  }
  return b;
}
// #define toijkl(i,j,k,l)  ( (i<<15)+(j<<10)+(k<<5)+l )
int toijkl(int i,int j,int k,int l){ return (i<<15)+(j<<10)+(k<<5)+l; }
// #define geti(ijkl) ( ijkl>>15 )
__host__ __device__ int geti(int ijkl){ return ijkl>>15; }
// #define getj(ijkl) ( (ijkl>>10) & 31 )
__host__ __device__ int getj(int ijkl){ return (ijkl>>10) & 31; }
// #define getk(ijkl) ( (ijkl>>5) & 31 )
__host__ __device__ int getk(int ijkl){ return (ijkl>>5) & 31; }
// #define getl(ijkl) ( ijkl & 31 )
__host__ __device__ int getl(int ijkl){ return ijkl & 31; }
/**
  左右のミラー 与えられたクイーンの配置を左右ミラーリングします。
  各クイーンの位置を取得し、列インデックスを N-1 から引いた位置に変更します（左右反転）。
  行インデックスはそのままにします。
*/
//#define mirvert(ijkl,N) ( toijkl(N-1-geti(ijkl),N-1-getj(ijkl),getl(ijkl),getk(ijkl)) )
int mirvert(int ijkl,int N){ return toijkl(N-1-geti(ijkl),N-1-getj(ijkl),getl(ijkl),getk(ijkl)); }
/**
  時計回りに90度回転
  rot90 メソッドは、90度の右回転（時計回り）を行います
  元の位置 (row,col) が、回転後の位置 (col,N-1-row) になります。
*/
//#define rot90(ijkl,N) ( ((N-1-getk(ijkl))<<15)+((N-1-getl(ijkl))<<10)+(getj(ijkl)<<5)+geti(ijkl) )
int rot90(int ijkl,int N){ return ((N-1-getk(ijkl))<<15)+((N-1-getl(ijkl))<<10)+(getj(ijkl)<<5)+geti(ijkl); }
/**
  対称性のための計算と、ijklを扱うためのヘルパー関数。
  開始コンステレーションが回転90に対して対称である場合
*/
// #define symmetry90(ijkl,N) ( ((geti(ijkl)<<15)+(getj(ijkl)<<10)+(getk(ijkl)<<5)+getl(ijkl)) == (((N-1-getk(ijkl))<<15)+((N-1-getl(ijkl))<<10)+(getj(ijkl)<<5)+geti(ijkl)) )
__host__ __device__ int symmetry90(int ijkl,int N){
  return ((geti(ijkl)<<15)+(getj(ijkl)<<10)+(getk(ijkl)<<5)+getl(ijkl)) == (((N-1-getk(ijkl))<<15)+((N-1-getl(ijkl))<<10)+(getj(ijkl)<<5)+geti(ijkl));
}
/**
  この開始コンステレーションで、見つかった解がカウントされる頻度
*/
// #define symmetry(ijkl,N) ( (geti(ijkl)==N-1-getj(ijkl) && getk(ijkl)==N-1-getl(ijkl)) ? (symmetry90(ijkl,N) ? 2 : 4 ) : 8 )
__host__ __device__ int symmetry(int ijkl,int N){
  // コンステレーションをrot180で対称に開始するか？
  if(geti(ijkl)==N-1-getj(ijkl) && getk(ijkl)==N-1-getl(ijkl)){
    if(symmetry90(ijkl,N)){
      return 2;
    }else{
      return 4;
    }
  }else{
    return 8;
  }
}
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
// IntHashSetの関数プロトタイプ
//IntHashSet* create_int_hashset();
void free_int_hashset(IntHashSet* set);
int int_hashset_contains(IntHashSet* set,int value);
void int_hashset_add(IntHashSet* set,int value);
// ビット操作関数プロトタイプ
int checkRotations(IntHashSet* set,int i,int j,int k,int l,int N);
int toijkl(int i,int j,int k,int l);
__device__ int geti(int sc);
__device__ int getj(int sc);
__device__ int getk(int sc);
__device__ int getl(int sc);
int rot90(int ijkl,int N);
int symmetry90(int ijkl,int N);
__device__ int symmetry(int ijkl,int N);
int mirvert(int ijkl,int N);
*/
/**
// ID
int get_id(Constellation* constellation){ return constellation->id; }
void set_id(Constellation* constellation,int id){ constellation->id=id; }
// LD
int get_ld(Constellation* constellation){ return constellation->ld; }
void set_ld(Constellation* constellation,int ld){ constellation->ld=ld; }
// RD
int get_rd(Constellation* constellation){ return constellation->rd; }
void set_rd(Constellation* constellation,int rd){ constellation->rd=rd; }
// COL
int get_col(Constellation* constellation){ return constellation->col; }
void set_col(Constellation* constellation,int col){ constellation->col=col; }
// startIJKL
int get_startijkl(Constellation* constellation){ return constellation->startijkl; }
void set_startijkl(Constellation* constellation,int startijkl){ constellation->startijkl=startijkl; }
// solutions
long get_solutions(Constellation* constellation){ return constellation->solutions; }
void set_solutions(Constellation* constellation,long solutions){ constellation->solutions=solutions; }
// IJKL
int get_ijkl(Constellation* constellation){
  return constellation->startijkl & 0xFFFFF;// Equivalent to 0b11111111111111111111
}
**/
void setPreQueens(int ld,int rd,int col,int k,int l,int row,int queens,int LD,int RD,int *counter,ConstellationArrayList* constellations,int N);
void execSolutions(ConstellationArrayList* constellations,int N);
void genConstellations(IntHashSet* ijklList,ConstellationArrayList* constellations,int N);
long calcSolutions(ConstellationArrayList* constellations,long solutions);
int jasmin(int ijkl,int N);
void add_constellation(int ld,int rd,int col,int startijkl,ConstellationArrayList* constellations);
/**
 * 関数プロトタイプ
 */
__device__ void SQBkBlBjrB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQBklBjrB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQBlBjrB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQBjrB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQBlBkBjrB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQBlkBjrB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQBkBjrB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQBjlBkBlBjrB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQBjlBklBjrB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQBjlBlBkBjrB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQBjlBlkBjrB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQd2BkBlB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQd2BklB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQd2BlB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQd2B(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQd2BlBkB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQd2BlkB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQd1BkBlB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQd1BklB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQd1BlB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQd1B(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQd1BlBkB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQd1BlkB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQd0B(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQd0BkB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQd2BkB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );
__device__ void SQd1BkB(
    int ld,int rd,int col,int start,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    );

/**
 * IntHashSet の関数実装
 */
IntHashSet* create_int_hashset(){
  IntHashSet* set=(IntHashSet*)malloc(sizeof(IntHashSet));
  set->data=(int*)malloc(INITIAL_CAPACITY * sizeof(int));
  set->size=0;
  set->capacity=INITIAL_CAPACITY;
  return set;
}
/**
 *
 */
void free_int_hashset(IntHashSet* set){
  free(set->data);
  free(set);
}
/**
 *
 */
int int_hashset_contains(IntHashSet* set,int value){
  for(int i=0;i<set->size;i++){
    if(set->data[i]==value){ return 1; }
  }
  return 0;
}
/**
 *
 */
void int_hashset_add(IntHashSet* set,int value){
  if(!int_hashset_contains(set,value)){
    if(set->size==set->capacity){
      set->capacity *= 2;
      set->data=(int*)realloc(set->data,set->capacity * sizeof(int));
    }
    set->data[set->size++]=value;
  }
}
/**
 * ConstellationArrayList の関数実装
 */
ConstellationArrayList* create_constellation_arraylist(){
  ConstellationArrayList* list=(ConstellationArrayList*)malloc(sizeof(ConstellationArrayList));
  list->data=(Constellation*)malloc(INITIAL_CAPACITY * sizeof(Constellation));
  list->size=0;
  list->capacity=INITIAL_CAPACITY;
  return list;
}
/**
 *
 */
void free_constellation_arraylist(ConstellationArrayList* list){
  free(list->data);
  free(list);
}
/**
 *
 */
void constellation_arraylist_add(ConstellationArrayList* list,Constellation value){
  if(list->size==list->capacity){
    list->capacity *= 2;
    list->data=(Constellation*)realloc(list->data,list->capacity * sizeof(Constellation));
  }
  list->data[list->size++]=value;
}
/**
 *
 */
Constellation* create_constellation(){
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
 *
 */
Constellation* create_constellation_with_values(int id,int ld,int rd,int col,int startijkl,long solutions){
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
 *
 */
void add_constellation(int ld,int rd,int col,int startijkl,ConstellationArrayList* constellations){
  Constellation new_constellation={0,ld,rd,col,startijkl,-1};
  constellation_arraylist_add(constellations,new_constellation);
}
// コンステレーションの比較関数
int compareConstellations(const void* a, const void* b) {
    Constellation* const1 = (Constellation*)a;
    Constellation* const2 = (Constellation*)b;

    // startijkl の最初の 15 ビットを取得
    int jkl1 = const1->startijkl & ((1 << 15) - 1);
    int jkl2 = const2->startijkl & ((1 << 15) - 1);

    // jkl に基づいてソート
    if (jkl1 < jkl2) {
        return -1;
    } else if (jkl1 > jkl2) {
        return 1;
    } else {
        return 0;
    }
}



// コンステレーションリストをソートする関数
void sortConstellations(ConstellationArrayList* constellations) {
    // qsort を使ってソート
    qsort(constellations->data, constellations->size, sizeof(Constellation), compareConstellations);
}


// トラッシュコンステレーションを追加
void addTrashConstellation(ConstellationArrayList* list, int ijkl) {
    // トラッシュ用のダミーコンステレーションを作成
    int ld = -1;
    int rd = -1;
    int col = -1;
    int startijkl = (69 << 20) | ijkl;
    // トラッシュコンステレーションをリストに追加
    add_constellation(ld, rd, col, startijkl, list);
}

// fillWithTrash 関数
ConstellationArrayList* fillWithTrash(ConstellationArrayList* constellations, int workgroupSize) {
    // コンステレーションのリストをソート
    sortConstellations(constellations);

    // 新しいリストを作成
    ConstellationArrayList* newConstellations = create_constellation_arraylist();

    // 最初のコンステレーションの currentJkl を取得
    int currentJkl = constellations->data[0].startijkl & ((1 << 15) - 1);

    // 各コンステレーションに対してループ
    for (int i = 0; i < constellations->size; i++) {
        Constellation c = constellations->data[i];
        // 既にソリューションがあるものは無視
        if (c.solutions >= 0) continue;

        // 新しい ijkl グループの開始を確認
        if ((c.startijkl & ((1 << 15) - 1)) != currentJkl) {
            // workgroupSize の倍数になるまでトラッシュを追加
            while (newConstellations->size % workgroupSize != 0) {
                addTrashConstellation(newConstellations, currentJkl);
            }
            currentJkl = c.startijkl & ((1 << 15) - 1);
        }

        // コンステレーションを追加
        add_constellation(c.ld, c.rd, c.col, c.startijkl, newConstellations);
    }

    // 最後に残った分を埋める
    while (newConstellations->size % workgroupSize != 0) {
        addTrashConstellation(newConstellations, currentJkl);
    }

    return newConstellations;
}



/**
  3つまたは4つのクイーンを使って開始コンステレーションごとにサブコンステレー
  ションを生成する。この関数 setPreQueens は、与えられた配置に基づいて、指定
  された数のクイーン (presetQueens) を配置するためのサブコンステレーション
  （部分配置）を生成します。この関数は再帰的に呼び出され、ボード上のクイーン
  の配置を計算します。ボード上に3つまたは4つのクイーンを使って、開始コンステ
  レーションからサブコンステレーションを生成します。
  ld: 左対角線のビットマスク。
  rd: 右対角線のビットマスク。
  col: 列のビットマスク。
  k: クイーンを配置する行の1つ目のインデックス。
  l: クイーンを配置する行の2つ目のインデックス。
  row: 現在の行のインデックス。
  queens: 現在配置されているクイーンの数。
*/
void setPreQueens(int ld,int rd,int col,int k,int l,int row,int queens,int LD,int RD,int *counter,ConstellationArrayList* constellations,int N){
  int mask=(1<<N)-1;//setPreQueensで使用
  // k行とl行はさらに進む
  if(row==k || row==l){
    setPreQueens(ld<<1,rd>>1,col,k,l,row+1,queens,LD,RD,counter,constellations,N);
    return;
  }
  /**
    preQueensのクイーンが揃うまでクイーンを追加する。
    現在のクイーンの数が presetQueens に達した場合、
    現在の状態を新しいコンステレーションとして追加し、カウンターを増加させる。
  */
  if(queens==presetQueens){
    // リストに４個クイーンを置いたセットを追加する
    add_constellation(ld,rd,col,row<<20,constellations);
    (*counter)++;
    return;
  }
  // k列かl列が終わっていなければ、クイーンを置いてボードを占領し、さらに先に進む。
  else{
    // 現在の行にクイーンを配置できる位置（自由な位置）を計算
    int free=~(ld | rd | col | (LD>>(N-1-row)) | (RD<<(N-1-row))) & mask;
    int bit;
    while(free){
      bit=free & (-free);
      free -= bit;
      // クイーンをおける場所があれば、その位置にクイーンを配置し、再帰的に次の行に進む
      setPreQueens((ld | bit)<<1,(rd | bit)>>1,col | bit,k,l,row+1,queens+1,LD,RD,counter,constellations,N);
    }
  }
}
/**
  いずれかの角度で回転させた座標がすでに見つかっている場合、trueを返す。
 */
int checkRotations(IntHashSet* ijklList,int i,int j,int k,int l,int N){
  int rot90=((N-1-k)<<15)+((N-1-l)<<10)+(j<<5)+i;
  int rot180=((N-1-j)<<15)+((N-1-i)<<10)+((N-1-l)<<5)+(N-1-k);
  int rot270=(l<<15)+(k<<10)+((N-1-i)<<5)+(N-1-j);
  if(int_hashset_contains(ijklList,rot90)){ return 1; }
  if(int_hashset_contains(ijklList,rot180)){ return 1; }
  if(int_hashset_contains(ijklList,rot270)){ return 1; }
  return 0;
}
/**
  i,j,k,lをijklに変換し、特定のエントリーを取得する関数
  各クイーンの位置を取得し、最も左上に近い位置を見つけます
  最小の値を持つクイーンを基準に回転とミラーリングを行い、配置を最も左上に近
  い標準形に変換します。
  最小値を持つクイーンの位置を最下行に移動させる
  i は最初の行（上端） 90度回転2回
  j は最後の行（下端） 90度回転0回
  k は最初の列（左端） 90度回転3回
  l は最後の列（右端） 90度回転1回
  優先順位が l>k>i>j の理由は？
  l は右端の列に位置するため、その位置を基準に回転させることで、配置を最も標
  準形に近づけることができます。
  k は左端の列に位置しますが、l ほど標準形に寄せる影響が大きくないため、次に
  優先されます。
  i は上端の行に位置するため、行の位置を基準にするよりも列の位置を基準にする
  方が配置の標準化に効果的です。
  j は下端の行に位置するため、優先順位が最も低くなります。
*/
int jasmin(int ijkl,int N){
  //j は最後の行（下端） 90度回転0回
  int min=ffmin(getj(ijkl),N-1-getj(ijkl));
  int arg=0;
  //i は最初の行（上端） 90度回転2回
  if(ffmin(geti(ijkl),N-1-geti(ijkl))<min){
    arg=2;
    min=ffmin(geti(ijkl),N-1-geti(ijkl));
  }
  //k は最初の列（左端） 90度回転3回
  if(ffmin(getk(ijkl),N-1-getk(ijkl))<min){
    arg=3;
    min=ffmin(getk(ijkl),N-1-getk(ijkl));
  }
  //l は最後の列（右端） 90度回転1回
  if(ffmin(getl(ijkl),N-1-getl(ijkl))<min){
    arg=1;
    min=ffmin(getl(ijkl),N-1-getl(ijkl));
  }
  for(int i=0;i<arg;i++){
    ijkl=rot90(ijkl,N);
  }
  if(getj(ijkl)<N-1-getj(ijkl)){
    ijkl=mirvert(ijkl,N);
  }
  return ijkl;
}
/**
 *
 */
long calcSolutions(ConstellationArrayList* constellations,long solutions){
  Constellation* c;
  for(int i=0;i<constellations->size;i++){
    c=&constellations->data[i];
    if(c->solutions > 0){
      solutions += c->solutions;
    }
  }
  return solutions;
}
__global__ void execSolutionsKernel(Constellation* constellations,int N, int totalSize){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 範囲外アクセスのチェック
    if (idx >= totalSize) return;

    Constellation* constellation = &constellations[idx];

    int j = getj(constellation->startijkl);
    int k = getk(constellation->startijkl);
    int l = getl(constellation->startijkl);
    int ijkl = constellation->startijkl & ((1 << 20) - 1);
    int ld = constellation->ld >> 1;
    int rd = constellation->rd >> 1;
    int col = (constellation->col >> 1) | (~((1 << (N - 2)) - 1));
    long tempcounter = 0;

    int start = constellation->startijkl >> 20;
    int LD = (1 << (N - 1) >> j) | (1 << (N - 1) >> l);

    ld |= LD>>(N-start);

    if(start>k){
      rd |= (1<<(N-1)>>(start-k+1));
    }
    if(j >= 2 * N-33-start){// クイーンjからのrdがない場合のみ追加する
      rd |= (1<<(N-1)>>j)<<(N-2-start);// 符号ビットを占有する！
    }

    int free=~(ld | rd | col);
    int jmark = j + 1;
    int endmark = N - 2;
    int mark1, mark2;

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
                SQBkBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
              }else{// kとlの間に空行がない場合
                SQBklBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
              }
            }else{// もしkがすでに開始前に来ていて、lだけが残っている場合
              SQBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
            }
          }else{// kとlの両方が開始前にすでに来ていた場合
            SQBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
          }
        }else{// l<k
          mark1=l-1;
          mark2=k-1;
          if(start<k){// 少なくともkがまだ来ていない場合
            if(start<l){// lがまだ来ていない場合
              if(k != l+1){// lとkの間に少なくとも1つの自由行がある場合
                SQBlBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
              }else{// lとkの間に自由行がない場合
                SQBlkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
              }
            }else{ // lがすでに来ていて、kだけがまだ来ていない場合
              SQBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
            }
          }else{// lとkの両方が開始前にすでに来ていた場合
            SQBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
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
            SQBjlBkBlBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
          }else{// lがkの直後に来る場合
            SQBjlBklBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
          }
        }else{  // l<k
          mark1=l-1;
          mark2=k-1;
          if(k != l+1){// l行とk行の間には、少なくともefree行が存在する。
            SQBjlBlBkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
          }else{// kがlの直後に来る場合
            SQBjlBlkBjrB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
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
              SQd2BkBlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
            }else{
              SQd2BklB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
            }
          }else{// k が開始前に設定されていた場合
            mark2=l-1;
            SQd2BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
          }
        }else{ // もしkとlが開始前にすでに来ていた場合
          SQd2B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
        }
      }else{// l<k
        mark1=l-1;
        mark2=k-1;
        endmark=N-2;
        if(start<k){// 少なくともkがまだ来ていない場合
          if(start<l){// lがまだ来ていない場合
            if(k != l+1){// lとkの間に空行がある場合
              SQd2BlBkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
            }else{// lとkの間に空行がない場合
              SQd2BlkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
            }
          }else{ // l が開始前に来た場合
            mark2=k-1;
            SQd2BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
          }
        }else{ // lとkの両方が開始前にすでに来ていた場合
          SQd2B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
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
              SQd1BkBlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
            }else{
              SQd1BklB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
            }
          }else{// lがまだ来ていないなら
            mark2=l-1;
            SQd1BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
          }
        }else{// すでにkとlが来ている場合
          SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
        }
      }else{ // l<k
        if(start<k){// 少なくともkがまだ来ていない場合
          if(start<l){ // lがまだ来ていない場合
            if(k<N-2){// kが末尾にない場合
              mark1=l-1;
              endmark=N-2;

              if(k != l+1){// lとkの間に空行がある場合
                mark2=k-1;
                SQd1BlBkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
              }else{// lとkの間に空行がない場合
                SQd1BlkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
              }
            }else{// kが末尾の場合
              if(l != (N-3)){// lがkの直前でない場合
                mark2=l-1;
                endmark=(N-3);
                SQd1BlB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
              }else{// lがkの直前にある場合
                endmark=(N-4);
                SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
              }
            }
          }else{ // もしkがまだ来ていないなら
            if(k != N-2){// kが末尾にない場合
              mark2=k-1;
              endmark=N-2;
              SQd1BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
            }else{// kが末尾の場合
              endmark=(N-3);
              SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
            }
          }
        }else{// kとlはスタートの前
          endmark=N-2;
          SQd1B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
        }
      }
    }else{// クイーンjがコーナーに置かれている場合
      endmark=N-2;
      if(start>k){
        SQd0B(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
      }else{
        /**
          クイーンをコーナーに置いて星座を組み立てる方法と、ジャスミンを適用
          する方法によって、Kは最後列に入ることはできない。
        */
        mark1=k-1;
        SQd0BkB(ld,rd,col,start,free,jmark,endmark,mark1,mark2,&tempcounter,N);
      }
    }
    // 完成した開始コンステレーションを削除する。
    constellation->solutions=tempcounter * symmetry(ijkl,N);

}
/**
 *
 */

/**
 *
 */
void genConstellations(IntHashSet* ijklList,ConstellationArrayList* constellations,int N){
  int halfN=(N+1) / 2;// N の半分を切り上げる
  int L=1<<(N-1);//Lは左端に1を立てる
  /**
    コーナーにクイーンがいない場合の開始コンステレーションを計算する
    最初のcolを通過する
    k: 最初の列（左端）に配置されるクイーンの行のインデックス。
  */
  for(int k=1;k<halfN;k++){
    /**
      l: 最後の列（右端）に配置されるクイーンの行のインデックス。
      l を k より後の行に配置する理由は、回転対称性を考慮して配置の重複を避け
      るためです。
      このアプローチにより、探索空間が効率化され、N-クイーン問題の解決が迅速
      かつ効率的に行えるようになります。
      最後のcolを通過する
    */
    for(int l=k+1;l<(N-1);l++){
      /**
        i: 最初の行（上端）に配置されるクイーンの列のインデックス。
        最初の行を通過する
        k よりも下の行に配置することで、ボード上の対称性や回転対称性を考慮し
        て、重複した解を避けるための配慮がされています。
      */
      for(int i=k+1;i<(N-1);i++){
        // i==N-1-lは、行iが列lの「対角線上」にあるかどうかをチェックしています。
        if(i==(N-1)-l){
          continue;
        }
        /**
            j: 最後の行（下端）に配置されるクイーンの列のインデックス。
            最後の行を通過する
        */
        for(int j=N-k-2;j>0;j--){
        /**
          同じ列や行にクイーンが配置されている場合は、その配置が有効でない
          ためスキップ
        */
          if(j==i || l==j){
            continue;
          }
          /**
            回転対称でスタートしない場合
            checkRotationsで回転対称性をチェックし、対称でない場合にijklList
            に配置を追加します。
          */
          if(!checkRotations(ijklList,i,j,k,l,N)){
            int_hashset_add(ijklList,toijkl(i,j,k,l));
          }
        }
      }
    }
  }
  /**
    コーナーにクイーンがある場合の開始コンステレーションを計算する
    最初のクイーンを盤面の左上隅（0,0）に固定
    j は最後の行に置かれるクイーンの列インデックスです。これは 1 から N-3 ま
    での値を取ります。
  */
  for(int j=1;j<N-2;j++){// jは最終行のクイーンのidx
    for(int l=j+1;l<(N-1);l++){// lは最終列のクイーンのidx
      int_hashset_add(ijklList,toijkl(0,j,0,l));
    }
  }
  IntHashSet* ijklListJasmin=create_int_hashset();
  int startConstellation;
  for(int i=0;i<ijklList->size;i++){
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
  for(int s=0;s<ijklList->size;s++){
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
    for(int a=0;a<counter;a++){
      constellations->data[currentSize-a-1].startijkl |= toijkl(i,j,k,l);
    }
  }
}
/**
 * 未使用変数対応
 */
void f(int unuse,char* argv[]){
  printf("%d%s\n",unuse,argv[0]);
}
/**
 * メインメソッド
 */
int main(int argc,char** argv){
  if(!InitCUDA()){return 0;}
  f(argc,argv);
  int min=4;
  int targetN=17;
  int workgroupSize=64;
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
  for(int size=min;size<=targetN;++size){
    ijklList=create_int_hashset();
    constellations=create_constellation_arraylist();
    TOTAL=0;
    UNIQUE=0;
    gettimeofday(&t0,NULL);
    genConstellations(ijklList,constellations,size);
    // ソート
    ConstellationArrayList* fillconstellations = fillWithTrash(constellations, workgroupSize);
    int totalSize = fillconstellations->size;
    int steps=24576;
    for (int offset = 0; offset < totalSize; offset += steps) {
      int currentSize = fmin(steps, totalSize - offset);
      Constellation* deviceMemory;
      cudaMalloc((void**)&deviceMemory, currentSize * sizeof(Constellation));
      // デバイスにコピー
      cudaMemcpy(deviceMemory, &fillconstellations->data[offset], currentSize * sizeof(Constellation), cudaMemcpyHostToDevice);

      int blockSize = workgroupSize;  // スレッド数
      int gridSize = (currentSize + blockSize - 1) / blockSize;  // グリッドサイズ
      // カーネルを実行
      execSolutionsKernel<<<gridSize, blockSize>>>(deviceMemory, size, currentSize);

      // カーネル実行後にデバイスメモリからホストにコピー
      cudaMemcpy(&fillconstellations->data[offset], deviceMemory, currentSize * sizeof(Constellation), cudaMemcpyDeviceToHost);

      // 取得したsolutionsをホスト側で集計
      for (int i = 0; i < currentSize; i++) {
        TOTAL += fillconstellations->data[offset + i].solutions;
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
    free_constellation_arraylist(fillconstellations);
  }
  return 0;
}
/**
 * 関数プロトタイプ
 */
__device__ void SQd0B(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  if(row==endmark){
    (*tempcounter)++;
    return;
  }
  int bit;
  int nextfree;
  int next_ld;
  int next_rd;
  int next_col;
  while(free){
    free-=bit=free&(-free);;
    next_ld=((ld|bit)<<1);
    next_rd=((rd|bit)>>1);
    next_col=(col|bit);
    nextfree=~(next_ld|next_rd|next_col);
    if(nextfree){
      if(row<endmark-1){
        if(~((next_ld<<1)|(next_rd>>1)|(next_col))>0)
          SQd0B(next_ld,next_rd,next_col,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }else{
        SQd0B(next_ld,next_rd,next_col,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }
    }
  }
}
__device__ void SQd0BkB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int N3=N-3;
  int bit;
  int nextfree;
  if(row==mark1){
    while(free){
      free-=bit=free&(-free);;
      nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1<<(N3));
      if(nextfree){
        SQd0B((ld|bit)<<2,((rd|bit)>>2)|1<<(N3),col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }
    }
    return;
  }
  while(free){
    free-=bit=free&(-free);;
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQd0BkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQd1BklB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int N4=N-4;
  int bit;
  int nextfree;
  if(row==mark1){
    while(free){
      free-=bit=free&(-free);;
      nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|1|1<<(N4));
      if(nextfree){
        SQd1B(((ld|bit)<<3)|1,((rd|bit)>>3)|1<<(N4),col|bit,row+3,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }
    }
    return;
  }
  while(free){
    free-=bit=free&(-free);;
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQd1BklB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQd1B(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  if(row==endmark){
    (*tempcounter)++;
    return;
  }
  int bit;
  int nextfree;
  int next_ld;
  int next_rd;
  int next_col;
  while(free){
    free-=bit=free&(-free);;
    next_ld=((ld|bit)<<1);
    next_rd=((rd|bit)>>1);
    next_col=(col|bit);
    nextfree=~(next_ld|next_rd|next_col);
    if(nextfree){
      if(row+1<endmark){
        if(~((next_ld<<1)|(next_rd>>1)|(next_col))>0)
          SQd1B(next_ld,next_rd,next_col,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }else{
        SQd1B(next_ld,next_rd,next_col,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }
    }
  }
}
__device__ void SQd1BkBlB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int N3=N-3;
  int bit;
  int nextfree;
  if(row==mark1){
    while(free){
      free-=bit=free&(-free);;
      nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1<<(N3));
      if(nextfree){
        SQd1BlB(((ld|bit)<<2),((rd|bit)>>2)|1<<(N3),col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }
    }
    return;
  }
  while(free){
    free-=bit=free&(-free);
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQd1BkBlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQd1BlB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int bit;
  int nextfree;
  int next_ld;
  int next_rd;
  int next_col;
  if(row==mark2){
    while(free){
      free-=bit=free&(-free);
      next_ld=((ld|bit)<<2)|1;
      next_rd=((rd|bit)>>2);
      next_col=(col|bit);
      nextfree=~(next_ld|next_rd|next_col);
      if(nextfree){
        if(row+2<endmark){
          if(~((next_ld<<1)|(next_rd>>1)|(next_col))>0)
            SQd1B(next_ld,next_rd,next_col,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
        }else{
          SQd1B(next_ld,next_rd,next_col,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
        }
      }
    }
    return;
  }
  while(free){
    free-=bit=free&(-free);
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQd1BlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQd1BlkB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int N3=N-3;
  int bit;
  int nextfree;
  if(row==mark1){
    while(free){
      free-=bit=free&(-free);;
      nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|2|1<<(N3));
      if(nextfree){
        SQd1B(((ld|bit)<<3)|2,((rd|bit)>>3)|1<<(N3),col|bit,row+3,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }
    }
    return;
  }
  while(free){
    free-=bit=free&(-free);
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQd1BlkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQd1BlBkB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int bit;
  int nextfree;
  if(row==mark1){
    while(free){
      free-=bit=free&(-free);
      nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1);
      if(nextfree){
        SQd1BkB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }
    }
    return;
  }
  while(free){
    free-=bit=free&(-free);
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQd1BlBkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQd1BkB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int N3=N-3;
  int bit;
  int nextfree;
  if(row==mark2){
    while(free){
      free-=bit=free&(-free);
      nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1<<(N3));
      if(nextfree){
        SQd1B(((ld|bit)<<2),((rd|bit)>>2)|1<<(N3),col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }
    }
    return;
  }
  while(free){
    free-=bit=free&(-free);
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQd1BkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQd2BlkB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int N3=N-3;
  int bit;
  int nextfree;
  if(row==mark1){
    while(free){
      free-=bit=free&(-free);
      nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|1<<(N3)|2);
      if(nextfree){
        SQd2B(((ld|bit)<<3)|2,((rd|bit)>>3)|1<<(N3),col|bit,row+3,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }
    }
    return;
  }
  while(free){
    free-=bit=free&(-free);
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQd2BlkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQd2BklB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int N4=N-4;
  int bit;
  int nextfree;
  if(row==mark1){
    while(free){
      free-=bit=free&(-free);
      nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|1<<(N4)|1);
      if(nextfree){
        SQd2B(((ld|bit)<<3)|1,((rd|bit)>>3)|1<<(N4),col|bit,row+3,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }
    }
    return;
  }
  while(free){
    free-=bit=free&(-free);
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQd2BklB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQd2BkB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int N3=N-3;
  int bit;
  int nextfree;
  if(row==mark2){
    while(free){
      free-=bit=free&(-free);
      nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1<<(N3));
      if(nextfree){
        SQd2B(((ld|bit)<<2),((rd|bit)>>2)|1<<(N3),col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }
    }
    return;
  }
  while(free){
    free-=bit=free&(-free);
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQd2BkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQd2BlBkB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int bit;
  int nextfree;
  if(row==mark1){
    while(free){
      free-=bit=free&(-free);
      nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1);
      if(nextfree){
        SQd2BkB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }
    }
    return;
  }
  while(free){
    free-=bit=free&(-free);
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQd2BlBkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQd2BlB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int bit;
  int nextfree;
  if(row==mark2){
    while(free){
      free-=bit=free&(-free);
      nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1);
      if(nextfree){
        SQd2B(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }
    }
    return;
  }
  while(free){
    free-=bit=free&(-free);
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQd2BlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQd2BkBlB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int N3=N-3;
  int bit;
  int nextfree;
  if(row==mark1){
    while(free){
      free-=bit=free&(-free);
      nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<(N3)));
      if(nextfree){
        SQd2BlB(((ld|bit)<<2),((rd|bit)>>2)|(1<<(N3)),col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }
    }
    return;
  }
  while(free){
    free-=bit=free&(-free);
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQd2BkBlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQd2B(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  if(row==endmark){
    if((free&(~1))>0){
      (*tempcounter)++;
    }
    return;
  }
  int bit;
  int nextfree;
  int next_ld;
  int next_rd;
  int next_col;
  while(free){
    free-=bit=free&(-free);
    next_ld=((ld|bit)<<1);
    next_rd=((rd|bit)>>1);
    next_col=(col|bit);
    nextfree=~(next_ld|next_rd|next_col);
    if(nextfree){
      if(row<endmark-1){
        if(~((next_ld<<1)|(next_rd>>1)|(next_col))>0)
          SQd2B(next_ld,next_rd,next_col,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }else{
        SQd2B(next_ld,next_rd,next_col,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }
    }
  }
}
__device__ void SQBlBjrB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int bit;
  int nextfree;
  if(row==mark2){
    while(free){
      free-=bit=free&(-free);
      nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1);
      if(nextfree){
        SQBjrB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }
    }
    return;
  }
  while(free){
    free-=bit=free&(-free);
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQBlBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQBkBlBjrB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int N3=N-3;
  int bit;
  int nextfree;
  if(row==mark1){
    while(free){
      free-=bit=free&(-free);
      nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<(N3)));
      if(nextfree){
        SQBlBjrB(((ld|bit)<<2),((rd|bit)>>2)|(1<<(N3)),col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }
    }
    return;
  }
  while(free){
    free-=bit=free&(-free);
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQBkBlBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQBjrB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int bit;
  int nextfree;
  if(row==jmark){
    free&=(~1);
    ld|=1;
    while(free){
      free-=bit=free&(-free);
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree){
        SQB(((ld|bit)<<1),(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }
    }
    return;
  }
  while(free){
    free-=bit=free&(-free);
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  if(row==endmark){
    (*tempcounter)++;
    return;
  }
  int bit;
  int nextfree;
  int next_ld;
  int next_rd;
  int next_col;
  while(free){
    free-=bit=free&(-free);
    next_ld=((ld|bit)<<1);
    next_rd=((rd|bit)>>1);
    next_col=(col|bit);
    nextfree=~(next_ld|next_rd|next_col);
    if(nextfree){
      if(row<endmark-1){
        if(~((next_ld<<1)|(next_rd>>1)|(next_col))>0){
          SQB(next_ld,next_rd,next_col,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
        }
      }else{
        SQB(next_ld,next_rd,next_col,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }
    }
  }
}
__device__ void SQBlBkBjrB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int bit;
  int nextfree;
  if(row==mark1){
    while(free){
      free-=bit=free&(-free);
      nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1);
      if(nextfree){
        SQBkBjrB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }
    }
    return;
  }
  while(free){
    free-=bit=free&(-free);
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQBlBkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQBkBjrB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int bit;
  int nextfree;
  int N3=N-3;
  if(row==mark2){
    while(free){
      free-=bit=free&(-free);
      nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1<<(N3));
      if(nextfree){
        SQBjrB(((ld|bit)<<2),((rd|bit)>>2)|1<<(N3),col|bit,row+2,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }
    }
    return;
  }
  while(free){
    free-=bit=free&(-free);
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQBkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQBklBjrB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int N4=N-4;
  int bit;
  int nextfree;
  if(row==mark1){
    while(free){
      free-=bit=free&(-free);
      nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|1<<(N4)|1);
      if(nextfree){
        SQBjrB(((ld|bit)<<3)|1,((rd|bit)>>3)|1<<(N4),col|bit,row+3,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
      }
    }
    return;
  }
  while(free){
    free-=bit=free&(-free);
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQBklBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQBlkBjrB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int N3=N-3;
  int bit;
  int nextfree;
  if(row==mark1){
    while(free){
      free-=bit=free&(-free);
      nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|1<<(N3)|2);
      if(nextfree)
        SQBjrB(((ld|bit)<<3)|2,((rd|bit)>>3)|1<<(N3),col|bit,row+3,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
    return;
  }
  while(free){
    free-=bit=free&(-free);
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQBlkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQBjlBkBlBjrB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int N1=N-1;
  int bit;
  int nextfree;
  if(row==N1-jmark){
    rd|=1<<(N1);
    free&=~1<<(N1);
    SQBkBlBjrB(ld,rd,col,row,free,jmark,endmark,mark1,mark2,tempcounter,N);
    return;
  }
  while(free){
    free-=bit=free&(-free);
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQBjlBkBlBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQBjlBlBkBjrB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int N1=N-1;
  int bit;
  int nextfree;
  if(row==N1-jmark){
    rd|=1<<(N1);
    free&=~1<<(N1);
    SQBlBkBjrB(ld,rd,col,row,free,jmark,endmark,mark1,mark2,tempcounter,N);
    return;
  }
  while(free){
    free-=bit=free&(-free);
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQBjlBlBkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQBjlBklBjrB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int N1=N-1;
  int bit;
  int nextfree;
  if(row==N1-jmark){
    rd|=1<<(N1);
    free&=~1<<(N1);
    SQBklBjrB(ld,rd,col,row,free,jmark,endmark,mark1,mark2,tempcounter,N);
    return;
  }
  while(free){
    free-=bit=free&(-free);
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQBjlBklBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
__device__ void SQBjlBlkBjrB(
    int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2,long* tempcounter,int N
    )
{
  int N1=N-1;
  int bit;
  int nextfree;
  if(row==N1-jmark){
    rd|=1<<(N1);
    free&=~1<<(N1);
    SQBlkBjrB(ld,rd,col,row,free,jmark,endmark,mark1,mark2,tempcounter,N);
    return;
  }
  while(free){
    free-=bit=free&(-free);
    nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
    if(nextfree){
      SQBjlBlkBjrB( (ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree ,jmark,endmark,mark1,mark2,tempcounter,N);
    }
  }
}
