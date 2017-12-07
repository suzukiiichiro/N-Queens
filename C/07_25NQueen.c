/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)

 <>25．最適化 									        NQueen25() N17=03:63

		コンパイルと実行
		$ make nq25 && ./07_25NQueen

=== 1 ===
 G構造体に格納していた int si int siE int lTotal int lUniqueを
 グローバル変数に置き換えました。ちょっと速くなりました。

=== 2 ===
 L構造体に格納していたC2/C4/C8カウンターの置き場所を変えて比較

1.
// long C2[MAX]; //グローバル環境に置くと N=17: 08.04
// long C4[MAX];
// long C8[MAX];

2.
//  long C2; // 構造体の中の配列をなくすとN=17: 05.24
//  long C4;
//  long C8;

3. 構造体の中でポインタアクセスにしてみる // N=17 : 05.87
   さらにcallocにより、宣言時に適切なメモリサイズを割り当てる
// int *ab; 
//  l[B1].aB=calloc(G.si,sizeof(int));

4.
  long C2[MAX];//構造体の中の配列を活かすと   N=17: 04.33
  long C4[MAX];
  long C8[MAX];

 よって、カウンターはL構造体の中に配置し、スレッド毎にカウンター
を管理する配列で構築しました。
同様に、カウントする箇所は以下のように書き換えました。

			l->C4[l->B1]++;

これによりちょっと速くなりました。

=== 3 ===
　symmetryOps_bm()/trackBack1()/trackBack2()のメソッドないで宣言されている
ローカル変数を撲滅しました。
　symmetryOps_bm()の中では以下の通りです。

  int own,ptn,you,bit;

こちらは全てL構造体でもち、
　l->own などでアクセスするようにしました。構造体に配置すると遅くなる
　という本をよく見ますが、激しく呼び出されるメソッドで変数が都度生成される
　コストと比べると計測から見れば、構造体で持った方が速いと言うことがわかりました。
これによりちょっと速くなりました。

=== 4 ===
 backTrack1()/backTrack2()のbm以外の変数はbitだけです。こちらは簡単に構造体に
　格納して実装することができました。問題はbm(bitmap)です。
　こちらは、再帰で変化する変数で、スレッド毎に値も変わることから値渡しである
　必要があります。よって関数の引数の中に格納することとしました。

void backTrack2(int y,int left,int down,int right,int bm,local *l){
void backTrack1(int y,int left,int down,int right,int bm,local *l){
これによりちょっと速くなりました。

=== 5 ===
pthreadや構造体 lは　#defineで宣言されるMAX=27を使って初期化していました。
si siEをグローバル変数としたことで、これらもNの値で初期化することとしました。

void *NQueenThread(){
  //pthread_t pt[G.si];//スレッド childThread
  pthread_t pt[si];//スレッド childThread
  //local l[MAX];//構造体 local型 
  local l[si];//構造体 local型 

=== 6 ===
  メインスレッドを生成するときに、
  pthread_join()の直後にpthread_detach()をいれて終了したスレッドを
　解放するようにした。

  pthread_join(pth, NULL); //スレッドの終了を待つ
  pthread_detach(pth);

=== 7 ===
  チルドスレッドを生成するときに、
  pthread_join()の直後にpthread_detach()をいれて終了したスレッドを
　解放するようにした。

  for(int B1=1;B1<siE;B1++){ 
    pthread_join(pt[B1],NULL); 
  }
  for(int B1=1;B1<siE;B1++){ 
    pthread_detach(pt[B1]);
  }

=== 8 ===
  以下の関数でforループする範囲を精査した
  void *NQueenThread(){

  B1は実際０から存在するがカウントされることはないためB1は１からスタート
  さらにB2がNでカウントされることはないので、N-1を限界値とした。

  for(int B1=1,B2=siE-1;B1<siE;B1++,B2--){

=== 9 ===
  symmetryOps_bm()のメソッド内を最適化。
  whileを使うことで変数を初期化する箇所をforに変更して、関数末尾での
　インクリメントやデクリメントをなくした。
　可能な処理に関してはインラインメソッドにした。

      for(l->bit=1,l->you=siE;(l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit);l->bit<<=1,l->you--);;

=== 10 ===
 CPU affinity 論理CPUにスレッドを割り当てる

#if _GNU_SOURCE
#define _GNU_SOURCE
#include <sched.h> 
#include <unistd.h>
#include <sys/syscall.h>
#include <errno.h>
#define handle_error_en(en, msg) do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)
#endif


# run ソース部分：
#ifdef _GNU_SOURCE
  pthread_t thread = pthread_self();
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(l->B1, &cpuset);
  int s=pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  if (s != 0){ handle_error_en(s, "pthread_setaffinity_np"); }
  s=pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  if (s != 0){ handle_error_en(s, "pthread_getaffinity_np"); }
  //printf("pid:%10d#l->B1:%2d#cpuset:%d\n",thread,l->B1,&cpuset);
#endif

  実行結果  スレッドが立ってる
pid:139811194939136#l->B1: 7#cpuset:1419517552
pid:139811100808960#l->B1: 4#cpuset:1325387376
pid:139811084023552#l->B1: 3#cpuset:1308601968
pid:139811109201664#l->B1: 5#cpuset:1333780080
pid:139811203331840#l->B1: 8#cpuset:1427910256
pid:139811117594368#l->B1: 6#cpuset:1342172784
pid:139811067238144#l->B1: 2#cpuset:1291816560
pid:139811050452736#l->B1: 1#cpuset:1275031152
16:        14772512          1846955          00:00:00:00.56
pid:139811228509952#l->B1:12#cpuset:1453088368
pid:139811084023552#l->B1:14#cpuset:1308601968
pid:139811211724544#l->B1:10#cpuset:1436302960
pid:139811067238144#l->B1:15#cpuset:1291816560
pid:139811220117248#l->B1:11#cpuset:1444695664
pid:139811050452736#l->B1:16#cpuset:1275031152
pid:139811236902656#l->B1:13#cpuset:1461481072
pid:139811203331840#l->B1: 9#cpuset:1427910256
pid:139811109201664#l->B1: 6#cpuset:1333780080
pid:139811058845440#l->B1: 2#cpuset:1283423856
pid:139811117594368#l->B1: 7#cpuset:1342172784
pid:139811042060032#l->B1: 1#cpuset:1266638448
pid:139811092416256#l->B1: 4#cpuset:1316994672
pid:139811100808960#l->B1: 5#cpuset:1325387376
pid:139811075630848#l->B1: 3#cpuset:1300209264
pid:139811194939136#l->B1: 8#cpuset:1419517552
17:        95815104         11977939          00:00:00:03.63

=== 11 ===
コンパイラ比較
$ /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang -v
Apple LLVM version 8.1.0 (clang-802.0.42)
Target: x86_64-apple-darwin16.6.0
Thread model: posix
InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin

15:         2279184           285053          00:00:00:00.16
16:        14772512          1846955          00:00:00:01.02
17:        95815104         11977939          00:00:00:06.55

$ /usr/local/Cellar/llvm/4.0.1/bin/clang -v
clang version 4.0.1 (tags/RELEASE_401/final)
Target: x86_64-apple-darwin16.6.0
Thread model: posix
InstalledDir: /usr/local/bin
Found CUDA installation: /usr/local/cuda, version 7.5

15:         2279184           285053          00:00:00:00.15
16:        14772512          1846955          00:00:00:00.95
17:        95815104         11977939          00:00:00:06.27

$ /opt/local/bin/gcc-mp-7 -v
Using built-in specs.
COLLECT_GCC=/opt/local/bin/gcc-mp-7
COLLECT_LTO_WRAPPER=/opt/local/libexec/gcc/x86_64-apple-darwin16/7.1.1/lto-wrapper
Target: x86_64-apple-darwin16
Configured with: /opt/local/var/macports/build/_opt_bblocal_var_buildworker_ports_build_ports_lang_gcc7/gcc7/work/gcc-7-20170622/configure --prefix=/opt/local --build=x86_64-apple-darwin16 --enable-languages=c,c++,objc,obj-c++,lto,fortran --libdir=/opt/local/lib/gcc7 --includedir=/opt/local/include/gcc7 --infodir=/opt/local/share/info --mandir=/opt/local/share/man --datarootdir=/opt/local/share/gcc-7 --with-local-prefix=/opt/local --with-system-zlib --disable-nls --program-suffix=-mp-7 --with-gxx-include-dir=/opt/local/include/gcc7/c++/ --with-gmp=/opt/local --with-mpfr=/opt/local --with-mpc=/opt/local --with-isl=/opt/local --enable-stage1-checking --disable-multilib --enable-lto --enable-libstdcxx-time --with-build-config=bootstrap-debug --with-as=/opt/local/bin/as --with-ld=/opt/local/bin/ld --with-ar=/opt/local/bin/ar --with-bugurl=https://trac.macports.org/newticket --with-pkgversion='MacPorts gcc7 7-20170622_0'
Thread model: posix
gcc version 7.1.1 20170622 (MacPorts gcc7 7-20170622_0)

15:         2279184           285053          00:00:00:00.12
16:        14772512          1846955          00:00:00:00.71
17:        95815104         11977939          00:00:00:04.64

$ /opt/local/bin/gcc-mp-6 -v
Using built-in specs.
COLLECT_GCC=/opt/local/bin/gcc-mp-6
COLLECT_LTO_WRAPPER=/opt/local/libexec/gcc/x86_64-apple-darwin16/6.3.0/lto-wrapper
Target: x86_64-apple-darwin16
Configured with: /opt/local/var/macports/build/_opt_bblocal_var_buildworker_ports_build_ports_lang_gcc6/gcc6/work/gcc-6.3.0/configure --prefix=/opt/local --build=x86_64-apple-darwin16 --enable-languages=c,c++,objc,obj-c++,lto,fortran --libdir=/opt/local/lib/gcc6 --includedir=/opt/local/include/gcc6 --infodir=/opt/local/share/info --mandir=/opt/local/share/man --datarootdir=/opt/local/share/gcc-6 --with-local-prefix=/opt/local --with-system-zlib --disable-nls --program-suffix=-mp-6 --with-gxx-include-dir=/opt/local/include/gcc6/c++/ --with-gmp=/opt/local --with-mpfr=/opt/local --with-mpc=/opt/local --with-isl=/opt/local --enable-stage1-checking --disable-multilib --enable-lto --enable-libstdcxx-time --with-build-config=bootstrap-debug --with-as=/opt/local/bin/as --with-ld=/opt/local/bin/ld --with-ar=/opt/local/bin/ar --with-bugurl=https://trac.macports.org/newticket --with-pkgversion='MacPorts gcc6 6.3.0_2'
Thread model: posix
gcc version 6.3.0 (MacPorts gcc6 6.3.0_2)

15:         2279184           285053          00:00:00:00.11
16:        14772512          1846955          00:00:00:00.69
17:        95815104         11977939          00:00:00:04.61


$ /usr/local/Cellar/gcc\@5/5.4.0_1/bin/gcc-5 -v
Using built-in specs.
COLLECT_GCC=/usr/local/Cellar/gcc@5/5.4.0_1/bin/gcc-5
COLLECT_LTO_WRAPPER=/usr/local/Cellar/gcc@5/5.4.0_1/libexec/gcc/x86_64-apple-darwin16.3.0/5.4.0/lto-wrapper
Target: x86_64-apple-darwin16.3.0
Configured with: ../configure --build=x86_64-apple-darwin16.3.0 --prefix=/usr/local/Cellar/gcc@5/5.4.0_1 --libdir=/usr/local/Cellar/gcc@5/5.4.0_1/lib/gcc/5 --enable-languages=c,c++,objc,obj-c++,fortran --program-suffix=-5 --with-gmp=/usr/local/opt/gmp --with-mpfr=/usr/local/opt/mpfr --with-mpc=/usr/local/opt/libmpc --with-isl=/usr/local/opt/isl@0.14 --with-system-zlib --enable-libstdcxx-time=yes --enable-stage1-checking --enable-checking=release --enable-lto --disable-werror --with-pkgversion='Homebrew GCC 5.4.0_1' --with-bugurl=https://github.com/Homebrew/homebrew-core/issues --enable-plugin --disable-nls --enable-multilib
Thread model: posix
gcc version 5.4.0 (Homebrew GCC 5.4.0_1)

15:         2279184           285053          00:00:00:00.11
16:        14772512          1846955          00:00:00:00.69
17:        95815104         11977939          00:00:00:04.62

$ /usr/local/Cellar/gcc\@4.9/4.9.4/bin/gcc-4.9 -v
Using built-in specs.
COLLECT_GCC=/usr/local/Cellar/gcc@4.9/4.9.4/bin/gcc-4.9
COLLECT_LTO_WRAPPER=/usr/local/Cellar/gcc@4.9/4.9.4/libexec/gcc/x86_64-apple-darwin16.4.0/4.9.4/lto-wrapper
Target: x86_64-apple-darwin16.4.0
Configured with: ../configure --build=x86_64-apple-darwin16.4.0 --prefix=/usr/local/Cellar/gcc@4.9/4.9.4 --libdir=/usr/local/Cellar/gcc@4.9/4.9.4/lib/gcc/4.9 --enable-languages=c,c++,objc,obj-c++,fortran --program-suffix=-4.9 --with-gmp=/usr/local/opt/gmp@4 --with-mpfr=/usr/local/opt/mpfr@2 --with-mpc=/usr/local/opt/libmpc@0.8 --with-cloog=/usr/local/opt/cloog --with-isl=/usr/local/opt/isl@0.12 --with-system-zlib --enable-libstdcxx-time=yes --enable-stage1-checking --enable-checking=release --enable-lto --with-build-config=bootstrap-debug --disable-werror --with-pkgversion='Homebrew GCC 4.9.4' --with-bugurl=https://github.com/Homebrew/homebrew-core/issues MAKEINFO=missing --enable-plugin --disable-nls --enable-multilib
Thread model: posix
gcc version 4.9.4 (Homebrew GCC 4.9.4)

15:         2279184           285053          00:00:00:00.11
16:        14772512          1846955          00:00:00:00.71
17:        95815104         11977939          00:00:00:04.63

  実行結果 
 N:        Total       Unique                 dd:hh:mm:ss.ms
 2:               0                0          00:00:00:00.00
 3:               0                0          00:00:00:00.00
 4:               2                1          00:00:00:00.00
 5:              10                2          00:00:00:00.00
 6:               4                1          00:00:00:00.00
 7:              40                6          00:00:00:00.00
 8:              92               12          00:00:00:00.00
 9:             352               46          00:00:00:00.00
10:             724               92          00:00:00:00.00
11:            2680              341          00:00:00:00.00
12:           14200             1787          00:00:00:00.00
13:           73712             9233          00:00:00:00.00
14:          365596            45752          00:00:00:00.01
15:         2279184           285053          00:00:00:00.10
16:        14772512          1846955          00:00:00:00.67
17:        95815104         11977939          00:00:00:04.47
18:       666090624         83263591          00:00:00:35.44

  参考（Bash版 07_8NQueen.lua）
  13:           73712             9233                99
  14:          365596            45752               573
  15:         2279184           285053              3511

  参考（Lua版 07_8NQueen.lua）
  14:          365596            45752          00:00:00
  15:         2279184           285053          00:00:03
  16:        14772512          1846955          00:00:20

  参考（Java版 NQueen8.java マルチスレット）
  16:        14772512          1846955          00:00:00
  17:        95815104         11977939          00:00:04
  18:       666090624         83263591          00:00:34
  19:      4968057848        621012754          00:04:18
  20:     39029188884       4878666808          00:35:07
  21:    314666222712      39333324973          04:41:36
  22:   2691008701644     336376244042          39:14:59
 *
*/

#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

#define MAX 27
#define DEBUG 0

#ifdef _GNU_SOURCE
/** cpu affinityを有効にするときは以下の１行（#define _GNU_SOURCE)を、
 * #ifdef _GNU_SOURCE の上に移動 
 * CPU Affinity はLinuxのみ動作します。　Macでは動きません*/
#define _GNU_SOURCE   
#include <sched.h> 
#include <unistd.h>
#include <sys/syscall.h>
#include <errno.h>
#define handle_error_en(en, msg) do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)
#endif

//ローカル構造体
typedef struct{
	int bit;
  int own;
	int ptn;
	int you;
  int B1;
  int B2;
  int TB;
  int EB;
  int msk;
  int SM;
  int LM;
  int aB[MAX];
  long C2[MAX];//構造体の中の配列を活かすと   N=17: 04.33
  long C4[MAX];
  long C8[MAX];
}local ;

int si;  //si siE lTotal lUnique をグローバルに置くと N=17: 04.26
int siE;
long lTotal;
long lUnique;

void backTrack2(int y,int left,int down,int right,int bm,local *l);
void backTrack1(int y,int left,int down,int right,int bm,local *l);
void *run(void *args);
void *NQueenThread();
void NQueen();
void symmetryOps_bm(local *l);

#ifdef DEBUG
const int spc[]={'/', '-', '\\', '|'};
const int spl=sizeof(spc)/sizeof(spc[0]);
void thMonitor(local *l,int i);
void hoge();
void hoge(){
  clock_t t;
  t = clock() + CLOCKS_PER_SEC/10;
  while(t>clock());
}
#endif

void thMonitor(local *l,int i){
  printf("\033[G");
  if(i==2){ printf("\rN:%2d C2[%c] C4[ ] C8[ ] C8BT[ ] B1[%2d] B2[%2d]",si,spc[l->C2[l->B1]%spl],l->B1,l->B2); }
  else if(i==4){ printf("\rN:%2d C2[ ] C4[%c] C8[ ] C8BT[ ] B1[%2d] B2[%2d]",si,spc[l->C4[l->B1]%spl],l->B1,l->B2); }
  else if(i==8){ printf("\rN:%2d C2[ ] C4[ ] C8[%c] C8BT[ ] B1[%2d] B2[%2d]",si,spc[l->C8[l->B1]%spl],l->B1,l->B2); }
  else if(i==82){ printf("\rN:%2d C2[ ] C4[ ] C8[ ] C8BT[%c] B1[%2d] B2[%2d]",si,spc[l->C8[l->B1]%spl],l->B1,l->B2); }
  printf("\033[G");
}

/**********************************************/
/* 最上段行のクイーンが角以外にある場合の探索 */
/**********************************************/
/**
  １行目角にクイーンが無い場合、クイーン位置より右位置の８対称位置にクイーンを
  置くことはできない
  １行目位置が確定した時点で、配置可能位置を計算しておく（☓の位置）
  lt, dn, lt 位置は効きチェックで配置不可能となる
  回転対称チェックが必要となるのは、クイーンがａ, ｂ, ｃにある場合だけなので、 
  90度、180度、270度回転した状態のユニーク判定値との比較を行うだけで済む

  【枝刈り図】
  x x - - - Q x x    
  x - - - / | ＼x    
  c - - / - | -rt    
  - - / - - | - -    
  - / - - - | - -    
  lt- - - - | - a    
  x - - - - | - x    
  x x b - - dnx x    
  */
void backTrack2(int y,int left,int down,int right,int bm,local *l){
  l->bit=0;
  bm=l->msk&~(left|down|right); //配置可能フィールド
  if(y==siE){
    if(bm){
      if((bm&l->LM)==0){  //【枝刈り】最下段枝刈り
        l->aB[y]=bm;
        symmetryOps_bm(l);        //対称解除法
      }
    }
  }else{
    //【枝刈り】上部サイド枝刈り
    if(y<l->B1){             
      bm&=~l->SM; 
    }else if(y==l->B2){        //【枝刈り】下部サイド枝刈り
      if((down&l->SM)==0){ return; }
      if((down&l->SM)!=l->SM){ bm&=l->SM; }
    }
    while(bm){
      bm^=l->aB[y]=l->bit=-bm&bm;//最も下位の１ビットを抽出
      backTrack2(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }
  }
}
/**********************************************/
/* 最上段行のクイーンが角にある場合の探索     */
/**********************************************/
/* 
   １行目角にクイーンがある場合、回転対称形チェックを省略することが出来る
   １行目角にクイーンがある場合、他の角にクイーンを配置することは不可
   鏡像についても、主対角線鏡像のみを判定すればよい
   ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい 
   */
void backTrack1(int y,int left,int down,int right,int bm,local *l){
  l->bit=0;
  bm=l->msk&~(left|down|right);  //配置可能フィールド
  if(y==siE) {
    if(bm){//【枝刈り】１行目角にクイーンがある場合回転対称チェックを省略
      l->aB[y]=bm;
      l->C8[l->B1]++;
      if(DEBUG>0) thMonitor(l,82);
    }
  }else{
    if(y<l->B1){
      //【枝刈り】鏡像についても主対角線鏡像のみを判定すればよい
      // ２行目、２列目を数値とみなし、２行目＜２列目という条件を課せばよい
      bm&=~2; // bm|=2; bm^=2; (bm&=~2と同等)
    }
    while(bm){
      bm^=l->aB[y]=l->bit=-bm&bm;//最も下位の１ビットを抽出
      backTrack1(y+1,(left|l->bit)<<1,down|l->bit,(right|l->bit)>>1,bm,l);
    }
  } 
}
void *run(void *args){
  local *l=(local *)args;
#ifdef _GNU_SOURCE
  pthread_t thread = pthread_self();
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(l->B1, &cpuset);
  int s=pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  if (s != 0){ handle_error_en(s, "pthread_setaffinity_np"); }
  s=pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  if (s != 0){ handle_error_en(s, "pthread_getaffinity_np"); }
  printf("pid:%10ld#l->B1:%2d#cpuset:%p\n",thread,l->B1,&cpuset);
#endif
  //int bit ;
  //l->bit=0 ; l->aB[0]=1; l->msk=(1<<G.si)-1; l->TB=1<<G.siE;
  l->bit=0 ; 
  l->aB[0]=1; 
  l->msk=(1<<si)-1; 
  l->TB=1<<siE;
  // 最上段のクイーンが角にある場合の探索
  if(l->B1>1 && l->B1<siE) { 
    //if(l->B1<G.siE) {
      // 角にクイーンを配置 
      l->aB[1]=l->bit=(1<<l->B1);
      //２行目から探索
      backTrack1(2,(2|l->bit)<<1,(1|l->bit),(l->bit>>1),0,l);
    //}
  }
  l->EB=(l->TB>>l->B1);
  l->SM=l->LM=(l->TB|1);
  /* 最上段行のクイーンが角以外にある場合の探索 
     ユニーク解に対する左右対称解を予め削除するには、
     左半分だけにクイーンを配置するようにすればよい */
  if(l->B1>0&&l->B2<siE&&l->B1<l->B2){ 
    for(int i=1; i<l->B1; i++){
      l->LM=l->LM|l->LM>>1|l->LM<<1;
    }
    //if(l->B1<l->B2) {
      l->aB[0]=l->bit=(1<<l->B1);
      backTrack2(1,l->bit<<1,l->bit,l->bit>>1,0,l);
    // }
    l->EB>>=si;
  }
  return 0;
}
/**********************************************/
/* マルチスレッド */
/**********************************************/
/**
 *
 * N=8の場合は8つのスレッドがおのおののrowを担当し処理を行います。

 メインスレッド  N=8
 +--B1=7----- run()
 +--B1=6----- run()
 +--B1=5----- run()
 +--B1=4----- run()
 +--B1=3----- run()
 +--B1=2----- run()
 +--B1=1----- run()
 +--B1=0----- run()

 * そこで、それぞれのスレッド毎にスレッドローカルな構造体を持ちます。
 *
// スレッドローカル構造体 
struct local{
int bit;
int B1;
int B2;
int TB;
int EB;
int msk;
int SM;
int LM;
int aB[MAX];
};
 * 
 * スレッドローカルな構造体の宣言は以下の通りです。
 *
 *    //スレッドローカルな構造体
 *    struct local l[MAX];
 *
 * アクセスはグローバル構造体同様 . ドットでアクセスします。
 l[B1].B1=B1;
 l[B1].B2=B2;
 *
 */
void *NQueenThread(){
  pthread_t pt[si];//スレッド childThread
  // スレッドローカルな構造体
  local l[si];//構造体 local型 
  // B1から順にスレッドを生成しながら処理を分担する 
  for(int B1=1,B2=siE-1;B1<siE;B1++,B2--){
    //B1 と B2を初期化
    l[B1].B1=B1; l[B1].B2=B2;
    //aB[]の初期化
    for(int j=0;j<siE;j++){ l[l->B1].aB[j]=j; } 
    //カウンターの初期化
    l[B1].C2[B1]=l[B1].C4[B1]=l[B1].C8[B1]=0;	
    // チルドスレッドの生成
    int iFbRet=pthread_create(&pt[B1],NULL,&run,(void*)&l[B1]);
    if(iFbRet>0){ printf("[Thread] pthread_create #%d: %d\n", l[B1].B1, iFbRet); }
  }
  for(int B1=1;B1<siE;B1++){ pthread_join(pt[B1],NULL); }
  for(int B1=1;B1<siE;B1++){ pthread_detach(pt[B1]); }
  //スレッド毎のカウンターを合計
  for(int B1=1;B1<siE;B1++){
    lTotal+=l[B1].C2[B1]*2+l[B1].C4[B1]*4+l[B1].C8[B1]*8;
    lUnique+=l[B1].C2[B1]+l[B1].C4[B1]+l[B1].C8[B1]; 
  }
  return 0;
}
/**********************************************/
/*  マルチスレッド pthread                    */
/**********************************************/
/**
 *  マルチスレッドには pthreadを使います。
 *  pthread を宣言するには pthread_t 型の変数を宣言します。
 *
 pthread_t tId;  //スレッド変数

 スレッドを生成するには pthread_create()を呼び出します。
 戻り値iFbRetにはスレッドの状態が格納されます。正常作成は0になります。
 pthread_join()はスレッドの終了を待ちます。
 */
void NQueen(){
  pthread_t pth;  //スレッド変数
  // メインスレッドの生成
  int iFbRet=pthread_create(&pth, NULL, &NQueenThread, NULL);
  //エラー出力デバッグ用
  if(iFbRet>0){ printf("[main] pthread_create: %d\n", iFbRet); }
  //スレッドの終了を待つ
  pthread_join(pth, NULL);
  pthread_detach(pth);
}
/**********************************************/
/*  メイン関数                                */
/**********************************************/
/**
 * N=2 から順を追って 実行関数 NQueen()を呼び出します。
 * 最大値は 先頭行でMAXをdefineしています。
 * G は グローバル構造体で宣言しています。

//グローバル構造体
typedef struct {
int nThread;
int si;
int siE;
long C2;
long C4;
long C8;
}GCLASS, *GClass;
GCLASS G; //グローバル構造体

グローバル構造体を使う場合は
G.si=i ; 
のようにドットを使ってアクセスします。

NQueen()実行関数は forの中の値iがインクリメントする度に
Nのサイズが大きくなりクイーンの数を解法します。 
*/
int main(void){
  struct timeval t0;
  struct timeval t1;
  int min=2;
  printf("%s\n"," N:        Total       Unique                 dd:hh:mm:ss.ms");
  for(int i=min;i<=MAX;i++){
    si=i; siE=i-1; 
    lTotal=lUnique=0;
    gettimeofday(&t0, NULL);
    NQueen();     // 実行関数
    gettimeofday(&t1, NULL);
    int ss;int ms;int dd;
    if (t1.tv_usec<t0.tv_usec) {
      dd=(t1.tv_sec-t0.tv_sec-1)/86400; 
      ss=(t1.tv_sec-t0.tv_sec-1)%86400; 
      ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000; 
    } else { 
      dd=(t1.tv_sec-t0.tv_sec)/86400; 
      ss=(t1.tv_sec-t0.tv_sec)%86400; 
      ms=(t1.tv_usec-t0.tv_usec+500)/10000; 
    }
    int hh=ss/3600; 
    int mm=(ss-hh*3600)/60; 
    ss%=60;
    printf("%2d:%16ld%17ld%12.2d:%02d:%02d:%02d.%02d\n", i,lTotal,lUnique,dd,hh,mm,ss,ms); 
  } 
}
/**********************************************/
/** 対称解除法                               **/
/** ユニーク解から全解への展開               **/
/**********************************************/
/**
  ひとつの解には、盤面を90度・180度・270度回転、及びそれらの鏡像の合計8個の対称解が存在する

  １２ ４１ ３４ ２３
  ４３ ３２ ２１ １４

  ２１ １４ ４３ ３２
  ３４ ２３ １２ ４１

  上図左上がユニーク解。
  1行目はユニーク解を90度、180度、270度回転したもの
  2行目は1行目のそれぞれを左右反転したもの。
  2行目はユニーク解を左右反転、対角反転、上下反転、逆対角反転したものとも解釈可 
  ただし、 回転・線対称な解もある
  クイーンが右上角にあるユニーク解を考えます。
  斜軸で反転したパターンがオリジナルと同型になることは有り得ないことと(×２)、
  右上角のクイーンを他の３つの角に写像させることができるので(×４)、
  このユニーク解が属するグループの要素数は必ず８個(＝２×４)になります。

  (1) 90度回転させてオリジナルと同型になる場合、さらに90度回転(オリジナルから180度回転)
  　　させても、さらに90度回転(オリジナルから270度回転)させてもオリジナルと同型になる。 
  (2) 90度回転させてオリジナルと異なる場合は、270度回転させても必ずオリジナルとは異なる。
  　　ただし、180度回転させた場合はオリジナルと同型になることも有り得る。

  　(1)に該当するユニーク解が属するグループの要素数は、左右反転させたパターンを加えて
  ２個しかありません。(2)に該当するユニーク解が属するグループの要素数は、180度回転させ
  て同型になる場合は４個(左右反転×縦横回転)、そして180度回転させてもオリジナルと異なる
  場合は８個になります。(左右反転×縦横回転×上下反転)
  */
void symmetryOps_bm(local *l){
  l->own=l->ptn=l->you=l->bit=0;
  l->C8[l->B1]++;
  if(DEBUG>0) thMonitor(l,8); 
  //90度回転
  if(l->aB[l->B2]==1){ 
    for(l->own=1,l->ptn=2;l->own<=siE;l->own++,l->ptn<<=1){ 
      for(l->bit=1,l->you=siE;(l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit);l->bit<<=1,l->you--){}
      if(l->aB[l->own]>l->bit){ l->C8[l->B1]--; return; 
      }else if(l->aB[l->own]<l->bit){ break; } }
    /** 90度回転して同型なら180度/270度回転も同型である */
    if(l->own>siE){ l->C2[l->B1]++; l->C8[l->B1]--; if(DEBUG>0) thMonitor(l,2); return ; } 
  }
  //180度回転
  if(l->aB[siE]==l->EB){ 
    for(l->own=1,l->you=siE-1;l->own<=siE;l->own++,l->you--){ 
      for(l->bit=1,l->ptn=l->TB;(l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit);l->bit<<=1,l->ptn>>=1){}
      if(l->aB[l->own]>l->bit){ l->C8[l->B1]--; return; } 
      else if(l->aB[l->own]<l->bit){ break; } }
    /** 90度回転が同型でなくても180度回転が同型である事もある */
    if(l->own>siE){ l->C4[l->B1]++; l->C8[l->B1]--; if(DEBUG>0) thMonitor(l,4); return; } }
  //270度回転
  if(l->aB[l->B1]==l->TB){ 
    for(l->own=1,l->ptn=l->TB>>1;l->own<=siE;l->own++,l->ptn>>=1){ 
      for(l->bit=1,l->you=0;(l->aB[l->you]!=l->ptn)&&(l->aB[l->own]>=l->bit);l->bit<<=1,l->you++){}
      if(l->aB[l->own]>l->bit){ l->C8[l->B1]--; return; } 
      else if(l->aB[l->own]<l->bit){ break; } } }
}
