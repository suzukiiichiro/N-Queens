#undef TRACE
//#define TRACE
#include <cstdint>
#include <cassert>
#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string.h>
#define MAX 27
//ローカル構造体
typedef struct
{
  uint64_t bv;
  uint64_t bh;
  uint64_t bu;
  uint64_t bd;
  int x[MAX];
}Board ;
Board B;
//ローカル構造体
typedef struct
{
  char unsigned  a;
  char unsigned  b;
}pres_t ;
//
//再帰でクイーンを置いていく
uint64_t countCompletions(uint64_t  bv,
				     uint64_t  bh,
				     uint64_t  bu,
				     uint64_t  bd) 
{
  // Placement Complete?
  if(bh+1 == 0)  return  1;
  // -> at least one more queen to place
  while((bv&1) != 0) { // Column is covered by pre-placement
    bv >>= 1;
    bu <<= 1;
    bd >>= 1;
  }
  bv >>= 1;

  // Column needs to be placed
  uint64_t  cnt = 0;
  for(uint64_t  slots = ~(bh|bu|bd); slots != 0;) {
    uint64_t const  slot = slots & -slots;
    cnt += countCompletions(bv, bh|slot, (bu|slot) << 1, (bd|slot) >> 1);
    slots ^= slot;
  }
  return  cnt;
} // countCompletions()
//
bool board_placement(int si,int x,int y)
{
  //同じ場所に置くかチェック
  //printf("i:%d:x:%d:y:%d\n",i,B.x[i],B.y[i]);
  if(B.x[x]==y){
    //printf("Duplicate x:%d:y:%d\n",x,y);
    return true;  
  }
  B.x[x]=y;
  uint64_t bv=1<<x;
  uint64_t bh=1<<y;
  uint64_t bu=1<<(si-1-x+y);
  uint64_t bd=1<<(x+y);
  //printf("check valid x:%d:y:%d:p.N-1-x+y:%d;x+y:%d\n",x,y,si-1-x+y,x+y);
  //printf("check valid pbv:%d:bv:%d:pbh:%d:bh:%d:pbu:%d:bu:%d:pbd:%d:bd:%d\n",B.bv,bv,B.bh,bh,B.bu,bu,B.bd,bd);
  //printf("bvcheck:%d:bhcheck:%d:bucheck:%d:bdcheck:%d\n",B.bv&bv,B.bh&bh,B.bu&bu,B.bd&bd);
  if((B.bv&bv)||(B.bh&bh)||(B.bu&bu)||(B.bd&bd)){
    //printf("valid_false\n");
    return false;
  }     
  //printf("before pbv:%d:bv:%d:pbh:%d:bh:%d:pbu:%d:bu:%d:pbd:%d:bd:%d\n",B.bv,bv,B.bh,bh,B.bu,bu,B.bd,bd);
  B.bv |=bv;
  B.bh |=bh;
  B.bu |=bu;
  B.bd |=bd;
  //printf("after pbv:%d:bv:%d:pbh:%d:bh:%d:pbu:%d:bu:%d:pbd:%d:bd:%d\n",B.bv,bv,B.bh,bh,B.bu,bu,B.bd,bd);
  //printf("valid_true\n");
  return true;
}
//クイーンを置いていく
void process(int si,Board B,int sym,uint64_t *pre,uint64_t *cnt)
{
  //printf("process\n");
  pre[sym]++;
  //printf("N:%d\n",si);
  //BVは行 x 
  //printf("getBV:%d\n",B.bv);
  //BHはdown y
  //printf("getBH:%d\n",B.bh);
  //BU left N-1-x+y 右上から左下
  //printf("getBU:%d\n",B.bu);
  //BD right x+y 左上から右下
  //printf("getBD:%d\n",B.bd);
  //printf("before_cnt_sym:%d\n",cnt[sym]);
  cnt[sym] += countCompletions(B.bv >> 2,
      ((((B.bh>>2)|(~0<<(si-4)))+1)<<(si-5))-1,
      B.bu>>4,
      (B.bd>>4)<<(si-5));

  //行 brd.getBV()>>2 右2ビット削除 すでに上２行はクイーンを置いているので進める BVは右端を１ビットずつ削っていく
  //列 down ((((brd.getBH()>>2)|(~0<<(N-4)))+1)<<(brd.N-5))-1 8だと左に1シフト 9:2 10:3 
  //brd.getBU()>>4 left  右４ビット削除
  //(brd.getBD()>>4)<<(N-5)) right 右４ビット削除後N-5個分左にシフト
  //printf("cnt_sym:%d\n",cnt[sym]);
}
int main(int const  argc, char const* const argv[]) 
{
  unsigned const  N = argc < 2? 0 : (unsigned)strtoul(argv[argc-1], 0, 0);
  uint64_t  cnt[3];
  uint64_t  pre[3];
  for(int s=0;s<3;s++){  
    cnt[s] = 0;
    pre[s] = 0;
  }
  // Check Arguments
  if((N < 5) || (32 < N)) {
    return  1;
  }
  std::cout << N << "-Queens Puzzle\n" << std::endl;
  /**
   * The number of valid pre-placements in two adjacent columns (rows) is
   * 2*(N-2) + (N-2)*(N-3) for the outmost and inner positions in the the
   * first column, respectively. Thus, the total is (N-2)*(N-1).
   */
  pres_t pres[(N-2)*(N-1)];
  // Compute all valid two-column pre-placements in order:
  //上下左右２行、２列にクイーンを配置する
  // (a0, b0) < (a1, b1) if a0<a1 || (a0==a1 && b0<b1)
  unsigned  idx = 0;
  for(unsigned  a = 0; a < N; a++) {
    for(unsigned  b = 0; b < N; b++) {
      if((a>=b&&(a-b)<=1)||(b>a&&(b-a)<=1)){
        continue;
      }     
      pres[idx].a = a;
      pres[idx].b = b;
      idx++;
    }
  }
  std::cout << "First side bound: ("
    << (unsigned)pres[(N/2)*(N-3)  ].a << ", " << (unsigned)pres[(N/2)*(N-3)  ].b << ") / ("
    << (unsigned)pres[(N/2)*(N-3)+1].a << ", " << (unsigned)pres[(N/2)*(N-3)+1].b << ')'
    << std::endl;
  // Generate coronal Placements
  //Board  board(N);
  //上2行は2分の1だけ実行(ミラー)
  Board wB=B;
  for(unsigned  w = 0; w <= (N/2)*(N-3); w++) {
    B=wB;
    B.bv=0;
    B.bh=0;
    B.bu=0;
    B.bd=0;
    for(int i=0;i<N;i++){
      B.x[i]=-1;
    }
    unsigned const  wa = pres[w].a;
    unsigned const  wb = pres[w].b;
#ifdef TRACE
    std::cerr << '(' << wa << ", " << wb << ')' << std::endl;
#else
    std::cout << "\rProgress: " << w << '/' << ((N/2)*(N-3)) << std::flush;
#endif
    //上２行　0行目,1行目にクイーンを置く
    //Board::Placement  pwa(board.place(0, wa));
    //Board::Placement  pwb(board.place(1, wb));
    board_placement(N,0,wa);
    board_placement(N,1,wb);
    Board nB=B;
    for(unsigned  n = w; n < (N-2)*(N-1)-w; n++) {
      B=nB;
      unsigned const  na = pres[n].a;
      unsigned const  nb = pres[n].b;
#ifdef TRACE
      std::cerr << '(' << wa << ", " << wb << ')'
        << '(' << na << ", " << nb << ')' << std::endl;
#endif
      //pre-place 左２列にクイーンを置く
      bool pna=board_placement(N,na,N-1);
      if(pna==false){
        //printf("pnaskip:na:%d:N-1:%d\n",na,size-1);
        continue;
      }
      bool pnb=board_placement(N,nb,N-2);
      if(pnb==false){
        //printf("pnbskip:nb:%d:N-2:%d\n",nb,size-2);
        continue;
      }
      Board eB=B;
      for(unsigned  e = w; e < (N-2)*(N-1)-w; e++) {
        B=eB;
        unsigned const  ea = pres[e].a;
        unsigned const  eb = pres[e].b;
#ifdef TRACE
        std::cerr << '(' << wa << ", " << wb << ')'
          << '(' << na << ", " << nb << ')'
          << '(' << ea << ", " << eb << ')' << std::endl;
#endif
        //下２行に置く
        bool pea=board_placement(N,N-1,N-1-ea);
        if(pea==false){
          //printf("peaskip:N-1:%d:N-1-ea:%d\n",size-1,size-1-ea);
          continue;
        }
        //printf("placement_peb:xk(N-2):%d:y:%d\n",size-2,size-1-eb);
        bool peb=board_placement(N,N-2,N-1-eb);
        if(peb==false){
          //printf("pebskip:N-2:%d:N-1-eb:%d\n",size-2,size-1-eb);
          continue;
        }
        Board sB=B;
        for(unsigned  s = w; s < (N-2)*(N-1)-w; s++) {
          B=sB;
          unsigned const  sa = pres[s].a;
          unsigned const  sb = pres[s].b;
#ifdef TRACE
          std::cerr << '(' << wa << ", " << wb << ')'
            << '(' << na << ", " << nb << ')'
            << '(' << ea << ", " << eb << ')'
            << '(' << sa << ", " << sb << ')' << std::endl;
#endif
          //右２列に置く
          // We have a successful complete pre-placement with
          bool psa=board_placement(N,N-1-sa,0);
          if(psa==false){
            //printf("psaskip:N-1-sa:%d:0\n",size-1-sa);
            continue;
          }
          //printf("psb:x:%d:yk(1):1\n",size-1-sb);
          bool psb=board_placement(N,N-1-sb,1);
          if(psb==false){
            //printf("psbskip:N-1-sb:%d:1\n",size-1-sb);
            continue;
          }
          //   w <= n, e, s < (N-2)*(N-1)-w
          //
          // Thus, the placement is definitely a canonical minimum unless
          // one or more of n, e, s are equal to w or (N-2)*(N-1)-1-w.
          { // Check for minimum if n, e, s = (N-2)*(N-1)-1-w
            unsigned const  ww = (N-2)*(N-1)-1-w;
            if(s == ww) {
              // check if flip about the up diagonal is smaller
              if(n < (N-2)*(N-1)-1-e) {
                //print('S', wa, wb, na, nb, ea, eb, sa, sb);
                continue;
              }
            }
            if(e == ww) {
              // check if flip about the vertical center is smaller
              if(n > (N-2)*(N-1)-1-n) {
                //print('E', wa, wb, na, nb, ea, eb, sa, sb);
                continue;
              }
            }
            if(n == ww) {
              // check if flip about the down diagonal is smaller
              if(e > (N-2)*(N-1)-1-s) {
                //print('N', wa, wb, na, nb, ea, eb, sa, sb);
                continue;
              }
            }   
          }
          // Check for minimum if n, e, s = w
          if(s == w) {
            //右回転で同じ場合w=n=e=sでなければ値が小さいのでskip
            // right rotation is smaller unless  w = n = e = s
            if((n != w) || (e != w)) {
              //print('s', wa, wb, na, nb, ea, eb, sa, sb);
              continue;
            }
            //w=n=e=sであれば90度回転で同じ可能性*2
            process(N,B,0,pre,cnt);
            continue;
          }
          if(e == w) {
            //e==wは180度回転して同じ*4
            // check if 180°-rotation is smaller
            //180度回転して同じ時n>=sの時はsmaller
            if(n >= s) {
              if(n > s) {
                //print('e', wa, wb, na, nb, ea, eb, sa, sb);
                continue;
              }
              process(N,B,1,pre,cnt);
              continue;
            } 
          }
          // n = w is okay
          //*8
          //print('o', wa, wb, na, nb, ea, eb, sa, sb);
          process(N,B,2,pre,cnt);
        } // s
      } // e
    } // n
  } // w
  //集計
  uint64_t  uniq;
  uint64_t  total;
  uniq=cnt[0]+cnt[1]+cnt[2];
  total=cnt[0]*2+cnt[1]*4+cnt[2]*8;
  printf("\nTOTAL:%llu UNIQ:%llu\n",total,uniq);
}
