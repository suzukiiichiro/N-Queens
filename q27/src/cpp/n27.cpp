#undef TRACE
//#define TRACE

#include <cstdint>
#include <cassert>
#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <string.h>

  class Board {
  public:
    unsigned const  N;

  private:
    signed *const  board;

    uint64_t  bv;
    uint64_t  bh;
    uint64_t  bu;
    uint64_t  bd;

  public:
    Board(unsigned const  dim)
      : N(dim), board(new signed[dim]),
	bv(0), bh(0), bu(0), bd(0) {
      for(unsigned  i = 0; i < dim; board[i++] = -1);
    }
    ~Board() {
      delete [] board;
    }

  private:
    class Cell {
      signed         &col;
      unsigned const  y;

    public:
      Cell(signed &_col, unsigned const _y) : col(_col), y(_y) {}
      ~Cell() {}

    public:
      operator bool() const { return  col == (signed)y; }
      Cell& operator=(bool const  v) {
	if(v) {
	  assert(col == -1);
	  col = (signed)y;
	}
	else {
	  assert(col == (signed)y);
	  col = -1;
	}
	return *this;
      }
    }; // class Cell

  public:
    bool operator()(unsigned  x, unsigned  y) const {
      return  board[x]==(signed)y;
    }
  private:
    Cell operator()(unsigned  x, unsigned  y) {
      return  Cell(board[x], y);
    }

  public:
    class Placement {
      Board &parent;

    public:
      unsigned const  x;
      unsigned const  y;
private: bool  valid; bool  owner; 
    private:
      friend class Board;
      Placement(Board &_parent, unsigned const _x, unsigned const _y)
	: parent(_parent), x(_x), y(_y) {
	if(parent(x, y)) {
	  // Duplicate Placement
	  valid = true;
	  owner = false;
	  return;
	}

	// Check Validity of new Placement
	uint64_t const  bv = UINT64_C(1)<<x;
	uint64_t const  bh = UINT64_C(1)<<y;
	uint64_t const  bu = UINT64_C(1)<<(parent.N-1-x+y);
	uint64_t const  bd = UINT64_C(1)<<(           x+y);
	if((parent.bv&bv)||(parent.bh&bh)||(parent.bu&bu)||(parent.bd&bd)) {
	  valid = false;
	  owner = false;
	  return;
	}
	parent(x, y) = true;
	parent.bv |= bv;
	parent.bh |= bh;
	parent.bu |= bu;
	parent.bd |= bd;
	valid = true;
	owner = true;
      }
    public:
      ~Placement() {
	if(owner) {
	  parent.bv ^= UINT64_C(1)<<x;
	  parent.bh ^= UINT64_C(1)<<y;
	  parent.bu ^= UINT64_C(1)<<(parent.N-1-x+y);
	  parent.bd ^= UINT64_C(1)<<(           x+y);
	  parent(x, y) = false;
	}
      }

    public:
      operator bool() { return  valid; }

    }; // class Placement

    Placement place(unsigned  x, unsigned  y) {
      return  Placement(*this, x, y);
    }

    uint64_t getBV() const { return  bv; }
    uint64_t getBH() const { return  bh; }
    uint64_t getBU() const { return  bu; }
    uint64_t getBD() const { return  bd; }
  }; // class Board

//再帰でクイーンを置いていく
uint64_t countCompletions(uint64_t  bv,
				     uint64_t  bh,
				     uint64_t  bu,
				     uint64_t  bd) {
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
//クイーンを置いていく
void process(Board const &brd, int  sym,uint64_t *pre,uint64_t *cnt) {
  pre[sym]++;
  if(cnt) {
    unsigned const  N = brd.N;
    cnt[sym] += countCompletions(brd.getBV() >> 2,
				     ((((brd.getBH()>>2)|(~0<<(N-4)))+1)<<(brd.N-5))-1,
				     brd.getBU()>>4,
				     (brd.getBD()>>4)<<(N-5));
  }
} // process()
int main(int const  argc, char const* const argv[]) {
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
  struct pres_t {
    char unsigned  a;
    char unsigned  b;
  };
  std::unique_ptr<pres_t[]>  pres(new pres_t[(N-2)*(N-1)]);

  { // Compute all valid two-column pre-placements in order:
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
  }
  std::cout << "First side bound: ("
	    << (unsigned)pres[(N/2)*(N-3)  ].a << ", " << (unsigned)pres[(N/2)*(N-3)  ].b << ") / ("
	    << (unsigned)pres[(N/2)*(N-3)+1].a << ", " << (unsigned)pres[(N/2)*(N-3)+1].b << ')'
	    << std::endl;

  // Generate coronal Placements
  Board  board(N);
  //上2行は2分の1だけ実行(ミラー)
  for(unsigned  w = 0; w <= (N/2)*(N-3); w++) {
    unsigned const  wa = pres[w].a;
    unsigned const  wb = pres[w].b;
#ifdef TRACE
    std::cerr << '(' << wa << ", " << wb << ')' << std::endl;
#else
    std::cout << "\rProgress: " << w << '/' << ((N/2)*(N-3)) << std::flush;
#endif

    //上２行　0行目,1行目にクイーンを置く
    Board::Placement  pwa(board.place(0, wa));
    Board::Placement  pwb(board.place(1, wb));

    for(unsigned  n = w; n < (N-2)*(N-1)-w; n++) {
      unsigned const  na = pres[n].a;
      unsigned const  nb = pres[n].b;
#ifdef TRACE
      std::cerr << '(' << wa << ", " << wb << ')'
		<< '(' << na << ", " << nb << ')' << std::endl;
#endif

      //pre-place 左２列にクイーンを置く
      Board::Placement  pna(board.place(na, N-1)); if(!pna)  continue;
      Board::Placement  pnb(board.place(nb, N-2)); if(!pnb)  continue;

      for(unsigned  e = w; e < (N-2)*(N-1)-w; e++) {
	unsigned const  ea = pres[e].a;
	unsigned const  eb = pres[e].b;
#ifdef TRACE
	std::cerr << '(' << wa << ", " << wb << ')'
		  << '(' << na << ", " << nb << ')'
		  << '(' << ea << ", " << eb << ')' << std::endl;
#endif

        //下２行に置く
	Board::Placement  pea(board.place(N-1, N-1-ea)); if(!pea)  continue;
	Board::Placement  peb(board.place(N-2, N-1-eb)); if(!peb)  continue;

	for(unsigned  s = w; s < (N-2)*(N-1)-w; s++) {
	  unsigned const  sa = pres[s].a;
	  unsigned const  sb = pres[s].b;
#ifdef TRACE
	  std::cerr << '(' << wa << ", " << wb << ')'
		    << '(' << na << ", " << nb << ')'
		    << '(' << ea << ", " << eb << ')'
		    << '(' << sa << ", " << sb << ')' << std::endl;
#endif

          //右２列に置く
	  Board::Placement  psa(board.place(N-1-sa, 0)); if(!psa)  continue;
	  Board::Placement  psb(board.place(N-1-sb, 1)); if(!psb)  continue;

	  // We have a successful complete pre-placement with
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
	    //w=n=e=sであれば90度回転で同じ可能性
            process(board,0,pre,cnt);
	    continue;
	  }
	  if(e == w) {
          //e==wは180度回転して同じ
	  // check if 180°-rotation is smaller
            //180度回転して同じ時n>=sの時はsmaller
	    if(n >= s) {
	      if(n > s) {
		//print('e', wa, wb, na, nb, ea, eb, sa, sb);
		continue;
	      }
              process(board,1,pre,cnt);
	      continue;
	    }
	  }
	  // n = w is okay

	  //print('o', wa, wb, na, nb, ea, eb, sa, sb);
          process(board,2,pre,cnt);

	} // s
      } // e
    } // n
  } // w

  //std::cout << "\n\n" << *act << std::endl;
  uint64_t  uniq;
  uint64_t  total;
  uniq=cnt[0]+cnt[1]+cnt[2];
  total=cnt[0]*2+cnt[1]*4+cnt[2]*8;
  printf("\nTOTAL:%llu UNIQ:%llu\n",total,uniq);
}
