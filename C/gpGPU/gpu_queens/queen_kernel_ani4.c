// Without OPENCL_STYLE defined, this program will compile with gcc/clang,
// which facilitates testing and experimentation. Without it defined, it
// compiles as an OpenCL shader.
#ifndef OPENCL_STYLE
  // Declarations appropriate to this program being compiled with gcc.
  #include "stdio.h"
  #include "stdint.h"
  typedef int64_t qint;
  // A stub for OpenCL's get_global_id function.
  int get_global_id(int dimension) { return 0; }
  #define CL_KERNEL_KEYWORD
  #define CL_GLOBAL_KEYWORD
  #define CL_CONSTANT_KEYWORD
  #define CL_PACKED_KEYWORD
  #define NUM_QUEENS 13
#else
  // Declarations appropriate to this program being compiled as an OpenCL
  // kernel. OpenCL has a 64 bit long and requires special keywords to designate
  // where and how different objects are stored in memory.
  typedef long qint;
  typedef long int64_t;
  typedef ulong uint64_t;
  typedef ushort uint16_t;
  #define CL_KERNEL_KEYWORD __kernel
  #define CL_GLOBAL_KEYWORD __global
  #define CL_CONSTANT_KEYWORD __constant
  #define CL_PACKED_KEYWORD  __attribute__ ((packed))
#endif

enum { PLACE, REMOVE, DONE };

// State of individual computation
struct CL_PACKED_KEYWORD queenState
{
  int id;
  qint aB[NUM_QUEENS];
  uint64_t lTotal; // Number of solutinos found so far.
  int step;
  int y;
  int startCol; // First column this individual computation was tasked with filling.
  qint bm;
  qint down;
  qint right;
  qint left;
  qint BOUND1;
};

CL_CONSTANT_KEYWORD const qint msk = (1 << NUM_QUEENS) - 1;

CL_KERNEL_KEYWORD void place(CL_GLOBAL_KEYWORD struct queenState * state)
{
  int index = get_global_id(0);
  int id= state[index].id;
  qint aB[NUM_QUEENS];
  for (int i = 0; i < NUM_QUEENS; i++)
    aB[i] = state[index].aB[i];

  uint64_t lTotal = state[index].lTotal;
  int step      = state[index].step;
  int y       = state[index].y;
  int startCol  = state[index].startCol;
  qint bm     = state[index].bm;
  qint down     = state[index].down;
  qint right      = state[index].right;
  qint left      = state[index].left;
  qint BOUND1   = state[index].BOUND1;

  //printf("bound:%d:startCol:%d\n", BOUND1,startCol);
  long i = 1;
  while (i != 0)
  {
  	i++;
    if (step == REMOVE)
    {
      if (y == startCol)
      {
        step = DONE;
        break;
      }
      --y;
      bm = aB[y];
    }
    qint bit;
    if(y==0){
      if(bm & (1<<BOUND1)){
        bit=1<<BOUND1;
      }else{
        step=DONE;
        break;
      }
    }else{
      bit = bm & -bm;
    }
    down ^= bit;
    right  ^= bit << y;
    left  ^= bit << (NUM_QUEENS - 1 - y);

    if (step == PLACE)
    {
      aB[y] = bm;
      ++y;

      if (y != NUM_QUEENS)
      {
        bm = msk & ~(down | (right >> y) | (left >> ((NUM_QUEENS - 1) - y)));

        if (bm == 0)
          step = REMOVE;
      }
      else
      {
        lTotal += 1;
        step = REMOVE;
      }
    }
    else
    {
      bm ^= bit;

      if (bm == 0)
        step = REMOVE;
      else
        step = PLACE;
    }
  }

  // Save kernel state for next round.
  state[index].id      = id;
  for (int i = 0; i < NUM_QUEENS; i++)
    state[index].aB[i] = aB[i];
    state[index].lTotal = lTotal;
    state[index].step      = step;
    state[index].y       = y;
    state[index].startCol  = startCol;
    state[index].bm      = bm;
    state[index].down      = down;
    state[index].right       = right;
    state[index].left       = left;
    state[index].BOUND1 = BOUND1;
}

#ifdef GCC_STYLE

int main()
{
    struct queenState state = { };
    state.bm = msk;

    place(&state);

    printf("%llu\n", state.lTotal);

    return 0;
}

#endif
