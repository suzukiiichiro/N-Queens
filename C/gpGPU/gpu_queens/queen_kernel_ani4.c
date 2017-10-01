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
  #define NUM_QUEENS 12
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
  qint masks[NUM_QUEENS];
  uint64_t solutions; // Number of solutinos found so far.
  int step;
  int col;
  int startCol; // First column this individual computation was tasked with filling.
  qint mask;
  qint rook;
  qint add;
  qint sub;
  qint BOUND1;
};

CL_CONSTANT_KEYWORD const qint dodge = (1 << NUM_QUEENS) - 1;

CL_KERNEL_KEYWORD void place(CL_GLOBAL_KEYWORD struct queenState * state)
{
  int index = get_global_id(0);
  int id= state[index].id;
  qint masks[NUM_QUEENS];
  for (int i = 0; i < NUM_QUEENS; i++)
    masks[i] = state[index].masks[i];

  uint64_t solutions = state[index].solutions;
  char step      = state[index].step;
  int col       = state[index].col;
  int startCol  = state[index].startCol;
  qint mask     = state[index].mask;
  qint rook     = state[index].rook;
  qint add      = state[index].add;
  qint sub      = state[index].sub;
  qint BOUND1   = state[index].BOUND1;

  //printf("bound:%d:startCol:%d\n", BOUND1,startCol);
  uint16_t i = 1;
  while (i != 0)
  {
  	i++;

    if (step == REMOVE)
    {
      if (col == startCol)
      {
        step = DONE;
        break;
      }
      --col;
      mask = masks[col];
    }
    qint rext;
    if(col==0){
      if(mask & (1<<BOUND1)){
        rext=1<<BOUND1;
      }else{
        step=DONE;
        break;
      }
    }else{
      rext = mask & -mask;
    }
    rook ^= rext;
    add  ^= rext << col;
    sub  ^= rext << (NUM_QUEENS - 1 - col);

    if (step == PLACE)
    {
      masks[col] = mask;
      ++col;

      if (col != NUM_QUEENS)
      {
        mask = dodge & ~(rook | (add >> col) | (sub >> ((NUM_QUEENS - 1) - col)));

        if (mask == 0)
          step = REMOVE;
      }
      else
      {
        solutions += 1;
        step = REMOVE;
      }
    }
    else
    {
      mask ^= rext;

      if (mask == 0)
        step = REMOVE;
      else
        step = PLACE;
    }
  }

  // Save kernel state for next round.
  state[index].id      = id;
  for (int i = 0; i < NUM_QUEENS; i++)
    state[index].masks[i] = masks[i];
    state[index].solutions = solutions;
    state[index].step      = step;
    state[index].col       = col;
    state[index].startCol  = startCol;
    state[index].mask      = mask;
    state[index].rook      = rook;
    state[index].add       = add;
    state[index].sub       = sub;
    state[index].BOUND1 = BOUND1;
}

#ifdef GCC_STYLE

int main()
{
    struct queenState state = { };
    state.mask = dodge;

    place(&state);

    printf("%llu\n", state.solutions);

    return 0;
}

#endif
