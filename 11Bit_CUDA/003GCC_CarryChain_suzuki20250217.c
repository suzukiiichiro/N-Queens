#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#define MAX 18

typedef struct{
  unsigned long total; // long → unsigned long
} NQueens17;

unsigned long solve(unsigned int row, unsigned int left, unsigned int down, unsigned int right){
  unsigned long total = 0;
  if ((down + 1) == 0) return 1;
  while (row & 1) {
    row >>= 1;
    left <<= 1;
    right >>= 1;
  }
  row >>= 1;
  unsigned int bitmap = ~(left | down | right);
  while (bitmap != 0) {
    unsigned int bit = bitmap & -bitmap;
    total += solve(row, (left | bit) << 1, down | bit, (right | bit) >> 1);
    bitmap ^= bit;
  }
  return total;
}

unsigned long process(int size, int sym, unsigned int B[]) {
  return sym * solve(B[0] >> 2, B[1] >> 4, (((((unsigned)B[2] >> 2) | ~(0U) << (size - 4)) + 1) << (size - 5)) - 1, (B[3] >> 4) << (size - 5));
}

unsigned long Symmetry(int size, int n, int w, int s, int e, unsigned int B[], int B4[]) {
  int ww = (size - 2) * (size - 1) - 1 - w;
  int w2 = (size - 2) * (size - 1) - 1;
  if (s == ww && n < (w2 - e)) return 0;
  if (e == ww && n > (w2 - n)) return 0;
  if (n == ww && e > (w2 - s)) return 0;
  if (!B4[0]) return process(size, 8, B);
  if (s == w) {
    if (n != w || e != w) return 0;
    return process(size, 2, B);
  }
  if (e == w && n >= s) {
    if (n > s) return 0;
    return process(size, 4, B);
  }
  return process(size, 8, B);
}

int placement(int size, int dimx, int dimy, unsigned int B[], int B4[]) {
  if (B4[dimx] == dimy) return 1;
  if (B4[0]) {
    if ((B4[0] != -1 && ((dimx < B4[0] || dimx >= size - B4[0]) && (dimy == 0 || dimy == size - 1))) ||
        ((dimx == size - 1) && (dimy <= B4[0] || dimy >= size - B4[0]))) return 0;
  } else if ((B4[1] != -1) && (B4[1] >= dimx && dimy == 1)) return 0;
  if ((B[0] & (1U << dimx)) || (B[1] & (1U << (size - 1 - dimx + dimy))) ||
      (B[2] & (1U << dimy)) || (B[3] & (1U << (dimx + dimy)))) return 0;
  B[0] |= 1U << dimx;
  B[1] |= 1U << (size - 1 - dimx + dimy);
  B[2] |= 1U << dimy;
  B[3] |= 1U << (dimx + dimy);
  B4[dimx] = dimy;
  return 1;
}

void deepcopy(unsigned int *src, unsigned int *dest, int size) {
  memcpy(dest, src, size * sizeof(unsigned int));
}

unsigned long buildChain(int size, int pres_a[], int pres_b[]) {
  unsigned long total = 0;
  unsigned int B[4] = {0, 0, 0, 0};
  int B4[MAX];
  for (int i = 0; i < size; i++) B4[i] = -1;
  int sizeE = size - 1;
  int sizeEE = size - 2;
  int range_size = (size / 2) * (size - 3) + 1;
  for (int w = 0; w < range_size; w++) {
    unsigned int wB[4];
    int wB4[MAX];
    deepcopy(B, wB, 4);
    memcpy(wB4, B4, sizeof(B4));
    if (!placement(size, 0, pres_a[w], wB, wB4) || !placement(size, 1, pres_b[w], wB, wB4)) continue;
    for (int n = w; n < (sizeEE) * (sizeE) - w; n++) {
      unsigned int nB[4];
      int nB4[MAX];
      deepcopy(wB, nB, 4);
      memcpy(nB4, wB4, sizeof(B4));
      if (!placement(size, pres_a[n], sizeE, nB, nB4) || !placement(size, pres_b[n], sizeEE, nB, nB4)) continue;
      for (int e = w; e < (sizeEE) * (sizeE) - w; e++) {
        unsigned int eB[4];
        int eB4[MAX];
        deepcopy(nB, eB, 4);
        memcpy(eB4, nB4, sizeof(B4));
        if (!placement(size, sizeE, sizeE - pres_a[e], eB, eB4) || !placement(size, sizeEE, sizeE - pres_b[e], eB, eB4)) continue;
        for (int s = w; s < (sizeEE) * (sizeE) - w; s++) {
          unsigned int sB[4];
          int sB4[MAX];
          deepcopy(eB, sB, 4);
          memcpy(sB4, eB4, sizeof(B4));
          if (!placement(size, sizeE - pres_a[s], 0, sB, sB4) || !placement(size, sizeE - pres_b[s], 1, sB, sB4)) continue;
          total += Symmetry(size, n, w, s, e, sB, sB4);
        }
      }
    }
  }
  return total;
}

