// con.c — single-file buildable (あなたの main + dump を活かしつつ、足りない型/関数を内蔵)

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <time.h>
#include <string.h>
#include <inttypes.h>
#include <stdlib.h>
#include <sys/time.h>

/*==================== clock_gettime フォールバック ====================*/
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
static int clock_gettime(int, struct timespec* ts) {
  struct timeval tv; gettimeofday(&tv, NULL);
  ts->tv_sec = tv.tv_sec;
  ts->tv_nsec = tv.tv_usec * 1000L;
  return 0;
}
#endif

/*==================== ここから：最小コンテナ実装 ====================*/
/* --- UIntSet: 20bit整数(ijkl)用の簡易Set（線形） --- */
typedef struct UIntSet {
  uint32_t* a;
  size_t n, cap;
} UIntSet;

static UIntSet* uintset_create(void){
  UIntSet* s = (UIntSet*)calloc(1, sizeof(UIntSet));
  s->cap = 256;
  s->a = (uint32_t*)calloc(s->cap, sizeof(uint32_t));
  return s;
}
static void uintset_free(UIntSet* s){ if(!s) return; free(s->a); free(s); }
static void uintset_clear(UIntSet* s){ if(!s) return; s->n = 0; }
static bool uintset_contains(const UIntSet* s, uint32_t v){
  if(!s) return false;
  for(size_t i=0;i<s->n;++i) if(s->a[i]==v) return true;
  return false;
}
static bool uintset_add(UIntSet* s, uint32_t v){
  if(!s) return false;
  if (uintset_contains(s, v)) return false;
  if (s->n == s->cap){
    s->cap <<= 1;
    s->a = (uint32_t*)realloc(s->a, s->cap * sizeof(uint32_t));
  }
  s->a[s->n++] = v;
  return true;
}
typedef struct { size_t i; } UIntSetIter;
static bool uintset_iter_next(const UIntSet* s, UIntSetIter* it, uint32_t* out){
  if(!s) return false;
  if (it->i >= s->n) return false;
  *out = s->a[it->i++];
  return true;
}
static size_t consts_dummy_zero(void){ return 0; } // ダミーで使うだけ

/* --- Constellation/Constellations: ベクタ --- */
typedef struct {
  uint32_t ld, rd, col, startijkl, solutions;
} Constellation;

typedef struct Constellations {
  Constellation* a;
  size_t n, cap;
} Constellations;

static Constellations* consts_create(void){
  Constellations* v = (Constellations*)calloc(1, sizeof(*v));
  v->cap = 128;
  v->a = (Constellation*)calloc(v->cap, sizeof(Constellation));
  return v;
}
static void consts_free(Constellations* v){ if(!v) return; free(v->a); free(v); }
static size_t consts_size(const Constellations* v){ return v ? v->n : 0; }
static void consts_push(Constellations* v, Constellation c){
  if (v->n == v->cap){
    v->cap <<= 1;
    v->a = (Constellation*)realloc(v->a, v->cap * sizeof(Constellation));
  }
  v->a[v->n++] = c;
}
static Constellation* consts_back_n(Constellations* v, size_t n_from_back){
  if (!v || v->n == 0 || n_from_back >= v->n) return NULL;
  return &v->a[v->n - 1 - n_from_back];
}
static const Constellation* consts_at(const Constellations* v, size_t idx){
  if (!v || idx >= v->n) return NULL;
  return &v->a[idx];
}

/* SigSet は set_pre_queens_cached に渡すだけのダミー */

/*==================== ここから：bit pack ヘルパ ====================*/
static inline uint32_t to_ijkl(int i,int j,int k,int l){
  return ((uint32_t)i<<15)|((uint32_t)j<<10)|((uint32_t)k<<5)|(uint32_t)l;
}
static inline int geti(uint32_t x){ return (int)((x>>15)&0x1F); }
static inline int getj(uint32_t x){ return (int)((x>>10)&0x1F); }
static inline int getk(uint32_t x){ return (int)((x>> 5)&0x1F); }
static inline int getl(uint32_t x){ return (int)( x      &0x1F); }
static inline int ffmin(int a,int b){ return a<b?a:b; }
static inline uint32_t mirvert(uint32_t ijkl,int N){
  return to_ijkl(N-1-geti(ijkl), N-1-getj(ijkl), getl(ijkl), getk(ijkl));
}
static inline uint32_t rot90(uint32_t ijkl,int N){
  return ((uint32_t)(N-1-getk(ijkl))<<15)|((uint32_t)(N-1-getl(ijkl))<<10)|((uint32_t)getj(ijkl)<<5)|(uint32_t)geti(ijkl);
}
static inline bool check_rotations(const UIntSet* S,int i,int j,int k,int l,int N){
  uint32_t r90  = to_ijkl((N-1-k),(N-1-l),j,i);
  uint32_t r180 = to_ijkl((N-1-j),(N-1-i),(N-1-l),(N-1-k));
  uint32_t r270 = to_ijkl(l,k,(N-1-i),(N-1-j));
  return uintset_contains(S,r90)||uintset_contains(S,r180)||uintset_contains(S,r270);
}
static inline uint32_t jasmin(uint32_t ijkl,int N){
  int arg=0;
  int minv = ffmin(getj(ijkl), N-1-getj(ijkl));
  { int v=ffmin(geti(ijkl),N-1-geti(ijkl)); if (v<minv){ arg=2; minv=v; } }
  { int v=ffmin(getk(ijkl),N-1-getk(ijkl)); if (v<minv){ arg=3; minv=v; } }
  { int v=ffmin(getl(ijkl),N-1-getl(ijkl)); if (v<minv){ arg=1; minv=v; } }
  for(int t=0;t<arg;++t) ijkl=rot90(ijkl,N);
  if (getj(ijkl) < N-1-getj(ijkl)) ijkl=mirvert(ijkl,N);
  return ijkl;
}
static inline uint32_t get_jasmin(uint32_t c,int N){ return jasmin(c,N); }

/*==================== 探索本体のスタブ ====================*/
/* ※ ここはあなたの実装に差し替えてください */
/* ==================== SigSet: constellation signature set (uint64) ==================== */
/* 6値タプル (ld,rd,col,k,l,row) を 64bit にハッシュして管理します */
typedef struct {
  uint64_t* keys;
  unsigned char* used;
  size_t cap, sz;
} SigSet;

static inline uint64_t mix64(uint64_t x){
  x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27; x *= 0x94d049bb133111ebULL;
  x ^= x >> 31; return x;
}
static inline uint64_t sig_key(uint64_t ld, uint64_t rd, uint64_t col, int k, int l, int row){
  /* boost::hash_combine 風に混ぜる */
  uint64_t h = 0x9e3779b97f4a7c15ULL;
  h ^= mix64(ld)  + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
  h ^= mix64(rd)  + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
  h ^= mix64(col) + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
  h ^= mix64((uint64_t)k) + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
  h ^= mix64((uint64_t)l) + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
  h ^= mix64((uint64_t)row)+ 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
  return h;
}
static void sigset_init(SigSet* s){ memset(s, 0, sizeof(*s)); }
static void sigset_grow(SigSet* s){
  size_t ncap = s->cap ? s->cap*2 : 1024;
  uint64_t* nkeys = (uint64_t*)calloc(ncap, sizeof(uint64_t));
  unsigned char* nused = (unsigned char*)calloc(ncap, 1);
  size_t mask = ncap - 1;

  for (size_t i=0;i<s->cap;++i) if (s->used[i]){
    uint64_t v = s->keys[i];
    size_t j = (size_t)mix64(v) & mask;
    while (nused[j]) j = (j+1) & mask;
    nused[j] = 1; nkeys[j] = v;
  }
  free(s->keys); free(s->used);
  s->keys = nkeys; s->used = nused; s->cap = ncap;
}
static bool sigset_add_if_absent(SigSet* s, uint64_t v){
  if (s->cap == 0 || (s->sz+1)*10 >= s->cap*7) sigset_grow(s);
  size_t mask = s->cap - 1;
  size_t i = (size_t)mix64(v) & mask;
  while (s->used[i]){
    if (s->keys[i] == v) return false; /* already */
    i = (i+1) & mask;
  }
  s->used[i] = 1; s->keys[i] = v; s->sz++;
  return true;
}

/* ==================== set_pre_queens / cached (C移植) ==================== */
/* 先にプロトタイプ（gen_constellations から呼ばれるため） */
static void set_pre_queens_cached(
  uint64_t ld, uint64_t rd, uint64_t col,
  int k, int l, int row, int queens,
  uint64_t LD, uint64_t RD,
  int* counter,
  Constellations* constellations,
  int N, int preset_queens,
  UIntSet* visited,
  SigSet* constellation_signatures
);
static void set_pre_queens(
  uint64_t ld, uint64_t rd, uint64_t col,
  int k, int l, int row, int queens,
  uint64_t LD, uint64_t RD,
  int* counter,
  Constellations* constellations,
  int N, int preset_queens,
  UIntSet* visited,
  SigSet* constellation_signatures
);

/* Python版はローカルsetを毎回作っていたので実質ノーキャッシュ。
   ロジックを変えず、ここでも単に本体へ委譲します。 */
static void set_pre_queens_cached(
  uint64_t ld, uint64_t rd, uint64_t col,
  int k, int l, int row, int queens,
  uint64_t LD, uint64_t RD,
  int* counter,
  Constellations* constellations,
  int N, int preset_queens,
  UIntSet* visited,
  SigSet* constellation_signatures
){
  set_pre_queens(ld, rd, col, k, l, row, queens, LD, RD,
                 counter, constellations, N, preset_queens,
                 visited, constellation_signatures);
}

/* 本体 */
static void set_pre_queens(
  uint64_t ld, uint64_t rd, uint64_t col,
  int k, int l, int row, int queens,
  uint64_t LD, uint64_t RD,
  int* counter,
  Constellations* constellations,
  int N, int preset_queens,
  UIntSet* visited,
  SigSet* constellation_signatures
){
  const uint64_t mask = ((uint64_t)1 << N) - 1u;

  /* --- state hash (軽量O(1)) による枝刈り --- */
  uint64_t h64 =
      (ld<<3) ^ (rd<<2) ^ (col<<1) ^ (uint64_t)row ^
      ((uint64_t)queens<<7) ^ ((uint64_t)k<<12) ^ ((uint64_t)l<<17) ^
      (LD<<22) ^ (RD<<27) ^ ((uint64_t)N<<1);
  /* visited は UIntSet(uint32) なので32bitに圧縮（衝突許容、元コードも衝突可能性を許容） */
  uint32_t h = (uint32_t)(h64 ^ (h64>>32));
  if (uintset_contains(visited, h)) return;
  uintset_add(visited, h);

  /* --- k 行 / l 行はスキップ --- */
  if (row == k || row == l) {
    set_pre_queens_cached(ld<<1, rd>>1, col,
                          k, l, row+1, queens,
                          LD, RD,
                          counter, constellations,
                          N, preset_queens, visited, constellation_signatures);
    return;
  }

  /* --- 既定数のクイーンを置いたら星座を保存（signature重複排除） --- */
  if (queens == preset_queens) {
    if (constellation_signatures && constellation_signatures->cap == 0)
      sigset_init(constellation_signatures);

    uint64_t sig = sig_key(ld, rd, col, k, l, row);
    bool fresh = true;
    if (constellation_signatures)
      fresh = sigset_add_if_absent(constellation_signatures, sig);

    if (fresh) {
      Constellation c;
      c.ld  = (uint32_t)ld;   /* dump用途は32bitで十分 */
      c.rd  = (uint32_t)rd;
      c.col = (uint32_t)col;
      c.startijkl = (uint32_t)(row << 20); /* Python: row<<20 を踏襲 */
      c.solutions = 0;
      consts_push(constellations, c);
      if (counter) (*counter)++;
    }
    return;
  }

  /* --- 次の行に置けるビット --- */
  uint64_t free = ~(ld | rd | col | (LD >> (N-1-row)) | (RD << (N-1-row))) & mask;

  while (free) {
    uint64_t bit = free & -free;
    free &= free - 1;

    set_pre_queens_cached( (ld|bit)<<1, (rd|bit)>>1, col|bit,
                           k, l, row+1, queens+1,
                           LD, RD,
                           counter, constellations,
                           N, preset_queens, visited, constellation_signatures);
  }
}


/*==================== gen_constellations（翻訳版） ====================*/
static void gen_constellations(UIntSet* ijkl_list,
                               Constellations* constellations,
                               int N, int preset_queens)
{
  (void)preset_queens; /* 本体側で使用 */
  const int halfN=(N+1)/2, N1=N-1;

  /* 奇数N: 中央列特別処理 */
  if (N & 1){
    const int center=N/2;
    for (int l=center+1; l<N1; ++l)
      for (int i=center+1; i<N1; ++i){
        if (i==(N1)-l) continue;
        for (int j=N-center-2; j>0; --j){
          if (j==i || j==l) continue;
          if (!check_rotations(ijkl_list,i,j,center,l,N))
            uintset_add(ijkl_list, to_ijkl(i,j,center,l));
        }
      }
  }

  /* 一般候補 */
  for (int k=1; k<halfN; ++k)
    for (int l=k+1; l<N1; ++l)
      for (int i=k+1; i<N-1; ++i){
        if (i==(N-1)-l) continue;
        for (int j=N-k-2; j>0; --j){
          if (j==i || j==l) continue;
          if (!check_rotations(ijkl_list,i,j,k,l,N))
            uintset_add(ijkl_list, to_ijkl(i,j,k,l));
        }
      }

  /* 端の種 */
  for (int j=1; j<N-2; ++j)
    for (int l=j+1; l<N1; ++l)
      uintset_add(ijkl_list, to_ijkl(0,j,0,l));

  /* それぞれを jasmin 正規化 */
  UIntSet* jas = uintset_create();
  { UIntSetIter it={0}; uint32_t v;
    while (uintset_iter_next(ijkl_list,&it,&v)) uintset_add(jas, get_jasmin(v,N));
  }
  uintset_clear(ijkl_list);
  { UIntSetIter it={0}; uint32_t v;
    while (uintset_iter_next(jas,&it,&v)) uintset_add(ijkl_list, v);
  }
  uintset_free(jas);

  /* 各 canonical start からサブコンステレーション生成 */
  const uint32_t L = (uint32_t)1u << N1;

  { UIntSetIter it={0}; uint32_t sc;
    while (uintset_iter_next(ijkl_list,&it,&sc)) {
      SigSet sigs = {0}; /* ダミー */

      const int i=geti(sc), j=getj(sc), k=getk(sc), l=getl(sc);
      const uint32_t Lj=(L>>j), Li=(L>>i), Ll=(L>>l);
      const uint32_t ld=((i>0)?(L>>(i-1)):0u) | (1u<<(N-k));
      const uint32_t rd=(L>>(i+1)) | (1u<<(l-1));
      const uint32_t col=1u|L|Li|Lj;
      const uint32_t LD = Lj|Ll;
      const uint32_t RD = Lj|(1u<<k);

      int counter = 0;
      UIntSet* visited = uintset_create();

      set_pre_queens_cached(ld,rd,col,k,l,1,(j==N1)?3:4,LD,RD,
                            &counter,constellations,N,preset_queens,visited,&sigs);

      const uint32_t base = to_ijkl(i,j,k,l);
      for (int a=0; a<counter; ++a) {
        Constellation* last_a = consts_back_n(constellations,(size_t)a);
        if (last_a) last_a->startijkl |= base;
      }
      uintset_free(visited);
    }
  }
}

/*==================== N<=5 のフォールバック（スタブ） ====================*/
static int bit_total(int N){ return (N==5)?10:0; }

/*==================== あなたの main + dump（そのまま） ====================*/
static void dump_constellations(const Constellations* v, int N, int limit) {
  size_t cnt = consts_size(v);
  int show = (limit < 0 || (size_t)limit > cnt) ? (int)cnt : limit;

  printf("---- dump constellations (N=%d, total=%zu, show=%d) ----\n",
         N, cnt, show);

  for (int idx = 0; idx < show; ++idx) {
    const Constellation* c = consts_at(v, (size_t)idx);
    if (!c) continue;

    int i = geti(c->startijkl);
    int j = getj(c->startijkl);
    int k = getk(c->startijkl);
    int l = getl(c->startijkl);

    printf("#%d: startijkl=0x%05" PRIX32 " (i=%d, j=%d, k=%d, l=%d)"
           "  solutions=%" PRIu32
           "  ld=0x%08" PRIX32 "  rd=0x%08" PRIX32 "  col=0x%08" PRIX32 "\n",
           idx, c->startijkl, i, j, k, l,
           c->solutions, c->ld, c->rd, c->col);
  }

  if (show < (int)cnt) {
    printf("... (%zu more not shown)\n", cnt - (size_t)show);
  }
}

static void format_duration(char* out, size_t outsz,
                            struct timespec t0, struct timespec t1) {
  long sec  = (long)(t1.tv_sec - t0.tv_sec);
  long nsec = t1.tv_nsec - t0.tv_nsec;
  if (nsec < 0) { nsec += 1000000000L; sec -= 1; }
  long ms = nsec / 1000000L;

  long h = sec / 3600;
  long m = (sec % 3600) / 60;
  long s = sec % 60;
  snprintf(out, outsz, "%ld:%02ld:%02ld.%03ld", h, m, s, ms);
}


/*==================== exec_solutions + dfs ====================*/

typedef struct { int next_funcid; int funcptn; int availptn; } FuncMeta;

static inline bool symmetry90_u32(uint32_t ijkl, int N){
  /* 90度回転後と一致するか */
  uint32_t rot = rot90(ijkl, N);
  return ijkl == rot;
}
static inline int symmetry_u32(uint32_t ijkl, int N){
  if (symmetry90_u32(ijkl, N)) return 2;
  /* 180度対称: (i,j)=(N-1-j,N-1-i) と (k,l)=(N-1-l,N-1-k) */
  int i=geti(ijkl), j=getj(ijkl), k=getk(ijkl), l=getl(ijkl);
  if (i == (N-1-j) && k == (N-1-l)) return 4;
  return 8;
}

/* FID 定義（Codonの並びと一致） */
enum {
  FID_SQBkBlBjrB=0, FID_SQBlBjrB=1, FID_SQBjrB=2, FID_SQB=3,
  FID_SQBklBjrB=4, FID_SQBlBkBjrB=5, FID_SQBkBjrB=6, FID_SQBlkBjrB=7,
  FID_SQBjlBkBlBjrB=8, FID_SQBjlBklBjrB=9, FID_SQBjlBlBkBjrB=10, FID_SQBjlBlkBjrB=11,
  FID_SQd2BkBlB=12, FID_SQd2BlB=13, FID_SQd2B=14, FID_SQd2BklB=15, FID_SQd2BlBkB=16,
  FID_SQd2BkB=17, FID_SQd2BlkB=18, FID_SQd1BkBlB=19, FID_SQd1BlB=20, FID_SQd1B=21,
  FID_SQd1BklB=22, FID_SQd1BlBkB=23, FID_SQd1BlkB=24, FID_SQd1BkB=25, FID_SQd0B=26, FID_SQd0BkB=27
};

/* dfs 本体（Codonのロジックを忠実移植） */
static int dfs_exec(
  int functionid,
  uint32_t ld, uint32_t rd, uint32_t col,
  int row, uint32_t free_mask,
  int jmark, int endmark, int mark1, int mark2,
  uint32_t board_mask,
  const uint32_t* blockK_by_funcid, const uint32_t* blockl_by_funcid,
  const FuncMeta* meta,
  int N, int N1, uint32_t NK, uint32_t NJ
){
  uint32_t avail = free_mask;
  int total = 0;

  int next_funcid = meta[functionid].next_funcid;
  int funcptn     = meta[functionid].funcptn;
  int avail_flag  = meta[functionid].availptn;

  /* ---- P6: endmark 基底 ---- */
  if (funcptn == 5 && row == endmark){
    if (functionid == FID_SQd2B){
      /* avail & (~1) が立っていれば 1 */
      return ((avail & (~(uint32_t)1)) != 0) ? 1 : 0;
    }
    return 1;
  }

  /* ---- P5: N-1-jmark 入口（行据え置きの一手前処理）---- */
  if (funcptn == 4){
    if (row == N1 - jmark){
      rd |= NJ; /* rd |= 1<<N1 */
      uint32_t next_free = board_mask & ~((ld<<1) | (rd>>1) | col);
      if (next_free){
        total += dfs_exec(next_funcid, ld<<1, rd>>1, col, row,
                          next_free, jmark, endmark, mark1, mark2,
                          board_mask,
                          blockK_by_funcid, blockl_by_funcid, meta,
                          N, N1, NK, NJ);
      }
      return total;
    }
  }

  /* 以降：共通配置ループ（既定は +1 進む） */
  int step = 1;
  uint32_t add1 = 0;
  int row_step = row + step;
  bool use_blocks = false;
  bool use_future = (avail_flag == 1);
  uint32_t blockK = 0, blockl = 0;
  int local_next_funcid = functionid;

  /* --- P4: jmark 特殊を前処理 --- */
  if (funcptn == 3 && row == jmark){
    avail &= ~1u; /* 列0禁止 */
    ld |= 1u;     /* 左斜線 LSB を立てる */
    local_next_funcid = next_funcid;
  }
  /* --- P1/P2/P3: mark 行で step=2/3 + block 適用 --- */
  else if (funcptn == 0 || funcptn == 1 || funcptn == 2){
    bool at_mark = (funcptn == 1) ? (row == mark2) : (row == mark1);
    if (at_mark && avail){
      step = (funcptn == 2) ? 3 : 2;
      add1 = (funcptn == 1 && functionid == FID_SQd1BlB) ? 1u : 0u; /* FID 20 のみ */
      row_step = row + step;

      blockK = blockK_by_funcid[functionid];
      blockl = blockl_by_funcid[functionid];

      use_blocks = true;
      use_future = false; /* ここは next_free のみで分岐 */
      local_next_funcid = next_funcid;
    }
  }

  if (use_blocks){
    while (avail){
      uint32_t bit = avail & -avail;
      avail &= avail - 1;

      uint32_t next_ld  = ((ld | bit) << step) | add1;
      uint32_t next_rd  = (rd | bit) >> step;
      uint32_t next_col =  col | bit;

      next_ld |= blockl;
      next_rd |= blockK;

      uint32_t blocked   = next_ld | next_rd | next_col;
      uint32_t next_free = board_mask & ~blocked;
      if (next_free){
        total += dfs_exec(local_next_funcid, next_ld, next_rd, next_col, row_step,
                          next_free, jmark, endmark, mark1, mark2,
                          board_mask,
                          blockK_by_funcid, blockl_by_funcid, meta,
                          N, N1, NK, NJ);
      }
    }
  } else {
    if (!use_future){
      while (avail){
        uint32_t bit = avail & -avail;
        avail &= avail - 1;

        uint32_t next_ld  = ((ld | bit) << step) | add1;
        uint32_t next_rd  = (rd | bit) >> step;
        uint32_t next_col =  col | bit;

        uint32_t blocked   = next_ld | next_rd | next_col;
        uint32_t next_free = board_mask & ~blocked;
        if (next_free){
          total += dfs_exec(local_next_funcid, next_ld, next_rd, next_col, row_step,
                            next_free, jmark, endmark, mark1, mark2,
                            board_mask,
                            blockK_by_funcid, blockl_by_funcid, meta,
                            N, N1, NK, NJ);
        }
      }
    } else {
      /* “+1 with 先読み” */
      while (avail){
        uint32_t bit = avail & -avail;
        avail &= avail - 1;

        uint32_t next_ld  = ((ld | bit) << step) | add1;
        uint32_t next_rd  = (rd | bit) >> step;
        uint32_t next_col =  col | bit;

        uint32_t blocked   = next_ld | next_rd | next_col;
        uint32_t next_free = board_mask & ~blocked;
        if (!next_free) continue;

        if (row_step >= endmark){
          total += dfs_exec(local_next_funcid, next_ld, next_rd, next_col, row_step,
                            next_free, jmark, endmark, mark1, mark2,
                            board_mask,
                            blockK_by_funcid, blockl_by_funcid, meta,
                            N, N1, NK, NJ);
          continue;
        }

        int use_j = (funcptn == 4); /* P5 ファミリのみ J 行を有効化 */
        int m1 = (row_step == mark1) ? 1 : 0;
        int m2 = (row_step == mark2) ? 1 : 0;
        int mj = (use_j && (row_step == (N1 - jmark))) ? 1 : 0;

        uint32_t extra  = ((uint32_t)(m1 | m2) * NK) | ((uint32_t)mj * NJ);
        uint32_t future = board_mask & ~(((next_ld<<1) | (next_rd>>1) | next_col) | extra);
        if (future){
          total += dfs_exec(local_next_funcid, next_ld, next_rd, next_col, row_step,
                            next_free, jmark, endmark, mark1, mark2,
                            board_mask,
                            blockK_by_funcid, blockl_by_funcid, meta,
                            N, N1, NK, NJ);
        }
      }
    }
  }
  return total;
}

static void exec_solutions(Constellations* constellations, int N){
  const int N1 = N - 1;
  const int N2 = N - 2;
  const uint32_t small_mask = (N2 >= 0) ? ((1u << (uint32_t)N2) - 1u) : 0u;
  const uint32_t board_mask = (N  >  0) ? ((1u << (uint32_t)N ) - 1u) : 0u;

  /* func_meta: next_funcid, funcptn, availptn */
  static const FuncMeta meta[28] = {
    {1,0,0},  /*  0 SQBkBlBjrB   -> P1, 先読みなし */
    {2,1,0},  /*  1 SQBlBjrB     -> P2, 先読みなし */
    {3,3,1},  /*  2 SQBjrB       -> P4, 先読みあり */
    {3,5,1},  /*  3 SQB          -> P6, 先読みあり */
    {2,2,0},  /*  4 SQBklBjrB    -> P3, 先読みなし */
    {6,0,0},  /*  5 SQBlBkBjrB   -> P1, 先読みなし */
    {2,1,0},  /*  6 SQBkBjrB     -> P2, 先読みなし */
    {2,2,0},  /*  7 SQBlkBjrB    -> P3, 先読みなし */
    {0,4,1},  /*  8 SQBjlBkBlBjrB-> P5, 先読みあり */
    {4,4,1},  /*  9 SQBjlBklBjrB -> P5, 先読みあり */
    {5,4,1},  /* 10 SQBjlBlBkBjrB-> P5, 先読みあり */
    {7,4,1},  /* 11 SQBjlBlkBjrB -> P5, 先読みあり */
    {13,0,0}, /* 12 SQd2BkBlB    -> P1, 先読みなし */
    {14,1,0}, /* 13 SQd2BlB      -> P2, 先読みなし */
    {14,5,1}, /* 14 SQd2B        -> P6, 先読みあり（avail 特例） */
    {14,2,0}, /* 15 SQd2BklB     -> P3, 先読みなし */
    {17,0,0}, /* 16 SQd2BlBkB    -> P1, 先読みなし */
    {14,1,0}, /* 17 SQd2BkB      -> P2, 先読みなし */
    {14,2,0}, /* 18 SQd2BlkB     -> P3, 先読みなし */
    {20,0,0}, /* 19 SQd1BkBlB    -> P1, 先読みなし */
    {21,1,0}, /* 20 SQd1BlB      -> P2, 先読みなし（add1=1 は dfs 内で扱う） */
    {21,5,1}, /* 21 SQd1B        -> P6, 先読みあり */
    {21,2,0}, /* 22 SQd1BklB     -> P3, 先読みなし */
    {25,0,0}, /* 23 SQd1BlBkB    -> P1, 先読みなし */
    {21,2,0}, /* 24 SQd1BlkB     -> P3, 先読みなし */
    {21,1,0}, /* 25 SQd1BkB      -> P2, 先読みなし */
    {26,5,1}, /* 26 SQd0B        -> P6, 先読みあり */
    {26,0,0}, /* 27 SQd0BkB      -> P1, 先読みなし */
  };

  /* blockl_by_funcid は固定配列（Codonどおり） */
  uint32_t blockl_by_funcid[28] =
    {0,1,0,0,1,1,0,2,0,0,0,0,0,1,0,1,1,0,2,0,0,0,1,1,2,0,0,0};

  /* blockK_by_funcid はカテゴリで N に依存 */
  uint32_t blockK_by_funcid[28] = {0};
  const uint32_t n3 = (N >= 3) ? (1u << (uint32_t)(N-3)) : 1u;
  const uint32_t n4 = (N >= 4) ? (1u << (uint32_t)(N-4)) : 1u;
  /* cat=3 の fid 群（N-3） */
  int cat3[] = {FID_SQBkBlBjrB, FID_SQBlkBjrB, FID_SQBkBjrB,
                FID_SQd2BkBlB, FID_SQd2BkB, FID_SQd2BlkB,
                FID_SQd1BkBlB, FID_SQd1BlkB, FID_SQd1BkB, FID_SQd0BkB};
  for (size_t x=0; x<sizeof(cat3)/sizeof(cat3[0]); ++x) blockK_by_funcid[cat3[x]] = n3;
  /* cat=4 の fid 群（N-4） */
  int cat4[] = {FID_SQBklBjrB, FID_SQd2BklB, FID_SQd1BklB};
  for (size_t x=0; x<sizeof(cat4)/sizeof(cat4[0]); ++x) blockK_by_funcid[cat4[x]] = n4;

  const uint32_t NK = (N >= 3) ? (1u << (uint32_t)(N-3)) : 0u;
  const uint32_t NJ = (N1 >= 0) ? (1u << (uint32_t)N1) : 0u;

  size_t m = consts_size(constellations);
  for (size_t idx=0; idx<m; ++idx){
    Constellation* c = (Constellation*)consts_at(constellations, idx);
    if (!c) continue;

    int jmark=0, mark1=0, mark2=0, endmark=0, target=0;

    uint32_t start_ijkl = c->startijkl;
    int start = (int)(start_ijkl >> 20);
    uint32_t ijkl = start_ijkl & ((1u << 20) - 1u);
    int j = getj(ijkl), k = getk(ijkl), l = getl(ijkl);

    uint32_t ld  = (c->ld  >> 1);
    uint32_t rd  = (c->rd  >> 1);
    uint32_t col = (c->col >> 1) | (~small_mask);

    uint32_t LD = (1u << (uint32_t)(N1 - j)) | (1u << (uint32_t)(N1 - l));
    ld |= (LD >> (uint32_t)(N - start));

    if (start > k){
      rd |= (1u << (uint32_t)(N1 - (start - k + 1)));
    }
    if (j >= 2*N - 33 - start){
      rd |= ( (1u << (uint32_t)(N1 - j)) << (uint32_t)(N2 - start) );
    }

    uint32_t free_mask = ~(ld | rd | col);

    if (j < (N - 3)){
      jmark = j + 1; endmark = N2;
      if (j > 2*N - 34 - start){
        if (k < l){
          mark1 = k-1; mark2 = l-1;
          if (start < l){
            if (start < k){
              target = (l != k+1) ? FID_SQBkBlBjrB : FID_SQBklBjrB;
            } else target = FID_SQBlBjrB;
          } else target = FID_SQBjrB;
        } else {
          mark1 = l-1; mark2 = k-1;
          if (start < k){
            if (start < l){
              target = (k != l+1) ? FID_SQBlBkBjrB : FID_SQBlkBjrB;
            } else target = FID_SQBkBjrB;
          } else target = FID_SQBjrB;
        }
      } else {
        if (k < l){
          mark1 = k-1; mark2 = l-1;
          target = (l != k+1) ? FID_SQBjlBkBlBjrB : FID_SQBjlBklBjrB;
        } else {
          mark1 = l-1; mark2 = k-1;
          target = (k != l+1) ? FID_SQBjlBlBkBjrB : FID_SQBjlBlkBjrB;
        }
      }
    } else if (j == (N - 3)){
      endmark = N2;
      if (k < l){
        mark1 = k-1; mark2 = l-1;
        if (start < l){
          if (start < k){
            target = (l != k+1) ? FID_SQd2BkBlB : FID_SQd2BklB;
          } else { mark2 = l-1; target = FID_SQd2BlB; }
        } else target = FID_SQd2B;
      } else {
        mark1 = l-1; mark2 = k-1;
        if (start < k){
          if (start < l){
            target = (k != l+1) ? FID_SQd2BlBkB : FID_SQd2BlkB;
          } else { mark2 = k-1; target = FID_SQd2BkB; }
        } else target = FID_SQd2B;
      }
    } else if (j == N2){
      if (k < l){
        endmark = N2;
        if (start < l){
          if (start < k){
            mark1 = k-1;
            if (l != k+1){ mark2 = l-1; target = FID_SQd1BkBlB; }
            else target = FID_SQd1BklB;
          } else { mark2 = l-1; target = FID_SQd1BlB; }
        } else target = FID_SQd1B;
      } else { /* l < k */
        if (start < k){
          if (start < l){
            if (k < N2){
              mark1 = l-1; endmark = N2;
              if (k != l+1){ mark2 = k-1; target = FID_SQd1BlBkB; }
              else target = FID_SQd1BlkB;
            } else {
              if (l != (N-3)){ mark2 = l-1; endmark = N-3; target = FID_SQd1BlB; }
              else { endmark = N-4; target = FID_SQd1B; }
            }
          } else {
            if (k != N2){ mark2 = k-1; endmark = N2; target = FID_SQd1BkB; }
            else { endmark = N-3; target = FID_SQd1B; }
          }
        } else { endmark = N2; target = FID_SQd1B; }
      }
    } else { /* j がコーナー */
      endmark = N2;
      if (start > k) target = FID_SQd0B;
      else { mark1 = k-1; target = FID_SQd0BkB; }
    }

    int cnt = dfs_exec(target, ld, rd, col, start,
                       free_mask, jmark, endmark, mark1, mark2,
                       board_mask,
                       blockK_by_funcid, blockl_by_funcid, meta,
                       N, N1, NK, NJ);

    c->solutions = (uint32_t)(cnt * symmetry_u32(ijkl, N));
  }
}



int main(void) {
  const int nmin = 5;
  const int nmax = 18;
  const int preset_queens = 4;

  const int expected[] = {0,0,0,0,0, 10,4,40,92,352,724,2680,14200,73712,365596,2279184,14772512,95815104};
  const int expected_len = (int)(sizeof(expected) / sizeof(expected[0]));

  printf(" N:        Total       Unique        hh:mm:ss.ms\n");

  for (int size = nmin; size < nmax; ++size) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    if (size <= 5) {
      int total = bit_total(size);
      clock_gettime(CLOCK_MONOTONIC, &t1);
      char buf[64]; format_duration(buf, sizeof(buf), t0, t1);
      printf("%2d:%13d%13d%20s\n", size, total, 0, buf);
      continue;
    }

    UIntSet* ijkl_list = uintset_create();
    Constellations* constellations = consts_create();

    gen_constellations(ijkl_list, constellations, size, preset_queens);

    exec_solutions(constellations, size);  

    //const int DUMP_LIMIT = -1;   /* -1: 全件 / 0: 何も出さない / N: 先頭N件 */
    //if (DUMP_LIMIT != 0) dump_constellations(constellations, size, DUMP_LIMIT);

    long long total = 0;
    size_t cnt = consts_size(constellations);
    for (size_t i = 0; i < cnt; ++i) {
      const Constellation* c = consts_at(constellations, i);
      if (!c) continue;
      if ((int)c->solutions > 0) total += (long long)c->solutions;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    char buf[64]; format_duration(buf, sizeof(buf), t0, t1);

    char status[64] = "ok";
    if (size >= 0 && size < expected_len) {
      if ((long long)expected[size] != total) {
        snprintf(status, sizeof(status), "ng(%lld!=%d)", total, expected[size]);
      }
    }
    printf("%2d:%13lld%13d%20s    %s\n", size, total, 0, buf, status);

    consts_free(constellations);
    uintset_free(ijkl_list);
  }
  return 0;
}
