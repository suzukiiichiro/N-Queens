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
#include <sys/stat.h>

static inline uint64_t mix64(uint64_t x){
//splitmix64 のミキサ最終段。64bit 値 x を (>>/XOR/乗算) の 3 段で拡散して返す。 Zobrist テーブルの乱数品質を担保するために使用。  
  x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27; x *= 0x94d049bb133111ebULL;
  x ^= x >> 31; return x;
}

/* 追加の hash-combine ヘルパ（既存 mix64 を利用） */
static inline void hcombine_u64(uint64_t* h, uint64_t v){
  uint64_t x = *h;
  x ^= mix64(v) + 0x9e3779b97f4a7c15ULL + (x<<6) + (x>>2);
  *h = x;
}
//サブコンステレーション生成状態のメモ化（実行中の重複再帰を抑制）
static inline uint64_t StateKey(
  uint64_t ld, uint64_t rd, uint64_t col,
  int k, int l, int row, int queens,
  uint64_t LD, uint64_t RD,
  int N, int preset_queens
){
  uint64_t h = 0x9e3779b97f4a7c15ULL;
  hcombine_u64(&h, ld);
  hcombine_u64(&h, rd);
  hcombine_u64(&h, col);
  hcombine_u64(&h, (uint64_t)k);
  hcombine_u64(&h, (uint64_t)l);
  hcombine_u64(&h, (uint64_t)row);
  hcombine_u64(&h, (uint64_t)queens);
  hcombine_u64(&h, LD);
  hcombine_u64(&h, RD);
  hcombine_u64(&h, (uint64_t)N);
  hcombine_u64(&h, (uint64_t)preset_queens);
  return h;
}
/*==================== JASMIN CACHE ====================*/
/* N ごとにリセットする簡易キャッシュ（open addressing） */
typedef struct {
  uint32_t* keys;      /* 入力 ijkl (20bit) */
  uint32_t* vals;      /* 出力 jasmin(ijkl, N) */
  unsigned char* used; /* 空き=0/使用=1 */
  size_t cap, sz;
  int currN;           /* 現在の N（違ったら全リセット） */
} jasmin_cache;

static jasmin_cache g_jas = {0};

static inline uint32_t mix32(uint32_t x){
  /* 32bit の軽量ミキサ */
  x ^= x >> 16; x *= 0x7feb352dU;
  x ^= x >> 15; x *= 0x846ca68bU;
  x ^= x >> 16; return x;
}

static void jas_cache_free(void){
  free(g_jas.keys); free(g_jas.vals); free(g_jas.used);
  memset(&g_jas, 0, sizeof(g_jas));
}

static void jas_cache_reset(int N){
  if (g_jas.currN == N && g_jas.cap) {
    /* N 同一なら内容だけクリア */
    memset(g_jas.used, 0, g_jas.cap);
    g_jas.sz = 0;
    return;
  }
  /* N 変更 or 初期化 */
  jas_cache_free();
  g_jas.currN = N;
  g_jas.cap = 4096; /* 初期容量：必要なら自動拡張 */
  g_jas.keys = (uint32_t*)calloc(g_jas.cap, sizeof(uint32_t));
  g_jas.vals = (uint32_t*)calloc(g_jas.cap, sizeof(uint32_t));
  g_jas.used = (unsigned char*)calloc(g_jas.cap, 1);
  g_jas.sz = 0;
}

static void jas_cache_grow(void){
  size_t ncap = g_jas.cap ? g_jas.cap * 2 : 4096;
  uint32_t* nkeys = (uint32_t*)calloc(ncap, sizeof(uint32_t));
  uint32_t* nvals = (uint32_t*)calloc(ncap, sizeof(uint32_t));
  unsigned char* nused = (unsigned char*)calloc(ncap, 1);
  size_t mask = ncap - 1;

  for (size_t i=0;i<g_jas.cap;++i){
    if (!g_jas.used[i]) continue;
    uint32_t k = g_jas.keys[i];
    size_t j = (size_t)mix32(k) & mask;
    while (nused[j]) j = (j+1) & mask;
    nused[j] = 1; nkeys[j] = k; nvals[j] = g_jas.vals[i];
  }
  free(g_jas.keys); free(g_jas.vals); free(g_jas.used);
  g_jas.keys = nkeys; g_jas.vals = nvals; g_jas.used = nused; g_jas.cap = ncap;
}

static bool jas_cache_get(uint32_t ijkl, uint32_t* out){
  if (!g_jas.cap) return false;
  size_t mask = g_jas.cap - 1;
  size_t i = (size_t)mix32(ijkl) & mask;
  while (g_jas.used[i]){
    if (g_jas.keys[i] == ijkl){
      *out = g_jas.vals[i]; return true;
    }
    i = (i+1) & mask;
  }
  return false;
}

static void jas_cache_put(uint32_t ijkl, uint32_t val){
  if (!g_jas.cap) jas_cache_reset(g_jas.currN);
  if ((g_jas.sz+1) * 10 >= g_jas.cap * 7) jas_cache_grow();
  size_t mask = g_jas.cap - 1;
  size_t i = (size_t)mix32(ijkl) & mask;
  while (g_jas.used[i]) i = (i+1) & mask;
  g_jas.used[i] = 1; g_jas.keys[i] = ijkl; g_jas.vals[i] = val; g_jas.sz++;
}


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



/*(i,j,k,l) を 5bit×4=20bit にパック/アンパックするユーティリティ。 mirvert は上下ミラー（行: N-1-?）＋ (k,l) の入れ替えで左右ミラー相当を実現。*/
static inline uint32_t to_ijkl(int i,int j,int k,int l){
  return ((uint32_t)i<<15)|((uint32_t)j<<10)|((uint32_t)k<<5)|(uint32_t)l;
}
static inline int ffmin(int a,int b){ return a<b?a:b; }
static inline int geti(uint32_t x){ return (int)((x>>15)&0x1F); }
static inline int getj(uint32_t x){ return (int)((x>>10)&0x1F); }
static inline int getk(uint32_t x){ return (int)((x>> 5)&0x1F); }
static inline int getl(uint32_t x){ return (int)( x      &0x1F); }
static inline uint32_t mirvert(uint32_t ijkl,int N){
  return to_ijkl(N-1-geti(ijkl), N-1-getj(ijkl), getl(ijkl), getk(ijkl));
}
//(i,j,k,l) パック値に対して盤面 90°/180° 回転を適用した新しいパック値を返す。 回転の定義: (r,c) -> (c, N-1-r)。対称性チェック・正規化に利用。
static inline uint32_t rot90(uint32_t ijkl,int N){
  return ((uint32_t)(N-1-getk(ijkl))<<15)|((uint32_t)(N-1-getl(ijkl))<<10)|((uint32_t)getj(ijkl)<<5)|(uint32_t)geti(ijkl);
}
static inline bool symmetry90(uint32_t ijkl, int N){
  /* 90度回転後と一致するか */
  uint32_t rot = rot90(ijkl, N);
  return ijkl == rot;
}

static inline int symmetry(uint32_t ijkl, int N){
  if (symmetry90(ijkl, N)) return 2;
  /* 180度対称: (i,j)=(N-1-j,N-1-i) と (k,l)=(N-1-l,N-1-k) */
  int i=geti(ijkl), j=getj(ijkl), k=getk(ijkl), l=getl(ijkl);
  if (i == (N-1-j) && k == (N-1-l)) return 4;
  return 8;
}
//与えた (i,j,k,l) の 90/180/270° 回転形が既出集合 ijkl_list に含まれるかを判定する
static inline bool check_rotations(const UIntSet* S,int i,int j,int k,int l,int N){
  uint32_t r90  = to_ijkl((N-1-k),(N-1-l),j,i);
  uint32_t r180 = to_ijkl((N-1-j),(N-1-i),(N-1-l),(N-1-k));
  uint32_t r270 = to_ijkl(l,k,(N-1-i),(N-1-j));
  return uintset_contains(S,r90)||uintset_contains(S,r180)||uintset_contains(S,r270);
}
static inline uint32_t jasmin(uint32_t ijkl,int N){
  //最初の最小値と引数を設定
  int arg=0;  
  int minv = ffmin(getj(ijkl), N-1-getj(ijkl));
  //i: 最初の行（上端） 90度回転2回
  { int v=ffmin(geti(ijkl),N-1-geti(ijkl)); if (v<minv){ arg=2; minv=v; } }
  //k: 最初の列（左端） 90度回転3回
  { int v=ffmin(getk(ijkl),N-1-getk(ijkl)); if (v<minv){ arg=3; minv=v; } }
  //l: 最後の列（右端） 90度回転1回
  { int v=ffmin(getl(ijkl),N-1-getl(ijkl)); if (v<minv){ arg=1; minv=v; } }
  //90度回転を arg 回繰り返す
  for(int t=0;t<arg;++t) ijkl=rot90(ijkl,N);
  //必要に応じて垂直方向のミラーリングを実行
  if (getj(ijkl) < N-1-getj(ijkl)) ijkl=mirvert(ijkl,N);
  return ijkl;
}

/* キャッシュ付き Jasmin 正規化ラッパ */
static inline uint32_t get_jasmin(uint32_t c, int N){
  /* Jasmin 正規化のキャッシュ付ラッパ。盤面パック値 c を回転/ミラーで規約化した代表値を返す。
    ※ キャッシュは self.jasmin_cache[(c,N)] に保持。
    [Opt-08] キャッシュ付き jasmin() のラッパー  */
  if (g_jas.currN != N || !g_jas.cap) jas_cache_reset(N);

  uint32_t v;
  if (jas_cache_get(c, &v)) return v;  /* ← cache hit */

  v = jasmin(c, N);                     /* ← miss: 計算 */
  jas_cache_put(c, v);                  /* ← 追加 */
  return v;
}
typedef struct { int next_funcid; int funcptn; int availptn; } FuncMeta;

static int dfs(
  /*
  汎用 DFS カーネル。古い SQ???? 関数群を 1 本化し、func_meta の記述に従って
  (1) mark 行での step=2/3 + 追加ブロック、(2) jmark 特殊、(3) ゴール判定、(4) +1 先読み
  を切り替える。引数:
  functionid: 現在の分岐モード ID（次の ID, パターン, 先読み有無は func_meta で決定）
  ld/rd/col:   斜線/列の占有
  row/free:    現在行と空きビット
  jmark/endmark/mark1/mark2: 特殊行/探索終端
  board_mask:  盤面全域のビットマスク
  blockK_by_funcid/blockl_by_funcid: 関数 ID に紐づく追加ブロック
  func_meta:   (next_id, funcptn, availptn) のメタ情報配列
  */
  int functionid,
  uint32_t ld, uint32_t rd, uint32_t col,
  int row, uint32_t free_mask,
  int jmark, int endmark, int mark1, int mark2,
  uint32_t board_mask,
  const uint32_t* blockK_by_funcid, const uint32_t* blockL_by_funcid,
  const FuncMeta* meta,
  int N, int N1, uint32_t NK, uint32_t NJ
){
  (void)N; (void)NK; /* 本移植では未使用（codon 側と同様の分岐） */

  /* ---- ローカル束縛（属性アクセス最小化）---- */
  const int next_funcid = meta[functionid].next_funcid;
  const int funcptn     = meta[functionid].funcptn;
  const int avail_flag  = meta[functionid].availptn;

  uint32_t avail = free_mask;
  if (avail == 0u) return 0;

  int      total    = 0;
  /* ----N10:47 P6: 早期終了（基底）---- */
  if (funcptn == 5 && row == endmark){
    if (functionid == 14){ /* SQd2B 特例（列0以外が残っていれば1） */
      return ((avail >> 1) != 0u) ? 1 : 0;
    }
    return 1;
  }
  int      step     = 1;
  uint32_t add1     = 0;
  int      row_step = row + 1;
  bool     use_blocks = false;                 /* blockK/blockL を噛ませるか */
  bool     use_future = (avail_flag == 1);     /* _should_go_plus1 を使うか */

  uint32_t blockK = 0, blockL = 0;
  int      local_next_funcid = functionid;

  /* ---- N10:538 P1/P2/P3: mark 行での step=2/3 ＋ block 適用を共通ループへ設定---- */
  if (funcptn == 0 || funcptn == 1 || funcptn == 2){
    const bool at_mark = (funcptn == 1) ? (row == mark2) : (row == mark1);
    if (at_mark && avail){
      step     = (funcptn == 2) ? 3 : 2;
      add1     = (funcptn == 1 && functionid == 20) ? 1u : 0u; /* SQd1BlB のときだけ +1 */
      row_step = row + step;
      blockK   = blockK_by_funcid[functionid];
      blockL   = blockL_by_funcid[functionid];
      use_blocks        = true;
      use_future        = false;
      local_next_funcid = next_funcid;
    }
  }
  /* ---- N10:3 P4: jmark 特殊（入口一回だけ）---- */
  else if (funcptn == 3 && row == jmark){
    avail &= ~1u;     /* 列0禁止 */
    ld    |=  1u;     /* 左斜線 LSB を立てる */
    local_next_funcid = next_funcid;
    if (avail == 0u) return 0;
  }

  /* ==== N10:267 ループ１：block 適用（step=2/3 系のホットパス）==== */
  if (use_blocks){
    const int s  = step;
    const uint32_t a1 = add1;
    const uint32_t bK = blockK;
    const uint32_t bL = blockL;

    while (avail){
      uint32_t bit = avail & (~avail + 1u);
      avail &= (avail - 1u);

      uint32_t nld  = ((ld | bit) << s) | a1 | bL;
      uint32_t nrd  = ((rd | bit) >> s) | bK;
      uint32_t ncol = (col | bit);

      uint32_t nf = board_mask & ~(nld | nrd | ncol);
      if (nf){
        total += dfs(local_next_funcid, nld, nrd, ncol, row_step, nf,
                     jmark, endmark, mark1, mark2,
                     board_mask,
                     blockK_by_funcid, blockL_by_funcid, meta,
                     N, N1, NK, NJ);
      }
    }
    return total;
  }

  /* ==== N10:271 ループ２：+1 素朴（先読みなし）==== */
  else if (!use_future){
    while (avail){
      uint32_t bit = avail & (~avail + 1u);
      avail &= (avail - 1u);

      uint32_t nld  = (ld | bit) << 1;
      uint32_t nrd  = (rd | bit) >> 1;
      uint32_t ncol =  col | bit;

      uint32_t nf = board_mask & ~(nld | nrd | ncol);
      if (nf){
        total += dfs(local_next_funcid, nld, nrd, ncol, row_step, nf,
                     jmark, endmark, mark1, mark2,
                     board_mask,
                     blockK_by_funcid, blockL_by_funcid, meta,
                     N, N1, NK, NJ);
      }
    }
    return total;
  }

  /* ==== N10:92 ループ３：+1 先読み（row_step >= endmark は基底で十分）==== */
  else if (row_step >= endmark){
    //もう1手置いたらゴール層に達する → 普通の分岐で十分
    while (avail){
      uint32_t bit = avail & (~avail + 1u);
      avail &= (avail - 1u);

      uint32_t nld  = (ld | bit) << 1;
      uint32_t nrd  = (rd | bit) >> 1;
      uint32_t ncol =  col | bit;

      uint32_t nf = board_mask & ~(nld | nrd | ncol);
      if (nf){
        total += dfs(local_next_funcid, nld, nrd, ncol, row_step, nf,
                     jmark, endmark, mark1, mark2,
                     board_mask,
                     blockK_by_funcid, blockL_by_funcid, meta,
                     N, N1, NK, NJ);
      }
    }
    return total;
  }

  /* ==== N10:402 ループ３B：+1 先読み本体（1手先の空きがゼロなら枝刈り）==== */
  while (avail){
    uint32_t bit = avail & (~avail + 1u);
    avail &= (avail - 1u);

    uint32_t nld  = (ld | bit) << 1;
    uint32_t nrd  = (rd | bit) >> 1;
    uint32_t ncol =  col | bit;

    uint32_t nf = board_mask & ~(nld | nrd | ncol);
    if (!nf) continue;

    /* 1手先の空きをその場で素早くチェック（余計な再帰を抑止） */
    /*   next_free_next = board_mask & ~(((nld << 1) | (nrd >> 1) | ncol)); */
    /*   if (next_free_next == 0) continue; */
    if (board_mask & ~((nld << 1) | (nrd >> 1) | ncol)){
      total += dfs(local_next_funcid, nld, nrd, ncol, row_step, nf,
                   jmark, endmark, mark1, mark2,
                   board_mask,
                   blockK_by_funcid, blockL_by_funcid, meta,
                   N, N1, NK, NJ);
    }
  }
  return total;
}

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
 static void exec_solutions(Constellations* constellations, int N){
  /* 各 Constellation（部分盤面）ごとに最適分岐（functionid）を選び、`dfs()` で解数を取得。 結果は `solutions` に書き込み、最後に `symmetry()` の重みで補正する。前段で SoA 展開し 並列化区間のループ体を軽量化。 */
  const int N2 = N - 2;
  const uint32_t small_mask = (N2 >= 0) ? ((1u << (uint32_t)N2) - 1u) : 0u;
  const uint32_t board_mask = (N  >  0) ? ((1u << (uint32_t)N ) - 1u) : 0u;

  /* ===== FUNC_CATEGORY: fid → {3:N-3, 4:N-4, 0:その他} ===== */
  static const unsigned char FUNC_CATEGORY[28] = {
    3,0,0,0, 4,0,3,3, 0,0,0,0, 3,0,0,4, 0,3,3,3, 0,0,4,0, 3,3,0,3
  };
  /* ===== FID（Codonの並び） ===== */
  enum {
    FID_SQBkBlBjrB=0, FID_SQBlBjrB=1, FID_SQBjrB=2, FID_SQB=3,
    FID_SQBklBjrB=4, FID_SQBlBkBjrB=5, FID_SQBkBjrB=6, FID_SQBlkBjrB=7,
    FID_SQBjlBkBlBjrB=8, FID_SQBjlBklBjrB=9, FID_SQBjlBlBkBjrB=10, FID_SQBjlBlkBjrB=11,
    FID_SQd2BkBlB=12, FID_SQd2BlB=13, FID_SQd2B=14, FID_SQd2BklB=15, FID_SQd2BlBkB=16,
    FID_SQd2BkB=17, FID_SQd2BlkB=18, FID_SQd1BkBlB=19, FID_SQd1BlB=20, FID_SQd1B=21,
    FID_SQd1BklB=22, FID_SQd1BlBkB=23, FID_SQd1BlkB=24, FID_SQd1BkB=25, FID_SQd0B=26, FID_SQd0BkB=27
  };

  /* next_funcid, funcptn, availptn の3つだけ持つ */
  typedef struct { int next_funcid; int funcptn; int availptn; } FuncMeta;
  static const FuncMeta func_meta[28] = {
    {1,0,0},//0 SQBkBlBjrB   -> P1, 先読みなし
    {2,1,0},//1 SQBlBjrB     -> P2, 先読みなし  
    {3,3,1},//2 SQBjrB       -> P4, 先読みあり  
    {3,5,1},//3 SQB          -> P6, 先読みあり
    {2,2,0},//4 SQBklBjrB    -> P3, 先読みなし  
    {6,0,0},//5 SQBlBkBjrB   -> P1, 先読みなし  
    {2,1,0},//6 SQBkBjrB     -> P2, 先読みなし  
    {2,2,0},//7 SQBlkBjrB    -> P3, 先読みなし
    {0,4,1},//8 SQBjlBkBlBjrB-> P5, 先読みあり  
    {4,4,1},//9 SQBjlBklBjrB -> P5, 先読みあり  
    {5,4,1},//10 SQBjlBlBkBjrB-> P5, 先読みあり  
    {7,4,1},//11 SQBjlBlkBjrB -> P5, 先読みあり
    {13,0,0},//12 SQd2BkBlB    -> P1, 先読みなし 
    {14,1,0},//13 SQd2BlB      -> P2, 先読みなし 
    {14,5,1},//14 SQd2B        -> P6, 先読みあり（avail 特例） 
    {14,2,0},//15 SQd2BklB     -> P3, 先読みなし
    {17,0,0},//16 SQd2BlBkB    -> P1, 先読みなし 
    {14,1,0},//17 SQd2BkB      -> P2, 先読みなし 
    {14,2,0},//18 SQd2BlkB     -> P3, 先読みなし 
    {20,0,0},//19 SQd1BkBlB    -> P1, 先読みなし
    {21,1,0},//20 SQd1BlB      -> P2, 先読みなし（add1=1 は dfs 内で特別扱い） 
    {21,5,1},//21 SQd1B        -> P6, 先読みあり 
    {21,2,0},//22 SQd1BklB     -> P3, 先読みなし 
    {25,0,0},//23 SQd1BlBkB    -> P1, 先読みなし
    {21,2,0},//24 SQd1BlkB     -> P3, 先読みなし 
    {21,1,0},//25 SQd1BkB      -> P2, 先読みなし 
    {26,5,1},//26 SQd0B        -> P6, 先読みあり 
    {26,0,0},//27 SQd0BkB      -> P1, 先読みなし
  };
  /* ===== block 配列 ===== */
  const uint32_t n3 = (N >= 3) ? (1u << (uint32_t)(N-3)) : 1u;//# 念のため負シフト防止
  const uint32_t n4 = (N >= 4) ? (1u << (uint32_t)(N-4)) : 1u;
  uint32_t blockK_by_funcid[28] = {0};
  uint32_t blockl_by_funcid[28] =
    {0,1,0,0,1,1,0,2,0,0,0,0,0,1,0,1,1,0,2,0,0,0,1,1,2,0,0,0};
  for (int fid = 0; fid < 28; ++fid) {
    unsigned char cat = FUNC_CATEGORY[fid];
    blockK_by_funcid[fid] = (cat == 3) ? n3 : (cat == 4 ? n4 : 0u);
  }
  /* ===== 前処理ステージ（単一スレッド） ===== */
  const size_t m = consts_size(constellations);
  //if (m == 0) return;
  /*  SoA（Structure of Arrays）に展開：並列本体が軽くなる */
  uint32_t *ld_arr   = (uint32_t*)malloc(sizeof(uint32_t)*m);
  uint32_t *rd_arr   = (uint32_t*)malloc(sizeof(uint32_t)*m);
  uint32_t *col_arr  = (uint32_t*)malloc(sizeof(uint32_t)*m);
  int      *row_arr  = (int*)     malloc(sizeof(int)*m);
  uint32_t *free_arr = (uint32_t*)malloc(sizeof(uint32_t)*m);
  int *jmark_arr = (int*)malloc(sizeof(int)*m);
  int *end_arr   = (int*)malloc(sizeof(int)*m);
  int *mark1_arr = (int*)malloc(sizeof(int)*m);
  int *mark2_arr = (int*)malloc(sizeof(int)*m);
  int      *funcid_arr = (int*)     malloc(sizeof(int)*m);
  uint32_t *ijkl_arr   = (uint32_t*)malloc(sizeof(uint32_t)*m);
  const int N1 = N - 1;
  const uint32_t NK = (N >= 3) ? (1u << (uint32_t)(N-3)) : 0u;
  const uint32_t NJ = (N1 >= 0) ? (1u << (uint32_t)N1) : 0u;
  int      *results    = (int*)     malloc(sizeof(int)*m);

  /* メモリ確保失敗の簡易チェック（本番なら全チェック推奨） */
  if (!ld_arr||!rd_arr||!col_arr||!row_arr||!free_arr||
      !jmark_arr||!end_arr||!mark1_arr||!mark2_arr||!funcid_arr||!ijkl_arr||!results){
    /* 後始末して戻る */
    free(ld_arr); free(rd_arr); free(col_arr); free(row_arr); free(free_arr);
    free(jmark_arr); free(end_arr); free(mark1_arr); free(mark2_arr);
    free(funcid_arr); free(ijkl_arr); free(results);
    return;
  }

  /* ====== (1) 前処理ステージ：配列へ格納 ======
     Codon の if 分岐で target/jmark/endmark/mark1/mark2 を決定し、
     ld/rd/col/row/free もここで確定させて SoA に詰める。 */
  for (size_t i = 0; i < m; ++i){
    const Constellation* c = consts_at(constellations, i);
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
    uint32_t free = ~(ld | rd | col);

    //よく使う比較はフラグ化（1回だけ計算）
    const bool j_lt_N3    = (j <  (N - 3));
    const bool j_eq_N3    = (j == (N - 3));
    const bool j_eq_N2    = (j == N2);
    const bool k_lt_l     = (k <  l);
    const bool start_lt_k = (start < k);
    const bool start_lt_l = (start < l);
    const bool l_eq_kp1   = (l == k + 1);
    const bool k_eq_lp1   = (k == l + 1);
    const bool j_gate     = (j > 2 * N - 34 - start);//既存の「ゲート」条件


    /* --- ここから分岐（大分類 → 小分類）。同じ式の再評価をなくし、ネストを浅く。 --- */
    if (j_lt_N3) {
  jmark = j + 1;
  endmark = N2;
  if (j_gate) {
    if (k_lt_l) {
      mark1 = k - 1; mark2 = l - 1;
      if (start_lt_l) {
        target = start_lt_k ? (l_eq_kp1 ? 4 : 0) : 1;   /* SQBklBjrB / SQBkBlBjrB / SQBlBjrB */
      } else {
        target = 2;                                      /* SQBjrB */
      }
    } else {
      mark1 = l - 1; mark2 = k - 1;
      if (start_lt_k) {
        target = start_lt_l ? (k_eq_lp1 ? 7 : 5) : 6;   /* SQBlkBjrB / SQBlBkBjrB / SQBkBjrB */
      } else {
        target = 2;                                      /* SQBjrB */
      }
    }
  } else {
    if (k_lt_l) {
      mark1 = k - 1; mark2 = l - 1;
      target = l_eq_kp1 ? 9 : 8;                         /* SQBjlBklBjrB / SQBjlBkBlBjrB */
    } else {
      mark1 = l - 1; mark2 = k - 1;
      target = k_eq_lp1 ? 11 : 10;                       /* SQBjlBlkBjrB / SQBjlBlBkBjrB */
    }
  }
} else if (j_eq_N3) {
  endmark = N2;
  if (k_lt_l) {
    mark1 = k - 1; mark2 = l - 1;
    if (start_lt_l) {
      if (start_lt_k) {
        target = l_eq_kp1 ? 15 : 12;                     /* SQd2BklB / SQd2BkBlB */
      } else {
        mark2 = l - 1;
        target = 13;                                     /* SQd2BlB */
      }
    } else {
      target = 14;                                       /* SQd2B */
    }
  } else {
    mark1 = l - 1; mark2 = k - 1;
    if (start_lt_k) {
      if (start_lt_l) {
        target = k_eq_lp1 ? 18 : 16;                     /* SQd2BlkB / SQd2BlBkB */
      } else {
        mark2 = k - 1;
        target = 17;                                     /* SQd2BkB */
      }
    } else {
      target = 14;                                       /* SQd2B */
    }
  }
} else if (j_eq_N2) { /* j がコーナーから1列内側 */
  if (k_lt_l) {
    endmark = N2;
    if (start_lt_l) {
      if (start_lt_k) {
        mark1 = k - 1;
        if (!l_eq_kp1) { mark2 = l - 1; target = 19; }   /* SQd1BkBlB */
        else            {               target = 22; }   /* SQd1BklB */
      } else {
        mark2 = l - 1;
        target = 20;                                     /* SQd1BlB */
      }
    } else {
      target = 21;                                       /* SQd1B */
    }
  } else { /* l < k */
    if (start_lt_k) {
      if (start_lt_l) {
        if (k < N2) {
          mark1 = l - 1; endmark = N2;
          if (!k_eq_lp1) { mark2 = k - 1; target = 23; } /* SQd1BlBkB */
          else            {               target = 24; } /* SQd1BlkB */
        } else {
          if (l != (N - 3)) {
            mark2 = l - 1; endmark = N - 3; target = 20; /* SQd1BlB */
          } else {
            endmark = N - 4; target = 21;                /* SQd1B */
          }
        }
      } else {
        if (k != N2) { mark2 = k - 1; endmark = N2; target = 25; }  /* SQd1BkB */
        else          { endmark = N - 3; target = 21; }             /* SQd1B */
      }
    } else {
      endmark = N2; target = 21;                                     /* SQd1B */
    }
  }
} else { /* j がコーナー */
  endmark = N2;
  if (start > k) {
    target = 26;                                                     /* SQd0B */
  } else {
    mark1 = k - 1; target = 27;                                      /* SQd0BkB */
  }
}
    
    /* ---- 配列へ格納（★ここがCUDA入力バッファになる） ---- */
    ld_arr[i]   = ld;
    rd_arr[i]   = rd;
    col_arr[i]  = col;
    row_arr[i]  = start;
    free_arr[i] = free;
    jmark_arr[i] = jmark;
    end_arr[i]   = endmark;
    mark1_arr[i] = mark1;
    mark2_arr[i] = mark2;
    funcid_arr[i] = target;
    ijkl_arr[i]   = ijkl;   /* symmetry 計算用 */
  }

  /* ====== (2) 並列ステージ：計算だけ ======
     ここは後で CUDA の kernel <<<grid,block>>> に置き換えやすい形。 */
  for (size_t i = 0; i < m; ++i){
    int cnt = dfs(
      funcid_arr[i],
      ld_arr[i], rd_arr[i], col_arr[i],
      row_arr[i], free_arr[i],
      jmark_arr[i], end_arr[i], mark1_arr[i], mark2_arr[i],
      board_mask,
      blockK_by_funcid, blockl_by_funcid, func_meta,
      N, N1, NK, NJ
    );
    results[i] = cnt * symmetry(ijkl_arr[i], N);
  }

  /* ====== (3) 書き戻し（単一スレッド） ====== */
  for (size_t i = 0; i < m; ++i){
    Constellation* c = (Constellation*)consts_at(constellations, i);
    c->solutions = (uint32_t)results[i];
  }

  /* 後始末 */
  free(ld_arr); free(rd_arr); free(col_arr); free(row_arr); free(free_arr);
  free(jmark_arr); free(end_arr); free(mark1_arr); free(mark2_arr);
  free(funcid_arr); free(ijkl_arr); free(results);
}


/* ==================== set_pre_queens / cached (C移植) ==================== */
typedef struct {
  uint64_t* keys;
  unsigned char* used;
  size_t cap, sz;
} SigSet;


static void sigset_free(SigSet* s){
  if (!s) return;
  free(s->keys);
  free(s->used);
  s->keys = NULL;
  s->used = NULL;
  s->cap = s->sz = 0;
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

static SigSet g_subconst = {0};
static int g_subconst_N = -1;
static int g_subconst_preset = -1;

static void subconst_cache_reset(int N, int preset_queens){
  /* N or preset が変わったら完全リセット、同じなら内容だけクリア */
  if (g_subconst.cap == 0 || g_subconst_N != N || g_subconst_preset != preset_queens){
    free(g_subconst.keys); free(g_subconst.used);
    memset(&g_subconst, 0, sizeof(g_subconst));
    g_subconst_N = N;
    g_subconst_preset = preset_queens;
  } else {
    /* 既存テーブルを空に */
    memset(g_subconst.used, 0, g_subconst.cap);
    g_subconst.sz = 0;
  }
}


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
  //サブコンステレーション生成のキャッシュ付ラッパ。StateKey で一意化し、 同一状態での重複再帰を回避して生成量を抑制する。
  uint64_t ld, uint64_t rd, uint64_t col,
  int k, int l, int row, int queens,
  uint64_t LD, uint64_t RD,
  int* counter,
  Constellations* constellations,
  int N, int preset_queens,
  UIntSet* visited,
  SigSet* constellation_signatures
){
  /* --- 共有キャッシュ（再入/重複抑止） --- */
  uint64_t key = StateKey(ld, rd, col, k, l, row, queens, LD, RD, N, preset_queens);
  /* SigSet を使って存在チェック＋追加（あれば重複なので即 return） */
  if (!sigset_add_if_absent(&g_subconst, key)){
    return;
  }
  set_pre_queens(ld, rd, col, k, l, row, queens, LD, RD,
                 counter, constellations, N, preset_queens,
                 visited, constellation_signatures);
}

/* 新規実行（従来通りset_pre_queensの本体処理へ） */
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

  /* --- state hash (軽量O(1)) による枝刈り ---
  その場で数個の ^ と << を混ぜるだけの O(1) 計算。
  生成されるキーも 単一の int なので、set/dict の操作が最速＆省メモリ。
  ただし理論上は衝突し得ます（実際はN≤17の範囲なら実害が出にくい設計にしていればOK）。
  [Opt-09] Zobrist Hash（Opt-09）の導入とその用途
  ビットボード設計でも、「盤面のハッシュ」→「探索済みフラグ」で枝刈りは可能です
   */
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
  /* クイーンの数がpreset_queensに達した場合、現在の状態を保存  */
  if (queens == preset_queens) {
    if (constellation_signatures && constellation_signatures->cap == 0)
      sigset_init(constellation_signatures);
    // signatureの生成
    uint64_t signature = sig_key(ld, rd, col, k, l, row);//必要な変数でOK
    bool fresh = true;
    if (constellation_signatures)
      fresh = sigset_add_if_absent(constellation_signatures, signature);

    if (fresh) {
      Constellation c;
      c.ld  = (uint32_t)ld;   /* dump用途は32bitで十分 */
      c.rd  = (uint32_t)rd;
      c.col = (uint32_t)col;
      c.startijkl = (uint32_t)(row << 20); /* Python: row<<20 を踏襲 */
      c.solutions = 0;
      consts_push(constellations, c);//星座データ追加
      if (counter) (*counter)++;
    }
    return;
  }

  /* --- 現在の行にクイーンを配置できる位置を計算 --- */
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
  /*
  開始コンステレーション（代表部分盤面）の列挙。中央列（奇数 N）特例、回転重複排除 （`check_rotations`）、Jasmin 正規化（`get_jasmin`）を経て、各 sc から `set_pre_queens_cached` でサブ構成を作る。
  実行ごとにメモ化をリセット（N や preset_queens が変わるとキーも変わるが、
  長時間プロセスなら念のためクリアしておくのが安全）
  */
  (void)preset_queens; /* 本体側で使用 */
  const int halfN=(N+1)/2, N1=N-1;//Nの半分を切り上げ

  //--- [Opt-03] 中央列特別処理（奇数Nの場合のみ） ---
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

  //--- [Opt-03] 中央列特別処理（奇数Nの場合のみ） ---
  //コーナーにクイーンがいない場合の開始コンステレーションを計算する
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

  //コーナーにクイーンがある場合の開始コンステレーションを計算する
  for (int j=1; j<N-2; ++j)
    for (int l=j+1; l<N1; ++l)
      uintset_add(ijkl_list, to_ijkl(0,j,0,l));

  /* Jasmin変換 */
  UIntSet* jas = uintset_create();
  { UIntSetIter it={0}; uint32_t v;
    while (uintset_iter_next(ijkl_list,&it,&v)) uintset_add(jas, get_jasmin(v,N));
  }
  //ここで毎回クリア（＝この sc だけの重複抑止に限定）
  uintset_clear(ijkl_list);
  { UIntSetIter it={0}; uint32_t v;
    while (uintset_iter_next(jas,&it,&v)) uintset_add(ijkl_list, v);
  }
  uintset_free(jas);

  /* 各 canonical start からサブコンステレーション生成 */
  subconst_cache_reset(N, preset_queens);   /* ★ 追加：共有キャッシュ初期化（N/preset単位） */
  /* 各 canonical start からサブコンステレーション生成 */
  const uint32_t L = (uint32_t)1u << N1;// Lは左端に1を立てる

  { UIntSetIter it={0}; uint32_t sc;
    while (uintset_iter_next(ijkl_list,&it,&sc)) {
      SigSet sigs = {0}; 

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
      sigset_free(&sigs);
      uintset_free(visited);
    }
  }
}


/* startijkl の 20bit 下位は ijkl、上位は start(row) */
static bool validate_ijkl_row(uint32_t startijkl, int N){
  uint32_t ijkl = startijkl & ((1u<<20)-1u);
  int i=geti(ijkl), j=getj(ijkl), k=getk(ijkl), l=getl(ijkl);
  int start = (int)(startijkl >> 20);
  if (N<=0) return false;
  if (i<0||j<0||k<0||l<0||start<0) return false;
  if (i>=N||j>=N||k>=N||l>=N||start>=N) return false;
  return true;
}

/* 盤面幅Nに対して ld/rd/col が N ビット内に収まっているかを大まかにチェック */
static bool validate_masks(uint32_t ld, uint32_t rd, uint32_t col, int N){
  if (N<=0 || N>=32) return false;
  uint32_t bm = (N==32) ? 0xFFFFFFFFu : ((1u<<(uint32_t)N) - 1u);
  return ( (ld & ~bm)==0 && (rd & ~bm)==0 && (col & ~bm)==0 );
}

/* constellations の各要素が {ld, rd, col, startijkl} を全て持つかを検証する。 */
static bool validate_constellation_list(const Constellations* v, int N){
  if (!v || v->n==0) return false;
  for (size_t i=0;i<v->n;++i){
    const Constellation* c = &v->a[i];
    if (!validate_masks(c->ld, c->rd, c->col, N)) return false;
    if (!validate_ijkl_row(c->startijkl, N)) return false;
  }
  return true;
}

//32bit little-endian の相互変換ヘルパ。Codon/CPython の差異に注意。
static uint32_t read_u32_le(const unsigned char* p){
  return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}
static void int_to_le_bytes(FILE* f, uint32_t v){
  unsigned char b[4];
  b[0] = (unsigned char)(v & 0xFFu);
  b[1] = (unsigned char)((v>>8) & 0xFFu);
  b[2] = (unsigned char)((v>>16)& 0xFFu);
  b[3] = (unsigned char)((v>>24)& 0xFFu);
  fwrite(b,1,4,f);
}

static bool file_exists(const char* path){
//ファイル存在チェック（読み取り open の可否で判定）。  
  struct stat st; return stat(path, &st) == 0;
}

static bool validate_bin_file(const char* path){
//bin キャッシュのサイズ妥当性確認（1 レコード 16 バイトの整数倍か）。  
  struct stat st;
  if (stat(path, &st) != 0) return false;
  /* 16 バイト（ld,rd,col,startijkl）の倍数であること */
  return (st.st_size > 0) && ((st.st_size % 16) == 0);
}
//テキスト形式で constellations を保存/復元する（1 行 5 数値: ld rd col startijkl solutions）。
static bool save_constellations_txt(const char* path, const Constellations* v){
  if (!path || !v) return false;
  FILE* f = fopen(path, "w");
  if (!f) return false;
  for (size_t i=0;i<v->n;++i){
    const Constellation* c = &v->a[i];
    /* solutions は現在の構造体の値をそのまま保存 */
    if (fprintf(f, "%u %u %u %u %u\n",
                c->ld, c->rd, c->col, c->startijkl, c->solutions) < 0){
      fclose(f); return false;
    }
  }
  fclose(f);
  return true;
}
//テキスト形式で constellations を保存/復元する（1 行 5 数値: ld rd col startijkl solutions）。
static bool load_constellations_txt(const char* path, Constellations* out){
  if (!path || !out) return false;
  FILE* f = fopen(path, "r");
  if (!f) return false;
  /* 既存内容はクリアする（必要なら別APIに） */
  out->n = 0;
  uint32_t ld, rd, col, startijkl, solutions;
  while (true){
    int n = fscanf(f, "%u %u %u %u %u", &ld,&rd,&col,&startijkl,&solutions);
    if (n == EOF || n == 0) break;
    if (n != 5){
      /* 行スキップ：残り捨て */
      int ch; while ((ch=fgetc(f))!='\n' && ch!=EOF){}
      continue;
    }
    Constellation c = { ld, rd, col, startijkl, solutions };
    consts_push(out, c);
  }
  fclose(f);
  return true;
}

/*テキストキャッシュを読み込み。壊れていれば `gen_constellations()` で再生成して保存する。*/
static bool load_or_build_constellations_txt(
  UIntSet* ijkl_list, Constellations* constellations,
  int N, int preset_queens)
{
  char fname[128];
  snprintf(fname, sizeof(fname), "constellations_N%d_%d.txt", N, preset_queens);

  if (file_exists(fname)){
    Constellations tmp = {0};
    tmp.cap = 128;
    tmp.a = (Constellation*)calloc(tmp.cap, sizeof(Constellation));
    if (!tmp.a) return false;

    bool ok = load_constellations_txt(fname, &tmp)
           && validate_constellation_list(&tmp, N);
    if (ok){
      /* swap into out */
      free(constellations->a);
      *constellations = tmp;
      return true;
    }
    /* 破損：再生成へ */
    free(tmp.a);
  }

  /* 生成 */
  constellations->n = 0;
  gen_constellations(ijkl_list, constellations, N, preset_queens);
  /* 保存（失敗しても致命ではないため戻り値は見ない） */
  save_constellations_txt(fname, constellations);
  return true;
}

/*bin 形式で constellations を保存/復元。Codon では str をバイト列として扱う前提のため、CPython では bytes で書き込むよう分岐/注意が必要。*/
static bool save_constellations_bin(const char* path, const Constellations* v){
  if (!path || !v) return false;
  FILE* f = fopen(path, "wb");
  if (!f) return false;
  for (size_t i=0;i<v->n;++i){
    const Constellation* c = &v->a[i];
    int_to_le_bytes(f, c->ld);
    int_to_le_bytes(f, c->rd);
    int_to_le_bytes(f, c->col);
    int_to_le_bytes(f, c->startijkl);
    /* solutions はバイナリには含めない（Python版踏襲） */
  }
  fclose(f);
  return true;
}

//bin 形式で constellations を保存/復元。Codon では str をバイト列として扱う前提のため、CPython では bytes で書き込むよう分岐/注意が必要。
static bool load_constellations_bin(const char* path, Constellations* out){
  if (!path || !out) return false;
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  out->n = 0;
  unsigned char buf[16];
  while (true){
    size_t n = fread(buf,1,16,f);
    if (n==0) break;
    if (n<16){ /* 端数は破損扱い */
      fclose(f); return false;
    }
    uint32_t ld   = read_u32_le(buf+0);
    uint32_t rd   = read_u32_le(buf+4);
    uint32_t col  = read_u32_le(buf+8);
    uint32_t stkl = read_u32_le(buf+12);
    Constellation c = { ld, rd, col, stkl, 0u };
    consts_push(out, c);
  }
  fclose(f);
  return true;
}

/*bin キャッシュを読み込み。検証に失敗した場合は再生成して保存し、その結果を返す。*/
static bool load_or_build_constellations_bin(
  UIntSet* ijkl_list, Constellations* constellations,
  int N, int preset_queens)
{
  char fname[128];
  snprintf(fname, sizeof(fname), "constellations_N%d_%d.bin", N, preset_queens);

  if (file_exists(fname) && validate_bin_file(fname)){
    Constellations tmp = {0};
    tmp.cap = 128;
    tmp.a = (Constellation*)calloc(tmp.cap, sizeof(Constellation));
    if (!tmp.a) return false;

    bool ok = load_constellations_bin(fname, &tmp)
           && validate_constellation_list(&tmp, N);
    if (ok){
      free(constellations->a);
      *constellations = tmp;
      return true;
    }
    free(tmp.a);
  }

  /* 生成 */
  constellations->n = 0;
  gen_constellations(ijkl_list, constellations, N, preset_queens);
  save_constellations_bin(fname, constellations);
  return true;
}




/*==================== N<=5 のフォールバック（スタブ） ====================*/
static int bit_total(int N){ return (N==5)?10:0; }

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


int main(void) {
  /*
  N=5..17 の合計解を計測。N<=5 は `_bit_total()` のフォールバック、それ以外は星座キャッシュ（.bin/.txt）→ `exec_solutions()` → 合計→既知解 `expected` と照合。
  */
  const int nmin = 5;
  const int nmax = 18;
  const int preset_queens = 4;

  const int expected[] = {0,0,0,0,0, 10,4,40,92,352,724,2680,14200,73712,365596,2279184,14772512,95815104};
  const int expected_len = (int)(sizeof(expected) / sizeof(expected[0]));

  printf(" N:        Total       Unique        hh:mm:ss.ms\n");

  for (int size = nmin; size < nmax; ++size) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    if (size <= 5) {//フォールバック：N=5はここで正しい10を得る
      int total = bit_total(size);
      clock_gettime(CLOCK_MONOTONIC, &t1);
      char buf[64]; format_duration(buf, sizeof(buf), t0, t1);
      printf("%2d:%13d%13d%20s\n", size, total, 0, buf);
      continue;
    }

    UIntSet* ijkl_list = uintset_create();
    Constellations* constellations = consts_create();

    //キャッシュを使わない
    gen_constellations(ijkl_list, constellations, size, preset_queens);
    //キャッシュを使う、キャッシュの整合性もチェック
    //load_or_build_constellations_bin(ijkl_list, constellations, size, preset_queens);
    //load_or_build_constellations_txt(ijkl_list, constellations, size, preset_queens);


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
