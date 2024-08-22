#include <stdio.h>
#include <stdlib.h>
// #include <math.h>
#include <stdint.h>
//#include <time.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// #define fmin(a,b) (((a) < (b)) ? (a) : (b))

// Constellation構造体の定義
typedef struct {
    int id;
    int ld;
    int rd;
    int col;
    int startijkl;
    long solutions;
} Constellation;

// IntHashSet構造体の定義
typedef struct {
    int* data;
    size_t size;
    size_t capacity;
} IntHashSet;

// ConstellationArrayList構造体の定義
typedef struct {
    Constellation* data;
    size_t size;
    size_t capacity;
} ConstellationArrayList;




#define INITIAL_CAPACITY 1000
#define PRESETQUEENS 4
// IntHashSetの関数プロトタイプ
//IntHashSet* create_int_hashset();
void free_int_hashset(IntHashSet* set);
int int_hashset_contains(IntHashSet* set, int value);
void int_hashset_add(IntHashSet* set, int value);

// ビット操作関数プロトタイプ
int checkRotations(IntHashSet* set, int i, int j, int k, int l,int N);
int toijkl(int i, int j, int k, int l);
int geti(int sc);
int getj(int sc);
int getk(int sc);
int getl(int sc);
int jasmin(int ijkl,int N);
int mirvert(int ijkl,int N);
int rot90(int ijkl,int N);
int symmetry90(int ijkl,int N);
int symmetry(int ijkl,int N);
void add_constellation(int ld, int rd, int col, int startijkl,ConstellationArrayList* constellations);

// 関数プロトタイプ宣言
void SQBkBlBjrB(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQBklBjrB(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQBlBjrB(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQBjrB(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQB(int ld, int rd, int col, int raw, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQBlBkBjrB(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQBlkBjrB(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQBkBjrB(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQBjlBkBlBjrB(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQBjlBklBjrB(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQBjlBlBkBjrB(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQBjlBlkBjrB(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQd2BkBlB(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQd2BklB(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQd2BlB(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQd2B(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQd2BlBkB(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQd2BlkB(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQd1BkBlB(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQd1BklB(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQd1BlB(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQd1B(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQd1BlBkB(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQd1BlkB(int ld, int rd, int col, int start, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQd0B(int ld, int rd, int col, int row, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQd0BkB(int ld, int rd, int col, int row, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQd2BkB(int ld, int rd, int col, int row, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);
void SQd1BkB(int ld, int rd, int col, int row, int free, int jmark, int endmark, int mark1, int mark2,long* tempcounter,int N,int N3,int N4,int L,int L3,int L4);

// その他の関数プロトタイプ
void setPreQueens(int ld, int rd, int col, int k, int l, int row, int queens, int LD, int RD, int *counter,ConstellationArrayList* constellations,int N);
void execSolutions(ConstellationArrayList* constellations,int N);
void genConstellations(IntHashSet* ijklList,ConstellationArrayList* constellations,int N);

// 関数プロトタイプ宣言

long calcSolutions(ConstellationArrayList* constellations,long solutions);

// 関数の実装

// IntHashSet の関数実装
IntHashSet* create_int_hashset() {
    IntHashSet* set = (IntHashSet*)malloc(sizeof(IntHashSet));
    set->data = (int*)malloc(INITIAL_CAPACITY * sizeof(int));
    set->size = 0;
    set->capacity = INITIAL_CAPACITY;
    return set;
}

void free_int_hashset(IntHashSet* set) {
    free(set->data);
    free(set);
}

int int_hashset_contains(IntHashSet* set, int value) {
    for (size_t i = 0; i < set->size; i++) {
        if (set->data[i] == value) {
            return 1;
        }
    }
    return 0;
}

void int_hashset_add(IntHashSet* set, int value) {
    if (!int_hashset_contains(set, value)) {
        if (set->size == set->capacity) {
            set->capacity *= 2;
            set->data = (int*)realloc(set->data, set->capacity * sizeof(int));
        }
        set->data[set->size++] = value;
    }
}

// ConstellationArrayList の関数実装
ConstellationArrayList* create_constellation_arraylist() {
    ConstellationArrayList* list = (ConstellationArrayList*)malloc(sizeof(ConstellationArrayList));
    list->data = (Constellation*)malloc(INITIAL_CAPACITY * sizeof(Constellation));
    list->size = 0;
    list->capacity = INITIAL_CAPACITY;
    return list;
}

void free_constellation_arraylist(ConstellationArrayList* list) {
    free(list->data);
    free(list);
}

void constellation_arraylist_add(ConstellationArrayList* list, Constellation value) {
    if (list->size == list->capacity) {
        list->capacity *= 2;
        list->data = (Constellation*)realloc(list->data, list->capacity * sizeof(Constellation));
    }
    list->data[list->size++] = value;
}

Constellation* create_constellation() {
    Constellation* new_constellation = (Constellation*)malloc(sizeof(Constellation));
    if (new_constellation) {
        new_constellation->id = 0;
        new_constellation->ld = 0;
        new_constellation->rd = 0;
        new_constellation->col = 0;
        new_constellation->startijkl = 0;
        new_constellation->solutions = 0;
    }
    return new_constellation;
}

// Constellation* create_constellation_with_values(int id, int ld, int rd, int col, int startijkl,ConstellationArrayList* constellations, long solutions) {
Constellation* create_constellation_with_values(int id, int ld, int rd, int col, int startijkl, long solutions) {
    Constellation* new_constellation = (Constellation*)malloc(sizeof(Constellation));
    if (new_constellation) {
        new_constellation->id = id;
        new_constellation->ld = ld;
        new_constellation->rd = rd;
        new_constellation->col = col;
        new_constellation->startijkl = startijkl;
        new_constellation->solutions = solutions;
    }
    return new_constellation;
}


long calcSolutions(ConstellationArrayList* constellations,long solutions) {
    for (size_t i = 0; i < constellations->size; i++) {
        Constellation* c = &constellations->data[i];
        if (c->solutions >= 0) {
            solutions += c->solutions;
        }
    }
    return solutions;
}


int get_id(Constellation* constellation) {
    return constellation->id;
}

void set_id(Constellation* constellation, int id) {
    constellation->id = id;
}

int get_ld(Constellation* constellation) {
    return constellation->ld;
}

void set_ld(Constellation* constellation, int ld) {
    constellation->ld = ld;
}

int get_rd(Constellation* constellation) {
    return constellation->rd;
}

void set_rd(Constellation* constellation, int rd) {
    constellation->rd = rd;
}

int get_col(Constellation* constellation) {
    return constellation->col;
}

void set_col(Constellation* constellation, int col) {
    constellation->col = col;
}

int get_startijkl(Constellation* constellation) {
    return constellation->startijkl;
}

void set_startijkl(Constellation* constellation, int startijkl) {
    constellation->startijkl = startijkl;
}

long get_solutions(Constellation* constellation) {
    return constellation->solutions;
}

void set_solutions(Constellation* constellation, long solutions) {
    constellation->solutions = solutions;
}

int get_ijkl(Constellation* constellation) {
    return constellation->startijkl & 0xFFFFF; // Equivalent to 0b11111111111111111111
}

int checkRotations(IntHashSet* ijklList, int i, int j, int k, int l,int N) {
    int rot90 = ((N-1-k) << 15) + ((N-1-l) << 10) + (j << 5) + i;
    int rot180 = ((N-1-j) << 15) + ((N-1-i) << 10) + ((N-1-l) << 5) + (N-1-k);
    int rot270 = (l << 15) + (k << 10) + ((N-1-i) << 5) + (N-1-j);

    if (int_hashset_contains(ijklList, rot90)) {
        return 1; // true
    }
    if (int_hashset_contains(ijklList, rot180)) {
        return 1; // true
    }
    if (int_hashset_contains(ijklList, rot270)) {
        return 1; // true
    }
    return 0; // false
}

int toijkl(int i, int j, int k, int l) {
    return (i << 15) + (j << 10) + (k << 5) + l;
}

int geti(int ijkl) {
    return ijkl >> 15;
}

int getj(int ijkl) {
    return (ijkl >> 10) & 31;
}

int getk(int ijkl) {
    return (ijkl >> 5) & 31;
}

int getl(int ijkl) {
    return ijkl & 31;
}

int mirvert(int ijkl,int N) {
    return toijkl(N - 1 - geti(ijkl), N - 1 - getj(ijkl), getl(ijkl), getk(ijkl));
}

int rot90(int ijkl,int N) {
    return ((N - 1 - getk(ijkl)) << 15) + ((N - 1 - getl(ijkl)) << 10) + (getj(ijkl) << 5) + geti(ijkl);
}

int jasmin(int ijkl,int N) {
    int min = fmin(getj(ijkl), N - 1 - getj(ijkl));
    int arg = 0;

    if (fmin(geti(ijkl), N - 1 - geti(ijkl)) < min) {
        arg = 2;
        min = fmin(geti(ijkl), N - 1 - geti(ijkl));
    }

    if (fmin(getk(ijkl), N - 1 - getk(ijkl)) < min) {
        arg = 3;
        min = fmin(getk(ijkl), N - 1 - getk(ijkl));
    }

    if (fmin(getl(ijkl), N - 1 - getl(ijkl)) < min) {
        arg = 1;
        min = fmin(getl(ijkl), N - 1 - getl(ijkl));
    }

    for (int i = 0; i < arg; i++) {
        ijkl = rot90(ijkl,N);
    }

    if (getj(ijkl) < N - 1 - getj(ijkl)) {
        ijkl = mirvert(ijkl,N);
    }

    return ijkl;
}

int symmetry90(int ijkl,int N) {
    return ((geti(ijkl) << 15) + (getj(ijkl) << 10) + (getk(ijkl) << 5) + getl(ijkl)) ==
           (((N - 1 - getk(ijkl)) << 15) + ((N - 1 - getl(ijkl)) << 10) + (getj(ijkl) << 5) + geti(ijkl));
}

int symmetry(int ijkl,int N) {
    if (geti(ijkl) == N - 1 - getj(ijkl) && getk(ijkl) == N - 1 - getl(ijkl)) {
        if (symmetry90(ijkl,N)) {
            return 2;
        } else {
            return 4;
        }
    } else {
        return 8;
    }
}



void add_constellation(int ld, int rd, int col, int startijkl,ConstellationArrayList* constellations) {
    Constellation new_constellation = {0, ld, rd, col, startijkl, 0};
    constellation_arraylist_add(constellations, new_constellation);
}

void setPreQueens(int ld, int rd, int col, int k, int l, int row, int queens, int LD, int RD, int *counter,ConstellationArrayList* constellations,int N) {
    int mask = (1 << N) - 1;//setPreQueensで使用
    // k行とl行はさらに進む
    if (row == k || row == l) {
        setPreQueens(ld << 1, rd >> 1, col, k, l, row + 1, queens, LD, RD, counter,constellations,N);
        return;
    }

    // preQueensのクイーンが揃うまでクイーンを追加する
    if (queens == PRESETQUEENS) {
        add_constellation(ld, rd, col, row << 20,constellations);
        (*counter)++;
        return;
    } else {
        // 現在の行にクイーンを配置できる位置（自由な位置）を計算
        uint32_t free = ~(ld | rd | col | (LD >> (N - 1 - row)) | (RD << (N - 1 - row))) & mask;
        uint32_t bit;
        while (free > 0) {
            bit = free & (-free);
            free -= bit;
            // 自由な位置がある限り、その位置にクイーンを配置し、再帰的に次の行に進む
            setPreQueens((ld | bit) << 1, (rd | bit) >> 1, col | bit, k, l, row + 1, queens + 1, LD, RD, counter,constellations,N);
        }
    }
}

void execSolutions(ConstellationArrayList* constellations,int N) {
    //
    int N3 = N - 3;//execSolutionで使用
    int N4 = N - 4;//execSolutionで使用
    int L = 1 << (N - 1);//execSolutionで使用
    int L3 = 1 << N3;//execSolutionで使用
    int L4 = 1 << N4;//execSolutionで使用    
    int j=0, k=0, l=0, ijkl=0, ld=0, rd=0, col=0, startIjkl=0, start=0, free=0, LD=0,jmark=0,endmark=0,mark1=0,mark2=0;
    int smallmask = (1 << (N - 2)) - 1;
    long tempcounter = 0;
    //for (int i = 0; i < constellations->size; i++) {
    for (size_t i = 0; i < constellations->size; i++) {
        Constellation* constellation = &constellations->data[i];
        startIjkl = constellation->startijkl;
        start = startIjkl >> 20;
        ijkl = startIjkl & ((1 << 20) - 1);
        j = getj(ijkl);
        k = getk(ijkl);
        l = getl(ijkl);

        // LDとrdを1つずつ右にずらすが、これは右列は重要ではないから（常に女王lが占有している）。
        LD = (L >> j) | (L >> l);
        ld = constellation->ld >> 1;
        ld |= LD >> (N - start);
        rd = constellation->rd >> 1;

        if (start > k) {
            rd |= (L >> (start - k + 1));
        }
        if (j >= 2 * N - 33 - start) {
            rd |= (L >> j) << (N - 2 - start);
        }

        col = (constellation->col >> 1) | (~smallmask);
        free = ~(ld | rd | col);

        if (j < N - 3) {
            jmark = j + 1;
            endmark = N - 2;

            if (j > 2 * N - 34 - start) {
                if (k < l) {
                    mark1 = k - 1;
                    mark2 = l - 1;

                    if (start < l) {
                        if (start < k) {
                            if (l != k + 1) {
                                SQBkBlBjrB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                            } else {
                                SQBklBjrB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                            }
                        } else {
                            SQBlBjrB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                        }
                    } else {
                        SQBjrB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                    }
                } else {
                    mark1 = l - 1;
                    mark2 = k - 1;

                    if (start < k) {
                        if (start < l) {
                            if (k != l + 1) {
                                SQBlBkBjrB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                            } else {
                                SQBlkBjrB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                            }
                        } else {
                            SQBkBjrB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                        }
                    } else {
                        SQBjrB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                    }
                }
            } else {
                if (k < l) {
                    mark1 = k - 1;
                    mark2 = l - 1;

                    if (l != k + 1) {
                        SQBjlBkBlBjrB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                    } else {
                        SQBjlBklBjrB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                    }
                } else {
                    mark1 = l - 1;
                    mark2 = k - 1;

                    if (k != l + 1) {
                        SQBjlBlBkBjrB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                    } else {
                        SQBjlBlkBjrB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                    }
                }
            }
        } else if (j == N - 3) {
            endmark = N - 2;

            if (k < l) {
                mark1 = k - 1;
                mark2 = l - 1;

                if (start < l) {
                    if (start < k) {
                        if (l != k + 1) {
                            SQd2BkBlB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                        } else {
                            SQd2BklB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                        }
                    } else {
                        mark2=l-1;
                        SQd2BlB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                    }
                } else {
                    SQd2B(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                }
            } else {
                mark1 = l - 1;
                mark2 = k - 1;
                endmark = N - 2;

                if (start < k) {
                    if (start < l) {
                        if (k != l + 1) {
                            SQd2BlBkB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                        } else {
                            SQd2BlkB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                        }
                    } else {
                        mark2=k-1;
                        SQd2BkB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                    }
                } else {
                    SQd2B(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                }
            }
        } else if (j == N - 2) {
            if (k < l) {
                endmark = N - 2;

                if (start < l) {
                    if (start < k) {
                        mark1 = k - 1;

                        if (l != k + 1) {
                            mark2 = l - 1;
                            SQd1BkBlB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                        } else {
                            SQd1BklB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                        }
                    } else {
                        mark2 = l - 1;
                        SQd1BlB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                    }
                } else {
                    SQd1B(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                }
            } else {
                if (start < k) {
                    if (start < l) {
                        if(k<N-2){
                            mark1 = l - 1;
                            endmark = N - 2;

                            if (k != l + 1) {
                                mark2 = k - 1;
                                SQd1BlBkB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                            } else {
                                SQd1BlkB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                            }
                        } else {
                            if (l != N - 3) {
                                mark2 = l - 1;
                                endmark = N - 3;
                                SQd1BlB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                            } else {
                                endmark = N - 4;
                                SQd1B(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                            }
                        }
                    } else {
                        if (k != N - 2) {
                            mark2 = k - 1;
                            endmark = N - 2;
                            SQd1BkB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                        } else {
                            endmark = N - 3;
                            SQd1B(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                        }
                    }
                } else {
                    endmark = N - 2;
                    SQd1B(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
                }
            }
        } else {
            endmark = N - 2;

            if (start > k) {
                SQd0B(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
            } else {
                mark1 = k - 1;
                SQd0BkB(ld, rd, col, start, free,jmark,endmark,mark1,mark2, &tempcounter,N,N3,N4,L,L3,L4);
            }
        }

        // 完成した開始コンステレーションを削除する。
        constellation->solutions = tempcounter * symmetry(ijkl,N);
        tempcounter = 0;
    }
}

void genConstellations(IntHashSet* ijklList,ConstellationArrayList* constellations,int N) {
    int halfN = (N + 1) / 2;
    int L = 1 << (N - 1);
    
    for (int k = 1; k < halfN; k++) {
        for (int l = k + 1; l < N - 1; l++) {
            for (int i = k + 1; i < N - 1; i++) {
                if (i == N - 1 - l) {
                    continue;
                }
                for (int j = N - k - 2; j > 0; j--) {
                    if (j == i || l == j) {
                        continue;
                    }
                    //printf("check\n");
                    //printf("Checking i=%d, j=%d, k=%d, l=%d\n", i, j, k, l); // デバッグ用プリント
                    if (!checkRotations(ijklList, i, j, k, l,N)) {
                        int_hashset_add(ijklList, toijkl(i, j, k, l));
                    }
                }
            }
        }
    }

    for (int j = 1; j < N - 2; j++) {
        for (int l = j + 1; l < N - 1; l++) {
            int_hashset_add(ijklList, toijkl(0, j, 0, l));
        }
    }

    IntHashSet* ijklListJasmin = create_int_hashset();
    for (size_t i = 0; i < ijklList->size; i++) {
        int startConstellation = ijklList->data[i];
        int_hashset_add(ijklListJasmin, jasmin(startConstellation,N));
    }
    //free_int_hashset(ijklList);
    ijklList = ijklListJasmin;

    //int i, j, k, l, ld, rd, col, currentSize = 0;
    for (size_t s = 0; s < ijklList->size; s++) {
        int sc = ijklList->data[s];
        int i = geti(sc);
        int j = getj(sc);
        int k = getk(sc);
        int l = getl(sc);

        int ld = (L >> (i - 1)) | (1 << (N - k));
        int rd = (L >> (i + 1)) | (1 << (l - 1));
        int col = 1 | L | (L >> i) | (L >> j);
        int LD = (L >> j) | (L >> l);
        int RD = (L >> j) | (1 << k);
        int counter = 0;

        setPreQueens(ld, rd, col, k, l, 1, j == N - 1 ? 3 : 4, LD, RD, &counter,constellations,N);
        int currentSize = constellations->size;

        for (int a = 0; a < counter; a++) {
            constellations->data[currentSize - a - 1].startijkl |= toijkl(i, j, k, l);
        }
    }
}

   void SQd0B(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    if(row==endmark){
      (*tempcounter)++;
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit=free&(-free);
      free-=bit;
      int next_ld=((ld|bit)<<1);
      int next_rd=((rd|bit)>>1);
      int next_col=(col|bit);
      nextfree=~(next_ld|next_rd|next_col);
      if(nextfree>0){
        if(row<endmark-1){
          if(~((next_ld<<1)|(next_rd>>1)|(next_col))>0)
            SQd0B(next_ld,next_rd,next_col,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }else{
          SQd0B(next_ld,next_rd,next_col,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }
      }
    }
  }
   void SQd0BkB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|L3);
        if(nextfree>0){
          SQd0B((ld|bit)<<2,((rd|bit)>>2)|L3,col|bit,row+2,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd0BkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
   void SQd1BklB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|1|L4);
        if(nextfree>0){
          SQd1B(((ld|bit)<<3)|1,((rd|bit)>>3)|L4,col|bit,row+3,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd1BklB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
   void SQd1B(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    if(row==endmark){
      (*tempcounter)++;
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit=free&(-free);
      free-=bit;
      int next_ld=((ld|bit)<<1);
      int next_rd=((rd|bit)>>1);
      int next_col=(col|bit);
      nextfree=~(next_ld|next_rd|next_col);
      if(nextfree>0){
        if(row+1<endmark){
          if(~((next_ld<<1)|(next_rd>>1)|(next_col))>0)
            SQd1B(next_ld,next_rd,next_col,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }else{
          SQd1B(next_ld,next_rd,next_col,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }
      }
    }
  }
   void SQd1BkBlB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|L3);
        if(nextfree>0){
          SQd1BlB(((ld|bit)<<2),((rd|bit)>>2)|L3,col|bit,row+2,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd1BkBlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
   void SQd1BlB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    int bit;
    int nextfree;
    if(row==mark2){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        int next_ld=((ld|bit)<<2)|1;
        int next_rd=((rd|bit)>>2);
        int next_col=(col|bit);
        nextfree=~(next_ld|next_rd|next_col);
        if(nextfree>0){
          if(row+2<endmark){
            if(~((next_ld<<1)|(next_rd>>1)|(next_col))>0)
              SQd1B(next_ld,next_rd,next_col,row+2,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
          }else{
            SQd1B(next_ld,next_rd,next_col,row+2,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
          }
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd1BlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
   void SQd1BlkB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|2|L3);
        if(nextfree>0){
          SQd1B(((ld|bit)<<3)|2,((rd|bit)>>3)|L3,col|bit,row+3,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd1BlkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
   void SQd1BlBkB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1);
        if(nextfree>0){
          SQd1BkB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd1BlBkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
   void SQd1BkB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    int bit;
    int nextfree;
    if(row==mark2){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|L3);
        if(nextfree>0){
          SQd1B(((ld|bit)<<2),((rd|bit)>>2)|L3,col|bit,row+2,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd1BkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
   void SQd2BlkB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|L3|2);
        if(nextfree>0){
          SQd2B(((ld|bit)<<3)|2,((rd|bit)>>3)|L3,col|bit,row+3,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd2BlkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
   void SQd2BklB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|L4|1);
        if(nextfree>0){
          SQd2B(((ld|bit)<<3)|1,((rd|bit)>>3)|L4,col|bit,row+3,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd2BklB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
   void SQd2BlBkB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1);
        if(nextfree>0){
          SQd2BkB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd2BlBkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
   void SQd2BkBlB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<(N3)));
        if(nextfree>0){
          SQd2BlB(((ld|bit)<<2),((rd|bit)>>2)|(1<<(N3)),col|bit,row+2,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd2BkBlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
   void SQd2BlB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    int bit;
    int nextfree;
    if(row==mark2){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1);
        if(nextfree>0){
          SQd2B(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd2BlB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
   void SQd2BkB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    int bit;
    int nextfree;
    if(row==mark2){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|L3);
        if(nextfree>0){
          SQd2B(((ld|bit)<<2),((rd|bit)>>2)|L3,col|bit,row+2,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQd2BkB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
   void SQd2B(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    if(row==endmark){
      if((free&(~1))>0){
        (*tempcounter)++;
      }
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit=free&(-free);
      free-=bit;
      int next_ld=((ld|bit)<<1);
      int next_rd=((rd|bit)>>1);
      int next_col=(col|bit);
      nextfree=~(next_ld|next_rd|next_col);
      if(nextfree>0){
        if(row<endmark-1){
          if(~((next_ld<<1)|(next_rd>>1)|(next_col))>0)
            SQd2B(next_ld,next_rd,next_col,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }else{
          SQd2B(next_ld,next_rd,next_col,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }
      }
    }
  }
  // for d>2 but d <small enough>
   void SQBkBlBjrB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|(1<<(N3)));
        if(nextfree>0){
          SQBlBjrB(((ld|bit)<<2),((rd|bit)>>2)|(1<<(N3)),col|bit,row+2,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQBkBlBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
   void SQBlBjrB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    int bit;
    int nextfree;
    if(row==mark2){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1);
        if(nextfree>0){
          SQBjrB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQBlBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
   void SQBjrB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    int bit;
    int nextfree;
    if(row==jmark){
      free&=(~1);
      ld|=1;
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
        if(nextfree>0){
          SQB(((ld|bit)<<1),(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
   void SQB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    if(row==endmark){
      (*tempcounter)++;
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit=free&(-free);
      free-=bit;
      int next_ld=((ld|bit)<<1);
      int next_rd=((rd|bit)>>1);
      int next_col=(col|bit);
      nextfree=~(next_ld|next_rd|next_col);
      if(nextfree>0){
        if(row<endmark-1){
          if(~((next_ld<<1)|(next_rd>>1)|(next_col))>0){
            SQB(next_ld,next_rd,next_col,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
          }
        }else{
          SQB(next_ld,next_rd,next_col,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }
      }
    }
  }
   void SQBlBkBjrB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|1);
        if(nextfree>0){
          SQBkBjrB(((ld|bit)<<2)|1,(rd|bit)>>2,col|bit,row+2,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQBlBkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
   void SQBkBjrB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    int bit;
    int nextfree;
    if(row==mark2){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<2)|((rd|bit)>>2)|(col|bit)|L3);
        if(nextfree>0){
          SQBjrB(((ld|bit)<<2),((rd|bit)>>2)|L3,col|bit,row+2,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQBkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
   void SQBklBjrB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|L4|1);
        if(nextfree>0){
          SQBjrB(((ld|bit)<<3)|1,((rd|bit)>>3)|L4,col|bit,row+3,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
        }
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQBklBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
   void SQBlkBjrB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    int bit;
    int nextfree;
    if(row==mark1){
      while(free>0){
        bit=free&(-free);
        free-=bit;
        nextfree=~(((ld|bit)<<3)|((rd|bit)>>3)|(col|bit)|L3|2);
        if(nextfree>0)
          SQBjrB(((ld|bit)<<3)|2,((rd|bit)>>3)|L3,col|bit,row+3,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
      return;
    }
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQBlkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
  // for d <big>
   void SQBjlBkBlBjrB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    if(row==N-1-jmark){
      rd|=L;
      free&=~L;
      SQBkBlBjrB(ld,rd,col,row,free,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQBjlBkBlBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
   void SQBjlBlBkBjrB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    if(row==N-1-jmark){
      rd|=L;
      free&=~L;
      SQBlBkBjrB(ld,rd,col,row,free,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQBjlBlBkBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
   void SQBjlBklBjrB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    if(row==N-1-jmark){
      rd|=L;
      free&=~L;
      SQBklBjrB(ld,rd,col,row,free,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQBjlBklBjrB((ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }
   void SQBjlBlkBjrB(int ld,int rd,int col,int row,int free,int jmark,int endmark,int mark1,int mark2, long* tempcounter,int N,int N3,int N4,int L,int L3,int L4)
  {
    if(row==N-1-jmark){
      rd|=L;
      free&=~L;
      SQBlkBjrB(ld,rd,col,row,free,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      return;
    }
    int bit;
    int nextfree;
    while(free>0){
      bit=free&(-free);
      free-=bit;
      nextfree=~(((ld|bit)<<1)|((rd|bit)>>1)|(col|bit));
      if(nextfree>0){
        SQBjlBlkBjrB( (ld|bit)<<1,(rd|bit)>>1,col|bit,row+1,nextfree ,jmark,endmark,mark1,mark2, tempcounter,N,N3,N4,L,L3,L4);
      }
    }
  }

// 未使用変数対応
void f(int unuse,char* argv[]){
  printf("%d%s\n",unuse,argv[0]);
}
// メインメソッド
int main(int argc, char** argv) {
  f(argc,argv);
  unsigned int min=6;
  unsigned int targetN=17;
  struct timeval t0;
  struct timeval t1;
  for(unsigned int size=min;size<=targetN;++size){
    gettimeofday(&t0, NULL);
    long solutions=0;
    IntHashSet* ijklList = create_int_hashset();
    ConstellationArrayList* constellations = create_constellation_arraylist();

    // 関数呼び出し
    genConstellations(ijklList,constellations,size);
    execSolutions(constellations,size);
    solutions=calcSolutions(constellations,solutions);
    gettimeofday(&t1, NULL);
    int ss;int ms;int dd;
    if(t1.tv_usec<t0.tv_usec) {
      dd=(t1.tv_sec-t0.tv_sec-1)/86400;
      ss=(t1.tv_sec-t0.tv_sec-1)%86400;
      ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
    }else {
      dd=(t1.tv_sec-t0.tv_sec)/86400;
      ss=(t1.tv_sec-t0.tv_sec)%86400;
      ms=(t1.tv_usec-t0.tv_usec+500)/10000;
    }
    int hh=ss/3600;
    int mm=(ss-hh*3600)/60;
    ss%=60;
    printf("%2d:%13ld%10.2d:%02d:%02d:%02d.%02d\n",size,solutions,dd,hh,mm,ss,ms);    
   
    // 後処理
    free_int_hashset(ijklList);
    free_constellation_arraylist(constellations);
    
  }  
    return 0;
}
