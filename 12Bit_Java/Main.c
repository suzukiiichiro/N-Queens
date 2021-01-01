#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

// グローバル変数の宣言
int L, mask, LD, RD, counter;
int N;
int presetQueens;
long solutions, duration, storedDuration;
int N3, N4, L3, L4;
long tempcounter = 0;
int mark1, mark2, endmark, jmark;
// グローバル変数の宣言
long solutions = 0;

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

IntHashSet* ijklList;
ConstellationArrayList* constellations;

#define INITIAL_CAPACITY 1000

// IntHashSetの関数プロトタイプ
IntHashSet* create_int_hashset();
void free_int_hashset(IntHashSet* set);
int int_hashset_contains(IntHashSet* set, int value);
void int_hashset_add(IntHashSet* set, int value);

// ビット操作関数プロトタイプ
int checkRotations(IntHashSet* set, int i, int j, int k, int l);
int toijkl(int i, int j, int k, int l);
int geti(int sc);
int getj(int sc);
int getk(int sc);
int getl(int sc);
int jasmin(int ijkl);
int mirvert(int ijkl);
int rot90(int ijkl);
int symmetry90(int ijkl);
int symmetry(int ijkl);
void add_constellation(int ld, int rd, int col, int startijkl);

// 関数プロトタイプ宣言
void SQBkBlBjrB(int ld, int rd, int col, int start, int free);
void SQBklBjrB(int ld, int rd, int col, int start, int free);
void SQBlBjrB(int ld, int rd, int col, int start, int free);
void SQBjrB(int ld, int rd, int col, int start, int free);
void SQBlBkBjrB(int ld, int rd, int col, int start, int free);
void SQBlkBjrB(int ld, int rd, int col, int start, int free);
void SQBkBjrB(int ld, int rd, int col, int start, int free);
void SQBjlBkBlBjrB(int ld, int rd, int col, int start, int free);
void SQBjlBklBjrB(int ld, int rd, int col, int start, int free);
void SQBjlBlBkBjrB(int ld, int rd, int col, int start, int free);
void SQBjlBlkBjrB(int ld, int rd, int col, int start, int free);
void SQd2BkBlB(int ld, int rd, int col, int start, int free);
void SQd2BklB(int ld, int rd, int col, int start, int free);
void SQd2BlB(int ld, int rd, int col, int start, int free);
void SQd2B(int ld, int rd, int col, int start, int free);
void SQd2BlBkB(int ld, int rd, int col, int start, int free);
void SQd2BlkB(int ld, int rd, int col, int start, int free);
void SQd1BkBlB(int ld, int rd, int col, int start, int free);
void SQd1BklB(int ld, int rd, int col, int start, int free);
void SQd1BlB(int ld, int rd, int col, int start, int free);
void SQd1B(int ld, int rd, int col, int start, int free);
void SQd1BlBkB(int ld, int rd, int col, int start, int free);
void SQd1BlkB(int ld, int rd, int col, int start, int free);
void SQd0B(int ld, int rd, int col, int row, int free);
void SQd0BkB(int ld, int rd, int col, int row, int free);
void SQd2BkB(int ld, int rd, int col, int row, int free);
void SQd1BkB(int ld, int rd, int col, int row, int free);

// その他の関数プロトタイプ
void setPreQueens(int ld, int rd, int col, int k, int l, int row, int queens);
void execSolutions();
void genConstellations();
void initialize(int sn);
void print_constellations(ConstellationArrayList* list);

// 関数プロトタイプ宣言
long getSolutions();
void calcSolutions();

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

// 関数実装
long getSolutions() {
    return solutions;
}

void calcSolutions() {
    for (size_t i = 0; i < constellations->size; i++) {
        Constellation* c = &constellations->data[i];
        if (c->solutions >= 0) {
            solutions += c->solutions;
        }
    }
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

int checkRotations(IntHashSet* ijklList, int i, int j, int k, int l) {
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

int mirvert(int ijkl) {
    return toijkl(N - 1 - geti(ijkl), N - 1 - getj(ijkl), getl(ijkl), getk(ijkl));
}

int rot90(int ijkl) {
    return ((N - 1 - getk(ijkl)) << 15) + ((N - 1 - getl(ijkl)) << 10) + (getj(ijkl) << 5) + geti(ijkl);
}

int jasmin(int ijkl) {
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
        ijkl = rot90(ijkl);
    }

    if (getj(ijkl) < N - 1 - getj(ijkl)) {
        ijkl = mirvert(ijkl);
    }

    return ijkl;
}

int symmetry90(int ijkl) {
    return ((geti(ijkl) << 15) + (getj(ijkl) << 10) + (getk(ijkl) << 5) + getl(ijkl)) ==
           (((N - 1 - getk(ijkl)) << 15) + ((N - 1 - getl(ijkl)) << 10) + (getj(ijkl) << 5) + geti(ijkl));
}

int symmetry(int ijkl) {
    if (geti(ijkl) == N - 1 - getj(ijkl) && getk(ijkl) == N - 1 - getl(ijkl)) {
        if (symmetry90(ijkl)) {
            return 2;
        } else {
            return 4;
        }
    } else {
        return 8;
    }
}

void initialize(int sn) {
    N = sn;
    presetQueens = 4;
    solutions = 0;
    N3 = N - 3;
    N4 = N - 4;
    L = 1 << (N - 1);
    L3 = 1 << N3;
    L4 = 1 << N4;
}

void print_constellations(ConstellationArrayList* list) {
    for (size_t i = 0; i < list->size; i++) {
        Constellation* c = &list->data[i];
        printf("Constellation %zu:\n", i + 1);
        printf("  id: %d\n", c->id);
        printf("  ld: %d\n", c->ld);
        printf("  rd: %d\n", c->rd);
        printf("  col: %d\n", c->col);
        printf("  startijkl: %d\n", c->startijkl);
        printf("  solutions: %ld\n", c->solutions);
        printf("\n");
    }
}

void add_constellation(int ld, int rd, int col, int startijkl) {
    Constellation new_constellation = {0, ld, rd, col, startijkl, 0};
    constellation_arraylist_add(constellations, new_constellation);
}

void setPreQueens(int ld, int rd, int col, int k, int l, int row, int queens) {
    // k行とl行はさらに進む
    if (row == k || row == l) {
        setPreQueens(ld << 1, rd >> 1, col, k, l, row + 1, queens);
        return;
    }

    // preQueensのクイーンが揃うまでクイーンを追加する
    if (queens == presetQueens) {
        add_constellation(ld, rd, col, row << 20);
        counter++;
        return;
    } else {
        // 現在の行にクイーンを配置できる位置（自由な位置）を計算
        uint32_t free = ~(ld | rd | col | (LD >> (N - 1 - row)) | (RD << (N - 1 - row))) & mask;
        uint32_t bit;
        while (free > 0) {
            bit = free & (-free);
            free -= bit;
            // 自由な位置がある限り、その位置にクイーンを配置し、再帰的に次の行に進む
            setPreQueens((ld | bit) << 1, (rd | bit) >> 1, col | bit, k, l, row + 1, queens + 1);
        }
    }
}

void execSolutions() {
    int j, k, l, ijkl, ld, rd, col, startIjkl, start, free, LD;
    int smallmask = (1 << (N - 2)) - 1;
    for (int i = 0; i < constellations->size; i++) {
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
                                SQBkBlBjrB(ld, rd, col, start, free);
                            } else {
                                SQBklBjrB(ld, rd, col, start, free);
                            }
                        } else {
                            SQBlBjrB(ld, rd, col, start, free);
                        }
                    } else {
                        SQBjrB(ld, rd, col, start, free);
                    }
                } else {
                    mark1 = l - 1;
                    mark2 = k - 1;

                    if (start < k) {
                        if (start < l) {
                            if (k != l + 1) {
                                SQBlBkBjrB(ld, rd, col, start, free);
                            } else {
                                SQBlkBjrB(ld, rd, col, start, free);
                            }
                        } else {
                            SQBkBjrB(ld, rd, col, start, free);
                        }
                    } else {
                        SQBjrB(ld, rd, col, start, free);
                    }
                }
            } else {
                if (k < l) {
                    mark1 = k - 1;
                    mark2 = l - 1;

                    if (l != k + 1) {
                        SQBjlBkBlBjrB(ld, rd, col, start, free);
                    } else {
                        SQBjlBklBjrB(ld, rd, col, start, free);
                    }
                } else {
                    mark1 = l - 1;
                    mark2 = k - 1;

                    if (k != l + 1) {
                        SQBjlBlBkBjrB(ld, rd, col, start, free);
                    } else {
                        SQBjlBlkBjrB(ld, rd, col, start, free);
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
                            SQd2BkBlB(ld, rd, col, start, free);
                        } else {
                            SQd2BklB(ld, rd, col, start, free);
                        }
                    } else {
                        SQd2BlB(ld, rd, col, start, free);
                    }
                } else {
                    SQd2B(ld, rd, col, start, free);
                }
            } else {
                mark1 = l - 1;
                mark2 = k - 1;
                endmark = N - 2;

                if (start < k) {
                    if (start < l) {
                        if (k != l + 1) {
                            SQd2BlBkB(ld, rd, col, start, free);
                        } else {
                            SQd2BlkB(ld, rd, col, start, free);
                        }
                    } else {
                        SQd2BkB(ld, rd, col, start, free);
                    }
                } else {
                    SQd2B(ld, rd, col, start, free);
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
                            SQd1BkBlB(ld, rd, col, start, free);
                        } else {
                            SQd1BklB(ld, rd, col, start, free);
                        }
                    } else {
                        mark2 = l - 1;
                        SQd1BlB(ld, rd, col, start, free);
                    }
                } else {
                    SQd1B(ld, rd, col, start, free);
                }
            } else {
                if (start < k) {
                    if (start < l) {
                        if (k != l + 1) {
                            mark1 = l - 1;
                            endmark = N - 2;

                            if (k != l + 1) {
                                mark2 = k - 1;
                                SQd1BlBkB(ld, rd, col, start, free);
                            } else {
                                SQd1BlkB(ld, rd, col, start, free);
                            }
                        } else {
                            if (l != N - 3) {
                                mark2 = l - 1;
                                endmark = N - 3;
                                SQd1BlB(ld, rd, col, start, free);
                            } else {
                                endmark = N - 4;
                                SQd1B(ld, rd, col, start, free);
                            }
                        }
                    } else {
                        if (k != N - 2) {
                            mark2 = k - 1;
                            endmark = N - 2;
                            SQd1BkB(ld, rd, col, start, free);
                        } else {
                            endmark = N - 3;
                            SQd1B(ld, rd, col, start, free);
                        }
                    }
                } else {
                    endmark = N - 2;
                    SQd1B(ld, rd, col, start, free);
                }
            }
        } else {
            endmark = N - 2;

            if (start > k) {
                SQd0B(ld, rd, col, start, free);
            } else {
                mark1 = k - 1;
                SQd0BkB(ld, rd, col, start, free);
            }
        }

        // 完成した開始コンステレーションを削除する。
        constellation->solutions = tempcounter * symmetry(ijkl);
        tempcounter = 0;
    }
}

void genConstellations() {
    int halfN = (N + 1) / 2;
    L = 1 << (N - 1);
    mask = (1 << N) - 1;

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
                    printf("check\n");
                    printf("Checking i=%d, j=%d, k=%d, l=%d\n", i, j, k, l); // デバッグ用プリント
                    if (!checkRotations(ijklList, i, j, k, l)) {
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
        int_hashset_add(ijklListJasmin, jasmin(startConstellation));
    }
    free_int_hashset(ijklList);
    ijklList = ijklListJasmin;

    int i, j, k, l, ld, rd, col, currentSize = 0;
    for (size_t i = 0; i < ijklList->size; i++) {
        int sc = ijklList->data[i];
        i = geti(sc);
        j = getj(sc);
        k = getk(sc);
        l = getl(sc);

        ld = (L >> (i - 1)) | (1 << (N - k));
        rd = (L >> (i + 1)) | (1 << (l - 1));
        col = 1 | L | (L >> i) | (L >> j);
        LD = (L >> j) | (L >> l);
        RD = (L >> j) | (1 << k);
        counter = 0;

        setPreQueens(ld, rd, col, k, l, 1, j == N - 1 ? 3 : 4);
        currentSize = constellations->size;

        for (int a = 0; a < counter; a++) {
            constellations->data[currentSize - a - 1].startijkl |= toijkl(i, j, k, l);
        }
    }
}

void SQd0B(int ld, int rd, int col, int row, int free) {
    if (row == endmark) {
        tempcounter++;
        return;
    }
    int bit, nextfree;
    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        int next_ld = ((ld | bit) << 1);
        int next_rd = ((rd | bit) >> 1);
        int next_col = (col | bit);
        nextfree = ~(next_ld | next_rd | next_col);
        if (nextfree > 0) {
            if (row < endmark - 1) {
                if (~((next_ld << 1) | (next_rd >> 1) | (next_col)) > 0)
                    SQd0B(next_ld, next_rd, next_col, row + 1, nextfree);
            } else {
                SQd0B(next_ld, next_rd, next_col, row + 1, nextfree);
            }
        }
    }
}

void SQd0BkB(int ld, int rd, int col, int row, int free) {
    int bit, nextfree;
    if (row == mark1) {
        while (free > 0) {
            bit = free & (-free);
            free -= bit;
            nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | L3);
            if (nextfree > 0) {
                SQd0B((ld | bit) << 2, ((rd | bit) >> 2) | L3, col | bit, row + 2, nextfree);
            }
        }
        return;
    }
    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if (nextfree > 0) {
            SQd0BkB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
    }
}

void SQd1BklB(int ld, int rd, int col, int row, int free) {
    int bit, nextfree;
    if (row == mark1) {
        while (free > 0) {
            bit = free & (-free);
            free -= bit;
            nextfree = ~(((ld | bit) << 3) | ((rd | bit) >> 3) | (col | bit) | 1 | L4);
            if (nextfree > 0) {
                SQd1B(((ld | bit) << 3) | 1, ((rd | bit) >> 3) | L4, col | bit, row + 3, nextfree);
            }
        }
        return;
    }
    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if (nextfree > 0) {
            SQd1BklB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
    }
}

void SQd1B(int ld, int rd, int col, int row, int free) {
    if (row == endmark) {
        tempcounter++;
        return;
    }
    int bit, nextfree;
    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        int next_ld = ((ld | bit) << 1);
        int next_rd = ((rd | bit) >> 1);
        int next_col = (col | bit);
        nextfree = ~(next_ld | next_rd | next_col);
        if (nextfree > 0) {
            if (row + 1 < endmark) {
                if (~((next_ld << 1) | (next_rd >> 1) | (next_col)) > 0)
                    SQd1B(next_ld, next_rd, next_col, row + 1, nextfree);
            } else {
                SQd1B(next_ld, next_rd, next_col, row + 1, nextfree);
            }
        }
    }
}

void SQd1BkBlB(int ld, int rd, int col, int row, int free) {
    int bit, nextfree;
    if (row == mark1) {
        while (free > 0) {
            bit = free & (-free);
            free -= bit;
            nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | L3);
            if (nextfree > 0) {
                SQd1BlB(((ld | bit) << 2), ((rd | bit) >> 2) | L3, col | bit, row + 2, nextfree);
            }
        }
        return;
    }
    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if (nextfree > 0) {
            SQd1BkBlB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
    }
}

void SQd1BlB(int ld, int rd, int col, int row, int free) {
    int bit, nextfree;
    if (row == mark1) {
        while (free > 0) {
            bit = free & (-free);
            free -= bit;
            nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | 1 | L4);
            if (nextfree > 0) {
                SQd1BkBlB(((ld | bit) << 2) | 1, ((rd | bit) >> 2) | L4, col | bit, row + 2, nextfree);
            }
        }
        return;
    }
    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if (nextfree > 0) {
            SQd1BlB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
    }
}

void SQd1BlkB(int ld, int rd, int col, int row, int free) {
    int bit, nextfree;
    if (row == mark1) {
        while (free > 0) {
            bit = free & (-free);
            free -= bit;
            nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | 1);
            if (nextfree > 0) {
                SQd1BlBkB(((ld | bit) << 2) | 1, ((rd | bit) >> 2) | L4, col | bit, row + 2, nextfree);
            }
        }
        return;
    }
    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if (nextfree > 0) {
            SQd1BlkB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
    }
}

void SQd1BlBkB(int ld, int rd, int col, int row, int free) {
    int bit, nextfree;
    if (row == mark1) {
        while (free > 0) {
            bit = free & (-free);
            free -= bit;
            nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | 1 | L3);
            if (nextfree > 0) {
                SQd1BlkB(((ld | bit) << 2) | 1, ((rd | bit) >> 2) | L4, col | bit, row + 2, nextfree);
            }
        }
        return;
    }
    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if (nextfree > 0) {
            SQd1BlBkB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
    }
}

void SQd1BkB(int ld, int rd, int col, int row, int free) {
    int bit, nextfree;
    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if (nextfree > 0) {
            SQd1BkB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
    }
}

void SQd2BlkB(int ld, int rd, int col, int row, int free) {
    int bit, nextfree;
    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 3) | ((rd | bit) >> 3) | (col | bit) | 1 | L4);
        if (nextfree > 0) {
            SQd2BlB(((ld | bit) << 3) | 1, ((rd | bit) >> 3) | L4, col | bit, row + 3, nextfree);
        }
    }
}

void SQd2BklB(int ld, int rd, int col, int row, int free) {
    int bit, nextfree;
    if (row == mark1) {
        while (free > 0) {
            bit = free & (-free);
            free -= bit;
            nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit) | 1 | L4);
            if (nextfree > 0) {
                SQd2BlB(((ld | bit) << 1) | 1, ((rd | bit) >> 1) | L4, col | bit, row + 1, nextfree);
            }
        }
        return;
    }
    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if (nextfree > 0) {
            SQd2BklB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
    }
}

void SQd2BlBkB(int ld, int rd, int col, int row, int free) {
    int bit, nextfree;
    if (row == mark1) {
        while (free > 0) {
            bit = free & (-free);
            free -= bit;
            nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | 1 | L3);
            if (nextfree > 0) {
                SQd2BlkB(((ld | bit) << 2) | 1, ((rd | bit) >> 2) | L4, col | bit, row + 2, nextfree);
            }
        }
        return;
    }
    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if (nextfree > 0) {
            SQd2BlBkB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
    }
}

void SQd2BkBlB(int ld, int rd, int col, int row, int free) {
    int bit, nextfree;
    if (row == mark1) {
        while (free > 0) {
            bit = free & (-free);
            free -= bit;
            nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | 1);
            if (nextfree > 0) {
                SQd2BlBkB(((ld | bit) << 2) | 1, ((rd | bit) >> 2) | L4, col | bit, row + 2, nextfree);
            }
        }
        return;
    }
    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if (nextfree > 0) {
            SQd2BkBlB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
    }
}

void SQd2BlB(int ld, int rd, int col, int row, int free) {
    int bit, nextfree;
    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if (nextfree > 0) {
            SQd2BlB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
    }
}

void SQd2BkB(int ld, int rd, int col, int row, int free) {
    int bit, nextfree;
    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | 1);
        if (nextfree > 0) {
            SQd2BkB(((ld | bit) << 2) | 1, ((rd | bit) >> 2) | L4, col | bit, row + 2, nextfree);
        }
    }
}

void SQBkBlBjrB(int ld, int rd, int col, int row, int free) {
    int bit;
    int nextfree;

    if (row == mark1) {
        while (free > 0) {
            bit = free & (-free);
            free -= bit;
            nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | (1 << N3));
            if (nextfree > 0) {
                SQBlBjrB((ld | bit) << 2, (rd | bit) >> 2 | (1 << N3), col | bit, row + 2, nextfree);
            }
        }
        return;
    }

    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if (nextfree > 0) {
            SQBkBlBjrB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
    }
}

// SQBklBjrB 関数
void SQBklBjrB(int ld, int rd, int col, int row, int free) {
    int bit;
    int nextfree;

    if (row == mark1) {
        while (free > 0) {
            bit = free & (-free);
            free -= bit;
            nextfree = ~(((ld | bit) << 3) | ((rd | bit) >> 3) | (col | bit) | L4 | 1);
            if (nextfree > 0) {
                SQBjrB(((ld | bit) << 3) | 1, ((rd | bit) >> 3) | L4, col | bit, row + 3, nextfree);
            }
        }
        return;
    }

    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if (nextfree > 0) {
            SQBklBjrB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
    }
}

// SQBlBjrB 関数
void SQBlBjrB(int ld, int rd, int col, int row, int free) {
    int bit;
    int nextfree;

    if (row == mark2) {
        while (free > 0) {
            bit = free & (-free);
            free -= bit;
            nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | 1);
            if (nextfree > 0) {
                SQBjrB(((ld | bit) << 2) | 1, (rd | bit) >> 2, col | bit, row + 2, nextfree);
            }
        }
        return;
    }

    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if (nextfree > 0) {
            SQBlBjrB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
    }
}

// SQBjrB 関数
void SQBjrB(int ld, int rd, int col, int row, int free) {
    int bit;
    int nextfree;

    if (row == jmark) {
        free &= (~1);
        ld |= 1;
        while (free > 0) {
            bit = free & (-free);
            free -= bit;
            nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
            if (nextfree > 0) {
                SQd2B((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
            }
        }
        return;
    }

    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if (nextfree > 0) {
            SQBjrB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
    }
}

// SQBlBkBjrB 関数
void SQBlBkBjrB(int ld, int rd, int col, int row, int free) {
    int bit;
    int nextfree;

    if (row == mark1) {
        while (free > 0) {
            bit = free & (-free);
            free -= bit;
            nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | 1);
            if (nextfree > 0) {
                SQBkBjrB(((ld | bit) << 2) | 1, (rd | bit) >> 2, col | bit, row + 2, nextfree);
            }
        }
        return;
    }

    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if (nextfree > 0) {
            SQBlBkBjrB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
    }
}

// SQBlkBjrB 関数
void SQBlkBjrB(int ld, int rd, int col, int row, int free) {
    int bit;
    int nextfree;

    if (row == mark1) {
        while (free > 0) {
            bit = free & (-free);
            free -= bit;
            nextfree = ~(((ld | bit) << 3) | ((rd | bit) >> 3) | (col | bit) | L3 | 2);
            if (nextfree > 0) {
                SQBjrB(((ld | bit) << 3) | 2, ((rd | bit) >> 3) | L3, col | bit, row + 3, nextfree);
            }
        }
        return;
    }

    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if (nextfree > 0) {
            SQBlkBjrB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
    }
}

// SQBkBjrB 関数
void SQBkBjrB(int ld, int rd, int col, int row, int free) {
    int bit;
    int nextfree;

    if (row == mark2) {
        while (free > 0) {
            bit = free & (-free);
            free -= bit;
            nextfree = ~(((ld | bit) << 2) | ((rd | bit) >> 2) | (col | bit) | L3);
            if (nextfree > 0) {
                SQBjrB((ld | bit) << 2, (rd | bit) >> 2 | L3, col | bit, row + 2, nextfree);
            }
        }
        return;
    }

    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if (nextfree > 0) {
            SQBkBjrB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
    }
}

// SQBjlBkBlBjrB 関数
void SQBjlBkBlBjrB(int ld, int rd, int col, int row, int free) {
    if (row == N - 1 - jmark) {
        rd |= L;
        free &= ~L;
        SQBkBlBjrB(ld, rd, col, row, free);
        return;
    }
    int bit;
    int nextfree;
    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if (nextfree > 0) {
            SQBjlBkBlBjrB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
    }
}

// SQBjlBklBjrB 関数
void SQBjlBklBjrB(int ld, int rd, int col, int row, int free) {
    if (row == N - 1 - jmark) {
        rd |= L;
        free &= ~L;
        SQBklBjrB(ld, rd, col, row, free);
        return;
    }
    int bit;
    int nextfree;
    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if (nextfree > 0) {
            SQBjlBklBjrB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
    }
}

// SQBjlBlBkBjrB 関数
void SQBjlBlBkBjrB(int ld, int rd, int col, int row, int free) {
    if (row == N - 1 - jmark) {
        rd |= L;
        free &= ~L;
        SQBlBkBjrB(ld, rd, col, row, free);
        return;
    }
    int bit;
    int nextfree;
    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if (nextfree > 0) {
            SQBjlBlBkBjrB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
    }
}

// SQBjlBlkBjrB 関数
void SQBjlBlkBjrB(int ld, int rd, int col, int row, int free) {
    if (row == N - 1 - jmark) {
        rd |= L;
        free &= ~L;
        SQBlkBjrB(ld, rd, col, row, free);
        return;
    }
    int bit;
    int nextfree;
    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        nextfree = ~(((ld | bit) << 1) | ((rd | bit) >> 1) | (col | bit));
        if (nextfree > 0) {
            SQBjlBlkBjrB((ld | bit) << 1, (rd | bit) >> 1, col | bit, row + 1, nextfree);
        }
    }
}

// SQd2B 関数
void SQd2B(int ld, int rd, int col, int row, int free) {
    if (row == endmark) {
        if ((free & (~1)) > 0) {
            tempcounter++;
        }
        return;
    }
    int bit;
    int nextfree;
    while (free > 0) {
        bit = free & (-free);
        free -= bit;
        int next_ld = ((ld | bit) << 1);
        int next_rd = ((rd | bit) >> 1);
        int next_col = (col | bit);
        nextfree = ~(next_ld | next_rd | next_col);
        if (nextfree > 0) {
            if (row < endmark - 1) {
                if (~((next_ld << 1) | (next_rd >> 1) | (next_col)) > 0) {
                    SQd2B(next_ld, next_rd, next_col, row + 1, nextfree);
                }
            } else {
                SQd2B(next_ld, next_rd, next_col, row + 1, nextfree);
            }
        }
    }
}

// メインメソッド
int main(int argc, char** argv) {
    initialize(6); // Example for 6-Queens problem
    ijklList = create_int_hashset();
    constellations = create_constellation_arraylist();

    // 関数呼び出し
    genConstellations();
    execSolutions();
    calcSolutions();
    printf("%17ld\n", getSolutions());

    print_constellations(constellations);

    // 後処理
    free_int_hashset(ijklList);
    free_constellation_arraylist(constellations);
    return 0;
}