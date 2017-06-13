/**
  Cで学ぶアルゴリズムとデータ構造  
  ステップバイステップでＮ−クイーン問題を最適化
  一般社団法人  共同通信社  情報技術局  鈴木  維一郎(suzuki.iichiro@kyodonews.jp)
  
  Java版 N-Queen
  https://github.com/suzukiiichiro/AI_Algorithm_N-Queen
  Bash版 N-Queen
  https://github.com/suzukiiichiro/AI_Algorithm_Bash
  Lua版  N-Queen
  https://github.com/suzukiiichiro/AI_Algorithm_Lua
 
  ステップバイステップでＮ−クイーン問題を最適化
   １．ブルートフォース（力まかせ探索） NQueen01()
   ２．配置フラグ（制約テスト高速化）   NQueen02()
   ３．バックトラック                   NQueen03() N16: 1:07
   ４．対称解除法(回転と斜軸）          NQueen04() N16: 1:09
   ５．枝刈りと最適化                   NQueen05() N16: 0:18
   ６．ビットマップ                     NQueen06() N16: 0:13
   ７．ビットマップ+対称解除法          NQueen07() N16: 0:21
   ８．ビットマップ+クイーンの場所で分岐NQueen08() N16: 0:13
   ９．ビットマップ+枝刈りと最適化      NQueen09() N16: 0:02
 <>10．もっとビットマップ               NQueen10()
   11．マルチスレッド                   NQueen11()

  実行結果

*/
#include <stdio.h>
#include <time.h>

#define  MAXSIZE   30
#define  MINSIZE    5
//#define  i64  __int64
#define  i64  int

//**********************************************
// Display the Board Image
//**********************************************
void Display(int n, int *board)
{
    int  y, bit, topb = 1 << (n - 1);

    printf("N= %d\n", n);
    for (y=0; y<n; y++) {
        for (bit=topb; bit; bit>>=1)
            printf("%s ", (board[y] & bit)? "Q": "-");
        printf("\n");
    }
    printf("\n");
}
//**********************************************
// Check Unique Solutions
//**********************************************
void Check(int *board, int size, int last, int topb, int posa, int posb, int posc, i64 *cnt8, i64 *cnt4, i64 *cnt2)
{
    int  pos1, pos2, bit1, bit2;

    // 90-degree rotation
    if (board[posa] == 1) {
        for (pos1=1,bit2=2; pos1<size; pos1++,bit2<<=1) {
            for (pos2=last,bit1=1; board[pos1]!=bit1 && board[pos2]!=bit2; pos2--,bit1<<=1);
            if (board[pos1] != bit1) return;
            if (board[pos2] != bit2) break;
        }
        if (pos1 == size) {
            (*cnt2)++;
            //Display(size, board);
            return;
        }
    }

    // 180-degree rotation
    if (board[last] == posb) {
        for (pos1=1,pos2=size-2; pos1<size; pos1++,pos2--) {
            for (bit2=topb,bit1=1; board[pos1]!=bit1 && board[pos2]!=bit2; bit2>>=1,bit1<<=1);
            if (board[pos1] != bit1) return;
            if (board[pos2] != bit2) break;
        }
        if (pos1 == size) {
            (*cnt4)++;
            //Display(size, board);
            return;
        }
    }

    // 270-degree rotation
    if (board[posc] == topb) {
        for (pos1=1,bit2=topb>>1; pos1<size; pos1++,bit2>>=1) {
            for (pos2=0,bit1=1; board[pos1]!=bit1 && board[pos2]!=bit2; pos2++,bit1<<=1);
            if (board[pos1] != bit1) return;
            if (board[pos2] != bit2) break;
        }
    }

    (*cnt8)++;
    //Display(size, board);
}
//**********************************************
// First queen is inside
//**********************************************
void Inside(int n, int x0, int x1, i64 *uniq, i64 *allc)
{
    int  size, last, y, i;
    int  bits, bit, mask, left, rigt;
    int  posa, posb, posc, topb, side, gate;
    int  board[MAXSIZE];
    int  s_mask[MAXSIZE];
    int  s_left[MAXSIZE];
    int  s_rigt[MAXSIZE];
    int  s_bits[MAXSIZE];
    i64  cnt8, cnt4, cnt2;

    // Initialize
    size = n;
    last = n - 1;
    mask = (1 << n) - 1;
    cnt8 = cnt4 = cnt2 = 0;

    // ControlValue
    topb = 1 << last;
    side = topb | 1;
    gate = (mask >> x0) & (mask << x0);
    posa = last - x0;
    posb = topb >> x0;
    posc = x0;

    // y=0: 000001110 (select)
    // y=1: 111111111 (select)
    board[0] = 1 << x0;
    board[1] = bit = 1 << x1;
    mask = mask ^ (board[0] | bit);
    left = board[0] << 2 | bit << 1;
    rigt = board[0] >> 2 | bit >> 1;
    y = i = 2;

    // y -> posc
    if (posc == 1) goto NEXT2;
    mask = mask ^ side;
NEXT1:
    if (i == posc) {
        mask |= side;
        goto NEXT2;
    }
    bits = mask & ~(left | rigt);
    if (bits) {
        s_mask[i] = mask;
        s_left[i] = left;
        s_rigt[i] = rigt;
PROC1:
        bits ^= bit = -bits & bits;
        board[i] = bit;
        s_bits[i++] = bits;
        mask = mask ^ bit;
        left = (left | bit) << 1;
        rigt = (rigt | bit) >> 1;
        goto NEXT1;
BACK1:
        bits = s_bits[--i];
        if (bits) {
            mask = s_mask[i];
            left = s_left[i];
            rigt = s_rigt[i];
            goto PROC1;
        }
    }
    if (i == y) goto FINISH;
    goto BACK1;

    // posc -> posa
NEXT2:
    bits = mask & ~(left | rigt);
    if (bits) {
        s_mask[i] = mask;
        s_left[i] = left;
        s_rigt[i] = rigt;
PROC2:
        bits ^= bit = -bits & bits;
        board[i] = bit;
        s_bits[i++] = bits;
        mask = mask ^ bit;
        left = (left | bit) << 1;
        rigt = (rigt | bit) >> 1;
        if (i == posa) {
            if (mask & topb) goto BACK2;
            if (mask & 1) {
                if ((left | rigt) & 1) goto BACK2;
                bits = 1;
            } else {
                bits = mask & ~(left | rigt);
                if (!bits) goto BACK2;
            }
            goto NEXT3;
        } else {
            goto NEXT2;
        }
BACK2:
        bits = s_bits[--i];
        if (bits) {
            mask = s_mask[i];
            left = s_left[i];
            rigt = s_rigt[i];
            goto PROC2;
        }
    }
    if (i == y) goto FINISH;
    if (i > posc) goto BACK2;
    goto BACK1;

    // posa -> last
NEXT3:
    if (i == last) {
        if (bits & gate) {
            board[i] = bits;
            Check(board, size, last, topb, posa, posb, posc, &cnt8, &cnt4, &cnt2);
        }
        goto BACK3;
    }
    s_mask[i] = mask;
    s_left[i] = left;
    s_rigt[i] = rigt;
PROC3:
    bits ^= bit = -bits & bits;
    board[i] = bit;
    s_bits[i++] = bits;
    mask = mask ^ bit;
    left = (left | bit) << 1;
    rigt = (rigt | bit) >> 1;
    bits = mask & ~(left | rigt);
    if (bits) goto NEXT3;
BACK3:
    bits = s_bits[--i];
    if (bits) {
        mask = s_mask[i];
        left = s_left[i];
        rigt = s_rigt[i];
        goto PROC3;
    }
    if (i > posa) goto BACK3;
    goto BACK2;

FINISH:
    *uniq = cnt8     + cnt4     + cnt2;
    *allc = cnt8 * 8 + cnt4 * 4 + cnt2 * 2;
}
//**********************************************
// First queen is in the corner
//**********************************************
void Corner(int n, int x1, i64 *uniq, i64 *allc)
{
    int  size, last, y, i;
    int  bits, bit, mask, left, rigt;
    int  posa;
    int  board[MAXSIZE];
    int  s_mask[MAXSIZE];
    int  s_left[MAXSIZE];
    int  s_rigt[MAXSIZE];
    int  s_bits[MAXSIZE];
    i64  cnt8;

    // Initialize
    size = n;
    last = n - 1;
    mask = (1 << n) - 1;
    cnt8 = 0;

    // ControlValue
    posa = x1;

    // y=0: 000000001 (static)
    // y=1: 011111100 (select)
    board[0] = 1;
    board[1] = bit = 1 << x1;
    mask = mask ^ (1 | bit);
    left = 1 << 2 | bit << 1;
    rigt = 1 >> 2 | bit >> 1;
    y = i = 2;

    // y -> posa
    mask = mask ^ 2;
NEXT1:
    if (i == posa) {
        mask |= 2;
        goto NEXT2;
    }
    bits = mask & ~(left | rigt);
    if (bits) {
        s_mask[i] = mask;
        s_left[i] = left;
        s_rigt[i] = rigt;
PROC1:
        bits ^= bit = -bits & bits;
        board[i] = bit;
        s_bits[i++] = bits;
        mask = mask ^ bit;
        left = (left | bit) << 1;
        rigt = (rigt | bit) >> 1;
        goto NEXT1;
BACK1:
        bits = s_bits[--i];
        if (bits) {
            mask = s_mask[i];
            left = s_left[i];
            rigt = s_rigt[i];
            goto PROC1;
        }
    }
    if (i == y) goto FINISH;
    goto BACK1;

    // posa -> last
NEXT2:
    bits = mask & ~(left | rigt);
    if (bits) {
        if (i == last) {
            board[i] = bits;
            cnt8++;
            //Display(size, board);
            goto BACK2;
        }
        s_mask[i] = mask;
        s_left[i] = left;
        s_rigt[i] = rigt;
PROC2:
        bits ^= bit = -bits & bits;
        board[i] = bit;
        s_bits[i++] = bits;
        mask = mask ^ bit;
        left = (left | bit) << 1;
        rigt = (rigt | bit) >> 1;
        goto NEXT2;
BACK2:
        bits = s_bits[--i];
        if (bits) {
            mask = s_mask[i];
            left = s_left[i];
            rigt = s_rigt[i];
            goto PROC2;
        }
    }
    if (i == y) goto FINISH;
    if (i > posa) goto BACK2;
    goto BACK1;

FINISH:
    *uniq = cnt8;
    *allc = cnt8 * 8;
}
//**********************************************
// Search of N-Queens
//**********************************************
void NQueens(int n, i64 *unique, i64 *allcnt)
{
    int  x0, x1;
    i64  uniq, allc;

    *unique = *allcnt = 0;

    for (x0=0; x0<n/2; x0++) {
        for (x1=0; x1<n; x1++) {
            if (x0 == 0) {
                // y=0: 000000001 (static)
                // y=1: 011111100 (select)
                if (x1<2 || x1==n-1) continue;
                Corner(n, x1, &uniq, &allc);
            } else {
                // y=0: 000001110 (select)
                // y=1: 111111111 (select)
                if (x1>=x0-1 && x1<=x0+1) continue;
                if (x0>1 && (x1==0 || x1==n-1)) continue;
                Inside(n, x0, x1, &uniq, &allc);
            }
            *unique += uniq;
            *allcnt += allc;
        }
    }
}
//**********************************************
// Format of Used Time
//**********************************************
void TimeFormat(clock_t utime, char *form)
{
    int  dd, hh, mm;
    double ftime, ss;

    ftime = (double)utime / CLOCKS_PER_SEC;

    mm = (int)ftime / 60;
    ss = ftime - (double)(mm * 60);
    dd = mm / (24 * 60);
    mm = mm % (24 * 60);
    hh = mm / 60;
    mm = mm % 60;

    if (dd) sprintf(form, "%4d %02d:%02d:%05.2f", dd, hh, mm, ss);
    else if (hh) sprintf(form, "     %2d:%02d:%05.2f", hh, mm, ss);
    else if (mm) sprintf(form, "        %2d:%05.2f", mm, ss);
    else sprintf(form, "           %5.2f", ss);
}
//**********************************************
// N-Queens Solutions MAIN
//**********************************************
int main(void)
{
    int  n;
    i64  unique, allcnt;
    clock_t starttime;
    char form[20], line[100];

    printf("<------  N-Queens Solutions  -----> <---- time ---->\n");
    printf(" N:           Total          Unique days hh:mm:ss.--\n");
    for (n=MINSIZE; n<=MAXSIZE; n++) {
        starttime = clock();
        NQueens(n, &unique, &allcnt);
        TimeFormat(clock() - starttime, form);
        // sprintf(line, "%2d:%16I64d%16I64d %s\n", n, allcnt, unique, form);
        sprintf(line, "%2d:%16d%16d %s\n", n, allcnt, unique, form);
        printf("%s", line);
    }

    return 0;
}
