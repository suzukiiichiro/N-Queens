// nqueen_dg.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//
/*-----------------------------------*/
/*  copyright deepgreen  2011.07.31  */
/*                                   */
/*  author    deepgreen  2010.07.07  */
/*-----------------------------------*/

// #include "stdafx.h"


#include "time.h"
#include "stdio.h"
#include "stddef.h"
#include "stdlib.h"
#include "string.h"

//#define  u64  unsigned __int64
#define  u64  unsigned long long
#define  u32  unsigned int
#define  u16  unsigned short
#define  BITRVS(x)  (rBit[x>>N] | (rBit[x&MASK]<<N))

typedef struct {
    int     A;
    int     B;
    int     C;
    int     D;
    int     a;
    int     b;
    int     c;
    int     d;
	int     ix[4];
	char    flag[4];
    char    sym;
    char    dup;
    char    N2M;
    char    fsym;

}  FRAME;

typedef struct {
	int     x;
	int     y;
	int     u;
	int     v;
	int     p;
}  QFRAME;

typedef struct {
    int     u;
	short   v;
	short   dmy;
	short   p;
	short   r;
} PARTS;

typedef struct {
	short   x;
	short   y;
	short   x2;
	short   sym;
	int     hkey;
	int     jkey;
	short   p;
	short   q;
	short   r;
} HPARTS;

typedef struct {
	int     jkey;
} PARTSJ;


void     Solver();

void     init(int N);
void     clear(int N);

//int      FrameCheck(int A,int B,int C,int D);
//int      SymCheck(short pa,short pb,short pc,short pd);
int      SymCheck(HPARTS *abdtp,HPARTS *cddtp);

void     QJoin(int M);
void     HJoin();

int		 QCache(int A,int c,PARTSJ *QIXP[],PARTSJ **Qdtp);

void	 ABJoin(int A,int C,HPARTS **Dtp);
void     CDJoin(int A,int C,HPARTS **Dtp);
void     setHalf(int stage,PARTS *cdtp,PARTS *ddtp);

void     PartsGen(int N,int M,int *IXC,PARTS *IXP[],PARTS **Dtp);
void     nq(int nest,int K,int X,int Y,int U,int V,int N,int M);

int      QSymCheck(int X,int Y) ;
void     QStore(int Qsym,int X,int Y,int U,int V);
u64      Qpos(int t);

int      bitpos(unsigned int x);
int      bitcount(int x);

void     BitSel(int nest,int bit,int Bits,int M,int BSset[]);

void     printNQ(char ID[],int Pos[],int N);
void     printNQ2(char ID[],u64 POS,int N);
void     printAns(char ID[],u64 A,u64 B,u64 C,u64 D,int N);

void     dumpAns(char ID[],PARTS *pa,PARTS *pb,PARTS *pc,PARTS *pd);


  time_t    t1,t2;
  FILE      *logid;
  int       result;
  char      str[128];
  int       tc;

  u64       Ans;
  u64       Cnt;
  
  int       N,N2,M,M2;
  
  int       Stage;
  u64       Acnt,Bcnt,Ccnt,Dcnt,ABcnt,CDcnt;
  u64       TABcnt,TCDcnt;
  u64       Mjcnt,Mncnt;
  u64       hit,Hit;

  u64       U,V,bit64[64];

  int       Pos[32];
  int       Bit[32];
  int       Bit2[32];
  int       rBit[2<<12];  /* reverse Bit table */
  char      Bitpos[256];
  int       MASK;         /* N  mask */
  int       MASK2;        /* NN mask */
  int       MASKU;        /* U  mask */
  int       MASKOFF;      /* N0 mask */

  int       MASKJ;        /* join mask */

  int       MASKJA,MASKJB;
  int       MASKJ2;
  int       JAsize,JBsize,JIXsize;
  int       JAXsize,JBXsize;



//  JCACHE    *JCp;
//  JCACHE    **JCixp;

  int       IXlen;
  int       IXbitsize;

  int       *MjIXC;
  PARTS     **MjIXP;
  
  int       MjDtlen;
  PARTS     *MjDtp;

  int       *MnIXC;
  PARTS     **MnIXP;
  
  int       MnDtlen;
  PARTS     *MnDtp;
  
  int		*XYIXC;
  PARTS		**XYIXP;
  PARTS		*XYDtp;
  
// Half Join Cache

  int       BIXlen;
  PARTSJ    **BIXP;
  int       Bdtlen;
  PARTSJ    *Bdtp;

  int       DIXlen;
  PARTSJ    **DIXP;
  int       Ddtlen;
  PARTSJ    *Ddtp;

// Half parts pool (C-D)
  int       UIXlen;
  int       *Uixc;
  PARTSJ    **Uixp;
  PARTSJ    *Ulp;
  HPARTS    *CDdtp;

  HPARTS    *ABdtp;

  int       DtCnt;
  int       Dtlen;
  int       *Tixc;
  HPARTS    *Tdtp;
  
// preJoin Table
//  u16       **BJX;
//  u16       *BJT;

  PARTS     parts;
  FRAME     frame;

//  PARTS     JAB[1000],JAC[1000],JAD[1000];
//  PARTS     JACB[1000],JACD[1000];

  int       MjSet[1000];
  int       BScnt;

  int       Vcnt,Ucnt;

