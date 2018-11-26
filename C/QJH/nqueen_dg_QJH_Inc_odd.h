
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

	char    dmy;
	char    I;
	char    J;
	char    M;

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
	int     hkey;
	int     jkey;
} HPARTS2;

typedef struct {
	int     jkey;
} PARTSJ;


void     Solver();

void     init(int N);
void     clear(int N);

int      Filter(int Q,int I,int J);

//int      FrameCheck(int A,int B,int C,int D);
//int      SymCheck(short pa,short pb,short pc,short pd);
int      SymCheck(HPARTS *abdtp,HPARTS *cddtp);

void     QJoin(int M);
void	 QJoin2(int M,int I,int J);
void     HJoin();
void     HJoin2();

//int		 QCache(int hflg,int A,int c,PARTSJ *QIXP[],PARTS **Qdtp);
//int      QCache2(int A,int c,int UM,PARTS *IXP[],PARTSJ *QIXP[],PARTSJ **QJdtp,PARTS **Qdtp);
int      QSUB(int BSet[],int XM,int YM,int UM,int VM,
              PARTS *IXP[],PARTSJ *QIXP[],PARTSJ **QJdtp,PARTS **Qdtp);

void	 ABJoin(int A,int C,HPARTS **Dtp);
void	 ABJoin2(int A,int C,int I,int J);
void     CDJoin(int A,int C,HPARTS **Dtp);
void	 CDJoin2(int A,int C,int I,int J);
void     setHalf(int stage,PARTS *cdtp,PARTS *ddtp);
void     setHalf2(int stage,PARTS *cdtp,PARTS *ddtp);

void     PartsGen(int N,int M,int *IXC,PARTS *IXP[],PARTS **Dtp);
void     nq(int nest,int K,int X,int Y,int U,int V,int N,int M);

int      QSymCheck(int X,int Y) ;
void     QStore(int Qsym,int X,int Y,int U,int V);
u64      Qpos(int t);

int      bitpos(unsigned int x);
int      bitcount(int x);

void     BitSel(int nest,int bit,int Bits,int M,int BSset[]);
void     setFRI(int BSset[],int *FRI);

void     printNQ(char ID[],int Pos[],int N);
void     printNQ2(char ID[],u64 POS,int N);
void     printAns(char ID[],u64 A,u64 B,u64 C,u64 D,int N);

void     dumpAns(char ID[],PARTS *pa,PARTS *pb,PARTS *pc,PARTS *pd);
void     dumpAns2(char ID[],HPARTS *abparts,HPARTS *cdparts);
void     dumpHparts(char ID[],HPARTS *hparts);
void     dumpParts(PARTS *parts);

int      AnsCheck(char ID[],HPARTS *abdtp,HPARTS *cddtp);
void     setX(char board[32][32],int X,int Y,int mask);
void     setY(char board[32][32],int X,int Y,int mask);
void     setQ(char board[32][32],int X,int Y);
void     printBoard(char ID[],char board[32][32]);


  time_t    t1,t2;
  FILE      *logid;
  int       result;
  char      str[128];
  int       tc;

  u64       Ans;
  u64       Cnt;
  
  int       N,N2,N3,M,M2,M3;
  
  int       Stage;
  u64       Acnt,Bcnt,Ccnt,Dcnt,ABcnt,CDcnt;
  u64       TABcnt,TCDcnt;
  u64       Mmcnt,Mjcnt,Mncnt;
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
//  int       IXbitsize;
  int       IXsize;

  int       *MmIXC;
  PARTS     **MmIXP;
  
  int       MmDtlen;
  PARTS     *MmDtp;

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
  
// QSUB parts
  PARTS     *Adtp;
  PARTS     *Bdtp;
  PARTS     *Cdtp;
  PARTS     *Ddtp;

  PARTSJ    *AJdtp;
  PARTSJ    *BJdtp;
  PARTSJ    *CJdtp;
  PARTSJ    *DJdtp;

//  PARTS     **AIXP;
//  PARTS     **BIXP;
//  PARTS     **CIXP;
//  PARTS     **DIXP;
  PARTSJ    **AIXP;
  PARTSJ    **BIXP;
  PARTSJ    **CIXP;
  PARTSJ    **DIXP;
  
 // Half Join Cache

//  int       BIXlen;
//  PARTSJ    **BJIXP;
//  int       Bdtlen;
//  PARTSJ    *BJdtp;

//  int       DIXlen;
//  PARTSJ    **DJIXP;
//  int       Ddtlen;
//  PARTSJ    *DJdtp;

// Half parts pool (C-D)
  int       UIXlen;
  int       *Uixc;
  PARTSJ    **Uixp;
  PARTSJ    *Ulp;
  HPARTS    *CDdtp;

  HPARTS    *ABdtp;

  PARTSJ    **Uixp2;
  PARTSJ    *Ulp2;
//  HPARTS2   *ABdtp2;
//  HPARTS2   *CDdtp2;

  int       DtCnt;
  int       Dtlen;
  int       *Tixc;
  HPARTS    *Tdtp;

  int       DtCnt2;
  int       Dtlen2;
  HPARTS2   *Tdtp2;
  
  PARTS     parts;
  FRAME     frame;


  int       MmSet[1000],MjSet[1000],MnSet[1000];
//  int       BScnt,MmScnt,MjScnt,MnScnt;
  int       *FRI,*FRIm,*FRIj,*FRIn,FRIMAX;

  int       Vcnt,Ucnt;
