
#include "nqueen_dg_QJH_inc_even.h"
#include <stdio.h>
#include <stdlib.h>

/* malloc監視 */
void *MyMalloc_Rep( size_t sz, const char *pcFileName, int nLine )
{
void *ptr;
ptr = malloc(sz);
//printf(stderr, "malloc at %s %-4d : %p - %p, %ld byte", pcFileName, nLine, ptr, ptr+sz, sz);
fprintf(stderr, "malloc at %s %-4d : %p %ld byte\n", pcFileName, nLine, ptr, sz);
return ptr;
}

/* free監視 */
void MyFree_Rep( void *ptr, const char *pcFileName, int nLine )
{
fprintf(stderr, "freeing at %s %-4d : %p\n", pcFileName, nLine, ptr);
free(ptr);
}

// 以下のマクロを定義しておく 
#define Malloc(S) MyMalloc_Rep( (S), __FILE__, __LINE__ )
#define Free(P) MyFree_Rep( (P), __FILE__, __LINE__ )

//int _tmain(int argc, _TCHAR* argv[])
int main() {
  char *stime[30],*etime[30];
  u64  tick1,tick2;

  tick1 = clock();
  logid = fopen("log.txt","wt"); 
  time(&t1);
  *stime = ctime(&t1);
  fprintf(logid,"*** pazul started : %s \n",*stime);
  // for (N=2; N<=10; N++) {
  for (N=8; N<=8; N++) {
    Solver();
  }
  tick2 = clock();
  result = (int) (tick2 -tick1);
  time(&t2);
  *etime = ctime(&t2);
  fprintf(logid,"\n*** pazul  ended  time = %d ms  : %s \n",result,*etime);
  fclose(logid);
  return 0;
}
//-------------------------------------------------------------------------
void Solver()
{
 u64  tick1,tick2,t1,t2;
 u64  Tans,Tcnt,Mans,Mcnt;
 u64  TAcnt,TBcnt,TCcnt,TDcnt;
 u64  Mjloop,Mnloop;
 int  etime;
 int  i,j,k,l,n;
 int  A,a,B,b,C,c,D,d;
 int  ia,ib,ic,id; 
//                 0  1  2  3  4    5   6    7   8     9   10     11     12      
 u64  AnsAll[] = { 0L,0L,0L,0L,2L, 10L, 4L, 40L, 92L, 352L, 724L, 2680L, 14200L,
//	   13       14       15         16        17          18         19           20
      73712L, 365596L, 2279184L, 14772512L, 95815104L, 666090624L, 4968057848L, 39029188884L,
//      21              22              23               24   
	  314666222712L, 2691008701644L,  24233937684440L, 227514171973736L,
//      25                 26	  
	  2207893435808352L, 22317699616364044L,  };
//                 0  1  2  3  4    5   6    7   8     9   10     11     12      
 u64  AnsUnq[] = { 0L,0L,0L,0L,1L,  2L, 1L,  6L, 12L,  46L,  92L,  341L,  1787L,
//	   13       14       15         16        17          18         19           20
	  9233L,  45752L,  285053L,  1846955L,  11977939L,  83263591L, 621012754L,  4878666808L,
//      21            22            23              24
	  39333324973L, 336376244042, 3029242658210L, 28439272956934L,
//      25                26
      275986683743434L, 2789712466510289L,    };
 
// int  t;
	 tick1 = clock();
	 N2 = N * 2;
 
init(N);


     Tans = 0;
	 Tcnt = 0;
//	 for (M=(N+1)/2; M<N; M++) {
	 for (M=1; M<=N/2; M++) {
//		 if (M != 4)   continue;
         t1 = clock();
	     Mans = 0;
	     Mcnt = 0;
	     
		 M2 = N - M;
		 frame.N2M = (M == M2)?  1 : 0;

		 PartsGen(N,M ,MjIXC,MjIXP,&MjDtp);
		 Mjcnt = Cnt;


		 PartsGen(N,M2,MnIXC,MnIXP,&MnDtp);
		 Mncnt = Cnt;
         hit = 0;
         Hit = 0;
	             
		 QJoin(M);
		 Tans += hit;
     Tcnt += Hit;
		 Mans += hit;
		 Mcnt += Hit;

		 //free(MjDtp);
		 //free(MnDtp);
		 Free(MjDtp);
		 Free(MnDtp);
		
		 t2 = clock();
     etime = (int)(t2-t1);
		 fprintf(logid,"----- M=%2d   %12lld   %12lld    %7d        Mj : %9lld  Mn : %9lld  /   ABcnt : %9lld   CDcnt : %9lld\n", M,Mans,Mcnt,etime,Mjcnt,Mncnt,TABcnt,TCDcnt);
     fflush(logid);
	 }
   clear(N);
	 tick2 = clock();
   etime = (int)(tick2-tick1);
	 fprintf(logid,"\n***** N=%2d   %14lld  %14lld   %10d \n\n\n",N2,Tans,Tcnt,etime);
	 if ((Tans != AnsUnq[N2]) || (Tcnt != AnsAll[N2])) 
		 fprintf(logid,"?????? Ans Error  %14lld  %14lld\n",AnsUnq[N2],AnsAll[N2]);
	 fflush(logid);
}
//-------------------------------------------------------------------------
void init(int N)
{
 int k,n,w,r,pJXlen;
 u64 one=1;
 for (k=0; k<32; k++)  Bit[k] = 1 << k; 
 for (k=0; k<N; k++)   Bit2[k] = 1 << (N-k-1);
 for (k=0; k<64; k++)  bit64[k] = one << k;
   
	memset(Bitpos,0x0f,sizeof(Bitpos));
	for (k=0; k<8; k++)  Bitpos[Bit[k]] = k;
	for (k=0; k<Bit[N]; k++) {
		r = 0;
		w = k;
		for (n=0; n<N; n++)  { 
			r = (r<<1) + (w & 0x01);
			w = w >> 1;		
		}  
		rBit[k] = r;
	}
    
	MASK  = Bit[N] - 1;
	MASK2 = Bit[N2] - 1;
//	MASKU = Bit[N2-1] - 1;
	MASKOFF = MASK << N;
// hash size	
	if (N < 6)  { 
//	   JAsize = N-1; JBsize = N-1;
	   JAsize = 1; JBsize = 1;
	} 
	else if (N <10) { 
//	   JAsize = N-3; JBsize = N-4;
//	   JAsize = 3; JBsize = 3;
	   JAsize = 4; JBsize = 3;
	}
	else {
//	   JAsize = N-3; JBsize = N-3;
	   JAsize = 4; JBsize = 2;
	}
//	JIXsize = JAsize + JBsize;
	JIXsize = N2 - 1 - (JAsize + JBsize);
	MASKJA = Bit[JAsize] - 1;
    MASKJB = Bit[JBsize] - 1;
	MASKJ2=  Bit[JIXsize] - 1;
	MASKJ = Bit[N-1] - 1;
// PARTS pool Index
	IXbitsize = N2;
    IXlen = (Bit[IXbitsize] + 1) * sizeof(int);
/*
	MjIXC = (int*) malloc(IXlen);
	MjIXP = (PARTS**) malloc(IXlen);
	MnIXC = (int*) malloc(IXlen);
	MnIXP = (PARTS**) malloc(IXlen);
// A-B Join Cashe
	BIXlen = (Bit[N]+1) * sizeof(PARTSJ*);
	BIXP = (PARTSJ**) malloc(BIXlen);
// C-D Join Cache
	DIXlen = (Bit[N]+1) * sizeof(PARTSJ*);
	DIXP = (PARTSJ**) malloc(DIXlen);
// Half(AB-CD) join Cache
	UIXlen= (Bit[JIXsize]+1) * sizeof(int);
	Uixc  = (int*) malloc(UIXlen);
	Uixp  = (PARTSJ**) malloc(UIXlen);
	DtCnt = 1 << N;
	Dtlen = DtCnt * sizeof(HPARTS);
	Tdtp = (HPARTS*) malloc(Dtlen);
*/

	MjIXC = (int*) Malloc(IXlen);
	MjIXP = (PARTS**) Malloc(IXlen);
	MnIXC = (int*) Malloc(IXlen);
	MnIXP = (PARTS**) Malloc(IXlen);
// A-B Join Cashe
	BIXlen = (Bit[N]+1) * sizeof(PARTSJ*);
	BIXP = (PARTSJ**) Malloc(BIXlen);
// C-D Join Cache
	DIXlen = (Bit[N]+1) * sizeof(PARTSJ*);
	DIXP = (PARTSJ**) Malloc(DIXlen);
// Half(AB-CD) join Cache
	UIXlen= (Bit[JIXsize]+1) * sizeof(int);
	Uixc  = (int*) Malloc(UIXlen);
	Uixp  = (PARTSJ**) Malloc(UIXlen);
	DtCnt = 1 << N;
	Dtlen = DtCnt * sizeof(HPARTS);
	Tdtp = (HPARTS*) Malloc(Dtlen);

}
//-------------------------------------------------------------------------
void clear(int N)
{
/*
	free(MjIXC);
	free(MjIXP);
	free(MnIXC);
	free(MnIXP);
	free(Uixc);
	free(Uixp);
	
	free(BIXP);
	free(DIXP);
	free(Tdtp);
*/

	Free(MjIXC);
	Free(MjIXP);
	Free(MnIXC);
	Free(MnIXP);
	Free(Uixc);
	Free(Uixp);
	
	Free(BIXP);
	Free(DIXP);
	Free(Tdtp);
}
//-------------------------------------------------------------------------
int  QCache(int A,int c,PARTSJ *QIXP[],PARTSJ **Qdtp)
{
 int QClen,x,X,xy;
 PARTSJ *qdtp;
 PARTS  *dtp;
 
	 QClen = MnIXP[(c+1)<<N] - MnIXP[c<<N];
	 if (QClen == 0)   return -1;
	 //*Qdtp = (PARTSJ*) malloc(QClen*sizeof(PARTSJ));
	 *Qdtp = (PARTSJ*) Malloc(QClen*sizeof(PARTSJ));
	 
	 qdtp  = *Qdtp;
	 
	 for (x=0; x<Bit[N]; x++) {
	 	 QIXP[x] = qdtp;
		 X = rBit[x];
	 	 if ((X ^ MASK) < frame.A)            continue;
		 if ((frame.N2M) && (X < frame.A))    continue;
	 	 xy = (x << N) + c;
	 	 for (dtp=MnIXP[xy]; dtp<MnIXP[xy+1]; dtp++) {
	 	 	 qdtp->jkey = (dtp->v << N) + (dtp->u & MASKJ);  
	 	 	 qdtp++;
	 	 }
	 }
	 return 0;
}
//-------------------------------------------------------------------------
void QJoin(int M)
{
 int ia,ib,ic,id,xy,xya,xyb,xyc,xyd;
 int A,C;
	 BScnt= 0;
     BitSel(0,0,MASK,M,MjSet);
     Mjcnt = BScnt;
	 TABcnt = 0;
	 TCDcnt = 0;
	 for (ia=0; ia<Mjcnt; ia++) {
	   
		 frame.A = MjSet[ia];
		 frame.a = frame.A ^ MASK;
		 if (frame.N2M) {
	   		 if (frame.A > frame.a)                   continue;    //5. Y check
		 }
	//setup D-cache
	   	 A = rBit[frame.A];
		 if (QCache(A,A^MASK,DIXP,&Ddtp) < 0)         continue;
	   	 
		 for (ic=0; ic<Mjcnt; ic++) {
	     
			 frame.C = MjSet[ic];
			 frame.c = frame.C ^ MASK;
			 frame.sym = 0;
			 if (frame.A >  frame.C)                              continue;
			 if (frame.A == frame.C)        frame.sym |= 0x04;   // 3. R180 check  
			 if (frame.N2M) {
			 	 if (frame.A >  frame.c)                          continue;
			 	 if (frame.A == frame.c)    frame.sym |= 0x40;   // 4. X check
			 }
	//setup B-cache
			 C = rBit[frame.C];
		   	 if (QCache(A,C^MASK,BIXP,&Bdtp) < 0)     continue;
			 ABJoin(A,C,&ABdtp);
			 ABcnt   = Cnt;
			 TABcnt += ABcnt;
			 
			 CDJoin(A,C,&CDdtp);
			 CDcnt   = Cnt;
			 TCDcnt += CDcnt;
			 
			 if ((ABcnt > 0) && (CDcnt > 0)) {
				 HJoin();
			 }
			 free(ABdtp);
			 free(CDdtp);
			 free(Ulp);
			 free(Bdtp);
		 }
		 free(Ddtp);
	 }
}
//-------------------------------------------------------------------------
void HJoin()
{
 int    abix,cdix,dup,cdinc,cdinc2,cdmask,jkey;
 HPARTS *abdtp;
 HPARTS *cddtp;
 PARTSJ *ucp;
	 for (abdtp=ABdtp; abdtp<ABdtp+ABcnt; abdtp++) {
		 abix = abdtp->hkey;
		 jkey = abdtp->jkey;
		 cdinc= abix & (-abix);
//		 if (cdinc == 0)  cdinc = MASKJA + 1;  // set any nonzero value
		 if (cdinc == 0)  cdinc = MASKJ2 + 1;  // set any nonzero value
		 cdinc2 = abix + cdinc;
		 cdmask = MASK2 ^ abix;
		 cdix = 0;
         while(cdix <= MASKJ2) {
			 
	//		 fprintf(logid,"      abix=%x  cdix=%x cdinc=%x\n",abix,cdix,cdinc);
	//		 fflush(logid);
	 		 for (ucp=Uixp[cdix]; ucp<Uixp[cdix+cdinc]; ucp++) {
				 if (jkey & ucp->jkey)   continue;
				 cddtp = CDdtp + (ucp-Ulp);
		//			 tc++;
				 dup = SymCheck(abdtp,cddtp);
				 if (dup <= 0)                        continue;
		//			 sprintf(str,"--- No. %4d   %7d   sym:%d ---",(int)tc,(int)Ans,sym);
		//			 printAns(str,abdtp->pos,abdtp->pos2,cddtp->pos,cddtp->pos2,N);
//skip:
				 hit++;
                 Hit += dup;
             }
		     cdix = (cdix + cdinc2) & cdmask;
	     }
	     
	 }
}
//-------------------------------------------------------------------------
void ABJoin(int A,int C,HPARTS **Dtp)
{
 int b,c,B,xya,xyb;
 int jkey,ix,k,dtlen,ullen,sym;
 PARTS  *adtp,*bdtp;
 PARTSJ *qbdtp,*ulp;
 HPARTS *hparts,*tdtp,*dtp;
	 Cnt = 0;
	 hparts = Tdtp;
	 memset(Uixc,0,UIXlen);
	 
	 c = C ^ MASK;
	 sym = frame.sym;
	 for (B=0; B<Bit[N]; B++) {
		 frame.sym = sym;                   //reset sym
		 frame.B = rBit[B];
		 if (frame.A >  frame.B)                         continue;   // 1. V check 
		 if (frame.A == frame.B)             frame.sym |= 0x01;
		 if (frame.N2M) {
			 frame.b = frame.B ^ MASK;
			 if ((frame.A == frame.c) &&                             // 4. X check
	    		 (frame.B >  frame.b))                   continue; 
			 if (frame.A >  frame.b)                     continue;   // 6. R+90 check
			 if (frame.A == frame.b) {
				 if (frame.B >  frame.c)                 continue;
				 if (frame.B == frame.c)	 frame.sym |= 0x10;
			 }
		 }
		 b = B ^ MASK;
		 xya = (A << N) + B;
		 xyb = (b << N) + c;
		 for (adtp= MjIXP[xya]; adtp<MjIXP[xya+1]; adtp++) {
		 
			 jkey = ((rBit[adtp->u>>N]) << (N-1)) + adtp->v;
		 
			 bdtp = MnIXP[xyb] - 1;
			 for (qbdtp= BIXP[b]; qbdtp<BIXP[b+1]; qbdtp++) {
				 bdtp++;
				 if (jkey & qbdtp->jkey)     continue;
				 setHalf(1,adtp,bdtp);
			 }
		 }
	 }
	 dtlen = Cnt*sizeof(HPARTS);
     //*Dtp  = (HPARTS*) malloc(dtlen);
     *Dtp  = (HPARTS*) Malloc(dtlen);
     memset(*Dtp,0xff,dtlen);
     tdtp  = Tdtp;
         
	 Uixp[0] = 0;
	 for (k=1; k<=Bit[JIXsize]; k++)  Uixp[k] = Uixp[k-1] + Uixc[k-1]; 
	 memset(Uixc,0,UIXlen);
	 for (k=0; k<Cnt; k++) {
		 ix  = tdtp->hkey;
         ulp = Uixp[ix] + Uixc[ix];
		 dtp = *Dtp + (ulp-Uixp[0]);            //relocate to CDdtp
         memcpy(dtp,tdtp,sizeof(HPARTS));
         Uixc[ix]++;
         tdtp++;
     }
}
//-------------------------------------------------------------------------
void CDJoin(int A,int C,HPARTS **Dtp)
{
 int a,d,D,xyc,xyd;
 int jkey,ix,k,dtlen,ullen,sym;
 PARTS  *cdtp,*ddtp;
 PARTSJ *qddtp,*ulp;
 HPARTS *hparts,*tdtp,*dtp;
	 Cnt = 0;
	 hparts = Tdtp;
	 memset(Uixc,0,UIXlen);
	 
	 a = A ^ MASK;
	 sym = frame.sym;
	 
	 for (D=0; D<Bit[N]; D++) {
		 frame.sym = sym;                     // reset sym
		 frame.D = rBit[D];
		 if (frame.A >  frame.D)                         continue;   // 2. U check
		 if (frame.A == frame.D)               frame.sym |= 0x02;
		 if (frame.N2M) {
			 frame.d = frame.D ^ MASK;
			 if (frame.A >  frame.d)                     continue;   // 7. R-90 check
		     if (frame.A == frame.d)           frame.sym |= 0x20;
		 }
		 d = D ^ MASK;
		 xyc = (C << N) + D;
		 xyd = (d << N) + a;
		 for (cdtp= MjIXP[xyc]; cdtp<MjIXP[xyc+1]; cdtp++) {
		 
			 jkey = ((rBit[cdtp->u>>N]) << (N-1)) + cdtp->v;
		 
			 ddtp = MnIXP[xyd] - 1;
			 for (qddtp= DIXP[d]; qddtp<DIXP[d+1]; qddtp++) {
				 ddtp++;
				 if (jkey & qddtp->jkey)     continue;
				 setHalf(2,cdtp,ddtp);
			 }
		 }
	 }
	 dtlen = Cnt*sizeof(HPARTS);
     *Dtp  = (HPARTS*) Malloc(dtlen);
     memset(*Dtp,0xff,dtlen);
     tdtp  = Tdtp;
     ullen = Cnt*sizeof(PARTSJ);
     Ulp   = (PARTSJ*) Malloc(ullen);
     memset(Ulp,0xff,ullen);
         
     Uixp[0] = Ulp;
	 for (k=1; k<=Bit[JIXsize]; k++)  Uixp[k] = Uixp[k-1] + Uixc[k-1]; 
	 memset(Uixc,0,UIXlen);
	 for (k=0; k<Cnt; k++) {
		 ix  = tdtp->hkey;
         ulp = Uixp[ix] + Uixc[ix];
		 ulp->jkey = tdtp->jkey;
		 dtp = *Dtp + (ulp-Ulp);            //relocate to CDdtp
         memcpy(dtp,tdtp,sizeof(HPARTS));
         Uixc[ix]++;
         tdtp++;
     }
}
//-------------------------------------------------------------------------
void setHalf(int stage,PARTS *p,PARTS *q)
{
 HPARTS *hparts,*dtp;
 int    sym,u,u2,h1,h2,hkey,jkey;
	 if (Cnt >= DtCnt) {
		 dtp = (HPARTS*)Malloc(Dtlen*2);
		 memcpy(dtp,Tdtp,Dtlen);
		 free(Tdtp);
		 Tdtp = dtp;
		 DtCnt *= 2;
		 Dtlen *= 2;
	 }
	 hparts = Tdtp + Cnt;
//change pos-info 2010.05.07
	 hparts->p  = p->p;
	 hparts->q  = q->p;
	 hparts->r  = p->r;       
	 hparts->sym= frame.sym;
	 u  = p->u | (rBit[q->v] << (N-1));
	 u2 = q->u | p->v;
	 if (stage == 1) {
		 hparts->x  = frame.A;
		 hparts->y  = frame.B;
		 hparts->x2 = frame.C ^ MASK;
		    
		 u2 = BITRVS(u2) >> 1;
	 }
	 else {
		 hparts->x  = frame.C;
		 hparts->y  = frame.D;
		 hparts->x2 = frame.A ^ MASK;
		 u = BITRVS(u) >> 1;
	 }
	 
//	 h1   = (u  >> N) & MASKJA;
//	 h2   = (u2 >> N) & MASKJB;
	 
//	 hkey = h1 + (h2 << JAsize);
	 
//	 jkey = (u2 & MASK) + ((u2 >> (JBsize+N)) << N);
//	 jkey = (u  & MASK) + ((u  >> (JAsize+N)) << N) + (jkey << (N2-1-JAsize));
	 hkey = (u2 >> JAsize) & MASKJ2;
	 jkey = (u << (JAsize+JBsize)) + ((u2 & MASKJA) << JBsize) + (u2 >> (JIXsize+JAsize));
	 hparts->hkey = hkey;
	 hparts->jkey = jkey;
	 Uixc[hkey]++ ;
	 Cnt++;
}
//-------------------------------------------------------------------------
int  SymCheck(HPARTS *abdtp,HPARTS *cddtp)
{
 int   sym,dup;
 int   A,B,C,D,a,b,c,d;
 short pa,par,pb,pc,pcr,pd;
	 if ((abdtp->sym == 0) && (cddtp->sym == 0))   return 8;
     sym = 0;
 	 dup = 8;
	 A = abdtp->x;
	 B = abdtp->y;
     C = cddtp->x;
     D = cddtp->y;
// 1. V check
//	 if (A > B)                     return -11;    //for 1. V     A <= B
	 if (A == B) {
         if (C > D)                 return -12;    //             C <= D
		 if (C == D)       sym |= 0x01; 
	 }
// 2. U check
//	 if (A > D)                     return -21;    //for 2. U     A <= D
	 if (A == D) {
         if (B > C)                 return -22;    //             B <= C
		 if (B == C)       sym |= 0x02; 
	 }
// 3. R180 check
//	 if (A > C)                     return -31;    //for 3. R180  A <= C  checked at main-line
	 if (A == C) {
		 if (B > D)                 return -32;    //             B <= D
		 if (B == D)       sym |= 0x04;
	 }
     if (frame.N2M) {
//		 a = frame.a;
//		 b = frame.b;
//		 c = frame.c;
//		 d = frame.d;
		 a = A ^ MASK;
		 b = B ^ MASK;
		 c = C ^ MASK;
		 d = D ^ MASK;
		 
// 4. X check
//	     if (A > c)                     return -41;    //for 4. X     A <= c, B <= b
//  check at ABJoin
//		 if ((A == c) &&
//	    	 (B > b))                   return -42;  
// 5. Y check
//		 if (A > a)                     return -51;    //for 5. Y     A <= a  checked at main-line
// 6. R+90 check
//		 if (A > b)                     return -61;    //for 6. R+90  A <= b, B <= c
// checked at ABJoin
		 if (A == b) {
			 if (B >  c)                return -62;
		 	 if (B == c)   sym |= 0x10;
		 }
// 7. R-90 check
//		 if (A > d)                     return -71;    //for 7. R-90  A <= d
		 if (A == d) {
			 if (B > a)                 return -72;    //             B <= a
			 if (B == a)   sym |= 0x20;
	 	 }
	 }
//
	 if (sym) {
		 pa = abdtp->p;
		 pb = abdtp->q;
		 pc = cddtp->p;
		 pd = cddtp->q;
		 if (sym & 0x01) {          // V-axis  A = B & C = D
		 	 if (pa > abdtp->r)                return -111;
		 	 if ((pa == abdtp->r) &&
		 	 	 (pc >  cddtp->r))             return -112;
		 	 	 
//		     if (pa > (pa ^ 0x08))             return -111;
//		     if ((pa & 0x01) && 
//		         (pc > (pc ^ 0x08)))           return -112;
		 }
		 if (sym & 0x02) {          // U-axis A = D & B = C
		 	 if (pa > cddtp->r)                return -121;
//		     if (pa > (pc ^ 0x08))             return -121;
		 }
		 if (sym & 0x04) {          // Rot180 A = C & B = D
		     if (pa > pc)                      return -131;
		     if (pa == pc) { 
		         if (pb > pd)                  return -132;
		         if (pb == pd)      dup = 4;
			 }
		 }
// bugfix 2011.05.01
//		 if (sym & 0x10) {
//           if (pa > pb)                      return -161;
//		     if (pa == pb) { 
//		         if (pb > pc)                  return -162;
//		         if (pb == pc) {
//		             if (pc != pd)             return -163;
//		             dup = 2;
//		         }
//		     }
//		 }
//		 if (sym & 0x20) {
//           if (pa > pd)                      return -171;
//		     if (pa == pd) { 
//		         if (pd > pc)                  return -172;
//		         if ((pd == pc) && (pc != pb))      return -173;
//		         if ((pd <  pc) && (pa != pb))      return -174;
//		     }
//		 }
//newcode
		 if ((sym & 0x30) == 0)                return dup;
		 if (sym & 0x10) {
             if (pa > pb)                      return -161;
		     if (pa == pb) { 
				 sym |= 0x40;
		     }
		 }
		 if (sym & 0x20) {
             if (pa > pd)                      return -171;
		     if (pa == pd) { 
				 sym |= 0x80;
		     }
		 }
		 if ((sym & 0x30) == 0x30) {
			 switch (sym & 0xc0) {
				 case 0x00:
		//		    if (pa > pc)              return -181;
		//			if ((pa == pc) && (pb > pd))  return -182;
					break;
				 case 0x40:
                    if (pa == pc)             return -183;
					if (pd >= pc)             return -184;
					break;
				 case 0x80:
				    if (pa == pc)             return -185;
					if (pb >  pc)             return -186;
					break;
				 case 0xc0:
				    if (pa == pc)  dup = 2;
                    break;
				 default:
				    break;
			 }
		 }
 //new code end
	  }
      return dup;
}
//-------------------------------------------------------------------------
void PartsGen(int N,int M,int *IXC,PARTS *IXP[],PARTS **Dtp) {
  int k;

  XYIXC = IXC;
  XYIXP = (PARTS**)IXP;

  Stage = 1;

  memset(XYIXC,0,IXlen);
  Cnt = 0;

  nq(0,0,0,0,0,0,N,M);
  *Dtp = XYDtp = (PARTS *) Malloc(Cnt * sizeof(PARTS));

  Stage = 2;
  Cnt = 0;
  XYIXP[0] = XYDtp;
  for (k=0; k<Bit[N2]; k++)  XYIXP[k+1] = XYIXP[k] + XYIXC[k];

  memset(XYIXC,0,IXlen);
  nq(0,0,0,0,0,0,N,M);
}
//-------------------------------------------------------------------------
void nq(int nest,int K,int X,int Y,int U,int V,int N,int M)
{
int b,m,n,xy,qsym;
	 if (K == M) {
        
		for (n=nest; n<N; n++)  Pos[n] = 0;
		U = (nest == N)?  U >> 1 :  U << (N-nest-1);
//		if ((M != bitcount(U)) || (M != bitcount(V))) {
// 		    sprintf(str,"----qsym =%3d     %lld  %x %x ----",qsym,Cnt,X,Y);
//    	    printNQ(str,Pos,N);
//		}
        qsym = QSymCheck(X,Y);
//		if ((Stage == 1) || (Stage == 3)) {
//			if ((X == 0x27) && (Y == 0x27)) {
// 		    fprintf(logid,"----qsym =%3d     %lld  %x %x %x %x ----\n",qsym,Cnt,X,Y,U,V);
// 		    sprintf(str,"----qsym =%3d     %lld  %x %x %x %x   %llx  %llx----",qsym,Cnt,X,Y,U,V,Qpos(0),Qpos(-qsym));
//    	    printNQ(str,Pos,N);
//			}
//		}
		if (qsym < 0)                      return;
        QStore(qsym,X,Y,U,V);
		return;
	 }
     if ((M-K) > (N-nest))  return;
	 m  = MASK & ~(X | U | (V>>nest));
	 while(m != 0) {
		 b = m & (-m);
		 m = m ^ b;
		 Pos[nest] = b;
         nq(nest+1,K+1,X|b,Y|Bit[nest],(U|b)<<1,V|(b<<nest),N,M);
	 }
	 Pos[nest] = 0;
	 if (nest+1 < N) {
         nq(nest+1,K,X,Y,U<<1,V,N,M);
	 }
}
//-------------------------------------------------------------------------
int  QSymCheck(int X,int Y) 
{
 int k,n,t,qsym;
 u64 pos0,p,q,r,s,w;
	 qsym = 0;
     if (X >  rBit[X])                     return -11;
     if (Y >  rBit[Y])                     return -21;
     if (X >  Y)                           return -31;
	 if (X == rBit[X])  qsym |= 0x01;
	 if (Y == rBit[Y])  qsym |= 0x02;
	 if (X == Y)        qsym |= 0x04;
	 pos0 = Qpos(0);
	 for (t=1; t<8; t++) {
		 if (((t & qsym) == t) && 	
		     (pos0 > Qpos(t)))               return t-100;
	 }
	 return 0;
}
//-------------------------------------------------------------------------
void QStore(int Qsym,int X,int Y,int U,int V) 
{
int x,y,u,v,t,s,w,xy,p;
PARTS *dtp;
QFRAME Qframe[8];
	 memset(Qframe,0,sizeof(Qframe));
	 p = XYIXC[(X << N) + Y];
	 for (t=0; t<8; t++) {
		 x = X;
		 y = Y;
		 u = U;
		 v = V;
		 if (t & 0x01) {              // X rvs
			 x = rBit[x];
			 w = u;
			 u = BITRVS(v) >> 1;
			 v = BITRVS(w) >> 1;
		 }
		 if (t & 0x02) {              // Y rvs
			 y = rBit[y];
			 w = u;
			 u = v;
			 v = w;
		 }
		 if (t & 0x04) {              // X <-> Y
			 w = y;
			 y = x;
			 x = w;
			 u = BITRVS(u) >> 1;
		 }
		 Qframe[t].x = x;
		 Qframe[t].y = y;
		 Qframe[t].u = u;
		 Qframe[t].v = v;
		 Qframe[t].p = t;
		 for (s=0; s<t; s++) {
		 	 if (Qframe[s].x != Qframe[t].x)   continue;
		 	 if (Qframe[s].y != Qframe[t].y)   continue;
		 	 if (Qframe[s].u != Qframe[t].u)   continue;
		 	 if (Qframe[s].v != Qframe[t].v)   continue;
			 if (Qpos(s)     != Qpos(t))       continue;
			 Qframe[t].p = Qframe[s].p;
			 break;
		 }
	 }	 
// duplicate check
	 for (t=0; t<8; t++) {
		 if (Qframe[t].p != t)                 continue;
	 	 Cnt++;
		 xy = (Qframe[t].x << N) + Qframe[t].y;
		 switch (Stage) {
			case 1:
	           XYIXC[xy]++;
			   break;
			case 2:
               dtp = XYIXP[xy] + XYIXC[xy];
			   dtp->u = Qframe[t].u;
			   dtp->v = Qframe[t].v >> N;
			   dtp->p = (p << 4) + Qframe[t].p;
			   dtp->r = (p << 4) + Qframe[t^0x04].p;
			   XYIXC[xy]++;
               break;
			default:
               break;
		 }
	 }
}
//-------------------------------------------------------------------------
u64  Qpos(int t)
{
 int k,s,x,y;
 u64 q,w;
 
	 q = 1;
	 q = (q << (N * 4)) - 1;
	 for (k=0; k<N; k++) {
		 x = bitpos(Pos[k]);
	 	 y = k;
		 if ((t & 0x01) && (x != 0x0f))  x = N - 1- x;
	 	 if ((t & 0x02) && (y != 0x0f))  y = N - 1- y;
	 	 if  (t & 0x04)      { s = x; x = y; y = s; }
	 	 
	 	 if (y == 0x0f)   continue;
	 	 
	     w = 0x0f ^ x;
	     q = q ^ (w << ((N-1-y)*4));
	 }
     return q;
}
//-------------------------------------------------------------------------
int  bitpos(unsigned int x)
{
 int n;
	 n = (x < 0x100)?  Bitpos[x] : 8 + bitpos(x>>8);
	 return n;
}
//-------------------------------------------------------------------------
int  bitcount(int x)
{
 int c;
     c = 0;
	 while(x !=0) { c++; x = (x-1) & x; }
	 return c;
}
//-------------------------------------------------------------------------
void BitSel(int nest,int bit,int Bits,int M,int BSset[])
{
 int b,w;
	 if ((nest == M) || (nest == -M)) {
		 BSset[BScnt++] = bit;
		 return;
	 }
	 if (nest < -M)  {
		 BSset[BScnt++] = bit;
	 }
	 w = Bits;
	 while (w != 0) {
		 b = w & (-w);
		 w = w ^ b;
		 BitSel(nest+1,bit|b,w,M,BSset);
	 }
}
//-------------------------------------------------------------------------
void printNQ(char ID[],int Pos[],int N)
{
int i,j;
	fprintf(logid,"\n%s\n",ID);
	
	for (i=0; i<N; i++) {
		for (j=0; j<N; j++) {
			(Pos[i] & Bit[j])?  fprintf(logid," Q") : fprintf(logid," .") ;
		}
		fprintf(logid,"\n");
	}
	fflush(logid);
}
//-------------------------------------------------------------------------
void printNQ2(char ID[],u64 POS,int N)
{
int k,n,pos[32];
	for (k=N-1; k>=0; k--)  {
		n   = POS & 0x0f;
		POS = POS >> 4;
		pos[k] = (n == 0x0f)?  0 : Bit[n];
	}
	printNQ(ID,pos,N);
}
//-------------------------------------------------------------------------
void printAns(char ID[],u64 A,u64 B,u64 C,u64 D,int N)
{
 int pos[32] = { 0 };
 int n,k;
 u64 w;
     fprintf(logid,"\n----- POSdump  %llx  %llx  %llx  %llx\n",A,B,C,D);
     w = A;
	 for (k=0; k<N; k++) {
		 n = w & 0x0f;
		 w = w >> 4;
		 if (n == 0x0f)   continue;
		 pos[k] |= Bit[n];
	 }
     w = B;
	 for (k=0; k<N; k++) {
		 n = w & 0x0f;
		 w = w >> 4;
		 if (n == 0x0f)   continue;
		 pos[n] |= Bit[N*2-k-1];
	 }
     w = C;
	 for (k=0; k<N; k++) {
		 n = w & 0x0f;
		 w = w >> 4;
		 if (n == 0x0f)   continue;
		 pos[N*2-k-1] |= Bit[N*2-n-1];
	 }
     w = D;
	 for (k=0; k<N; k++) {
		 n = w & 0x0f;
		 w = w >> 4;
		 if (n == 0x0f)   continue;
		 pos[N*2-n-1] |= Bit[k];
	 }
	 printNQ(ID,pos,N*2);
}
//-------------------------------------------------------------------------
void  dumpAns(char ID[],PARTS *pa,PARTS *pb,PARTS *pc,PARTS *pd) 
{
int  u1,u2,u3,u4;
//	 if ((pa->p&0x01) || (pb->p&0x01) || (pc->p&0x01) || (pd->p&0x01)){
	 fprintf(logid,"%s  sym:%x  dup:%x  (%x %x %x %x)  ( %x %x %x %x)\n",
		   ID,frame.sym,frame.dup,frame.A,frame.B,frame.C,frame.D,frame.a,frame.b,frame.c,frame.d);
	 
	 if (pa->u & pd->v)   fprintf(logid,"xxxx a.u=%x d.v=%x\n",pa->u,pd->v);
	 u1 = pa->u | pd->v;
	 if (pc->u & pb->v)   fprintf(logid,"xxxx c.u=%x b.v=%x\n",pc->u,pb->v);
	 u2 = pc->u | pb->v;
	 u2 = BITRVS(u2) >> 1;
	 if (u1 & u2)   fprintf(logid,"xxxx u1=%x u2r=%x\n",u1,u2);
	 if (pb->u & pa->v)   fprintf(logid,"xxxx b.u=%x a.v=%x\n",pb->u,pa->v);
	 u3 = pb->u | pa->v;
	 if (pd->u & pc->v)   fprintf(logid,"xxxx d.u=%x c.v=%x\n",pd->u,pc->v);
	 u4 = pd->u | pc->v;
	 u4 = BITRVS(u4) >> 1;
	 if (u3 & u4)   fprintf(logid,"xxxx u3=%x u4r=%x\n",u3,u4);
	 fprintf(logid,"---- A : %x %x %x   B : %x %x %x   C :%x %x %x   D : %x %x %x\n",
		     pa->u,pa->v,pa->p,pb->u,pb->v,pb->p,pc->u,pc->v,pc->p,pd->u,pd->v,pd->p);
//	 }
	 fflush(logid);
}
