// nqueen_dg.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//
// #include "stdafx.h"
#include "nqueen_dg_QJH_Inc_odd.h"

/*-----------------------------------*/
/*  copyright deepgreen  2011.07.31  */
/*                                   */
/*  author    deepgreen  2010.07.07  */
/*  bugfix    SymCheck   2011.05.01  */
/*-----------------------------------*/

//-------------------------------------------------------------------------
//int _tmain(int argc, _TCHAR* argv[])
int main()
{
 char *stime[30],*etime[30];
 u64  tick1,tick2;


    tick1 = clock();

    logid = fopen("log.txt","wt"); 

    time(&t1);
    *stime = ctime(&t1);
    fprintf(logid,"*** pazul started : %s \n",*stime);

	for (N=2; N<=9; N++) {

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
 int  I,J;

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


	 tick1 = clock();

	 N2 = N * 2;
	 N3 = N2 + 1;
 
 	 init(N);


     Tans = 0;
	 Tcnt = 0;

	 for (M=1; M<N; M++) {

//		 if (M != 4)   continue;

         t1 = clock();
	     Mans = 0;
	     Mcnt = 0;
	     
		 M2 = N - M;
		 M3 = M - 1;

		 frame.N2M = (M == M2)?  1 : 0;


		 MmSet[0] = 0;
    	 BitSel(0,0,MASK,M3,MmSet);

    	 setFRI(MmSet,FRIm);

		 MjSet[0] = 0;
    	 BitSel(0,0,MASK,M,MjSet);

    	 setFRI(MjSet,FRIj);

		 MnSet[0] = 0;
    	 BitSel(0,0,MASK,M2,MnSet);

    	 setFRI(MnSet,FRIn);

//
		 FRI = FRIm;  FRIMAX = MmSet[0];
		 PartsGen(N,M3,MmIXC,MmIXP,&MmDtp);
		 Mmcnt = Cnt;

		 FRI = FRIj;  FRIMAX = MjSet[0];
		 PartsGen(N,M ,MjIXC,MjIXP,&MjDtp);
		 Mjcnt = Cnt;

		 FRI = FRIn;  FRIMAX = MnSet[0];
		 PartsGen(N,M2,MnIXC,MnIXP,&MnDtp);
		 Mncnt = Cnt;

//		 fprintf(logid,"\n---- Cnt = ( %I64d, %I64d, %I64d )\n",Mmcnt,Mjcnt,Mncnt);
//		 fflush(logid);

         hit = 0;
         Hit = 0;


	     for (I=0; I<N; I++) {
	     	 for (J=I+1; J<N; J++) {

				 frame.I = I;
				 frame.J = J;
				 frame.M = 0xff;

				 Acnt = Filter(0,I,J);
				 Bcnt = Filter(1,I,J);
				 Ccnt = Filter(2,I,J);
				 Dcnt = Filter(3,I,J);

//	     	 	 fprintf(logid,"\n\n-------------------- I=%d J=%d  ( %I64d, %I64d, %I64d, %I64d )\n",
//								I,J,Acnt,Bcnt,Ccnt,Dcnt);
//	     	 	 fflush(logid);

	     	 	 QJoin2(M,I,J);
	     	 	 
	     	 	 free(Adtp);
	     	 	 free(Bdtp);
	     	 	 free(Cdtp);
	     	 	 free(Ddtp);
	     	 	 free(AJdtp);
	     	 	 free(BJdtp);
	     	 	 free(CJdtp);
	     	 	 free(DJdtp);
	     	 }
	     }
		 if (M <= M2) {

			 frame.I = 0xff;
			 frame.J = 0xff;
			 frame.M = M;

			 Acnt = Filter(4,M,M);
			 Bcnt = Filter(5,M,M);
			 Ccnt = Filter(6,M,M);
			 Dcnt = Filter(7,M,M);

//     	 	 fprintf(logid,"\n\n-------------------- M=%d ( %I64d, %I64d, %I64d, %I64d )\n",
//							M,Acnt,Bcnt,Ccnt,Dcnt);
//	     	 fflush(logid);
		 
		     QJoin(M);

		 	 free(Adtp);
		 	 free(Bdtp);
		 	 free(Cdtp);
		 	 free(Ddtp);
		 	 free(AJdtp);
		 	 free(BJdtp);
		 	 free(CJdtp);
		 	 free(DJdtp);
		 }
		 Tans += hit;
	     Tcnt += Hit;
		 Mans += hit;
		 Mcnt += Hit;

		 free(MmDtp);
		 free(MjDtp);
		 free(MnDtp);
		
		 t2 = clock();
         etime = (int)(t2-t1);
		 fprintf(logid,"----- M=%2d   %12I64d   %12I64d    %7d        Mm : %9I64d  Mj : %9I64d  Mn : %9I64d  /   ABcnt : %9I64d   CDcnt : %9I64d\n",
		         M,Mans,Mcnt,etime,Mmcnt,Mjcnt,Mncnt,TABcnt,TCDcnt);
         fflush(logid);

	 }
     clear(N);

	 tick2 = clock();
     etime = (int)(tick2-tick1);

	 printf("\n***** N=%2d   %14I64d  %14I64d   %10d \n",N3,Tans,Tcnt,etime);

	 fprintf(logid,"\n***** N=%2d   %14I64d  %14I64d   %10d \n\n\n",N3,Tans,Tcnt,etime);
	 if ((Tans != AnsUnq[N3]) || (Tcnt != AnsAll[N3])) 
		 fprintf(logid,"?????? Ans Error  %14I64d  %14I64d\n",AnsUnq[N3],AnsAll[N3]);
	 fflush(logid);

}

//-------------------------------------------------------------------------
void init(int N)
{
 int k,n,w,r;
 u64 one=1;
//               1 2 3 4  5  6  7  8   9  10  11  12   13
 int nCr[] = { 0,1,2,3,6,10,20,35,70,126,252,462,924,1716 };
	           

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
	   JAsize = 5; JBsize = 4;
//	   JAsize = 4; JBsize = 4;
//	   JAsize = 4; JBsize = 3;   //base version
	}
	else {
//	   JAsize = N-3; JBsize = N-3;
	   JAsize = 5; JBsize = 3;
//	   JAsize = 4; JBsize = 2;   //base version
	}
//	JIXsize = JAsize + JBsize;
	JIXsize = N2 - 1 - (JAsize + JBsize);

	MASKJA = Bit[JAsize] - 1;
    MASKJB = Bit[JBsize] - 1;
	MASKJ2=  Bit[JIXsize] - 1;

//odd	MASKJ = Bit[N-1] - 1;
	MASKJ = Bit[N-2] - 1;

// PARTS pool Index
//	IXbitsize = N2;
//  IXlen = (Bit[IXbitsize] + 1) * sizeof(int);
	IXsize= nCr[N] * nCr[N];
    IXlen = (IXsize + 1) * sizeof(int);

	MmIXC = (int*) malloc(IXlen);
	MmIXP = (PARTS**) malloc(IXlen);

	MjIXC = (int*) malloc(IXlen);
	MjIXP = (PARTS**) malloc(IXlen);

	MnIXC = (int*) malloc(IXlen);
	MnIXP = (PARTS**) malloc(IXlen);

	AIXP  = (PARTSJ**) malloc(IXlen);
	BIXP  = (PARTSJ**) malloc(IXlen);
	CIXP  = (PARTSJ**) malloc(IXlen);
	DIXP  = (PARTSJ**) malloc(IXlen);

// Half(AB-CD) join Cache
	UIXlen= (Bit[JIXsize]+1) * sizeof(int);
	Uixc  = (int*) malloc(UIXlen);
	Uixp  = (PARTSJ**) malloc(UIXlen);

	Uixp2 = (PARTSJ**) malloc(UIXlen);

	DtCnt = 1 << N;
	Dtlen = DtCnt * sizeof(HPARTS);
	Tdtp = (HPARTS*) malloc(Dtlen);


	DtCnt2 = 1 << N;
	Dtlen2 = DtCnt2 * sizeof(HPARTS);
	Tdtp2 = (HPARTS2*) malloc(Dtlen2);

	FRIm = (int*) malloc(Bit[N]*sizeof(int));
	FRIj = (int*) malloc(Bit[N]*sizeof(int));
	FRIn = (int*) malloc(Bit[N]*sizeof(int));
}

//-------------------------------------------------------------------------
void clear(int N)
{
	free(MmIXC);
	free(MmIXP);
	
	free(MjIXC);
	free(MjIXP);
	
	free(MnIXC);
	free(MnIXP);
	
	free(AIXP);
	free(BIXP);
	free(CIXP);
	free(DIXP);

	free(Uixc);
	free(Uixp);

	free(Uixp2);
	
	free(Tdtp);
	free(Tdtp2);

	free(FRIm);
	free(FRIj);
	free(FRIn);

}

//-------------------------------------------------------------------------
int  Filter(int Q,int I,int J)
{
 int UM,Cnt;
 
	 switch (Q) {
	 	 case 0:  // parts A
	 	 	UM  = (I > 0)? Bit[I-1] : 0;
	 	 	UM |= Bit[N2-J-1];
	 	 	Cnt = QSUB(MmSet,Bit[I],Bit[J],UM,Bit[I]+Bit[J],MmIXP,AIXP,&AJdtp,&Adtp);
	 	 	break;
	 	   
	 	 case 1:  // parts B
	 	 	UM  = (I > 0)? Bit[I-1] : 0;
	 	 	UM |= Bit[J-1];
//	 	 	Cnt = QSUB(Bit[J],0,UM,Bit[J],MnIXP,BIXP,&Bdtp);
	 	 	Cnt = QSUB(MnSet,0,Bit[J],UM,Bit[J],MnIXP,BIXP,&BJdtp,&Bdtp);
	 	 	break;

	 	 case 2:  // parts C
	 	 	UM  = Bit[J-1];
	 	 	UM |= Bit[N2-I-1];
	 	 	Cnt = QSUB(MjSet,0,0,UM,0,MjIXP,CIXP,&CJdtp,&Cdtp);
	 	 	break;

	 	 case 3:  // parts D
	 	 	UM  = Bit[N2-I-1];
	 	 	UM |= Bit[N2-J-1];
//	 	 	Cnt = QSUB(0,Bit[I],UM,Bit[I],MnIXP,DIXP,&Ddtp);
	 	 	Cnt = QSUB(MnSet,Bit[I],0,UM,Bit[I],MnIXP,DIXP,&DJdtp,&Ddtp);
	 	 	break;

		 case 4:
	 	 	Cnt = QSUB(MjSet,0,0,Bit[N-1],0,MjIXP,AIXP,&AJdtp,&Adtp);
	 	   break;

	 	 case 5:
	 	 	Cnt = QSUB(MnSet,0,0,Bit[N-1],0,MnIXP,BIXP,&BJdtp,&Bdtp);
	 	   break;

	 	 case 6:
	 	 	Cnt = QSUB(MjSet,0,0,Bit[N-1],0,MjIXP,CIXP,&CJdtp,&Cdtp);
	 	   break;

	 	 case 7:
	 	 	Cnt = QSUB(MnSet,0,0,Bit[N-1],0,MnIXP,DIXP,&DJdtp,&Ddtp);
	 	   break;

	 }
	 return Cnt;
}
//-------------------------------------------------------------------------
int  QSUB(int BSet[],int XM,int YM,int UM,int VM,
          PARTS *IXP[],PARTSJ *QIXP[],PARTSJ **QJdtp,PARTS **Qdtp)
{
 PARTS  *dtp,*qdtp,*qdtp2;
 PARTSJ *qjdtp;
 int    QSlen,x,y,xy,k,ix,iy,ixy;
 

//	 QSlen = IXP[Bit[N2]] - IXP[0];
	 QSlen = IXP[IXsize] - IXP[0];

	 *Qdtp = (PARTS*) malloc(QSlen*sizeof(PARTS));
	 qdtp  = *Qdtp;

	 *QJdtp = (PARTSJ*) malloc(QSlen*sizeof(PARTSJ));
	 qjdtp  = *QJdtp;
	 
//	 for (x=0; x<Bit[N]; x++) {
	 for (ix=0; ix<BSet[0]; ix++) {
	 	 x = BSet[ix+1];
	 	 
//	 	 for (y=0; y<Bit[N]; y++) {
		 for (iy=0; iy<BSet[0]; iy++) {
		 	 y = BSet[iy+1];
			 
		 	 ixy = (ix * BSet[0]) + iy;

	 	 	 QIXP[ixy] = qjdtp;
	 	 
			 if (XM & x)              continue;
			 if (YM & y)              continue;

	 		 for (dtp=IXP[ixy]; dtp<IXP[ixy+1]; dtp++) {
	 	 
	 	 		 if (UM & dtp->u)     continue;
	 	 		 if (VM & dtp->v)     continue;

	 	 		 qdtp->u = dtp->u;
	 	 		 qdtp->v = dtp->v;
	 	 		 qdtp->p = dtp->p;
	 	 		 qdtp->r = dtp->r;

				 if (IXP == MmIXP) {
					 qdtp->u |= UM;
					 qdtp->v |= VM;
				 }

	 	 		 qjdtp->jkey = (IXP == MnIXP)? 
	 	 			         ((dtp->v >> 1) << N) + (dtp->u & MASKJ) :
							 ((rBit[dtp->u>>(N+1)]) << (N-2)) + (dtp->v >> 1);

		 	 	 qdtp++;
		 	 	 qjdtp++;
			 }
	 	 }
	 }
 	 QIXP[ixy+1] = qjdtp;       //set last

	 return (qdtp - *Qdtp);
}

//-------------------------------------------------------------------------
void QJoin(int M)
{
 int ia,ib,ic,id,xy,xya,xyb,xyc,xyd;
 int A,C;


	 TABcnt = 0;
	 TCDcnt = 0;

	 for (ia=0; ia<MjSet[0]; ia++) {
	   
		 frame.A = MjSet[ia+1];
		 frame.a = frame.A ^ MASK;

		 if (frame.N2M) {
	   		 if (frame.A > frame.a)                     continue;    //5. Y check
		 }

		 A = rBit[frame.A];

		 for (ic=0; ic<MjSet[0]; ic++) {
	     
			 frame.C = MjSet[ic+1];
			 frame.c = frame.C ^ MASK;

			 frame.sym = 0;

			 if (frame.A >  frame.C)                              continue;
			 if (frame.A == frame.C)        frame.sym |= 0x04;   // 3. R180 check  
			 if (frame.N2M) {
			 	 if (frame.A >  frame.c)                          continue;
			 	 if (frame.A == frame.c)    frame.sym |= 0x40;   // 4. X check
			 }

			 C = rBit[frame.C];

			 ABJoin(A,C,&ABdtp);
			 ABcnt   = Cnt;
			 TABcnt += ABcnt;
			 

			 if (ABcnt) {
				 CDJoin(A,C,&CDdtp);
				 CDcnt   = Cnt;
				 TCDcnt += CDcnt;
			 
				 if (CDcnt > 0) {

					 HJoin();

					 free(CDdtp);
					 free(Ulp);
				 }
				 free(ABdtp);
			 }
		 }
	 }
}
//-------------------------------------------------------------------------
void QJoin2(int M,int I,int J)
{
 int ia,ib,ic,id,xy,xya,xyb,xyc,xyd;
 int A,C,XM,YM,UM,VM;


	 TABcnt = 0;
	 TCDcnt = 0;

	 for (ia=0; ia<MmSet[0]; ia++) {
	   
	   	 frame.A = MmSet[ia+1];

		 A = rBit[frame.A];
		 if (A & Bit[I])           continue;

	   	 frame.a = (frame.A | Bit[I]) ^ MASK;

		 for (ic=0; ic<MjSet[0]; ic++) {
	     
			 frame.C = MjSet[ic+1];
			 frame.c = frame.C ^ MASK;
			 C = rBit[frame.C];

			 ABJoin2(A,C,I,J);
			 ABcnt   = Cnt;
			 TABcnt += ABcnt;
			 
			 if (ABcnt) {
				 CDJoin2(A,C,I,J);
				 CDcnt   = Cnt;
				 TCDcnt += CDcnt;
			 
				 if (CDcnt > 0) {

					 HJoin2();
				
					 free(Ulp);
				 }
				 free(Ulp2);
			 }
		 }
	 }
}
//-------------------------------------------------------------------------
void HJoin()
{
 int    abix,cdix,dup,cdinc,cdinc2,cdmask,jkey;
 HPARTS *abdtp;
 HPARTS *cddtp;
 PARTSJ *ucp;

 //return;

	 for (abdtp=ABdtp; abdtp<ABdtp+ABcnt; abdtp++) {

		 abix = abdtp->hkey;
		 jkey = abdtp->jkey;

		 cdinc= abix & (-abix);
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
		//		 if (AnsCheck("???",abdtp,cddtp)) {

		//			 dumpAns2("????",abdtp,cddtp);
		//		 }

//				 dup = (DUP)?  DUP : SymCheck(abdtp,cddtp);
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
void HJoin2()
{
 int    OK,abix,cdix,dup,cdinc,cdinc2,cdmask,hkey,jkey;
// HPARTS2 *abdtp;
// HPARTS2 *cddtp;
 PARTSJ *ucp,*ucp2;

 //return;

	 OK = 0;

	 for (hkey=0; hkey<=MASKJ2; hkey++) {
	 
	 	 for (ucp2=Uixp2[hkey]; ucp2<Uixp2[hkey+1]; ucp2++) {

//		 abix = abdtp->hkey;
			 abix = hkey;
			 jkey = ucp2->jkey;

			 cdinc= abix & (-abix);
			 if (cdinc == 0)  cdinc = MASKJ2 + 1;  // set any nonzero value

			 cdinc2 = abix + cdinc;
			 cdmask = MASK2 ^ abix;

			 cdix = 0;
        	 while(cdix <= MASKJ2) {
			 
	 			 for (ucp=Uixp[cdix]; ucp<Uixp[cdix+cdinc]; ucp++) {

					 if (jkey & ucp->jkey)       continue;
					 OK++;

            	 }
		    	 cdix = (cdix + cdinc2) & cdmask;
	    	 }
	     }
	 }
	 hit += OK;
	 Hit += OK * 8;
}

//-------------------------------------------------------------------------
void ABJoin(int A,int C,HPARTS **Dtp)
{
 int b,c,B,xya,xyb;
 int jkey,ix,k,dtlen,ullen,sym;
 int IA,IB,ic,ib;
 PARTS  *adtp,*bdtp;
 PARTSJ *qadtp,*qbdtp,*ulp;
 HPARTS *tdtp,*dtp;

	 Cnt = 0;
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
		 
//		 xya = (A << N) + B;
//		 xyb = (b << N) + c;
//		 xyb = (c << N) + b;
		 IA = FRIj[A];
		 IB = FRIj[B];
		 ic = FRIn[c];
		 ib = FRIn[b];
		 if (IB < 0)              continue;

		 xya = (IA * MjSet[0]) + IB;
		 xyb = (ic * MnSet[0]) + ib;

		 adtp = Adtp + (AIXP[xya] - AJdtp);
		 for (qadtp= AIXP[xya]; qadtp<AIXP[xya+1]; qadtp++,adtp++) {
		 
			 jkey = qadtp->jkey;
		 
			 bdtp = Bdtp + (BIXP[xyb] - BJdtp);
			 for (qbdtp=BIXP[xyb]; qbdtp<BIXP[xyb+1]; qbdtp++,bdtp++) {

				 if (jkey & qbdtp->jkey)     continue;

				 setHalf(1,adtp,bdtp);

			 }
		 }
	 }
	 dtlen = Cnt*sizeof(HPARTS);
     *Dtp  = (HPARTS*) malloc(dtlen);
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
void ABJoin2(int A,int C,int I,int J)
{
 int b,c,B,xya,xyb;
 int jkey,ix,k,dtlen,ullen,sym;
 int IA,IB,ic,ib;
 PARTS  *adtp,*bdtp;
 PARTSJ *qadtp,*qbdtp,*ulp;
 HPARTS2 *hparts,*tdtp,*dtp;

	 Cnt = 0;
//	 hparts = Tdtp;
	 memset(Uixc,0,UIXlen);

//	 fprintf(logid,"\n--- ABjoin2 A=%x C=%x\n",A,C);
//	 fflush(logid);
	 
	 if (A & Bit[I])              return;
	 
	 c = C ^ MASK;
	 

	 for (B=0; B<Bit[N]; B++) {

		 if (B & Bit[J])          continue;

		 b = (B | Bit[J]) ^ MASK;
		 frame.B = rBit[B];
		 frame.b = rBit[b];
		 
//		 xya = (A << N) + B;
//		 xyb = (b << N) + c;
//		 xyb = (c << N) + b;
		 IA = FRIm[A];
		 IB = FRIm[B];
		 ic = FRIn[c];
		 ib = FRIn[b];
		 if (IB < 0)              continue;
		 
		 xya = (IA * MmSet[0]) + IB;
		 xyb = (ic * MnSet[0]) + ib;

		 adtp = Adtp + (AIXP[xya] - AJdtp);
		 for (qadtp= AIXP[xya]; qadtp<AIXP[xya+1]; qadtp++,adtp++) {
		 
			 jkey = qadtp->jkey;
		 
			 bdtp = Bdtp + (BIXP[xyb] - BJdtp);
			 for (qbdtp=BIXP[xyb]; qbdtp<BIXP[xyb+1]; qbdtp++,bdtp++) {

				 if (jkey & qbdtp->jkey)     continue;

				 setHalf2(1,adtp,bdtp);

			 }
		 }
	 }
//	 dtlen = Cnt*sizeof(HPARTS2);
//     *Dtp  = (HPARTS2*) malloc(dtlen);
//     memset(*Dtp,0xff,dtlen);
     tdtp  = Tdtp2;

     ullen = Cnt*sizeof(PARTSJ);
     Ulp2  = (PARTSJ*) malloc(ullen);
     memset(Ulp2,0xff,ullen);
         
     Uixp2[0] = Ulp2;
	 for (k=1; k<=Bit[JIXsize]; k++)  Uixp2[k] = Uixp2[k-1] + Uixc[k-1]; 

	 memset(Uixc,0,UIXlen);

	 for (k=0; k<Cnt; k++) {
		 ix  = tdtp->hkey;
         ulp = Uixp2[ix] + Uixc[ix];
		 ulp->jkey = tdtp->jkey;
		 
//		 dtp = *Dtp + (ulp-Uixp[0]);            //relocate to CDdtp
//         memcpy(dtp,tdtp,sizeof(HPARTS2));
         Uixc[ix]++;
         tdtp++;
     }
}
//-------------------------------------------------------------------------
void CDJoin(int A,int C,HPARTS **Dtp)
{
 int a,d,D,xyc,xyd;
 int jkey,ix,k,dtlen,ullen,sym;
 int IC,ID,ia,id;
 PARTS  *cdtp,*ddtp;
 PARTSJ *qcdtp,*qddtp,*ulp;
 HPARTS *tdtp,*dtp;

	 Cnt = 0;
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
//		 xyc = (C << N) + D;
//		 xyd = (d << N) + a;
//		 xyd = (a << N) + d;
		 IC = FRIj[C];
		 ID = FRIj[D];
		 ia = FRIn[a];
		 id = FRIn[d];
		 if (ID < 0)     continue;

		 xyc = (IC * MjSet[0]) + ID;
		 xyd = (ia * MnSet[0]) + id;

		 cdtp = Cdtp + (CIXP[xyc] - CJdtp);
		 for (qcdtp= CIXP[xyc]; qcdtp<CIXP[xyc+1]; qcdtp++,cdtp++) {
		 
			 jkey = qcdtp->jkey;
		 
			 ddtp = Ddtp + (DIXP[xyd] - DJdtp);
			 for (qddtp= DIXP[xyd]; qddtp<DIXP[xyd+1]; qddtp++,ddtp++) {

				 if (jkey & qddtp->jkey)     continue;

				 setHalf(2,cdtp,ddtp);

			 }
		 }
	 }
	 dtlen = Cnt*sizeof(HPARTS);
     *Dtp  = (HPARTS*) malloc(dtlen);
     memset(*Dtp,0xff,dtlen);
     tdtp  = Tdtp;

     ullen = Cnt*sizeof(PARTSJ);
     Ulp   = (PARTSJ*) malloc(ullen);
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
void CDJoin2(int A,int C,int I,int J)
{
 int a,d,D,xyc,xyd;
 int jkey,ix,k,dtlen,ullen,sym;
 int IC,ID,ia,id;
 PARTS  *cdtp,*ddtp;
 PARTSJ *qcdtp,*qddtp,*ulp;
 HPARTS2 *tdtp,*dtp;

	 Cnt = 0;
	 memset(Uixc,0,UIXlen);
	 
	 if (A & Bit[I])              return;

	 a = (A | Bit[I]) ^ MASK;
	 frame.A = rBit[A];
	 frame.a = rBit[a];


	 for (D=0; D<Bit[N]; D++) {

		 d = D ^ MASK;
		 frame.D = rBit[D];
		 frame.d = rBit[d];

//		 xyc = (C << N) + D;
//		 xyd = (d << N) + a;
//		 xyd = (a << N) + d;
		 IC = FRIj[C];
		 ID = FRIj[D];
		 ia = FRIn[a];
		 id = FRIn[d];
		 if (ID < 0)     continue;
		 
		 xyc = (IC * MjSet[0]) + ID;
		 xyd = (ia * MnSet[0]) + id;

		 cdtp = Cdtp + (CIXP[xyc] - CJdtp);
		 for (qcdtp= CIXP[xyc]; qcdtp<CIXP[xyc+1]; qcdtp++,cdtp++) {

			 jkey = qcdtp->jkey;
		 
			 ddtp = Ddtp + (DIXP[xyd] - DJdtp);
			 for (qddtp= DIXP[xyd]; qddtp<DIXP[xyd+1]; qddtp++,ddtp++) {

				 if (jkey & qddtp->jkey)     continue;

				 setHalf2(2,cdtp,ddtp);

			 }
		 }
	 }
     tdtp  = Tdtp2;

     ullen = Cnt*sizeof(PARTSJ);
     Ulp   = (PARTSJ*) malloc(ullen);
     memset(Ulp,0xff,ullen);
         
     Uixp[0] = Ulp;
	 for (k=1; k<=Bit[JIXsize]; k++)  Uixp[k] = Uixp[k-1] + Uixc[k-1]; 

	 memset(Uixc,0,UIXlen);

	 for (k=0; k<Cnt; k++) {
		 ix  = tdtp->hkey;
         ulp = Uixp[ix] + Uixc[ix];
		 ulp->jkey = tdtp->jkey;

         Uixc[ix]++;
         tdtp++;
     }
}
//-------------------------------------------------------------------------
void setHalf(int stage,PARTS *p,PARTS *q)
{
 HPARTS *hparts,*dtp;
 int    sym,u,u2,w,h1,h2,hkey,jkey;

	 if (Cnt >= DtCnt) {
		 dtp = (HPARTS*)malloc(Dtlen*2);

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

//	 u  = p->u | (rBit[q->v] << (N-1));
//	 u2 = q->u | p->v;
	 u  = p->u | (rBit[(q->v >> 1)] << (N-1));   //bugfix 2010.06.30
	 u2 = q->u | (p->v >> 1);

	 if (stage == 1) {
		 hparts->x  = frame.A;
		 hparts->y  = frame.B;
		 hparts->x2 = frame.c;
		    
		 u2 = BITRVS(u2) >> 1;
	 }
	 else {
		 hparts->x  = frame.C;
		 hparts->y  = frame.D;
		 hparts->x2 = frame.a;

		 u = BITRVS(u) >> 1;
	 }
	 if (M >= M2)  {
		 w = u;
		 u = u2;
		 u2= w;
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

//	 fprintf(logid,"\n    sethalf  : u=%x u2=%x  /parts  ",u,u2);
//	 dumpParts(p);
//	 dumpParts(q);
//	 dumpHparts(hparts);
//	 fprintf(logid,"\n");
//	 fflush(logid);
}

//-------------------------------------------------------------------------
void setHalf2(int stage,PARTS *p,PARTS *q)
{
 HPARTS2 *hparts,*dtp;
 int    sym,u,u2,w,h1,h2,hkey,jkey;

	 if (Cnt >= DtCnt2) {
		 dtp = (HPARTS2*)malloc(Dtlen2*2);

		 memcpy(dtp,Tdtp2,Dtlen2);

		 free(Tdtp2);

		 Tdtp2 = dtp;
		 DtCnt2 *= 2;
		 Dtlen2 *= 2;
	 }
	 hparts = Tdtp2 + Cnt;

//change pos-info 2010.05.07
//	 hparts->p  = p->p;
//	 hparts->q  = q->p;
//	 hparts->r  = p->r;       

//	 hparts->sym= frame.sym;

//	 u  = p->u | (rBit[q->v] << (N-1));
//	 u2 = q->u | p->v;
	 u  = p->u | (rBit[(q->v >> 1)] << (N-1));   //bugfix 2010.06.30
	 u2 = q->u | (p->v >> 1);

	 if (stage == 1) {
//		 hparts->x  = frame.A;
//		 hparts->y  = frame.B;
//		 hparts->x2 = frame.c;
		    
		 u2 = BITRVS(u2) >> 1;
	 }
	 else {
//		 hparts->x  = frame.C;
//		 hparts->y  = frame.D;
//		 hparts->x2 = frame.a;

		 u = BITRVS(u) >> 1;
	 }
	 if (M >= M2)  {
		 w = u;
		 u = u2;
		 u2= w;
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

//	 fprintf(logid,"\n    sethalf  : u=%x u2=%x  /parts  ",u,u2);
//	 dumpParts(p);
//	 dumpParts(q);
//	 dumpHparts(hparts);
//	 fprintf(logid,"\n");
//	 fflush(logid);
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
void PartsGen(int N,int M,int *IXC,PARTS *IXP[],PARTS **Dtp) 
{
 int k;
 
	 XYIXC = IXC;
	 XYIXP = (PARTS**)IXP;
	 
   Stage = 1;
     
	 memset(XYIXC,0,IXlen);
	 Cnt = 0;
	 
     nq(0,0,0,0,0,0,N,M);

	 *Dtp = XYDtp = (PARTS *) malloc(Cnt * sizeof(PARTS));
	 
   Stage = 2;

     Cnt = 0;
	 XYIXP[0] = XYDtp;
//	 for (k=0; k<Bit[N2]; k++)  XYIXP[k+1] = XYIXP[k] + XYIXC[k];
	 for (k=0; k<IXsize; k++)  XYIXP[k+1] = XYIXP[k] + XYIXC[k];
	 
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


        qsym = QSymCheck(X,Y);
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

	 p = (XYIXC == MnIXC)? 
		 XYIXC[(FRI[X] * FRIMAX) + FRI[Y]] :
		 XYIXC[(FRI[Y] * FRIMAX) + FRI[X]] ;

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

//		 xy = (Qframe[t].x << N) + Qframe[t].y;
		 xy = (XYIXC == MnIXC)? 
		 	  (FRI[Qframe[t].y] * FRIMAX) + FRI[Qframe[t].x] :
			  (FRI[Qframe[t].x] * FRIMAX) + FRI[Qframe[t].y] ;

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
		 BSset[++BSset[0]] = bit;
		 return;
	 }
	 if (nest < -M)  {
		 BSset[++BSset[0]] = bit;
	 }
	 w = Bits;
	 while (w != 0) {
		 b = w & (-w);
		 w = w ^ b;
		 BitSel(nest+1,bit|b,w,M,BSset);
	 }
}

//-------------------------------------------------------------------------
void setFRI(int BSset[],int *FRI)
{
 int i,j,k,w;

	 for (i=1; i<=BSset[0]; i++) {
		 for (j=i+1; j<=BSset[0]; j++) {
			 if (BSset[i] > BSset[j]) {
				 w = BSset[i];
				 BSset[i] = BSset[j];
				 BSset[j] = w;
			 }
		 }
	 }
	 memset(FRI,0xff,Bit[N]*sizeof(int));

	 for (k=0; k<BSset[0]; k++)   FRI[BSset[k+1]] = k;
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

     fprintf(logid,"\n----- POSdump  %I64x  %I64x  %I64x  %I64x\n",A,B,C,D);

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
void  dumpAns2(char ID[],HPARTS *abparts,HPARTS *cdparts)
{
 char strAB[64],strCD[64];

	  dumpHparts(strAB,abparts);
	  dumpHparts(strCD,cdparts);
	  sprintf(str,"\n---- %s :  AB %s    CD %s",ID,strAB,strCD);
}
//-------------------------------------------------------------------------
void  dumpHparts(char str[],HPARTS *hparts)
{
	  sprintf(str," x=%x x2=%x y=%x hkey=%x jkey=%x",
		     hparts->x,hparts->x2,hparts->y,hparts->hkey,hparts->jkey);
}
//-------------------------------------------------------------------------
void  dumpParts(PARTS *parts)
{
	  fprintf(logid," u=%x v=%x",
		     parts->u,parts->v);
}
//-------------------------------------------------------------------------
int   AnsCheck(char ID[],HPARTS *abdtp,HPARTS *cddtp)
{
 char board[32][32];
 int  i,j,k,qx,qy,sx,sy,mx,my,retry;

	  memset(board,0,sizeof(board));

	  setX(board,0,0,rBit[abdtp->x]);
	  setX(board,N+1,0,abdtp->x2);
	  setY(board,0,0,rBit[abdtp->y]);
	  setY(board,N+1,0,(rBit[abdtp->y]|Bit[frame.J])^MASK);

	  setX(board,0,N+1,rBit[cddtp->x2]);
	  setX(board,N+1,N+1,cddtp->x);
	  setY(board,0,N+1,cddtp->y^MASK);
	  setY(board,N+1,N+1,cddtp->y);

//	  printBoard("???",board);

	  if (frame.I >= 0)  setQ(board,frame.I,N);
	  if (frame.J >= 0)  setQ(board,N,frame.J);
	  if (frame.M >= 0)  setQ(board,N,N);

//	  printBoard("---",board);

	  mx = 0;
	  my = 0;
	  do {
		  retry = 0;
		  for (i=0; i<N3; i++) {
		  	  qx = 0;
		  	  qy = 0;
		  	  for (j=0; j<N3; j++) {
		  	      if (board[i][j] == 2)  { qx++;  sx = j; }
		  	      if (board[j][i] == 2)  { qy++;  sy = j; }
		  	  }
		  	  if  ((qx == 0) || (qy == 0))  break;
			  if ((qx == 1) && ((mx & Bit[sx]) == 0))  {  mx |= Bit[sx]; setQ(board,sx,i); retry = 1; }
			  if ((qy == 1) && ((my & Bit[sy]) == 0))  {  my |= Bit[sy]; setQ(board,i,sy); retry = 1; }
		  }
		  if (i < N3) {
			  dumpAns2("??? ",abdtp,cddtp);
			  printBoard(str,board);
			  return -1;
		  }
	  } while(retry);
	  
	  dumpAns2("+++ ",abdtp,cddtp);
	  printBoard(str,board);

	  return 0;
}
//-------------------------------------------------------------------------
void  setX(char board[32][32],int X,int Y,int mask)
{
 int  k,n;

	  for (k=0; k<N; k++) {
		  if ((mask & Bit[k]) == 0)    continue;
		  for (n=0; n<N; n++)  board[n+Y][k+X]++;
	  }

}
//-------------------------------------------------------------------------
void  setY(char board[32][32],int X,int Y,int mask)
{
 int  k,n;

	  for (k=0; k<N; k++) {
		  if ((mask & Bit[k]) == 0)    continue;
		  for (n=0; n<N; n++)  board[k+Y][n+X]++;
	  }

}
//-------------------------------------------------------------------------
void  setQ(char board[32][32],int X,int Y)
{
 int  k,n,z;

	  for (k=0; k<N3; k++) {
	  	  board[Y][k] = 0;
	  	  board[k][X] = 0;
		  z = Y - X + k;
		  if ((z >= 0) && (z < N3))  board[z][k] = 0;
		  z = X + Y - k;
		  if ((z >= 0) && (z < N3))  board[z][k] = 0;
	  }
	  board[Y][X] = 2;
}
//-------------------------------------------------------------------------
void  printBoard(char ID[],char board[32][32])
{
 int  i,j;

	  fprintf(logid,"\n%s\n",ID);
	  for (i=0; i<N3; i++) {
		  for (j=0; j<N3; j++) {
			  (board[i][j])?  fprintf(logid," %d",board[i][j]) :
							  fprintf(logid," .");
		  }
		  fprintf(logid,"\n");
	  }
	  fflush(logid);
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
