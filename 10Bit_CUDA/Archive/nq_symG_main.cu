/*------------------------------------------------------------*/
/*  author  Kazuji   HAGINOYA      2012.03.05                 */
/*          Keigo    TANAKA *1                                */
/*          Noriyuki FUJIMOTO *1                              */
/* *1 Department of mathematics and Information Sciences,     */
/*    Graduate School of Science, Osaka Prefecture University */
/*------------------------------------------------------------*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <nq_symG_kernel.cu>

#include <stdlib.h>
#include <windows.h>

#define nsize     19
#define BLKSZ     64
#define MPSZ      96

//#include "cuda_def.h"     
//#include "cuda_dmy_main.cpp"

#define PoolSize  2500000

 typedef struct {
    int  status;
	int  kbn;
    int  t;
    void *buf;
 } TCA;
   
    int  N,MASK,J,Mid,X0,MASKH;

	unsigned long long *d_cnt;
	int T_all,T8,T4,T2;	                /* 1/2ïîï™âÇ…ÇÊÇÈå¬êî*/
	int Kline;
	int kubun,Pcnt;
	
	unsigned long long cnt,Total;
	int4 *proper8,*proper4,*proper2;	/* 1/2ïîï™âÇÃmaskîzóÒ*/
	int4 *d_proper;
	unsigned int *d_counter; 
	unsigned int *counter;			

	int rBit[1 << 13];
	int Bits[1 << 13];

	
	DWORD WINAPI RunGPU(LPVOID);
	HANDLE       hThread;
	DWORD        dwThreadID;
	CRITICAL_SECTION lock;

	TCA tca;
	
	dim3 block(BLKSZ,1,1);
	dim3 grid(MPSZ,1,1);
    
//-------------------------------------------------------------------------
int Wait(TCA *tca,int Val)
{
 int rc = -1;
 
//    printf("Wait %d  %d\n",tca->status,Val);

    EnterCriticalSection(&lock);
    
    while(tca->status >= 0) {
        if (tca->status == Val)  { rc = 0; break; };
		LeaveCriticalSection(&lock);

        Sleep(1);
	    
		EnterCriticalSection(&lock);
    }
	LeaveCriticalSection(&lock);
	
//    printf("Wait exit %d\n",tca->status);
    
    return rc; 
}
//-------------------------------------------------------------------------
int Swap(TCA *tca,int X,int Y)
{
 int rc = -1;
 
//    printf("Swap %d  %d %d\n",tca->status,X,Y);
    
    EnterCriticalSection(&lock);

    if (tca->status == X) {
        tca->status = Y;
        rc = 0;
    }
	LeaveCriticalSection(&lock);

//    printf("Swap status  %d\n",tca->status);
    
    return rc;
}

//-------------------------------------------------------------------------
void  submit(int Kbn,int *T,int4 *proper)
{
        tca.kbn = Kbn;
        tca.t   = *T;
        tca.buf = proper;
	    Swap(&tca,0,1);  //triger RunGPU
	    
		Sleep(1);

	    Wait(&tca,0);    //wait for buf-empty
}
//-------------------------------------------------------------------------
void  HStore(int Kbn,int X,int U,int V,int BM,int *T,int4 *proper)
{
    if (*T >= PoolSize) {
        submit(Kbn,T,proper);
        *T = 0;
    }
//	printf(" %d   (%x %x %x)\n",*T,X,U&MASK,V);

    proper[*T].x = X ^ MASK;
    proper[*T].y = U;
    proper[*T].z = V;
    proper[*T].w = BM;
       
    (*T)++;
}
//-------------------------------------------------------------------------
int RVS(int X,int N)
{
int k,r,w;

    w = X;
	r = 0;
	for (k=0; k<N; k++)  { r = (r << 1) | (w & 0x01); w = w >> 1; }
	return r;
}
//-------------------------------------------------------------------------
int Bitcount(int X)
{
int c,w;

	c = 0;
	w = X;
	while (w) { c++;  w &= w - 1; }
	return c;
}
//-------------------------------------------------------------------------
void nq(int nest,int X,int U,int V,int B,int N)
{
int b,m,K,JM,BM,A,BB,c;
int XX,UU,VV;

     if ((nest >= Kline) && (J > 0)) {

		 K  = (N - nest) * BLKSZ;
		 JM = (1 << J) - 1;
		 
         BM = JM & ~(X | (U<<(N-nest)) | (V>>(N-nest)));
    	 if (BM)
//             HStore(0,X,U,V,(nest<<16)+J,&T8,proper8);
             HStore(0,X,U,V,(K<<16)+JM,&T8,proper8);
         return;
     } 
	 if (nest == N) {

//             if (X0 > Mid)    return;

//             BM = Mid & ~(X | U | V);
//             if (BM)
//                 HStore(2,X,U,V,0,&T2,proper2);
//	         return;
//
         A = X & MASKH;
         c = rBit[X >> (N+1)];
	
	     if (Bits[A] == Bits[c]) {

			 if (X0 > Mid)    return;

             BM = Mid & ~(X | U | V);
             if (BM) {
//                 HStore(2,X,U,V,0,&T2,proper2);  // D free
                 XX =  X | Mid;
                 UU = (U | Mid) << 1;
                 VV = (V | Mid) >> 1;
                 BM = MASK & ~(XX | UU | VV);
                 HStore(2,XX,UU,VV,BM,&T2,proper2);  // D free
             }
	         return;
	     }

		 if (Bits[A] <  Bits[c]) {

			 BM = Mid & ~(X | U | V);

			 if (BM) { 
//				if (A <  B)  HStore(1,X,U,V,0,&T4,proper4);
//				if (A == B)  HStore(2,X,U,V,0,&T2,proper2);
                 XX =  X | Mid;
                 UU = (U | Mid) << 1;
                 VV = (V | Mid) >> 1;
                 BM = MASK & ~(XX | UU | VV);
				 if (A <  B)  HStore(1,XX,UU,VV,BM,&T4,proper4);
				 if (A == B)  HStore(2,XX,UU,VV,BM,&T2,proper2);
			 }
		 }
 	     return; 
	 }

	 m  = MASK & ~(X | U | V);
	 while(m != 0) {
		 b = m & (-m);
		 m = m ^ b;

		 if (nest == 0) { 
             if (b == Mid)          continue; 		
			 X0 = b;
		 }
         if (b == Mid)  J = nest;

		 BB = (b <  Mid)?  B | (1 << nest) : B ;
		 
		 nq(nest+1, X|b, (U|b)<<1, (V|b)>>1, BB, N);

		 if (b == Mid)  J = -1;

	 }
}
//-------------------------------------------------------------------------
DWORD WINAPI SeedsGen(LPVOID)
//void RunGPU(void)
{
 int k;
  
    printf("==== SeedsGen started ====\n");

	N    = nsize >> 1;
	Mid  = 1 << N;
	MASK = (1 << nsize) - 1;
	MASKH= Mid - 1;

	Kline= N - 1; 
//	Kline= N - 2; 
    J    = -1;

	for (k=0; k<(1<<N); k++)  { rBit[k] = RVS(k,N);  Bits[k] = Bitcount(k); }

	T_all = 0;
    T8    = 0;
    T4    = 0;
    T2    = 0;
    
    nq(0,0,0,0,0,N);
    
//flush buf
    submit(0,&T8,proper8);
    submit(1,&T4,proper4);
    submit(2,&T2,proper2); 
	T8 = 0;
	submit(0,&T8,proper8);
	
	tca.status = -1;  //SeedsGen completed  &  halt GPU-thread
    
    printf("==== SeedsGen terminated ====\n");
    Sleep(100);

	return 0;
}
//-------------------------------------------------------------------------
int main(int argc, char* argv[])
{
 const int Maps[] = { 8, 4, 2 };
 int   kbn,kubun;
 int4  *proper;
 
	printf("\n==== nq_symG ver.65   nsize=%d,MPSZ=%d,BLKSZ=%d,Queens=%d,PoolSize=%d\n",
	        nsize,MPSZ,BLKSZ,Queens,PoolSize); 

	unsigned int timer = 0;
//temp modfied by deepgreen
//	 u64  tick1,tick2;
//	 printf("----- N = %d -----\n",nsize);
//     tick1 = clock();

	cutilCheckError( cutCreateTimer( &timer));
	cutilCheckError( cutStartTimer( timer));

//	proper8 = (int4*)malloc(sizeof(int4)*PoolSize);
//	proper4 = (int4*)malloc(sizeof(int4)*PoolSize);
//	proper2 = (int4*)malloc(sizeof(int4)*PoolSize);
	cudaHostAlloc((void**) &proper8,sizeof(int4)*PoolSize,cudaHostAllocDefault);
	cudaHostAlloc((void**) &proper4,sizeof(int4)*PoolSize,cudaHostAllocDefault);
	cudaHostAlloc((void**) &proper2,sizeof(int4)*PoolSize,cudaHostAllocDefault);
	
	counter = (unsigned int*)malloc(sizeof(unsigned int));
	
	cudaMalloc((void**) &d_cnt, sizeof(unsigned long long));
	cudaMalloc((void**) &d_proper, sizeof(int4)*PoolSize);
	cudaMalloc((void**) &d_counter, sizeof(unsigned int) );
//	d_cnt    = (unsigned long long *) malloc(sizeof(unsigned long long));
//	d_proper = (int4*)                malloc(sizeof(int4)*PoolSize);
//	d_counter= (unsigned int *)       malloc(sizeof(unsigned int));


//***** thread launch section

    tca.status = 0;    //initialize thread communication area  

	InitializeCriticalSection(&lock);

	hThread=CreateThread(NULL, //default security attributes
						0, //default stack size
						SeedsGen, //function name
						0, // parameter
						0, // start the thread immediately after creation
						&dwThreadID);
	if (hThread) {
//		printf ("SeedsGen-Thread launched successfully\n");
		CloseHandle (hThread);
	}

//****** GPU control section

	Total = 0;
	Pcnt  = 0;

    while(tca.status >= 0) {
    
    	if (Wait(&tca,1) < 0)   break;  //wait for buf-full
    
    	kbn   = tca.kbn;
		kubun = Maps[kbn];
		T_all = tca.t;
    	proper= (int4*)tca.buf;
        
		if (T_all)
	 	    cudaMemcpy(d_proper, proper, sizeof(int4)*T_all, cudaMemcpyHostToDevice);

    	Swap(&tca,1,0); //release buf
//		printf("---- release buf   T_all = %d\n",T_all);
    	
		if (T_all == 0)      break;

    	cudaMemset(d_cnt, 0, sizeof(unsigned long long));

		counter[0] = BLKSZ * MPSZ;			/* blocksize * multiprocessorêî */
//    	counter[0] = 1;
    	cudaMemcpy(d_counter, counter,sizeof(unsigned int), cudaMemcpyHostToDevice);
	
//temp modified by deepgreen
//	assign<<< grid, block >>>(n, N_assign, d_cnt, T_all, d_proper, BLKSZ, d_counter);	
//    	assign(kbn, nsize, d_cnt, T_all, d_proper, 1, d_counter);
    	assign<<< grid, block >>>(kbn, nsize, d_cnt, T_all, d_proper, BLKSZ, d_counter);

		cutilCheckMsg("Kernel execution failed");
    	
    	cudaMemcpy(&cnt, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
		Total += kubun * cnt;

		Pcnt++;
		if ((T_all != PoolSize) ||
		   ((Pcnt & 0x03) == 1)) {
			printf("---- kick (%d) ---- %8d  %9d      %I64d  %I64d\n",kbn,Pcnt,T_all,Total,cnt);
		}
    }

//temp modfied by deepgreen
	cutilCheckError( cutStopTimer( timer));
	printf("Processing time: %f (s)\n", cutGetTimerValue( timer)/1000);
	cutilCheckError( cutDeleteTimer( timer));
//    tick2 = clock();
//	printf("Processing time: %d\n", (int)(tick2-tick1));
	
//	printf("âÇÃå¬êî");
//	printf("%llu\n", cnt*2);
	printf("%llu\n", Total);

//	free(proper);
//	free(proper8);
//	free(proper4);
//	free(proper2);
	cudaFreeHost(proper8);
	cudaFreeHost(proper4);
	cudaFreeHost(proper2);
	free(counter);
	cudaFree(d_cnt);
	cudaFree(d_proper);
	cudaFree(d_counter);

	DeleteCriticalSection(&lock);
 
//    printf("SeedsGen terminated\n\n" );

//	CUT_EXIT(argc, argv);

	return 0;
}
