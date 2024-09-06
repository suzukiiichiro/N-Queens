/*------------------------------------------------------------*/
/*  author  Kazuji   HAGINOYA      2012.03.05                 */
/*          Keigo    TANAKA *1                                */
/*          Noriyuki FUJIMOTO *1                              */
/* *1 Department of mathematics and Information Sciences,     */
/*    Graduate School of Science, Osaka Prefecture University */
/*------------------------------------------------------------*/

//#define nsize 19	/* nsize : N-QUEENのサイズ*/
//#include "cuda_def.h"     
//#include "cuda_dmy_kernel.cpp"

#define  Queens   16    // MAX Queens
#define  Threads  64    // MAX Threads / Block

__device__
void NQPart(int n, unsigned int number, int2 *Q, int mask, int BLKSZ,
					 unsigned int T_all, unsigned long long* d_cnt, unsigned int *d_counter, int4* d_proper,int pos)
{
    unsigned __int64 cnt = 0;	/* cnt : 成功数のカウント */
    int   x,s,h,e,bm,J,K,JM,Mid,MB;
    unsigned int  c,bit,rr;
    unsigned __int64  bit2,uu,ll;
    
    Mid = (n - 1) / 2;
	MB  = Mid * BLKSZ;
	
	x = number;
    s = x;

	rr = d_proper[pos].x;
	uu = d_proper[pos].y;
	ll = d_proper[pos].z;
	J    = d_proper[pos].w;
	K    = J >> 16;
    JM   = J & 0xffff;
    h    = number + K;
    e    = h + MB;
    
	bm = rr & ~(uu | ll);
	ll = ll << 32;

    if (x == h)  bm &= JM;
    
    for(;;) {

        if (bm) {
            bit = bit2 = -bm & bm; 
              
            x += BLKSZ;
            Q[x].x = bit;
            Q[x].y = bm ^ bit;
            
            rr =  rr ^ bit;
            uu = (uu | bit2) << 1;
            ll = (ll | (bit2<<32)) >> 1;

            bm = rr & ~(uu | (ll>>32));
        } 
		else {
			bit= bit2 = Q[x].x;
			bm = Q[x].y;
            rr =  rr | bit;
            uu = (uu >> 1) ^ bit2;
            ll = (ll << 1) ^ (bit2<<32);
            
		    x -= BLKSZ;
            if (x == e)   cnt++;
			if (x < s){
				c = atomicAdd(d_counter, 1);
				if (c < T_all){					
					x = s;
					rr = d_proper[c].x;
                	uu = d_proper[c].y;
					ll = d_proper[c].z;
    				J    = d_proper[c].w;
					K    = J >> 16;
				    JM   = J & 0xffff;
				    h    = number + K;
				    e    = h + MB;
				    
					bm = rr & ~(uu | ll);
					ll = ll << 32;
				}
				else{
					atomicAdd(d_cnt, cnt);
					return;
				}
			}
        }
        if (x == h)  bm &= JM;
	}
}
__device__
void NQPart2(int n, unsigned int number, int2 *Q, int mask, int BLKSZ,
					 unsigned int T_all, unsigned long long* d_cnt, unsigned int *d_counter, int4* d_proper,int pos)
{
    unsigned __int64 cnt = 0;	/* cnt : 成功数のカウント */
    int   x,s,e,bm,N,Mid;
    unsigned int   c,bit,rr;
    unsigned __int64  bit2,uu,ll;
    
    N   = (n - 1) / 2;
	Mid = 1 << N;
	
	x = number;
    s = x;
//    e    = number + (n-N-1) * BLKSZ;
    e    = number + (n-N-2) * BLKSZ;

	rr = d_proper[pos].x;
	uu = d_proper[pos].y;
	ll = d_proper[pos].z;
	ll = ll << 32;
//	bm        = Mid;
	bm = d_proper[pos].w;

    for(;;) {

        if (bm) {
            bit = bit2 = -bm & bm;

            x += BLKSZ;
            
            Q[x].x = bit;
            Q[x].y = bm ^ bit;
            rr =  rr ^ bit;
            uu = (uu | bit2) << 1;
            ll = (ll | (bit2<<32)) >> 1;

            bm = rr & ~(uu | (ll>>32));
        } 
		else {
			bit= bit2 = Q[x].x;
			bm = Q[x].y;
            rr =  rr | bit;
            uu = (uu >> 1) ^ bit2;
            ll = (ll << 1) ^ (bit2<<32);

			x -= BLKSZ;
			if (x == e)  cnt++;
			if (x < s){
				c = atomicAdd(d_counter, 1);
				if(c < T_all){

					x = s;
					rr = d_proper[c].x;
                	uu = d_proper[c].y;
					ll = d_proper[c].z;
					ll = ll << 32;
//					bm        = Mid;
					bm = d_proper[c].w;
				}
				else{
					atomicAdd(d_cnt, cnt);
					return;
				}
			}
        }
    }
}
__global__ void assign(int Kbn, int n, unsigned long long* d_cnt, unsigned int T_all, int4* d_proper, int BLKSZ, unsigned int *d_counter){
			/* n : N-QUEENのサイズ, number : thread番号, nassign : 分解する列数 */
	unsigned int anumber = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int number = threadIdx.x;
	int mask = (1 << n) - 1; 

	__shared__ int2 Q[Queens*Threads];
	
	if(T_all <= anumber)	return;
	
//	NQPart(n, nassign, number, bitmap, u, r, l, mask, BLKSZ, T_all, d_cnt, d_counter, d_proper);
    (Kbn == 0)? 
		NQPart(n, number, Q, mask, BLKSZ, T_all, d_cnt, d_counter, d_proper,anumber) :
		NQPart2(n, number, Q, mask, BLKSZ, T_all, d_cnt, d_counter, d_proper,anumber);
	
	return;
}
