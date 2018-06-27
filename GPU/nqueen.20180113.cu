
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define THREAD_NUM		96


/* int bunk=0;		// this is a dummy variable used for making sure clock() are not optimized out */

/*
 * ----------------------------------------------------------------
 * This is a recursive version of n-queen backtracking solver.
 * A non-recursive version is used instead.
 * ----------------------------------------------------------------
*/
long long solve_nqueen_internal(int n,unsigned int mask,unsigned int l_mask,unsigned int r_mask,unsigned int t_mask) {
	if(mask==t_mask){
		return 1;
	}
	unsigned int m=(mask | l_mask | r_mask);
	if((m & t_mask)==t_mask){
		return 0;
	}
	long long total=0;
	unsigned int index=(m+1) & ~m;
	while((index & t_mask) != 0){
		total += solve_nqueen_internal(n,mask | index,(l_mask | index) << 1,(r_mask | index) >> 1,t_mask);
		m |= index;
		index=(m+1) & ~m;
	}
	return total;
}
long long solve_nqueen_recursive(int n){
	return solve_nqueen_internal(n,0,0,0,(1 << n) - 1);
}
/* -------------------------------------------------------------------
 * This is a non-recursive version of n-queen backtracking solver.
 * This provides the basis for the CUDA version.
 * -------------------------------------------------------------------
 */
long long solve_nqueen_nonRecursive(int n){
  unsigned int mask[32];
  unsigned int l_mask[32];
  unsigned int r_mask[32];
  unsigned int m[32];

  if(n<=0 || n>32){
    return 0;
  }

  const unsigned int t_mask=(1 << n) - 1;
  long long total=0;
  long long upper_total=0;
  int i=0,j;
  unsigned int index;

  mask[0]=0;
  l_mask[0]=0;
  r_mask[0]=0;
  m[0]=0;

  for(j=0;j<(n+1) / 2;j++){
    index=(1 << j);
    m[0] |= index;

    mask[1]=index;
    l_mask[1]=index << 1;
    r_mask[1]=index >> 1;
    m[1]=(mask[1] | l_mask[1] | r_mask[1]);
    i=1;

    if(n % 2==1 && j==(n+1) / 2 - 1){
      upper_total=total;
      total=0;
    }

    while(i>0){
      if((m[i] & t_mask)==t_mask){
        i--;
      }
      else {
        index=((m[i]+1) ^ m[i]) & ~m[i];
        m[i] |= index;
        if((index & t_mask) != 0){
          if(i+1==n){
            total++;
            i--;
          }
          else {
            mask[i+1]=mask[i] | index;
            l_mask[i+1]=(l_mask[i] | index) << 1;
            r_mask[i+1]=(r_mask[i] | index) >> 1;
            m[i+1]=(mask[i+1] | l_mask[i+1] | r_mask[i+1]);
            i++;
          }
        }
        else {
          i --;
        }
      }
    }
  }

  /* bunk=2; */

  if(n % 2==0){
    return total * 2;
  }
  else {
    return upper_total * 2+total;
  }
}
/* --------------------------------------------------------------------------
 * This is a non-recursive version of n-queen backtracking solver for CUDA.
 * It receives multiple initial conditions from a CPU iterator,and count
 * each conditions.
 * --------------------------------------------------------------------------
 */
__global__ void solve_nqueen_cuda_kernel(int n,int mark,unsigned int* total_masks,unsigned int* total_l_masks,unsigned int* total_r_masks,unsigned int* results,int total_conditions)
{
  const int tid=threadIdx.x;
  const int bid=blockIdx.x;
  const int idx=bid * blockDim.x+tid;

  __shared__ unsigned int mask[THREAD_NUM][10];
  __shared__ unsigned int l_mask[THREAD_NUM][10];
  __shared__ unsigned int r_mask[THREAD_NUM][10];
  __shared__ unsigned int m[THREAD_NUM][10];

  __shared__ unsigned int sum[THREAD_NUM];

  const unsigned int t_mask=(1 << n) - 1;
  int total=0;
  int i=0;
  unsigned int index;

  if(idx<total_conditions){
    mask[tid][i]=total_masks[idx];
    l_mask[tid][i]=total_l_masks[idx];
    r_mask[tid][i]=total_r_masks[idx];
    m[tid][i]=mask[tid][i] | l_mask[tid][i] | r_mask[tid][i];

    while(i>=0){
      if((m[tid][i] & t_mask)==t_mask){
        i--;
      }
      else {
        index=(m[tid][i]+1) & ~m[tid][i];
        m[tid][i] |= index;
        if((index & t_mask) != 0){
          if(i+1==mark){
            total++;
            i--;
          }
          else {
            mask[tid][i+1]=mask[tid][i] | index;
            l_mask[tid][i+1]=(l_mask[tid][i] | index) << 1;
            r_mask[tid][i+1]=(r_mask[tid][i] | index) >> 1;
            m[tid][i+1]=(mask[tid][i+1] | l_mask[tid][i+1] | r_mask[tid][i+1]);
            i++;
          }
        }
        else {
          i --;
        }
      }
    }

    sum[tid]=total;
  }
  else {
    sum[tid]=0;
  }

  __syncthreads();

  // reduction
  if(tid<64 && tid+64<THREAD_NUM){ sum[tid] += sum[tid+64];} __syncthreads();
  if(tid<32){ sum[tid] += sum[tid+32];} __syncthreads();
  if(tid<16){ sum[tid] += sum[tid+16];} __syncthreads();
  if(tid<8){ sum[tid] += sum[tid+8];} __syncthreads();
  if(tid<4){ sum[tid] += sum[tid+4];} __syncthreads();
  if(tid<2){ sum[tid] += sum[tid+2];} __syncthreads();
  if(tid<1){ sum[tid] += sum[tid+1];} __syncthreads();

  if(tid==0){
    results[bid]=sum[0];
  }
}


long long solve_nqueen_cuda(int n,int steps)
{
  // generating start conditions
  unsigned int mask[32];
  unsigned int l_mask[32];
  unsigned int r_mask[32];
  unsigned int m[32];
  unsigned int index;

  if(n<=0 || n>32){
    return 0;
  }

  unsigned int* total_masks=new unsigned int[steps];
  unsigned int* total_l_masks=new unsigned int[steps];
  unsigned int* total_r_masks=new unsigned int[steps];
  unsigned int* results=new unsigned int[steps];

  unsigned int* masks_cuda;
  unsigned int* l_masks_cuda;
  unsigned int* r_masks_cuda;
  unsigned int* results_cuda;

  cudaMalloc((void**) &masks_cuda,sizeof(int) * steps);
  cudaMalloc((void**) &l_masks_cuda,sizeof(int) * steps);
  cudaMalloc((void**) &r_masks_cuda,sizeof(int) * steps);
  cudaMalloc((void**) &results_cuda,sizeof(int) * steps / THREAD_NUM);

  const unsigned int t_mask=(1 << n) - 1;
  const unsigned int mark=n>11 ? n - 10 : 2;
  long long total=0;
  int total_conditions=0;
  int i=0,j;

  mask[0]=0;
  l_mask[0]=0;
  r_mask[0]=0;
  m[0]=0;

  bool computed=false;

  for(j=0;j<n / 2;j++){
    index=(1 << j);
    m[0] |= index;

    mask[1]=index;
    l_mask[1]=index << 1;
    r_mask[1]=index >> 1;
    m[1]=(mask[1] | l_mask[1] | r_mask[1]);
    i=1;

    while(i>0){
      if((m[i] & t_mask)==t_mask){
        i--;
      }
      else {
        index=(m[i]+1) & ~m[i];
        m[i] |= index;
        if((index & t_mask) != 0){
          mask[i+1]=mask[i] | index;
          l_mask[i+1]=(l_mask[i] | index) << 1;
          r_mask[i+1]=(r_mask[i] | index) >> 1;
          m[i+1]=(mask[i+1] | l_mask[i+1] | r_mask[i+1]);
          i++;
          if(i==mark){
            total_masks[total_conditions]=mask[i];
            total_l_masks[total_conditions]=l_mask[i];
            total_r_masks[total_conditions]=r_mask[i];
            total_conditions++;
            if(total_conditions==steps){
              if(computed){
                cudaMemcpy(results,results_cuda,sizeof(int) * steps / THREAD_NUM,cudaMemcpyDeviceToHost);

                for(int j=0;j<steps / THREAD_NUM;j++){
                  total += results[j];
                }

                computed=false;
              }

              // start computation
              cudaMemcpy(masks_cuda,total_masks,sizeof(int) * total_conditions,cudaMemcpyHostToDevice);
              cudaMemcpy(l_masks_cuda,total_l_masks,sizeof(int) * total_conditions,cudaMemcpyHostToDevice);
              cudaMemcpy(r_masks_cuda,total_r_masks,sizeof(int) * total_conditions,cudaMemcpyHostToDevice);

              solve_nqueen_cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM>>>(n,n - mark,masks_cuda,l_masks_cuda,r_masks_cuda,results_cuda,total_conditions);

              computed=true;

              total_conditions=0;
            }
            i--;
          }
        }
        else {
          i --;
        }
      }
    }
  }


  if(computed){
    cudaMemcpy(results,results_cuda,sizeof(int) * steps / THREAD_NUM,cudaMemcpyDeviceToHost);

    for(int j=0;j<steps / THREAD_NUM;j++){
      total += results[j];
    }
    computed=false;
  }

  cudaMemcpy(masks_cuda,total_masks,sizeof(int) * total_conditions,cudaMemcpyHostToDevice);
  cudaMemcpy(l_masks_cuda,total_l_masks,sizeof(int) * total_conditions,cudaMemcpyHostToDevice);
  cudaMemcpy(r_masks_cuda,total_r_masks,sizeof(int) * total_conditions,cudaMemcpyHostToDevice);

  solve_nqueen_cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM>>>(n,n - mark,masks_cuda,l_masks_cuda,r_masks_cuda,results_cuda,total_conditions);

  cudaMemcpy(results,results_cuda,sizeof(int) * steps / THREAD_NUM,cudaMemcpyDeviceToHost);

  for(int j=0;j<steps / THREAD_NUM;j++){
    total += results[j];
  }	

  total *= 2;

  if(n % 2==1){
    computed=false;
    total_conditions=0;

    index=(1 << (n - 1) / 2);
    m[0] |= index;

    mask[1]=index;
    l_mask[1]=index << 1;
    r_mask[1]=index >> 1;
    m[1]=(mask[1] | l_mask[1] | r_mask[1]);
    i=1;

    while(i>0){
      if((m[i] & t_mask)==t_mask){
        i--;
      }
      else {
        index=(m[i]+1) & ~m[i];
        m[i] |= index;
        if((index & t_mask) != 0){
          mask[i+1]=mask[i] | index;
          l_mask[i+1]=(l_mask[i] | index) << 1;
          r_mask[i+1]=(r_mask[i] | index) >> 1;
          m[i+1]=(mask[i+1] | l_mask[i+1] | r_mask[i+1]);
          i++;
          if(i==mark){
            total_masks[total_conditions]=mask[i];
            total_l_masks[total_conditions]=l_mask[i];
            total_r_masks[total_conditions]=r_mask[i];
            total_conditions++;
            if(total_conditions==steps){
              if(computed){
                cudaMemcpy(results,results_cuda,sizeof(int) * steps / THREAD_NUM,cudaMemcpyDeviceToHost);

                for(int j=0;j<steps / THREAD_NUM;j++){
                  total += results[j];
                }

                computed=false;
              }

              // start computation
              cudaMemcpy(masks_cuda,total_masks,sizeof(int) * total_conditions,cudaMemcpyHostToDevice);
              cudaMemcpy(l_masks_cuda,total_l_masks,sizeof(int) * total_conditions,cudaMemcpyHostToDevice);
              cudaMemcpy(r_masks_cuda,total_r_masks,sizeof(int) * total_conditions,cudaMemcpyHostToDevice);

              solve_nqueen_cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM>>>(n,n - mark,masks_cuda,l_masks_cuda,r_masks_cuda,results_cuda,total_conditions);

              computed=true;

              total_conditions=0;
            }
            i--;
          }
        }
        else {
          i --;
        }
      }
    }

    if(computed){
      cudaMemcpy(results,results_cuda,sizeof(int) * steps / THREAD_NUM,cudaMemcpyDeviceToHost);

      for(int j=0;j<steps / THREAD_NUM;j++){
        total += results[j];
      }

      computed=false;
    }

    cudaMemcpy(masks_cuda,total_masks,sizeof(int) * total_conditions,cudaMemcpyHostToDevice);
    cudaMemcpy(l_masks_cuda,total_l_masks,sizeof(int) * total_conditions,cudaMemcpyHostToDevice);
    cudaMemcpy(r_masks_cuda,total_r_masks,sizeof(int) * total_conditions,cudaMemcpyHostToDevice);

    solve_nqueen_cuda_kernel<<<steps/THREAD_NUM,THREAD_NUM>>>(n,n - mark,masks_cuda,l_masks_cuda,r_masks_cuda,results_cuda,total_conditions);

    cudaMemcpy(results,results_cuda,sizeof(int) * steps / THREAD_NUM,cudaMemcpyDeviceToHost);

    for(int j=0;j<steps / THREAD_NUM;j++){
      total += results[j];
    }
  }

  cudaFree(masks_cuda);
  cudaFree(l_masks_cuda);
  cudaFree(r_masks_cuda);
  cudaFree(results_cuda);

  delete[] total_masks;
  delete[] total_l_masks;
  delete[] total_r_masks;
  delete[] results;

  /* bunk=1; */

  return total;
}
/** CUDA 初期化 **/
bool InitCUDA(){
  int count;
  cudaGetDeviceCount(&count);
  if(count==0){
    fprintf(stderr,"There is no device.\n");
    return false;
  }
  int i;
  for(i=0;i<count;i++){
    cudaDeviceProp prop;
    if(cudaGetDeviceProperties(&prop,i)==cudaSuccess){
      if(prop.major>=1){
        break;
      }
    }
  }
  if(i==count){
    fprintf(stderr,"There is no device supporting CUDA 1.x.\n");
    return false;
  }
  cudaSetDevice(i);
  return true;
}
int main(int argc,char** argv) {
  /* int n=4; */
  long long solution;
  bool cpu=true,gpu=true;
  int argstart=1,steps=24576;
  /** パラメータの処理 */
  if(argc>=2 && argv[1][0]=='-'){
    if(argv[1][1]=='c' || argv[1][1]=='C'){
      gpu=false;
    }
    else if(argv[1][1]=='g' || argv[1][1]=='G'){
      cpu=false;
    }
    argstart=2;
  }
  //if(argc<argstart+1){
  if(argc<argstart){
    printf("Usage: %s [-c|-g] n steps\n",argv[0]);
    printf("  -c: CPU only\n");
    printf("  -g: GPU only\n");
    printf("Default to 8 queen\n");
  }
  /** 出力と実行 */
  int min=4;
  int targetN=16;
  /** CPU */
  if(cpu){
    struct timeval t0; struct timeval t1; int ss;int ms;int dd;
    printf("%s\n"," N:          Total        Unique                 dd:hh:mm:ss.ms");
    for(int i=min;i<=targetN;i++){
      gettimeofday(&t0,NULL);    // 計測開始
      /** Recursive */
      solution=solve_nqueen_recursive(i);
      /** nonRecursive */
      //solution=solve_nqueen_nonRecursive(i);
      gettimeofday(&t1,NULL);    // 計測終了
      if (t1.tv_usec<t0.tv_usec) {
        dd=(int)(t1.tv_sec-t0.tv_sec-1)/86400;
        ss=(t1.tv_sec-t0.tv_sec-1)%86400;
        ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
      } else { 
        dd=(int)(t1.tv_sec-t0.tv_sec)/86400;
        ss=(t1.tv_sec-t0.tv_sec)%86400;
        ms=(t1.tv_usec-t0.tv_usec+500)/10000;
      }
      int hh=ss/3600;
      int mm=(ss-hh*3600)/60;
      ss%=60;
      long lGUnique=0;
      printf("%2d:%18llu%18llu%12.2d:%02d:%02d:%02d.%02d\n", i,(unsigned long long)solution,(unsigned long long)lGUnique,dd,hh,mm,ss,ms);
    }
  }
  /** GPU */
  if(gpu){
    if(!InitCUDA()){
      return 0;
    }
    struct timeval t0; struct timeval t1; int ss;int ms;int dd;
    printf("%s\n"," N:          Total        Unique                 dd:hh:mm:ss.ms");
    for(int i=min;i<=targetN;i++){
      gettimeofday(&t0,NULL);    // 計測開始
      solution=solve_nqueen_cuda(i,steps);
      gettimeofday(&t1,NULL);    // 計測終了
      if (t1.tv_usec<t0.tv_usec) {
        dd=(int)(t1.tv_sec-t0.tv_sec-1)/86400;
        ss=(t1.tv_sec-t0.tv_sec-1)%86400;
        ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
      } else { 
        dd=(int)(t1.tv_sec-t0.tv_sec)/86400;
        ss=(t1.tv_sec-t0.tv_sec)%86400;
        ms=(t1.tv_usec-t0.tv_usec+500)/10000;
      }
      int hh=ss/3600;
      int mm=(ss-hh*3600)/60;
      ss%=60;
      long lGUnique=0;
      printf("%2d:%18llu%18llu%12.2d:%02d:%02d:%02d.%02d\n", i,(unsigned long long)solution,(unsigned long long)lGUnique,dd,hh,mm,ss,ms);
    }
  }
  return 0;
}
