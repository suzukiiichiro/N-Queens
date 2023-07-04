#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string>
#include <time.h>
#include <sys/time.h>

unsigned long TOTAL;
unsigned long UNIQUE;
//
// $B%/%$!<%s$N8z$-$rH=Dj$7$F2r$rJV$9(B
__host__ __device__ 
long nQueens(int size,long left,long down,long right)
{
  long mask=(1<<size)-1;
  long counter = 0;
  if (down==mask) { // down$B$,$9$Y$F@lM-$5$l2r$,8+$D$+$k(B
    return 1;
  }
  long bit=0;
  for(long bitmap=mask&~(left|down|right);bitmap;bitmap^=bit){
    bit=-bitmap&bitmap;
    counter += nQueens(size,(left|bit)>>1,(down|bit),(right|bit)<< 1); 
  }
  return counter;
}
//
// i $BHVL\$N%a%s%P$r(B i $BHVL\$NItJ,LZ$N2r$GKd$a$k(B
__global__ 
void calculateSolutions(int size,long* nodes, long* solutions, int numElements)
{
  int i=blockDim.x * blockIdx.x + threadIdx.x;
  if(i<numElements){
    solutions[i]=nQueens(size,nodes[3 * i],nodes[3 * i + 1],nodes[3 * i + 2]);
  }
}
//
// 0$B0J30$N(Bbit$B$r%+%&%s%H(B
int countBits(long n)
{
  int counter = 0;
  while (n){
    n &= (n - 1); // $B1&C<$N%<%m0J30$N?t;z$r:o=|(B
    counter++;
  }
  return counter;
}
//
// $B%N!<%I$r(Bk$BHVL\$N%l%$%d!<$N%N!<%I$GKd$a$k(B
long kLayer(int size,std::vector<long>& nodes, int k, long left, long down, long right)
{
  long counter=0;
  long mask=(1<<size)-1;
  // $B$9$Y$F$N(Bdown$B$,Kd$^$C$?$i!"2r7h:v$r8+$D$1$?$3$H$K$J$k!#(B
  if (countBits(down) == k) {
    nodes.push_back(left);
    nodes.push_back(down);
    nodes.push_back(right);
    return 1;
  }
  long bit=0;
  for(long bitmap=mask&~(left|down|right);bitmap;bitmap^=bit){
    bit=-bitmap&bitmap;
    // $B2r$r2C$($FBP3Q@~$r$:$i$9(B
    counter+=kLayer(size,nodes,k,(left|bit)>>1,(down|bit),(right|bit)<<1); 
  }
  return counter;
}
//
// k $BHVL\$N%l%$%d$N$9$Y$F$N%N!<%I$r4^$`%Y%/%H%k$rJV$9!#(B
std::vector<long> kLayer(int size,int k)
{
  std::vector<long> nodes{};
  kLayer(size,nodes, k, 0, 0, 0);
  return nodes;
}
//
// $B%N!<%I%l%$%d!<$N:n@.(B
void nodeLayer(int size)
{
  //int size=16;
  // $B%D%j!<$N(B3$BHVL\$N%l%$%d!<$K$"$k%N!<%I(B
  //$B!J$=$l$>$lO"B3$9$k(B3$B$D$N?t;z$G%(%s%3!<%I$5$l$k!K$N%Y%/%H%k!#(B
  // $B%l%$%d!<(B2$B0J9_$O%N!<%I$N?t$,6QEy$J$N$G!"BP>N@-$rMxMQ$G$-$k!#(B
  // $B%l%$%d(B4$B$K$O==J,$J%N!<%I$,$"$k!J(BN16$B$N>l9g!"(B9844$B!K!#(B
  std::vector<long> nodes = kLayer(size,4); 

  // $B%G%P%$%9$K$O%/%i%9$,$J$$$N$G!"(B
  // $B:G=i$NMWAG$r;XDj$7$F$+$i%G%P%$%9$K%3%T!<$9$k!#(B
  size_t nodeSize = nodes.size() * sizeof(long);
  long* hostNodes = (long*)malloc(nodeSize);
  hostNodes = &nodes[0];
  long* deviceNodes = NULL;
  cudaMalloc((void**)&deviceNodes, nodeSize);
  cudaMemcpy(deviceNodes, hostNodes, nodeSize, cudaMemcpyHostToDevice);

  // $B%G%P%$%9=PNO$N3d$jEv$F(B
  long* deviceSolutions = NULL;
  int numSolutions = nodes.size() / 6; // We only need half of the nodes, and each node is encoded by 3 integers.
  size_t solutionSize = numSolutions * sizeof(long);
  cudaMalloc((void**)&deviceSolutions, solutionSize);

  // nQueens CUDA$B%+!<%M%k$r5/F0$9$k!#(B
  int threadsPerBlock = 256;
  int blocksPerGrid = (numSolutions + threadsPerBlock - 1) / threadsPerBlock;
  calculateSolutions <<<blocksPerGrid, threadsPerBlock >>> (size,deviceNodes, deviceSolutions, numSolutions);

  // $B7k2L$r%[%9%H$K%3%T!<(B
  long* hostSolutions = (long*)malloc(solutionSize);
  cudaMemcpy(hostSolutions, deviceSolutions, solutionSize, cudaMemcpyDeviceToHost);

  // $BItJ,2r$r2C;;$7!"7k2L$rI=<($9$k!#(B
  long solutions = 0;
  for (long i = 0; i < numSolutions; i++) {
      solutions += 2*hostSolutions[i]; // Symmetry
  }

  // $B=PNO(B
  //std::cout << "We have " << solutions << " solutions on a " << size << " by " << size << " board." << std::endl;
  TOTAL=solutions;
  //return 0;
}
//
// CUDA $B=i4|2=(B
bool InitCUDA()
{
  int count;
  cudaGetDeviceCount(&count);
  if(count==0){fprintf(stderr,"There is no device.\n");return false;}
  int i;
  for(i=0;i<count;i++){
    struct cudaDeviceProp prop;
    if(cudaGetDeviceProperties(&prop,i)==cudaSuccess){if(prop.major>=1){break;} }
  }
  if(i==count){fprintf(stderr,"There is no device supporting CUDA 1.x.\n");return false;}
  cudaSetDevice(i);
  return true;
}
//
//$B%a%$%s(B
int main(int argc,char** argv)
{
  bool cpu=false,cpur=false,gpu=false,sgpu=false;
  int argstart=1;
  if(argc>=2&&argv[1][0]=='-'){
    if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='r'||argv[1][1]=='R'){cpur=true;}
    else if(argv[1][1]=='c'||argv[1][1]=='C'){cpu=true;}
    else if(argv[1][1]=='g'||argv[1][1]=='G'){gpu=true;}
    else{ gpu=true; } //$B%G%U%)%k%H$r(Bgpu$B$H$9$k(B
    argstart=2;
  }
  if(argc<argstart){
    printf("Usage: %s [-c|-g|-r|-s] n steps\n",argv[0]);
    printf("  -r: CPUR only\n");
    printf("  -c: CPU only\n");
    printf("  -g: GPU only\n");
  }
  if(cpu){ printf("\n\n$B%-%c%j!<%A%'!<%s(B $BHs:F5"(B \n"); }
  else if(cpur){ printf("\n\n$B%-%c%j!<%A%'!<%s(B $B:F5"(B \n"); }
  else if(gpu){ printf("\n\n$B%-%c%j!<%A%'!<%s(B GPGPU/CUDA \n"); }
  if(cpu||cpur){
    int min=4;
    int targetN=17;
    struct timeval t0;
    struct timeval t1;
    printf("%s\n"," N:           Total           Unique          dd:hh:mm:ss.ms");
    for(int size=min;size<=targetN;size++){
      TOTAL=UNIQUE=0;
      gettimeofday(&t0, NULL);//$B7WB,3+;O(B
      if(cpur){ //$B:F5"(B
        // bluteForce_R(size,0);//$B%V%k!<%H%U%)!<%9(B
        // backTracking_R(size,0); //$B%P%C%/%H%i%C%/(B
        // postFlag_R(size,0);     //$BG[CV%U%i%0(B
        // bitmap_R(size,0,0,0,0); //$B%S%C%H%^%C%W(B
        // mirror_R(size);         //$B%_%i!<(B
        // symmetry_R(size);       //$BBP>N2r=|K!(B
        // g.size=size;
        // carryChain();           //$B%-%c%j!<%A%'!<%s(B
        nodeLayer(size);
      }
      if(cpu){ //$BHs:F5"(B
        //bluteForce_NR(size,0);//$B%V%k!<%H%U%)!<%9(B
        // backTracking_NR(size,0);//$B%P%C%/%H%i%C%/(B
        // postFlag_NR(size,0);     //$BG[CV%U%i%0(B
        // bitmap_NR(size,0);  //$B%S%C%H%^%C%W(B
        // mirror_NR(size);         //$B%_%i!<(B
        // symmetry_NR(size);       //$BBP>N2r=|K!(B
        // g.size=size;
        // carryChain();           //$B%-%c%j!<%A%'!<%s(B
        nodeLayer(size);
      }
      //
      gettimeofday(&t1, NULL);//$B7WB,=*N;(B
      int ss;int ms;int dd;
      if(t1.tv_usec<t0.tv_usec) {
        dd=(t1.tv_sec-t0.tv_sec-1)/86400;
        ss=(t1.tv_sec-t0.tv_sec-1)%86400;
        ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
      }else {
        dd=(t1.tv_sec-t0.tv_sec)/86400;
        ss=(t1.tv_sec-t0.tv_sec)%86400;
        ms=(t1.tv_usec-t0.tv_usec+500)/10000;
      }//end if
      int hh=ss/3600;
      int mm=(ss-hh*3600)/60;
      ss%=60;
      printf("%2d:%16ld%17ld%12.2d:%02d:%02d:%02d.%02d\n",
          size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
    } //end for
  }//end if
  if(gpu||sgpu){
    if(!InitCUDA()){return 0;}
    int min=4;
    int targetN=21;
    struct timeval t0;
    struct timeval t1;
    printf("%s\n"," N:        Total      Unique      dd:hh:mm:ss.ms");
    for(int size=min;size<=targetN;size++){
      gettimeofday(&t0,NULL);   // $B7WB,3+;O(B
      if(gpu){
        TOTAL=UNIQUE=0;
        nodeLayer(size);
      }
      gettimeofday(&t1,NULL);   // $B7WB,=*N;(B
      int ss;int ms;int dd;
      if (t1.tv_usec<t0.tv_usec) {
        dd=(int)(t1.tv_sec-t0.tv_sec-1)/86400;
        ss=(t1.tv_sec-t0.tv_sec-1)%86400;
        ms=(1000000+t1.tv_usec-t0.tv_usec+500)/10000;
      } else {
        dd=(int)(t1.tv_sec-t0.tv_sec)/86400;
        ss=(t1.tv_sec-t0.tv_sec)%86400;
        ms=(t1.tv_usec-t0.tv_usec+500)/10000;
      }//end if
      int hh=ss/3600;
      int mm=(ss-hh*3600)/60;
      ss%=60;
      printf("%2d:%13ld%16ld%4.2d:%02d:%02d:%02d.%02d\n",
          size,TOTAL,UNIQUE,dd,hh,mm,ss,ms);
    }//end for
  }//end if
  return 0;
}
