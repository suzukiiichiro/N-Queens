/* 395c_ctx_holder.cu -- hold a CUDA context (and optionally a device
 * allocation) for N seconds, doing nothing else. Used to test whether the
 * mere presence of another context on the GPU is what makes the CRunner
 * ~4% faster when launched from the Codon dispatcher than directly.
 *
 * Build:  /usr/local/cuda/bin/nvcc -O2 -arch=sm_86 -o 395c_ctx_holder 395c_ctx_holder.cu
 * Run:    ./395c_ctx_holder <alloc_mb> <seconds>
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>
int main(int argc, char **argv) {
    long mb = (argc > 1) ? atol(argv[1]) : 0;
    int  secs = (argc > 2) ? atoi(argv[2]) : 600;
    cudaError_t e = cudaFree(0);                 /* creates the primary context */
    if (e != cudaSuccess) { fprintf(stderr, "[ctx-holder] cudaFree(0) failed: %s\n", cudaGetErrorString(e)); return 1; }
    void *p = NULL;
    if (mb > 0) {
        e = cudaMalloc(&p, (size_t)mb << 20);
        if (e != cudaSuccess) { fprintf(stderr, "[ctx-holder] cudaMalloc(%ld MB) failed: %s\n", mb, cudaGetErrorString(e)); return 1; }
        e = cudaMemset(p, 0, (size_t)mb << 20);  /* touch it so it is really resident */
        if (e != cudaSuccess) { fprintf(stderr, "[ctx-holder] cudaMemset failed: %s\n", cudaGetErrorString(e)); return 1; }
        cudaDeviceSynchronize();
    }
    printf("[ctx-holder] context up pid=%d alloc_mb=%ld sleeping=%ds\n", (int)getpid(), mb, secs);
    fflush(stdout);
    sleep((unsigned)secs);
    if (p) cudaFree(p);
    printf("[ctx-holder] done\n");
    return 0;
}
