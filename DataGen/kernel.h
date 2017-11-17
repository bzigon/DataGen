#include <exception>  
#include <cuda_runtime.h>
using namespace std;


#define KernelVectorAddName "KernelVectorAdd"
void LaunchKernelVectorAdd(dim3& gs, dim3&bs, char **argv, int argc, int nextarg);
void QueryKernelVectorAdd(char *, int gs_start, int gs_end, int gs_inc, int numele);

#define KernelVectorAddCBName "KernelVectorAddCB"
void LaunchKernelVectorAddCB(dim3& gs, dim3&bs, char **argv, int argc, int nextarg);
void QueryKernelVectorAddCB(char *, int gs_start, int gs_end, int gs_inc, int numele);

#define KernelVectorAddCBTrigName "KernelVectorAddCBTrig"
void LaunchKernelVectorAddCBTrig(dim3& gs, dim3&bs, char **argv, int argc, int nextarg);
void QueryKernelVectorAddCBTrig(char *, int gs_start, int gs_end, int gs_inc, int numele);

#define KernelVectorAddCBTrigILP2Name "KernelVectorAddCBTrigILP2"
void LaunchKernelVectorAddCBTrigILP2(dim3& gs, dim3&bs, char **argv, int argc, int nextarg);
void QueryKernelVectorAddCBTrigILP2(char *, int gs_start, int gs_end, int gs_inc, int numele);

#define KernelVectorAddCBTrigILP4Name "KernelVectorAddCBTrigILP4"
void LaunchKernelVectorAddCBTrigILP4(dim3& gs, dim3&bs, char **argv, int argc, int nextarg);
void QueryKernelVectorAddCBTrigILP4(char *, int gs_start, int gs_end, int gs_inc, int numele);


#define KernelVectorAddCBTrigILP2_64Name "KernelVectorAddCBTrigILP2_64"
void LaunchKernelVectorAddCBTrigILP2_64(dim3& gs, dim3&bs, char **argv, int argc, int nextarg);
void QueryKernelVectorAddCBTrigILP2_64(char *, int gs_start, int gs_end, int gs_inc, int numele);

void InitRandomSequence(float *dBuffer, int NumberOfValues);

//
// https://devtalk.nvidia.com/default/topic/985255/?comment=5048162
// 
// Macro to catch CUDA errors in CUDA runtime calls
//
// Example usage
//		CUDA_SAFE_CALL (cudaMemset(d_a, 0x00, sizeof(d_a[0]) * opts.len)); // zero
//
//	myKernel << <dimGrid, dimBlock >> >(d_a, d_b, opts.len);
//	CHECK_LAUNCH_ERROR();
//

#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

// Macro to catch CUDA errors in kernel launches
#define CHECK_LAUNCH_ERROR()                                          \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaThreadSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)