
#include "kernel.h"
#include <cuda_runtime.h>
#include <curand_kernel.h> 
#include <curand.h>
#include <list>
#include <string>
#include <assert.h>
#include <omp.h>

using namespace::std;

// ==================================================================
// http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
//
// Kernels like this ...
//
//	kernel<<<1,1>>>(a);
//	gpuErrchk(cudaPeekAtLastError());
//	gpuErrchk(cudaDeviceSynchronize());
//
// API's like this ...
//
// gpuErrchk( cudaMalloc(&a_d, size*sizeof(int)) );
//

#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

// ==================================================================
//
// ==================================================================
typedef struct KernelVectorAddCBParams
{
public:
	dim3 m_bs;
	dim3 m_gs;
	int m_NumberOfElements;

	KernelVectorAddCBParams(int bsx, int bsy, int bsz, int gsx, int gsy, int gsz, int numele) :
		m_bs(bsx, bsy, bsz),
		m_gs(gsx, gsy, gsz),
		m_NumberOfElements(numele)
	{
		if (bsx < 1) { printf("\n***Error bsx < 1\n"); exit(EXIT_FAILURE); }
		if (bsx > 1024) { printf("\n***Error bsx > 1024\n"); exit(EXIT_FAILURE); }
		if (bsy != 1) { printf("\n***Error bsy != 1\n"); exit(EXIT_FAILURE); }
		if (bsz != 1) { printf("\n***Error bsz != 1\n"); exit(EXIT_FAILURE); }

		if (gsx < 1) { printf("\n***Error gsx < 1\n"); exit(EXIT_FAILURE); }
		if (gsy != 1) { printf("\n***Error gsy != 1\n"); exit(EXIT_FAILURE); }
		if (gsz != 1) { printf("\n***Error gsz != 1\n"); exit(EXIT_FAILURE); }
		if (numele < 1) { printf("\n***Error numele < 1\n"); exit(EXIT_FAILURE); }
	}

} KernelVectorAddCBParams_t;

void QueryKernelVectorAddCB(char *KernelName, int bs_start, int bs_end, int bs_inc, int gs_start, int gs_end, int gs_inc, int numele)
{
	list<KernelVectorAddCBParams_t*> params;

	for (int gsx = gs_start; gsx < gs_end; gsx += gs_inc)
		for (int bsx = bs_start; bsx < bs_end; bsx += bs_inc)
			params.push_back(new KernelVectorAddCBParams_t(bsx, 1, 1, gsx, 1, 1, numele));

	printf("#\n# %s\n#", KernelName);
	list<KernelVectorAddCBParams_t*>::iterator i = params.begin();
	printf("\n%s:   compile: params -bs %4d,%d,%d -gs %4d,%d,%d -numele %d",
		KernelName,
		(*i)->m_bs.x,
		(*i)->m_bs.y,
		(*i)->m_bs.z,
		(*i)->m_gs.x,
		(*i)->m_gs.y,
		(*i)->m_gs.z,
		(*i)->m_NumberOfElements);

	for (i++; i != params.end(); ++i)
	{
		printf("\n%s: nocompile: params -bs %4d,%d,%d -gs %4d,%d,%d -numele %d",
			KernelName,
			(*i)->m_bs.x,
			(*i)->m_bs.y,
			(*i)->m_bs.z,
			(*i)->m_gs.x,
			(*i)->m_gs.y,
			(*i)->m_gs.z,
			(*i)->m_NumberOfElements);
	}
	printf("\n");
}

//
// compute bound version of vector add kernel
//
__global__ void
kernelVectorAddCB(const float *A, const float *B, float *C, float K1, float K2, int numElements)
{
	const int stride = blockDim.x * gridDim.x;

	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < numElements; i += stride)
	{
		// C[i] = K1*A[i] + K2*B[i]
		float T1	= A[i];
		float T2	= B[i];
		float T3	= K1;
		float T4	= K2;
		float T5	= T1*T3;
		float T6	= T2*T4;

		C[i]		= T5 + T6;
	}
}

void LaunchKernelVectorAddCB(dim3& gs, dim3& bs, char **argv, int argc, int nextarg)
{
	printf("\nPreparing %s", KernelVectorAddCBName);
	if (strcmp(argv[nextarg], "-numele") == 0)
	{
		printf("\nAllocating RAM");

		cudaError_t err = cudaSuccess;
		int numElements = stoi(argv[nextarg + 1], nullptr);
		size_t size = numElements * sizeof(float);

		KernelVectorAddCBParams_t Verify(bs.x, bs.y, bs.z, gs.x, gs.y, gs.z, numElements);

		float *h_A = new float[numElements];
		float *h_C = new float[numElements];

		// Verify that allocations succeeded
		if (h_A == NULL || h_C == NULL)
		{
			printf("Failed to allocate host vectors in LaunchKernelVectorAddCB\n");
			exit(EXIT_FAILURE);
		}


		float *d_A = NULL;
		err = cudaMalloc((void **)&d_A, size);
		if (err != cudaSuccess)
		{
			printf("Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		float *d_B = NULL;
		err = cudaMalloc((void **)&d_B, size);
		if (err != cudaSuccess)
		{
			printf("Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		float *d_C = NULL;
		err = cudaMalloc((void **)&d_C, size);
		if (err != cudaSuccess)
		{
			printf("Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		printf("\nInitializing GPU RAM");
		InitRandomSequence(d_A, numElements);
		InitRandomSequence(d_B, numElements);

		printf("\nLaunching kernel: kernelVectorAddCB");
		printf("\n\tgridsize  (%d,%d,%d)", gs.x, gs.y, gs.z);
		printf("\n\tblocksize (%d,%d,%d)", bs.x, bs.y, bs.z);
		printf("\n\tNumElements %d", numElements);

		kernelVectorAddCB << <gs, bs >> > (d_A, d_B, d_C, 1.0f, 0.0f, numElements);
		cudaDeviceSynchronize();

		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf("Failed to launch kernelVectorAddCB (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			printf("Failed to copy vector A from device to host (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			printf("Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		// Verify that the result vector is correct
		printf("\nValidating results ...");
#pragma omp parallel for 
		for (int i = 0; i < numElements; ++i)
		{
			if (fabs(h_A[i] - h_C[i]) > 1e-5)
			{
				printf("Result verification failed at element %d!\n", i);
				exit(EXIT_FAILURE);
			}
		}
		printf(" success!\n");

		err = cudaFree(d_A);
		if (err != cudaSuccess)
		{
			printf("Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		err = cudaFree(d_B);
		if (err != cudaSuccess)
		{
			printf("Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		err = cudaFree(d_C);
		if (err != cudaSuccess)
		{
			printf("Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		delete[]h_A;
		delete[]h_C;
	}
	else
	{
		printf("\nExpecting -numele, but saw %s", argv[nextarg]);
		exit(EXIT_FAILURE);
	}
}

#define NumThreadsPerBlock 256

__global__ void GenerateRandomSequenceKernel(float *Dest, unsigned int seed, int NumberOfValues)
{
	const int gtid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ curandState_t RT[NumThreadsPerBlock];

	curand_init((gtid << 8) + seed, 0, 0, &RT[threadIdx.x]);

	for (int i = gtid; i < NumberOfValues; i += blockDim.x*gridDim.x)
	{
		float T = curand_normal(&RT[threadIdx.x]);
		Dest[i] = T;
	}

}


void InitRandomSequence(float *dBuffer, int NumberOfValues)
{
	GenerateRandomSequenceKernel << <30 * 32, NumThreadsPerBlock >> > (dBuffer, (unsigned int)time(0), NumberOfValues);
	CHECK_LAUNCH_ERROR();
//	cudaDeviceSynchronize();
//	cudaError_t err = cudaGetLastError();
//	if (err != cudaSuccess)
//	{
//		fprintf(stderr, "GenerateRandomSequenceKernel failed. (error code %s)!\n", cudaGetErrorString(err));
//		exit(EXIT_FAILURE);
//	}
}
