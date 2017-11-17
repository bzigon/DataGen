
#include "kernel.h"
#include <cuda_runtime.h>
#include <list>
#include <string>
#include <assert.h>
#include <omp.h>

using namespace::std;


typedef struct KernelVectorAddParams
{
public:
	dim3 m_bs;
	dim3 m_gs;
	int m_NumberOfElements;

	KernelVectorAddParams(int bsx, int bsy, int bsz, int gsx, int gsy, int gsz, int numele) :
		m_bs(bsx,bsy,bsz), 
		m_gs(gsx,gsy,gsz),
		m_NumberOfElements(numele)
	{
		if (bsx < 1)	{ printf("\n***Error bsx < 1\n"); exit(EXIT_FAILURE); }
		if (bsx > 1024) { printf("\n***Error bsx > 1024\n"); exit(EXIT_FAILURE); }
		if (bsy != 1)	{ printf("\n***Error bsy != 1\n"); exit(EXIT_FAILURE); }
		if (bsz != 1)	{ printf("\n***Error bsz != 1\n"); exit(EXIT_FAILURE); }

		if (gsx < 1)	{ printf("\n***Error gsx < 1\n"); exit(EXIT_FAILURE); }
		if (gsy != 1)	{ printf("\n***Error gsy != 1\n"); exit(EXIT_FAILURE); }
		if (gsz != 1)	{ printf("\n***Error gsz != 1\n"); exit(EXIT_FAILURE); }
		if (numele < 1) { printf("\n***Error numele < 1\n"); exit(EXIT_FAILURE); }
	}

} KernelVectorAddParams_t;

void QueryKernelVectorAdd(char *KernelName, int gs_start, int gs_end, int gs_inc, int numele)
{

	list<KernelVectorAddParams_t*> params;

	for (int gsx = gs_start; gsx < gs_end; gsx += gs_inc)
		for (int bsx = 32; bsx < 1025; bsx += 32)
			params.push_back(new KernelVectorAddParams_t(bsx, 1, 1, gsx, 1, 1, numele));

	printf("\n#\n# %s\n#", KernelName);
	list<KernelVectorAddParams_t*>::iterator i = params.begin();
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

__global__ void
kernelVectorAdd(const float *A, const float *B, float *C, int numElements)
{
	const int stride = blockDim.x * gridDim.x;
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = tid; i < numElements; i += stride)
	{
		C[i] = A[i] + B[i];
	}
}

void LaunchKernelVectorAdd(dim3& gs, dim3& bs, char **argv, int argc, int nextarg)
{
	printf("\nPreparing %s", KernelVectorAddName);
	if (strcmp(argv[nextarg], "-numele") == 0)
	{
		printf("\nAllocating RAM");

		cudaError_t err = cudaSuccess;
		int numElements = stoi(argv[nextarg + 1], nullptr);
		size_t size = numElements * sizeof(float);

		KernelVectorAddParams_t Verify(bs.x, bs.y, bs.z, gs.x, gs.y, gs.z, numElements);

		float *h_A = new float[numElements];
		float *h_B = new float[numElements];
		float *h_C = new float[numElements];

		// Verify that allocations succeeded
		if (h_A == NULL || h_B == NULL || h_C == NULL)
		{
			printf("Failed to allocate host vectors in LaunchVectorAddKernel\n");
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

		//printf("\nCopying HOST RAM to GPU");
		//err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
		//if (err != cudaSuccess)
		//{
		//	printf("Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		//	exit(EXIT_FAILURE);
		//}

		//err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
		//if (err != cudaSuccess)
		//{
		//	printf("Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
		//	exit(EXIT_FAILURE);
		//}

		printf("\nLaunching kernel: kernelVectorAdd");
		printf("\n\tgridsize  (%d,%d,%d)", gs.x, gs.y, gs.z);
		printf("\n\tblocksize (%d,%d,%d)", bs.x, bs.y, bs.z);
		printf("\n\tNumElements %d", numElements);

		kernelVectorAdd << <gs, bs >> > (d_A, d_B, d_C, numElements);
		cudaDeviceSynchronize();

		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf("Failed to launch kernelVectorAdd (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			printf("Failed to copy vector A from device to host (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		err = cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			printf("Failed to copy vector B from device to host (error code %s)!\n", cudaGetErrorString(err));
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
			if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
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
		delete[]h_B;
		delete[]h_C;
	}
	else
	{
		printf("\nExpecting -numele, but saw %s", argv[nextarg]);
		exit(EXIT_FAILURE);
	}
}