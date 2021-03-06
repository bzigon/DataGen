
#include "kernel.h"
#include <cuda_runtime.h>
#include <list>
#include <string>
#include <assert.h>
#include <omp.h>

using namespace::std;


typedef struct KernelVectorAddCBTrigILP4_128Params
{
public:
	dim3 m_bs;
	dim3 m_gs;
	int m_NumberOfElements;

	KernelVectorAddCBTrigILP4_128Params(int bsx, int bsy, int bsz, int gsx, int gsy, int gsz, int numele) :
		m_bs(bsx, bsy, bsz),
		m_gs(gsx, gsy, gsz),
		m_NumberOfElements(numele)
	{
		if (bsx < 1) { printf("\n***Error bsx < 1\n"); exit(EXIT_FAILURE); }
		if (bsx > 1024) { printf("\n***Error bsx > 128\n"); exit(EXIT_FAILURE); }
		if (bsy != 1) { printf("\n***Error bsy != 1\n"); exit(EXIT_FAILURE); }
		if (bsz != 1) { printf("\n***Error bsz != 1\n"); exit(EXIT_FAILURE); }

		if (gsx < 1) { printf("\n***Error gsx < 1\n"); exit(EXIT_FAILURE); }
		if (gsy != 1) { printf("\n***Error gsy != 1\n"); exit(EXIT_FAILURE); }
		if (gsz != 1) { printf("\n***Error gsz != 1\n"); exit(EXIT_FAILURE); }
		if (numele < 1) { printf("\n***Error numele < 1\n"); exit(EXIT_FAILURE); }
	}

} KernelVectorAddCBTrigILP4_128Params_t;

void QueryKernelVectorAddCBTrigILP4_128(char *KernelName, int bs_start, int bs_end, int bs_inc, int gs_start, int gs_end, int gs_inc, int numele)
{
	list<KernelVectorAddCBTrigILP4_128Params_t*> params;

	for (int gsx = gs_start; gsx < gs_end; gsx += gs_inc)
		for (int bsx = bs_start; bsx < bs_end; bsx += bs_inc)
			params.push_back(new KernelVectorAddCBTrigILP4_128Params_t(bsx, 1, 1, gsx, 1, 1, numele));

	printf("#\n# %s\n#", KernelName);
	list<KernelVectorAddCBTrigILP4_128Params_t*>::iterator i = params.begin();
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
kernelVectorAddCBTrigILP4_128(float *A, float *B, float *C, float K1, float K2, int numElements)
{
	const int stride = blockDim.x * gridDim.x;

	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < numElements / 4; i += stride)
	{
		float4 Ar = reinterpret_cast<float4*>(A)[i];	// Ar.x = A[i], Ar.y = A[i+1], Ar.z = A[i+2], Ar.w = A[i+3]
		float4 Br = reinterpret_cast<float4*>(B)[i];	// Br.x = B[i], Br.y = B[i+1], Br.z = B[i+2], Br.w = B[i+3]

		float T5 = sin(K1);
		float T6 = cos(K2);
		float T7 = sin(K1);
		float T8 = cos(K2);
		float T9 = sin(K1);
		float T10 = cos(K2);
		float T11 = sin(K1);
		float T12 = cos(K2);

		float4 C0, C1;
		C0.x = Ar.x*T5;
		C1.x = Br.x*T6;

		C0.y = Ar.y*T7;
		C1.y = Br.y*T8;

		C0.z = Ar.z*T9;
		C1.z = Br.z*T10;

		C0.w = Ar.w*T11;
		C1.w = Br.w*T12;

		float4 C2;
		C2.x = C0.x + C1.x;
		C2.y = C0.y + C1.y;
		C2.z = C0.z + C1.z;
		C2.w = C0.w + C1.w;

		reinterpret_cast<float4*>(C)[i] = C2;
	}

	// Process remaining elements
	for (int i = blockDim.x * blockIdx.x + threadIdx.x + (numElements / 4) * 4; i < numElements; i += blockDim.x * gridDim.x + threadIdx.x)
	{
		C[i] = sin(K1)*A[i] + cos(K2)*B[i];
	}
}


void LaunchKernelVectorAddCBTrigILP4_128(dim3& gs, dim3& bs, char **argv, int argc, int nextarg)
{
	printf("\nPreparing %s", KernelVectorAddCBTrigILP4_128Name);
	if (strcmp(argv[nextarg], "-numele") == 0)
	{
		printf("\nAllocating RAM");

		cudaError_t err = cudaSuccess;
		int numElements = stoi(argv[nextarg + 1], nullptr);
		size_t size = numElements * sizeof(float);

		KernelVectorAddCBTrigILP4_128Params_t Verify(bs.x, bs.y, bs.z, gs.x, gs.y, gs.z, numElements);

		float *h_B = new float[numElements];
		float *h_C = new float[numElements];

		// Verify that allocations succeeded
		if (h_B == NULL || h_C == NULL)
		{
			printf("Failed to allocate host vectors in LaunchKernelVectorAddCBTrigILP4_128\n");
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

		printf("\nLaunching kernel: kernelVectorAddCBTrigILP4_128");
		printf("\n\tgridsize  (%d,%d,%d)", gs.x, gs.y, gs.z);
		printf("\n\tblocksize (%d,%d,%d)", bs.x, bs.y, bs.z);
		printf("\n\tNumElements %d", numElements);

		kernelVectorAddCBTrigILP4_128 << <gs, bs >> > (d_A, d_B, d_C, 0.0f, 0.0f, numElements);
		cudaDeviceSynchronize();

		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			printf("Failed to launch kernelVectorAddCBTrigILP4_128 (error code %s)!\n", cudaGetErrorString(err));
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
			if (fabs(h_B[i] - h_C[i]) > 1e-5)
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

		delete[]h_B;
		delete[]h_C;
	}
	else
	{
		printf("\nExpecting -numele, but saw %s", argv[nextarg]);
		exit(EXIT_FAILURE);
	}
}