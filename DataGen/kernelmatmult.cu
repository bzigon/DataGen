
#include "kernel.h"
#include <cuda_runtime.h>
#include <list>
#include <string>
#include <assert.h>
#include <omp.h>
#include <xmmintrin.h>

using namespace::std;


typedef struct KernelMatMultParams
{
public:
	dim3 m_bs;
	dim3 m_gs;
	int m_NumberOfElements;

	KernelMatMultParams(int bsx, int bsy, int bsz, int gsx, int gsy, int gsz, int numele) :
		m_bs(bsx, bsy, bsz),
		m_gs(gsx, gsy, gsz),
		m_NumberOfElements(numele)
	{
		if (bsx != 8 && bsx != 16 && bsx != 32) { printf("\n***Error bsx !=8,16,32\n"); exit(EXIT_FAILURE); }
		if (bsy != 8 && bsy != 16 && bsy != 32) { printf("\n***Error bsy !=8,16,32\n"); exit(EXIT_FAILURE); }
		if (bsz != 1) { printf("\n***Error bsz != 1\n"); exit(EXIT_FAILURE); }

		if (gsx < 1) { printf("\n***Error gsx < 1\n"); exit(EXIT_FAILURE); }
		if (gsy < 1) { printf("\n***Error gsy < 1\n"); exit(EXIT_FAILURE); }
		if (gsz != 1) { printf("\n***Error gsz != 1\n"); exit(EXIT_FAILURE); }
		if (numele < 1) { printf("\n***Error numele < 1\n"); exit(EXIT_FAILURE); }

		if (bsx*gsx*bsy*gsy != numele) { printf("\n***Error bsx*gsx*bsy*gsy != numele (%d,%d,%d,%d,%d)\n", bsx, gsx, bsy, gsy, numele); exit(EXIT_FAILURE); }
	}

} KernelMatMultParams_t;

void QueryKernelMatMult(char *KernelName)
{
	int bsize[] = { 8, 16, 32, 0 };
	int gsize[] = { 16, 32, 64, 128, 256, 0 };

	list<KernelMatMultParams_t*> params;

	for (int *gs = &gsize[0]; *gs != 0; gs++)
		for (int *bs = &bsize[0]; *bs != 0; bs++)
		{
			int ms = (*gs)*(*bs);
			params.push_back(new KernelMatMultParams_t(*bs, *bs, 1, *gs, *gs, 1, ms));
		}

	printf("\n#\n# %s\n#", KernelName);
	list<KernelMatMultParams_t*>::iterator i = params.begin();
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


// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
	int width;
	int height;
	float* elements;
} Matrix;

// Thread block size
//#define BLOCK_SIZE 16

__global__ void kernelMatMult(Matrix A, Matrix B, Matrix C)
{
	// Each thread computes one element of C
	// by accumulating results into Cvalue
	float Cvalue = 0;
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row > A.height || col > B.width)
		return;

	for (int e = 0; e < A.width; ++e)
	{
			Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
	}
	C.elements[row * C.width + col] = Cvalue;
}

static inline float _mm256_reduce_add_ps(__m256 x) {
	/* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
	const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
	/* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
	const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
	/* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
	const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
	/* Conversion to float is a no-op on x86-64 */
	return _mm_cvtss_f32(x32);
}

void LaunchKernelMatMult(dim3& gs, dim3& bs, char **argv, int argc, int nextarg)
{
	printf("\nPreparing %s", KernelMatMultName);
	if (strcmp(argv[nextarg], "-numele") == 0)
	{
		printf("\nAllocating RAM");

		cudaError_t err = cudaSuccess;
		int numElements = stoi(argv[nextarg + 1], nullptr);
		const int numElementsSq = numElements*numElements;

		KernelMatMultParams_t Verify(bs.x, bs.y, bs.z, gs.x, gs.y, gs.z, numElementsSq);

		Matrix d_A;
		d_A.width = numElements; d_A.height = numElements;
		size_t size = d_A.width * d_A.height * sizeof(float);
		err = cudaMalloc(&d_A.elements, size);
		if (err != cudaSuccess)
		{
			printf("Failed to allocate device matrix A (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		Matrix d_B;
		d_B.width = numElements; d_B.height = numElements;
		size = d_B.width * d_B.height * sizeof(float);
		err = cudaMalloc(&d_B.elements, size);
		if (err != cudaSuccess)
		{
			printf("Failed to allocate device matrix B (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		Matrix d_C;
		d_C.width = numElements; d_C.height = numElements;
		size = d_C.width * d_C.height * sizeof(float);
		err = cudaMalloc(&d_C.elements, size);
		if (err != cudaSuccess)
		{
			printf("Failed to allocate device matrix C (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		float *h_A = new float[numElementsSq];
		float *h_B = new float[numElementsSq];
		float *h_C = new float[numElementsSq];
		float *h_AB= new float[numElementsSq];

		// Verify that allocations succeeded
		if (h_A == NULL || h_B == NULL || h_C == NULL || h_AB == NULL)
		{
			printf("Failed to allocate host vectors in LaunchKernelMatMult\n");
			exit(EXIT_FAILURE);
		}

		printf("\nInitializing GPU RAM");
		InitRandomSequence(d_A.elements, numElementsSq);
		InitRandomSequence(d_B.elements, numElementsSq);


		printf("\nLaunching kernel: kernelMatMult");
		printf("\n\tgridsize  (%d,%d,%d)", gs.x, gs.y, gs.z);
		printf("\n\tblocksize (%d,%d,%d)", bs.x, bs.y, bs.z);
		printf("\n\tNumElements %d", numElementsSq);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);

		kernelMatMult << <gs, bs >> > (d_A, d_B, d_C);
//		CHECK_LAUNCH_ERROR();
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);

		err = cudaMemcpy(h_A, d_A.elements, size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			printf("Failed to copy matrix A from device to host (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		err = cudaMemcpy(h_B, d_B.elements, size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			printf("Failed to copy matrix B from device to host (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		err = cudaMemcpy(h_C, d_C.elements, size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			printf("Failed to copy matrix C from device to host (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}


		//
		// Now compute AB= A*B on the host so that we can compare it with the GPU.
		//
		// First transpose h_B matrix into h_T using SSE.
		// Then use OpenMP and AVX to perform the matrix multiplication 8 floats at a time.
		//
		printf("\nGPU finished %f milliseconds.\nComputing host solution ...", milliseconds);
		{
			float *h_T = new float[numElementsSq];

			for (int i = 0; i < numElements; i += 4)
			{
				for (int j = 0; j < numElements; j += 4)
				{
					__m128 B[4];
					for (int k = 0; k < 4; k++)
					{
						B[k] = _mm_load_ps(&h_B[(i + k)*numElements + j]);
					}

					_MM_TRANSPOSE4_PS(B[0], B[1], B[2], B[3]);

					for (int k = 0; k < 4; k++)
						_mm_store_ps(&h_T[(j + k)*numElements + i], B[k]);

				}
			}

#pragma omp parallel for
			for (int i = 0; i < numElements; i++)
			{
				for (int j = 0; j < numElements; j++)
				{
					__m256  T = _mm256_setzero_ps();

					for (int k = 0; k < numElements; k += 8)
					{
						__m256 A1 = _mm256_load_ps(&h_A[i*numElements + k]);
						__m256 T1 = _mm256_load_ps(&h_T[j*numElements + k]);
						__m256 C = _mm256_mul_ps(A1, T1);
						T = _mm256_add_ps(C, T);
					}

					float Q = _mm256_reduce_add_ps(T);
					h_AB[i*numElements + j] = Q;
				}
			}

			delete[]h_T;
		}


		// Verify that the result vector is correct
		printf("\nValidating results ...");
#pragma omp parallel for
		for (int i = 0; i < numElementsSq; ++i)
		{
			float T1 = h_AB[i];
			float T2 = h_C[i];
			if (fabs(T1-T2) > 0.009f)
			{
				printf("Result verification failed at element %d!\n", i);
				printf("h_AB[%d] = %f, h_C[%d]=%f\n", i, h_AB[i], i, h_C[i]);
				exit(EXIT_FAILURE);
			}
		}
		printf(" success!\n");

		err = cudaFree(d_A.elements);
		if (err != cudaSuccess)
		{
			printf("Failed to free device matrix A (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		err = cudaFree(d_B.elements);
		if (err != cudaSuccess)
		{
			printf("Failed to free device matrix B (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		err = cudaFree(d_C.elements);
		if (err != cudaSuccess)
		{
			printf("Failed to free device matrix C (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		delete[]h_A;
		delete[]h_B;
		delete[]h_C;
		delete[]h_AB;
	}
	else
	{
		printf("\nExpecting -numele, but saw %s", argv[nextarg]);
		exit(EXIT_FAILURE);
	}
}


