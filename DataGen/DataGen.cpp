//
// DataGen.cpp -- program to ...
//	1. query all kernels for their configurations, or
//  2. execute a given kernel with a given configuration
//
#include <iostream>
#include <string>
#include <exception>   
#include "kernel.h"
#include <cuda_runtime.h>
#include <chrono>

class Timer
{
public:
	Timer() : m_beg(m_clock::now()) {}
	void reset() { m_beg = m_clock::now(); }

	double elapsed() const {
		return std::chrono::duration_cast<m_second>
			(m_clock::now() - m_beg).count();
	}

private:
	typedef std::chrono::high_resolution_clock m_clock;
	typedef std::chrono::duration<double, std::ratio<1> > m_second;
	std::chrono::time_point<m_clock> m_beg;
};

// TitanX = SM52
// K40c   = SM35
// TitanXp= SM61

using namespace std;

typedef struct QueryKernelPair
{
	char *FuncName;
	void (*QueryFunc)(char *, int, int, int, int);
	void (*KernelFunc)(dim3&, dim3&, char**, int, int);
} QueryKernelPair_t;

QueryKernelPair_t FunctionTable[] =
{
	KernelVectorAddName,				QueryKernelVectorAdd,				LaunchKernelVectorAdd,
	KernelVectorAddCBName,				QueryKernelVectorAddCB,				LaunchKernelVectorAddCB,
	KernelVectorAddCBTrigName,			QueryKernelVectorAddCBTrig,			LaunchKernelVectorAddCBTrig,
	KernelVectorAddCBTrigILP2Name,		QueryKernelVectorAddCBTrigILP2,		LaunchKernelVectorAddCBTrigILP2,
	KernelVectorAddCBTrigILP4Name,		QueryKernelVectorAddCBTrigILP4,		LaunchKernelVectorAddCBTrigILP4,
	KernelVectorAddCBTrigILP2_64Name,	QueryKernelVectorAddCBTrigILP2_64,	LaunchKernelVectorAddCBTrigILP2_64,
};

const int FunctionTableLen = sizeof(FunctionTable) / sizeof(FunctionTable[0]);

//
// example calling sequence
//
// datagen -q
//	or
// datagen -x kernelVectorAdd -bs 16,16,1 -gs 10,1,1 -numele 1000000
//
// -q = query all kernels for their configurations
// -x = execute a kernel
//
int main(int argc, char **argv) 
{
	if (argc < 2)
	{
		printf("\ndatagen <option>");
		printf("\n\t-q = query all kernels for their configurations");
		printf("\n\t-x = optionally compile and execute a kernel");
		exit(EXIT_FAILURE);
	}

	if (strcmp(argv[1], "-q") == 0)
	{
		const int numele	= 64 * 1024 * 1024;
		const int gs_start	= 128;
		const int gs_inc	= 128;
		const int gs_end	= 1024 + 128;

		for (int i = 0; i < FunctionTableLen; i++)
			FunctionTable[i].QueryFunc(FunctionTable[i].FuncName, gs_start, gs_end, gs_inc, numele);

		exit(EXIT_SUCCESS);
	}
	else
	if (strcmp(argv[1], "-x") == 0)
	{
		char* KernelName = argv[2];

		const int BSIndex = 3;
		if (strcmp(argv[BSIndex], "-bs") == 0)
		{ 
			const int GSIndex = BSIndex + 2;
			dim3 bs;
			string::size_type sz0,sz1;
			bs.x = stoi(argv[BSIndex + 1], &sz0);
			bs.y = stoi(argv[BSIndex + 1] + sz0 + 1, &sz1);
			bs.z = stoi(argv[BSIndex + 1] + sz0 + sz1 + 2, nullptr);

			if (strcmp(argv[GSIndex], "-gs") == 0)
			{
				dim3 gs;
				string::size_type sz2, sz3;
				gs.x = stoi(argv[GSIndex + 1], &sz2);
				gs.y = stoi(argv[GSIndex + 1] + sz2 + 1, &sz3);
				gs.z = stoi(argv[GSIndex + 1] + sz2 + sz3 + 2, nullptr);

				for (int i = 0; i < FunctionTableLen; i++)
					if (strcmp(KernelName, FunctionTable[i].FuncName) == 0)
					{
						Timer tmr;

						FunctionTable[i].KernelFunc(gs, bs, argv, argc, GSIndex + 2);
						cudaDeviceReset();

						double DeltaTime = tmr.elapsed();
						cout << "Elapsed time " << DeltaTime << " seconds." << endl;

						exit(EXIT_SUCCESS);
						break;
					}
				printf("\nCant find kernel %s to launch.", KernelName);
				exit(EXIT_FAILURE);
			}
			else
			{
				printf("\nExpecting -gs, but saw %s", argv[GSIndex]);
				exit(EXIT_FAILURE);
			}
		}
		else
		{
			printf("\nExpecting -bs, but saw %s", argv[BSIndex]);
			exit(EXIT_FAILURE);
		}
	}
	else
	{
		printf("\nUnrecognized option %s", argv[1]);
		exit(EXIT_FAILURE);
	}
	exit(EXIT_SUCCESS);
}
