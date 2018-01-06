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

typedef struct QueryTrainingKernelPair
{
	char *FuncName;
	void (*QueryFunc)(char *, int, int, int, int, int, int, int);
	void (*KernelFunc)(dim3&, dim3&, char**, int, int);
} QueryTrainingKernelPair_t;

QueryTrainingKernelPair_t TrainingTable[] =
{
	KernelVectorAddName,				QueryKernelVectorAdd,				LaunchKernelVectorAdd,
	KernelVectorAddCBName,				QueryKernelVectorAddCB,				LaunchKernelVectorAddCB,
	KernelVectorAddCBTrigName,			QueryKernelVectorAddCBTrig,			LaunchKernelVectorAddCBTrig,
	KernelVectorAddCBTrigILP2Name,		QueryKernelVectorAddCBTrigILP2,		LaunchKernelVectorAddCBTrigILP2,
	KernelVectorAddCBTrigILP2_64Name,	QueryKernelVectorAddCBTrigILP2_64,	LaunchKernelVectorAddCBTrigILP2_64,
	KernelVectorAddCBTrigILP4Name,		QueryKernelVectorAddCBTrigILP4,		LaunchKernelVectorAddCBTrigILP4,
	KernelVectorAddCBTrigILP4_128Name,	QueryKernelVectorAddCBTrigILP4_128,	LaunchKernelVectorAddCBTrigILP4_128,
};

const int TrainingTableLen = sizeof(TrainingTable) / sizeof(TrainingTable[0]);

typedef struct QueryValidationKernelPair
{
	char *FuncName;
	void(*QueryFunc)(char *);
	void(*KernelFunc)(dim3&, dim3&, char**, int, int);
} QueryValidationKernelPair_t;

QueryValidationKernelPair_t ValidationTable[] =
{
	KernelMatMultName,					QueryKernelMatMult,					LaunchKernelMatMult,
	KernelMatMultFastName,				QueryKernelMatMultFast,				LaunchKernelMatMultFast
};

const int ValidationTableLen = sizeof(ValidationTable) / sizeof(ValidationTable[0]);


//
// example calling sequence
//
// datagen -q
//	or
// datagen -v
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
		printf("\n\t-q = query all training kernels for their configurations");
		printf("\n\t-v = query all validation kernels for their configurations");
		printf("\n\t-x = optionally compile and execute a kernel");
		exit(EXIT_FAILURE);
	}

	cudaDeviceProp prop;
	const int DevNum = 0;
	cudaGetDeviceProperties(&prop, DevNum);
	cudaError_t err = cudaSetDevice(DevNum);
#if 0
	if (strcmp(prop.name, "TITAN Xp") != 0)		// 6.1 board
	{
		printf("\n***Device 0 is not TITAN Xp. Found ... %s", prop.name);
		exit(EXIT_FAILURE);
	}
#endif

#if 1
	if (strcmp(prop.name, "TITAN V") != 0)		// 6.1 board
	{
		printf("\n***Device 0 is not TITAN V. Found ... %s", prop.name);
		exit(EXIT_FAILURE);
	}
#endif

#if 0
	if (strcmp(prop.name, "GeForce GTX TITAN X") != 0)		// 5.2 board
	{
		printf("\n***Device 0 is not GeForce GTX TITAN X. Found ... %s", prop.name);
		exit(EXIT_FAILURE);
	}
#endif

	if (strcmp(argv[1], "-q") == 0)
	{
		const int numele = 200000033;	// a prime number
		const int gs_start = 128;
		const int gs_inc = 128;
		const int gs_end = 1024 + 128;

		for (int i = 0; i < TrainingTableLen; i++)
			TrainingTable[i].QueryFunc(TrainingTable[i].FuncName, 32, 1025, 16, gs_start, gs_end, gs_inc, numele);

		exit(EXIT_SUCCESS);
	}
	else
	if (strcmp(argv[1], "-v") == 0)
	{
		for (int i = 0; i < ValidationTableLen; i++)
			ValidationTable[i].QueryFunc(ValidationTable[i].FuncName);

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

				for (int i = 0; i < TrainingTableLen; i++)
					if (strcmp(KernelName, TrainingTable[i].FuncName) == 0)
					{
						Timer tmr;

						TrainingTable[i].KernelFunc(gs, bs, argv, argc, GSIndex + 2);
						cudaDeviceReset();

						double DeltaTime = tmr.elapsed();
						cout << "\nElapsed time " << DeltaTime << " seconds." << endl;

						exit(EXIT_SUCCESS);
						break;
					}

				for (int i = 0; i < ValidationTableLen; i++)
					if (strcmp(KernelName, ValidationTable[i].FuncName) == 0)
					{
						Timer tmr;

						ValidationTable[i].KernelFunc(gs, bs, argv, argc, GSIndex + 2);
						cudaDeviceReset();

						double DeltaTime = tmr.elapsed();
						cout << "\nElapsed time " << DeltaTime << " seconds." << endl;

						exit(EXIT_SUCCESS);
						break;
					}
				printf("\nCant find kernel %s (in training or validation table) to launch.", KernelName);
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
