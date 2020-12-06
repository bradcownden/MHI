#pragma once
#include "cuda_runtime.h"

enum class IoError
{
	SUCCESS,
	FILE_READ_ERROR,
	FILE_WRITE_ERROR
};

extern __host__ void readFile(char* inFile, const int rowsA, double* inPtr);
extern __host__ void writeFile(char* outFile, const int rowsA, double* outPtr);
extern __host__ void getIOError(IoError error);