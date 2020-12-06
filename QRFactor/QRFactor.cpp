/*
 * Copyright 2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cstdio>
#include <assert.h>
#include <iostream>
#include <chrono>

#include "cusolverSp.h"

#include "cusolverSp_LOWLEVEL_PREVIEW.h"

#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "helper_cusolver.h"

#include "QRFactor_Helper.h"

// Ignore unsafe/deprecated warnings for sprintf and fopen
#pragma warning(disable : 4996)

// For loading the sparse matrix
template <typename T_ELEM>
int loadMMSparseMatrix(
    char* filename,
    char elem_type,
    bool csrFormat,
    int* m,
    int* n,
    int* nnz,
    T_ELEM** aVal,
    int** aRowInd,
    int** aColInd,
    int extendSymMatrix);

// Check for input file existing
bool fileExists(const char* fname)
{
    std::ifstream ifile(fname);
    bool out = (ifile) ? true : false;
    return out;
}

// Check for existing/applicable GPUs
bool gpuDetect(int argc, char* argv[], struct QRfactorOpts& qropts)
{
    const int t_verb = qropts.verbose; // Get verbosity
#ifdef DEBUG
    if (t_verb > 1) // Regular verbosity will give device ID with compute capability
    {
        printf("Detecting CUDA devices...\n");
        findCudaDevice(argc, (const char**)argv);
    }
    else if (t_verb == 1) // Minimum verbosity suppresses all output
    {
        int devID = 0;
        if (checkCmdLineFlag(argc, (const char**)argv, "device")) 
        {
            devID = getCmdLineArgumentInt(argc, (const char**)argv, "device=");

            if (devID < 0) {
                printf("Invalid command line parameter for gpuDetect. Expecting 'device='\n ");
                exit(EXIT_FAILURE);
            }
            else {
                devID = gpuDeviceInit(devID);

                if (devID < 0) {
                    printf("exiting...\n");
                    exit(EXIT_FAILURE);
                }
            }
        }
        else {
            // Otherwise pick the device with highest Gflops/s
            devID = gpuGetMaxGflopsDeviceId();
            checkCudaErrors(cudaSetDevice(devID));
        }
    }
    else
    {
        fprintf(stderr, "\nERROR in gpuDetect(): verbosity must be [1,3] but recieved %d\n",
            t_verb);
        exit(EXIT_FAILURE);
    }
#endif // DEBUG
    return true;
}
// Usage instructions for command line execution
void UsageSP(void)
{
    printf("Usage: QRFactor -matrix=<filename> -data=<fileroot> -v=<int>\n");
    printf("-h: display this help\n");
    printf("-matrix=<filename>: filename containing a matrix in MM format\n");
    printf("-data=<fileroot>: filenames for b(t) inputs. Expects format <dir>/<fileroot>_t%%d.txt\n");
    printf("-v=<int>: value of output verbosity 1, 2, 3\n");
    printf("\tv=1: minimal output to terminal only\n");
    printf("\tv=2: greater output to terminal only\n");
    printf("\tv=3: output to terminal and data files written to current directory\n");

    exit(0);
}

// Load command line arguments based on presence of flags
void parseCommandLineArguments(int argc, char* argv[], struct QRfactorOpts& qropts)
{
    memset(&qropts, 0, sizeof(qropts));

    // Return usage message if help flag or no arguments given
    if (checkCmdLineFlag(argc, (const char**)argv, "-h") || argc == 1)
    {
        UsageSP();
    }

    // Matrix file flag
    if (checkCmdLineFlag(argc, (const char**)argv, "matrix"))
    {
        char* fileName = 0;
        getCmdLineArgumentString(argc, (const char**)argv, "matrix", &fileName);

        if (fileName)
        {
            qropts.sparse_mat_filename = fileName;
        }
        else
        {
            printf("\nIncorrect filename passed to -matrix\n ");
            UsageSP();
        }
    }

    // Input data directory flag
    if (checkCmdLineFlag(argc, (const char**)argv, "data"))
    {
        char* inData = 0;
        getCmdLineArgumentString(argc, (const char**)argv, "data", &inData);

        if (inData)
        {
            qropts.data_files = inData;
        }
        else
        {
            printf("\nIncorrect data directory passed to -data\n ");
            UsageSP();
        }
    }

    // Verbosity flag
    if (checkCmdLineFlag(argc, (const char**)argv, "v"))
    {
        qropts.verbose = getCmdLineArgumentInt(argc, (const char**)argv, "v");

        if (qropts.verbose == 0 || qropts.verbose > 3)
        {
            printf("\nIncorrect verbosity level passed to -v. Defaulting to 1\n");
            qropts.verbose = 1;
        }
    }
    else
    {
        qropts.verbose = 1;
    }

    if (qropts.verbose > 1)
    {
        printf("Input parameters are\n");
        printf("Input matrix: %s\n", qropts.sparse_mat_filename);
        printf("Data directory: %s\n", qropts.data_files);
        printf("Verbosity: %d\n\n", qropts.verbose);
    }
}

// Read the input data file
void readB(char* inFile, const int rowsA, double* inPtr, const int v) 
{
    if (inFile)
    {   
        std::ifstream file(inFile);
        if (file.is_open())
        {
#ifdef DEBUG
            if (v > 1) printf("Reading data from %s\n", inFile);
#endif // DEBUG
            std::string line;
            std::string::size_type val;
            int count = 0;
            while (count < rowsA) // Read the correct number of rows
            {
                getline(file, line);
                inPtr[count] = std::stod(line, &val);
                ++count;
            }
        }
    }
    else
    {
#ifdef DEBUG
        printf("Error: couldn't find file %s\n", inFile);
#endif // DEBUG
    }
}

int main(int argc, char* argv[])
{
    struct QRfactorOpts qropts; // command line inputs 
    cusolverSpHandle_t cusolverSpH = NULL; // handle for sparse matrix
    cusparseHandle_t   cusparseH = NULL;   // residual evaluation
    cudaStream_t stream = NULL;
    cusparseMatDescr_t descrA = NULL; // descriptor for data type, symmetry

    csrqrInfo_t d_info = NULL; // opaque info structure for QR factoring

    cudaEvent_t factor_start, factor_stop, solve_start, solve_stop; // CUDA timing events
    cudaEventCreate(&factor_start);
    cudaEventCreate(&factor_stop);
    cudaEventCreate(&solve_start);
    cudaEventCreate(&solve_stop);
    float gpufactor_time = 0.;
    float gpusolve_time = 0.;

    int rowsA = 0; // number of rows of A
    int colsA = 0; // number of columns of A
    int nnzA = 0; // number of nonzeros of A
    int baseA = 0; // base index in CSR format

    // CSR(A) from I/O
    int* h_csrRowPtrA = NULL; // <int> n+1
    int* h_csrColIndA = NULL; // <int> nnzA
    double* h_csrValA = NULL; // <double> nnzA

    double* h_x = NULL; // <double> n,  x = A \ b
    double* h_b = NULL; // <double> n, b = ones(m,1)
    double* h_bcopy = NULL; // <double> n, b = ones(m,1)
    double* h_r = NULL; // <double> n, r = b - A*x

    size_t size_internal = 0;
    size_t size_chol = 0; // size of working space for csrlu
    void* buffer_gpu = NULL; // working space for factoring

    int* d_csrRowPtrA = NULL; // <int> n+1
    int* d_csrColIndA = NULL; // <int> nnzA
    double* d_csrValA = NULL; // <double> nnzA
    double* d_x = NULL; // <double> n, x = A \ b
    double* d_b = NULL; // <double> n, a copy of h_b
    double* d_r = NULL; // <double> n, r = b - A*x

    const double zero = 0.0; 
    const double tol = 1.e-16; // tolerance for invertibility
    int singularity = 0; // singularity is -1 if A is invertible under tol

#ifdef DEBUG
    printf("/********************************************/\n\n");
    printf("           Starting QRFactor...\n\n");
    printf("/********************************************/\n\n");
#endif // DEBUG

    // Get command line arguments
    parseCommandLineArguments(argc, argv, qropts);
    const int verb = qropts.verbose; // Sets the verbosity

    // Set device to be either the specified device or the best available device
    gpuDetect(argc, argv, qropts);
    // findCudaDevice(argc, (const char**)argv);
    
    // Initial matrix read
    if (qropts.sparse_mat_filename == NULL)
    {
        qropts.sparse_mat_filename = sdkFindFilePath("lap2D_5pt_n32.mtx", argv[0]);
        if (qropts.sparse_mat_filename != NULL)
        {
            if (verb > 1)
                printf("No input matrix detected. Using default input file [%s]\n", qropts.sparse_mat_filename);
        }
        else
        {
            fprintf(stderr, "Error: could not find default matrix lap2D_5pt_n32.mtx\n");
            printf("Exiting program...\n");
            return 1;
        }
    }
    else
    {
        if (verb > 1)
        {
#ifdef DEBUG
            printf("Using input file [%s]\n", qropts.sparse_mat_filename);
#endif // DEBUG
        }
    }

    if (qropts.sparse_mat_filename) {
        if (loadMMSparseMatrix<double>(qropts.sparse_mat_filename, 'd', true, &rowsA, &colsA,
            &nnzA, &h_csrValA, &h_csrRowPtrA, &h_csrColIndA, true)) {
            return 1;
        }
        baseA = h_csrRowPtrA[0]; // baseA = {0,1}
    }
    else {
        fprintf(stderr, "Error: could not find input matrix\n");
        return 1;
    }

    if (rowsA != colsA) {
        fprintf(stderr, "Error: only supports square matrix types\n");
        return 1;
    }

    if (verb > 1)
    {
        printf("Sparse matrix A is %d x %d with %d nonzeros, base=%d\n\n", rowsA, colsA, nnzA, baseA);
    }
    
    // Create matrix handles and bind to streams
    checkCudaErrors(cusolverSpCreate(&cusolverSpH));
    checkCudaErrors(cusparseCreate(&cusparseH));
    checkCudaErrors(cudaStreamCreate(&stream));
    checkCudaErrors(cusolverSpSetStream(cusolverSpH, stream));
    checkCudaErrors(cusparseSetStream(cusparseH, stream));

    // Create matrix descriptor for data type and base value
    checkCudaErrors(cusparseCreateMatDescr(&descrA));
    checkCudaErrors(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    if (baseA)
    {
        checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));
    }
    else
    {
        checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    }

    std::cout << "cusparseMatrixType_t: " << cusparseGetMatType(descrA) << "\n";
    std::cout << "cusparseFillMode_t: " << cusparseGetMatFillMode(descrA) << "\n";
    std::cout << "cusparseDiagType_t: " << cusparseGetMatDiagType(descrA) << "\n";
    std::cout << "cusparseIndexBase_t: " << cusparseGetMatIndexBase(descrA) << "\n";

    // Host variables
    h_x = (double*)malloc(sizeof(double) * colsA);
    h_b = (double*)malloc(sizeof(double) * rowsA);
    h_bcopy = (double*)malloc(sizeof(double) * rowsA);
    h_r = (double*)malloc(sizeof(double) * rowsA);

    assert(NULL != h_x);
    assert(NULL != h_b);
    assert(NULL != h_bcopy);
    assert(NULL != h_r);

    // Device variables
    checkCudaErrors(cudaMalloc((void**)&d_csrRowPtrA, sizeof(int) * (rowsA + 1)));
    checkCudaErrors(cudaMalloc((void**)&d_csrColIndA, sizeof(int) * nnzA));
    checkCudaErrors(cudaMalloc((void**)&d_csrValA, sizeof(double) * nnzA));
    checkCudaErrors(cudaMalloc((void**)&d_x, sizeof(double) * colsA));
    checkCudaErrors(cudaMalloc((void**)&d_b, sizeof(double) * rowsA));
    checkCudaErrors(cudaMalloc((void**)&d_r, sizeof(double) * rowsA));
    
    // Transfer matrix pointers from host to device
    checkCudaErrors(cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, sizeof(int) * (rowsA + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int) * nnzA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrValA, h_csrValA, sizeof(double) * nnzA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_bcopy, sizeof(double) * rowsA, cudaMemcpyHostToDevice));

#ifdef DEBUG
    printf("Starting GPU factoring.\n");
#endif // DEBUG
    
    checkCudaErrors(cudaEventRecord(factor_start)); // Timing for GPU solve

#ifdef DEBUG
    if (verb > 1) printf("Step 1: create opaque matrix data structure\n");
#endif // DEBUG

    // Create opaque data structure
    checkCudaErrors(cusolverSpCreateCsrqrInfo(&d_info));
    

#ifdef DEBUG
    if (verb > 1) printf("Step 2: analyze qr(A)\n");
#endif // DEBUG

    // Analyze qr(A)
    checkCudaErrors(cusolverSpXcsrqrAnalysis(
        cusolverSpH, rowsA, colsA, nnzA,
        descrA, d_csrRowPtrA, d_csrColIndA,
        d_info));


#ifdef DEBUG
    if (verb > 1) printf("Step 3: determine device workspace for qr(A)\n");
#endif // DEBUG

    // Determine workspace requirement for factoring
    checkCudaErrors(cusolverSpDcsrqrBufferInfo(
        cusolverSpH, rowsA, colsA, nnzA,
        descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
        d_info,
        &size_internal,
        &size_chol));

 
#ifdef DEBUG
    if (verb > 1) printf("Step 4: allocate memory on the device for qr(A)\n");
#endif // DEBUG

    // Allocate buffer
    if (buffer_gpu) {
        checkCudaErrors(cudaFree(buffer_gpu));
    }
    checkCudaErrors(cudaMalloc(&buffer_gpu, sizeof(char) * size_chol));

#ifdef DEBUG
    if (verb > 1) printf("Step 5: compute the factored matrices\n");
#endif // DEBUG

    // Compute decomposition
    checkCudaErrors(cusolverSpDcsrqrSetup(
        cusolverSpH, rowsA, colsA, nnzA,
        descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
        zero,
        d_info));
    checkCudaErrors(cusolverSpDcsrqrFactor(
        cusolverSpH, rowsA, colsA, nnzA,
        NULL, NULL,
        d_info,
        buffer_gpu));

#ifdef DEBUG
    if (verb > 1) printf("Step 6: check for singularity\n");
#endif // DEBUG
    
    // Check for singularity condition
    {
        checkCudaErrors(cusolverSpDcsrqrZeroPivot(
            cusolverSpH, d_info, tol, &singularity));
        if (0 <= singularity) {
#ifdef DEBUG
            fprintf(stderr, "Error: A is not invertible, singularity=%d\n", singularity);
#endif // DEBUG
            return 1;
        }
    }

    // Stop timing and calculate GPU factoring time
    checkCudaErrors(cudaEventRecord(factor_stop)); 
    checkCudaErrors(cudaEventSynchronize(factor_stop));
    checkCudaErrors(cudaEventElapsedTime(&gpufactor_time, factor_start, factor_stop));

#ifdef DEBUG
    if (verb > 1) printf("GPU factoring time: %E ms\n", gpufactor_time);
    printf("\nUsing factored matrix to solve for output at each time step.\n");
#endif // DEBUG

    // Loop over all input data
    int bcount = 1;
    while (true)
    {
        // Make the name of the file based on the current time step
        char bfile[500];
        {
            std::string filetemp(qropts.data_files);
            filetemp = filetemp + "_t%d.txt";
            sprintf(bfile, (const char*)filetemp.c_str(), bcount);
        }

        // Continue if file exists
        if (fileExists(bfile))
        {
            // Read the data into the host vector
            readB(bfile, rowsA, h_b, verb);

            // Solving time
            checkCudaErrors(cudaEventRecord(solve_start));

            // Send host input data to device
            checkCudaErrors(cudaMemcpy(d_b, h_b, sizeof(double) * rowsA, cudaMemcpyHostToDevice));
            // Do the solve
            checkCudaErrors(cusolverSpDcsrqrSolve(
                cusolverSpH, rowsA, colsA, d_b, d_x, d_info, buffer_gpu));
            // Bring the result back to the host
            checkCudaErrors(cudaMemcpy(h_x, d_x, sizeof(double) * rowsA, cudaMemcpyDeviceToHost));
            
            // Calculate the timings
            checkCudaErrors(cudaEventRecord(solve_stop));
            checkCudaErrors(cudaEventSynchronize(solve_stop));
            checkCudaErrors(cudaEventElapsedTime(&gpusolve_time, solve_start, solve_stop));

            // Print GPU timing
#ifdef DEBUG
            if (verb > 1) printf("GPU solve time: %E ms\n", gpusolve_time);
#endif // DEBUG
            
            // Highest verbosity writes the result to file
#ifdef DEBUG
            if (verb > 2)
            {
                // Write out GPU data
                printf("Writing out result\n");
                char xfile[500];
                sprintf(xfile, "GPUFactor_t%d.txt", bcount);
                FILE* GPU_out = fopen(xfile, "w");
                if (GPU_out == NULL)
                {
                    fprintf(stderr, "Error: Couldn't write ouput to file\n");
                    exit(1);
                }
                else
                {
                    for (int i = 0; i < rowsA; ++i)
                    {
                        fprintf(GPU_out, "%1.15e\n", h_x[i]);
                    }
                }
                fclose(GPU_out);
            }
#endif // DEBUG

            // Increment the time step and continue
            ++bcount;
        }
        // If input file does not exist, finish the loop
        else
        {
#ifdef DEBUG
            printf("\nEnd of directory. Exiting program.\n");
#endif // DEBUG

            break;
        }
    }

    if (cusolverSpH) { checkCudaErrors(cusolverSpDestroy(cusolverSpH)); }
    if (cusparseH) { checkCudaErrors(cusparseDestroy(cusparseH)); }
    if (stream) { checkCudaErrors(cudaStreamDestroy(stream)); }
    if (descrA) { checkCudaErrors(cusparseDestroyMatDescr(descrA)); }
    if (d_info) { checkCudaErrors(cusolverSpDestroyCsrqrInfo(d_info)); }

    if (h_csrValA) { free(h_csrValA); }
    if (h_csrRowPtrA) { free(h_csrRowPtrA); }
    if (h_csrColIndA) { free(h_csrColIndA); }

    if (h_x) { free(h_x); }
    if (h_b) { free(h_b); }
    if (h_bcopy) { free(h_bcopy); }
    if (h_r) { free(h_r); }

    if (buffer_gpu) { checkCudaErrors(cudaFree(buffer_gpu)); }

    if (d_csrValA) { checkCudaErrors(cudaFree(d_csrValA)); }
    if (d_csrRowPtrA) { checkCudaErrors(cudaFree(d_csrRowPtrA)); }
    if (d_csrColIndA) { checkCudaErrors(cudaFree(d_csrColIndA)); }
    if (d_x) { checkCudaErrors(cudaFree(d_x)); }
    if (d_b) { checkCudaErrors(cudaFree(d_b)); }
    if (d_r) { checkCudaErrors(cudaFree(d_r)); }

    if (factor_start) { checkCudaErrors(cudaEventDestroy(factor_start)); }
    if (factor_stop) { checkCudaErrors(cudaEventDestroy(factor_stop)); }
    if (solve_start) { checkCudaErrors(cudaEventDestroy(solve_start)); }
    if (solve_stop) { checkCudaErrors(cudaEventDestroy(solve_stop)); }


    return 0;
}
