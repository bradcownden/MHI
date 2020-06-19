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
#include <assert.h>
#include <iostream>
#include <chrono>

#include "cusolverSp.h"

#include "cusolverSp_LOWLEVEL_PREVIEW.h"

#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "helper_cusolver.h"

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

bool fileExists(const char* fname)
{
    std::ifstream ifile(fname);
    bool out = (ifile) ? true : false;
    return out;
}

void UsageSP(void)
{
    printf("<options>\n");
    printf("-h          : display this help\n");
    printf("-file=<filename> : filename containing a matrix in MM format\n");
    printf("-device=<device_id> : <device_id> if want to run on specific GPU\n");

    exit(0);
}

void parseCommandLineArguments(int argc, char* argv[], struct testOpts& opts)
{
    memset(&opts, 0, sizeof(opts));

    if (checkCmdLineFlag(argc, (const char**)argv, "-h"))
    {
        UsageSP();
    }

    if (checkCmdLineFlag(argc, (const char**)argv, "file"))
    {
        char* fileName = 0;
        getCmdLineArgumentString(argc, (const char**)argv, "file", &fileName);

        if (fileName)
        {
            opts.sparse_mat_filename = fileName;
        }
        else
        {
            printf("\nIncorrect filename passed to -file \n ");
            UsageSP();
        }
    }
}

void gpuFactor(csrqrInfo_t d_info, cusolverSpHandle_t cusolverSpH, cusparseMatDescr_t descrA, void* buffer_gpu,
    int rowsA, int colsA, int nnzA, int* d_csrRowPtrA, int* d_csrColIndA, double* d_csrValA)
{
    int singularity = 0;
    const double tol = 1.0e-16;
    const double zero = 0.0;
    size_t size_qr = 0;
    size_t size_internal = 0;

    // Create opaque data structure
    checkCudaErrors(cusolverSpCreateCsrqrInfo(&d_info));

    // Analyze qr(A)
    checkCudaErrors(cusolverSpXcsrqrAnalysis(
        cusolverSpH, rowsA, colsA, nnzA,
        descrA, d_csrRowPtrA, d_csrColIndA,
        d_info));

    // Create workspace for qr(A)
    checkCudaErrors(cusolverSpDcsrqrBufferInfo(
        cusolverSpH, rowsA, colsA, nnzA,
        descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
        d_info,
        &size_internal,
        &size_qr));

    if (buffer_gpu)
    {
        checkCudaErrors(cudaFree(buffer_gpu));
    }
    checkCudaErrors(cudaMalloc(&buffer_gpu, sizeof(char) * size_qr));
    assert(NULL != buffer_gpu);

    // Set up, then factor
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

    // Check for singularity
    checkCudaErrors(cusolverSpDcsrqrZeroPivot(
        cusolverSpH, d_info, tol, &singularity));

    if (0 <= singularity) {
        fprintf(stderr, "Error: A is not invertible, singularity=%d\n", singularity);
        exit(1);
    }
}

void cpuFactor(csrqrInfoHost_t h_info, cusolverSpHandle_t cusolverSpH, cudaStream_t stream, cusparseMatDescr_t descrA,
    int rowsA, int colsA, int nnzA, int* h_csrRowPtrA, int* h_csrColIndA, double* h_csrValA)
{
    int singularity = 0;
    const double tol = 1.0e-16;
    const double zero = 0.0;
    size_t size_qr = 0;
    size_t size_internal = 0;
    void* buffer_cpu = NULL;

    // Create opaque info structure
    checkCudaErrors(cusolverSpCreateCsrqrInfoHost(&h_info));

    // Analyze qr(A)
    checkCudaErrors(cusolverSpXcsrqrAnalysisHost(
        cusolverSpH, rowsA, colsA, nnzA,
        descrA, h_csrRowPtrA, h_csrColIndA,
        h_info));

    // Create workspace for qr(A)
    checkCudaErrors(cusolverSpDcsrqrBufferInfoHost(
        cusolverSpH, rowsA, colsA, nnzA,
        descrA, h_csrValA, h_csrRowPtrA, h_csrColIndA,
        h_info,
        &size_internal,
        &size_qr));

    if (buffer_cpu) {
        free(buffer_cpu);
    }
    buffer_cpu = (void*)malloc(sizeof(char) * size_qr);
    //assert(NULL != buffer_cpu);
    std::cout << "buffer_cpu allocated to " << buffer_cpu << std::endl;

    // Set up, then factor
    checkCudaErrors(cusolverSpDcsrqrSetupHost(
        cusolverSpH, rowsA, colsA, nnzA,
        descrA, h_csrValA, h_csrRowPtrA, h_csrColIndA,
        zero,
        h_info));

    checkCudaErrors(cusolverSpDcsrqrFactorHost(
        cusolverSpH, rowsA, colsA, nnzA,
        NULL, NULL,
        h_info,
        buffer_cpu));

    // Check for singularity
    checkCudaErrors(cusolverSpDcsrqrZeroPivotHost(
        cusolverSpH, h_info, tol, &singularity));

    if (0 <= singularity) {
        fprintf(stderr, "Error: A is not invertible, singularity=%d\n", singularity);
        exit(1);
    }

}

void readB(char* inFile, char* argv[], const int rowsA, double* inPtr) // Read in b vectors
{
    if (inFile)
    {
        printf("Reading file %s\n", inFile);
        std::ifstream file(inFile);
        if (file.is_open())
        {
            std::string line;
            std::string::size_type val;
            int count = 0;
            while (count < rowsA) // Stopgap for some data files having different lengths
            {
                getline(file, line);
                inPtr[count] = std::stod(line, &val);
                //printf("%s\n", line.c_str());
                ++count;
            }
            if (count != rowsA)
            {
                printf("\nERROR: input file has %d rows, but matrix row size %d\n\n", count, rowsA);
            }
        }
    }
    else
    {
        printf("\nERROR: couldn't find file %s\n\n", inFile);
    }
}

int main(int argc, char* argv[])
{
    struct testOpts opts;
    cusolverSpHandle_t cusolverSpH = NULL; // reordering, permutation and 1st LU factorization
    cusparseHandle_t   cusparseH = NULL;   // residual evaluation
    cudaStream_t stream = NULL;
    cusparseMatDescr_t descrA = NULL; // A is a base-0 general matrix

    csrqrInfoHost_t h_info = NULL; // opaque info structure for QR factoring
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
    void* buffer_cpu = NULL; // working space for factoring
    void* buffer_gpu = NULL; // working space for factoring

    int* d_csrRowPtrA = NULL; // <int> n+1
    int* d_csrColIndA = NULL; // <int> nnzA
    double* d_csrValA = NULL; // <double> nnzA
    double* d_x = NULL; // <double> n, x = A \ b
    double* d_b = NULL; // <double> n, a copy of h_b
    double* d_r = NULL; // <double> n, r = b - A*x

    // the constants used in residual evaluation, r = b - A*x
    const double minus_one = -1.0;
    const double one = 1.0;
    const double zero = 0.0;
    // the constant used in cusolverSp
    // singularity is -1 if A is invertible under tol
    // tol determines the condition of singularity
    int singularity = 0;
    const double tol = 1.e-16;

    const int checkFreq = 1;
    const int checkTot = 100;

    // Initialize random number for checks
    srand(time(NULL));

    // Initial matrix read
    parseCommandLineArguments(argc, argv, opts);

    findCudaDevice(argc, (const char**)argv);

    if (opts.sparse_mat_filename == NULL)
    {
        opts.sparse_mat_filename = sdkFindFilePath("sysMatA_t1.mtx", argv[0]);
        if (opts.sparse_mat_filename != NULL)
            printf("Using default input file [%s]\n", opts.sparse_mat_filename);
        else
            printf("Could not find sysMatA_t1.mtx\n");
    }
    else
    {
        printf("Using input file [%s]\n", opts.sparse_mat_filename);
    }


    printf("step 1: read matrix market format\n");

    if (opts.sparse_mat_filename) {
        if (loadMMSparseMatrix<double>(opts.sparse_mat_filename, 'd', true, &rowsA, &colsA,
            &nnzA, &h_csrValA, &h_csrRowPtrA, &h_csrColIndA, true)) {
            return 1;
        }
        baseA = h_csrRowPtrA[0]; // baseA = {0,1}
    }
    else {
        fprintf(stderr, "Error: input matrix is not provided\n");
        return 1;
    }

    if (rowsA != colsA) {
        fprintf(stderr, "Error: only support square matrix\n");
        return 1;
    }

    printf("sparse matrix A is %d x %d with %d nonzeros, base=%d\n", rowsA, colsA, nnzA, baseA);

    checkCudaErrors(cusolverSpCreate(&cusolverSpH));
    checkCudaErrors(cusparseCreate(&cusparseH));
    checkCudaErrors(cudaStreamCreate(&stream));
    checkCudaErrors(cusolverSpSetStream(cusolverSpH, stream));
    checkCudaErrors(cusparseSetStream(cusparseH, stream));

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

    h_x = (double*)malloc(sizeof(double) * colsA);
    h_b = (double*)malloc(sizeof(double) * rowsA);
    h_bcopy = (double*)malloc(sizeof(double) * rowsA);
    h_r = (double*)malloc(sizeof(double) * rowsA);

    assert(NULL != h_x);
    assert(NULL != h_b);
    assert(NULL != h_bcopy);
    assert(NULL != h_r);

    checkCudaErrors(cudaMalloc((void**)&d_csrRowPtrA, sizeof(int) * (rowsA + 1)));
    checkCudaErrors(cudaMalloc((void**)&d_csrColIndA, sizeof(int) * nnzA));
    checkCudaErrors(cudaMalloc((void**)&d_csrValA, sizeof(double) * nnzA));
    checkCudaErrors(cudaMalloc((void**)&d_x, sizeof(double) * colsA));
    checkCudaErrors(cudaMalloc((void**)&d_b, sizeof(double) * rowsA));
    checkCudaErrors(cudaMalloc((void**)&d_r, sizeof(double) * rowsA));

    /*
    for (int row = 0; row < rowsA; row++)
    {
        h_b[row] = 1.0;
    }
    */

    //readB("../Output/Province/system/sysVecB_t0.txt", argv, rowsA, h_b);

    //memcpy(h_bcopy, h_b, sizeof(double) * rowsA);

    // Do CPU factoring for checking solutions later
    auto cpu_start = std::chrono::high_resolution_clock::now();

    printf("CPU factoring...\n");
    checkCudaErrors(cusolverSpCreateCsrqrInfoHost(&h_info));

    checkCudaErrors(cusolverSpXcsrqrAnalysisHost(
        cusolverSpH, rowsA, colsA, nnzA,
        descrA, h_csrRowPtrA, h_csrColIndA,
        h_info));

    checkCudaErrors(cusolverSpDcsrqrBufferInfoHost(
        cusolverSpH, rowsA, colsA, nnzA,
        descrA, h_csrValA, h_csrRowPtrA, h_csrColIndA,
        h_info,
        &size_internal,
        &size_chol));

    if (buffer_cpu) {
        free(buffer_cpu);
    }
    buffer_cpu = (void*)malloc(sizeof(char) * size_chol);
    assert(NULL != buffer_cpu);

    checkCudaErrors(cusolverSpDcsrqrSetupHost(
        cusolverSpH, rowsA, colsA, nnzA,
        descrA, h_csrValA, h_csrRowPtrA, h_csrColIndA,
        zero,
        h_info));
    checkCudaErrors(cusolverSpDcsrqrFactorHost(
        cusolverSpH, rowsA, colsA, nnzA,
        NULL, NULL,
        h_info,
        buffer_cpu));

    checkCudaErrors(cusolverSpDcsrqrZeroPivotHost(
        cusolverSpH, h_info, tol, &singularity));

    if (0 <= singularity) {
        fprintf(stderr, "Error: A is not invertible, singularity=%d\n", singularity);
        return 1;
    }

    /*
    printf("step 7: solve A*x = b \n");
    checkCudaErrors(cusolverSpDcsrqrSolveHost(
        cusolverSpH, rowsA, colsA, h_b, h_x, h_info, buffer_cpu));
    */

    auto cpu_stop = std::chrono::high_resolution_clock::now(); // CPU timing stop
    std::chrono::duration<double, std::milli> cpu_time = cpu_stop - cpu_start;
    printf("CPU factoring time: %E ms\n", cpu_time.count());

   
    //printf("step 8: evaluate residual r = b - A*x (result on CPU)\n");
    // use GPU gemv to compute r = b - A*x

    // Transfer matrix pointers from host to device
    checkCudaErrors(cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, sizeof(int) * (rowsA + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrColIndA, h_csrColIndA, sizeof(int) * nnzA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrValA, h_csrValA, sizeof(double) * nnzA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_bcopy, sizeof(double) * rowsA, cudaMemcpyHostToDevice));

    /*
    checkCudaErrors(cudaMemcpy(d_r, h_bcopy, sizeof(double) * rowsA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_x, h_x, sizeof(double) * colsA, cudaMemcpyHostToDevice));

    checkCudaErrors(cusparseDcsrmv(cusparseH,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        rowsA,
        colsA,
        nnzA,
        &minus_one,
        descrA,
        d_csrValA,
        d_csrRowPtrA,
        d_csrColIndA,
        d_x,
        &one,
        d_r));

    checkCudaErrors(cudaMemcpy(h_r, d_r, sizeof(double) * rowsA, cudaMemcpyDeviceToHost));

    x_inf = vec_norminf(colsA, h_x);
    r_inf = vec_norminf(rowsA, h_r);
    A_inf = csr_mat_norminf(rowsA, colsA, nnzA, descrA, h_csrValA, h_csrRowPtrA, h_csrColIndA);

    printf("(CPU) |b - A*x| = %E \n", r_inf);
    printf("(CPU) |A| = %E \n", A_inf);
    printf("(CPU) |x| = %E \n", x_inf);
    printf("(CPU) |b - A*x|/(|A|*|x|) = %E \n", r_inf / (A_inf * x_inf));
    */

    printf("Beginning GPU factoring...\n");

    checkCudaErrors(cudaEventRecord(factor_start)); // Timing for GPU solve

    // Create opaque data structure
    checkCudaErrors(cusolverSpCreateCsrqrInfo(&d_info));

    // Analyze qr(A)
    checkCudaErrors(cusolverSpXcsrqrAnalysis(
        cusolverSpH, rowsA, colsA, nnzA,
        descrA, d_csrRowPtrA, d_csrColIndA,
        d_info));

    // Determine workspace requirement for factoring
    checkCudaErrors(cusolverSpDcsrqrBufferInfo(
        cusolverSpH, rowsA, colsA, nnzA,
        descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
        d_info,
        &size_internal,
        &size_chol));

    if (buffer_gpu) {
        checkCudaErrors(cudaFree(buffer_gpu));
    }
    checkCudaErrors(cudaMalloc(&buffer_gpu, sizeof(char) * size_chol));

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

    // Check for singularity condition
    {
        checkCudaErrors(cusolverSpDcsrqrZeroPivot(
            cusolverSpH, d_info, tol, &singularity));
        if (0 <= singularity) {
            fprintf(stderr, "Error: A is not invertible, singularity=%d\n", singularity);
            return 1;
        }
    }

    checkCudaErrors(cudaEventRecord(factor_stop)); // Stop timing and calculate GPU execution time
    checkCudaErrors(cudaEventSynchronize(factor_stop));
    checkCudaErrors(cudaEventElapsedTime(&gpufactor_time, factor_start, factor_stop));
    printf("GPU factoring time: %E ms\n", gpufactor_time);

    // Loop to solve multiple data files
    int bcount = 1;
    while (true)
    {
        char bfile[500];
        // Updated data (non-ill defined system)
        sprintf(bfile, "C:/Users/bradc/Documents/MHI/GPU_Data/CompilerGF462/SysVecB_t%d.txt", bcount);
        // Check if file exists
        if (fileExists(bfile))
        {
            readB(bfile, argv, rowsA, h_b);

            // Solve timing
            checkCudaErrors(cudaEventRecord(solve_start));
            checkCudaErrors(cudaMemcpy(d_b, h_b, sizeof(double) * rowsA, cudaMemcpyHostToDevice));
            // printf("Solve A*x = b with RHS from %s\n", bfile);
            checkCudaErrors(cusolverSpDcsrqrSolve(
                cusolverSpH, rowsA, colsA, d_b, d_x, d_info, buffer_gpu));
            checkCudaErrors(cudaMemcpy(h_x, d_x, sizeof(double) * rowsA, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaEventRecord(solve_stop));
            checkCudaErrors(cudaEventSynchronize(solve_stop));
            checkCudaErrors(cudaEventElapsedTime(&gpusolve_time, solve_start, solve_stop));
            printf("GPU solve time: %E ms\n", gpusolve_time);

            // Use random number generator to create spot checks of cpu
            // vs gpu results
            int v = rand() % checkTot;
            //if (v <= checkFreq)
            if (false)
            {
                // Write out GPU data
                char xfile[500];
                sprintf(xfile, "GPUFactor_t%d.txt", bcount);
                FILE* GPU_out = fopen(xfile, "w");
                if (GPU_out == NULL)
                {
                   std::cout << "\nERROR: Couldn't write to file " << xfile << "\n";
                   exit(1);
                }
                for (int i = 0; i < rowsA; ++i)
                {
                    fprintf(GPU_out, "%1.15e\n", h_x[i]);
                }
                fclose(GPU_out);

                // Do CPU solving and write out
                checkCudaErrors(cusolverSpDcsrqrSolveHost(
                    cusolverSpH, rowsA, colsA, h_b, h_x, h_info, buffer_cpu));

                char Outname[500];
                sprintf(Outname, "CPUFactor_t%d.txt", bcount);
                FILE* CPU_out = fopen(Outname, "w");
                if (CPU_out == NULL)
                {
                    std::cout << "\nERROR: Couldn't write to file " << Outname << "\n";
                    exit(1);
                }
                for (int i = 0; i < rowsA; ++i)
                {
                    fprintf(CPU_out, "%1.15e\n", h_x[i]);
                }
                fclose(CPU_out);
            }
            ++bcount;
        }
        else
        {
            printf("End of file directory. Exiting program.\n");
            break;
        }
    }

    /*
    printf("step 14: solve A*x = b \n");
    checkCudaErrors(cusolverSpDcsrqrSolve(
        cusolverSpH, rowsA, colsA, d_b, d_x, d_info, buffer_gpu));

    checkCudaErrors(cudaMemcpy(d_r, h_bcopy, sizeof(double) * rowsA, cudaMemcpyHostToDevice));

    checkCudaErrors(cusparseDcsrmv(cusparseH,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        rowsA,
        colsA,
        nnzA,
        &minus_one,
        descrA,
        d_csrValA,
        d_csrRowPtrA,
        d_csrColIndA,
        d_x,
        &one,
        d_r));

    //checkCudaErrors(cudaMemcpy(h_r, d_r, sizeof(double) * rowsA, cudaMemcpyDeviceToHost));

    //r_inf = vec_norminf(rowsA, h_r);

    //printf("(GPU) |b - A*x| = %E \n", r_inf);
    //printf("(GPU) |b - A*x|/(|A|*|x|) = %E \n", r_inf / (A_inf * x_inf));

    */

    if (cusolverSpH) { checkCudaErrors(cusolverSpDestroy(cusolverSpH)); }
    if (cusparseH) { checkCudaErrors(cusparseDestroy(cusparseH)); }
    if (stream) { checkCudaErrors(cudaStreamDestroy(stream)); }
    if (descrA) { checkCudaErrors(cusparseDestroyMatDescr(descrA)); }
    if (h_info) { checkCudaErrors(cusolverSpDestroyCsrqrInfoHost(h_info)); }
    if (d_info) { checkCudaErrors(cusolverSpDestroyCsrqrInfo(d_info)); }

    if (h_csrValA) { free(h_csrValA); }
    if (h_csrRowPtrA) { free(h_csrRowPtrA); }
    if (h_csrColIndA) { free(h_csrColIndA); }

    if (h_x) { free(h_x); }
    if (h_b) { free(h_b); }
    if (h_bcopy) { free(h_bcopy); }
    if (h_r) { free(h_r); }

    if (buffer_cpu) { free(buffer_cpu); }
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
