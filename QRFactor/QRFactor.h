#pragma once
#include <QRFactor.cpp>
#ifndef QRFACTOR_H
#define QRFACTOR_H

class GPUFactor
{
public:

    double* b_in;
	double* x_out;

private:
    cusolverSpHandle_t cusolverSpH = NULL; // reordering, permutation and 1st LU factorization
    cusparseHandle_t   cusparseH = NULL;   // residual evaluation
    cudaStream_t stream = NULL;
    cusparseMatDescr_t descrA = NULL; // A is a base-0 general matrix

    csrqrInfoHost_t h_info = NULL; // opaque info structure for LU with parital pivoting
    csrqrInfo_t d_info = NULL; // opaque info structure for LU with parital pivoting

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

    size_t size_internal = 0;
    size_t size_chol = 0; // size of working space for csrlu
    void* buffer_cpu = NULL; // working space for Cholesky
    void* buffer_gpu = NULL; // working space for Cholesky

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
};


class CPUFactor
{
public:

    double* b_in;
    double* x_out;

private:


};


#endif
