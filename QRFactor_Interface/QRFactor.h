#pragma once
#include <vector>
#include <Eigen/SparseCore>		// Eigen sparse library

#include "cusolverSp.h"			// CUDA sparse solving library
#include <cuda_runtime.h>		// __host__ and __device__
#include "helper_cuda.h"


class QRFactor
{
	typedef Eigen::Triplet<double> T; // Shorthand for Eigen triplet type

// Member functions, constructor, destructor are public
public:

	__host__ QRFactor() = default; // default constructor

	/*
	* Main factoring and solving functions
	*/
	__host__ __device__ void factor();
	__host__ __device__ void solve(const double* bVector, double*& xVector);
	
	/*
	* Building sparse matrix
	*/
	void buildTriplets(const double* inputArr, int rows, int cols);
	void buildSparseMatrix();

	void setTripletsSize(const int tripletSize);
	Eigen::SparseMatrix<double> getSparseMatrix() const { return m_sparse; }

	/*
	* Destructor
	*/ 
	~QRFactor()
	{
		if (m_cusolverSpH) { checkCudaErrors(cusolverSpDestroy(m_cusolverSpH)); }
		if (m_stream) { checkCudaErrors(cudaStreamDestroy(m_stream)); }
		if (m_info) { checkCudaErrors(cusolverSpDestroyCsrqrInfo(m_info)); }
	}

private:

	size_t m_rowsA{ 0 };
	size_t m_colsA{ 0 };
	int m_rowOffset{ 0 };
	int m_colOffset{ 0 };
	size_t m_nnzA{ 0 };
	std::vector<T> m_coefficients;		// Vector of triplets
	// Sparse matrix from Eigen package: http://eigen.tuxfamily.org/index.php?title=Main_Page
	Eigen::SparseMatrix<double, Eigen::StorageOptions::RowMajor> m_sparse{}; // matrix is column-major by default

	/*
	* CUSOLVER/CUSPARSE variables 
	*/
	size_t msize_factor = 0;		
	csrqrInfo_t m_info = NULL;
	cusolverSpHandle_t m_cusolverSpH = NULL;
	cudaStream_t m_stream = NULL;
	
};


