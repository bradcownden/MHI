#include <iostream>

#include <cusparse_v2.h>					// CUDA sparse library
#include "cusolverSp_LOWLEVEL_PREVIEW.h"	// CUDA low-level sparse functions
											// required in factoring
#include "helper_cusolver.h"
#include "helper_cuda.h"

#include "QRFactor.h"	// QR class
#include "QRError.h"	// QR error codes
#include "debug.h"		// debug on/off



__host__ __device__ void QRFactor::solve(const double* bVector, double*& xVector)
{
	/*
	* Use the QR form of the system matrix to solve for x
	* given an input vector b
	*/

	QRFactor_Error qr_error{ QRFactor_Error::SUCCESS };

	void* buffer = NULL;
	if (buffer != NULL)
	{
		checkCudaErrors(cudaFree(buffer));
	}
	checkCudaErrors(cudaMalloc(&buffer, sizeof(char)* msize_factor));
	if (buffer == NULL) { qr_error = QRFactor_Error::BAD_BUFFER_SOLVE; }
	getQRFactorError(qr_error);
	
	// Create device-side x and b vectors
	double* d_b = NULL;
	double* d_x = NULL;

	checkCudaErrors(cudaMalloc((void**)&d_b, sizeof(double) * m_rowsA));
	checkCudaErrors(cudaMalloc((void**)&d_x, sizeof(double) * m_rowsA));
	
	// Send host input data to device
	checkCudaErrors(cudaMemcpy(d_b, bVector, sizeof(double) * m_rowsA, cudaMemcpyHostToDevice));

	// Do the solve
	checkCudaErrors(cusolverSpDcsrqrSolve(
		m_cusolverSpH, static_cast<int>(m_rowsA), static_cast<int>(m_rowsA), 
		d_b, d_x, m_info, buffer));
	// Bring the result back to the host
	checkCudaErrors(cudaMemcpy(xVector, d_x, sizeof(double) * m_rowsA, cudaMemcpyDeviceToHost));
	
}

__host__ __device__ void QRFactor::factor()
{
	/*
	* Error handler
	*/
	QRFactor_Error qrError{ QRFactor_Error::SUCCESS };

	/*
	* Device-side factoring
	*/
	size_t size_internal = 0; // size for holding matrix data
	size_t size_factor = 0;
	cusparseMatDescr_t descrA = NULL; // descriptor for data type, symmetry
	void* buffer = NULL;
	csrqrInfo_t info = NULL;
	cusolverSpHandle_t cusolverSpH = NULL;
	cudaStream_t stream = NULL;

	/*
	* Device pointers
	*/
	int* d_csrRowPtrA = NULL;
	int* d_csrColIndA = NULL;
	double* d_csrValA = NULL;

	/*
	* Constants
	*/
	int rowsA = static_cast<int>(m_rowsA); // number of rows of A
	int colsA = static_cast<int>(m_colsA); // number of columns of A
	if (rowsA != colsA) 
	{ 
		qrError = QRFactor_Error::SIZE_ERROR;
	}
	int nnzA = static_cast<int>(m_nnzA); // number of nonzeros of A
	const double tol = 1.e-16; // tolerance for invertibility
	const double zero = 0.0;
	int singularity = 0; // singularity is -1 if A is invertible under tol

	/*
	* Host pointers
	*/
	int* h_csrRowPtrA = m_sparse.outerIndexPtr();	// sizeof(int) * (ncols + 1)
	int* h_csrColIndA = m_sparse.innerIndexPtr();	// sizeof(int) * nnz
	double* h_csrValPtrA = m_sparse.valuePtr();		// sizeof(double) * nnz
	int baseA = static_cast<int>(h_csrRowPtrA[0]);  // base index in CSR format
	
	/*
	* CUDA sparse matrix handles & descriptors
	*/
	checkCudaErrors(cusolverSpCreateCsrqrInfo(&m_info));
	checkCudaErrors(cusparseCreateMatDescr(&descrA));
	// Create matrix handles and bind to streams
	checkCudaErrors(cusolverSpCreate(&m_cusolverSpH));
	checkCudaErrors(cudaStreamCreate(&m_stream));
	checkCudaErrors(cusolverSpSetStream(m_cusolverSpH, m_stream));

	// Report setup errors
	getQRFactorError(qrError);
	
	/*
	* Start factoring
	*/
	// Set cusparse matrix type and index base
	checkCudaErrors(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	if (baseA)
	{
		checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));
	}
	else
	{
		checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
	}
	
	// Allocate device memory for sparse matrix, input, and output
	checkCudaErrors(cudaMalloc((void**)&d_csrRowPtrA, sizeof(int) * (rowsA + 1)));
	checkCudaErrors(cudaMalloc((void**)&d_csrColIndA, sizeof(int) * nnzA));
	checkCudaErrors(cudaMalloc((void**)&d_csrValA, sizeof(double) * nnzA));

	// Transfer matrix pointers from host to device
	checkCudaErrors(cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, 
		sizeof(int) * (rowsA + 1), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_csrColIndA, h_csrColIndA, 
		sizeof(int) * nnzA, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_csrValA, h_csrValPtrA, 
		sizeof(double) * nnzA, cudaMemcpyHostToDevice));

#ifdef SPARSE_DEBUG	
	int* trow = NULL;
	int* tcol = NULL;
	double* tval = NULL;
	trow = (int*)malloc(sizeof(int) * (rowsA + 1));
	tcol = (int*)malloc(sizeof(int) * nnzA);
	tval = (double*)malloc(sizeof(double) * nnzA);

	checkCudaErrors(cudaMemcpy(trow, d_csrRowPtrA, sizeof(int) * (rowsA + 1),
		cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(tcol, d_csrColIndA, sizeof(int) * nnzA,
		cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(tval, d_csrValA, sizeof(double) * nnzA,
		cudaMemcpyDeviceToHost));

	std::cout << "Row pointer from device: ";
	for (int i = 0; i < (rowsA + 1); i++)
	{
		std::cout << trow[i] << " ";
	}
	std::cout << "\n";
	std::cout << "Column pointer from device: ";
	for (int i = 0; i < nnzA; i++)
	{
		std::cout << tcol[i] << " ";
	}
	std::cout << "\n";
	std::cout << "Value pointer from device: ";
	for (int i = 0; i < nnzA; i++)
	{
		std::cout << tval[i] << " ";
	}
	std::cout << "\n";
#endif // SPARSE_DEBUG

	checkCudaErrors(cusolverSpXcsrqrAnalysis(m_cusolverSpH, rowsA, colsA, nnzA,
		descrA, d_csrRowPtrA, d_csrColIndA, m_info));

	// Determine workspace requirement for factoring
	checkCudaErrors(cusolverSpDcsrqrBufferInfo(m_cusolverSpH, rowsA, colsA, nnzA,
		descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA, m_info, 
		&size_internal, &msize_factor));

	// Allocate buffer on device
	if (buffer) {
		checkCudaErrors(cudaFree(buffer));
	}
	checkCudaErrors(cudaMalloc(&buffer, sizeof(char) * msize_factor));
	if (buffer == NULL) { qrError = QRFactor_Error::BAD_BUFFER_FACTOR; }
	getQRFactorError(qrError);

	// Compute decomposition
	checkCudaErrors(cusolverSpDcsrqrSetup(m_cusolverSpH, rowsA, colsA, nnzA,
		descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA, zero, m_info));
	checkCudaErrors(cusolverSpDcsrqrFactor(m_cusolverSpH, rowsA, colsA, nnzA,
		NULL, NULL, m_info, buffer));

	// Check for singularity condition
	{
		checkCudaErrors(cusolverSpDcsrqrZeroPivot(m_cusolverSpH, m_info, tol,
			&singularity));
		if (0 <= singularity) {
			qrError = QRFactor_Error::SINGULARITY_ERROR;
#ifdef DEBUG
			fprintf(stderr, "Error: A is not invertible, singularity=%d\n", 
				singularity);
#endif // DEBUG
		}
	}

	// Factoring complete. Report any errors
	getQRFactorError(qrError);

}

// If the total number of non-zero entries is known, set the 
// total size of the triplets for better performance
void QRFactor::setTripletsSize(const int tripletSize)
{
	m_coefficients.reserve(m_coefficients.size() + tripletSize);
}

// Use Eigen triplets to build a sparse system matrix from many dense inputs
// per Oct 16, 2020 email from Rajendra
void QRFactor::buildTriplets(const double* inputArr, int rows, int cols)
{
	/*
	* Read the inputs of inputArr into class triplets list
	* Update internal row and column offset after
	* the data has been read in
	*/

	// Find the number of non-zero entries
	int count = 0;
	for (int i = 0; i < (rows * cols); i++)
	{
		if (inputArr[i] != 0.0)
		{
			count++;
		}
	}
	// Index for new non-zero elements
	int oldSize{ static_cast<int>(m_coefficients.size()) };

#ifdef SPARSE_DEBUG
	std::cout << "Number of non-zero values read in: " << count << "\n";
	std::cout << "Length of m_coefficients: " << m_coefficients.size() << "\n";
#endif // SPARSE_DEBUG

	// Resize the triplets list to accomodate new entries
	m_coefficients.resize(count + m_coefficients.size());
	int newSize{ static_cast<int>(m_coefficients.size()) };

#ifdef SPARSE_DEBUG
	std::cout << "New length of m_coefficients: " << m_coefficients.size() << "\n";
#endif // SPARSE_DEBUG

	// Load new entries
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			// Append non-zero values to triplet list
			if (inputArr[j * rows + i] != 0.0)
			{
				if (oldSize <= newSize)
				{
					m_coefficients[oldSize] = T((i + m_rowOffset), (j + m_colOffset),
						inputArr[j * rows + i]);
					oldSize++;
				}
				else
				{
#ifdef SPARSE_DEBUG
					std::cerr << "\nERROR: m_coefficients was not properly resized. "
						<< "Attempted to add too many non-zero elements.\n";
#endif // SPARSE_DEBUG

				}
			}
		}
	}
#ifdef SPARSE_DEBUG
	std::cout << "m_coefficients: \n";
	for (int i = 0; i < m_coefficients.size(); i++)
	{
		std::cout << m_coefficients.data()[i].value() << " ";
	}
	std::cout << "\n";
#endif // SPARSE_DEBUG

	// Update internal offsets
	m_rowOffset += rows;
	m_colOffset += cols;
}

// After all the triplets have been read in, build the matrix
void QRFactor::buildSparseMatrix()
{
	// Set the total required size of the sparse matrix using the 
	// final values for the row and column offsets
	m_sparse.resize(m_rowOffset, m_colOffset);

	// Feed triplets to sparse matrix
	m_sparse.setFromTriplets(m_coefficients.begin(), m_coefficients.end());

	// Free the initially allocated memory
	m_sparse.data().squeeze();

	// Set class members
	m_nnzA = m_sparse.nonZeros();
	m_rowsA = m_sparse.rows();
	m_colsA = m_sparse.cols();

#ifdef SPARSE_DEBUG
	std::cout << "Sparse matrix after loading...\n" << m_sparse << "\n";
	std::cout << "Row pointer: ";
	for (int i = 0; i < m_sparse.cols() + 1; i++)
	{
		std::cout << m_sparse.outerIndexPtr()[i] << " ";
	}
	std::cout << "\n";
	std::cout << "Column pointer: ";
	for (int i = 0; i < m_sparse.nonZeros(); i++)
	{
		std::cout << m_sparse.innerIndexPtr()[i] << " ";
	}
	std::cout << "\n";
	std::cout << "Value pointer: ";
	for (int i = 0; i < m_sparse.nonZeros(); i++)
	{
		std::cout << m_sparse.valuePtr()[i] << " ";
	}
	std::cout << "\n";
#endif // SPARSE_DEBUG

}
