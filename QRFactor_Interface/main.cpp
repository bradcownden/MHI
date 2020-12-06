#include <iostream>
#include <stdio.h>

#include "Io.h"
#include "debug.h"
#include "QRFactor.h"

#ifdef DEBUG
#include <Eigen/Dense>	// Direct solve test
#include <chrono>		// Timing
#endif // DEBUG


/* 
 * Interface testing for QRFactor
 * 
 * Primary purpose: to test the ability to compile QRFactor
 * separately from a main .cpp file and call it as a 
 * function. This test should include an input vector and
 * a dense matrix that will be converted into a sparse system 
 * matrix before undergoing QR factoring on the device. 
 * The input vector will then be used to solve Ax = b and the result
 * will be returned to the main program.
*/

int main()
{
	QRFactor qr;

	// Known factorable matrix from https://eigen.tuxfamily.org/dox/group__TutorialSparse.html
	// Column-major order
	double inArray[25] = { 0.0, 22.0, 7.0, 0.0, 0.0,
						   3.0, 0.0, 5.0, 0.0, 0.0,
						   0.0, 0.0, 0.0, 7.0, 14.0,
						   0.0, 0.0, 1.0, 0.0, 0.0,
						   0.0, 17.0, 0.0, 0.0, 8.0 };
	double* inPtr;
	inPtr = inArray;
	int rows = 5;
	int cols = 5;
	int* rPtr;
	int* cPtr;
	rPtr = &rows;
	cPtr = &cols;

#ifdef DEBUG
	auto start = std::chrono::high_resolution_clock::now();		// Matrix build timings
#endif // DEBUG

	// If the total number of non-zero entries is known, set the 
	// size of the list of triplets. This would be calculated for 
	// each dense matrix
	int nnzCount = 9;
	qr.setTripletsSize(nnzCount);
	// Build the non-zero entries with input dense matrices
	qr.buildTriplets(inPtr, *rPtr, *cPtr);
	// All dense inputs have been read in. Build the sparse matrix and convert to CSR
	qr.buildSparseMatrix();

#ifdef DEBUG
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "Sparse matrix building took " << duration.count() << " us\n";
#endif // DEBUG

	qr.factor();
	// Factoring is now complete. All subsequent calls to QRFactor will
	// be for solving only

	// Some example input data
	double* b = NULL;
	double* x = NULL;
	b = (double*)malloc(sizeof(double) * 5);
	x = (double*)malloc(sizeof(double) * 5);
	for (int i = 0; i < 5; i++)
	{
		b[i] = float(i) + 1.0;
	}

	// Do the solving
	qr.solve(const_cast<double*>(b), x);
	
#ifdef DEBUG
	// Print the result of the solving step
	std::cout << "Solving sparse matrix on GPU. Result is\n";
	for (int i = 0; i < 5; i++)
	{
		std::cout << x[i] << "\n";
	}

	// Host-only solving for the sample data as a check 
	// on the GPU result

	std::cout << "Eigen package solve -- direct\n";
	Eigen::Matrix<double, 5, 5> A;
	A << 0.0, 3.0, 0.0, 0.0, 0.0, 22.0, 0.0, 0.0, 0.0, 17.0, 7.0, 5.0, 0.0, 1.0, 0.0,
		0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 14.0, 0.0, 8.0;
	Eigen::SparseMatrix<double, Eigen::StorageOptions::RowMajor> A_CSR = A.sparseView();
	std::cout << "CSR Row pointer: ";
	for (int i = 0; i < A_CSR.rows() + 1; i++)
	{
		std::cout << A_CSR.outerIndexPtr()[i] << " ";
	}
	std::cout << "\n";
	std::cout << "CSR Colum pointer: ";
	for (int i = 0; i < A_CSR.nonZeros(); i++)
	{
		std::cout << A_CSR.innerIndexPtr()[i] << " ";
	}
	std::cout << "\n";
	std::cout << "CSR Value pointer: ";
	for (int i = 0; i < A_CSR.nonZeros(); i++)
	{
		std::cout << A_CSR.valuePtr()[i] << " ";
	}
	std::cout << "\n";

	Eigen::VectorXd B(5);
	B << 1.0, 2.0, 3.0, 4.0, 5.0;
	std::cout << "Input matrix:\n" << A << std::endl;
	Eigen::Matrix<double, 5, 1> X = A.fullPivLu().solve(B);
	std::cout << "Solution:\n" << X << std::endl;
#endif // DEBUG

	return 0;
	
}