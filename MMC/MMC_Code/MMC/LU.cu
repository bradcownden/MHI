#pragma once
#include "LU.cuh"



// Take in a column-major list of values and factor upon initialization
LUgpu::LUgpu(double* A, int& size): n(size)
{
	// Wrap input pointer with thrust pointer
	thrust::device_ptr<double> d_ptr(A);
	// Initialize lu
	for (int i = 0; i < n*n; i+=n)
	{
		lu[i] = thrust::device_vector<double>(&d_ptr[i], &d_ptr[i] + n);
	}

	// Print lu
	for (size_t i = 0; i < lu.size(); i++)
	{
		std::cout << lu[i] << "\n";
		
	}
}