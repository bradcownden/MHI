#pragma once
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/device_ptr.h"
#include "thrust/device_reference.h"
#include "thrust/fill.h"

struct LUgpu
{
	typedef thrust::device_vector<thrust::device_vector<double>> d_mat;
	typedef thrust::device_vector<double> d_vec;
	size_t n{ 0 };
	d_mat lu{};
	d_vec indx{};
	double d{ 0.0 };

	__device__ void solve(d_vec b, d_vec x);
	__device__ void solve(d_mat B, d_mat X);
public:
	LUgpu(double* A, int& size);
	double** inverse();
};

