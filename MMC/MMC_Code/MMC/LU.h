#pragma once
#include <iostream>

class LU
{
	int n{ 0 };
	double** lu{};
	int* indx{};
	double d{ 0.0 };

	void solve(double*& b, double*& x);
	void solve(double**& B, double**& X);
public:
	LU(double**& A, int& size);
	double** inverse();
};