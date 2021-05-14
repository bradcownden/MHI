#include "LU.h"
#include <iterator>


// Factor on initialization. Method from NR ludcmp.h
LU::LU(double**& A, int& size): n(size)
{
	const double EPS = 1.0e-40;
	indx = new int[n];
	double big{ 0.0 };
	double temp{ 0.0 };
	double* vv = new double[n];
	int i, j, k, imax;

	// Initialize lu matrix
	lu = new double* [n];
	for (i = 0; i < n; i++)
	{
		lu[i] = new double [n];
		for (j = 0; j < n; j++) { lu[i][j] = A[i][j]; }
	}
	
	d = 1.0;
	
	for (i = 0; i < n; i++) 
	{
		big = 0.0;
		for (j = 0; j < n; j++)
		{
			if ((temp = fabs(lu[i][j])) > big) big = temp;
		}
		if (big == 0.0) { exit(EXIT_FAILURE); }
		vv[i] = 1.0 / big;
	}
	for (k = 0; k < n; k++) 
	{
		big = 0.0;
		imax = k;
		for (i = k; i < n; i++) 
		{
			temp = vv[i] * fabs(lu[i][k]);
			if (temp > big) 
			{
				big = temp;
				imax = i;
			}
		}
		if (k != imax) 
		{
			for (j = 0; j < n; j++) 
			{
				temp = lu[imax][j];
				lu[imax][j] = lu[k][j];
				lu[k][j] = temp;
			}
			d = -d;
			vv[imax] = vv[k];
		}
		indx[k] = imax;
		if (lu[k][k] == 0.0) { lu[k][k] = EPS; }
		for (i = k + 1; i < n; i++) 
		{
			temp = lu[i][k] /= lu[k][k];
			for (j = k + 1; j < n; j++)
			{
				lu[i][j] -= temp * lu[k][j];
			}
		}
	}
}

// Vector version of solve
void LU::solve(double*& b, double*& x)
{
	int i, ip, j;
	int ii{ 0 };
	double sum{ 0.0 };
	
	for (i = 0; i < n; i++) { x[i] = b[i]; }
	for (i = 0; i < n; i++) 
	{
		ip = indx[i];
		sum = x[ip];
		x[ip] = x[i];
		if (ii != 0)
			for (j = ii - 1; j < i; j++) { sum -= lu[i][j] * x[j]; }
		else if (sum != 0.0)
			ii = i + 1;
		x[i] = sum;
	}
	for (i = n - 1; i >= 0; i--) 
	{
		sum = x[i];
		for (j = i + 1; j < n; j++) { sum -= lu[i][j] * x[j]; }
		x[i] = sum / lu[i][i];
	}
}

// Matrix version of solve
void LU::solve(double**& B, double**& X)
{
	int i, j;
	double* xx = new double[n];
	for (j = 0; j < n; j++) 
	{
		for (i = 0; i < n; i++) xx[i] = B[i][j];
		solve(xx, xx);
		for (i = 0; i < n; i++) X[i][j] = xx[i];
	}
}

// Use the solve function to compute the inverse of A
double** LU::inverse()
{
	int i, j;
	double** Ainv = new double* [n];
	for (i = 0; i < n; i++) 
	{
		Ainv[i] = new double[n];
		for (j = 0; j < n; j++) { Ainv[i][j] = 0.; }
		Ainv[i][i] = 1.;
	}
	solve(Ainv, Ainv);
	return Ainv;
}