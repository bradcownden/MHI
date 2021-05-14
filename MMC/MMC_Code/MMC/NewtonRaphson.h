#include <iostream>

/*
* Apply the Newton-Raphson solving technique to non-linear components in the
* MMC model. This N-R algorithm comes from Numerical Recipes Ch 9.4 and includes
* a fail-safe routine that utilizes a combination of bisection and N-R: it takes
* a bisection step whenever N-R would take the solution out of bounds, or
* whenever N-R is not reducing the size of the brackets rapidly enough
*/

// Structure that contains a function and its derivative
struct Func
{
	double f(double x)
	{
		return 5.0 * x * x * x - 2.0 * x * x + 10.0 * x - 7.0;
	}
	double df(double x)
	{
		return 15.0 * x * x - 4.0 * x + 10.0;
	}
};

struct NRSolver
{
	const int MAXITS{ 100 };	// Maximum number of iterations
	const double TOL{ 1.0E-8 };	// Convergence criteria

	double root(Func& f, const double& a, const double& b);
};