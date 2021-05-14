#include "NewtonRaphson.h"

// Return the root of the function data within the bound [a,b]
double NRSolver::root(Func& f, const double& a, const double& b)
{
	double xh{ 0.0 };
	double xl{ 0.0 };
	double fl{ f.f(a) };
	double fh{ f.f(b) };
	// Check for bracketing
	if ((fl < 0.0 && fh < 0.0) || (fl > 0.0 && fh > 0.0))
	{
		std::cerr << "\nERROR: root must be bracketed in NRSolver\n";
	}
	// Check initial guesses
	if (fl == 0.0) return xl;
	if (fh == 0.0) return xh;
	// Orient search so that f(xl) < 0
	if (fl < 0.0)
	{
		xl = a;
		xh = b;
	}
	else
	{
		xh = b;
		xl = a;
	}
	// Initial values
	double rt{ 0.5 * (a + b) };
	double dx_old{ abs(b - a) };
	double dx{ dx_old };
	double val{ f.f(rt) };
	double dval{ f.df(rt) };
	// Do the search
	for (int i = 0; i < MAXITS; i++)
	{
		// Bisect of out of range or not converging rapidly enough
		if ((((rt - xh) * dval - val) * ((rt - xl) * dval - val) > 0.0)
			|| (abs(2.0 * val) > abs(dx_old * dval)))
		{
			dx_old = dx;
			dx = 0.5 * (xh - xl);
			rt = xl + dx;
			if (xl == rt) return rt;
		}
		// If root is valid, continue with Newton Raphson
		else
		{
			dx_old = dx;
			dx = val / dval;
			double temp = rt;
			rt -= dx;
			if (temp == rt) return rt;
		}
		// Check for convergence
		if (abs(dx) < TOL) return rt;
		// New function evaluation
		val = f.f(rt);
		dval = f.df(rt);
		// Maintain bracket on the root
		if (val < 0.0)
		{
			xl = rt;
		}
		else
		{
			xh = rt;
		}
	}
	// If root hasn't been found, throw an error
	std::cerr << "\nERROR: MAXITS exceeded in Newton-Raphson solver\n";
	exit(EXIT_FAILURE);
}