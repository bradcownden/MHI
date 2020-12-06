#include <iostream>

#include "QRError.h"
#include "debug.h"

 void getQRFactorError(QRFactor_Error &error)
{
	 // Given the error code, return a description of what raised the error

	 switch (error)
	 {
	 case QRFactor_Error::SINGULARITY_ERROR:
		 std::cerr << "\nERROR: system matrix A is not invertible\n";
		 break;
	 case QRFactor_Error::BUILD_ERROR:
		 std::cerr << "\nERROR: error building system matrix A\n";
		 break;
	 case QRFactor_Error::SIZE_ERROR:
		 std::cerr << "\nERROR: system matrix A must be square\n";
		 break;
	 case QRFactor_Error::SUCCESS:
		 break;
	 case QRFactor_Error::SPARSE_BUILD_ERROR:
		 std::cerr << "\nERROR: error building sparse matrix\n";
		 break;
	 case QRFactor_Error::SPARSE_SIZE_ERROR:
		 std::cerr << "\nERROR: QRFactor sparse matrix is the incorrect size\n";
		 break;
	 case QRFactor_Error::BAD_BUFFER_SOLVE:
		 std::cerr << "\nERROR: Working buffer could not be allocated in solve()\n";
		 break;
	 case QRFactor_Error::BAD_BUFFER_FACTOR:
		 std::cerr << "\nERROR: Working buffer could not be allocated in factor()\n";
		 break;
	 default:
		 std::cerr << "\nUnrecognized QRFactor class error\n";
		 break;
	 }

	 error = QRFactor_Error::SUCCESS;

}
