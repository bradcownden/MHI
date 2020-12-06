#pragma once

enum class QRFactor_Error
{
	SUCCESS,
	BUILD_ERROR,
	SIZE_ERROR,
	SPARSE_SIZE_ERROR,
	SPARSE_BUILD_ERROR,
	SINGULARITY_ERROR,
	BAD_BUFFER_SOLVE,
	BAD_BUFFER_FACTOR
};

void getQRFactorError(QRFactor_Error &error);
