#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>

#include "Io.h"
#include "debug.h"

// Ignore unsafe/deprecated warnings for sprintf and fopen
#pragma warning(disable : 4996)

// File I/O error handling
void getIOError(IoError error, char* filename)
{
	switch (error)
	{
	case IoError::FILE_READ_ERROR:
		std::cerr << "\nERROR: could not read " << filename << "\n";
		break;
	case IoError::FILE_WRITE_ERROR:
		std::cerr << "\nERROR: could not write to " << filename << "\n";
		break;
	case IoError::SUCCESS:
		break;
	default:
		std::cerr << "\nUnrecognized I/O error\n";
		break;
	}
}

// Read the data from a file and into a vector
void readFile(char* inFile, const int rowsA, double* inPtr)
{
	IoError error;
	std::ifstream file(inFile);
	if (file.is_open())
	{

#ifdef DEBUG
		printf("Reading data from %s\n", inFile);
#endif // DEBUG

		std::string line;
		std::string::size_type val;
		int count{ 0 };
		while (count < rowsA) // Test to ensure too much data is not read
		{
			getline(file, line);
			inPtr[count] = std::stod(line, &val);
			++count;
		}
		error = IoError::SUCCESS;
	}
	else
	{
		error = IoError::FILE_READ_ERROR;
	}

	getIOError(error, inFile);

	file.close();

}

// Write the data from a vector to a file
void writeFile(char* outFile, const int rowsA, double* outPtr)
{
	IoError error;
	FILE* out = fopen(const_cast<char*>(outFile), "w");
	if (out)
	{

#ifdef DEBUG
		printf("Writing data to %s\n", outFile);
#endif // DEBUG
		
		int count{ 0 };
		while (count < rowsA) // Test to ensure too much data is not written
		{
			fprintf(out, "%1.15e\n", outPtr[count]);
			++count;
		}

		error = IoError::SUCCESS;
	}
	else
	{
		error = IoError::FILE_WRITE_ERROR;
	}

	getIOError(error, outFile);

	fclose(out);

}
