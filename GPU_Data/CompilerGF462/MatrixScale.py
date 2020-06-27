"""
Read through subsystems and assemble the full system matrix
at each time step
"""

import numpy as np
import os, sys, datetime

#####################################
#####################################

MAINDIR = "C:/Users/bradc/Documents/MHI/GPU_Data/CompilerGF462/"
LINES = 7362

#####################################
#####################################

# Read the non-zero, column-ordered data from the existing matrix file
def readMatrix(infile):
    matdata = {}
    with open(infile, 'r') as f:
        for line in f:
            line = line.split()
            # Skip lines that start with %
            if '%' in line:
                #print("Read comment line; skipping")
                #print(line)
                pass
            elif len(line) == 3:
                matdata[(int(line[0]), int(line[1]))] = float(line[2])
            else:
                #print("ERROR: unexpected data line")
                #print(line)
                pass
    return matdata

# Read data from a vector file
def readData(infile):
    indata = []
    if os.path.isfile(infile):
        with open(infile, 'r') as f:
            for line in f:
                indata.append(float(line))
    else:
        print("ERROR: Could not find file", infile)

    return indata

# Write out vector data
def writeVector(data, tstep, factor):
    fout = "SysVecB_n" + str(factor) + "_t" + str(tstep) + ".txt"
    with open(MAINDIR + fout, 'a') as f:
        for val in data:
            f.write("%1.14e\n" % val)

# Write non-zero entries to .mtx file in column-ordered list
def writeMatrix(outfile, row, col, val):
    with open(outfile, 'a') as f:
        f.write("%d %d %1.12e\n" % (row, col, val))

# Write header to .mtx file
def writeIntro(outfile, size, nnz):
    date = datetime.datetime.now()
    with open(outfile, 'w+') as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write("%--------------------------------------------------------------\n")
        f.write("% \n")
        f.write("% " + "date: %s\n" % str(str(date.year) + '-' + str(date.month) + '-' + str(date.day)))     
        f.write("% \n")           
        f.write("%--------------------------------------------------------------\n")
        f.write("%d %d %d\n" % (size, size, nnz))

# Clean the w+ files from any previous runs
def cleanVectors(dirpath):
    files = [x for x in os.listdir(dirpath) if "SysVecB_n" in x]
    for f in files:
        os.remove(f)

#####################################
#####################################
    

def main(dirpath, factor):
    # Multiply the data set by the input factor so that the n x n matrix
    # becomes (factor * n) x (factor * n) and the length n input data becomes
    # length (factor * n)

    print("Multiplying dataset by a factor of %d..." % factor)
    mat_out = dirpath + "sysMat_n" + str(factor) + "_t1.mtx"
    if os.path.exists(mat_out):
        pass
    else:
        matrix = readMatrix(dirpath + "sysMatA_t1.mtx")
        # Get the number of non-zero entries
        nnz = len(matrix)
        new_mat = {} # Empty matrix
        new_mat.update(matrix) # Add existing matrix values to scaled matrix
        # Add duplicate values to scaled-up matrix
        for key in matrix.keys():
            new_mat[(key[0] * factor, key[1] * factor)] = matrix.get(key)

        # Write the new matrix to file
        writeIntro(mat_out, LINES * factor, nnz * factor)
        ii = 0
        for key in sorted(new_mat.keys(), key = lambda x: x[1]):
            print("\rWrote %d%% of data." % int(ii / (nnz * factor) * 100),
                end='')
            writeMatrix(mat_out, key[0], key[1], new_mat.get(key))
            ii += 1

        print("\rWrote 100%% of data.\n", end='')
        print("Wrote factored matrix to", mat_out)

    cleanVectors(dirpath)

    ii = 1
    while True:
        infile = dirpath + "SysVecB_t" + str(ii) + ".txt"
        if os.path.isfile(infile):
            print("Reading data from", infile)
            data = readData(infile)
            n = factor
            while (n > 0):
                writeVector(data, ii, factor)
                n -= 1
            print("Finished scaling", infile)
            ii += 1
        else:
            break
    

#####################################
#####################################

if len(sys.argv) != 2:
    print("Usage:  python MatrixScale.py factor")
    print("\tfactor: integer to multiply the size of the original system" +
        " to arrive at the new system")
else:
    factor = int(sys.argv[1])
    main(MAINDIR, factor)

#####################################
#####################################
