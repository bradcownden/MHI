# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:07:44 2019

@author: bcownden
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys

#####################################
#####################################

def usage():
    print("Iterate through folders and create a single matrix for the entire system")
    print("usage:  python3 MHIMakeSystemMatrix.py /path/to/Province\n")

def getTotalMatrixSize(infile):
    size = 0
    if os.path.isfile(infile):
        with open(infile, 'r') as f:
            for line in f:
                line = line.split()
                # Test for empty line
                if len(line) > 0:
                    try:
                        # Test for whole number that is not zero (ie. part of data)
                        if (float(line[0]) % 1) == 0.0 and float(line[0]) != 0.0:
                            size += float(line[-1])
                    # Catch conversion from string to float
                    except ValueError:
                        pass
                else:
                    pass
    else:
        print("ERROR: Could not find file", infile)
        
    return int(size)
        
def readMatrix(infile):
    matdata = {}
    if os.path.isfile(infile):
        # Go through the file and load the data into a dictionary        
        lcount = 1
        with open(infile, 'r') as f:
            for line in f:
                lcount += 1
                line = line.split()
                # Test for empty line
                if len(line) > 0:
                    try:
                        # Test for whole number that is not zero (ie. part of data)
                        if (float(line[0]) % 1) == 0.0 and float(line[0]) != 0.0:
                            # This line contains the subsystem number, the timestep,
                            # the ncols, and nrows. The next line is all dashes.
                            # The order is then: a row with ncol data values,
                            # a row of empty space, a row with ncol data values, etc.
                            # After 2 x nrows, the data finishes.
                            size = int(line[-1])
                            # Skip up to current line number, read line number + size
                            # of rows and store in dictionary
                            matdata[int(line[0])] = [size, np.genfromtxt(infile, skip_header=lcount,
                                    max_rows=size, autostrip=True)]
                                
                    # Catch conversion from string to float
                    except:
                        pass
                else:
                    pass    

    else:
        print("\n\nERROR: Could not find", infile)
        
    return matdata

#####################################
#####################################
    
def main(dirpath):
    
    # Verbosity for the debugging
    verbose = True
    
    os.chdir(dirpath)

    """
    Psuedo code:
        Each directory has a matrixA.txt file that is broken up into subsystems. Read the size of
        all the subsystems for the total shape of that directory's matrix.

        Each subsystem needs to be added together along the diagonal of a directory matrix -> add
        this in blocks maybe: np.block()

        The total system matrix comes from adding each of the directory matrices together along 
        the diagonal -> add these with a diagonal call?

        System matrix is now a huge sparse matrix -> write out to .mtx format so it can be 
        read into the CUDA program.
    """
    # iterate through directories and get the total size of the system matrix
    sizes = []
    for i in range(1,40):
        target = "./" + str(i) + "/matrixA.txt"
        if os.path.isfile(target):
            sizes.append(getTotalMatrixSize(target))
        else:
            print("Could not find", target)
            print("Ending loop at i =", i)
            break


    
    Msys = {}
    major_offset = 0
    for i in range(1,40):
        target = "./" + str(i) + "/matrixA.txt"
        # Get the submatrices for the current directory
        if os.path.isfile(target):
            submat = readMatrix(target)
            minor_offset = 0
            # Increment the minor offset by the size of each subsystem,
            # so that the row position is major_offset + minor_offset + i
            for key in sorted(submat.keys()):
                temp = submat.get(key)[1:][0]
                if key == 1:
                    print(temp)
                # Get the indices of the nonzero values
                ivals, jvals = np.nonzero(temp)
                # Nonzero indices are Msys key (base 1), element value is Msys value
                for x,y in zip(ivals, jvals):
                    Msys[(x + major_offset + minor_offset + 1, 
                    y + major_offset + minor_offset + 1)] = temp[x][y]
                # Increment minor offset value for next submatrix
                minor_offset += submat.get(key)[0]
            # Increment major offset value for next directory
            #print(sizes[i])
            major_offset += sizes[i - 1]
            print(major_offset)
        else:
            print("Could not find", target)
    
    #print(Msys)

    sys_size = sum(sizes)
    # Make a visualization
    M = np.zeros((sys_size, sys_size))
    for key in Msys.keys():
        i, j = key
        M[i - 1, j - 1] = 1.0
    
    plt.figure(figsize=(8,6))
    plt.grid()
    plt.spy(M, markersize=2)
    plt.show()

    """
    To sort dictionary by second entry in dictionary value (x, y), use 
    sorted(array, key=lambda x: x[1])
    """

#####################################
#####################################

"""
if len(sys.argv) != 2:
    usage()
else:
    dirpath = sys.argv[1]
    main(dirpath)
"""
main("C:\\Users\\bcownden\\MHI_Data\\Output\\Province")