# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:07:44 2019

@author: bcownden
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys, datetime

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

# Write non-zero entries to .mtx file in column-ordered list
def writeData(outfile, row, col, val):
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


#####################################
#####################################
    
def main(dirpath):
    
    
    os.chdir(dirpath)
    print("Current working directory:", dirpath)
    
    
    # Iterate through directories and get the total size of the system matrix
    sizes = []
    for i in range(1,40):
        target = "./" + str(i) + "/matrixA.txt"
        if os.path.isfile(target):
            sizes.append(getTotalMatrixSize(target))
        else:
            print("Could not find", target)
            print("Ending loop at i =", i)
            break


    # Load non-zero values for each submatrix in each directory into a system
    # M matrix
    print("Loading sparse matrices.")
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
                # Get the indices of the nonzero values
                ivals, jvals = np.nonzero(temp)
                # Nonzero indices are Msys key (base 1), element value is Msys value
                for x,y in zip(ivals, jvals):
                    Msys[(x + major_offset + minor_offset + 1, 
                    y + major_offset + minor_offset + 1)] = temp[x][y]
                # Increment minor offset value for next submatrix
                minor_offset += submat.get(key)[0]
            # Increment major offset value for next directory
            major_offset += sizes[i - 1]
        else:
            print("Could not find", target)
    

    # Write header of .mtx output file
    m_out = "sysMatA.mtx"
    sys_size = sum(sizes)
    writeIntro(m_out, sys_size, len(Msys))
    
    
    # Sort system matrix into column-major format
    # and write to file in .mtx format
    print("Writing system matrix to", m_out)
    ii = 0
    total = len(Msys)
    for key in sorted(Msys.keys(), key = lambda x: x[1]):
        print("\rFinished %d%%..." % int(ii / total * 100), end='')
        row, col = key
        val = Msys.get(key)
        #writeData(m_out, row, col, val)
        ii += 1

    print("\rFinished 100%...", end='')
    print("\nDone.")
    
    
    """
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
    
    
    # Hardcoded test feature
    mode = "test"
    directory = 36
    fail = False
    if mode == "test":
        # Test values in system matrix against individual directories
        test_mat = readMatrix(str(directory) + "/matrixA.txt")
        print("Testing system matrix against direct reading of %s" %
              str(directory) + "/matrixA.txt")
        # Offset will be the sum of the sizes of the directories up
        # to the test directory
        offset = sum(sizes[:(directory - 1)])
        # Shift is the sum of the sizes of subsystems
        shift = 0
        for key in sorted(test_mat):
            # Compare non-zero values of subsystem to system matrix 
            temp = test_mat.get(key)[1:][0]
            sub_size = test_mat.get(key)[0]
            # Find non-zero values in subsystem
            for i in range(sub_size):
                for j in range(sub_size):
                    if temp[i][j] != 0.0:
                        # Compare this value to the value found in the 
                        # full system matrix
                        test_key = (offset + shift + i + 1, offset + shift + j + 1)
                        if test_key in Msys:
                            difference = temp[i][j] - Msys.get(test_key)
                            if difference != 0.0:
                                print("ERROR: non-zero difference between system and direct read matrices!")
                                print("From direct read: (%d, %d) = %f" % (i, j, temp[i][j]))
                                print("Corresponding element in sysMatA:", Msys.get(test_key))
                                print("Difference:", difference)
                                print(temp)
                                fail = True
                        else:
                            print("ERROR: failed to find value in Msys with key", test_key)
                            fail = True
                    else:
                        pass
                        
            # Increment shift by subsystem size
            shift += sub_size
         
        # Return success/fail    
        if fail:
            print("*******************************")
            print("\nERROR: system matrix differs from direct read of %s\n" %
                  str(directory) + "/matrixA.txt")
            print("*******************************")
        else:
            print("Check completed! System matrix matches direct read of", 
                  str(directory) + "/matrixA.txt")
    # Ignore if not testing
    else:
        pass
    

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