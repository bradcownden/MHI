"""
Read through subsystems and assemble the full system matrix
at each time step
"""

import numpy as np
import os, sys, datetime
import matplotlib.pyplot as plt

#####################################
#####################################

LINES = 7362

#####################################
#####################################

def getTotalMatrixSize(infile):
    # Matrix size will be constant in time, therefore only need to read
    # the t = 0 entries
    size = 0
    t = 0
    matdims = {}
    if os.path.isfile(infile):
        with open(infile, 'r') as f:
            for line in f:
                line = line.split()
                # Test for keys
                if len(line) == 4 and '.' not in line[0]:
                    keys = tuple(int(x) for x in line)
                    # If time value is not 0, exit
                    if keys[1] != 0:
                        #print("Reached time step", keys[1])
                        #print("Breaking.")
                        break
                    # If key value is new, add the size to the total
                    elif keys not in matdims.keys():
                        matdims[keys] = []
                        size += keys[-1]
                    else:
                        #print("Detected douplicate key set. Skipping...")
                        pass
                else:
                    pass
    else:
        print("ERROR: Could not find file", infile)
        
    return int(size)
        
def readMatrix(infile, timestep, MatrixSize):
    matdata = {}
    key = ()
    matsize = 0
    fsize = 0
    if os.path.isfile(infile):
        # Get total file size
        with open(infile, 'r') as f:
            for line in f:
                fsize += 1

        # Go through the file and load the data into a dictionary        
        lcount = 0
        f = open(infile, 'r')
        while (lcount < fsize):
            lcount += 1
            line = f.readline().split()
            # Test for empty line
            if len(line) == 0:
                print("Empty line at line %d of %s" % (lcount, infile))
            # Test for text
            if 'For' in line[0] or 'On' in line[0] or 'Size' in line[0]:
                pass
            # Test for delimiter
            elif '-----------------------------------' in line[0]:
                pass
            # Test for key set
            elif len(line) == 4 and '.' not in line[0]:
                key = tuple(int(x) for x in line)
            # Test for data line
            else:
                # Key length should be non-zero, having been set by a previous
                # line read
                if key[1] == timestep:
                    # There should be a key[2] x key[3] block of data to be read
                    # Continue reading lines until the size is correct
                    vals = [float(x) for x in line]
                    if (len(vals) < key[-1] * key[-1]):
                        # Some data span multiple lines and multiple blocks. Continue
                        # reading lines until the correct number of elements have been read
                        while len(vals) < (key[-1] * key[-1]):
                            vals += [float(x) for x in f.readline().split()]
                            lcount += 1
                    else:
                        pass
                    # Make sure the lengths are correct
                    if len(vals) != (key[-2] * key[-1]):
                        print("ERROR: expected to read %d x %d = %d elements" %\
                            (key[-2], key[-1], key[-2] * key[-1]))
                        print("for key", key, "in", infile)
                        print("But only got", len(vals))
                    else:
                        # Add the data the matrix dictionary with a key of
                        # (subsystem, ncols)
                        matdata[(key[0], key[-1])] = vals
                        key = ()
                # If time value exceeds target value, break loop
                elif key[1] > timestep:
                    #print("Target time step: %d. Current time step: %d. Breaking loop." %\
                    #    (timestep, key[1]))
                    break
                # If time value is less than time step, continue reading
                else:
                    pass
        f.close()
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
    # Iterate through directories and get the total size of the system matrix
    # This will not change with time
    sizes = []
    for i in range(1,40):
        target = dirpath + "Province39/" + str(i) + "/matrixA.txt"
        if os.path.isfile(target):
            sizes.append(getTotalMatrixSize(target))
        else:
            print("Could not find", target)
            print("Ending loop at i =", i)
            break
    """
    print("Matrix sizes from each subfolder:")
    for i in range(len(sizes)):
        print("Folder %d, size %d" % (i + 1, sizes[i]))
    """
    # Check total system size against known size of system vectors
    if np.sum(sizes) != LINES:
        print("ERROR: missing lines from system matrix. Should be %d x %d" %\
            (LINES, LINES))
        print("Instead, only found %d x %d" % (np.sum(sizes), np.sum(sizes)))
    else:
        print("Total system size:", np.sum(sizes))

    # Read through the folders and store all non-zero matrix values
    # in column-major format for each time step
    time = 0
    while(True):
        print("Reading sparse matrices at time t = %d..." % time)
        Msys = {}
        major_offset = 0
        minor_offset = 0
        # For each folder, read the submatrices at a specific time step
        # and reassemble them. The row and column positions of the non-zero
        # matrix elements will be key values to a system matrix
        for i in range(1,40):
            matrix = readMatrix(dirpath + "Province39/" + str(i) + "/MatrixA.txt", time, sizes[i-1])
            minor_offset = 0
            # Test the matrix size against what it is supposed to be
            readSize = np.sum([key[-1] for key in matrix.keys()])
            if readSize != sizes[i-1]:
                print("ERROR: size of matrix in folder %d should be %d x %d" %\
                    (i, sizes[i - 1], sizes[i - 1]))
                print("But read %d x %d instead" % (readSize, readSize))
                break
            # If not every folder has matrix data for all the time steps, stop the process
            elif not matrix:
                print("ERROR: incomplete matrix data for time =", time + 1)
                print("Only %d/39 folders contain matrix data at this time step" % i)
                print("WARNING: full system data ONLY for t <", time)
                break
            else:
                # Sort by subsystem
                # Increment the minor offset by the size of each subsystem so that
                # the row position is major_offset + minor_offset
                for key in sorted(matrix.keys()):
                    # Reassemble the submatrix into the correct shape
                    data = np.reshape(matrix.get(key), (key[1], key[1]))
                    """
                    print("%d/MatrixA.txt subsystem %d:" % (i, key[0]))
                    print(data)
                    """
                    # Get the positions of the non-zero values
                    ivals, jvals = np.nonzero(data)
                    # Non-zero element indicies are the key values for the system matrix (base 1)
                    for x, y in zip(ivals, jvals):
                        Msys[(x + major_offset + minor_offset + 1,
                        y + major_offset + minor_offset + 1)] = data[x, y]
                    # Increment minor offset value for next subsystem
                    minor_offset += key[1]
                # Increment major offset value for the next directory
                major_offset += sizes[i - 1]
        
        print("Done. Writing to file...")

        # System matrix now assembled; write to file in MM Format starting with t = 1
        nnz = len(Msys)
        mfile = dirpath + "sysMatA_t" + str(time + 1) + ".mtx"
        writeIntro(mfile, np.sum(sizes), nnz)

        # Write to file 
        ii = 0
        for key in sorted(Msys.keys(), key = lambda x: x[1]):
            print("\rWrote %d%% of data." % int(ii / nnz * 100), end='')
            writeData(mfile, key[0], key[1], Msys.get(key))
            ii += 1

        print("\rWrote 100% of data.", end='')
        print("\nDone. Wrote", mfile)

        time += 1

    """
    # Hardcoded test feature
    #mode = "test"
    mode = "run"
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
    """

#####################################
#####################################

main("C:/Users/bradc/Documents/MHI/GPU_Data/CompilerGF462/")

#####################################
#####################################
