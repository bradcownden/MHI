# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:07:44 2019

@author: bcownden
"""

import numpy as np
import os, sys

#####################################
#####################################

def usage():
    print("\nRead the Province data set by directory and reconstruct the " +
          "time dependence of the full system. Will read the VectorB.txt file "
          + "in each directory, combine the subsystems into a single vector at "
          + "each time step, and write out the result.\n")
    print("usage:  python %s /path/to/Province fstart fstop" % sys.argv[1])
    print("\t/path/to/Province: the location of the Province directory, " +
          "which contains the 39 subfolders.")
    print("\tfstart: initial directory (int)")
    print("\tfstop: final directory, inclusive (int). Note that it is assumed " +
          "that subsequent directories are indexed by 1.\n")

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

def readVector(infile, lncnt, verbose):
    vecdata = {}
    if os.path.isfile(infile):
        print("------------------")
        print("Reading", infile)
        # Read subsystem number, timestep. Store in dictionary
        lcount = 1
        with open(infile, 'r') as f:
            for line in f:
                if (verbose):
                    print("\rReading line %d/%d." % (lcount, lncnt), end='')
                else:
                    pass
                lcount += 1
                line = line.split()
                # Skip empty lines
                if len(line) > 0:
                    try:
                        # Find timestep value descriptor line: whole number that is not
                        # part of the data
                        if (float(line[0]) % 1) == 0.0 and float(line[0]) != 0.0:
                            # This line contains the subsystem number, timestep, and size.
                            # The next row is all dashes, followed by a single row with
                            # (size) number of entries, another of dashes, an empty line.
                            subsys = int(line[0])
                            tstep = int(line[1])
                            # Skip to current line number, read a single line and store in
                            # the dictionary with the key as the timstep as the first value 
                            # and the subsystem number as the second value.
                            if (subsys, tstep) not in vecdata.keys():
                                vecdata[(subsys, tstep)] = [np.genfromtxt(infile, skip_header=lcount,
                                    max_rows=1, autostrip=True)]
                            else:
                                if (verbose):
                                    print("\nFound duplicate: (subsys, tstep) = (%d, %d)" % (subsys, tstep))
                                else:
                                    pass
                        else:
                            pass
                                                
                    # Catch conversion from string to float
                    except ValueError:
                        pass
                else:
                    pass
        print("\nFinished reading", infile)        

    # Catch file DNE
    else:
        print("\n\nERROR: Could not find", infile)
    
    return vecdata
                            

#####################################
#####################################
    
def main(dirpath, fstart, fstop):
    
    # Verbosity for the debugging
    verbose = True
    
    os.chdir(dirpath)
    dirs = sorted(os.listdir(), key=int)
    
    while (int(fstart) <= int(fstop)):
        # move into a folder
        os.chdir('./' + fstart)
        if verbose:
            print("Current directory:", os.getcwd())
        else:
            pass
        
        # get total numer of lines in VectorB.txt
        lncnt = 0
        with open('VectorB.test.txt', 'r') as f:
            for line in f:
                lncnt += 1
        
        # Start reading the RHS file
        subvec = readVector('VectorB.test.txt', lncnt, verbose)
        nsys, maxtime = sorted(subvec.keys())[-1]
        # Timestep numbering is base 0;
        maxtime += 1
        # subsystem numbering is base 1
        
        if verbose:
            print("(Number of subsystems, maximum timesteps) = (%d, %d)" 
                  % (nsys, maxtime))
        else:
            pass
        
        keylist = sorted(subvec.keys())
            
        if verbose:
            print("\nWriting full system vector at each timestep into",
                  os.getcwd())
        else:
            pass
        
        # Do the writing
        for i in range(maxtime):
            vout = []
            vecoutfile = 'f' + fstart + '_VectorBtest_t' + str(i) + '.txt'
            for j in range(nsys):
                if verbose:
                    print(subvec.get(keylist[i + j * maxtime])[0])
                vout.append(subvec.get(keylist[i + j * maxtime])[0])
                # Flatten the list of lists
            vout = [item for sublist in vout for item in sublist]
            if verbose:
                print(vout)
                print("--------------------")
                # Save the full VectorB for this folder and timestep
                np.savetxt(vecoutfile, vout, fmt='%1.16e')
        
        # increment and move out of the folder        
        fstart = str(int(fstart) + 1)
        os.chdir("../")
    
    """
     # Iterate through folders and plot each reconstructed matrix
    for folder in sorted(dirs):
        matinfile = indir + '/' + folder + '/matrixA.txt'    
        vecinfile = indir + '/' + folder + '/VectorB.txt'
        
        # Read all the subsystems at a certain timestep, combine into one
        # large vector in subsystem order and save with timstep number
        # in the filename. Repeat for next timestep.

        # Get RHS vectors
        subvecs = readVector(vecinfile)
        try:
            nsys, maxtime = sorted(subvecs.keys())[-1]
            # Timestep numbering is base 0;
            maxtime += 1
            # subsystem numbering is base 1
            
            keylist = sorted(subvecs.keys())
            
            # Remove VectorB.txt from file name and get folder number
            foo = vecinfile[:-len("\VectorB.txt")]
            # Change directory to print in current folder
            os.chdir(foo)
    
            print("\nWriting full system vector at each timestep into",
                  os.getcwd())
            for i in range(maxtime):
                vout = []
                vecoutfile = 'f' + folder + '_VectorB_t' + str(i) + '.txt'
                for j in range(nsys):
                    #print(subvecs.get(keylist[i + j * maxtime])[0])
                    vout.append(subvecs.get(keylist[i + j * maxtime])[0])
                # Flatten the list of lists
                vout = [item for sublist in vout for item in sublist]
                print(vout)
                print("--------------------")
                # Save the full VectorB for this folder and timestep
                np.savetxt(vecoutfile, vout, fmt='%1.16e')
                
        except IndexError:
            pass
            
     # check to see if this folder has already been started
        if (os.path.isfile('f' + fstart + '_VectorB_t0.txt')):
            print("Found file:", 'f' + fstart + '_VectorB_t0.txt')
            # if so, check to see if all timesteps have been completed
            with open('VectorB.txt', 'r') as f:
                # get the maximum timestep value
                maxtime = 0
                for line in f:
                    line = line.split()
                    if len(line) > 0:
                        try:
                            # Find timestep value descriptor line: whole number that is not
                            # part of the data
                            if (float(line[0]) % 1) == 0.0 and float(line[0]) != 0.0:
                                # This line contains the subsystem number, timestep, and size.
                                # The next row is all dashes, followed by a single row with
                                # (size) number of entries, another of dashes, an empty line.
                                maxtime = max(maxtime, int(line[1]))
                            else: 
                                pass
                        # catch conversion from string to float
                        except ValueError:
                            pass
                    else:
                        pass

            # see if final timestep file has been written already                    
            if (os.path.isfile('f' + fstart + '_VectorB_t' + str(maxtime) + '.txt')):
                # if this file exists, skip the directory
                if verbose:
                    print("All %d timestep files have been printed in %s. Continuing to next directory." 
                          % (maxtime, fstart))
                skip = True
                os.chdir("../")
            # if the final timstep file doesn't exist, restart this directory
            else:
                if verbose:
                    print("Not all %d timestep files have been printed. Starting data read...")
                else:
                    pass
            
        # if folder hasn't been started, perform the data read/write

    """

#####################################
#####################################

"""
if len(sys.argv) != 4:
    usage()
else:
    dirpath, fstart, fstop = tuple(sys.argv[1:])
    main(dirpath, fstart, fstop)
"""

dirpath = "C:\\Users\\bcownden\\MHI_Data\\Output\\Province"
fstart = "2"
fstop = "2"
main(dirpath, fstart, fstop)

#####################################
#####################################