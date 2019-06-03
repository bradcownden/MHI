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
    print("\tfstart: initial directory")
    print("\tfstop: final directory (inclusive). Note that it is assumed " +
          "that subsequent directories are indexed by 1.\n")

def readVec(infile):
    pass



#####################################
#####################################
    
def main(dirpath, fstart, fstop):
    os.chdir(dirpath)
    print(os.listdir())

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
fstart = "1"
fstop = "1"
main(dirpath, fstart, fstop)

#####################################
#####################################