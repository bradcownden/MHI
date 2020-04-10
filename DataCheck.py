import numpy as np
import os, re

#########################################
#########################################

ROWSA = 7362

#########################################
#########################################

def readFile(infile):
    data = np.genfromtxt(infile, dtype=np.double)
    return len(data)

def replaceFile(file):
    foo = file.split("_")[1][1:]
    tout = int(foo.split(".")[0])
    tin = tout - 1
    fin = file.split("_")[0] + "_t" + str(tin) + ".txt"
    print("Rewriting %s with data from %s..." % (file, fin))
    np.savetxt(file, np.genfromtxt(fin, dtype=np.double), fmt='%1.15e')

#########################################
#########################################

def main():
    # Move to the data directory
    os.chdir("C:/Users/bradc/Documents/MHI/Output/Province/system")
    # Get a list of the text files in the directory
    flist = [file for file in os.listdir() if 'sysVecB' in file]
    flist.sort(key=lambda x: int(re.sub('\D', '', x)))
    # Iterate through the files and check that they are the correct
    # length. If not, copy the data from the previous file into
    # the current file
    for file in flist:
        size = readFile(file)
        if size != ROWSA:
            print("WARNING: file %s is not of length %d" % (file, ROWSA))
            replaceFile(file)
        else:
            pass
    print("Done!")

#########################################
#########################################

main()

#########################################
#########################################
