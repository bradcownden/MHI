"""
Read Province39/folder/VectorX.txt and combine subsystems into a
total system file at each time step
"""
import os, re
import numpy as np

##############################
##############################

MAINDIR = "C:/Users/bradc/Documents/MHI/GPU_Data/CompilerIF15/"
LINES = 7362

##############################
##############################

def writeX(fnum, data, tstep):
    fname = "f" + str(fnum) + "VecX_t" + str(tstep) + ".txt"
    with open(MAINDIR + fname, 'w') as f:
        for val in data:
            f.write("%1.14e\n" % val)

def writeSys(outfile, outdata):
    with open(outfile, 'w') as f:
        for val in outdata:
            f.write("%1.14e\n" % val)


def dataRead(infile):
    fsize, lcount = 0, 0
    key = ()
    data = {}
    # Get the total number of lines in the file
    with open(infile, 'r') as f:
        for line in f:
            fsize += 1

    # Read each line in the file and look for duplicate
    # data entries. According to MHI, if an entry is repeated
    # it is because the calculation was redone with a higher
    # resolution. So, the final repeat of the data is the 
    # correct answer. Also, look for lines where data length 
    # does not match the size given in the key value. This will
    # mean that the data continues on the next line and will need
    # to be appended to the current line
    
    f = open(infile, 'r')
    while(lcount < fsize):
        line = f.readline().split()
        lcount += 1
        # Test for empty line
        if len(line) == 0:
            print("Empty line at", lcount)
        # Test for text
        if 'For' in line[0] or 'On' in line[0] or 'Size' in line[0]:
            pass
        # Test for delimiter
        elif '-----------------------------------' in line[0]:
            pass
        # Test for key set
        elif len(line) == 3 and '.' not in line[0]:
            key = tuple(int(x) for x in line)
        # Test for data line
        else:
            vals = [float(x) for x in line]
            # If the key set gives a length longer than the number
            # of entries in that line, the next line down will need
            # to be read as well
            if key[-1] != len(vals):
                reRead = True
                # Keep reading data lines until the correct number of entries is found
                while (reRead):
                    vals += [float(x) for x in f.readline().split()]
                    lcount += 1
                    # Now check lengths
                    if key[-1] != len(vals):
                        #print("ERROR: additional data at line %d of %s not properly appended"\
                        #    % (lcount, f.name))
                        #print("Expected data of length %d, got data of length %d\n"\
                        #    % (key[-1], len(vals)))
                        print("Continuing to read data...")
                    elif key[-1] == len(vals):
                        #print("Done reading multi-line data from", f.name)
                        #print("Expected length: %d, real length: %d"\
                        #    % (key[-1], len(vals)))
                        reRead = False
                    else:
                        print("Length error in %s at line %d" % (f.name, lcount))
                        reRead = False
                # Lengths are now correct: add the data
                if key in data.keys():
                    #print("Repeated key", key, "at line", lcount)
                    pass
                data[key] = vals
                key = ()
            # Lengths are already correct
            else:
                if key in data.keys():
                        #print("Repeated key", key, "at line", lcount)
                        pass
                data[key] = vals
                key = ()
    f.close()
    #print("Read %d/%d lines" % (lcount, fsize))
    return data

##############################
##############################

def main(maindir):

    # Read the VectorX.txt file in each numbered directory and write out the 
    # folder file at each time step, e.g. f1VecX_t0.txt, by assembling the
    # subsystems

    indir = maindir + "Province39/"
    for i in range(1,40):
        datafile = indir + str(i) + "/VectorX.txt"
        subsys = dataRead(datafile)
        # Get maximum time step
        keys = sorted(subsys.keys(), key=lambda x:x[1], reverse=True)
        tmax = keys[0][1]
        print("Max time:", tmax)
        # Get highest subsystem number
        N = sorted(subsys.keys(), key=lambda x:x[0], reverse=True)
        N = N[0][0]
        print("Number of subsystems:", N)
        # Get total number of entries by summing over all the subsystems
        # from a single time step -- avoid the last time step as this seems
        # to be incomplete
        lines = 0
        for key in keys:
            if key[1] == 0:
                lines += len(subsys.get(key))
        print("Total number of lines:", lines)
        # Remove dictionary entires that don't have all the subsystems
        for t in range(tmax+1):
            subkeys = [key for key in keys if key[1] == t]
            #print(subkeys)
            if sorted(subkeys, key=lambda x:x[0], reverse=True)[0][0] < N:
                for keydel in subkeys:
                    print("Removing dictionary entry", keydel)
                    subsys.pop(keydel)
            else:
                pass
        # Update the set of keys now that entries have been removed
        keys = sorted(subsys.keys(), key=lambda x:x[1], reverse=True)
        # Write out the subsystems to file
        outvec = []
        for key in keys:
            (n, t) = key[0:2]
            #print(n, t)
            if n < N:
                outvec.append(subsys.get(key))
            elif n == N:
                outvec.append(subsys.get(key)) 
                # Flatten
                outvec = [x for y in outvec for x in y]
                # Length check
                if len(outvec) != lines:
                    print("ERROR: subsystem does not have the correct number" +
                    " of elements. Should be %d, got %d" % (lines, len(outvec)))
                    outvec = []
                else:
                    #print("Writing to file f%dVecX_t%d.txt" % (i, t + 1))
                    writeX(i, outvec, t)
                    outvec = []
            else:
                print("ERROR: number of subsystems or time step greater than" +
                " maximum values.")
                print("Subsystem: %d, time step: %d" % (n, t))

    
    # Read the output from the section above and combine each of the folder files 
    # into system files
    t = 1
    while True:
        sysfiles = [x for x in os.listdir(maindir) if ("VecX_t" + str(t) 
            + ".txt") in x]
        # Make sure all the folders are there
        if len(sysfiles) != 39:
            print("ERROR: Vector files missing from some folders.")
            print("Should have 39 folders, instead have", len(sysfiles))
            print("Maximum time step with all folders:", t - 1)
            break
        else:
            # Sort the files by folder
            sysfiles.sort(key=lambda f: int(re.sub('\D', '', f)))
            # Iterate through the files, each time appending to a single 
            # system vector. Write out system vector for each time step
            out = []
            for file in sysfiles:
                out.append(np.genfromtxt(maindir + file))
            # Flatten
            out = [x for y in out for x in y]
            # Length check
            if len(out) != LINES:
                print("ERROR: incorrect number of lines. Only %d of %d" % (len(out), LINES))
                out = []
            else:
                outfile = maindir + "SysVecX_t" + str(t) + ".txt"
                print("Writing full system vector for t =", t)
                writeSys(outfile, out)
                out = []
        t += 1

    
##############################
##############################

main(MAINDIR)

##############################
##############################