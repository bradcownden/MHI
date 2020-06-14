
import numpy as np
import matplotlib.pyplot as plt
import os, re

###############################################
###############################################

min_threshold = 1.0E-16
max_threshold = 1.0E-7
LINES = 7362


def computeRelativeDifference(mhi_file, my_file):
    # Calculate relative difference for each line
    # Return the line number and the difference value
    # if the difference exceeds max_threshold
    mhi_data = np.genfromtxt(mhi_file)
    my_data = np.genfromtxt(my_file)

    diff = []
    for i in range(len(mhi_data)):
        if mhi_data[i] == 0.0 and my_data[i] == 0.0:
            diff.append(min_threshold)
        else:
            foo = abs(mhi_data[i] - my_data[i]) / max(abs(mhi_data[i]),
                abs(my_data[i]))
            if foo < min_threshold:
                diff.append(min_threshold)
            else:
                diff.append(foo)
    
    return diff

def fileCheck(file1, file2):
    # Check to make sure the two files being compared are from
    # the same time step. All data starts with time step t = 1

    f1_t = file1.split("_t")[1]
    f1_t = f1_t.split(".txt")[0]

    f2_t = file2.split("_t")[1]
    f2_t = f2_t.split(".txt")[0]

    f1_data = np.genfromtxt(file1)
    #print(len(f1_data))
    f2_data = np.genfromtxt(file2)
    #print(len(f2_data))

    if f1_t != f2_t:
        print("ERROR: Unmatched file times!")
        print(file1, file2)
        return False
    elif (len(f1_data) != len(f2_data)):
        print("ERROR: Output vectors do not have the same length!")
        print("%s: %d \n%s: %d" % (file1, len(f1_data), file2, len(f2_data)))
        return False
    else:
        return True

###############################################
###############################################

def main():
    # Read all the output files from the two sets of compilers
    # and compare the results
    
    mhi_c1dir = "C:/Users/bradc/Documents/MHI/GPU_Data/CompilerIF15/"
    mhi_c2dir = "C:/Users/bradc/Documents/MHI/GPU_Data/CompilerGF462/"
    #data_dir = "C:/Users/bradc/Documents/MHI/QRFactor/"
    bfile = "SysVecB"
    xfile = "SysVecX"

    # Histogram for the relative differences and custom bin sizes
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,8))
    my_bins = [10 ** x for x in range(-16,1)]
    I = 0
    colour = 0
    MHI_comps = ["CompilerIF15", "CompilerGF462"]

    """
    # Compare both types of files from the two compilers
    for ftype in [xfile, bfile]:
        mhi_c1 = [mhi_c1dir + x for x in os.listdir(mhi_c1dir) if ftype in x]
        mhi_c2 = [mhi_c2dir + x for x in os.listdir(mhi_c2dir) if ftype in x]

        # Sort by time step
        mhi_c1.sort(key=lambda f: int(re.sub('\D', '', f)))
        mhi_c2.sort(key=lambda f: int(re.sub('\D', '', f)))

        # Trim files that are not present in all sets
        tmax = min(len(mhi_c1), len(mhi_c2))
        mhi_c1 = mhi_c1[0:tmax]
        mhi_c2 = mhi_c2[0:tmax]

        # Iterate through all files and create a histogram
        print("Calculating relative differences in %s files between MHI compilers..." % ftype)
        diffs = np.zeros((tmax, LINES))
        for i in range(tmax):
            # Do time/length check
            if (fileCheck(mhi_c1[i], mhi_c2[i])):
                diffs[i, :] = computeRelativeDifference(mhi_c1[i], mhi_c2[i])
            else:
                print("ERROR: file check failed for %s data and %s data" %\
                    (mhi_c1[i], mhi_c2[i]))
        axs[I].hist(diffs.flatten(), bins=my_bins, color='C' + str(I))
        axs[I].set_xscale('log')
        axs[I].set_xlabel("Relative Difference Value")
        axs[I].set_ylabel("Counts")
        axs[I].set_yscale('log')
        axs[I].grid(True)
        axs[I].set_ylim(0.5, 7E3 * 287)
        axs[I].set_title("%s vs %s: %s" % (MHI_comps[0], MHI_comps[1], ftype))
        I += 1
        print("Done!")

    # Show the result
    plt.tight_layout()
    outfile = "MHI_DirectCompare.pdf"
    plt.savefig(outfile, transparent=True, format='pdf')
    plt.show()
    plt.clf()
    """

    # Do the same comparison for the system matrix data from each compiler 
    # for all times
    time = 1
    while(True):
        matC1, matC2 = {}, {}
        matdirs = [mhi_c1dir, mhi_c2dir]
        matdicts = [matC1, matC2]
        # Test for existence of files at this time step
        if not os.path.isfile(matdirs[0] + 'sysMatA_t' + str(time) + '.mtx') or\
            not os.path.isfile(matdirs[1] + 'sysMatA_t' + str(time) + '.mtx'):
            break
        # Otherwise, continue with the comparison
        for i in range(2):
            print("Reading system matrix from", matdirs[i], "for time t =", time)
            with open(matdirs[i] + 'sysMatA_t' + str(time) + '.mtx', 'r') as f:
                for line in f:
                    # Skip lines with leading '%'
                    if '%' in line:
                        pass
                    # Otherwise, enter the matrix value
                    else:
                        try:
                            data = [float(x) for x in line.split()]
                            matdicts[i][(int(data[0]), int(data[1]))] = data[2]
                        except ValueError:
                            print("ERROR: tried to convert string to float:", line)
                            pass
            print("Done!")

        # Look for differences greater than the threshold difference
        for key in matC1.keys():
            if key not in matC2.keys():
                print("ERROR: Key value", key, "missing from matrix in", mhi_c2dir)
            else:
                diff = abs(matC1.get(key) - matC1.get(key))
                if matC1.get(key) == 0.0 and matC2.get(key) == 0.0:
                    pass
                else:
                    diff /= max(abs(matC1.get(key)), abs(matC2.get(key)))
                    if diff < min_threshold:
                        pass
                    else:
                        print("Relative error of", diff, "found at", key)
                        print("%s matrix value: %f\t %s matrix value: %f" %\
                            (MHI_comps[0], matC1.get(key), MHI_comps[1], matC2.get(key)))
        
        time += 1



###############################################
###############################################

main()

###############################################
###############################################
