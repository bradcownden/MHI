"""
Directly compare the output from QRFactor -- both GPU 
and CPU methods -- to the results provided by MHI. Furthermore,
use the data from both compiler methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import os, re, itertools

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
    foo = []
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
    # Read all the output files from the three GPUs tested
    # and compare to the output files
    
    p100_dir = "C:/Users/bradc/Documents/MHI/QRFactor/P100_Out/"
    v100_dir = "C:/Users/bradc/Documents/MHI/QRFactor/V100_Out/"
    data_dir = "C:/Users/bradc/Documents/MHI/QRFactor/"

    # Histogram for the relative differences and custom bin sizes
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10,6))
    my_bins = [10 ** x for x in range(-16,1)]
    I = 0
    GPUs = ["Quadro", "P100", "V100"]

    p100_files = [p100_dir + x for x in os.listdir(p100_dir) if "GPUFactor_t" in x]
    v100_files = [v100_dir + x for x in os.listdir(v100_dir) if "GPUFactor_t" in x]
    quadro_files = [data_dir + x for x in os.listdir(data_dir) if "GPUFactor_t" in x]


    # Generate unique combinations of items
    for x, y in itertools.combinations([p100_dir, v100_dir, data_dir], 2):
        print("%s vs. %s" % (x, y))
        x_files = [x + s for s in os.listdir(x) if "GPUFactor_t" in s]
        x_files.sort(key=lambda f: int(re.sub('\D', '', f)))
        y_files = [y + s for s in os.listdir(y) if "GPUFactor_t" in s]
        y_files.sort(key=lambda f: int(re.sub('\D', '', f)))
        
        diffs = np.zeros((len(x_files), LINES))
        for i in range(len(x_files)):
            # Do time/length check
            if (fileCheck(x_files[i], y_files[i])):
                diffs[i,:] = computeRelativeDifference(x_files[i], y_files[i])
            else:
                print("ERROR: File check failed for %s and %s" % (x_files[i],
                    y_files[i]))
        # Make the plot
        axs[I].hist(diffs.flatten(), bins=my_bins, color='C' + str(I))
        axs[I].set_xscale('log')
        axs[I].set_xlabel("Relative Difference Value")
        axs[I].set_ylabel("Counts")
        axs[I].set_yscale('log')
        axs[I].grid(True)
        axs[I].set_ylim(0.5, LINES * 298)
            
        # Make the title
        if x == data_dir:
            title1 = GPUs[0]
        elif x == p100_dir:
            title1 = GPUs[1]
        else:
            title1 = GPUs[2]

        if y == data_dir:
            title2 = GPUs[0]
        elif y == p100_dir:
            title2 = GPUs[1]
        else:
            title2 = GPUs[2]

        axs[I].set_title("%s vs %s" % (title1, title2))
        # Increment 
        I += 1
        print("Done!")
        
    # Show the result
    plt.tight_layout()
    outfile = "QRFactor_GPUCrossCompare.pdf"
    plt.savefig(outfile, transparent=True, format='pdf')
    plt.show()


###############################################
###############################################

main()

###############################################
###############################################
