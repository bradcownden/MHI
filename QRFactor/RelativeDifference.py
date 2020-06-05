import numpy as np
import random as ran
import os, re
import matplotlib.pyplot as plt

###############################################
###############################################

threshold = 1.0E-16

def computeRelativeDifference(cpu_data, gpu_data):
    # Calculate absolute difference in each line between the data sets
    diff = {}
    for i in range(len(cpu_data)):
        if cpu_data[i] == 0.0 and gpu_data[i] == 0.0:
            diff[i] = threshold
        else:
            diff[i] = abs(cpu_data[i] - gpu_data[i]) / max(abs(cpu_data[i]),
                abs(gpu_data[i]))
            if diff[i] < threshold:
                diff[i] = threshold
            else:
                pass
    """
    # Normalize to the average
    av = 0.0
    for val in rel.values():
        av += val
    av /= len(rel)
    for key in rel.keys():
        rel[key] = rel[key] / av
    return rel
    """
    return diff


def readFile(infile):
    # Test for file existing
    if os.path.isfile(infile):
        # Read in the data
        data = []
        with open(infile, 'r') as f:
            for line in f:
                try:
                    data.append(float(line.strip()))
                except:
                    print("\nERROR: non-numerical value in %s:" % infile, line)
                    print("\n")
        return data
    else:
        print("\nERROR: could not find file", infile)
        print("\n")

###############################################
###############################################

def main():
    # Read the three random output files and determine
    # the absolute normalized difference
    ran.seed()
    gfiles = [f for f in os.listdir() if "GPUFactor_t" in f]
    cfiles = [f for f in os.listdir() if "CPUFactor_t" in f]
    tvals = []

    for file in gfiles:
        t = file.split("_t")[1]
        tvals.append(int(t.split(".txt")[0]))
    ts = [ran.randint(1, len(tvals)) for i in range(3)]
    tvals = [tvals[i] for i in ts]
    print(tvals)
    files = []
    for t in tvals:
        files.append([(f,g) for f,g in zip(cfiles, gfiles) if\
            str(t) in f and str(t) in g])
    files = [a for b in files for a in b]
    print(files)

    plt.figure()
    colours = ['r', 'b', 'g']
    for i in range(len(files)):
        pair = files[i]
        print("Reading data files %s and %s..." % (pair[0], pair[1]))
        gpu_data = readFile(pair[0])
        cpu_data = readFile(pair[1])

        print("Done! Computing normalized absolute differences...")
        data = computeRelativeDifference(cpu_data, gpu_data)

        print("Done! Plotting data...")
        plotlabel = pair[0].split("_t")[1]
        plotlabel = plotlabel.split(".txt")[0]
        # Legend entry
        plt.plot(0, threshold, '.' + colours[i], markersize=2,
            label='t = ' + plotlabel)
        # Plot log_10(data) against line number
        for line, val in data.items():
            plt.plot(line + 1, val, '.' + colours[i], markersize=2)

    plt.xlabel(r'Line number')
    plt.ylabel(r'Relative Difference')
    plt.yscale('log')
    plt.legend()
    print("Done!")
    plt.savefig("CPUvsGPURelativeDifference_tolE-15.pdf", format='pdf',
        bbox_inches='tight')
    plt.show()

###############################################
###############################################

main()

###############################################
###############################################
