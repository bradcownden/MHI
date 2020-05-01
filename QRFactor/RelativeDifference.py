import numpy as np
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
    # Read the files at a specific time and determine
    # the absolute normalized difference
    files = ["Factor_t850.txt"]
    colours = ['b', 'r', 'g']
    plt.figure()
    for i in range(len(files)):
        gpu_file = 'GPU' + files[i]
        cpu_file = 'CPU' + files[i]
        print("Reading data files %s and %s..." % (cpu_file, gpu_file))
        gpu_data = readFile(gpu_file)
        cpu_data = readFile(cpu_file)

        print("Done! Computing normalized absolute differences...")
        data = computeRelativeDifference(cpu_data, gpu_data)

        print("Done! Plotting data...")
        plotlabel = files[i].split("_t")[1]
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
    plt.savefig("CPUvsGPUFactor2.pdf", transparent=True, format='pdf',
        bbox_inches='tight')
    plt.show()

###############################################
###############################################

main()

###############################################
###############################################
