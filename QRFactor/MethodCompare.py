"""
Program to read the output files from the debug version of QRFactor,
then compute and plot the relative difference between the results of
GPU an CPU methods
"""

import numpy as np
import os, re
import matplotlib.pyplot as plt

###############################################
###############################################

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

def computeRelativeDifference(cpu_data, gpu_data):
    # Calculate and return L^2 norm of the relative difference
    # between the data sets
    rel = []
    for i in range(len(cpu_data)):
        if cpu_data[i] == 0.0 and gpu_data[i] == 0.0:
            rel.append(0.0)
        else:
            rel.append(abs(cpu_data[i] - gpu_data[i]) / max(abs(cpu_data[i]),
            abs(gpu_data[i])))
    print(np.max(rel))
    return np.sum(rel) / len(cpu_data)

def timeCheck(cpu_file, gpu_file):
    # Ensure the time steps of either file match
    cpu_time = cpu_file.split("Factor_t")[1]
    cpu_time = int(cpu_time.split(".")[0])
    gpu_time = gpu_file.split("Factor_t")[1]
    gpu_time = int(gpu_time.split(".")[0])
    if cpu_time != gpu_time:
        print("\nERROR: CPU and GPU files are not from the same time step")
        print("Tried to compare %s to %s\n" % (cpu_file, gpu_file))
        return False
    else:
        return True

###############################################
###############################################

def main():
    # Get a list of the CPU and GPU output files sorted by time step
    cpu = [file for file in os.listdir() if "CPUFactor" in file]
    cpu.sort(key=lambda x: int(re.sub('\D', '', x)))
    gpu = [file for file in os.listdir() if "GPUFactor" in file]
    gpu.sort(key=lambda x: int(re.sub('\D', '', x)))
    # Iterate through the list and compare the output of the two methods
    if len(cpu) != len(gpu):
        print("\nERROR: list of output files for CPU and GPU methods are" +
        "different lengths. Find missing files.\n")
    else:
        rel_vals = np.zeros((len(cpu), ), dtype=np.double)
        for i in range(len(rel_vals)):
            if (timeCheck(cpu[i], gpu[i])):
                cpu_data = readFile(cpu[i])
                gpu_data = readFile(gpu[i])
                if len(cpu_data) != len(gpu_data):
                    print("\nERROR: data files have different lengths")
                    print("%s:" % cpu[i], len(cpu_data))
                    print("%s:" % gpu[i], len(gpu_data))
                    print("\n")
                else:
                    rel_vals[i] = computeRelativeDifference(cpu_data, gpu_data)
        print(rel_vals)

    plt.figure()
    plt.plot(np.arange(1, len(rel_vals) + 1), rel_vals[:], '.g', markersize=6)
    plt.title("Relative Difference Between CPU and GPU \nMethods From Random Sampling")
    plt.ylabel("Relative Difference")
    plt.xlabel("")
    plt.yscale('log')
    plt.grid()
    plt.show()

###############################################
###############################################

main()

###############################################
###############################################
