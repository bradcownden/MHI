"""
Compare the MHI result data from the two compilers
and plot the relaive difference
"""

import os, sys, re
import numpy as np
import matplotlib.pyplot as plt

##############################
##############################

threshold = 1.0E-16

def fileCheck(c1_file, c2_file):
    # Check to make sure the two files being compared are from
    # the same time step
    c1_t = c1_file.split("_t")[1]
    c1_t = c1_t.split(".txt")[0]
    c2_t = c2_file.split("_t")[1]
    c2_t = c2_t.split(".txt")[0]
    if c1_t == c2_t:
        return True
    else:
        print("ERROR: Unmatched files!")
        print(c1_file, c2_file)
        return False

def relativeDifference(c1_data, c2_data):
    # Calculate relative difference 
    diff = []
    for i in range(len(c1_data)):
        if c1_data[i] == 0.0 and c2_data[i] == 0.0:
            diff.append(threshold)
        else:
            diff.append(abs(c1_data[i] - c2_data[i]) / max(abs(c1_data[i]),
                abs(c2_data[i])))
            if diff[-1] < threshold:
                diff[-1] = threshold
            else:
                pass
    return diff

##############################
##############################

def main():
    # Get lists of files from each directory
    c1_dir = "./CompilerGF462/Province_AllinOne"
    c2_dir = "./CompilerIF15/Province_AllinOne"
    c1_list = os.listdir(c1_dir)
    c2_list = os.listdir(c2_dir)
    c1_list = [c1_dir + "/" + f for f in c1_list if "VecX_t" in f]
    c2_list = [c2_dir + "/" + f for f in c2_list if "VecX_t" in f]
    # Sort by timestep
    c1_list.sort(key=lambda f: int(re.sub('\D', '', f)))
    c2_list.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    # Plot a histogram of the relative differences
    fig, ax = plt.subplots()
    my_bins = [10 ** x for x in range(-16,1)]

    for i in range(len(c1_list)):
        if fileCheck(c1_list[i], c2_list[i]):
            c1_data = np.genfromtxt(c1_list[i], dtype=np.double)
            c2_data = np.genfromtxt(c2_list[i], dtype=np.double)
            t = c1_list[i].split("_t")[1]
            t = t.split(".txt")[0]
            plt.hist(relativeDifference(c1_data, c2_data),
                bins=my_bins)
            plt.text(0.5,0.9,"t = " + t, transform=ax.transAxes)
            plt.xscale('log')
            plt.yscale('log')
            plt.ylim(0.5, 7E3)
            plt.ylabel("Counts")
            plt.xlabel("Relative Difference in Vector_X Values \n" + 
                "Between Compilers")
            plt.tight_layout()
            outfile = "CompilerComp_t" + t + ".png"
            plt.savefig(outfile)
            plt.clf()
        else:
            pass


##############################
##############################

main()

##############################
##############################
