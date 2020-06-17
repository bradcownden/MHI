"""

Read data from FullTimings.txt and plot the result
Read data from cedar_timings.txt and compare

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import codecs # Reading Windows data files

###########################
###########################

# Windows encoding
dcode = 'utf_16_le'

###########################
###########################

def readData(infile):
    t_factor, t_solve = [], []
    if os.path.isfile(infile):
        if infile.split("/")[-1] == 'FullTimings.txt':
            with codecs.open(infile, 'r', encoding=dcode) as f:
                for line in f:
                    # Read time to factor matrix on GPU
                    if "GPU factor timing" in line:
                        line = line.split()
                        t_factor.append(float(line[-2]))
                    elif "GPU solve time" in line:
                        line = line.split()
                        t_solve.append(float(line[-2]))
                    else:
                        pass
        else:
            with open(infile, 'r') as f:
                for line in f:
                    # Read time to factor matrix on GPU
                    if "GPU factoring time" in line:
                        line = line.split()
                        t_factor.append(float(line[-2]))
                    elif "GPU solve time" in line:
                        line = line.split()
                        t_solve.append(float(line[-2]))
                    else:
                        pass
    else:
        print("ERROR: Could not find data file", infile)

    return t_factor, t_solve

###########################
###########################

def main(datadir):
    # Data files
    d1file = datadir + "FullTimings.txt"
    d2file = datadir + "cedar_timings.txt"

    fig = plt.figure()
    gs = fig.add_gridspec(2, 3)

    # Laptop timings
    fig_ax1 = fig.add_subplot(gs[0,0:2])
    factor1, solve1 = readData(d1file)
    fig_ax1.plot(np.arange(len(solve1)), solve1, '.', color='C0')
    fig_ax1.set_title(r"Laptop: solve $R\mathbf{X} = \mathbf{B}Q^{-1}$")

    # Cedar timings
    fig_ax2 = fig.add_subplot(gs[1,0:2])
    factor2, solve2 = readData(d2file)
    fig_ax2.plot(np.arange(len(solve2)), solve2, '.', color='C3')
    fig_ax2.set_title(r"P100 GPU: solve $R\mathbf{X} = \mathbf{B}Q^{-1}$")

    # Matrix factoring
    fig_ax3 = fig.add_subplot(gs[:,-1:])
    fig_ax3.bar(r"$A=QR$", factor1, color='C0', label=r"Laptop")
    fig_ax3.bar(r"$A=QR$", factor2, color='C3', label=r"P100 GPU")
    #fig_ax3.set_title(r"Factor matrix $A$")
    fig_ax3.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=1, mode="expand", borderaxespad=0.)

    f_ymax = max(factor1[0], factor2[0])
    f_ymin = min(factor1[0], factor2[0])
    s_ymax = max(np.max(solve1), np.max(solve2))
    s_ymin = min(np.min(solve1), np.min(solve2))
    
    # Add axis labels
    for axis in [fig_ax1, fig_ax2, fig_ax3]:
        axis.set_ylabel("Time (ms)")
        axis.grid(True)
    for axis in [fig_ax1, fig_ax2]:
        axis.set_xlabel("Time Step")
        axis.set_ylim(0.0, s_ymax * 1.1)
    for axis in [fig_ax3]:
        axis.set_ylim(0.0, f_ymax * 1.1)
    
    plt.tight_layout()
    outfile = datadir + "FullTimings.pdf"
    plt.savefig(outfile, transparent=True, format='pdf', bbox='inches')
    plt.show()

###########################
###########################

main("C:/Users/bradc/Documents/MHI/Output/")

###########################
###########################