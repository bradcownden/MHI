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
        # Try reading with windows encoding
        with codecs.open(infile, 'r', encoding=dcode) as f:
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
        # If windows codecs doesn't work, try without it
        if len(t_solve) == 0:
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
    d3file = datadir + "cedar_v100_timings.txt"

    fig = plt.figure()
    gs = fig.add_gridspec(1, 3)

    # Laptop timings
    fig_ax1 = fig.add_subplot(gs[0,0:2])
    factor1, solve1 = readData(d1file)
    fig_ax1.plot(np.arange(len(solve1)), solve1, '.', color='C0', label=r"Laptop: Quadro RTX 3000")

    # Cedar_P100 timings
    factor2, solve2 = readData(d2file)
    fig_ax1.plot(np.arange(len(solve2)), solve2, '.', color='C3', label=r"P100 GPU")

    # Cedar_V100 timings
    factor3, solve3 = readData(d3file)
    fig_ax1.plot(np.arange(len(solve3)), solve3, '.', color='C2', label=r"V100 GPU")

    # Matrix factoring
    fig_ax2 = fig.add_subplot(gs[0,-1:])
    fig_ax2.bar(r"$A=QR$", factor1, color='C0', label=r"Laptop")
    fig_ax2.bar(r"$A=QR$", factor3, color='C2', label=r"V100 GPU")
    fig_ax2.bar(r"$A=QR$", factor2, color='C3', label=r"P100 GPU")
    fig_ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=1, mode="expand", borderaxespad=0.)

    f_ymax = max(factor1[0], max(factor2[0], factor3[0]))
    f_ymin = min(factor1[0], min(factor2[0], factor3[0]))
    s_ymax = max(np.max(solve1), max(np.max(solve2), np.max(solve3)))
    s_ymin = min(np.min(solve1), min(np.min(solve2), np.min(solve3)))
    
    # Add axis labels
    fig_ax1.set_ylabel(r"Time per solve $R\mathbf{x} = \mathbf{b}Q^{-1}$ (ms)")
    fig_ax1.set_xlabel("Time Step")
    fig_ax1.grid(True)
    fig_ax1.legend(loc='best')
    fig_ax1.set_ylim(0.0, s_ymax * 1.1)
    fig_ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)

    fig_ax2.set_ylabel("Time (ms)")
    fig_ax2.set_ylim(0.0, f_ymax * 1.1)
    fig_ax2.grid(True)
    
    plt.tight_layout()
    outfile = datadir + "FullTimings.pdf"
    plt.savefig(outfile, transparent=True, format='pdf', bbox='inches')
    #plt.show()

###########################
###########################

main("C:/Users/bradc/Documents/MHI/Output/")

###########################
###########################