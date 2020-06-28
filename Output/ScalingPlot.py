"""

Read data from laptop, P100, and V100 timings. Compute the 
average solve times and plot as a function of size of system

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import codecs # Reading Windows data files

###########################
###########################

# Windows encoding
dcode = 'utf_16_le'

# Size of data set
LINES = 7362

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
    d1file = "FullTimings"
    d2file = "cedar_timings_p100"
    d3file = "cedar_timings_v100"

    # Read the data for each system size, find the per-time step average
    # and plot how the timings scale with system size for each type of GPU
    quadro_factor, quadro_solve = [], []
    p100_factor, p100_solve = [], []
    v100_factor, v100_solve = [], []

    for n in [0,2,4]:
        if n == 0:
            for f in [d1file, d2file, d3file]:
                temp_factor, temp_solve = readData(datadir + f + ".txt")
                if f == d1file:
                    quadro_factor.append(temp_factor)
                    quadro_solve.append(np.average(temp_solve))
                elif f == d2file:
                    p100_factor.append(temp_factor)
                    p100_solve.append(np.average(temp_solve))
                else:
                    v100_factor.append(temp_factor)
                    v100_solve.append(np.average(temp_solve))
        else:
            for f in [d1file, d2file, d3file]:
                temp_factor, temp_solve = readData(datadir + f + "_n" + str(n) + ".txt")
                if f == d1file:
                    quadro_factor.append(temp_factor)
                    quadro_solve.append(np.average(temp_solve))
                elif f == d2file:
                    p100_factor.append(temp_factor)
                    p100_solve.append(np.average(temp_solve))
                else:
                    v100_factor.append(temp_factor)
                    v100_solve.append(np.average(temp_solve))

    # Flatten factor lists
    quadro_factor = [x for y in quadro_factor for x in y]
    p100_factor =  [x for y in p100_factor for x in y]
    v100_factor =  [x for y in v100_factor for x in y]

    # Plot the result vs system size
    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0, top=0.9)
    fig.suptitle("Timings as a Function of System Scaling Number")

    # Solve time
    ax[0].plot([2 ** x for x in np.arange(0,3)], quadro_solve[:], 'sy',
        label=r"Quadro RTX 3000")
    ax[0].plot([2 ** x for x in np.arange(0,3)], quadro_solve[:], '-y')
    ax[0].plot([2 ** x for x in np.arange(0,3)], p100_solve[:], 'sb',
        label=r"P100 GPU")
    ax[0].plot([2 ** x for x in np.arange(0,3)], p100_solve[:], '-b')
    ax[0].plot([2 ** x for x in np.arange(0,3)], v100_solve[:], 'sr',
        label=r"V100 GPU")
    ax[0].plot([2 ** x for x in np.arange(0,3)], v100_solve[:], '-r')
    ax[0].set_xticks([2 ** x for x in np.arange(0,3)])
    ax[0].legend(loc='best')
    ax[0].set_ylabel(r'$\bar{t}_{step}$ (ms)')
    ax[0].grid(True)

    # Factor time
    ax[1].plot([2 ** x for x in np.arange(0,3)], quadro_factor[:], '^y',
        label=r"Quadro TRX 3000")
    ax[1].plot([2 ** x for x in np.arange(0,3)], quadro_factor[:], '-y')
    ax[1].plot([2 ** x for x in np.arange(0,3)], p100_factor[:], '^b',
        label=r"P100 GPU")
    ax[1].plot([2 ** x for x in np.arange(0,3)], p100_factor[:], '-b')
    ax[1].plot([2 ** x for x in np.arange(0,3)], v100_factor[:], '^r',
        label=r"V100 GPU")
    ax[1].plot([2 ** x for x in np.arange(0,3)], v100_factor[:], '-r')
    ax[1].legend(loc='best')
    ax[1].set_ylabel(r'$A = QR$ (ms)')
    ax[1].set_xlabel("System Size Factor, n")
    ax[1].grid(True)


    outname = datadir + "TimingScales.pdf"
    plt.savefig(outname, transparent=True, format='pdf')
    print("Saved figure to", outname)
    plt.show()

###########################
###########################

main("C:/Users/bradc/Documents/MHI/Output/")

###########################
###########################