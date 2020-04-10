"""

Read data from FullTimings.txt and plot the result

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import codecs # Reading Windows data files

# Data file
dfile = "FullTimings.txt"

# Windows encoding
dcode = 'utf_16_le'

# Read the data into an array
t_factor = []
t_solve = []

if os.path.isfile(dfile):
    with codecs.open(dfile, encoding=dcode) as f:
        for line in f:
            # Read time for QR factoring
            if "GPU factor timing" in line:
                foo = line.split()
                t_factor.append(float(foo[-2]))
            # Read time for solving
            elif "GPU solve" in line:
                foo = line.split()
                t_solve.append(float(foo[-2]))
            else:
                pass

else:
    print("\nERROR: couldn't find data file", dfile)
    print("Exiting...\n")

plt.figure()
plt.plot(np.arange(len(t_solve)), t_solve[:], '.b', label='Sparse Function Solve Time', markersize=3)
plt.xlabel('Time Step')
plt.ylabel('Solve Time (ms)')
plt.grid(True)
plt.title(r'Solving $R \mathbf{x} = Q^T \mathbf{b}$')
plt.savefig("FullTimings.pdf", transparent=True, format='pdf', bbox_inches='tight')
print("Saved figure as FullTimings.pdf")
plt.show()
