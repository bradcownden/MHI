import matplotlib.pyplot as plt
import numpy as np

indata = np.genfromtxt("sysMatA.mtx", comments='%')
(rowSize, colSize, nnz) = indata[0]
M = np.zeros((int(rowSize), int(rowSize)))
for row in indata[1:]:
	(i, j) = row[:2]
	M[int(i - 1), int(j - 1)] = 1.

plt.figure(figsize=(10,8))
plt.spy(M)
plt.show()
