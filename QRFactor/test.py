import numpy as np

gpuin = "GPUFactor_t2808.txt"
cpuin = "CPUFactor_t2808.txt"
input = "../Output/Province/system/sysVecB_t2808.txt"

threshold = 1.0E-11

gpu_data = np.genfromtxt(gpuin, dtype=np.double)
cpu_data = np.genfromtxt(cpuin, dtype=np.double)
input_data = np.genfromtxt(input, dtype=np.double)

print("File size:", len(gpu_data))

for i in range(len(gpu_data)):
    if cpu_data[i] == 0.0 and gpu_data[i] == 0.0:
        diff = 0.0
    else:
        diff = abs(gpu_data[i] - cpu_data[i]) / max(abs(gpu_data[i]),
        abs(cpu_data[i]))

    if diff > threshold:
        print("CPU_data: %E, GPU_data: %E" % (cpu_data[i], gpu_data[i]))
        print("Difference = %E at line %d" % (diff, i))
        print("Input file at this position: %E" % input_data[i])
        print("")
