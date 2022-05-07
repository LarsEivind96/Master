import os
import numpy as np

lidar_file = os.path.join("000016.bin")
assert os.path.exists(lidar_file)
arr = np.fromfile(lidar_file, dtype=np.float32)
print(arr)
arr = arr.reshape(-1, 4)

print(arr[0:30])

txt = "8__thin_ground_points.json_0.txt"
arr = np.loadtxt(txt)
print(arr[:, 0:4])
