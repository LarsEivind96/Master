import os
import numpy as np
import laspy

las_path = "C:/Users/LARLIE/School/Master/thinned_0_500/"
default_bin = "000016.bin"
new_bin = "damane.bin"

def read_bin_file():
    lidar_file = os.path.join(new_bin)
    assert os.path.exists(lidar_file)
    arr = np.fromfile(lidar_file, dtype=np.float32)
    print("np array of binary file", arr)
    arr = arr.reshape(-1, 4)
    print("np array of binary file", arr)

def write_bin_file():
    for filename in os.listdir(las_path):
        with laspy.open(las_path + filename) as fh:
            las = fh.read()
            x = las.x - fh.header.offsets[0]
            y = las.y - fh.header.offsets[1]
            z = las.z - fh.header.offsets[2]
            intensity = np.zeros(len(x))
            out_arr = np.asarray([x, y, z, intensity]).T
            print(out_arr)
            print(out_arr.reshape(-1, 1))
            out_arr.astype(np.float32).tofile(new_bin)
        return
    
    # save augment result to file
    '''pts_info = np.concatenate((pts_rect, pts_intensity.reshape(-1, 1)), axis=1)
    bin_file = os.path.join(data_save_dir, '%06d.bin' % (base_id + sample_id))
    pts_info.astype(np.float32).tofile(bin_file)
'''

write_bin_file()
read_bin_file()
