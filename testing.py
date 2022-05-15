import os
import numpy as np
# import laspy

las_path = "C:/Users/LARLIE/School/Master/thinned_0_500/"
default_bin = "000016.bin"
new_bin = "damane.bin"
meta_path = "/home/alfredla/Documents/Pointnet_Pointnet2_pytorch/data/meta_local.txt"

def read_fns_with_bump():
    fns = []
    with open(meta_path, 'r') as f:
        data = f.read().split('\n')
        for line in data:
            if len(line) > 10:
                path = line.split('\t')[1]
                fn = path.split('txt_local_files/')[1]
                fns.append(fn)
                print(fn)
    return fns

def count_point_cats():
    path = '/media/alfredla/F662-8A47/annotation1/txt_local_files/'
    fns = read_fns_with_bump()
    tot_reg_count = 0
    tot_bus_count = 0
    tot_road_count = 0
    tot_line_count = 0
    for file in fns: # os.listdir(path):
        print(file)
        with open(path + file, 'r') as f:
            data = f.read().split('\n')
            reg_count = 0
            bus_count = 0
            road_count = 0
            line_count = 0
            for line in data:
                if len(line) > 10:
                    class_val = line.split()[4]
                    line_count += 1
                    if class_val == '1.000000000000000000e+00':
                        reg_count += 1
                    elif class_val == '2.000000000000000000e+00':
                        bus_count += 1
                    else:
                        road_count += 1
            
            tot_reg_count += reg_count
            tot_bus_count += bus_count
            tot_road_count += road_count
            tot_line_count += line_count
            print("Relation reg_dump: " + str(reg_count / line_count))
            print("Relation bus_dump: " + str(bus_count / line_count))
            print("Relation road: " + str(road_count / line_count))
        f.close()
    print("\nTotal reg_dump count:", tot_reg_count)
    print("Total bus_dump count:", tot_bus_count)
    print("Total road count:", tot_road_count)
    print("Total line count:", tot_line_count)
    print("Relation reg_dump: " + str(tot_reg_count / tot_line_count))
    print("Relation bus_dump: " + str(tot_bus_count / tot_line_count))
    print("Relation road: " + str(tot_road_count / tot_line_count))
    print("Weight reg_dump:", str(1 - (tot_reg_count / tot_line_count)))
    print("Weight bus_dump:", str(1 - (tot_bus_count / tot_line_count)))
    print("Weight road:", str(1 - (tot_road_count / tot_line_count)))
    return

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

#write_bin_file()
#read_bin_file()
count_point_cats()

'''test = ['hei', 'hallo', 'damane']
with open("test.txt", 'w') as f:
    for word in test:
        f.write(word + "\n")'''
