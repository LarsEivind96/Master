import os
import json
import numpy as np
import shutil

pc_path = "/home/alfredla/Documents/Pointnet_Pointnet2_pytorch/data/00000000/"
train_test_split_path = "/home/alfredla/Documents/Pointnet_Pointnet2_pytorch/data/train_test_split_3/"
meta_path = "/home/alfredla/Documents/Pointnet_Pointnet2_pytorch/data/meta_local.txt"
prcnn_train_test_split_path = "/home/alfredla/Documents/PointRCNN/data/KITTI/ImageSets/"
prcnn_bin_path = "/home/alfredla/Documents/PointRCNN/data/KITTI/object/training/velodyne/"
prcnn_bin_path_testing = "/home/alfredla/Documents/PointRCNN/data/KITTI/object/testing/velodyne/"

def copy_files_to_testing(test_names):
    for name in test_names:
        basename = name[0:-1]
        shutil.move(prcnn_bin_path + basename + '.bin', prcnn_bin_path_testing)


def prcnn_create_val_test_train_file_list():
    if not os.path.exists(prcnn_train_test_split_path):
        os.makedirs(prcnn_train_test_split_path)
    fn_length = len(os.listdir(prcnn_bin_path))
    test_list = []
    val_list = []
    train_list = []
    train_val_list = []
    i = 0
    for fn in os.listdir(prcnn_bin_path):

        fn = fn[0:-4]
        # Create a list with 1/10 of files for test
        if i < fn_length / 10:
            test_list.append(fn + "\n")
        # Create a list with 1/10 of files for validation
        elif i < 2 * fn_length / 10:
            val_list.append(fn + "\n")
            train_val_list.append(fn + "\n")
        # Create a list with 8/10 of files for train
        else:
            train_list.append(fn + "\n")
            train_val_list.append(fn + "\n")
        i += 1

    # Create json files of each list.
    test_name = "test.txt"
    val_name = "val.txt"
    train_name = "train.txt"
    trainval_name = "trainval.txt"

    with open(prcnn_train_test_split_path + test_name, 'w') as f:
        f.writelines(test_list)
    with open(prcnn_train_test_split_path + val_name, 'w') as f:
        f.writelines(val_list)
    with open(prcnn_train_test_split_path + train_name, 'w') as f:
        f.writelines(train_list)
    with open(prcnn_train_test_split_path + trainval_name, 'w') as f:
        f.writelines(train_val_list)
    
    copy_files_to_testing(test_list)
    


def pnpp_create_val_test_train_file_list(reg_list):
    fns_bump = read_fns_with_bump()
    fn_length = len(os.listdir(pc_path))
    fn_length = len(fns_bump)
    fn_length = len(reg_list)
    # Filenames should be split into an 8:1:1 relation
    test_length = np.floor(fn_length / 10)
    val_length = np.floor(fn_length / 10)
    train_length = fn_length - test_length - val_length

    test_list = []
    val_list = []
    train_list = []

    i = 0
    for filename in reg_list: # os.listdir(pc_path):
        filename = filename[0:-4]
        # Create a list with 1/10 of files for test
        if i < test_length:
            test_list.append(filename)
        # Create a list with 1/10 of files for validation
        elif i < test_length + val_length:
            val_list.append(filename)
        # Create a list with 8/10 of files for train
        else:
            train_list.append(filename)
        i += 1
    
    i = 0
    for fn in train_list:
        if fn + '.txt' in fns_bump:
            i += 1
    print("Number of bump files in train_list:", i)

    i = 0
    for fn in test_list:
        if fn + '.txt' in fns_bump:
            i += 1
    print("Number of bump files in test_list:", i)

    i = 0
    for fn in val_list:
        if fn + '.txt' in fns_bump:
            i += 1
    print("Number of bump files in val_list:", i)

    # Create json files of each list.
    test_name = "shuffled_test_file_list.json"
    val_name = "shuffled_val_file_list.json"
    train_name = "shuffled_train_file_list.json"

    
    if not os.path.exists(train_test_split_path):
        os.makedirs(train_test_split_path)

    with open(train_test_split_path + test_name, 'w') as f:
        json.dump(test_list, f)
    with open(train_test_split_path + val_name, 'w') as f:
        json.dump(val_list, f)
    with open(train_test_split_path + train_name, 'w') as f:
        json.dump(train_list, f)


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

def find_bumpy_roads():
    fns = read_fns_with_bump()
    f_list_reg = []
    f_list_bus = []
    path = '/media/alfredla/F662-8A47/annotation1/txt_local_files/'
    for file in fns:
        print(file)
        with open(path + file, 'r') as f:
            data = f.read().split('\n')
            reg_count = 0
            bus_count = 0
            line_count = 0
            for line in data:
                if len(line) > 10:
                    class_val = line.split()[4]
                    line_count += 1
                    if class_val == '1.000000000000000000e+00':
                        reg_count += 1
                    elif class_val == '2.000000000000000000e+00':
                        bus_count += 1
            print("reg: " + str(reg_count / line_count))
            print("bus: " + str(bus_count / line_count))
            if (reg_count / line_count) > 0.15:
                # fn = path.split('txt_local_files/')[1]
                f_list_reg.append(file)
                
            elif (bus_count / line_count) > 0.05:
                # fn = path.split('txt_local_files/')[1]
                f_list_bus.append(file)
        f.close()
    print(f_list_reg)
    print(f_list_bus)
    with open('/media/alfredla/F662-8A47/annotation1/txt_bump_files/reg_bump_files.txt', 'w') as f_reg:
        for file in f_list_reg:
            f_reg.write(file + '\n')
    f_reg.close()
    with open('/media/alfredla/F662-8A47/annotation1/txt_bump_files/bus_bump_files.txt', 'w') as f_bus:
        for file in f_list_bus:
            f_bus.write(file + '\n')
    f_bus.close()

    return f_list_reg, f_list_bus

# read_fns_with_bump()
prcnn_create_val_test_train_file_list()
# f_list_reg, f_list_bus = find_bumpy_roads()
# pnpp_create_val_test_train_file_list(f_list_reg)
