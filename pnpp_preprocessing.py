import os
import json
import laspy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import path
import open3d as o3d
from operator import itemgetter
import random

'''
Preprocessing for Pointnet++
This script splits all point clouds into segments of ~n points, and specifies which class each point corresponds to (road, reg_dump, bus_dump), 
    by comparing the bounding boxes from annotation with the original point cloud (las) files. This, along with the point coordinates, is stored in a txt file.
    It also creates json files with the test, train and val split.

Annotate dataset in Supervise.ly
Save the annotation json files, and set the json_bbox_path to where the files are stored. 
Set las_path to where the las files are stored.
Set splittet_txt_path to where you want the txt files to be stored.
'''

las_path = "C:/Users/LARLIE/School/Master/ground_0_500/"
json_bbox_path = "C:/Users/LARLIE/Downloads/annotation_1 (6)/annotation_1/ds0/ann/"
splitted_txt_path = "C:/Users/LARLIE/School/Master/annotation2/txt_local_files/"
train_test_split_path = "C:/Users/LARLIE/School/Master/annotation2/train_test_split/"
meta_path = "C:/Users/LARLIE/School/Master/annotation2/meta.txt"

        
if not os.path.exists(splitted_txt_path):
    os.makedirs(splitted_txt_path)

if not os.path.exists(train_test_split_path):
    os.makedirs(train_test_split_path)
        
class_title_to_num = {"road": 0, "reg_dump": 1, "bus_dump": 2}
tag_title_to_num = {"normal": 0, "missing_part": 1, "bad_dump": 2, "missing_line": 3}

NUM_POINTS_PER_TXT_FILE = 16384

# Contains data over which files contain reg_dump or bus_dump
meta = []


def split_las_files_pnpp():
    i = 1
    # Loop through all json bbox files
    for filename in os.listdir(json_bbox_path):
        print(filename)
        annot_file = open(json_bbox_path + filename)
        annot_data = json.load(annot_file)
        bbox_list = fetch_bbox_coords(annot_data)
        if len(bbox_list) > 0:
            print("Different bump classes: ", np.asarray(bbox_list, dtype=object)[:, 3])
        
        las_filename = filename.split("_2022")[0] + ".las"
        with laspy.open(las_path + las_filename) as fh:
            las = fh.read()
            type = np.zeros(len(las.x))
            tag = np.zeros(len(las.x))
            
            x = las.x - fh.header.offsets[0]
            y = las.y - fh.header.offsets[1]
            z = las.z - fh.header.offsets[2]
            out_arr = np.transpose([x, y, z, las.intensity, type, tag, las.gps_time])
            out_arr = sorted(out_arr, key=itemgetter(6))
            out_arr = np.asarray(out_arr)
            
            # Loop through all bboxes and mark the points within
            print("Number of bounding boxes: ", len(bbox_list))
            for bbox in bbox_list:
                rotated_box_coords = bbox[0]
                min_z = bbox[1]
                max_z = bbox[2]
                class_num = bbox[3]
                tag_num = bbox[4]
                
                # Find all points within the xy path of the bounding box
                rect = path.Path(rotated_box_coords[:, 0:2])
                contains_points = rect.contains_points(out_arr[:, 0:2])
                
                for i in range(len(out_arr)):
                    if contains_points[i] and (out_arr[i][2] > min_z and out_arr[i][2] < max_z):
                        out_arr[i][4] = class_num
                        out_arr[i][5] = tag_num
                        # print(str(class_num) + " - point. Number: " + str(i))
            write_new_txt(las_filename.split('.las')[0], out_arr)
            
        with open(meta_path, 'w') as f:
            for item in meta:
                f.write("{}\t{}\n".format(item[0], item[1]))


def write_new_txt(base_name, arr):
    n_points = len(arr)
    divisor = int(np.ceil(n_points / NUM_POINTS_PER_TXT_FILE))
    print("divisor", divisor)
    if divisor == 0: divisor = 1
    new_n_points = int(np.floor(n_points / divisor))
    for i in range(divisor):
        new_arr = arr[i * new_n_points : (i+1) * new_n_points]
        txt_name = base_name + "_" + str(i) + ".txt"
        save_path = splitted_txt_path + txt_name
        print(save_path)
        if i == (divisor - 1):
            new_arr = arr[i * new_n_points : n_points]
        if 1 in new_arr[:, 4]:
            #print("Regular dump here: " + str(i))
            meta.append(["reg_dump", txt_name])
        if 2 in new_arr[:, 4]:
            #print("Bus dump here: " + str(i))
            meta.append(["bus_dump", txt_name])
        #if 1 in new_arr[:, 6] or 2 in new_arr[:, 6] or 3 in new_arr[:, 6]:
            #print("... A bad one")
        np.savetxt(save_path, new_arr)


# Find correct bounding box coordinates after rotation.
def fetch_bbox_coords(data):
    bbox_list = []
    for figure in data['figures']:
        geometry = figure['geometry']
        alfa = geometry['rotation']['z']
        p = geometry['position']
        d = geometry['dimensions']
        alfa = -alfa
        
        min_x = p['x'] - d['x']/2
        max_x = p['x'] + d['x']/2
        min_y = p['y'] - d['y']/2
        max_y = p['y'] + d['y']/2
        min_z = p['z'] - d['z']/2
        max_z = p['z'] + d['z']/2
        
        # Find xy coordinates of the rotated bounding box
        box_coords = np.asarray([[min_x, min_y, 1], [min_x, max_y, 1], [max_x, max_y, 1], [max_x, min_y, 1]])
        rotation = np.asarray([[np.cos(alfa), -np.sin(alfa), 0], [np.sin(alfa), np.cos(alfa), 0], [0, 0, 1]])
        translation = np.asarray([[1, 0, 0], [0, 1, 0], [p['x'], p['y'], 1]])
        translation_2 = np.asarray([[1, 0, 0], [0, 1, 0], [-p['x'], -p['y'], 1]])
        rotated_box_coords = box_coords.dot(translation_2).dot(rotation).dot(translation)
        
        # Find the box type of each figure
        obj_id = figure['objectKey']
        obj = list(filter(lambda x: x['key'] == obj_id, data['objects']))
        class_title = obj[0]['classTitle']
        class_num = class_title_to_num[class_title]
        
        tags = obj[0]['tags']
        tag_title = "normal"
        for tag in tags:
            tag_title = tag['name']
        
        tag_num = tag_title_to_num[tag_title]
        
        bbox_list.append([rotated_box_coords, min_z, max_z, class_num, tag_num])
        # bbox_list.append({"rotated_box_coords": rotated_box_coords, "min_z": min_z, "max_z": max_z, "class_num": class_num})
    return bbox_list


def pnpp_create_val_test_train_file_list():
    fns_bump = read_fns_with_bump()

    fn_length = len(fns_bump)

    # Filenames should be split into an 8:1:1 relation
    test_length = np.floor(fn_length / 10)
    val_length = np.floor(fn_length / 10)
    train_length = fn_length - test_length - val_length

    test_list = []
    val_list = []
    train_list = []

    i = 0
    for filename in fns_bump: # os.listdir(pc_path):
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

    # Create json files of each list.
    test_name = "shuffled_test_file_list.json"
    val_name = "shuffled_val_file_list.json"
    train_name = "shuffled_train_file_list.json"

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
            # Do not include the last "\n"
            if len(line) > 10:
                name = line.split('\t')[1]
                fns.append(name)
    random.shuffle(fns)
    return fns

if __name__ == '__main__':
    print("damane")
    split_las_files_pnpp()
    pnpp_create_val_test_train_file_list()