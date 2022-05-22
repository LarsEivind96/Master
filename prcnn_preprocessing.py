import os
import json
import laspy
import numpy as np
from matplotlib import path
from operator import itemgetter
import random
import shutil

'''
Preprocessing for PointRCNN
This script splits all point clouds into segments of ~n points, and saves as .bin files.
    It also converts the bboxes from json format to the same format as in KITTI dataset.
    It also creates json files with the test, train and val split.

Annotate dataset in Supervise.ly
Save the annotation json files, and set the json_bbox_path to where the files are stored. 
Set las_path to where the las files are stored.
set bin_path and bin_path_testing to where you want the bin files to be stored.
Set label_path_KITTI_format to where you want the txt files to be stored.
'''

prcnn_train_test_split_path = "C:/Users/LARLIE/School/Master/annotation3/prcnn/train_test_split/"
json_bbox_path = "C:/Users/LARLIE/Downloads/annotation_2/ds0/ann/"
new_json_bbox_path = "C:/Users/LARLIE/Downloads/annotation_2/ds0/ann_new/"
# prcnn_bin_path = "/home/alfredla/Documents/PointRCNN/data/KITTI/object/training/velodyne/"
# prcnn_bin_path_testing = "/home/alfredla/Documents/PointRCNN/data/KITTI/object/testing/velodyne/"
las_path = "C:/Users/LARLIE/School/Master/annotation2/thinned_las/"
bin_path = "C:/Users/LARLIE/School/Master/annotation2/prcnn/bin/"
bin_path_testing = "C:/Users/LARLIE/School/Master/annotation2/prcnn/bin_test/"
label_path_KITTI_format = "C:/Users/LARLIE/School/Master/annotation2/prcnn/label_txt/"


def split_las_files_to_bin():
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)
    for filename in os.listdir(las_path):
        print(filename)
        with laspy.open(las_path + filename) as fh:
            las = fh.read()
            x = las.x - fh.header.offsets[0]
            y = las.y - fh.header.offsets[1]
            z = las.z - fh.header.offsets[2]
            
            out_arr = np.transpose([x, y, z, las.intensity, las.gps_time])
            out_arr = sorted(out_arr, key=itemgetter(4))
            out_arr = np.asarray(out_arr)
            
            # Split points
            n_points = len(out_arr)
            divisor = int(np.ceil(n_points / 100000))
            if divisor == 0: divisor = 1
            new_n_points = int(np.floor(n_points / divisor))
            # print("divisor", divisor)
            # print(len(out_arr))
            for i in range(divisor):
                new_arr = out_arr[i * new_n_points : (i+1) * new_n_points]
                if i == (divisor - 1):
                    new_arr = out_arr[i * new_n_points : n_points]
                # print(i, len(new_arr))
                save_path = filename.split(".las")[0] + "_" + str(i) + ".bin"
                new_arr[:, 0:4].astype(np.float32).tofile(bin_path + save_path)


def get_bounding_box_coords(bounding_box):
    alfa = bounding_box['rotation']['z']
    p = bounding_box['position']
    d = bounding_box['dimensions']
    alfa = -alfa

    min_x = p['x'] - d['x']/2
    max_x = p['x'] + d['x']/2
    min_y = p['y'] - d['y']/2
    max_y = p['y'] + d['y']/2
    min_z = p['z'] - d['z']/2
    max_z = p['z'] + d['z']/2

    # Find xy coordinates of the rotated bounding box
    box_coords = np.asarray([[min_x, min_y, 1], [min_x, max_y, 1], [
                            max_x, max_y, 1], [max_x, min_y, 1]])
    rotation = np.asarray(
        [[np.cos(alfa), -np.sin(alfa), 0], [np.sin(alfa), np.cos(alfa), 0], [0, 0, 1]])
    translation = np.asarray([[1, 0, 0], [0, 1, 0], [p['x'], p['y'], 1]])
    translation_2 = np.asarray([[1, 0, 0], [0, 1, 0], [-p['x'], -p['y'], 1]])
    rotated_box_coords = box_coords.dot(
        translation_2).dot(rotation).dot(translation)

    # print(box_coords)
    # print(rotated_box_coords)

    return rotated_box_coords, min_z, max_z


def save_num_points_in_bbox_to_json():
    for filename in os.listdir(json_bbox_path):
        print(filename)
        annot_file = open(json_bbox_path + filename)
        annot_data = json.load(annot_file)
        las_name = filename.split("_2022")[0] + ".las"
        with laspy.open(las_path + las_name) as fh:
            las = fh.read()
            x = las.x - fh.header.offsets[0]
            y = las.y - fh.header.offsets[1]
            z = las.z - fh.header.offsets[2]
            out_arr = np.asarray([x, y, z]).T
            print(out_arr)

        # Loop through all figures, and fetch number of points within bounding box
        for figure in annot_data['figures']:
            rotated_box_coords, min_z, max_z = get_bounding_box_coords(figure['geometry'])

            # Find all points within the xy path of the bounding box
            rect = path.Path(rotated_box_coords[:, 0:2])
            contains_points = rect.contains_points(out_arr[:, 0:2])
            points_inside_box = out_arr[contains_points]

            # Check if points are within z boundaries
            bound_z = np.logical_and(
                points_inside_box[:, 2] > min_z, points_inside_box[:, 2] < max_z)
            points_inside_box = points_inside_box[bound_z]
            num_points = len(points_inside_box)
            print(len(points_inside_box))
            figure['num_points'] = num_points
        
        # Write into json file
        if not os.path.exists(new_json_bbox_path):
            os.makedirs(new_json_bbox_path)
        with open(new_json_bbox_path + filename, 'w') as new_json:
            json.dump(annot_data, new_json, indent=2)


# Convert label data from json to KITII format
def json_to_label():
    # Loop through json files
    for filename in os.listdir(new_json_bbox_path):
        annot_file = open(new_json_bbox_path + filename)
        annot_data = json.load(annot_file)
    
        # Find point cloud files corresponding to the json file
        base_name = filename.split("_2022")[0]
        all_bin_files = os.listdir(bin_path)
        
        # Loop through all pc files with same name but different index (ending)
        for bin_name in all_bin_files:
            if bin_name.startswith(base_name):
                print(bin_name)
                # Fetch point array from binary file
                lidar_file = os.path.join(bin_path, bin_name)
                assert os.path.isfile(lidar_file)
                arr = np.fromfile(lidar_file, dtype=np.float32)
                print("np array of binary file", arr[0:8])
                arr = arr.reshape(-1, 4)
                print("np array of binary file", arr[0:3])
                # Loop through all figures
                i = 0

                obj_list = []
                for figure in annot_data['figures']:
                    i += 1
                    pos = figure['geometry']['position']
                    pos_x = pos['x']
                    pos_y = pos['y']
                    pos_z = pos['z']
                    dim = figure['geometry']['dimensions']
                    dim_x = dim['x']
                    dim_y = dim['y']
                    dim_z = dim['z']
                    rot = figure['geometry']['rotation']
                    rot_z = rot['z']
                    tot_num_points = figure['num_points']

                    # Find class code
                    obj_id = figure['objectKey']
                    obj = list(filter(lambda x: x['key'] == obj_id, annot_data['objects']))
                    class_title = obj[0]['classTitle']
                    tags = obj[0]['tags']
                    for tag in tags:
                        print(tag['name'])

                    # Calculate truncation.. Find points within bbox for this segment and divide by total number of points within same bbox
                    rotated_box_coords, min_z, max_z = get_bounding_box_coords(
                        figure['geometry'])
                    
                    # Find all points within the xy path of the bounding box
                    rect = path.Path(rotated_box_coords[:, 0:2])
                    points_inside_box = arr[rect.contains_points(arr[:, 0:2])]

                    # Check if points are within z boundaries
                    bound_z = np.logical_and(
                        points_inside_box[:, 2] > min_z, points_inside_box[:, 2] < max_z)
                    points_inside_box = points_inside_box[bound_z]

                    truncation = 1 - len(points_inside_box) / tot_num_points

                    # if tags inneholder missing_line/missing_part -> 1, contains bad_dump -> 2, ??? -> 3, else 0 (som betyr at den er good)
                    occlusion = 0
                    if 'bad_dump' in tags:
                        occlusion = 3
                    elif 'missing_part' in tags:
                        occlusion = 2
                    elif 'missing_line' in tags:
                        occlusion = 1

                    # Write attributes to txt file (the same setup as in KITTI label files)
                    curr_obj = [class_title, truncation, occlusion, 0, 0, 0,
                                0, 0, dim_z, dim_y, dim_x, pos_x, pos_y, pos_z, rot_z]
                    obj_to_str = ' '.join([str(elem) for elem in curr_obj])
                    if truncation < 1: 
                        obj_list.append(obj_to_str)
                    
                    # File name on txt should be the same as the binary file name
                
                if not os.path.exists(label_path_KITTI_format):
                    os.makedirs(label_path_KITTI_format)
                f = open(label_path_KITTI_format + bin_name.split(".bin")[0] + ".txt", "a")
                for obj in obj_list:
                    f.write(obj + "\n")
                f.close()
                obj_list = []
                    # 1 - Class names
                    # 1 - Truncation (punkter i bbox i segment / tot punkter i bbox)
                    # 1 - Occlusion (0, 1, 2, 3, beskriver hvor synlig humpen er, f.eks 0 = fin hump, 1 = missing_line/missing_part, 2 = bad_dump, 3 = unknown)
                    # 1 - Alpha (sett til 0)
                    # 4 - Bounding box (på bilder, sett til 0)
                    # 3 - 3D dimensions (størrelse på bbox)
                    # 3 - Location (koordinater til midtpunktet)
                    # 1 - Rotation y (egt z tror vi)
    return


# TODO: Create train_test_split with only bumps?
def copy_files_to_testing(test_names):
    if not os.path.exists(bin_path_testing):
        os.makedirs(bin_path_testing)
    for name in test_names:
        basename = name[0:-1]
        shutil.move(bin_path + basename + '.bin', bin_path_testing)


# TODO: Create train_test_split with bumps?
def prcnn_create_val_test_train_file_list():
    if not os.path.exists(prcnn_train_test_split_path):
        os.makedirs(prcnn_train_test_split_path)
    fn_length = len(os.listdir(bin_path))
    test_list = []
    val_list = []
    train_list = []
    train_val_list = []
    i = 0
    for fn in os.listdir(bin_path):

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

if __name__ == '__main__':
    print("damane")
    split_las_files_to_bin()
    save_num_points_in_bbox_to_json()
    json_to_label()
    prcnn_create_val_test_train_file_list()
