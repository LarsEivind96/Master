# Split las files into files with a fixed number of points (e.g. 8196)

import os
import json
import laspy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import path
import open3d as o3d
from operator import itemgetter

las_path = "C:/Users/LARLIE/School/Master/las_to_pcd_1/"
las_path = "C:/Users/LARLIE/School/Master/ground_0_500/"
splitted_txt_path = "C:/Users/LARLIE/School/Master/annotation1/txt_files/"
splitted_txt_local_path = "C:/Users/LARLIE/School/Master/annotation1/txt_local_files/"
json_bbox_path = "C:/Users/LARLIE/Downloads/annotation_1 (6)/annotation_1/ds0/new_ann/"
json_bbox_local_coords_path = "C:/Users/LARLIE/Downloads/annotation_1 (6)/annotation_1/ds0/ann/"
bin_path = "C:/Users/LARLIE/School/Master/bin_thinned_0_500/"

class_title_to_num = {"road": 0, "reg_dump": 1, "bus_dump": 2}
tag_title_to_num = {"normal": 0, "missing_part": 1, "bad_dump": 2, "missing_line": 3}

meta = []


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
            print("divisor", divisor)
            print(len(out_arr))
            for i in range(divisor):
                new_arr = out_arr[i * new_n_points : (i+1) * new_n_points]
                if i == (divisor - 1):
                    new_arr = out_arr[i * new_n_points : n_points]
                print(i, len(new_arr))
                save_path = filename.split(".las")[0] + "_" + str(i) + ".bin"
                new_arr[:, 0:4].astype(np.float32).tofile(bin_path + save_path)


def split_las_files_pnpp():
    i = 1
    
    # Loop through all json bbox files
    for filename in os.listdir(json_bbox_local_coords_path):
        print(filename)
        '''i += 1
        if i == 2: continue
        if i > 3: return'''
        annot_file = open(json_bbox_local_coords_path + filename)
        annot_data = json.load(annot_file)
        # Fetch list of bboxes
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
            write_new_txt(splitted_txt_local_path + las_filename.split('.las')[0], out_arr)
            


def write_new_txt(filename, arr):
    n_points = len(arr)
    divisor = int(np.ceil(n_points / 16384))
    print("divisor", divisor)
    if divisor == 0: divisor = 1
    new_n_points = int(np.floor(n_points / divisor))
    for i in range(divisor):
        new_arr = arr[i * new_n_points : (i+1) * new_n_points]
        save_path = filename + "_" + str(i) + ".txt"
        print(save_path)
        if i == (divisor - 1):
            new_arr = arr[i * new_n_points : n_points]
        if 1 in new_arr[:, 4]:
            #print("Regular dump here: " + str(i))
            meta.append(["reg_dump", save_path])
        if 2 in new_arr[:, 4]:
            #print("Bus dump here: " + str(i))
            meta.append(["bus_dump", save_path])
        #if 1 in new_arr[:, 6] or 2 in new_arr[:, 6] or 3 in new_arr[:, 6]:
            #print("... A bad one")
        
        np.savetxt(save_path, new_arr)


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
    


def write_new_las(las_file, arr):
    
    header = laspy.LasHeader(point_format=las_file.header.point_format, version=las_file.header.version)
    new_las = laspy.LasData(header)
    new_arr = arr[0:16384]
    
    new_las.x = new_arr[:, 0]
    new_las.y = new_arr[:, 1]
    new_las.z = new_arr[:, 2]
    new_las.gps_time = new_arr[:, 3]
    new_las.intensity = new_arr[:, 4]
    
    # new_las.write(splitted_las_path + "new_file.las")
    
    
# split_las_files_to_bin()
split_las_files_pnpp()
meta_path = "C:/Users/LARLIE/School/Master/annotation1/meta_local.txt"
# meta = [["1", "2"], ["3", "4"]]

with open(meta_path, 'w') as f:
    for item in meta:
        f.write("{}\t{}\n".format(item[0], item[1]))
