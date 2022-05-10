import os
import laspy
import json
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import path
import time
import struct

json_folder_path = "C:/Users/LARLIE/Downloads/annotation_1 (6)/annotation_1/ds0/ann/"
bin_path = "C:/Users/LARLIE/Downloads/data_object_velodyne/training/velodyne/000000.bin"

bounding_box_test = {"position": {
                    "x": 90.78147767695194,
                    "y": -1782.6824804277383,
                    "z": 4.967639028873762
                },
                "rotation": {
                    "x": 0,
                    "y": 0,
                    "z": -2.0263469467579647
                },
                "dimensions": {
                    "x": 1.6869789820045218,
                    "y": 1.6307562062820506,
                    "z": 0.23338076799362972
                }}

def get_bounding_box_coords(bounding_box):
    alfa = bounding_box['rotation']['z']
    p = bounding_box['position']
    d = bounding_box['dimensions']
    
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
    
    return rotated_box_coords, min_z, max_z

def read_point_cloud():
    start = time.time()
    pcd = o3d.io.read_point_cloud("C:\\Users\\LARLIE\\School\\Master\\las_to_pcd_1\\10__thin_ground_points_2022-03-21_16h49_24_450.pcd")
    out_arr = np.asarray(pcd.points)
    
    rotated_box_coords, min_z, max_z = get_bounding_box_coords(bounding_box_test)
    
    # Find all points within the xy path of the bounding box
    rect = path.Path(rotated_box_coords[:, 0:2])
    contains_points = rect.contains_points(out_arr[:, 0:2])
    points_inside_box = out_arr[contains_points]
    
    # Check if points are within z boundaries
    bound_z = np.logical_and(points_inside_box[:, 2] > min_z, points_inside_box[:, 2] < max_z)
    points_inside_box = points_inside_box[bound_z]
    
    # print(points_inside_box)
    end = time.time()
    print("Time : ", end - start)
    
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points_inside_box[:, 0], points_inside_box[:, 1], points_inside_box[:, 2])
    ax.set_axis_off()
    plt.show()
    return points_inside_box


def print_annotation_statistics():
    num_figures = 0
    for filename in os.listdir(json_folder_path):
        annot_file = open(json_folder_path + filename)
        annot_data = json.load(annot_file)
        # x and y rotation should be 0
        check_rotation_x_y(annot_data, filename)
        # Fetch 7dof and bump type
        bounding_boxes = extract_boxes(annot_data, filename)
        num_figures += len(annot_data['figures'])
    print("Total number of annotated bumps", num_figures)


def check_rotation_x_y(data, filename):
    for figure in data['figures']:
        rot = figure['geometry']['rotation']
        rot_x = rot['x']
        rot_y = rot['y']
        if rot_x != 0 or rot_y != 0:
            print("Rotation x or y is not 0 ", filename)


def extract_boxes(file_data, filename):
    boxes = []
    i = 0
    for figure in file_data['figures']:
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
        rot_x = rot['x']
        rot_y = rot['y']
        rot_z = rot['z']
        
        # Find class code
        obj_id = figure['objectKey']
        obj = list(filter(lambda x: x['key'] == obj_id, file_data['objects']))
        class_title = obj[0]['classTitle']
        tags = obj[0]['tags']
        for tag in tags:
            print(tag['name'])
        
        dof = [class_title, pos_x, pos_y, pos_z, dim_x, dim_y, dim_z, rot_x, rot_y, rot_z]
        boxes.append(dof)
    print(i, filename)
    return boxes

def translate_box_position(file_data, x, y, z):
    for figure in file_data['figures']:
        pos = figure['geometry']['position']
        print(pos['x'])
        pos['x'] = pos['x'] + x
        pos['y'] = pos['y'] + y
        pos['z'] = pos['z'] + z
        print(pos['x'])
    with open('37new.json', 'w') as new_json:
        json.dump(file_data, new_json, indent=2)


# translate_box_position(data, -16, -60.002, -143.04)
#read_point_cloud()

#print_annotation_statistics()













def bin_to_pcd():
    """file = open(bin_path, "rb")
    print(list(file.read(50)))
    file.close"""
    # Load binary point cloud
    bin_pcd = np.fromfile("10__thin_ground_points_1.bin", dtype=np.float32)

    # Reshape and drop reflection values
    points = bin_pcd.reshape((-1, 4))[:, 0:4]
    print(points[0:10])
    
    print("\n\n")
    with open ("10__thin_ground_points_1.bin", "rb") as f:
        list_pcd = []
        size_float = 4
        byte = f.read(size_float*4)
        while byte:
            x,y,z,intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float*4)
    np_pcd = np.asarray(list_pcd)
    print(np_pcd[0:10])
    pcd = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector
    pcd.points = v3d(np_pcd)
    o3d.io.write_point_cloud("copy_of_fragment.pcd", pcd)
    return

bin_to_pcd()




"""

def damane    
    box_coords = np.asarray([[min_x, min_y, 1], [min_x, max_y, 1], [max_x, max_y, 1], [max_x, min_y, 1]])
    rotation = np.asarray([[np.cos(alfa), -np.sin(alfa), 0], [np.sin(alfa), np.cos(alfa), 0], [0, 0, 1]])
    translation = np.asarray([[1, 0, 0], [0, 1, 0], [p['x'], p['y'], 1]])
    translation_2 = np.asarray([[1, 0, 0], [0, 1, 0], [-p['x'], -p['y'], 1]])
    rotated_box_coords = box_coords.dot(translation_2).dot(rotation).dot(translation)
    print(box_coords)
    print(rotated_box_coords[:, 0:2])
    points_inside_box_1 = test_points(out_arr, rotated_box_coords[:, 0:2], min_z, max_z)

    inside_box = bounding_box(out_arr, min_x, max_x, min_y, max_y, min_z, max_z)
    points_inside_box = out_arr[inside_box]
    # print("points_inside_box : ", points_inside_box)
    
    
    
    end = time.time()
    print("Time : ", end - start)
    
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points_inside_box_1[:, 0], points_inside_box_1[:, 1], points_inside_box_1[:, 2])
    ax.set_axis_off()
    plt.show()

def is_on_right_side(x, y, xy0, xy1):
    x0, y0 = xy0
    x1, y1 = xy1
    a = float(y1 - y0)
    b = float(x0 - x1)
    c = - a*x0 - b*y0
    return a*x + b*y + c >= 0

def test_point(x, y, vertices):
    num_vert = len(vertices)
    is_right = [is_on_right_side(x, y, vertices[i], vertices[(i + 1) % num_vert]) for i in range(num_vert)]
    all_left = not any(is_right)
    all_right = all(is_right)
    return all_left or all_right

def test_points(points, vertices, min_z, max_z):
    inside = []
    for point in points:
        if test_point(point[0], point[1], vertices) and point[2] < max_z and point[2] > min_z:
            inside.append([point[0], point[1], point[2]])
    print(inside)
    return np.asarray(inside)

def bounding_box(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf, min_z=-np.inf, max_z=np.inf):

    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
    bound_z = np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)

    bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    return bb_filter
"""