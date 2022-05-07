import os
import json
import laspy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import path
import open3d as o3d

json_folder_path = "C:/Users/LARLIE/Downloads/annotation_1 (6)/annotation_1/ds0/ann/"
new_json_folder_path = "C:/Users/LARLIE/Downloads/annotation_1 (6)/annotation_1/ds0/new_ann/"
las_path = "C:/Users/LARLIE/School/Master/thinned_0_500/"
pcd_path = "C:/Users/LARLIE/School/Master/las_to_pcd_1/"

def convert_bbox_coords():
    num_figures = 0
    # Loop through all bbox json files
    i = 0
    for filename in os.listdir(json_folder_path):
        i += 1
        '''if i <= 2:
            continue
        if not filename.startswith("109__thin"):
            continue'''
        print(filename)
        # Fetch bounding_box
        annot_file = open(json_folder_path + filename)
        annot_data = json.load(annot_file)
        
        # Find the corresponding las file and fetch global shift values
        las_name = filename.split("_2022")[0] + ".las"
        json_name = filename.split("_2022")[0] + ".json"
        pcd_name = filename.split(".json")[0]
        with laspy.open(las_path + las_name) as fh:
            shift = fh.header.offsets
            translate_box_position(json_name, annot_data, shift[0], shift[1], shift[2])
        
        # Visualize the bumps in the las file
        # visualize_las_bumps(las_name, json_name, filename, pcd_name)
        # visualize_las_and_pcd_bumps(las_name, json_name, filename, pcd_name)


def translate_box_position(file_name, file_data, x, y, z):
    for figure in file_data['figures']:
        pos = figure['geometry']['position']
        pos['x'] = pos['x'] + x
        pos['y'] = pos['y'] + y
        pos['z'] = pos['z'] + z
    
    # Edit / create new json file with updated bbox coordinates
    if not os.path.exists(new_json_folder_path):
        os.makedirs(new_json_folder_path)
    with open(new_json_folder_path + file_name, 'w') as new_json:
        json.dump(file_data, new_json, indent=2)


def visualize_las_bumps(las_name, json_name, old_json_name, pcd_name):
    # Convert las x y z to numpy array
    with laspy.open(las_path + las_name) as fh:
        las = fh.read()
        # TODO: Save time and intensity variable here as well
        out_arr = np.asarray([las.x, las.y, las.z]).T
    
    
    # Find corner coordinates of bounding box
    annot_file = open(new_json_folder_path + json_name)
    annot_data = json.load(annot_file)
    print("Number of figures: ", len(annot_data['figures']))
    i = 0
    for figure in annot_data['figures']:
        i += 1
        # if i == 1: continue
        # if i > 2: return
        rotated_box_coords, min_z, max_z = get_bounding_box_coords(figure['geometry'])
        
        # Find all points within the xy path of the bounding box
        rect = path.Path(rotated_box_coords[:, 0:2])
        contains_points = rect.contains_points(out_arr[:, 0:2])
        points_inside_box = out_arr[contains_points]
        
        # Check if points are within z boundaries
        bound_z = np.logical_and(points_inside_box[:, 2] > min_z, points_inside_box[:, 2] < max_z)
        points_inside_box = points_inside_box[bound_z]
        
        # Visualize the bump
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points_inside_box[:, 0], points_inside_box[:, 1], points_inside_box[:, 2])
        ax.set_axis_off()
        plt.show()

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
    box_coords = np.asarray([[min_x, min_y, 1], [min_x, max_y, 1], [max_x, max_y, 1], [max_x, min_y, 1]])
    rotation = np.asarray([[np.cos(alfa), -np.sin(alfa), 0], [np.sin(alfa), np.cos(alfa), 0], [0, 0, 1]])
    translation = np.asarray([[1, 0, 0], [0, 1, 0], [p['x'], p['y'], 1]])
    translation_2 = np.asarray([[1, 0, 0], [0, 1, 0], [-p['x'], -p['y'], 1]])
    rotated_box_coords = box_coords.dot(translation_2).dot(rotation).dot(translation)
    
    print(box_coords)
    print(rotated_box_coords)
    
    return rotated_box_coords, min_z, max_z





def visualize_las_and_pcd_bumps(las_name, json_name, old_json_name, pcd_name):
    pcd = o3d.io.read_point_cloud(pcd_path + pcd_name)
    out_arr_pcd = np.asarray(pcd.points)
    
    # Convert las x y z to numpy array
    with laspy.open(las_path + las_name) as fh:
        las = fh.read()
        # TODO: Save time and intensity variable here as well
        out_arr = np.asarray([las.x, las.y, las.z]).T
    
    
    # Find corner coordinates of bounding box
    def find_bumps(json_filename, arr):
        annot_file = open(json_filename)
        annot_data = json.load(annot_file)
        print("Number of figures: ", len(annot_data['figures']))
        i = 0
        for figure in annot_data['figures']:
            i += 1
            rotated_box_coords, min_z, max_z = get_bounding_box_coords(figure['geometry'])
            
            # Find all points within the xy path of the bounding box
            rect = path.Path(rotated_box_coords[:, 0:2])
            contains_points = rect.contains_points(arr[:, 0:2])
            points_inside_box = arr[contains_points]
            
            # Check if points are within z boundaries
            bound_z = np.logical_and(points_inside_box[:, 2] > min_z, points_inside_box[:, 2] < max_z)
            points_inside_box = points_inside_box[bound_z]
            if i == 2: return points_inside_box
            
    las_points = find_bumps(new_json_folder_path + json_name, out_arr)
    pcd_points = find_bumps(json_folder_path + old_json_name, out_arr_pcd)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(las_points[:, 0], las_points[:, 1], las_points[:, 2])
    #ax.set_axis_off()
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(pcd_points[:, 0], pcd_points[:, 1], pcd_points[:, 2])
    #ax2.set_axis_off()
    plt.show()


convert_bbox_coords()
