import os
import numpy as np
from matplotlib import pyplot as plt
from itertools import product, combinations
import json

path = "results/"
json_path = "C:/Users/LARLIE/School/Master/all_json_annot/"

class_title_to_num = {"road": 0, "reg_dump": 1, "bus_dump": 2}

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
        
        bbox_list.append([rotated_box_coords, p, min_z, max_z, class_num])
    return bbox_list

i = 0
for fn in os.listdir(path):
    
    i += 1
    if i < 5 or i > 10:
        continue
    print(fn)
    file = np.loadtxt(path + fn)
    coords_and_label = file[:, 0:4]
    
    
    reg_flag = coords_and_label[:, 3] == 1
    bus_flag = coords_and_label[:, 3] == 2
    road_flag = coords_and_label[:, 3] == 0
    
    road_coord = coords_and_label[road_flag]
    reg_coord = coords_and_label[reg_flag]
    bus_coord = coords_and_label[bus_flag]
        
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(road_coord[:, 0], road_coord[:, 1], road_coord[:, 2], c="blue")
    ax.scatter(reg_coord[:, 0], reg_coord[:, 1], reg_coord[:, 2], c="green")
    ax.scatter(bus_coord[:, 0], bus_coord[:, 1], bus_coord[:, 2], c="red")
    
    max_x = max(coords_and_label[:, 0]) + 10
    min_x = min(coords_and_label[:, 0]) - 10
    max_y = max(coords_and_label[:, 1]) + 10
    min_y = min(coords_and_label[:, 1]) - 10
    print("Max-min x-y: {} {} {} {}".format(max_x, min_x, max_y, min_y))
    
    start_name = fn.split("points_")[0]
    for json_name in os.listdir(json_path):
        if json_name.startswith(start_name):
            print(json_name)
            f = open(json_path + json_name)
            data = json.load(f)
            bbox_list = fetch_bbox_coords(data)
            for bbox in bbox_list:
                rotated_box_coords = bbox[0]
                center = bbox[1]
                if center['x'] > min_x and center['x'] < max_x and center['y'] > min_y and center['y'] < max_y:
                    ax.plot(rotated_box_coords[:, 0], rotated_box_coords[:, 1])
                    print("damane")
                print("\n")
                for coord in rotated_box_coords:
                    if coord[0] > min_x and coord[0] < max_x and coord[1] > min_y and coord[1] < max_y:
                        ax.plot(rotated_box_coords[:, 0], rotated_box_coords[:, 1])
                    # print(coord)
        # ax.plot(rotated_box_coords[:, 0], rotated_box_coords[:, 1])
    
    '''r = [-1, 1]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="green")
        # ax.set_title("Cube")'''
    ax.set_axis_off()
    plt.show()
    # break

