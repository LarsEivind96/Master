import os
import json
import laspy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import path
# import open3d as o3d

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
            translate_box_position(json_name, annot_data,
                                   shift[0], shift[1], shift[2])

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


def add_total_points(file_name, file_data):
    for figure in file_data['figures']:
        figure['total_points'] = 0

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
        rotated_box_coords, min_z, max_z = get_bounding_box_coords(
            figure['geometry'])

        # Find all points within the xy path of the bounding box
        rect = path.Path(rotated_box_coords[:, 0:2])
        contains_points = rect.contains_points(out_arr[:, 0:2])
        points_inside_box = out_arr[contains_points]

        # Check if points are within z boundaries
        bound_z = np.logical_and(
            points_inside_box[:, 2] > min_z, points_inside_box[:, 2] < max_z)
        points_inside_box = points_inside_box[bound_z]

        # Visualize the bump
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            points_inside_box[:, 0], points_inside_box[:, 1], points_inside_box[:, 2])
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
    box_coords = np.asarray([[min_x, min_y, 1], [min_x, max_y, 1], [
                            max_x, max_y, 1], [max_x, min_y, 1]])
    rotation = np.asarray(
        [[np.cos(alfa), -np.sin(alfa), 0], [np.sin(alfa), np.cos(alfa), 0], [0, 0, 1]])
    translation = np.asarray([[1, 0, 0], [0, 1, 0], [p['x'], p['y'], 1]])
    translation_2 = np.asarray([[1, 0, 0], [0, 1, 0], [-p['x'], -p['y'], 1]])
    rotated_box_coords = box_coords.dot(
        translation_2).dot(rotation).dot(translation)

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
            rotated_box_coords, min_z, max_z = get_bounding_box_coords(
                figure['geometry'])

            # Find all points within the xy path of the bounding box
            rect = path.Path(rotated_box_coords[:, 0:2])
            contains_points = rect.contains_points(arr[:, 0:2])
            points_inside_box = arr[contains_points]

            # Check if points are within z boundaries
            bound_z = np.logical_and(
                points_inside_box[:, 2] > min_z, points_inside_box[:, 2] < max_z)
            points_inside_box = points_inside_box[bound_z]

    las_points = find_bumps(new_json_folder_path + json_name, out_arr)
    pcd_points = find_bumps(json_folder_path + old_json_name, out_arr_pcd)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(las_points[:, 0], las_points[:, 1], las_points[:, 2])
    # ax.set_axis_off()
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(pcd_points[:, 0], pcd_points[:, 1], pcd_points[:, 2])
    # ax2.set_axis_off()
    plt.show()


def json_to_label(json_filename, las_name):
    # TODO: fikse så koordinatene som blir brukt er de lokale

    annot_file = open(json_filename)
    annot_data = json.load(annot_file)
    # open file
    f = open("test_file1.txt", "a")

    boxes = []
    i = 0

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
        rot_x = rot['x']
        rot_y = rot['y']
        rot_z = rot['z']

        # Find class code
        obj_id = figure['objectKey']
        obj = list(filter(lambda x: x['key'] == obj_id, annot_data['objects']))
        class_title = obj[0]['classTitle']
        tags = obj[0]['tags']
        for tag in tags:
            print(tag['name'])

        with laspy.open(las_path + las_name) as fh:
            las = fh.read()
            # TODO: Save time and intensity variable here as well
            arr = np.asarray([las.x, las.y, las.z]).T

        rotated_box_coords, min_z, max_z = get_bounding_box_coords(
            figure['geometry'])
        # Find all points within the xy path of the bounding box
        rect = path.Path(rotated_box_coords[:, 0:2])
        contains_points = rect.contains_points(arr[:, 0:2])
        seg_point_num = arr[contains_points]

        # Check if points are within z boundaries
        bound_z = np.logical_and(
            points_inside_box[:, 2] > min_z, points_inside_box[:, 2] < max_z)
        points_inside_box = points_inside_box[bound_z]

        tot_points_inside_box = figure['total_points']

        #seg_point_num = 1
        tot_point_num = 1
        # TODO: se på denne etter vi vet om vi skal splitte opp filene eller ikke
        # TODO: fikse så lokale koordinater brukes og så transformerer vi til globale til slutt

        truncation = 1 - points_inside_box / tot_points_inside_box
        truncation = 0

        # if tags inneholder missing_line/missing_part -> 1, contains bad_dump -> 2, ??? -> 3, else 0 (som betyr at den er good)
        occlusion = 0
        something = 0
        if 'missing_part' in tags or 'missing_line' in tags:
            occlusion = 1
        elif 'bad_dump' in tags:
            occlusion = 2
        elif something:
            occlusion = 3

        # TODO: gjøre pos_x, pos_y, pos_z om til lokale koordinater
        curr_line = [class_title, truncation, occlusion, 0, 0, 0,
                     0, 0, dim_z, dim_y, dim_x, pos_x, pos_y, pos_z, rot_z]
        listToStr = ' '.join([str(elem) for elem in curr_line])
        f.write(listToStr + '\n')
    f.close()
    # 1 - Class names
    # 1 - Truncation (punkter i bbox i segment / tot punkter i bbox)
    # 1 - Occlusion (0, 1, 2, 3, beskriver hvor synlig humpen er, f.eks 0 = fin hump, 1 = missing_line/missing_part, 2 = bad_dump, 3 = unknown)
    # 1 - Alpha (sett til 0)
    # 4 - Bounding box (på bilder, sett til 0)
    # 3 - 3D dimensions (størrelse på bbox)
    # 3 - Location (koordinater til midtpunktet)
    # 1 - Rotation y (egt z tror vi)
    return


s = ['a', 'b', 'c']
listToStr = ' '.join([str(elem) for elem in s])

f = open("demofile2.txt", "a")
lines = []
for i in range(3):
    lines.append(listToStr)
print(lines)
f.writelines(lines)
f.close()

# open and read the file after the appending:
# f = open("demofile2.txt", "r")
# print(f.read())


'''# convert_bbox_coords()
annot_file = open('./annotation_files/10__thin_ground_points.json')
annot_data = json.load(annot_file)

print(len(annot_data['figures']))


ids = []
for i in range(len(annot_data['figures'])):
    if not ids.contains(annot_data['figures'][i]['key']):
        ids.append(annot_data['figures'][i]['key'])
print(len(ids))'''

# print(annot_data['objects'][0])
# {'key': '31b541188dc840dd82eb2b0e805f6091',
# 'classTitle': 'bus_dump',
# 'tags': [],
# 'labelerLogin': 'Larseivind',
# 'updatedAt': '2022-03-21T16:37:17.243Z',
# 'createdAt': '2022-03-21T16:37:17.243Z'}
# print(annot_data['figures'][0])
# {'key': 'dffceafdbd1d4ccfb659376256c0542a',
# 'objectKey': 'da0a5e8475b84d0cbb80cbcb39bae314',
# 'geometryType': 'cuboid_3d',
# 'geometry': {'position': {'x': 571182.583018077, 'y': 7029925.959727647, 'z': 122.47938413290164}, 'rotation': {'x': 0, 'y': 0, 'z': 1.7464851045250933}, 'dimensions': {'x': 5.860604543730506, 'y': 4.316090054277171, 'z': 0.3839025757244377}},
# 'labelerLogin': 'Larseivind',
# 'updatedAt': '2022-04-12T11:32:04.682Z',
# 'createdAt': '2022-04-12T11:30:29.176Z'}
