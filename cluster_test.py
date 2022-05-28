from collections import Counter
from fileinput import filename
import numpy as np
from sklearn.cluster import DBSCAN
import json


# opens the current file and moves points to the correct list, class_1 = reg_dump and class_2 = bus_dump, normal points are "discarded" here


def check_bumps(filename):
    class_1 = []
    class_2 = []
    with open('./unnorm_4096/' + filename) as file:
        counter = 0
        for line in file:
            if float(line.split()[3]) == 1:
                class_1.append([float(line.split()[0]), float(
                    line.split()[1]), float(line.split()[2])])
            elif float(line.split()[3]) == 2:
                class_2.append([float(line.split()[0]), float(
                    line.split()[1]), float(line.split()[2])])
    data_1 = np.array(class_1)
    data_2 = np.array(class_2)

    # print(data_1)
    # print(data_2)

    # creates clusters from the points
    if len(data_1) > 0:
        db_1 = DBSCAN(eps=0.5, min_samples=10).fit(data_1)
        labels_1 = db_1.labels_
    else:
        labels_1 = []
    if len(data_2) > 0:
        db_2 = DBSCAN(eps=0.5, min_samples=10).fit(data_2)
        labels_2 = db_2.labels_
    else:
        labels_2 = []
    print(labels_1)
    # print(labels_2)

    # counts the number of points in each cluster (or amount of noise points)
    count_1 = Counter(labels_1)
    count_2 = Counter(labels_2)

    print(count_1)
    print(count_2)

    # find the amount of unique clusters in the current file and if there are noise points in the file, subtract 1 to not count it as a cluster
    if np.isin(-1, labels_1):
        num_of_clusters_1 = len(set(labels_1)) - 1
    else:
        num_of_clusters_1 = len(set(labels_1))

    if np.isin(-1, labels_2):
        num_of_clusters_2 = len(set(labels_2)) - 1
    else:
        num_of_clusters_2 = len(set(labels_2))
    # creates empty sublists for every cluster
    clusters_1 = [[] for i in range(num_of_clusters_1)]
    clusters_2 = [[] for i in range(num_of_clusters_2)]

    return

    # noise counters
    noise_1 = 0
    noise_2 = 0

    # loops through all points and checks which clusters they belong to and adds them to the correct sublist
    for i in range(len(labels_1)):
        if labels_1[i] == -1:  # -1 means the point is labelled as noise
            noise_1 += 1
            continue
        else:
            # adds the points to the corresponding cluster "id"
            clusters_1[labels_1[i]].append(class_1[i])
    # repeated for bus dumps
    for i in range(len(labels_2)):
        if labels_2[i] == -1:
            noise_2 += 1
            continue
        else:
            clusters_2[labels_2[i]].append(class_2[i])

    print(len(clusters_2[0]))
    print(len(clusters_2[1]))

    avg_xy_1 = []
    avg_xy_2 = []
    # find average x and y coordinate of clusters
    for cluster in clusters_1:
        xy_1 = [item[0:1] for item in cluster]
        #ys_1 = [item[1] for item in cluster]
        print("avg xy1: ", str(np.average(xy_1)))
        # print("avg y1: ", str(np.average(ys_1)))
    for cluster in clusters_2:
        xy_2 = [item[0:2] for item in cluster]
        xy_2 = np.array(xy_2)
        avg_xy_2.append(np.mean(xy_2, axis=0))

    print("AAAA", avg_xy_1, avg_xy_2)

    good_clusters_2 = []
    bad_clusters_2 = []
    good_cluster_counter_2 = 0

    # opens the json-file containing the bounding boxes for the current file
    # TODO: fikse til standardiserte navn og bytte ut med (filename + '.json').replace('.txt', '')
    with open('annotation_1_ds0_163__thin_ground_points_2022-03-21_16h49_25_298.pcd.json') as file:
        data = json.load(file)
        class_type = dict([(p['id'], p['classTitle'])
                           for p in data['objects']])
        # goes through all figures (bounding boxes) and tries to match it to a cluster
        for figure in data['figures']:
            # TODO: fix for reg_dump
            if class_type[figure['objectId']] == 'reg_dump':
                for avg in avg_xy_1:
                    fig_xy = [figure['geometry']['position']['x'],
                              figure['geometry']['position']['y']]
                    print("REG")
            elif class_type[figure['objectId']] == 'bus_dump':
                # iterates through all the clusters to match it to the current figure
                for i in range(len(avg_xy_2)):
                    fig_xy = [figure['geometry']['position']['x'],
                              figure['geometry']['position']['y']]
                    cluster_to_gt = np.linalg.norm(fig_xy-avg_xy_2[i])
                    if cluster_to_gt < 1:
                        print(i, figure['id'])
                        print("distance: ", cluster_to_gt)
                        good_clusters_2.append([figure['id'], figure['geometry']['position']
                                                ['x'], figure['geometry']['position']['y'], count_2[i], avg_xy_2[i][0], avg_xy_2[i][1]])
                        good_cluster_counter_2 += 1
                        # TODO: add cluster/figure combo to good results
                        break  # figure has been matched to a cluster, so the next figure can be checked
            else:
                print("???")

    if len(good_clusters_2) > 0:
        with open('./result_folder/good_clusters_2/res_' + filename, 'w') as f:
            print("ok")
            write_string = ''
            for cluster in good_clusters_2:
                write_string += str(cluster) + '\n'
            write_string = write_string.replace('[', '')
            write_string = write_string.replace(']', '')
            write_string = write_string.replace(',', '')
            f.write(write_string)

    if len(bad_clusters_2) > 0:
        with open('./result_folder/bad_clusters_2/res_' + filename, 'w') as f:
            print("ok")
            write_string = ''
            for cluster in bad_clusters_2:
                write_string += str(cluster) + '\n'
            write_string = write_string.replace('[', '')
            write_string = write_string.replace(']', '')
            f.write(write_string)

    # filnavnet det skrives til matcher current filnavn
    # figure-id, figure-x, figure-y, amount of points in cluster, cluster-center-x, cluster-center-y
    print("GOOD CLUSTERS: ", good_clusters_2)


# check_bumps('163__thin_ground_points_123.txt')
check_bumps('31__thin_ground_points_26.txt')


# TODO: skriv til fil for "underfiler", good & bad


# TODO: oppdatér eller lag ny hovedfil-oversikt for precision/recall/true positives/false positives/false negatives


# TODO:
# output: good cluster-file, bad cluster-file
# filnavn, fig_id, fig_xy, cluster_avg_xy, cluster_point_num
# hente totalt antall bumps i hver fil fra json (antall figures), totalt antall true positives og false positives fra filene
# lage en file over hver "hele" fil med totalt true positives, false positives, false negatives som oppdateres hvis man er innom en "underfil"


# gå gjennom alle bounding boxes og finn koordinater (separér reg_dump og bus_dump fra hverandre)
# 2x dobbel for-loop: sammenlign koordinater av bboxes med midtpunkt av clusters

# open file
# ta bare med punkter med class = 1 eller 2 (evt fjerne punkter med class = 0)
# clustre for 1 og 2
# finne clustere og så midtpunkt (enten gjennomsnittskoordinater eller av clusteret)
# sjekke cluster-midtpunkt mot faktisk midtpunkt

# cluster_dbscan(self, eps, min_points, print_progress=False)
# eps (float) – Density parameter that is used to find neighbouring points.
# min_points (int) – Minimum number of points to form a cluster.
# print_progress (bool, optional, default=False) – If true the progress is visualized in the console.
