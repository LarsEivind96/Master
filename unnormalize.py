import numpy as np
import itertools


def unnorm(filename):
    file_orig = './orig_files/' + filename
    file_new = './new_files/' + filename

    m, centroid = pc_normalize(file_orig)
    print('m: ', m, '\ncentroid: ', centroid)

    pc = np.loadtxt(file_new).astype(np.float32)[:, 0:3]
    pc = pc * m
    pc = pc + centroid

    class_list = np.loadtxt(file_new).astype(np.float32)[:, 3]

    orig_pc = np.loadtxt(file_orig).astype(np.float32)[:, 0:3]

    print(np.max(orig_pc))
    print(np.max(pc))
    print(np.min(orig_pc))
    print(np.min(pc))

    if len(pc) == len(class_list):
        # for i in range(len(class_list)):
        new_file = './unnorm_4096/' + filename
        with open(new_file, "w") as file:
            for i in range(len(pc)):
                curr_point = np.append(pc[i], class_list[i])
                line = ' '.join(str(e) for e in (curr_point))
                print(line)
                file.write(line + '\n')
    else:
        print(len(pc), len(class_list))

    # write to file


def pc_normalize(filename):

    pc = np.loadtxt(filename).astype(np.float32)[:, 0:3]

    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return m, centroid

# for filename in filenames:
#   unnorm(filename)


unnorm('33__thin_ground_points_79.txt')
