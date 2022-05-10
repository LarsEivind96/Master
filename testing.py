import os
import numpy as np

lidar_file = os.path.join("000016.bin")
assert os.path.exists(lidar_file)
arr = np.fromfile(lidar_file, dtype=np.float32)
print(arr)
arr = arr.reshape(-1, 4)

print(arr)

'''txt = "8__thin_ground_points.json_0.txt"
arr = np.loadtxt(txt)
print(arr[:, 0:4])'''

def cart_to_hom( pts):
    """
    :param pts: (N, 3 or 2)
    :return pts_hom: (N, 4 or 3)
    """
    pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
    print("pts_hom:\n", pts_hom)
    
    print(max(pts_hom[:, 0]), max(pts_hom[:, 1]), max(pts_hom[:, 2]))
    print(min(pts_hom[:, 0]), min(pts_hom[:, 1]), min(pts_hom[:, 2]))
    return pts_hom

def lidar_to_rect( pts_lidar):
    """
    :param pts_lidar: (N, 3)
    :return pts_rect: (N, 3)
    """
    pts_lidar_hom = cart_to_hom(pts_lidar)
    damane = np.dot(V2C.T, R0.T)
    print("damane\n", damane)
    pts_rect = np.dot(pts_lidar_hom, damane)
    # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
    print("pts_rect:\n", pts_rect)
    print(max(pts_rect[:, 0]), max(pts_rect[:, 1]), max(pts_rect[:, 2]))
    print(min(pts_rect[:, 0]), min(pts_rect[:, 1]), min(pts_rect[:, 2]))
    return pts_rect

def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

calib = get_calib_from_file("000016_calib.txt")

P2 = calib['P2']  # 3 x 4
R0 = calib['R0']  # 3 x 3
V2C = calib['Tr_velo2cam']  # 3 x 4

lidar_to_rect(arr[:, 0:3])


