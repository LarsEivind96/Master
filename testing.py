import os
import numpy as np

test = np.zeros((10, 10))
test3 = np.asarray([True, False, False, False, False, False, False, True, False, True])
test4 = np.asarray([True, False, True, False, False, False, False, True, False, True])
test1 = np.asarray([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
test2 = np.asarray([1, 1, 1, 0, 0, 2, 2, 2, 2, 2])

print(test[test3])
print(np.logical_or(test1, test2))

meta_path = "C:/Users/LARLIE/School/Master/annotation1/"

with open(meta_path + "meta.txt", 'r') as f:
    #for line in f.readlines():
     #   print(line.split("\t")[0])
    print(len(f.readlines()))
    
