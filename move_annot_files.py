import os
import numpy as np
import shutil

thinned_path = "C:/Users/LARLIE/School/Master/thinned/"
new_path = "C:/Users/LARLIE/School/Master/annotation3/las/"
base_list = [1112,
             1124,
             1128,
             1359,
             1368,
             1439,
             1534,
             1632,
             1678,
             1680,
             1684,
             1716,
             1719,
             1730,
             1753,
             1931,
             1981,
             2024,
             2042,
             2053,
             2098,
             2151,
             2174,
             2175,
             2226,
             2261,
             2283,
             2284,
             2297,
             2465,
             2482,
             2490,
             2548,
             2549,
             2553,
             2726,
             2732,
             2889,
             2978,
             3272,
             3292,
             3315,
             3327,
             3343,
             3344,
             3798,
             3801,
             3849,
             4175,
             4419,
             4420,
             4441,
             4455,
             4467,
             4473,
             4545,
             4683,
             4703,
             4815,
             5158,
             5192,
             5232,
             5242,
             5736,
             5737,
             5775,
             5851,
             5866
             ]


def copy_files_to_testing():
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    for base in base_list:
        file_path = thinned_path + str(base) + '__thin.las'

        shutil.move(file_path, new_path)

copy_files_to_testing()