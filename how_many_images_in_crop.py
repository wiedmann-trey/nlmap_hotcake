dir_path = '/home/ifrah/longterm_semantic_map/nlmap_hotcake/clusters/batch_1122'

import os
# from hs import HashSet  

dir_path_1 = os.path.join(dir_path, '-1')
# dir_path_2 = os.path.join(dir_path, '0')
# dir_path_3 = os.path.join(dir_path, '1')
# dir_path_4 = os.path.join(dir_path, '2')

paths = [dir_path_1]

images = []

count = 0
for dir in paths:
    print(f"length of {len(os.listdir(dir))}")
    for path in os.listdir(dir):
        if path[54:96] not in images:
            images.append(path[54:96])
            count += 1
print(f"unique image count {count}")

