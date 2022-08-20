import os


seg_path = "data/zerowaste-f/test/sem_seg/"
data_path = "data/zerowaste-f/test/data/"

seg_list = os.listdir(seg_path)
data_list = os.listdir(data_path)


rm_list = []
for fname in data_list:
    if fname not in seg_list:
        rm_list.append(fname)

print(len(rm_list))

for fname in rm_list:
    fullname = os.path.join(data_path, fname)
    os.remove(fullname)