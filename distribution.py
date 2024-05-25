import h5py
import numpy as np
import os
from collections import Counter



folder_path = '/custom/dataset/vo_dataset/test-72exp'  # 文件夹路径，请替换成你的文件夹路径
folder_path = '/custom/PointNav-VO-single/dataset/vo_dataset'
# 使用 os.listdir() 列出文件夹中的所有文件
file_list = os.listdir(folder_path)

overall_files = [file for file in file_list if file.startswith('val_')]


length = 0
forward_num = 0
left_num = 0
right_num = 0
collision_num = 0
valid_idxes = 0
all_num = 0
forward_col_num = 0
left_col_num = 0
right_col_num = 0

def indexes(h5_f, scence_n_chunk_k):
    valid_act_idxes = np.arange(h5_f[scence_n_chunk_k]["actions"].shape[0])
    return list(valid_act_idxes)


for file_path in overall_files:
    chunk_splits = []
    with h5py.File(os.path.join(folder_path,file_path), "r", libver="latest") as f:
        for scence_n_chunk_k in sorted(f.keys()):
            chunk_splits.append(scence_n_chunk_k)
            # print(scence_n_chunk_k)
            valid_idxes = indexes(f, scence_n_chunk_k)

    for chunk_k in chunk_splits:
        with h5py.File(
                os.path.join(folder_path,file_path),
                "r",
                libver="latest",

                rdcc_nslots=1e7,
        ) as f:
            # get valid indexes of each chunk!!!!!!!!!

            actions = f[chunk_k]["actions"][()]
            collision = f[chunk_k]["collisions"][()]
            all_num = all_num + len(actions)
            for i in range(len(actions)):

                if actions[i] == 1 and collision[i] == 1:
                    forward_col_num = forward_col_num + 1
                elif actions[i] == 2 and collision[i] == 1:
                    left_col_num = left_col_num + 1
                elif actions[i] == 3 and collision[i] == 1:
                    right_col_num = right_col_num + 1

            counted_elements = Counter(actions)
            collision_elements = Counter(collision)
            # 输出特定元素的计数
            right_num = right_num + counted_elements[3]
            left_num = left_num + counted_elements[2]
            forward_num = forward_num + counted_elements[1]
            collision_num = collision_num + collision_elements[1]



            f.close()

right_per = ((right_num - right_col_num) / right_num) * 100
right_col_per = (right_col_num / right_num) * 100

left_per = ((left_num - left_col_num )/ left_num) * 100
left_col_per = (left_col_num / left_num) * 100

forward_per = ((forward_num - forward_col_num) / forward_num) * 100
forward_col_per = (forward_col_num / forward_num) * 100

collision_per = (collision_num / all_num) * 100
collision_all = right_col_num+left_col_num+forward_col_num
print(f"right numbers {right_num}")
print(f"right collisions numbers {right_col_num}")
print(f"right actions have collisions presentage {right_col_per}")


print(f"left numbers {left_num}")
print(f"left collisions numbers {left_col_num}")
print(f"rleft actions have collisions presentage {left_col_per}")

print(f"forward numbers {forward_num}")
print(f"forward collisions numbers {forward_col_num}")
print(f"forward actions have collisions presentage {forward_col_per}")


print(f"overall cool {collision_all}")
print(f"overall cool {collision_num}")