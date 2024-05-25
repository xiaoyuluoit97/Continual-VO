import h5py
import csv
import numpy as np
import random
import os
from tqdm import tqdm
from torch import FloatTensor

#print("NOW MAKE DIR")
#log_folder='.'
#os.makedirs(log_folder, exist_ok=True)
overall_forward_pred = "./overall_dataset_distribution.csv"
overall_forward_pred = open(
    overall_forward_pred, "w"
)
print(
    "forward_x_mean",
    "forward_x_var",
    "forward_yaw_mean",
    "forward_yaw_var",
    "forward_z_mean",
    "forward_z_var",
    "left_x_mean",
    "left_x_var",
    "left_yaw_mean",
    "left_yaw_var",
    "left_z_mean",
    "left_z_var",
    "right_x_mean",
    "right_x_var",
    "right_yaw_mean",
    "right_yaw_var",
    "right_z_mean",
    "right_z_var",
    "forward_num",
    "left_num",
    "right_num",
    sep=",",
    file=overall_forward_pred,
    flush=True,
)

h5_files = [file for file in os.listdir('/custom/dataset/vo_dataset/test-72exp') if file.endswith('.h5')]
DATA_FOLDER_PATH = '/custom/dataset/vo_dataset/test-72exp'
# Open the H5 file
Train_files = [f for f in h5_files if f.startswith("train")]
Val_files = [f for f in h5_files if f.startswith("val")]
Test_files = [f for f in h5_files if f.startswith("test")]
def _process_data(actions, delta_positions, delta_rotations,i,Action):

    delta_pos_cur_rel_to_prev = delta_positions[i, :]
    delta_x = FloatTensor([delta_pos_cur_rel_to_prev[0]])
    delta_z = FloatTensor([delta_pos_cur_rel_to_prev[2]])

    delta_rotation_quaternion_cur_rel_to_prev = delta_rotations[i, :]

    dyaw_cur_rel_to_prev = FloatTensor(
        [
            2
            * np.arctan2(
                delta_rotation_quaternion_cur_rel_to_prev[1],
                delta_rotation_quaternion_cur_rel_to_prev[3],
            )
        ]
    )
    delta_yaw = dyaw_cur_rel_to_prev

    if actions[i] == 1:
        Action[0][0].append(delta_x.item())
        Action[0][1].append(delta_yaw.item())
        Action[0][2].append(delta_z.item())
    elif actions[i] == 2:
        Action[1][0].append(delta_x.item())
        Action[1][1].append(delta_yaw.item())
        Action[1][2].append(delta_z.item())
    elif actions[i] == 3:
        Action[2][0].append(delta_x.item())
        Action[2][1].append(delta_yaw.item())
        Action[2][2].append(delta_z.item())

    return


# 按结尾的数字进行排序
def custom_sort(file_name):
    # 提取文件名中第三段的数字
    num = int(file_name.split("_")[2].split(".")[0])
    return num

train_files = sorted(Train_files, key=custom_sort)
val_files = sorted(Val_files, key=custom_sort)
test_files = sorted(Test_files, key=custom_sort)

def overall_pred_metrics(
        forward_x_mean,
        forward_x_var,
        forward_yaw_mean,
        forward_yaw_var,
        forward_z_mean,
        forward_z_var,
        left_x_mean,
        left_x_var,
        left_yaw_mean,
        left_yaw_var,
        left_z_mean,
        left_z_var,
        right_x_mean,
        right_x_var,
        right_yaw_mean,
        right_yaw_var,
        right_z_mean,
        right_z_var,
        forward_num,
        left_num,
        right_num,
):

    print(
            forward_x_mean,
            forward_x_var,
            forward_yaw_mean,
            forward_yaw_var,
            forward_z_mean,
            forward_z_var,
            left_x_mean,
            left_x_var,
            left_yaw_mean,
            left_yaw_var,
            left_z_mean,
            left_z_var,
            right_x_mean,
            right_x_var,
            right_yaw_mean,
            right_yaw_var,
            right_z_mean,
            right_z_var,
            forward_num,
            left_num,
            right_num,
            sep=",",
            file=overall_forward_pred,
            flush=True,
        )

for k in tqdm(range(len(train_files)), desc="Loading processing h5 file now", leave=False):
    forward_x = []
    forward_yaw = []
    forward_z = []
    Forward = [forward_x,forward_yaw,forward_z]
    left_x = []
    left_yaw = []
    left_z = []
    Left = [left_x,left_yaw,left_z]
    right_x = []
    right_yaw = []
    right_z = []
    Right = [right_x,right_yaw,right_z]

    Action = [Forward,Left,Right]
    print("Processing file:", train_files[k])
    print("Processing file:", val_files[k])
    print("Processing file:", test_files[k])
    with h5py.File(os.path.join(DATA_FOLDER_PATH, train_files[k]), 'r+') as f:
        chunk_numbers = list(f.keys())
        for chunk in chunk_numbers:
            actions = f[chunk]["actions"][()]
            collsions = f[chunk]["collisions"][()]
            delta_positions = f[chunk]["delta_positions"][()]
            delta_rotations = f[chunk]["delta_rotations"][()]

            for i in range(len(actions)):
                _process_data(actions,delta_positions,delta_rotations,i,Action)
        f.close()

    with h5py.File(os.path.join(DATA_FOLDER_PATH, val_files[k]), 'r+') as f:
        chunk_numbers = list(f.keys())
        for chunk in chunk_numbers:
            actions = f[chunk]["actions"][()]
            collsions = f[chunk]["collisions"][()]
            delta_positions = f[chunk]["delta_positions"][()]
            delta_rotations = f[chunk]["delta_rotations"][()]

            for i in range(len(actions)):
                _process_data(actions,delta_positions,delta_rotations,i,Action)
        f.close()

    with h5py.File(os.path.join(DATA_FOLDER_PATH, test_files[k]), 'r+') as f:
        chunk_numbers = list(f.keys())
        for chunk in chunk_numbers:
            actions = f[chunk]["actions"][()]
            collsions = f[chunk]["collisions"][()]
            delta_positions = f[chunk]["delta_positions"][()]
            delta_rotations = f[chunk]["delta_rotations"][()]

            for i in range(len(actions)):
                _process_data(actions,delta_positions,delta_rotations,i,Action)
        f.close()
    print(len(left_x))
    print(len(forward_x))
    print(len(right_x))

    overall_pred_metrics(
        np.mean(Action[0][0]),
        np.var(Action[0][0]),
        np.mean(Action[0][1]),
        np.var(Action[0][1]),
        np.mean(Action[0][2]),
        np.var(Action[0][2]),
        np.mean(Action[1][0]),
        np.var(Action[1][0]),
        np.mean(Action[1][1]),
        np.var(Action[1][1]),
        np.mean(Action[1][2]),
        np.var(Action[1][2]),
        np.mean(Action[2][0]),
        np.var(Action[2][0]),
        np.mean(Action[2][1]),
        np.var(Action[2][1]),
        np.mean(Action[2][2]),
        np.var(Action[2][2]),
        len(Action[0][0]),
        len(Action[1][0]),
        len(Action[2][0]),
    )


