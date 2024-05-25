import gc

import h5py
import numpy as np
import random
import argparse
import os
from tqdm import tqdm
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-CURRENT', metavar='N', type=int, nargs='?', default=10,
                    help='An integer for the accumulator')

args = parser.parse_args()
CURRENT = args.CURRENT


file_path = "/custom/dataset/vo_dataset/test-72exp"

sample_path = "/custom/dataset/vo_dataset/replaybuffer5120"
output_path = "/custom/dataset/vo_dataset"
train_h5_files = [file for file in os.listdir(file_path) if file.endswith('.h5') and file.startswith('train_')]
train_h5_files = sorted(train_h5_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

replay_h5_files = [file for file in os.listdir(sample_path) if file.endswith('.h5') and file.startswith('replayfor_')]
replay_h5_files = sorted(replay_h5_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

# Open the H5 file
# 13312, 5120, 1024
CHUNKSIZE = 512
VIS_SIZE_W= 341
VIS_SIZE_H= 192
# RGB stored with uint8
RGB_PAIR_SIZE = 2 * VIS_SIZE_W * VIS_SIZE_H * 3
# Depth stored with float16
DEPTH_PAIR_SIZE = 2 * VIS_SIZE_W * VIS_SIZE_H * 2
# global CHUNK_SIZE = int(CHUNK_BYTES / (RGB_PAIR_SIZE + DEPTH_PAIR_SIZE))

# 640MB
# global CHUNK_BYTES = 640 * 1024 * 1024
CHUNK_BYTES = int(np.ceil((RGB_PAIR_SIZE + DEPTH_PAIR_SIZE) * CHUNKSIZE))
H5PY_COMPRESS_KWARGS = {
    # "chunks": True,
    #"compression": "lzf",
    # "compression_opts": 4
}
CHUNK_SHAPE0 = 1
H5PY_COMPRESS_KWARGS_RGB = {
    "chunks": (CHUNK_SHAPE0, VIS_SIZE_W * VIS_SIZE_H * 3),
    "compression": "lzf",
    # "compression_opts": 4
}
H5PY_COMPRESS_KWARGS_DEPTH = {
    "chunks": (CHUNK_SHAPE0, VIS_SIZE_W * VIS_SIZE_H),
    "compression": "lzf",
    # "compression_opts": 4
}


def read_logic():
    all_data_dict = {}
    train_h5file = train_h5_files[CURRENT]
    print(train_h5file)
    replay_h5file = replay_h5_files[0]
    print(replay_h5file)
    assert int(train_h5file.split('_')[-1].split('.')[0]) == int(replay_h5file.split('_')[-1].split('.')[0])
    alldata_dict = {}
    with h5py.File(os.path.join(file_path, train_h5file), 'r+', swmr=True, libver='latest') as f:
        # Create a dictionary to store the data of each dataset
        # Iterate over each chunk

        chunk_numbers = list(f.keys())
        for chunk_number in tqdm(chunk_numbers, desc="Loading chunks into memory", leave=False):
            chunk = f[chunk_number]
            # Get the names of all datasets in the chunk
            dataset_names = list(chunk.keys())

            for dataset_name in dataset_names:
                if dataset_name not in alldata_dict:
                    alldata_dict[dataset_name] = []
                # Append the data to the corresponding dataset list in the dictionary
                alldata_dict[dataset_name].extend(chunk[dataset_name][:])
        f.close()

    # Accumulate data to replay_data_dict
    for key, value in alldata_dict.items():
        if key in all_data_dict:
            # If the key exists, append the data
            all_data_dict[key].extend(value)
        else:
            # If the key does not exist, add the key-value pair
            all_data_dict[key] = value
    del alldata_dict
    gc.collect()

    alldata_dict = {}
    with h5py.File(os.path.join(sample_path, replay_h5file), 'r+') as f:
        # Create a dictionary to store the data of each dataset
        # Iterate over each chunk
        chunk_numbers = list(f.keys())
        for chunk_number in tqdm(chunk_numbers, desc="Loading chunks into memory", leave=False):
            chunk = f[chunk_number]

            # Get the names of all datasets in the chunk
            dataset_names = list(chunk.keys())
            for dataset_name in dataset_names:
                if dataset_name not in alldata_dict:
                    alldata_dict[dataset_name] = []
                # Append the data to the corresponding dataset list in the dictionary
                alldata_dict[dataset_name].extend(chunk[dataset_name][:])
        f.close()

    for key, value in alldata_dict.items():
        all_data_dict[key].extend(value)

    del alldata_dict
    gc.collect()
    dataset_len = len(all_data_dict["actions"])
    print("now total lenth")
    print(dataset_len)
    # Generate shuffled indices
    shuffled_indices = list(range(dataset_len))
    random.shuffle(shuffled_indices)

    with h5py.File(os.path.join(output_path, "train_fulreplay_" + str(int(train_h5file.split('_')[-1].split('.')[0])) + ".h5"),
                   'w', libver="latest",
                   rdcc_nbytes=CHUNK_BYTES, rdcc_nslots=1e7) as f:
        for dataset_name, data in tqdm(all_data_dict.items(), desc="Writing in datasets"):
            idx = 0  # 用于追踪当前数据集中的位置
            chunk_idx = 0

            while idx < len(data):
                # 动态生成chunk名称，如 "chunk_0", "chunk_1" 等
                chunk_group_name = f"chunk_{chunk_idx}"
                if chunk_group_name not in f:
                    f.create_group(f"chunk_{chunk_idx}")

                # 确定当前chunk中数据的结束索引

                end_idx = min(idx + CHUNKSIZE, len(data))


                valid_indices = []
                for i in shuffled_indices[idx:end_idx]:
                    if i < len(data):
                        valid_indices.append(i)
                    else:
                        # 打印错误信息
                        print(f"Error: Index {i} out of range for dataset with length {len(data)}.")

                data_to_save = [data[i] for i in valid_indices]


                # 创建数据集并写入数据
                if "rgbs" in dataset_name:
                    f[chunk_group_name].create_dataset(dataset_name, data=data_to_save, **H5PY_COMPRESS_KWARGS_RGB)
                elif "depths" in dataset_name:
                    f[chunk_group_name].create_dataset(dataset_name, data=data_to_save,
                                                       **H5PY_COMPRESS_KWARGS_DEPTH)
                else:
                    f[chunk_group_name].create_dataset(dataset_name, data=data_to_save, **H5PY_COMPRESS_KWARGS)

                # 更新索引，准备处理下一块数据
                idx = idx + CHUNKSIZE
                chunk_idx = chunk_idx + 1
        f.flush()
        f.close()


def wode():
    total_size = 0
    with h5py.File(os.path.join("/custom/dataset/vo_dataset/test-buffer-5120", "train_fulreplay_57.h5"), 'r', libver="latest") as f:
        print("Chunk Sizes:")
        for chunk_name in f.keys():
            chunk = f[chunk_name]  # 获取 chunk
            if 'apartment' in chunk:  # 检查 chunk 中是否有 apartment 数据集
                apartment_data = chunk['apartment'][:]  # 读取 apartment 数据
                chunk_size = len(apartment_data)
                print(f"{chunk_name}: {chunk_size} elements")
                total_size += chunk_size  # 累加每个 chunk 的大小

    # 打印总大小
    print(f"Total size: {total_size} elements")


if __name__ == "__main__":
    #read_logic()
    wode()



