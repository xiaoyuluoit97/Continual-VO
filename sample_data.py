import h5py
import numpy as np
import gc
import os
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-CURRENT', metavar='N', type=int, nargs='?', default=10,
                    help='An integer for the accumulator')

args = parser.parse_args()
CURRENT = args.CURRENT

file_path = "/custom/dataset/vo_dataset/test-72exp"
output_path = "/custom/dataset/vo_dataset/replaybuffer5120"
h5_files = [file for file in os.listdir(file_path) if file.endswith('.h5') and file.startswith('train_')]
h5_files = sorted(h5_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
# Open the H5 file
# 13888, 5120, 1024 36开始
#"replayfor_"+str(current_exp)+".h5")
current_exp = CURRENT
buffer_size = 5120
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
    i = 1
    each_exp_load = int(buffer_size/(current_exp-1))
    last_exp_load_more = buffer_size % (current_exp-1)


    replay_data_dict = {}
    for h5file in h5_files[:(current_exp-1)]:

        chunk_needed = int(each_exp_load / CHUNKSIZE)

        if each_exp_load % CHUNKSIZE > 0:
            chunk_needed = chunk_needed + 1
            last_exp_chunk_needed = chunk_needed
        else:
            last_exp_chunk_needed = chunk_needed + 1

        if i == (current_exp-1):
            final_chunk_needed = last_exp_chunk_needed
        else:
            final_chunk_needed = chunk_needed


        print(os.path.join(file_path,h5file))
        alldata_dict = {}
        with h5py.File(os.path.join(file_path,h5file), 'r+') as f:
            # Create a dictionary to store the data of each dataset
            # Iterate over each chunk
            chunk_numbers = list(f.keys())
            for chunk_number in tqdm(chunk_numbers[:final_chunk_needed], desc="Loading chunks into memory", leave=False):
                chunk = f[chunk_number]
                # Get the names of all datasets in the chunk
                dataset_names = list(chunk.keys())
                for dataset_name in dataset_names:
                    if dataset_name not in alldata_dict:
                        alldata_dict[dataset_name] = []
                    # Append the data to the corresponding dataset list in the dictionary
                    alldata_dict[dataset_name].extend(chunk[dataset_name][:])

            f.close()

        if i == (current_exp-1):
            new_data_dict = {dataset_name: data[:(each_exp_load+last_exp_load_more)] for dataset_name, data in alldata_dict.items()}
        else:
            new_data_dict = {dataset_name: data[:each_exp_load] for dataset_name, data in alldata_dict.items()}

        del alldata_dict

        # 累加数据到 replay_data_dict
        for key, value in new_data_dict.items():
            if key in replay_data_dict:
                # 如果键已存在，追加数据
                replay_data_dict[key].extend(value)
            else:
                # 如果键不存在，添加键值对
                replay_data_dict[key] = value

        del new_data_dict
        if i == (current_exp-1) :
            assert len(replay_data_dict['actions']) == (i * each_exp_load) + last_exp_load_more
        else:
            assert len(replay_data_dict['actions']) == (i * each_exp_load)
        i = i + 1
        gc.collect()

    assert len(replay_data_dict["actions"]) == buffer_size

    with h5py.File(os.path.join(output_path, "replayfor_"+str(current_exp)+".h5"), 'w', libver="latest", rdcc_nbytes=CHUNK_BYTES, rdcc_nslots=1e7) as f:

        for dataset_name, data in tqdm(replay_data_dict.items(), desc="Writing in datasets"):
            idx = 0  # 用于追踪当前数据集中的位置
            chunk_idx = 0
            while idx < len(data):
                # 动态生成chunk名称，如 "chunk_0", "chunk_1" 等
                chunk_group_name = f"chunk_{chunk_idx}"
                if chunk_group_name not in f:
                    f.create_group(f"chunk_{chunk_idx}")

                # 确定当前chunk中数据的结束索引
                end_idx = min(idx + CHUNKSIZE, len(data))
                data_to_save = data[idx:end_idx]
                # 创建数据集并写入数据

                if "rgbs" in dataset_name:
                    f[chunk_group_name].create_dataset(dataset_name, data=data_to_save,**H5PY_COMPRESS_KWARGS_RGB)
                elif "depths" in dataset_name:
                    f[chunk_group_name].create_dataset(dataset_name, data=data_to_save, **H5PY_COMPRESS_KWARGS_DEPTH)
                else:
                    f[chunk_group_name].create_dataset(dataset_name, data=data_to_save, **H5PY_COMPRESS_KWARGS)

                # 更新索引，准备处理下一块数据
                idx = idx + CHUNKSIZE
                chunk_idx = chunk_idx+1
                #print(chunk_idx)

        f.flush()



if __name__ == "__main__":
    read_logic()
