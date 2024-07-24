import gc

import h5py
import numpy as np
import random
import os
from tqdm import tqdm

from collections import Counter


def wode():
    print("gogogo")
    with h5py.File('/custom/dataset/vo_dataset/test-buffer/train_fulreplay_7.h5', 'r') as f:
        all_apartment_data = []  # 初始化一个列表来累积所有 apartment 数据

        # 遍历文件中的每个 chunk
        for chunk_name in f.keys():
            chunk = f[chunk_name]  # 获取 chunk
            if 'apartment' in chunk:  # 检查 chunk 中是否有 apartment 数据集
                apartment_data = chunk['apartment'][:]  # 读取 apartment 数据
                all_apartment_data.extend(apartment_data)  # 累积 apartment 数据

                # 对当前 chunk 的 apartment 数据进行计数
                element_counts = Counter(apartment_data)
                print(f"{chunk_name} 中的 apartment 数据元素及其计数:")
                for element, count in element_counts.items():
                    print(f"元素 {element}: 出现 {count} 次")

        # 对所有 chunk 累积的 apartment 数据进行总体计数
        total_element_counts = Counter(all_apartment_data)
        print("\n所有 chunk 的总 apartment 数据元素及其计数:")
        for element, count in total_element_counts.items():
            print(f"元素 {element}: 总共出现 {count} 次")
            

if __name__ == "__main__":
    #read_logic()
    wode()