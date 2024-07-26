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
        all_apartment_data = []

        # check each chunk
        for chunk_name in f.keys():
            chunk = f[chunk_name]
            if 'apartment' in chunk:
                apartment_data = chunk['apartment'][:]
                all_apartment_data.extend(apartment_data)

                element_counts = Counter(apartment_data)
                print(f"{chunk_name} count:")
                for element, count in element_counts.items():
                    print(f"element {element}: appear {count} times")

        total_element_counts = Counter(all_apartment_data)
        print("\nall chunks")
        for element, count in total_element_counts.items():
            print(f"element {element}: appear {count} times")
            

if __name__ == "__main__":
    #read_logic()
    wode()