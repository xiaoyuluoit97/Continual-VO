import h5py
import numpy as np
import random
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-CURRENT', metavar='N', type=int, nargs='?', default=10,
                    help='An integer for the accumulator')

args = parser.parse_args()
CURRENT = args.CURRENT

file_path = "/custom/dataset/vo_dataset/test-72exp"
h5_files = [file for file in os.listdir(file_path) if file.endswith('.h5') and file.startswith('train_')]
h5_files = sorted(h5_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
# Open the H5 file
#11

def main():
    print(h5_files)
    for h5file in tqdm(h5_files[CURRENT:(CURRENT+1)]):
        print(os.path.join(file_path, h5file))
        with h5py.File(os.path.join(file_path,h5file), 'r+')  as f:
            # Create a dictionary to store the data of each dataset
            alldata_dict = {}
            chunk_sizes = {}  # Record the size of each chunk
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

                apartment_num = int(h5file.split('_')[-1].split('.')[0])
                alldata_dict["apartment"] = [apartment_num] * len(alldata_dict["actions"])
                # Record the size of the current chunk
                chunk_sizes[chunk_number] = len(chunk['actions'])


            dataset_len = sum(chunk_sizes.values())
            shuffled_indices = list(range(dataset_len))
            random.shuffle(shuffled_indices)

            for dataset_name, data in tqdm(alldata_dict.items(), desc="Writing in datasets"):
                idx = 0
                for chunk_number, chunk_size in chunk_sizes.items():
                    # Get the shuffled data in the current chunk
                    shuffled_data = [data[i] for i in shuffled_indices[idx:idx + chunk_size]]
                    # Update the data of the corresponding dataset in the current chunk
                    if dataset_name == "apartment":
                        f[chunk_number].create_dataset(dataset_name, data=shuffled_data)
                    else:
                        f[chunk_number][dataset_name][:] = shuffled_data
                    # Update the index position
                    idx += chunk_size
                f.flush()

            f.flush()
            f.close()
            f.close()


    print("Shuffling completed.")
if __name__ == "__main__":
    main()