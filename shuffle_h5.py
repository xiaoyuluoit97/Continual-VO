import h5py
import numpy as np
import random
import os
from tqdm import tqdm
#h5_files = [file for file in os.listdir('.') if file.endswith('.h5')]
# Open the H5 file
def main():

    #print("Processing file:", h5_file)
    #test_Brevort_20.h5
    with h5py.File("/custom/dataset/vo_dataset/test-72exp/train_Applewold_19.h5", 'r+') as f:
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
            # Record the size of the current chunk
            chunk_sizes[chunk_number] = len(chunk['actions'])

        # Calculate the total length of the dataset
        dataset_len = sum(chunk_sizes.values())
        # Generate shuffled indices
        shuffled_indices = list(range(dataset_len))
        random.shuffle(shuffled_indices)
        #with open('shuffled_indices.txt', 'w') as file:
            #for idx in shuffled_indices:
                #file.write(str(idx) + '\n')
        # Iterate over each dataset and rearrange the data according to the shuffled indices
        for dataset_name, data in tqdm(alldata_dict.items(), desc="Writing in datasets"):
            idx = 0
            for chunk_number, chunk_size in tqdm(chunk_sizes.items(), desc="Writing in one chunks", leave=False):
                # Get the shuffled data in the current chunk
                shuffled_data = [data[i] for i in shuffled_indices[idx:idx + chunk_size]]
                # Update the data of the corresponding dataset in the current chunk
                f[chunk_number][dataset_name][:] = shuffled_data
                # Update the index position
                idx += chunk_size
        f.flush()

    print("Shuffling completed.")
if __name__ == "__main__":
    main()