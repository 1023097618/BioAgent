# -*- coding: UTF-8 -*-
import numpy as np
import os
import glob

def load_data(file_pattern):
    data = {}
    files = glob.glob(file_pattern)
    for file in files:
        batch_data = np.load(file,allow_pickle=True)
        for key in batch_data:
            if key not in data:
                data[key] = []
            data[key].append(batch_data[key])
    for key in data:
        data[key] = np.concatenate(data[key], axis=0)
    print('load all,no memory problem,If there is memoryError,please use shuffle_npz_save.py')
    return data

def shuffle_and_split_data(data, batch_size):
    num_samples = data['matrix'].shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    shuffled_data = {key: value[indices] for key, value in data.items()}

    batches = []
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch = {key: value[start_idx:end_idx] for key, value in shuffled_data.items()}
        batches.append(batch)
    return batches

def save_batches(batches, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, batch in enumerate(batches):
        np.savez(os.path.join(output_dir, f'batch_{i}.npz'), **batch)
        print(f'saved to batch_{i}.npz')

def main(input_dir, output_dir, batch_size):
    file_pattern = os.path.join(input_dir, 'batch_*.npz')
    data = load_data(file_pattern)
    batches = shuffle_and_split_data(data, batch_size)
    save_batches(batches, output_dir)

if __name__ == '__main__':
    input_dir = os.path.join('..','..','output_npz')
    output_dir = os.path.join('..','..','output_npz')
    batch_size = 64  # Adjust this to your needs
    main(input_dir, output_dir, batch_size)
