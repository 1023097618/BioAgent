# -*- coding: UTF-8 -*-
import numpy as np
import os
import glob
import re

input_dir = os.path.join('..', '..', 'output_npz')
output_dir = os.path.join('..', '..', 'output_npz')

# Function to extract the batch number from the filename
def extract_batch_number(filename):
    match = re.search(r'batch_(\d+)\.npz', filename)
    return int(match.group(1)) if match else -1

# Step 1: 统计所有样本总数量
batch_files = sorted(glob.glob(os.path.join(input_dir, 'batch_*.npz')), key=extract_batch_number)
total_samples = 0
sample_shapes = None
label_shapes = {}
labels = []
batch_size = None

for batch_file in batch_files:
    data = np.load(batch_file)
    if sample_shapes is None:
        sample_shapes = data['matrix'].shape[1:]
        for key in data.keys():
            if key.startswith('label_'):
                label_shapes[key] = data[key].shape[1:]
                labels.append(key)
        batch_size = data['matrix'].shape[0]  # 这里确定 batch_size
    total_samples += data['matrix'].shape[0]

# Step 2: 生成打乱的索引
indices = np.arange(total_samples)
np.random.shuffle(indices)

# Step 3: 动态生成新的 batch_*.npz 文件
num_batches = (total_samples + batch_size - 1) // batch_size

current_index = 0
for i in range(num_batches):
    start_index = i * batch_size
    end_index = min(start_index + batch_size, total_samples)
    batch_indices = indices[start_index:end_index]

    # 创建新的 batch 数据
    new_matrix = np.zeros((len(batch_indices), *sample_shapes), dtype=data['matrix'].dtype)
    new_labels = {key: np.zeros((len(batch_indices), *label_shapes[key]), dtype=data[key].dtype) for key in labels}

    for j, idx in enumerate(batch_indices):
        batch_idx = idx // batch_size
        sample_idx = idx % batch_size
        batch_data = np.load(batch_files[batch_idx])
        new_matrix[j] = batch_data['matrix'][sample_idx]
        for key in labels:
            new_labels[key][j] = batch_data[key][sample_idx]

    # 保存新的 batch 文件
    output_file = os.path.join(output_dir, f'new_batch_{i}.npz')
    np.savez(output_file, matrix=new_matrix, **new_labels)

print("数据打乱并重新保存完成。")
