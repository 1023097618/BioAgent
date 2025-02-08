# -*- coding: UTF-8 -*-
import gc
import os
import sys
import time
from multiprocessing import Lock, Manager

import numpy as np
import pickle
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
from src.config.constants import ALL_SPECIES_NAME, LEVEL_NAMES


# 加载已有的 all_species 和 level_names
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


total_memory = psutil.virtual_memory().total


# 检查可用内存是否大于指定值（以字节为单位）
def check_memory(lock, check_name, *shapes, max_memory_gb=5):
    min_available_memory = sum(np.prod(shape, dtype=np.int64) for shape in shapes)
    available_memory = psutil.virtual_memory().free
    available_percentage = available_memory / total_memory

    # 修改部分：检查预估内存是否大于指定值
    if min_available_memory > max_memory_gb * 1024 ** 3:
        print(f"{os.getpid()}: Estimated memory for {check_name} exceeds {max_memory_gb} GB, skipping file...")
        return False

    while available_memory < min_available_memory and available_percentage < 0.05:
        print(
            f"{os.getpid()}: Not enough memory available for {check_name} ,which needs {min_available_memory} , waiting...")
        lock.release()
        time.sleep(60)
        lock.acquire()
        available_memory = psutil.virtual_memory().free
        available_percentage = available_memory / total_memory

    return True


# 解析文件并生成矩阵
def parse_file(file_path, all_species_level, num_species, labels, lock):
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("# Constructed from biom file"):
                continue
            parts = line.strip().split('\t')
            if line.startswith('#'):
                is_sample = [not part.startswith('#SampleID') for part in parts]
                num_sample = is_sample.count(True)
                lock.acquire()
                # print(f'{os.getpid()}: start free memory {psutil.virtual_memory().free}')
                if not check_memory(lock, file_path, (num_sample, 10),
                                    (num_sample, num_species, len(all_species_level), 10), (num_sample, 1000)):
                    lock.release()
                    return None
                total_abundance = np.zeros(num_sample, dtype=float)
                matrices = np.zeros((num_sample, num_species, len(all_species_level)))
                all_labels = {f'label_{i}': [label] * num_sample for i, label in enumerate(labels)}
                total_abundance.fill(0)
                matrices.fill(0)
                # print(f'{os.getpid()}:real used memory {sys.getsizeof(matrices)+sys.getsizeof(all_labels)+sys.getsizeof(total_abundance)}')
                # print(f'{os.getpid()}:end free memory {psutil.virtual_memory().free}')
                lock.release()
                continue
            abundances = [float(_) for i, _ in enumerate(parts) if is_sample[i]]
            taxonomy = [_ for _ in parts if _.startswith('sk__')]
            if not taxonomy:
                continue
            else:
                taxonomy = taxonomy[0]
            for i, abundance in enumerate(abundances):
                total_abundance[i] += abundance
                taxonomy_levels = taxonomy.strip().split(';')
                for j, taxonomy_level in enumerate(taxonomy_levels):
                    if taxonomy_level.endswith('__'):
                        continue
                    if taxonomy_level in all_species_level[j]:
                        indexes = all_species_level[j][taxonomy_level]
                        matrices[i][indexes, j] += abundance
        matrices = np.array([matrices[i] / total if total > 0 else matrices[i] for i, total in enumerate(total_abundance)])
    lock.acquire()
    del total_abundance
    gc.collect()
    lock.release()
    return matrices, num_sample, all_labels


# 获取所有层级的名称
def get_all_levels(sub_dirs):
    level_names = {}
    for sub_dir in sub_dirs:
        sub_dir = os.path.basename(sub_dir)
        sub_dir_levels = sub_dir.strip().split('-')
        for i, sub_dir_level in enumerate(sub_dir_levels):
            if i not in level_names:
                level_names[i] = set()
            level_names[i].add(sub_dir_level)
    for key in level_names.keys():
        level_names[key]=sorted(level_names[key])
    return level_names


# 生成层级的 one-hot 编码
def generate_one_hot(levels, level_names):
    one_hot_label = []
    for i in range(len(level_names)):
        one_hot = [0] * (len(level_names[i])+1)
        if i < len(levels):
            level = levels[i]
            level_names_list = level_names[i]
            if level in level_names_list:
                level_index = level_names_list.index(level)
                one_hot[level_index] = 1
        else:
            one_hot[len(level_names[i])] = 1
        one_hot_label.append(one_hot)
    return one_hot_label


# 递归遍历目录，读取所有的文本文件
def get_file_paths(root_dir):
    file_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.tsv') and 'SSU' in file:
                file_paths.append(os.path.join(root, file))
    return file_paths


# 解析所有的文本文件，统计所有的物种
def parse_all_files(file_paths):
    all_species = {}
    all_species_level = {}
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                taxonomy = [_ for _ in parts if _.startswith('sk__')]
                if not taxonomy:
                    continue
                else:
                    taxonomy = taxonomy[0]
                if taxonomy not in all_species:
                    all_species[taxonomy] = len(all_species)
                else:
                    continue
                species_index = all_species[taxonomy]
                taxonomy_levels = taxonomy.strip().split(';')
                for i, taxonomy_level in enumerate(taxonomy_levels):
                    if i not in all_species_level:
                        all_species_level[i] = {}
                    if taxonomy_level not in all_species_level[i]:
                        all_species_level[i][taxonomy_level] = []
                    all_species_level[i][taxonomy_level].append(species_index)
    return all_species, all_species_level


def process_file(file_path, all_species_level, num_species, level_names, temp_dir, lock):
    sub_dir_name = os.path.basename(os.path.dirname(file_path))
    levels = sub_dir_name.split('-')
    labels = generate_one_hot(levels, level_names)

    result = parse_file(file_path, all_species_level, num_species, labels, lock)
    if result is None:
        return None
    matrix, num_samples, all_labels = result

    # 保存每个文件的结果到单独的文件
    file_output_file = os.path.join(temp_dir, f"{os.path.basename(file_path)}.npz")
    np.savez(file_output_file, matrix=matrix, **all_labels)
    lock.acquire()
    del matrix
    del all_labels
    gc.collect()
    lock.release()
    return file_path


def combine_npz_files(npz_files, output_dir, batch_size):
    all_matrix = []
    all_labels = {}
    batch_index = 0

    for npz_file in npz_files:
        print(f"Processing file: {npz_file}")
        data = np.load(npz_file, allow_pickle=True)
        all_matrix.extend(data['matrix'])
        for key in data.files:
            if key != 'matrix':
                if key not in all_labels:
                    all_labels[key] = []
                all_labels[key].extend(data[key])

        # # 打印当前all_matrix和all_labels的形状
        # print(f"Current all_matrix shape: {np.shape(all_matrix)}")
        # for key in all_labels:
        #     print(f"Current all_labels[{key}] shape: {np.shape(all_labels[key])}")

        # 当累积的数据量达到 batch_size 时，保存一个批次
        while len(all_matrix) >= batch_size:
            save_batch(all_matrix[:batch_size], {key: all_labels[key][:batch_size] for key in all_labels}, output_dir,
                       batch_index)
            batch_index += 1
            all_matrix = all_matrix[batch_size:]
            for key in all_labels:
                all_labels[key] = all_labels[key][batch_size:]

    # 保存最后一个批次
    if all_matrix:
        save_batch(all_matrix, all_labels, output_dir, batch_index)


def save_batch(matrix_batch, labels_batch, output_dir, batch_index):
    # 打乱顺序
    indices = np.arange(len(matrix_batch))
    np.random.shuffle(indices)
    matrix_batch = np.array(matrix_batch)[indices]
    for key in labels_batch:
        labels_batch[key] = np.array(labels_batch[key])[indices]

    # # 检查matrix_batch的形状
    # if matrix_batch.shape == (64,):
    #     print(f"Warning: matrix_batch shape is (64,) in batch {batch_index}")
    #     print(f"matrix_batch content: {matrix_batch}")
    # 保存批次文件
    batch_file = os.path.join(output_dir, f"batch_{batch_index}.npz")
    np.savez(batch_file, matrix=matrix_batch, **labels_batch)


def fastq_to_npz(root_dir: str, output_dir: str, batch_size: int = 64, max_workers: int = 3):
    sub_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    file_paths = get_file_paths(root_dir)
    all_species, all_species_level = parse_all_files(file_paths)
    num_species = len(all_species)

    level_names = get_all_levels(sub_dirs)

    # 创建临时目录
    temp_dir = os.path.join(os.path.dirname(output_dir), 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # 使用多进程处理
    npz_files = []

    # 加载进度
    processed_files = set()
    progress_file = os.path.join(temp_dir, 'progress.pkl')
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'rb') as pf:
                processed_files = pickle.load(pf)
                for progress_file_name in processed_files:
                    npz_files.append(os.path.join(temp_dir, f"{os.path.basename(progress_file_name)}.npz"))
        except EOFError:
            with open(progress_file, 'wb') as pf:
                pickle.dump(processed_files, pf)

    with Manager() as manager:
        lock = manager.Lock()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            process_func = partial(process_file, all_species_level=all_species_level, num_species=num_species,
                                   level_names=level_names, temp_dir=temp_dir, lock=lock)
            futures = [executor.submit(process_func, file_path) for file_path in file_paths if
                       file_path not in processed_files]

            # 使用 tqdm 显示进度条
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                file_path = future.result()
                if file_path is None:
                    continue
                processed_files.add(file_path)
                file_output_file = os.path.join(temp_dir, f"{os.path.basename(file_path)}.npz")
                npz_files.append(file_output_file)

                # 保存进度
                with open(progress_file, 'wb') as pf:
                    pickle.dump(processed_files, pf)

                gc.collect()
        # 合并所有文件的结果，分批保存
    combine_npz_files(npz_files, output_dir, batch_size)

    with open(os.path.join(output_dir, ALL_SPECIES_NAME), 'wb') as bf:
        pickle.dump(all_species_level, bf)
    with open(os.path.join(output_dir, LEVEL_NAMES), 'wb') as lf:
        pickle.dump(level_names, lf)
