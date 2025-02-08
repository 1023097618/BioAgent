# -*- coding: UTF-8 -*-
import os
import numpy as np
import yaml
from src.config.constants import LEVEL_NAMES, ALL_SPECIES_NAME


def npz_to_yaml(input_dir, output_file):
    # 获取所有batch文件
    batch_files = [
        f for f in os.listdir(input_dir)
        if f.startswith('batch_') and f.endswith('.npz') and f[6:-4].isdigit() and int(f[6:-4]) >= 0
    ]
    if not batch_files:
        raise ValueError("No batch files found in the input directory.")

    # 加载第一个batch文件以获取矩阵和标签信息
    first_batch_file = os.path.join(input_dir, batch_files[0])
    data = np.load(first_batch_file)
    matrix = data['matrix']
    labels = {key: data[key] for key in data.files if key.startswith('label_')}

    # 计算feature_nums
    num_samples, feature_nums = matrix.shape[0], matrix.shape[1] * matrix.shape[2]

    # 获取label维度数
    label_dims = {key: value.shape[1] for key, value in labels.items()}

    # 计算batch_size
    batch_size = matrix.shape[0]

    # 生成yaml内容
    yaml_content = {
        "base_dir": "data/model",
        "data": {
            "npz_dir": input_dir,
            "model_parameter_file": output_file
        },
        "model": {
            "is_train": False,
            "feature_nums": feature_nums,
            "load_train_data": True
        },
        "train": {
            "batch_total_num": len(batch_files),
            "batch_size": batch_size,
            "learning_rate": 0.00001,
            "global_step": 0,
            "epochs": 30000,
            "min_delta": 0.0001,
            "save_to_memory": True,
            "train_test_split":0.7,
            'patience':10,
            'save_per_epoch':3
        }
    }

    # 添加label维度数到model部分
    yaml_content["model"].update(label_dims)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as file:
        yaml.dump(yaml_content, file, default_flow_style=False)

# 示例调用
# npz_to_yaml('path/to/input_dir', 'path/to/output_dir')
