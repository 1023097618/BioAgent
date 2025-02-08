import argparse
import os
import re

import numpy as np
import joblib
import pickle


def load_species_and_levels(all_species_pkl, level_names_pkl):
    """
    加载 all_species 和 level_names 数据。
    """
    with open(all_species_pkl, 'rb') as f:
        all_species_level = pickle.load(f)
    with open(level_names_pkl, 'rb') as f:
        level_names = pickle.load(f)
    return all_species_level, level_names


def parse_fastq_to_npz(fastq_file, all_species_level, num_species, level_names):
    """
    将 FASTQ 文件解析为特征矩阵 (NPZ 格式)。
    """
    with open(fastq_file, 'r') as f:
        for line in f:
            if line.startswith("# Constructed from biom file"):
                continue
            parts = line.strip().split('\t')
            if line.startswith('#'):
                is_sample = [not part.startswith('#SampleID') for part in parts]
                num_sample = is_sample.count(True)
                total_abundance = np.zeros(num_sample, dtype=float)
                matrices = np.zeros((num_sample, num_species, len(all_species_level)))
                total_abundance.fill(0)
                matrices.fill(0)
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
        matrices = np.array(
            [matrices[i] / total if total > 0 else matrices[i] for i, total in enumerate(total_abundance)])
    return matrices


def predict_with_decision_tree(fastq_files, model_files, all_species_pkl, level_names_pkl, threshold=0.0):
    """
    使用训练好的决策树模型对 FASTQ 文件进行预测。
    """
    # 加载 all_species 和 level_names
    all_species_level, level_names = load_species_and_levels(all_species_pkl, level_names_pkl)

    # 确定物种数量
    num_species = max(max(species.values()) for species in all_species_level.values())[0] + 1

    predictions = []

    for fastq_file in fastq_files:
        # 将 FASTQ 文件转换为特征矩阵
        x_data = parse_fastq_to_npz(fastq_file, all_species_level, num_species, level_names)

        # 展平特征矩阵
        x_data = x_data.reshape(x_data.shape[0], -1)

        # 加载每个标签的模型并进行预测
        file_predictions = []
        file_mapped_predictions = []
        for model_file in model_files:
            # 加载模型
            rf_model = joblib.load(model_file)

            # 使用模型进行预测
            probabilities = rf_model.predict_proba(x_data)
            tree_label = rf_model.classes_
            file_predictions.append(probabilities)

            # 将预测结果映射到 level_names
            mapped_names = []
            label_index = model_files.index(model_file)  # 当前模型对应的标签索引
            for sample_idx, sample_probs in enumerate(probabilities):
                layer_results = {}
                if label_index in level_names:
                    level_names_lst = sorted(level_names[label_index])
                    for class_idx, prob in enumerate(sample_probs):
                        level_names_idx = tree_label[class_idx]
                        if prob >= threshold:
                            if level_names_idx < len(level_names_lst):
                                layer_results[level_names_lst[level_names_idx]] = prob
                            else:
                                layer_results["unknown"] = prob
                mapped_names.append(layer_results)
            file_mapped_predictions.append(mapped_names)

        formatted_predictions = []
        for sample_idx in range(len(x_data)):
            sample_prediction = []
            for layer_idx, layer_predictions in enumerate(file_mapped_predictions):
                sample_prediction.append(layer_predictions[sample_idx])
            formatted_predictions.append(sample_prediction)
        predictions.append(formatted_predictions)

    return predictions

def get_model_files(model_dir="decisionModel"):
    """
    动态获取 decisionModel 目录下的所有模型文件，并按 label_X 的 X 值排序。
    """
    model_files = []
    pattern = re.compile(r"label_(\d+)_random_forest_model\.pkl")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' not found.")

    for filename in os.listdir(model_dir):
        match = pattern.match(filename)
        if match:
            label_index = int(match.group(1))
            model_files.append((label_index, os.path.join(model_dir, filename)))

    # 按 label_X 的 X 值排序
    model_files.sort(key=lambda x: x[0])

    # 只返回文件路径
    return [file[1] for file in model_files]


def getdata(fastq_files):
    model_files = get_model_files("decisionModel")
    all_species_pkl = "output_npz/all_species.pkl"
    level_names_pkl = "output_npz/level_names.pkl"

    # 进行预测
    predictions = predict_with_decision_tree(fastq_files, model_files, all_species_pkl, level_names_pkl, threshold=0.0)

    # 输出预测结果
    result_str = ""
    for i, file_predictions in enumerate(predictions):
        result_str += f"Predictions for file {fastq_files[i]}:\n"
        for sample_idx, sample_prediction in enumerate(file_predictions):
            if sample_idx == 0:
                result_str += f"Sample {sample_idx + 1}: {sample_prediction}\n"
    return result_str


# result = getdata(["data/root-Engineered-Biogas plant-Wet fermentation/ERP106205_taxonomy_abundances_SSU_v4.1.tsv"])
# print(result)
if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Predict with decision tree models.")
    parser.add_argument("fastq_file", help="Path to the FASTQ file for prediction.", nargs='+', type=str)
    args = parser.parse_args()

    # 调用 getdata 函数并打印结果
    print(args.fastq_file)
    result = getdata(args.fastq_file)
    # result = getdata(["data/root-Engineered-Built environment/ERP114086_taxonomy_abundances_SSU_v4.1.tsv"])
    print(result)
