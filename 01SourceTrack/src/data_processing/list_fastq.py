import os

def extract_sequence_ids_and_ecosystem(root_dir):
    """
    从指定目录中提取所有被处理文件的序列号及其父目录名称（生态位）。

    :param root_dir: 数据文件的根目录
    :return: 包含序列号和生态位的列表
    """
    sequence_data = []  # 用于存储序列号和生态位的列表

    # 遍历目录，查找符合条件的文件
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # 检查文件名是否符合条件
            if file.endswith('.tsv') and 'SSU' in file:
                # 提取序列号
                sequence_id = file.split('_')[0]
                # 提取父目录名称（生态位）
                ecosystem = os.path.basename(root)
                # 将序列号和生态位存储为元组
                sequence_data.append((sequence_id, ecosystem))

    # 按序列号排序
    sequence_data = sorted(sequence_data, key=lambda x: x[0])
    return sequence_data


def save_sequence_data(sequence_data, output_file):
    """
    将序列号和生态位信息保存到文件中。

    :param sequence_data: 包含序列号和生态位的列表
    :param output_file: 输出文件路径
    """
    with open(output_file, 'w') as f:
        # 写入标题行
        f.write("Sequence_ID\tEcosystem\n")
        # 写入每一行数据
        for seq_id, ecosystem in sequence_data:
            f.write(f"{seq_id}\t{ecosystem}\n")


if __name__ == "__main__":
    # 数据文件的根目录
    root_dir = "./data"  # 替换为实际的根目录路径

    # 输出文件路径
    output_file = "processed_sequence_data.txt"

    # 提取序列号和生态位信息
    sequence_data = extract_sequence_ids_and_ecosystem(root_dir)

    # 保存序列号和生态位信息到文件
    save_sequence_data(sequence_data, output_file)

    print(f"已处理的序列号和生态位信息已保存到 {output_file}")
