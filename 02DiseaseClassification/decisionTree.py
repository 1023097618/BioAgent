import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch._prims import le
from sklearn.metrics import classification_report

plt.rcParams.update({'font.size':20,
                     'font.family': 'serif',
                     'font.serif': ['Times New Roman']
                     })

# 读取数据
otu_table = pd.read_csv(r'data\D003424_Crohn_Disease\genus\otu_table.xls',sep='\t')
sample_data = pd.read_csv(r'data\D003424_Crohn_Disease\genus\sample_data.mf.xls',sep='\t')
# 转置 OTU 表格，使样本 ID 成为行索引
otu_table = otu_table.set_index('featureid').T.reset_index()
otu_table.rename(columns={'index': 'SampleID'}, inplace=True)
sample_data.rename(columns={'#SampleID':'SampleID'},inplace=True)
# 合并 OTU 表格和样本数据
merged_data = pd.merge(otu_table, sample_data, on='SampleID', how='inner')

# 检查数据
# print(merged_data.head())

# 编码疾病 ID
label_encoder = LabelEncoder()
merged_data['Disease.MESH.ID'] = label_encoder.fit_transform(merged_data['Disease.MESH.ID'])

# 分离特征和标签
X = merged_data.drop(columns=['SampleID', 'Disease.MESH.ID'])
y = merged_data['Disease.MESH.ID']
# print(f"sample num:{X.shape[0]}\nspecies num:{X.shape[1]}")
# 标准化特征值
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



# 将 y_train 转换为 NumPy 数组，避免索引问题
y_train = y_train.reset_index(drop=True)  # 重置索引
y_train = y_train.values  # 转换为 NumPy 数组


clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 评估模型
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
output_data = {
    "confusion_matrix": cm.tolist(),
    "scaled_features": X_scaled.tolist(),
    "labels": y.tolist(),  # 添加标签信息
    "label_mapping": {i: label for i, label in enumerate(label_encoder.classes_)}  # 添加标签映射
}

# 保存到文件
output_file = os.path.abspath("output_data.json")  # 获取绝对路径
with open(output_file, "w") as f:
    json.dump(output_data, f)

# 打印文件的绝对路径
print(output_file)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('Confusion_Matrix.svg', bbox_inches='tight', format='svg', transparent=True)  # 保存为 PNG 图片
plt.show()

# 使用 PCA 降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 使用 t-SNE 降维
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

# 可视化函数
def plot_2d_projection(data, labels, title, filename, label_encoder):
    unique_labels = np.unique(labels)
    colors = sns.color_palette("hsv", len(unique_labels))
    plt.figure(figsize=(8, 8))
    for i, label in enumerate(unique_labels):
        # 使用 label_encoder.inverse_transform 将数字标签映射回原始标签
        original_label = label_encoder.inverse_transform([label])[0]
        plt.scatter(data[labels == label, 0], data[labels == label, 1],
                    label=f"{original_label}", alpha=0.6, s=10, color=colors[i])
    plt.title(title)
    plt.legend()
    plt.savefig(filename, bbox_inches='tight', format='svg', transparent=True)  # 保存为 SVG 图片
    plt.show()

# 绘制 PCA 和 t-SNE 的结果，并分别保存为 SVG 图片
plot_2d_projection(X_pca, y.values, "PCA Projection", "PCA_Projection.svg", label_encoder)
plot_2d_projection(X_tsne, y.values, "t-SNE Projection", "tSNE_Projection.svg", label_encoder)
