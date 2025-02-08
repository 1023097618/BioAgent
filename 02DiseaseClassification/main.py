import glob
import json
import os
import subprocess
from types import SimpleNamespace

import joblib
import numpy as np
from autogen import register_function, ConversableAgent, AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils import get_openai_api_key
from typing_extensions import Annotated

# 获取OpenAI API密钥
get_openai_api_key()

# LLM配置
# llm_config = {"model": "gpt-4-turbo", "cache_seed": None}
llm_config = {"model": "gpt-4-turbo"}

# 本地命令行代码执行器
code_execution_config = LocalCommandLineCodeExecutor(
    timeout=60,
    work_dir="coding",
    virtual_env_context=SimpleNamespace(bin_path=r'E:\python3.9\paint\Scripts')
)

# 用户代理
user_proxy = ConversableAgent(
    name="user_proxy",
    llm_config=False,
    default_auto_reply="Continue",
    human_input_mode="ALWAYS",
    silent=False
)

executor = UserProxyAgent(
    name="Executor",
    system_message="Executor. Execute the code written by the engineer and report the result.",
    human_input_mode="NEVER",
    code_execution_config={"executor": code_execution_config}
)

# 代码编写助手代理
code_writer_agent = AssistantAgent(
    name="code_writer_agent",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
    system_message="""
You are a coding assistant. Your task is to execute the `perform_prediction_task` function to perform disease prediction using a trained Decision Tree model.

The `perform_prediction_task` function does the following:
1. Loads the necessary data and performs predictions using a trained model.
2. Generates the following outputs:
   - A confusion matrix.
   - Scaled features for dimensionality reduction.
   - Labels for the data points.
   - A label mapping that maps numerical labels to their original class names.
3. Stores these outputs in global variables for later use.

Your role is to:
1. Call the `perform_prediction_task` function to complete the prediction task.
2. Ensure that the outputs are correctly stored in the global variables.

You do not need to generate any code yourself. Simply call the `perform_prediction_task` function and confirm that it has been executed successfully.
    """,
    silent=False
)

# 绘图助手代理
plotter_agent = AssistantAgent(
    name="plotter_agent",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
    system_message="""
You are a plotting assistant. Your task is to implement the `visualize` function and call it with the file path returned by the `code_agent`.

Below is an example implementation of the `visualize` function. Use this as a reference to write the function code.

You must only generate Python code. Do not provide any explanations, comments, or additional text. At the end of the code, you must call the `visualize` function and pass the file path returned by the `code_agent` as an argument.
```python
def visualize(file_path: str):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import numpy as np
    import json
    import os

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    # Read the JSON file
    try:
        with open(file_path, "r") as f:
            output_data = json.load(f)

        # Extract data
        confusion_matrix = output_data["confusion_matrix"]
        scaled_features = np.array(output_data["scaled_features"])
        labels = np.array(output_data["labels"])
        label_mapping = output_data["label_mapping"]

        # Delete the file after reading
        os.remove(file_path)

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return

    # Convert label mapping to a list (ordered)
    tick_labels = [label_mapping[str(i)] for i in range(len(label_mapping))]

    # Plot confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=tick_labels, yticklabels=tick_labels, cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    # Save confusion matrix plot
    confusion_matrix_file = os.path.abspath("confusion_matrix.png")
    plt.savefig(confusion_matrix_file, bbox_inches='tight', format='png')
    plt.show()
    plt.close()
    print(f"Confusion matrix visualization saved to: {confusion_matrix_file}")

    # PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(scaled_features)

    plt.figure(figsize=(8, 8))
    unique_labels = np.unique(labels)
    colors = sns.color_palette("hsv", len(unique_labels))
    for i, label in enumerate(unique_labels):
        original_label = label_mapping[str(label)]
        plt.scatter(
            X_pca[labels == label, 0],
            X_pca[labels == label, 1],
            alpha=0.6,
            s=10,
            label=f"{original_label}",
            color=colors[i]
        )
    plt.title("PCA Projection")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()

    # Save PCA plot
    pca_file = os.path.abspath("pca_projection.png")
    plt.savefig(pca_file, bbox_inches='tight', format='png')
    plt.show()
    plt.close()
    print(f"PCA visualization saved to: {pca_file}")

    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(scaled_features)

    plt.figure(figsize=(8, 8))
    for i, label in enumerate(unique_labels):
        original_label = label_mapping[str(label)]
        plt.scatter(
            X_tsne[labels == label, 0],
            X_tsne[labels == label, 1],
            alpha=0.6,
            s=10,
            label=f"{original_label}",
            color=colors[i]
        )
    plt.title("t-SNE Projection")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()

    # Save t-SNE plot
    tsne_file = os.path.abspath("tsne_projection.png")
    plt.savefig(tsne_file, bbox_inches='tight', format='png')
    plt.show()
    plt.close()
    print(f"t-SNE visualization saved to: {tsne_file}")
# Call the visualize function with the file path returned by code_agent
file_path = "/absolute/path/to/output/file.json"  # Replace this with the actual file path
visualize(file_path)
```
"""
    ,
    silent=False
)

# 默认路径
default_path = "output_npz/"


@executor.register_for_execution()
@code_writer_agent.register_for_llm(
    description="Use a trained Decision Tree model to predict disease outcomes and return the predictions.")
def perform_prediction_task() -> str:
    """
    使用训练好的随机森林模型对指定的 FASTQ 文件进行预测，并返回预测结果。
    """
    # 调用被调用者脚本
    command = [
        r"E:\python3.9\classified\Scripts\python.exe",
        "decisionTree.py",
    ]
    try:
        # 执行脚本并捕获输出
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output_file = result.stdout.strip()  # 获取被调用者输出的文件路径
        return output_file

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")
    except FileNotFoundError:
        print(f"File {output_file} not found.")





def custom_speaker_selection_func(last_speaker, groupchat):
    """Define a customized speaker selection function.
    A recommended way is to define a transition for each speaker in the groupchat.

    Returns:
        Return an `Agent` class or a string from ['auto', 'manual', 'random', 'round_robin'] to select a default method to use.
    """
    messages = groupchat.messages
    rounds = len(messages)
    if rounds == 1:
        return code_writer_agent
    if rounds == 2:
        return executor

    if last_speaker is executor:
        return user_proxy

    if last_speaker is plotter_agent:
        return executor

    if last_speaker is user_proxy:
        return plotter_agent


# 创建群聊，将三个代理加入群聊
groupchat = GroupChat(
    agents=[user_proxy, code_writer_agent, plotter_agent, executor],
    messages=[],
    max_round=500,
    speaker_selection_method=custom_speaker_selection_func,
    enable_clear_history=True,
)

# 创建群聊管理器
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# 启动对话，执行随机森林分类并绘制图表
chat_result = user_proxy.initiate_chat(
    manager,
    message="Please perform disease prediction using the trained Decision Tree model and visualize the results, including the confusion matrix, PCA, and t-SNE plots.",
    max_turns=3
)

# 任务总结助手代理
task_summary_agent = AssistantAgent(
    name="task_summary_agent",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
    system_message="""
You are a task summarization assistant. Your job is to summarize the task that has been completed in the current session.
You will provide a concise and clear explanation of the following:
1. What the task was about.
2. The steps taken to complete the task.
3. The final output or result of the task.

Your summary should be written in a way that is easy to understand and provides a clear overview of the task's purpose and outcome.
You do not need to perform any task yourself; your role is purely to summarize what has already been done.
    """,
    silent=False
)
# 提取 chat_history 并转化为包含 role 和 name 的字符串
# chat_history_str = "\n".join(
#     [
#         f"Message {i} (Role: {message['role']}, Name: {message['name']}): {message['content']}"
#         for i, message in enumerate(chat_result.chat_history)
#     ]
# )


summary = task_summary_agent.generate_reply(
    messages=chat_result.chat_history
)
file_path = "task_summary.txt"
with open(file_path, "w", encoding="utf-8") as file:
    file.write(summary)

print(f"Task summary has been saved to {file_path}")
