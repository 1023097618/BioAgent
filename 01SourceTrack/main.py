import glob
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
from decitiontree import getdata
from typing_extensions import Annotated

# 获取OpenAI API密钥
get_openai_api_key()

# LLM配置
llm_config = {"model": "gpt-4-turbo", "cache_seed": None}
# llm_config = {"model": "gpt-4-turbo"}

# 本地命令行代码执行器
code_execution_config = LocalCommandLineCodeExecutor(
    timeout=60,
    work_dir="coding",
    virtual_env_context=SimpleNamespace(bin_path=r'E:\python\myenv\Scripts')
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
    #     system_message="""
    # You are a coding assistant. Your task is to help generate Python code for machine learning tasks.
    # You will load data from the specified directory, train a RandomForest classifier, save the trained models,
    # and use the trained models to perform predictions on specific samples.
    #
    # You have access to the following two registered functions:
    #
    # 1. `perform_random_forest_classification`:
    #     - This function performs RandomForest classification on the loaded data.
    #     - It splits the data into training and testing sets, trains a RandomForest classifier for each label,
    #       and calculates the accuracy for each label.
    #     - The function saves the trained models to disk and returns a summary of the accuracies for each label.
    #
    # 2. `perform_prediction_task`:
    #     - This function uses the trained RandomForest models to predict on specific samples.
    #     - It takes as input a list of file paths (e.g., FASTQ files) and returns the prediction results.
    #     - The prediction results are formatted as a list of tuples for each sample, where each tuple contains a label
    #       and its corresponding confidence score.
    #
    # Your job is to call these functions as needed to complete the tasks requested by the user.
    # You do not need to implement these functions; they are already provided.
    #     """,
    system_message="""
You are a coding assistant. Your task is to help generate Python code for machine learning tasks.
You will use the trained models to perform predictions on specific samples.

You have access to the following one registered functions:


1. `perform_prediction_task`:
    - This function uses the trained RandomForest models to predict on specific samples.
    - It takes as input a list of file paths (e.g., FASTQ files) and returns the prediction results.
    - The prediction results are formatted as a list of tuples for each sample, where each tuple contains a label 
      and its corresponding confidence score.

Your job is to call these functions as needed to complete the tasks requested by the user. 
You do not need to implement these functions; they are already provided.
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
You are a plotting assistant. Your task is to generate Python code to create plots based on the data provided.
You will use the 'matplotlib' library to generate plots. Specifically, you will visualize the prediction results for specific samples.

The prediction results will be provided in the following format:
    Sample 1: [{'label1': confidence1}, {'label2': confidence2}, {'label3': confidence3, 'label4': confidence4}, ...]
    Sample 2: [{'label1': confidence1}, {'label2': confidence2}, {'label3': confidence3, 'label4': confidence4}, ...]
    ...
Each sample contains a list of dictionaries, where each dictionary represents a hierarchical level of labels and their corresponding confidence scores.
specifically,there are 7 layers each sample
Your task is to create a pie chart for a specific sample (e.g., Sample 1) that visualizes the confidence scores for each hierarchical level. 
For each level, create a separate pie chart, where the slices represent the confidence scores of the labels at that level. 
Save all pie charts as PNG files, with filenames indicating the sample and the level (e.g., 'Sample1_Level1.png', 'Sample1_Level2.png', etc.).
For the specific sample, you should contain all the level in your code.
by default,you should not limit the display of the label and the percentage.
but if user required,you can limit the display of the label and the percentage which is less than a specific figure by this way
```python
        for key, value in level_data.items():
            sizes.append(value)
            if value >= 0.05:
                labels.append(f'{key} ({value*100:.1f}%)')
                autopct_list.append('%1.1f%%')
            else:
                labels.append('')
                autopct_list.append('')
        ax1.pie(sizes, labels=labels, autopct=lambda pct: f'{pct:.1f}%' if pct >= 5 else '', startangle=90)
```
    """,
    silent=False
)

# 默认路径
default_path = "output_npz/"


# 注册函数：加载数据并返回字符串
# @user_proxy.register_for_execution()
# @code_writer_agent.register_for_llm(description="Load all data from the specified directory and return the shape of each array as a string.")
def meta_load():
    def load_data(file_pattern):
        data = {}
        files = glob.glob(file_pattern)
        for file in files:
            batch_data = np.load(file, allow_pickle=True)
            for key in batch_data:
                if key not in data:
                    data[key] = []
                data[key].append(batch_data[key])
        for key in data:
            data[key] = np.concatenate(data[key], axis=0)
        print('load all,no memory problem')
        return data

    return load_data('output_npz/batch_*.npz')


@executor.register_for_execution()
@code_writer_agent.register_for_llm(
    description="Use trained RandomForest models to predict on FASTQ files and return the predictions.")
def perform_prediction_task():
    """
    使用训练好的随机森林模型对指定的 FASTQ 文件进行预测，并返回预测结果。
    """
    # 调用 getdata 函数执行预测任务
    fastq_file = ["data/root-Environmental-Aquatic-Marine-Pelagic/SRP064135_taxonomy_abundances_SSU_v5.0.tsv"]
    command = [
        r"E:\python3.9\myenv1\Scripts\python.exe",  # 调用 Python 解释器
        "decitiontree.py",  # 指定脚本文件名
        # 传入 FASTQ 文件路径作为参数
        *fastq_file
    ]
    try:
        # 使用 subprocess 调用命令行
        print(command)
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        # 返回命令行输出结果
        return result.stdout
    except subprocess.CalledProcessError as e:
        # 捕获错误并返回错误信息
        return f"Error occurred while running the prediction task: {e.stderr}"


# # 启动对话，调用数据加载工具
# chat_result = user_proxy.initiate_chat(
#     code_writer_agent,
#     message="Please perform RandomForest classification on the loaded data and return the accuracies for each label.",
#     max_turns=2
# )
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
    message="Please perform RandomForest classification, use the trained models to predict on specific samples, and plot the prediction results for Sample 1 as pie charts for each hierarchical level.",
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
