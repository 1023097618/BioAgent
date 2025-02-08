import subprocess
from types import SimpleNamespace

from autogen import ConversableAgent, AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor

from utils import get_openai_api_key

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
    By default, execute the `perform_qiime` function. If there are specific requirements, modify the code based on the following instructions.
    Running the `perform_qiime` function will invoke a batch command. The batch command is as follows:
    ```bat
    @echo off
    chcp 65001
    set SERVER_IP=192.168.29.130
    set USERNAME=root
    set REMOTE_COMMAND="bash -i -c 'cd /home/user/Desktop; source qiimestart.sh; cd /home/user/Desktop/qiime2Study; source clean.sh; source process.sh; source visualize.sh;'"
    set REMOTE_FOLDER=/home/user/Desktop/qiime2Study/web
    set LOCAL_PATH=%cd%
    
    ssh -t %USERNAME%@%SERVER_IP% %REMOTE_COMMAND%
    
    scp -r %USERNAME%@%SERVER_IP%:%REMOTE_FOLDER% %LOCAL_PATH%
    
    echo finish
    ```
    
    The content of `clean.sh` is as follows:
    ```sh
    rm -rf ./results/*
    rm -rf ./taxonomy/*
    rm -rf ./visualize/*
    rm -rf ./web/*
    ```
    
    The content of `process.sh` is as follows:
    ```sh
    qiime tools import \
      --type 'SampleData[PairedEndSequencesWithQuality]' \
      --input-path sample_data/manifest.txt  \
      --output-path results/paired-end-demux.qza \
      --input-format PairedEndFastqManifestPhred33
  
    qiime dada2 denoise-paired \
      --i-demultiplexed-seqs results/paired-end-demux.qza \
      --p-trunc-len-f 220 \
      --p-trunc-len-r 220 \
      --p-trim-left-f 0 \
      --p-trim-left-r 0 \
      --p-trunc-q  5 \
      --p-chimera-method pooled \
      --o-table results/table.qza \
      --o-representative-sequences results/rep-seqs.qza \
      --o-denoising-stats results/denoising-stats
      
    qiime feature-classifier classify-sklearn \
      --i-classifier sample_data/gg-13-8-99-nb-classifier.qza \
      --i-reads results/rep-seqs.qza \
      --o-classification results/taxonomy.qza
      
    qiime tools export  --input-path results/taxonomy.qza  --output-path taxonomy
    ```
    
    The content of `visualize.sh` is as follows:
    ```sh
    qiime taxa barplot \
      --i-table results/table.qza \
      --i-taxonomy results/taxonomy.qza \
      --m-metadata-file sample_data/mapping.txt \
      --o-visualization visualize/taxa-bar-plots.qzv
  
    qiime tools extract \
      --input-path visualize/taxa-bar-plots.qzv \
      --output-path web
    ```
    """,
    silent=False
)

working_dir = r"E:\mypython\MySourceTrack\ssh"


@executor.register_for_execution()
@code_writer_agent.register_for_llm(
    description="This function will invoke the `qiime.bat` script, which connects to a remote server, preprocesses DNA sequencing data, classifies it into an OTU abundance matrix, and downloads the visualization results locally.")
def perform_qiime():
    command = [
        "qiime.bat"
    ]
    result = subprocess.run(command, capture_output=True, check=True, text=True, cwd=working_dir)
    print(f"visualize result saved to {working_dir}")

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

    if last_speaker is code_writer_agent:
        return executor

    if last_speaker is user_proxy:
        return code_writer_agent


# 创建群聊，将三个代理加入群聊
groupchat = GroupChat(
    agents=[user_proxy, code_writer_agent, executor],
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
    message="Convert DNA sequencing data into an OTU abundance matrix",
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

summary = task_summary_agent.generate_reply(
    messages=chat_result.chat_history
)
file_path = "task_summary.txt"
with open(file_path, "w", encoding="utf-8") as file:
    file.write(summary)

print(f"Task summary has been saved to {file_path}")
