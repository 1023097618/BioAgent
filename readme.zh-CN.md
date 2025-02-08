要想运行这个程序，首先需要对环境进行以下配置
1. 在每个任务领域的`.env`中输入相关的[密钥文件](https://platform.openai.com/docs/overview)
2. 使用以下命令对环境进行配置，这就创建了一个名为`myenv`的虚拟环境，此处python的版本需要为3.12
``` cmd
python -m venv myenv
myenv\Scripts\activate
pip install -r myenv_python3.12.txt
```
3. 然后到[huggingface](https://huggingface.co/zhishidiannaoka/bioAgent/tree/main)上选择下载相关需要的文件，huggingface的文件目录与此项目的保持一致，所以huggingface上面相关文件放到哪你就下载到哪里就行。
4. 然后在`myenv`环境下运行每个任务下方的`main.py`就可以啦。