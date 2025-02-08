做完了项目根目录的环境适配之后，要完成本任务还需要以下的环境适配
1. 使用以下命令创建一个名为`classified`的虚拟环境,此处python的版本需要是3.9
``` cmd
python -m venv classified
classified\Scripts\activate
pip install -r classified_python3.9.txt
```

2. 在`classified`环境下，查看`python`解释器的绝对路径，并且将它填入`main.py` 215行中
查看绝对路径
``` cmd
where python
```
main.py上下文
``` python
    command = [
        r"E:\python3.9\classified\Scripts\python.exe",
        "decisionTree.py",
    ]
```

3. 将myenv的Script填入到main.py 29行中
main.py上下文
``` python
code_execution_config = LocalCommandLineCodeExecutor(
    timeout=60,
    work_dir="coding",
    virtual_env_context=SimpleNamespace(bin_path=r'E:\python\myenv\Scripts')
)
```