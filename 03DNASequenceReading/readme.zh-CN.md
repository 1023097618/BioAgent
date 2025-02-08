做完了项目根目录的环境适配之后，要完成本任务还需要以下的环境适配
1. 下载[ovf](https://huggingface.co/zhishidiannaoka/bioAgent/tree/main/03DNASequenceReading/VirtualMachine)部署到vmware 虚拟机中
2. 使用以下命令获取到虚拟机的ip地址
``` cmd
ip addr
```
3. 将获取到的ip地址写到qiime.bat的第3行
4. 将myenv的Script填入到main.py 20行中
main.py上下文
``` python
code_execution_config = LocalCommandLineCodeExecutor(
    timeout=60,
    work_dir="coding",
    virtual_env_context=SimpleNamespace(bin_path=r'E:\python3.9\paint\Scripts')
)
```