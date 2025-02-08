# Environment Setup Instructions

To run this program, you need to configure the environment as follows:

1. Add Secret Keys to `.env` Files  
For each task domain, enter the relevant [secret key](https://platform.openai.com/docs/overview) files into the corresponding `.env` file.

2. Set Up the Virtual Environment  
Use the following commands to configure the environment. This will create a virtual environment named `myenv`. Ensure that Python version **3.12** is used.

```cmd
python -m venv myenv
myenv\Scripts\activate
pip install -r myenv_python3.12.txt
```
3. Download Required Files from Hugging Face

Go to [Hugging Face](https://huggingface.co/zhishidiannaoka/bioAgent/tree/main) and download the necessary files. The directory structure on Hugging Face matches this project's structure, so download the files into the corresponding locations.

4. Run main.py for Each Task

Once the environment is set up, activate the myenv environment and run the main.py file for each task.