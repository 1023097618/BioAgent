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
