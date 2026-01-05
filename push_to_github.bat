@echo off
echo 正在初始化 Git 仓库并推送到 GitHub...
echo.

echo 1. 初始化 Git 仓库
git init

echo.
echo 2. 添加远程仓库
git remote add origin https://github.com/MarsZhanCZ/wxauto.git

echo.
echo 3. 添加所有文件到暂存区
git add .

echo.
echo 4. 提交更改
git commit -m "Add intelligent WeChat message repeater with DeepSeek AI integration - Added MyRepeaterNew_LLM.py with smart AI reply functionality - Integrated DeepSeek API for natural conversation responses - Added system prompts for better chat experience - Configured to avoid markdown formatting in replies - Limited response length to 100 characters for natural feel - Only processes private messages, filters out group chats - Prevents duplicate replies with message ID tracking"

echo.
echo 5. 设置主分支并推送
git branch -M main
git push -u origin main

echo.
echo 推送完成！
pause