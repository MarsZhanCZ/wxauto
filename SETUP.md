# 微信智能回复机器人设置指南

## 环境配置

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置 API Key
1. 复制 `.env.example` 文件为 `.env`
2. 在 `.env` 文件中填入你的 DeepSeek API Key：
   ```
   DEEPSEEK_API_KEY=你的真实API_Key
   ```

### 3. 运行机器人
```bash
python MyRepeaterNew_LLM.py
```

## 注意事项
- 确保微信已登录并正常运行
- 建议使用微信 3.9.X 版本
- `.env` 文件已被添加到 `.gitignore`，不会被提交到 git 仓库
- 机器人只会回复私聊消息，不会回复群聊消息

## 自定义配置
```python
from MyRepeaterNew_LLM import MessageRepeater

# 使用自定义系统提示词
custom_prompt = "你是一个专业的客服助手..."
repeater = MessageRepeater(system_prompt=custom_prompt)
repeater.run()
```