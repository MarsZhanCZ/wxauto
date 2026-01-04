# 微信智能回复机器人设置指南

## 环境配置

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置 API Key
1. 复制 `.env.example` 文件为 `.env`
2. 在 `.env` 文件中填入你的 API Key：
   ```
   DEEPSEEK_API_KEY=你的DeepSeek_API_Key
   SILICONFLOW_API_KEY=你的SiliconFlow_API_Key
   ```

### 3. 准备知识库文档（RAG版本）
将你的文档放入 `raw_docs` 目录：
- 支持 `.xlsx` / `.xls` 格式的 QA 对文件
- 支持纯文本文件

### 4. 运行机器人

#### 基础版（无RAG）
```bash
python MyRepeaterNew_LLM.py
```

#### RAG增强版
```bash
# 首次运行（构建知识库索引）
python MyRepeaterNew_RAG.py --rebuild

# 后续运行（使用已有索引）
python MyRepeaterNew_RAG.py

# 指定昵称以精确检测@我的消息
python MyRepeaterNew_RAG.py --nickname "你的微信昵称"

# 调试模式
python MyRepeaterNew_RAG.py --debug
```

## 注意事项
- 确保微信已登录并正常运行
- 建议使用微信 3.9.X 版本
- `.env` 文件已被添加到 `.gitignore`，不会被提交到 git 仓库
- 机器人只会回复好友私聊消息和群聊中@我的消息
- RAG版本使用 SiliconFlow 的 BGE-M3 模型进行向量化
- 建议使用 --nickname 参数指定昵称以精确检测@消息

## 自定义配置
```python
from MyRepeaterNew_RAG import MessageRepeaterRAG

# 重新构建知识库索引
repeater = MessageRepeaterRAG(rebuild_index=True)
repeater.run()
```

## 知识库文件格式

### Excel QA 对文件
Excel 文件应包含问题和答案两列，列名可以是：
- Q / A
- 问题 / 答案
- 问 / 答

### 文本文件
纯文本文件会被自动分割成多个文档块进行索引。