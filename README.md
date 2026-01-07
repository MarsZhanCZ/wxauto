# 微信智能回复机器人

基于 wxauto 和 LangChain 的微信消息智能回复机器人，支持 RAG 知识库检索和联网搜索功能。

## ✨ 功能特性

- 🤖 **智能回复**：基于 DeepSeek AI 模型的自然语言对话
- 📚 **RAG 知识库**：支持本地文档检索，提供准确的业务信息
- 🌐 **联网搜索**：集成百度搜索，获取实时信息
- 💬 **多场景支持**：支持私聊和群聊@回复
- 🔄 **消息去重**：防止重复处理同一消息
- 📊 **多格式文档**：支持 Excel QA 对和文本文件

## 🚀 快速开始

### 1. 环境准备

#### 微信版本要求
- **必须使用微信 3.9.X 版本**
- 64位系统默认只能安装4.X版本，需要使用兼容性启动器：
  - 下载：[WeChat 3.9 32bit 兼容性启动器](https://github.com/Skyler1n/WeChat3.9-32bit-Compatibility-Launcher)
  - 安装微信 3.9.X 版本并保持登录状态

#### Python 环境
```bash
# 创建虚拟环境（推荐）
conda create -n wxauto python=3.10
conda activate wxauto

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置设置

#### 环境变量配置
复制 `.env.example` 为 `.env` 并填入你的 API Keys：

```bash
cp .env.example .env
```

编辑 `.env` 文件：
```env
# DeepSeek API 配置（必需）
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# SiliconFlow Embedding API 配置（RAG功能必需）
SILICONFLOW_API_KEY=your_siliconflow_api_key_here

# 百度搜索 API 配置（联网搜索功能，可选）
BAIDU_SEARCH_API_KEY=your_baidu_search_api_key_here
```

#### API Key 获取方式
- **DeepSeek API**：访问 [DeepSeek 开放平台](https://platform.deepseek.com/) 注册获取
- **SiliconFlow API**：访问 [SiliconFlow](https://siliconflow.cn/) 注册获取
- **百度搜索 API**：访问 [百度智能云](https://cloud.baidu.com/) 获取千帆大模型API

### 3. 知识库准备

将你的文档放入 `raw_docs` 目录：

```
raw_docs/
├── 智云上海QA对.xlsx     # Excel格式的问答对
├── 智云上海简介          # 纯文本文件
├── 产品手册.xlsx         # 更多Excel文件
└── 服务指南.txt          # 更多文本文件
```

**支持的文件格式：**
- **Excel文件** (`.xlsx`, `.xls`)：自动识别问题/答案列
- **文本文件**：任意纯文本文件，自动分块处理

### 4. 运行机器人

#### 基础版本（仅 RAG 知识库）
```bash
# 首次运行（构建知识库索引）
python MyRepeaterNew_Agent.py --rebuild

# 后续运行
python MyRepeaterNew_Agent.py

# 支持群聊@回复（需要指定昵称）
python MyRepeaterNew_Agent.py --nickname "你的微信昵称"
```

#### 增强版本（RAG + 联网搜索）
```bash
# 双工具版本，支持知识库检索和联网搜索
python MyRepeaterNew_AgentDouble.py --nickname "你的微信昵称"
```

#### 命令行参数
- `--rebuild`：重新构建知识库索引
- `--debug`：开启调试模式
- `--nickname "昵称"`：指定微信昵称，用于群聊@检测

## 📖 版本说明

| 文件 | 功能 | 适用场景 |
|------|------|----------|
| `MyRepeaterNew_LLM.py` | 基础版，仅LLM对话 | 简单聊天机器人 |
| `MyRepeaterNew_RAG.py` | RAG版，知识库检索 | 企业客服、文档问答 |
| `MyRepeaterNew_Agent.py` | Agent版，智能工具调用 | 灵活的AI助手 |
| `MyRepeaterNew_AgentDouble.py` | 双工具版，RAG+联网搜索 | 全功能智能助手 |

## 🔧 高级配置

### 自定义系统提示词
```python
from MyRepeaterNew_Agent import MessageRepeaterAgent

custom_prompt = """你是一个专业的客服助手，请遵循以下原则：
1. 始终保持礼貌和专业
2. 快速准确地回答用户问题
3. 如果无法解决问题，引导用户联系人工客服
请用中文回复。"""

repeater = MessageRepeaterAgent(system_prompt=custom_prompt)
repeater.run()
```

### 知识库文档格式要求

#### Excel QA 对文件
Excel 文件应包含问题和答案两列，支持的列名：
- 问题列：`Q`, `问题`, `问`
- 答案列：`A`, `答案`, `答`, `回答`

示例：
| 问题 | 答案 |
|------|------|
| 智云上海是什么？ | 智云上海是中国电信与上海市政府合作打造的... |
| 如何开通服务？ | 您可以通过以下方式开通服务... |

#### 文本文件
纯文本文件会被自动分割成多个文档块进行索引，支持各种格式的文档内容。

## 🛠️ 开发文档

- **wxauto 库文档**：查看 `docs/` 目录下的详细文档
- **API 参考**：`docs/class/` 目录包含各个类的详细说明
- **使用示例**：`docs/example.md` 提供了完整的使用示例

## 📝 使用说明

### 消息处理规则
- ✅ **私聊消息**：自动回复所有好友私聊消息
- ✅ **群聊@消息**：只回复群聊中@你的消息（需设置昵称）
- ❌ **系统消息**：自动过滤系统通知、时间消息等
- ❌ **重复消息**：60秒内相同内容不重复处理

### 工具调用逻辑（Agent版本）
1. **知识库优先**：企业相关问题优先搜索本地知识库
2. **联网补充**：实时信息、新闻、天气等使用联网搜索
3. **智能判断**：Agent 自动决定是否需要使用工具

### 回复风格特点
- 🗣️ **自然对话**：像真人聊天一样亲切自然
- 📏 **简洁明了**：回复控制在100字以内
- 🚫 **无格式化**：不使用 Markdown 格式，适合微信聊天
- 😊 **表情符号**：适当使用表情让对话更生动

## ⚠️ 注意事项

1. **微信版本**：必须使用 3.9.X 版本，4.X 版本不兼容
2. **网络连接**：需要稳定的网络连接调用 API
3. **API 配额**：注意各个 API 的调用限制和费用
4. **合规使用**：请遵守相关法律法规，仅用于合法用途
5. **数据安全**：`.env` 文件包含敏感信息，不要提交到版本控制

## 🐛 故障排除

### 常见问题

**Q: 程序启动后没有反应？**
A: 检查微信是否已登录，版本是否为 3.9.X

**Q: 提示找不到微信窗口？**
A: 确保微信已打开并完全加载，尝试重启微信

**Q: RAG 搜索没有结果？**
A: 检查 `raw_docs` 目录是否有文档，运行时加 `--rebuild` 参数

**Q: 群聊@消息没有回复？**
A: 确保使用了 `--nickname` 参数指定正确的微信昵称

**Q: API 调用失败？**
A: 检查 `.env` 文件中的 API Key 是否正确配置

### 调试模式
```bash
# 开启调试模式查看详细日志
python MyRepeaterNew_Agent.py --debug --nickname "你的昵称"
```

## 📄 许可证

本项目仅用于学习和研究目的，请勿用于商业用途。使用时请遵守相关法律法规。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

---

**免责声明**：本项目仅用于技术学习和交流，请勿用于非法用途和商业用途。如因使用本项目产生任何法律纠纷，均与作者无关。