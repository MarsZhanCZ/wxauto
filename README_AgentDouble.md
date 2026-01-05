# 微信机器人双工具版本 - 修改说明

## 概述

`MyRepeaterNew_AgentDouble.py` 是在原有 `MyRepeaterNew_Agent.py` 基础上的增强版本，新增了百度联网搜索功能，使机器人既能查询本地知识库，也能获取实时网络信息。

## 主要修改内容

### 1. 新增百度搜索工具

#### 工具定义
```python
@tool
def baidu_search_tool(query: str) -> str:
    """
    使用百度搜索API进行联网搜索，获取最新的网络信息。
    
    当用户询问需要实时信息、最新新闻、当前事件、股价、天气、最新技术动态等问题时，
    或者本地知识库无法回答的问题时，使用此工具进行联网搜索。
    """
```

#### 核心功能
- 调用百度千帆AI搜索API
- 支持最近一个月的搜索结果过滤
- 完善的错误处理和超时机制
- 智能提取搜索结果内容

### 2. 环境变量配置

在 `.env.example` 中新增：
```bash
# 百度搜索 API 配置
# 用于联网搜索功能
BAIDU_SEARCH_API_KEY=your_baidu_search_api_key_here
```

### 3. 工具集成策略

#### 双工具架构
- **本地知识库搜索** (`search_knowledge_base`): 优先用于智云上海相关问题
- **百度联网搜索** (`baidu_search_tool`): 用于实时信息和本地知识库无法回答的问题

#### 智能工具选择
Agent会根据用户问题的性质自动选择合适的工具：

**使用本地知识库的场景：**
- 智云上海公司相关问题
- 智家服务介绍
- 算力服务咨询
- AI服务说明
- 城市智能化解决方案

**使用联网搜索的场景：**
- 实时新闻和事件
- 天气查询
- 股价信息
- 最新技术动态
- 当前时事
- 本地知识库无法回答的问题

### 4. 系统提示词优化

更新了系统提示词，明确了两个工具的使用策略：

```python
self.system_prompt = """你是一个友好的微信聊天助手，拥有两个强大的工具：

1. search_knowledge_base: 搜索智云上海相关的本地知识库
2. baidu_search_tool: 进行联网搜索获取最新信息

**工具使用策略：**
- 当用户询问智云上海、智家服务、算力、AI服务等公司相关问题时，优先使用 search_knowledge_base
- 当用户询问实时信息、最新新闻、天气、股价、当前事件等需要联网的问题时，使用 baidu_search_tool
- 如果本地知识库没有找到相关信息，可以尝试联网搜索
- 对于日常闲聊、问候等简单问题，可以直接回复，无需使用工具
"""
```

### 5. 增强的日志输出

新增了工具使用情况的日志显示：
- 显示使用了哪些工具（知识库、联网搜索）
- 区分直接回复和工具辅助回复
- 更清晰的执行过程追踪

## 使用方法

### 1. 环境配置

复制 `.env.example` 为 `.env` 并配置API密钥：

```bash
# DeepSeek API 配置
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# SiliconFlow Embedding API 配置
SILICONFLOW_API_KEY=your_siliconflow_api_key_here

# 百度搜索 API 配置（新增）
BAIDU_SEARCH_API_KEY=your_baidu_search_api_key_here
```

### 2. 运行程序

```bash
# 基本运行
python MyRepeaterNew_AgentDouble.py

# 重新构建知识库索引
python MyRepeaterNew_AgentDouble.py --rebuild

# 开启调试模式
python MyRepeaterNew_AgentDouble.py --debug

# 设置微信昵称（用于群聊@检测）
python MyRepeaterNew_AgentDouble.py --nickname "你的微信昵称"
```

### 3. 功能特性

#### 智能工具选择
- Agent会根据问题内容自动选择最合适的工具
- 支持工具链调用（先查本地，再联网搜索）
- 对于简单问候无需使用工具，直接回复

#### 容错机制
- 百度搜索API未配置时，仍可正常使用知识库功能
- 网络请求超时和异常处理
- 搜索失败时的友好提示

#### 回复质量
- 保持自然的聊天风格
- 控制回复长度在100字以内
- 避免markdown格式，适合微信聊天

## 技术架构

### 工具调用流程

1. **消息接收**: 微信消息监听和过滤
2. **Agent处理**: LangChain Agent分析用户意图
3. **工具选择**: 根据问题类型选择合适工具
4. **工具执行**: 
   - 本地知识库搜索 → FAISS向量检索
   - 联网搜索 → 百度千帆API调用
5. **结果整合**: Agent整合工具结果生成回复
6. **消息发送**: 通过微信API发送回复

### 依赖库

新增依赖：
- `requests`: HTTP请求库，用于调用百度搜索API
- `json`: JSON数据处理

其他依赖保持不变：
- `wxauto`: 微信自动化
- `langchain`: Agent框架
- `faiss`: 向量检索
- `pandas`: Excel文件处理

## 注意事项

1. **API密钥安全**: 请妥善保管各种API密钥，不要提交到版本控制系统
2. **搜索频率**: 百度搜索API可能有调用频率限制，请合理使用
3. **网络环境**: 联网搜索功能需要稳定的网络连接
4. **成本控制**: 联网搜索会产生API调用费用，建议监控使用量

## 未来扩展

可以考虑的功能扩展：
- 支持更多搜索引擎（Google、Bing等）
- 添加图片搜索功能
- 实现搜索结果缓存机制
- 支持多轮对话上下文
- 添加用户偏好学习功能