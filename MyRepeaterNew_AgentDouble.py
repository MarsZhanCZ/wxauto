from wxauto import WeChat
from wxauto.msgs import FriendMessage
import time
from datetime import datetime
from typing import Set, Optional, List
import os
import sys
from dotenv import load_dotenv
import pandas as pd
import requests
import json

# 强制刷新输出缓冲
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except:
    pass

# LangChain 1.1 相关导入
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

# 加载环境变量
load_dotenv()

print("模块导入完成", flush=True)


class RAGKnowledgeBase:
    """RAG 知识库管理类"""
    
    def __init__(self, docs_path: str = "raw_docs", persist_path: str = "faiss_index"):
        self.docs_path = docs_path
        self.persist_path = persist_path
        self.embeddings = OpenAIEmbeddings(
            model="BAAI/bge-m3",
            openai_api_key=os.getenv('SILICONFLOW_API_KEY'),
            openai_api_base="https://api.siliconflow.cn/v1"
        )
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            separators=["\n\n", "\n", " "]
        )
    
    def load_excel_qa(self, file_path: str) -> List[Document]:
        """加载 Excel QA 对文件"""
        documents = []
        try:
            df = pd.read_excel(file_path)
            q_col = None
            a_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                if 'q' in col_lower or '问' in col_lower or '问题' in col_lower:
                    q_col = col
                elif 'a' in col_lower or '答' in col_lower or '答案' in col_lower or '回答' in col_lower:
                    a_col = col
            
            if q_col is None and len(df.columns) >= 1:
                q_col = df.columns[0]
            if a_col is None and len(df.columns) >= 2:
                a_col = df.columns[1]
            
            for idx, row in df.iterrows():
                q = str(row[q_col]) if q_col and pd.notna(row[q_col]) else ""
                a = str(row[a_col]) if a_col and pd.notna(row[a_col]) else ""
                
                if q and a:
                    content = f"问题：{q}\n答案：{a}"
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": file_path, "type": "qa", "row": idx, "question": q}
                    ))
                    documents.append(Document(
                        page_content=q,
                        metadata={"source": file_path, "type": "question", "row": idx, "answer": a}
                    ))
                    if len(a) > 20:
                        documents.append(Document(
                            page_content=a,
                            metadata={"source": file_path, "type": "answer", "row": idx, "question": q}
                        ))
            
            print(f"  ✓ 从 {file_path} 加载了 {len(documents)} 条文档")
        except Exception as e:
            print(f"  ✗ 加载 Excel 文件失败: {e}")
        
        return documents
    
    def load_text_file(self, file_path: str) -> List[Document]:
        """加载文本文件"""
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = self.text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={"source": file_path, "type": "text", "chunk": i}
                ))
            
            print(f"  ✓ 从 {file_path} 加载了 {len(documents)} 个文本块")
        except Exception as e:
            print(f"  ✗ 加载文本文件失败: {e}")
        
        return documents
    
    def build_index(self):
        """构建 FAISS 向量索引"""
        print("正在构建知识库索引...")
        
        all_documents = []
        
        for filename in os.listdir(self.docs_path):
            file_path = os.path.join(self.docs_path, filename)
            
            if filename.endswith('.xlsx') or filename.endswith('.xls'):
                docs = self.load_excel_qa(file_path)
                all_documents.extend(docs)
            elif not filename.startswith('.'):
                docs = self.load_text_file(file_path)
                all_documents.extend(docs)
        
        if not all_documents:
            print("  ⚠ 未找到任何文档")
            return False
        
        print(f"共加载 {len(all_documents)} 个文档块，正在生成向量...")
        
        try:
            batch_size = 32
            vector_store = None
            
            for i in range(0, len(all_documents), batch_size):
                batch_docs = all_documents[i:i + batch_size]
                print(f"  正在处理第 {i//batch_size + 1} 批文档 ({len(batch_docs)} 个)...")
                
                if i == 0:
                    vector_store = FAISS.from_documents(batch_docs, self.embeddings)
                else:
                    temp_store = FAISS.from_documents(batch_docs, self.embeddings)
                    vector_store.merge_from(temp_store)
            
            self.vector_store = vector_store
            self.vector_store.save_local(self.persist_path)
            print(f"  ✓ 知识库索引构建完成，已保存到 {self.persist_path}")
            return True
        except Exception as e:
            print(f"  ✗ 构建索引失败: {e}")
            return False
    
    def load_index(self) -> bool:
        """加载已有的 FAISS 索引"""
        try:
            if os.path.exists(self.persist_path):
                self.vector_store = FAISS.load_local(
                    self.persist_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"  ✓ 已加载现有知识库索引")
                return True
        except Exception as e:
            print(f"  ⚠ 加载索引失败: {e}")
        return False
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """搜索相关文档"""
        if not self.vector_store:
            return []
        
        try:
            # 直接搜索，不做阈值过滤（让 LLM 自己判断相关性）
            docs = self.vector_store.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"  ⚠ 搜索失败: {e}")
            return []


# 全局知识库实例（供工具使用）
_knowledge_base: Optional[RAGKnowledgeBase] = None


def get_knowledge_base() -> RAGKnowledgeBase:
    """获取知识库实例"""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = RAGKnowledgeBase()
        if not _knowledge_base.load_index():
            _knowledge_base.build_index()
    return _knowledge_base


# 定义 RAG 搜索工具
@tool
def search_knowledge_base(query: str) -> str:
    """搜索智云上海相关的知识库，获取产品信息、服务介绍、常见问题解答等内容。
    
    当用户询问关于智云上海、智家服务、算力、AI服务、城市智能化等相关问题时，使用此工具搜索知识库获取准确信息。
    
    Args:
        query: 用户的问题或搜索关键词
        
    Returns:
        搜索到的相关信息，如果没有找到则返回空字符串
    """
    kb = get_knowledge_base()
    docs = kb.search(query, k=3)
    
    if not docs:
        return "未找到相关信息"
    
    results = []
    for doc in docs:
        results.append(doc.page_content)
    
    return "\n\n---\n\n".join(results)


# 定义百度搜索工具
@tool
def baidu_search_tool(query: str) -> str:
    """
    使用百度搜索API进行联网搜索，获取最新的网络信息。
    
    当用户询问需要实时信息、最新新闻、当前事件、股价、天气、最新技术动态等问题时，
    或者本地知识库无法回答的问题时，使用此工具进行联网搜索。
    
    Args:
        query: 搜索关键词或问题
        
    Returns:
        搜索结果的文本内容，包含相关的网络信息
    """
    api_key = os.getenv('BAIDU_SEARCH_API_KEY')
    if not api_key:
        return "百度搜索API密钥未配置，无法进行联网搜索"
    
    url = 'https://qianfan.baidubce.com/v2/ai_search'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    messages = [
        {
            "content": query,
            "role": "user"
        }
    ]
    
    data = {
        "messages": messages,
        "search_source": "baidu_search_v2",
        "search_recency_filter": "month"  # 搜索最近一个月的内容
    }
    
    try:
        print(f"  [百度搜索] 正在搜索: {query[:30]}...")
        response = requests.post(url, headers=headers, json=data, timeout=10)
        print(f"  [百度搜索] 响应状态: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # 提取搜索结果的主要内容
            if 'result' in result:
                content = result['result']
                # 如果返回的是字符串，直接返回
                if isinstance(content, str):
                    return content
                # 如果是字典，尝试提取主要内容
                elif isinstance(content, dict):
                    if 'content' in content:
                        return content['content']
                    elif 'answer' in content:
                        return content['answer']
                    else:
                        return str(content)
            else:
                # 如果没有result字段，返回整个响应的字符串形式
                return str(result)
                
        else:
            return f"搜索请求失败，状态码: {response.status_code}，错误信息: {response.text}"
            
    except requests.exceptions.Timeout:
        return "搜索请求超时，请稍后再试"
    except requests.exceptions.RequestException as e:
        return f"搜索请求异常: {str(e)}"
    except Exception as e:
        return f"搜索过程中发生错误: {str(e)}"


class MessageRepeaterAgent:
    """微信消息智能回复机器人（双工具版本）
    
    使用 LangChain 的 Agent 架构，集成了 RAG 知识库搜索和百度联网搜索两个工具。
    """
    
    def __init__(self, debug: bool = False, rebuild_index: bool = False, my_nickname: str = None):
        """初始化消息智能回复机器人
        
        Args:
            debug (bool): 是否开启调试模式
            rebuild_index (bool): 是否重新构建知识库索引
            my_nickname (str): 我的微信昵称，用于检测@我的消息
        """
        # 检查 API Key
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("请在.env文件中设置DEEPSEEK_API_KEY")
        
        self.siliconflow_key = os.getenv('SILICONFLOW_API_KEY')
        if not self.siliconflow_key:
            raise ValueError("请在.env文件中设置SILICONFLOW_API_KEY")
        
        # 百度搜索API Key（可选）
        self.baidu_key = os.getenv('BAIDU_SEARCH_API_KEY')
        if not self.baidu_key:
            print("  ⚠ 未设置BAIDU_SEARCH_API_KEY，联网搜索功能将不可用")
        
        # 初始化微信
        self.wx = WeChat(debug=debug)
        self.processed_msg_ids: Set[int] = set()
        # 增加基于内容的去重（防止同一消息被多次处理）
        self.processed_msg_contents: dict = {}  # {(sender, content): timestamp}
        self.content_dedup_window = 60  # 60秒内相同发送者的相同内容视为重复
        
        # 设置昵称
        self.my_nickname = my_nickname
        if not self.my_nickname:
            # 尝试从微信获取当前用户昵称
            try:
                if hasattr(self.wx, 'CurrentUserName'):
                    self.my_nickname = self.wx.CurrentUserName
                    print(f"  ✓ 自动获取到昵称: {self.my_nickname}")
                else:
                    print("  ⚠ 无法自动获取昵称，群聊@检测将不可用")
                    print("  建议使用 --nickname 参数指定昵称")
            except Exception as e:
                print(f"  ⚠ 获取昵称失败: {e}")
                print("  建议使用 --nickname 参数指定昵称")
        
        # 初始化知识库
        print("正在初始化知识库...")
        global _knowledge_base
        _knowledge_base = RAGKnowledgeBase()
        if rebuild_index or not _knowledge_base.load_index():
            _knowledge_base.build_index()
        
        # 初始化 LLM
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=self.api_key,
            openai_api_base="https://api.deepseek.com/v1",
            temperature=0.3,
            max_tokens=200
        )
        
        # 定义工具列表（包含知识库搜索和联网搜索）
        self.tools = [search_knowledge_base, baidu_search_tool]
        
        # 系统提示词
        self.system_prompt = """你是一个友好的微信聊天助手，拥有两个强大的工具：

1. search_knowledge_base: 搜索智云上海相关的本地知识库
2. baidu_search_tool: 进行联网搜索获取最新信息

请遵循以下原则：

**工具使用策略：**
- 当用户询问智云上海、智家服务、算力、AI服务等公司相关问题时，优先使用 search_knowledge_base
- 当用户询问实时信息、最新新闻、天气、股价、当前事件等需要联网的问题时，使用 baidu_search_tool
- 如果本地知识库没有找到相关信息，可以尝试联网搜索
- 对于日常闲聊、问候等简单问题，可以直接回复，无需使用工具

**回复风格：**
1. 回复要像真人聊天一样自然亲切
2. 控制回复长度在100字以内，简洁明了
3. 绝对不要使用任何markdown格式（如**加粗**、*斜体*、`代码`等）
4. 不要使用项目符号、编号列表等格式化内容
5. 可以适当使用表情符号让对话生动
6. 回复要像普通人发微信一样，不要显得太正式或机械

用中文回复，语气自然友好。"""
        
        # 创建 Agent（使用 LangGraph 的 create_react_agent）
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=self.system_prompt
        )
        
        print("=" * 50)
        print("微信消息智能回复机器人（双工具版本）已启动...")
        print("集成功能：")
        print("  ✓ RAG 知识库搜索（智云上海相关）")
        if self.baidu_key:
            print("  ✓ 百度联网搜索（实时信息）")
        else:
            print("  ✗ 百度联网搜索（未配置API Key）")
        print("等待接收新消息，按 Ctrl+C 停止...")
        print("=" * 50)
    
    def should_process_message(self, msg, chat_type: str = None) -> bool:
        """判断是否应该处理该消息"""
        if msg.attr == 'self':
            return False
        
        if msg.attr in ('system', 'time', 'tickle'):
            return False
        
        # 基于消息ID去重
        if msg.id in self.processed_msg_ids:
            return False
        
        # 基于内容+发送者去重（防止同一消息被多次触发）
        current_time = time.time()
        content_key = (getattr(msg, 'sender', ''), getattr(msg, 'content', ''))
        
        if content_key in self.processed_msg_contents:
            last_time = self.processed_msg_contents[content_key]
            if current_time - last_time < self.content_dedup_window:
                print(f"  [去重] 跳过重复消息: {content_key[1][:20]}...")
                return False
        
        if chat_type == 'friend':
            return True
        elif chat_type == 'group':
            # 调试群聊消息
            print(f"  [调试] 群聊消息检测:")
            print(f"    消息内容: {getattr(msg, 'content', 'None')}")
            print(f"    发送者: {getattr(msg, 'sender', 'None')}")
            print(f"    我的昵称: {self.my_nickname}")
            
            if hasattr(msg, 'content') and msg.content:
                content = str(msg.content)
                print(f"    是否包含@: {'@' in content}")
                
                if '@' in content:
                    if self.my_nickname:
                        at_me = f"@{self.my_nickname}" in content
                        print(f"    是否@我: {at_me}")
                        if at_me:
                            return True
                    else:
                        print(f"    未设置昵称，不处理群聊消息")
                        return False
                else:
                    print(f"    消息不包含@，跳过")
            else:
                print(f"    消息内容为空，跳过")
            return False
        else:
            print(f"  [调试] 未知聊天类型: {chat_type}")
            return False
    
    def get_ai_reply(self, user_message: str) -> Optional[str]:
        """获取 AI 回复（使用 Agent）
        
        Args:
            user_message: 用户消息
            
        Returns:
            AI 回复内容
        """
        try:
            # 调用 Agent
            result = self.agent.invoke({
                "messages": [("user", user_message)]
            })
            
            # 提取最终回复
            messages = result.get("messages", [])
            if messages:
                # 分析使用了哪些工具
                tools_used = []
                for msg in messages:
                    if isinstance(msg, ToolMessage):
                        if hasattr(msg, 'name'):
                            tools_used.append(msg.name)
                
                # 获取最后一条 AI 消息
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        if tools_used:
                            tool_names = []
                            if 'search_knowledge_base' in tools_used:
                                tool_names.append('知识库')
                            if 'baidu_search_tool' in tools_used:
                                tool_names.append('联网搜索')
                            print(f"  [Agent] 使用了工具: {', '.join(tool_names)}")
                        else:
                            print(f"  [Agent] 直接回复")
                        
                        return msg.content.strip()
            
            return None
            
        except Exception as e:
            print(f"  ✗ 获取AI回复失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def check_wechat_status(self) -> bool:
        """检查微信连接状态"""
        try:
            # 尝试获取微信窗口信息来检查连接状态
            if hasattr(self.wx, '_api') and hasattr(self.wx._api, 'main_window'):
                return self.wx._api.main_window is not None
            return True  # 如果无法检查，假设连接正常
        except Exception as e:
            print(f"  ⚠ 检查微信状态时发生异常: {e}")
            return False
    
    def process_new_messages(self):
        """处理新消息"""
        try:
            # 增加对 wxauto 内部异常的处理
            try:
                new_msg_data = self.wx.GetNextNewMessage(filter_mute=False)
            except KeyError as ke:
                # wxauto 内部 KeyError，通常是消息ID管理问题
                print(f"  ⚠ wxauto内部KeyError: {ke}，跳过本次消息检查")
                return
            except Exception as we:
                # 其他 wxauto 相关异常
                print(f"  ⚠ wxauto异常: {we}，跳过本次消息检查")
                return
            
            if not new_msg_data or 'msg' not in new_msg_data:
                return
            
            chat_name = new_msg_data.get('chat_name', '未知')
            chat_type = new_msg_data.get('chat_type', 'unknown')
            messages = new_msg_data.get('msg', [])
            
            if not isinstance(messages, list):
                messages = [messages]
            
            for msg in messages:
                try:
                    if not self.should_process_message(msg, chat_type):
                        continue
                    
                    self.processed_msg_ids.add(msg.id)
                    
                    # 记录内容用于去重
                    content_key = (msg.sender, msg.content)
                    self.processed_msg_contents[content_key] = time.time()
                    
                    # 清理过期的去重记录（避免内存泄漏）
                    current_time = time.time()
                    expired_keys = [k for k, v in self.processed_msg_contents.items() 
                                   if current_time - v > self.content_dedup_window * 2]
                    for k in expired_keys:
                        del self.processed_msg_contents[k]
                    
                    msg_content = msg.content
                    msg_sender = msg.sender
                    
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"\n[{timestamp}] 收到来自 [{chat_name}] ({msg_sender}) 的消息:")
                    print(f"  内容: {msg_content}")
                    print(f"  消息类型: {'私聊' if chat_type == 'friend' else '群聊@我'}")
                    
                    print(f"  正在生成回复...")
                    ai_reply = self.get_ai_reply(msg_content)
                    
                    if ai_reply:
                        print(f"  AI回复: {ai_reply}")
                        print(f"  正在发送回复给 [{chat_name}]...")
                        try:
                            result = self.wx.SendMsg(ai_reply, who=chat_name)
                            if result:
                                print(f"  ✓ 已成功发送回复")
                            else:
                                error_msg = result.get('message', '未知错误') if isinstance(result, dict) else '发送失败'
                                print(f"  ✗ 发送失败: {error_msg}")
                        except Exception as e:
                            print(f"  ✗ 发送异常: {e}")
                    else:
                        print(f"  ✗ 未能获取AI回复，跳过发送")
                    
                    # 处理完一条消息后稍作延迟
                    time.sleep(0.3)
                    
                except Exception as msg_e:
                    print(f"  ✗ 处理单条消息时发生错误: {msg_e}")
                    continue
                
        except Exception as e:
            print(f"处理消息时发生错误: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """运行消息智能回复机器人"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        try:
            while True:
                try:
                    # 定期检查微信连接状态
                    if consecutive_errors > 0 and not self.check_wechat_status():
                        print("  ⚠ 微信连接可能已断开，请检查微信状态")
                    
                    self.process_new_messages()
                    consecutive_errors = 0  # 成功处理后重置错误计数
                    time.sleep(0.5)
                    
                except KeyboardInterrupt:
                    raise  # 重新抛出键盘中断
                    
                except Exception as e:
                    consecutive_errors += 1
                    print(f"\n⚠ 主循环异常 (第{consecutive_errors}次): {e}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"\n✗ 连续发生{max_consecutive_errors}次错误，程序可能存在严重问题")
                        print("建议检查微信状态或重启程序")
                        break
                    
                    # 错误后等待更长时间再重试
                    wait_time = min(consecutive_errors * 2, 10)
                    print(f"  等待{wait_time}秒后重试...")
                    time.sleep(wait_time)
                
        except KeyboardInterrupt:
            print("\n" + "=" * 50)
            print("程序已停止")
            print("=" * 50)
        except Exception as e:
            print(f"\n发生严重错误: {e}")
            import traceback
            traceback.print_exc()
            print("程序已停止")


if __name__ == "__main__":
    import argparse
    
    print("正在解析命令行参数...")
    
    parser = argparse.ArgumentParser(description='微信消息智能回复机器人（双工具版本）')
    parser.add_argument('--rebuild', action='store_true', help='重新构建知识库索引')
    parser.add_argument('--debug', action='store_true', help='开启调试模式')
    parser.add_argument('--nickname', type=str, help='我的微信昵称，用于检测@我的消息')
    args = parser.parse_args()
    
    print(f"参数: rebuild={args.rebuild}, debug={args.debug}, nickname={args.nickname}")
    print("正在初始化机器人...")
    
    try:
        repeater = MessageRepeaterAgent(
            debug=args.debug, 
            rebuild_index=args.rebuild,
            my_nickname=args.nickname
        )
        repeater.run()
    except Exception as e:
        print(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()