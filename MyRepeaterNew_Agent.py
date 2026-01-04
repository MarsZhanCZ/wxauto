from wxauto import WeChat
from wxauto.msgs import FriendMessage
import time
from datetime import datetime
from typing import Set, Optional, List
import os
import sys
from dotenv import load_dotenv
import pandas as pd

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


class MessageRepeaterAgent:
    """微信消息智能回复机器人（Agent 版本）
    
    使用 LangChain  的 Agent 架构，将 RAG 作为工具使用。
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
        
        # 初始化微信
        self.wx = WeChat(debug=debug)
        self.processed_msg_ids: Set[int] = set()
        self.my_nickname = my_nickname
        
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
        
        # 定义工具列表
        self.tools = [search_knowledge_base]
        
        # 系统提示词
        self.system_prompt = """你是一个友好的微信聊天助手，请遵循以下原则：

1. 回复要像真人聊天一样自然亲切
2. 控制回复长度在100字以内，简洁明了
3. 绝对不要使用任何markdown格式（如**加粗**、*斜体*、`代码`等）
4. 不要使用项目符号、编号列表等格式化内容
5. 可以适当使用表情符号让对话生动
6. 回复要像普通人发微信一样，不要显得太正式或机械

当用户询问关于智云上海、智家服务、算力、AI服务等相关问题时，请先使用 search_knowledge_base 工具搜索知识库获取准确信息，然后基于搜索结果回答。

如果问题与知识库无关（如日常闲聊、问候等），直接友好回复即可。

用中文回复，语气自然友好。"""
        
        # 创建 Agent（使用 LangGraph 的 create_react_agent）
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=self.system_prompt
        )
        
        print("=" * 50)
        print("微信消息智能回复机器人（Agent版本）已启动...")
        print("使用 LangChain Agent + RAG 工具进行智能回复")
        print("等待接收新消息，按 Ctrl+C 停止...")
        print("=" * 50)
    
    def should_process_message(self, msg, chat_type: str = None) -> bool:
        """判断是否应该处理该消息"""
        if msg.attr == 'self':
            return False
        
        if msg.attr in ('system', 'time', 'tickle'):
            return False
        
        if msg.id in self.processed_msg_ids:
            return False
        
        if chat_type == 'friend':
            return True
        elif chat_type == 'group':
            if hasattr(msg, 'content') and msg.content:
                content = str(msg.content)
                if '@' in content:
                    if self.my_nickname:
                        if f"@{self.my_nickname}" in content:
                            return True
                    else:
                        return False
            return False
        else:
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
                # 检查是否使用了工具
                tool_used = any(isinstance(m, ToolMessage) for m in messages)
                
                # 获取最后一条 AI 消息
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        if tool_used:
                            print(f"  [Agent] 使用了 RAG 工具")
                        else:
                            print(f"  [Agent] 直接回复")
                        
                        return msg.content.strip()
            
            return None
            
        except Exception as e:
            print(f"  ✗ 获取AI回复失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_new_messages(self):
        """处理新消息"""
        try:
            new_msg_data = self.wx.GetNextNewMessage(filter_mute=False)
            
            if not new_msg_data or 'msg' not in new_msg_data:
                return
            
            chat_name = new_msg_data.get('chat_name', '未知')
            chat_type = new_msg_data.get('chat_type', 'unknown')
            messages = new_msg_data.get('msg', [])
            
            if not isinstance(messages, list):
                messages = [messages]
            
            for msg in messages:
                if not self.should_process_message(msg, chat_type):
                    continue
                
                self.processed_msg_ids.add(msg.id)
                
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
                            error_msg = result.get('message', '未知错误')
                            print(f"  ✗ 发送失败: {error_msg}")
                    except Exception as e:
                        print(f"  ✗ 发送异常: {e}")
                else:
                    print(f"  ✗ 未能获取AI回复，跳过发送")
                
                time.sleep(0.3)
                
        except Exception as e:
            print(f"处理消息时发生错误: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """运行消息智能回复机器人"""
        try:
            while True:
                self.process_new_messages()
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n" + "=" * 50)
            print("程序已停止")
            print("=" * 50)
        except Exception as e:
            print(f"\n发生错误: {e}")
            import traceback
            traceback.print_exc()
            print("程序已停止")


if __name__ == "__main__":
    import argparse
    
    print("正在解析命令行参数...")
    
    parser = argparse.ArgumentParser(description='微信消息智能回复机器人（Agent版本）')
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
