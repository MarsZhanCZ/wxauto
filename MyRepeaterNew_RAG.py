from wxauto import WeChat
from wxauto.msgs import FriendMessage
import time
from datetime import datetime
from typing import Set, Optional, List
import os
from dotenv import load_dotenv
import pandas as pd

# LangChain 相关导入
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 加载环境变量
load_dotenv()


class RAGKnowledgeBase:
    """RAG 知识库管理类"""
    
    def __init__(self, docs_path: str = "raw_docs", persist_path: str = "faiss_index"):
        """初始化知识库
        
        Args:
            docs_path: 原始文档目录
            persist_path: FAISS 索引持久化路径
        """
        self.docs_path = docs_path
        self.persist_path = persist_path
        # 使用 SiliconFlow 的 BGE-M3 embedding 模型
        self.embeddings = OpenAIEmbeddings(
            model="Qwen/Qwen3-Embedding-8B",
            openai_api_key=os.getenv('SILICONFLOW_API_KEY'),
            openai_api_base="https://api.siliconflow.cn/v1"
        )
        self.vector_store = None
        # 优化文本分割策略
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # 减小块大小，提高精确度
            chunk_overlap=100,  # 增加重叠，避免信息丢失
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]  # 增加更多分隔符
        )
    
    def load_excel_qa(self, file_path: str) -> List[Document]:
        """加载 Excel QA 对文件
        
        Args:
            file_path: Excel 文件路径
            
        Returns:
            Document 列表
        """
        documents = []
        try:
            df = pd.read_excel(file_path)
            # 假设 Excel 有 Q 和 A 两列，或者类似的列名
            # 尝试识别问题和答案列
            q_col = None
            a_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                if 'q' in col_lower or '问' in col_lower or '问题' in col_lower:
                    q_col = col
                elif 'a' in col_lower or '答' in col_lower or '答案' in col_lower or '回答' in col_lower:
                    a_col = col
            
            # 如果没找到，使用前两列
            if q_col is None and len(df.columns) >= 1:
                q_col = df.columns[0]
            if a_col is None and len(df.columns) >= 2:
                a_col = df.columns[1]
            
            for idx, row in df.iterrows():
                q = str(row[q_col]) if q_col and pd.notna(row[q_col]) else ""
                a = str(row[a_col]) if a_col and pd.notna(row[a_col]) else ""
                
                if q and a:
                    # 为QA对创建多个索引条目，提高召回率
                    # 1. 完整的QA对
                    content = f"问题：{q}\n答案：{a}"
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": file_path, "type": "qa", "row": idx, "question": q}
                    ))
                    
                    # 2. 单独的问题（用于问题匹配）
                    documents.append(Document(
                        page_content=q,
                        metadata={"source": file_path, "type": "question", "row": idx, "answer": a}
                    ))
                    
                    # 3. 单独的答案（用于内容匹配）
                    if len(a) > 20:  # 只有答案足够长才单独索引
                        documents.append(Document(
                            page_content=a,
                            metadata={"source": file_path, "type": "answer", "row": idx, "question": q}
                        ))
            
            print(f"  ✓ 从 {file_path} 加载了 {len(documents)} 条 QA 对")
        except Exception as e:
            print(f"  ✗ 加载 Excel 文件失败: {e}")
        
        return documents
    
    def load_text_file(self, file_path: str) -> List[Document]:
        """加载文本文件
        
        Args:
            file_path: 文本文件路径
            
        Returns:
            Document 列表
        """
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 分割文本
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
        
        # 遍历文档目录
        for filename in os.listdir(self.docs_path):
            file_path = os.path.join(self.docs_path, filename)
            
            if filename.endswith('.xlsx') or filename.endswith('.xls'):
                docs = self.load_excel_qa(file_path)
                all_documents.extend(docs)
            elif not filename.startswith('.'):
                # 尝试作为文本文件加载
                docs = self.load_text_file(file_path)
                all_documents.extend(docs)
        
        if not all_documents:
            print("  ⚠ 未找到任何文档")
            return False
        
        print(f"共加载 {len(all_documents)} 个文档块，正在生成向量...")
        
        try:
            # 分批处理文档，避免超过 API 批量限制
            batch_size = 32  # 设置为 32，留一些余量
            vector_stores = []
            
            for i in range(0, len(all_documents), batch_size):
                batch_docs = all_documents[i:i + batch_size]
                print(f"  正在处理第 {i//batch_size + 1} 批文档 ({len(batch_docs)} 个)...")
                
                if i == 0:
                    # 第一批创建新的向量存储
                    vector_store = FAISS.from_documents(batch_docs, self.embeddings)
                else:
                    # 后续批次创建临时向量存储然后合并
                    temp_store = FAISS.from_documents(batch_docs, self.embeddings)
                    vector_store.merge_from(temp_store)
            
            self.vector_store = vector_store
            # 保存索引
            self.vector_store.save_local(self.persist_path)
            print(f"  ✓ 知识库索引构建完成，已保存到 {self.persist_path}")
            return True
        except Exception as e:
            print(f"  ✗ 构建索引失败: {e}")
            return F
    
    def load_index(self) -> bool:
        """加载已有的 FAISS 索引
        
        Returns:
            是否加载成功
        """
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


class MessageRepeaterRAG:
    """微信消息智能回复机器人（RAG 增强版）
    
    使用 LangChain + FAISS 实现 RAG，结合知识库内容进行智能回复。
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
        self.my_nickname = my_nickname  # 存储我的昵称用于@检测
        
        # 初始化 LLM
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=self.api_key,
            openai_api_base="https://api.deepseek.com/v1",
            temperature=0.8,
            max_tokens=200
        )
        
        # 初始化知识库
        print("正在初始化知识库...")
        self.kb = RAGKnowledgeBase()
        
        # 加载或构建索引
        if rebuild_index or not self.kb.load_index():
            self.kb.build_index()
        
        # 定义提示词模板
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个友好的微信聊天助手，请遵循以下原则：

1. 回复要像真人聊天一样自然亲切
2. 控制回复长度在100字以内，简洁明了
3. 绝对不要使用任何markdown格式（如**加粗**、*斜体*、`代码`等）
4. 不要使用项目符号、编号列表等格式化内容
5. 可以适当使用表情符号让对话生动
6. 回复要像普通人发微信一样，不要显得太正式或机械

以下是可能相关的参考信息：
{context}

请根据参考信息（如果相关的话）回答用户问题。如果参考信息与问题无关，就直接用你的知识回答。"""),
            ("human", "{question}")
        ])
        
        self.direct_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个友好的微信聊天助手，请遵循以下原则：

1. 回复要像真人聊天一样自然亲切
2. 控制回复长度在100字以内，简洁明了
3. 绝对不要使用任何markdown格式（如**加粗**、*斜体*、`代码`等）
4. 不要使用项目符号、编号列表等格式化内容
5. 可以适当使用表情符号让对话生动
6. 回复要像普通人发微信一样，不要显得太正式或机械

用中文回复，语气自然友好。"""),
            ("human", "{question}")
        ])
        
        self.output_parser = StrOutputParser()
        
        print("=" * 50)
        print("微信消息智能回复机器人（RAG增强版）已启动...")
        print("使用 LangChain + FAISS + DeepSeek 进行智能回复")
        print("等待接收新消息，按 Ctrl+C 停止...")
        print("=" * 50)
    
    def should_process_message(self, msg, chat_type: str = None) -> bool:
        """判断是否应该处理该消息
        
        Args:
            msg: 消息对象
            chat_type: 聊天类型，'friend' 表示私聊，'group' 表示群聊
        """
        # 过滤掉自己发送的消息
        if msg.attr == 'self':
            return False
        
        # 过滤掉系统消息（如时间消息、系统通知等）
        if msg.attr in ('system', 'time', 'tickle'):
            return False
        
        # 检查消息是否已经处理过
        if msg.id in self.processed_msg_ids:
            return False
        
        # 根据 chat_type 判断是私聊还是群聊
        if chat_type == 'friend':
            # 私聊消息，直接处理
            return True
        elif chat_type == 'group':
            # 群聊消息，只处理@我的消息
            if hasattr(msg, 'content') and msg.content:
                content = str(msg.content)
                
                # 检查是否被@了
                if '@' in content:
                    # 如果设置了昵称，检查是否@了我
                    if self.my_nickname:
                        if f"@{self.my_nickname}" in content:
                            return True
                    else:
                        # 没有设置昵称时不自动回复群聊（避免误触发）
                        return False
            return False
        else:
            # 未知类型，不处理
            return False
    
    def get_ai_reply(self, user_message: str) -> Optional[str]:
        """获取 AI 回复（带 RAG）
        
        Args:
            user_message: 用户消息
            
        Returns:
            AI 回复内容
        """
        try:
            # 搜索相关文档
            relevant_docs = self.kb.search(user_message, k=3)
            
            if relevant_docs:
                # 有相关文档，使用 RAG 回复
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                chain = self.rag_prompt | self.llm | self.output_parser
                reply = chain.invoke({"context": context, "question": user_message})
                print(f"  [RAG] 找到 {len(relevant_docs)} 条相关文档:")
                for i, doc in enumerate(relevant_docs, 1):
                    preview = doc.page_content[:15].replace('\n', ' ') + "..."
                    doc_type = doc.metadata.get('type', 'unknown')
                    print(f"    {i}. [{doc_type}] {preview}")
            else:
                # 无相关文档，直接回复
                chain = self.direct_prompt | self.llm | self.output_parser
                reply = chain.invoke({"question": user_message})
                print(f"  [Direct] 未找到相关文档，直接回复")
            
            return reply.strip()
            
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
            chat_type = new_msg_data.get('chat_type', 'unknown')  # 获取聊天类型
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
    
    parser = argparse.ArgumentParser(description='微信消息智能回复机器人（RAG增强版）')
    parser.add_argument('--rebuild', action='store_true', help='重新构建知识库索引')
    parser.add_argument('--debug', action='store_true', help='开启调试模式')
    parser.add_argument('--nickname', type=str, help='我的微信昵称，用于检测@我的消息')
    args = parser.parse_args()
    
    repeater = MessageRepeaterRAG(
        debug=args.debug, 
        rebuild_index=args.rebuild,
        my_nickname=args.nickname
    )
    repeater.run()
