from wxauto import WeChat
from wxauto.msgs import FriendMessage
import time
from datetime import datetime
from typing import Set, Optional
import requests


class MessageRepeater:
    """微信消息智能回复机器人
    
    自动检测任意新用户发来的信息并使用DeepSeek AI模型回复，确保每条消息只回复一次。
    """
    
    def __init__(self, debug: bool = False, api_key: str = None, system_prompt: str = None):
        """初始化消息智能回复机器人
        
        Args:
            debug (bool): 是否开启调试模式，默认False
            api_key (str): DeepSeek API Key，默认使用内置的Key
            system_prompt (str): 自定义系统提示词，默认使用内置的提示词
        """
        self.wx = WeChat(debug=debug)
        # 用于记录已处理的消息ID，确保每条消息只回复一次
        self.processed_msg_ids: Set[int] = set()
        # DeepSeek API配置
        self.api_key = api_key or "sk-2e3314629b5f44a6a57f4188943ce488"
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        
        # 默认系统提示词
        self.system_prompt = system_prompt or """你是一个友好的微信聊天助手，请遵循以下原则：

1. 回复要像真人聊天一样自然亲切s
2. 控制回复长度在100字以内，简洁明了
3. 绝对不要使用任何markdown格式（如**加粗**、*斜体*、`代码`等）
4. 不要使用项目符号、编号列表等格式化内容
5. 根据对方语气调整回复风格，问候要热情，问题要有用，闲聊要轻松
6. 可以适当使用表情符号让对话生动
7. 不确定的信息要诚实说明
8. 回复要像普通人发微信一样，不要显得太正式或机械

用中文回复，语气自然友好。"""
        
        print("=" * 50)
        print("微信消息智能回复机器人已启动...")
        print("使用 DeepSeek AI 模型进行智能回复")
        print("等待接收新消息，按 Ctrl+C 停止...")
        print("=" * 50)
    
    def should_process_message(self, msg) -> bool:
        """判断是否应该处理该消息
        
        Args:
            msg: 消息对象
            
        Returns:
            bool: 是否应该处理
        """
        # 过滤掉自己发送的消息
        if msg.attr == 'self':
            return False
        
        # 过滤掉系统消息（如时间消息、系统通知等）
        if msg.attr in ('system', 'time', 'tickle'):
            return False
        
        # 只处理好友发送的消息（FriendMessage）
        if not isinstance(msg, FriendMessage):
            return False
        
        # 检查消息是否已经处理过
        if msg.id in self.processed_msg_ids:
            return False
        
        return True
    
    def call_deepseek_api(self, user_message: str) -> Optional[str]:
        """调用DeepSeek API获取AI回复
        
        Args:
            user_message (str): 用户输入的消息
            
        Returns:
            Optional[str]: AI回复内容，失败时返回None
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 系统提示词，定义AI的角色和回复风格
            system_prompt = self.system_prompt
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                "temperature": 0.8,
                "max_tokens": 150  # 进一步限制回复长度
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                # 提取AI回复内容
                if "choices" in result and len(result["choices"]) > 0:
                    ai_reply = result["choices"][0]["message"]["content"]
                    return ai_reply.strip()
                else:
                    print(f"  ⚠ API返回格式异常: {result}")
                    return None
            else:
                print(f"  ✗ API调用失败，状态码: {response.status_code}")
                print(f"  错误信息: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"  ✗ API调用超时")
            return None
        except requests.exceptions.RequestException as e:
            print(f"  ✗ API调用异常: {e}")
            return None
        except Exception as e:
            print(f"  ✗ 处理API响应时发生错误: {e}")
            return None
    
    def process_new_messages(self):
        """处理新消息"""
        try:
            # 获取下一个新消息
            # GetNextNewMessage 返回格式: {'chat_name': '...', 'chat_type': '...', 'msg': [...]}
            new_msg_data = self.wx.GetNextNewMessage(filter_mute=False)
            
            # 如果没有新消息，返回空字典
            if not new_msg_data or 'msg' not in new_msg_data:
                return
            
            chat_name = new_msg_data.get('chat_name', '未知')
            chat_type = new_msg_data.get('chat_type', 'unknown')
            messages = new_msg_data.get('msg', [])
            
            # 确保 messages 是列表
            if not isinstance(messages, list):
                messages = [messages]
            
            # 处理每条消息
            for msg in messages:
                # 判断是否应该处理该消息
                if not self.should_process_message(msg):
                    continue
                
                # 标记消息为已处理
                self.processed_msg_ids.add(msg.id)
                
                # 获取消息内容
                msg_content = msg.content
                msg_sender = msg.sender
                
                # 打印接收到的消息信息
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n[{timestamp}] 收到来自 [{chat_name}] ({msg_sender}) 的消息:")
                print(f"  内容: {msg_content}")
                
                # 调用DeepSeek API获取AI回复
                print(f"  正在调用 DeepSeek AI 生成回复...")
                ai_reply = self.call_deepseek_api(msg_content)
                
                if ai_reply:
                    print(f"  AI回复: {ai_reply}")
                    # 发送AI回复
                    print(f"  正在发送回复给 [{chat_name}]...")
                    try:
                        result = self.wx.SendMsg(ai_reply, who=chat_name)
                        
                        # WxResponse 实现了 __bool__ 方法，可以直接判断
                        if result:
                            print(f"  ✓ 已成功发送回复")
                        else:
                            # WxResponse 继承自 dict，可以直接获取 message
                            error_msg = result.get('message', '未知错误')
                            print(f"  ✗ 发送失败: {error_msg}")
                    except Exception as e:
                        print(f"  ✗ 发送异常: {e}")
                else:
                    print(f"  ✗ 未能获取AI回复，跳过发送")
                
                # 短暂休眠，避免快速连续处理导致的问题
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
                # 每0.5秒检查一次新消息
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
    # 创建并运行消息智能回复机器人
    repeater = MessageRepeater(debug=False)
    repeater.run()

