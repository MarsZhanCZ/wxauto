from wxauto import WeChat
import time
from datetime import datetime

class MessageRepeaterTEST:
    def __init__(self):
        self.wx = WeChat()
        
        print("微信消息复述机器人已启动...")
        print("等待接收新消息，按Ctrl+C停止...")
    
    def process_new_messages(self):
        """处理新消息"""
        try:
            # 获取新消息
            new_msg = self.wx.GetNextNewMessage()
            
            if new_msg:
                for sender, messages in new_msg.items():
                    # 过滤掉系统消息和自己发送的消息
                    if sender == 'Self' or '系统' in sender:
                        continue
                    
                    # 确保messages是列表格式
                    if not isinstance(messages, list):
                        messages = [messages]
                    
                    # 只处理最新的消息（避免重复处理历史消息）
                    # 当没有新消息时，wxauto可能会返回所有消息
                    # 所以我们只处理最后一条消息
                    if len(messages) > 1:
                        # 如果有多条消息，只处理最后一条
                        messages = [messages[-1]]
                    
                    for msg in messages:
                        # 处理FriendMessage对象
                        if hasattr(msg, 'content') and hasattr(msg, 'sender'):
                            msg_content = msg.content
                            msg_sender = msg.sender
                            
                            # 再次检查发送者，确保不是自己
                            if msg_sender == 'Self':
                                continue
                            
                            print(f"\n[{datetime.now().isoformat()}] 收到来自 [{sender}] 的消息: {msg_content}")
                            # 复述消息内容发送回去
                            print(f"正在复述消息给 [{sender}]...")
                            try:
                                self.wx.SendMsg(msg_content, who=sender)
                                print(f"已发送: {msg_content}")
                            except Exception as e:
                                print(f"发送失败: {e}")
                                # 短暂休眠，避免快速连续处理
                                time.sleep(0.25)
                        else:
                            # 如果是列表格式的消息（旧版本兼容）
                            if isinstance(msg, (list, tuple)) and len(msg) >= 3:
                                msg_sender, msg_content, msg_id = msg
                                # 再次检查发送者，确保不是自己
                                if msg_sender == 'Self':
                                    continue
                                # 打印接收到的消息
                                print(f"\n[{datetime.now().isoformat()}] 收到来自 [{sender}] 的消息: {msg_content}")
                                # 复述消息内容发送回去
                                print(f"正在复述消息给 [{sender}]...")
                                try:
                                    self.wx.SendMsg(msg_content, who=sender)
                                    print(f"已发送: {msg_content}")
                                except Exception as e:
                                    print(f"发送失败: {e}")
                                # 短暂休眠，避免快速连续处理
                                time.sleep(1)
                
        except Exception as e:
            print(f"处理消息时发生错误: {e}")
    
    def run(self):
        """运行消息复述机器人"""
        try:
            while True:
                self.process_new_messages()
                time.sleep(0.5)  # 每X秒检查一次新消息
                
        except KeyboardInterrupt:
            print("\n程序已停止")
        except Exception as e:
            print(f"\n发生错误: {e}")
            print("程序已停止")

if __name__ == "__main__":
    repeater = MessageRepeaterTEST()
    repeater.run()