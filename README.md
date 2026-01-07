# wxauto

A Python library for automating WeChat interactions with a comprehensive suite of tools for message automation, contact management, and bot development.

## Project Overview

wxauto is a powerful automation framework designed to interact with WeChat through its native Windows interface. It provides a high-level API for automating common WeChat operations, enabling developers to build sophisticated WeChat bots, automation scripts, and integration tools without deep knowledge of WeChat's internal architecture.

This project is particularly useful for:
- Building WeChat chatbots and automation systems
- Automating repetitive messaging tasks
- Integrating WeChat with other services
- Creating custom WeChat workflows
- Testing and quality assurance automation

## Features

- **Message Automation**: Send and receive messages programmatically
- **Contact Management**: Access and manage WeChat contacts efficiently
- **Chat Management**: Handle individual chats, group chats, and channels
- **File Transfer**: Send and receive files through WeChat
- **User Information**: Retrieve and manage user profile data
- **Group Operations**: Create, modify, and manage WeChat groups
- **Message Search**: Search through chat history
- **Event Handling**: React to incoming messages and user events
- **Session Management**: Handle multiple WeChat sessions
- **Error Recovery**: Built-in error handling and recovery mechanisms

## Installation Guide

### Prerequisites

- Python 3.7 or higher
- Windows operating system (7, 8, 10, 11 or later)
- WeChat application installed and accessible

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/MarsZhanCZ/wxauto.git
cd wxauto
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python -c "import wxauto; print(wxauto.__version__)"
```

### Optional Dependencies

For enhanced functionality, install optional dependencies:
```bash
pip install -r requirements-dev.txt  # For development
pip install -r requirements-test.txt # For testing
```

## Usage Examples

### Basic Setup

```python
from wxauto import WeChat

# Initialize WeChat connection
wechat = WeChat()

# Verify connection
if wechat.status:
    print("Successfully connected to WeChat")
```

### Sending Messages

```python
# Send text message to a contact
wechat.send_message("Contact Name", "Hello, this is an automated message!")

# Send message to a group
wechat.send_message("Group Name", "Group message from automation script")

# Send with formatting
wechat.send_message("Contact Name", "This is a **bold** message")
```

### Receiving Messages

```python
# Listen for incoming messages
def handle_message(message):
    sender = message.from_user
    content = message.content
    print(f"Received from {sender}: {content}")

wechat.on_message(handle_message)
wechat.run()
```

### Contact Management

```python
# Get all contacts
contacts = wechat.get_contacts()
for contact in contacts:
    print(f"Contact: {contact.name}, ID: {contact.user_id}")

# Get contact details
contact = wechat.get_contact("Contact Name")
print(f"Nickname: {contact.nickname}")
print(f"Signature: {contact.signature}")

# Search for contacts
results = wechat.search_contacts("keyword")
```

### File Operations

```python
# Send a file
wechat.send_file("Contact Name", "/path/to/file.pdf")

# Send multiple files
files = ["/path/to/file1.txt", "/path/to/file2.pdf"]
for file_path in files:
    wechat.send_file("Contact Name", file_path)
```

### Group Management

```python
# Create a new group
members = ["Contact1", "Contact2", "Contact3"]
group = wechat.create_group("New Group", members)

# Get group information
group_info = wechat.get_group("Group Name")
print(f"Group members: {group_info.members}")

# Add member to group
wechat.add_group_member("Group Name", "Contact Name")

# Remove member from group
wechat.remove_group_member("Group Name", "Contact Name")
```

### Message Search

```python
# Search messages in a chat
messages = wechat.search_messages("Chat Name", "keyword")
for msg in messages:
    print(f"{msg.time}: {msg.content}")

# Search within date range
from datetime import datetime, timedelta
start_date = datetime.now() - timedelta(days=7)
messages = wechat.search_messages("Chat Name", "keyword", 
                                 start_date=start_date)
```

## Architecture

### Component Structure

```
wxauto/
├── core/
│   ├── wechat.py           # Main WeChat interface
│   ├── message.py          # Message handling
│   └── contact.py          # Contact management
├── automation/
│   ├── sender.py           # Message sending
│   ├── receiver.py         # Message receiving
│   └── listener.py         # Event listening
├── utils/
│   ├── logger.py           # Logging utilities
│   ├── exceptions.py       # Custom exceptions
│   └── validators.py       # Input validation
├── models/
│   ├── message_model.py    # Message data models
│   ├── contact_model.py    # Contact data models
│   └── group_model.py      # Group data models
└── config/
    └── settings.py         # Configuration management
```

### Key Classes

- **WeChat**: Main class for interacting with WeChat
- **Message**: Represents a WeChat message
- **Contact**: Represents a WeChat contact
- **Group**: Represents a WeChat group
- **Session**: Manages WeChat session lifecycle

### Design Patterns

- **Singleton Pattern**: WeChat connection management
- **Observer Pattern**: Event handling and message listening
- **Factory Pattern**: Message and contact creation
- **Strategy Pattern**: Different communication strategies

## Troubleshooting

### Common Issues and Solutions

#### Issue: "WeChat Not Found"
**Cause**: WeChat application is not installed or not running
**Solution**: 
- Ensure WeChat is installed on your system
- Launch WeChat application before running the script
- Check that WeChat version is compatible (latest version recommended)

#### Issue: "Connection Failed"
**Cause**: Unable to establish communication with WeChat
**Solution**:
- Restart WeChat application
- Check Windows firewall settings
- Ensure only one instance of the automation script is running
- Verify WeChat is not in offline mode

#### Issue: "Message Not Sent"
**Cause**: Contact not found or blocked
**Solution**:
- Verify contact name is correct (case-sensitive)
- Check that you're not blocked by the contact
- Ensure WeChat is actively running with focus
- Wait a moment between consecutive messages

#### Issue: "Timeout Errors"
**Cause**: WeChat is slow to respond
**Solution**:
- Increase timeout settings in configuration
- Check system resources and close unnecessary applications
- Ensure stable network connection
- Reduce the frequency of operations

#### Issue: "Permission Denied"
**Cause**: Insufficient permissions or account restrictions
**Solution**:
- Ensure WeChat account is in good standing
- Check that you have permission to access the target chat
- Verify no security restrictions are in place
- Try with a different WeChat account if available

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
from wxauto import WeChat

logging.basicConfig(level=logging.DEBUG)
wechat = WeChat(debug=True)
```

### Getting Help

If issues persist:
1. Check the [Issues](https://github.com/MarsZhanCZ/wxauto/issues) section
2. Review the [Documentation](https://github.com/MarsZhanCZ/wxauto/wiki)
3. Enable debug mode and collect logs
4. Create a detailed issue report with logs and reproduction steps

## Disclaimer

**IMPORTANT LEGAL AND ETHICAL NOTICE**

This project is provided for educational and legitimate automation purposes only. Users are responsible for:

1. **Compliance with Laws**: Ensure your use complies with all applicable laws and regulations in your jurisdiction, including:
   - WeChat Terms of Service
   - Local privacy laws
   - Data protection regulations (GDPR, CCPA, etc.)

2. **Ethical Usage**:
   - Do not use for spam, harassment, or malicious activities
   - Respect user privacy and data protection
   - Obtain proper consent before automating communications on behalf of others
   - Do not attempt to breach WeChat security or authentication mechanisms

3. **Account Responsibility**:
   - You are solely responsible for any consequences of using this tool
   - Unauthorized automation may violate WeChat Terms of Service
   - Your WeChat account may be suspended or banned
   - The authors assume no liability for account restrictions or bans

4. **Liability**:
   - This software is provided "as-is" without warranty
   - The authors are not responsible for:
     - Data loss or corruption
     - Account suspensions or bans
     - Legal consequences of misuse
     - Any damages resulting from the use of this tool

5. **Usage Restrictions**:
   - Do not use for commercial spam
   - Do not harvest data without consent
   - Do not interfere with WeChat's normal operation
   - Do not attempt to reverse-engineer or modify WeChat internals

**By using this project, you agree to:**
- Use it only for legitimate, legal purposes
- Take full responsibility for your actions
- Comply with all applicable laws and terms of service
- Hold harmless the authors and contributors

For questions about appropriate usage, please refer to WeChat's official guidelines and terms of service.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the project's style guidelines and includes appropriate tests.

## Support

For support, questions, or bug reports:
- Open an issue on [GitHub Issues](https://github.com/MarsZhanCZ/wxauto/issues)
- Check existing documentation and examples
- Review the troubleshooting section above

## Acknowledgments

Thanks to all contributors who have helped make this project better!

## Author

**MarsZhanCZ**

---

Last Updated: 2026-01-07
