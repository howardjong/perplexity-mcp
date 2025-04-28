
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    name: Optional[str] = None
    
class ToolCall(BaseModel):
    tool_id: str
    name: str
    args: Dict[str, Any]
    
class ModelResponse(BaseModel):
    content: str
    role: MessageRole
    tool_calls: List[ToolCall]
    
class ToolResult(BaseModel):
    tool_call_id: str
    content: str
