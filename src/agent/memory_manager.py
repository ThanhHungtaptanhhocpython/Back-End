import uuid
from typing import Dict, Any, List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessage

class MemoryManager:
    """
    Manages conversational memory for the Agentic AI.
    For Phase 1, it uses in-memory dict to store ChatMessageHistory per session.
    """
    def __init__(self):
        self.sessions: Dict[str, ChatMessageHistory] = {}

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatMessageHistory()
        return self.sessions[session_id]

    def add_user_message(self, session_id: str, message: str) -> None:
        history = self.get_session_history(session_id)
        history.add_user_message(message)

    def add_ai_message(self, session_id: str, message: str) -> None:
        history = self.get_session_history(session_id)
        history.add_ai_message(message)

    def get_messages(self, session_id: str) -> List[BaseMessage]:
        return self.get_session_history(session_id).messages

    def clear_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            self.sessions[session_id].clear()

# Global instance for easy access across routers
memory_manager = MemoryManager()
