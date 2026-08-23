import os
from typing import Dict, List

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage


class MemoryManager:
    """
    Manages conversational memory for the Agentic AI.
    For Phase 1, it uses in-memory dict to store ChatMessageHistory per session.
    """

    def __init__(self):
        self.sessions: Dict[str, ChatMessageHistory] = {}
        self.max_messages = max(2, int(os.getenv("CHAT_HISTORY_MESSAGES", "6")))

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatMessageHistory()
        return self.sessions[session_id]

    def _trim_history(self, session_id: str) -> None:
        history = self.get_session_history(session_id)
        if len(history.messages) > self.max_messages:
            history.messages[:] = history.messages[-self.max_messages:]

    def add_user_message(self, session_id: str, message: str) -> None:
        history = self.get_session_history(session_id)
        history.add_user_message(message)
        self._trim_history(session_id)

    def add_ai_message(self, session_id: str, message: str) -> None:
        history = self.get_session_history(session_id)
        history.add_ai_message(message)
        self._trim_history(session_id)

    def get_messages(self, session_id: str) -> List[BaseMessage]:
        return self.get_session_history(session_id).messages

    def get_recent_messages(self, session_id: str, limit: int | None = None) -> List[BaseMessage]:
        messages = self.get_session_history(session_id).messages
        effective_limit = self.max_messages if limit is None else max(1, limit)
        return list(messages[-effective_limit:])

    def clear_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            self.sessions[session_id].clear()


# Global instance for easy access across routers
memory_manager = MemoryManager()
