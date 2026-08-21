import logging
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.agent.memory_manager import memory_manager
from src.agent.tools import agent_tools

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=True)

logger = logging.getLogger(__name__)
_agent_executor = None

SYSTEM_PROMPT = """You are an Agentic AI assistant for AIC 2026 multimodal video retrieval.
Your job is to help with KIS, AVS, and Conversational KIS tasks.

Available tools:
1. vector_search_tool: search video frames by natural-language visual description.
2. ocr_search_tool: search text visible on frames.
3. asr_search_tool: search speech/transcript content.
4. temporal_search_tool: search sequences of actions/events over time.
5. video_qa_tool: answer visual questions about a specific frame; it may return an 'answer' field.

Reasoning workflow:
1. Analyze whether the user needs broad visual search, OCR, ASR, VQA, or temporal search.
2. Call the most relevant tool or tools.
3. Synthesize results in natural Vietnamese.
4. Do not print raw JSON arrays. Extract useful fields such as video_key, frame_key, ocr_text, and answer.
5. If results are unclear, ask for more clues such as color, action, object, text, or time relationship.
"""




def _safe_error_message(exc: Exception) -> str:
    message = str(exc)
    if "integrate.api.nvidia.com" in message or "Nvcf-Reqid" in message:
        return (
            "NVIDIA provider returned 404. Backend is still using NVIDIA; "
            "set LLM_PROVIDER=anthropic in Back-End/.env and restart uvicorn."
        )
    return message
def _provider_name() -> str:
    return os.getenv("LLM_PROVIDER", "auto").strip().lower()


def get_llm():
    """Initialize an LLM from explicit provider env vars."""
    provider = _provider_name()
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if provider not in {"auto", "openai", "anthropic", "claude", "nvidia", "nim", "nv", "google", "gemini"}:
        raise ValueError("LLM_PROVIDER must be one of: auto, openai, anthropic, claude, nvidia, nim, nv, google, gemini.")

    logger.info("Initializing LLM provider: %s", provider)

    if provider in {"auto", "openai"} and openai_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

    if provider in {"auto", "anthropic", "claude"} and anthropic_key:
        try:
            from langchain_anthropic import ChatAnthropic
        except Exception as exc:
            raise RuntimeError(
                "Anthropic/Claude provider is selected but langchain-anthropic cannot be imported. "
                "Install langchain-anthropic or use LLM_PROVIDER=openai."
            ) from exc

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
            temperature=0,
            max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "2048")),
        )

    if provider in {"auto", "nvidia", "nim", "nv"} and nvidia_key:
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
        except Exception as exc:
            raise RuntimeError(
                "NVIDIA provider is selected but langchain-nvidia-ai-endpoints cannot be imported. "
                "Install langchain-nvidia-ai-endpoints or use LLM_PROVIDER=openai."
            ) from exc

        return ChatNVIDIA(
            model=os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b"),
            temperature=0,
            max_tokens=int(os.getenv("NVIDIA_MAX_TOKENS", "2048")),
            top_p=float(os.getenv("NVIDIA_TOP_P", "1.0")),
        )

    if provider in {"auto", "google", "gemini"} and google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception as exc:
            raise RuntimeError(
                "Google/Gemini provider is selected but langchain-google-genai cannot be imported. "
                "Use LLM_PROVIDER=openai or align langchain-google-genai with langchain-core."
            ) from exc

        return ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"), temperature=0)

    raise ValueError(
        "Missing API key for LLM planner. Set LLM_PROVIDER=openai with OPENAI_API_KEY, "
        "LLM_PROVIDER=anthropic with ANTHROPIC_API_KEY, LLM_PROVIDER=nvidia with NVIDIA_API_KEY, "
        "or LLM_PROVIDER=google with GOOGLE_API_KEY."
    )


def get_agent_executor():
    """Create the tool-calling agent lazily so FastAPI can start before keys are configured."""
    global _agent_executor
    if _agent_executor is not None:
        return _agent_executor

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(get_llm(), agent_tools, prompt)
    _agent_executor = AgentExecutor(agent=agent, tools=agent_tools, verbose=True, handle_parsing_errors=True)
    return _agent_executor


def execute_chat_turn(session_id: str, user_message: str) -> Dict[str, Any]:
    """Run one conversational turn through the LLM planner agent."""
    try:
        history = memory_manager.get_messages(session_id)
        response = get_agent_executor().invoke({
            "input": user_message,
            "chat_history": history,
        })

        output_text = response.get("output", "Toi khong co cau tra loi.")

        memory_manager.add_user_message(session_id, user_message)
        memory_manager.add_ai_message(session_id, output_text)

        return {
            "success": True,
            "response": output_text,
            "data": None,
        }
    except Exception as exc:
        safe_message = _safe_error_message(exc)
        logger.error("Error in LLM Planner execution: %s", safe_message)
        memory_manager.add_user_message(session_id, user_message)
        error_msg = f"Loi tac nhan: {safe_message}. (Vui long kiem tra API key/provider trong bien moi truong)"
        memory_manager.add_ai_message(session_id, error_msg)
        return {
            "success": False,
            "response": error_msg,
            "data": None,
        }