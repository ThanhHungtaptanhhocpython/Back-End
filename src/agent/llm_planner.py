import logging
from typing import Any, Dict

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.agent.memory_manager import memory_manager
from src.agent.tools import agent_tools
from src.config.settings import get_settings

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
4. Do not print raw JSON arrays. For every retrieval match, report video_id/video_key, frame_key and timestamp so the user can locate it. Include OCR text or answer only when relevant.
5. A tool result containing an 'error' field is a failed search. State that it failed and never claim that the user's description was found.
6. If results are unclear, ask for more clues such as color, action, object, text, or time relationship.
"""


def _provider_name() -> str:
    return get_settings().llm_provider.strip().lower()


def _is_prompt_limit_error(message: str) -> bool:
    lowered = str(message).lower()
    return "prompt tokens limit exceeded" in lowered or "maximum context length" in lowered


def _safe_error_message(exc: Exception) -> str:
    message = str(exc)
    if "integrate.api.nvidia.com" in message or "Nvcf-Reqid" in message:
        return (
            "NVIDIA provider returned 404. Backend is still using NVIDIA; "
            "switch provider in Back-End/.env and restart uvicorn."
        )
    if "openrouter" in message.lower() and ("401" in message or "authentication" in message.lower()):
        return "OpenRouter authentication failed. Check OPENROUTER_API_KEY and model, then restart uvicorn."
    if _is_prompt_limit_error(message):
        return (
            "Prompt vuot gioi han token cua provider hien tai. "
            "Da xoa bot history chat; hay gui lai cau hoi ngan hon hoac bat dau mot phien chat moi."
        )
    message = " ".join(message.split())
    if len(message) > 400:
        return f"{message[:400]}..."
    return message


def get_llm():
    """Initialize an LLM from provider settings."""
    settings = get_settings()
    provider = _provider_name()
    openai_key = settings.openai_api_key
    openrouter_key = settings.openrouter_api_key
    anthropic_key = settings.anthropic_api_key
    nvidia_key = settings.nvidia_api_key
    google_key = settings.google_api_key

    if provider not in {"auto", "openai", "openrouter", "anthropic", "claude", "nvidia", "nim", "nv", "google", "gemini"}:
        raise ValueError("LLM_PROVIDER must be one of: auto, openai, openrouter, anthropic, claude, nvidia, nim, nv, google, gemini.")

    logger.info("Initializing LLM provider: %s", provider)

    if provider in {"openrouter"} and openrouter_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openrouter_model,
            temperature=0,
            api_key=openrouter_key,
            max_tokens=settings.openrouter_max_tokens,
            base_url=settings.openrouter_base_url,
            default_headers={
                "HTTP-Referer": settings.openrouter_site_url,
                "X-Title": settings.openrouter_app_name,
            },
        )

    if provider in {"auto", "openai"} and openai_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.openai_model, temperature=0)

    if provider in {"auto"} and openrouter_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openrouter_model,
            temperature=0,
            api_key=openrouter_key,
            max_tokens=settings.openrouter_max_tokens,
            base_url=settings.openrouter_base_url,
            default_headers={
                "HTTP-Referer": settings.openrouter_site_url,
                "X-Title": settings.openrouter_app_name,
            },
        )

    if provider in {"auto", "anthropic", "claude"} and anthropic_key:
        try:
            from langchain_anthropic import ChatAnthropic
        except Exception as exc:
            raise RuntimeError(
                "Anthropic/Claude provider is selected but langchain-anthropic cannot be imported. "
                "Install langchain-anthropic or use LLM_PROVIDER=openai."
            ) from exc

        return ChatAnthropic(
            model=settings.anthropic_model,
            temperature=0,
            max_tokens=settings.anthropic_max_tokens,
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
            model=settings.nvidia_model,
            temperature=0,
            max_tokens=settings.nvidia_max_tokens,
            top_p=settings.nvidia_top_p,
        )

    if provider in {"auto", "google", "gemini"} and google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception as exc:
            raise RuntimeError(
                "Google/Gemini provider is selected but langchain-google-genai cannot be imported. "
                "Use LLM_PROVIDER=openai or align langchain-google-genai with langchain-core."
            ) from exc

        return ChatGoogleGenerativeAI(model=settings.google_model, temperature=0)

    raise ValueError(
        "Missing API key for LLM planner. Set LLM_PROVIDER=openai with OPENAI_API_KEY, "
        "LLM_PROVIDER=openrouter with OPENROUTER_API_KEY, "
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
    _agent_executor = AgentExecutor(agent=agent, tools=agent_tools, verbose=True, handle_parsing_errors=True, return_intermediate_steps=True)
    return _agent_executor


def _extract_tool_frames(intermediate_steps) -> list[dict]:
    """Flatten tool observations into safe, structured frame locators."""
    frames: list[dict] = []
    seen: set[tuple] = set()

    for _action, observation in intermediate_steps or []:
        if not isinstance(observation, list):
            continue
        for result in observation:
            if not isinstance(result, dict) or result.get("error"):
                continue

            nested = result.get("keyframes")
            candidates = nested if isinstance(nested, list) else [result]
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                item = dict(candidate)
                video_id = (
                    item.get("video_id")
                    or item.get("video_key")
                    or result.get("video_id")
                    or result.get("video_key")
                )
                frame_key = (
                    item.get("frame_key")
                    or item.get("frame_name")
                    or item.get("frame_id")
                )
                timestamp = item.get("timestamp")
                if not video_id or (frame_key is None and timestamp is None):
                    continue

                item.setdefault("video_id", video_id)
                item.setdefault("video_key", video_id)
                if frame_key is not None:
                    item.setdefault("frame_key", frame_key)
                    item.setdefault("frame_name", str(frame_key))
                identity = (str(video_id), str(frame_key), timestamp)
                if identity in seen:
                    continue
                seen.add(identity)
                frames.append(item)
                if len(frames) >= 24:
                    return frames

    return frames

def execute_chat_turn(session_id: str, user_message: str) -> Dict[str, Any]:
    """Run one conversational turn through the LLM planner agent."""
    try:
        history = memory_manager.get_recent_messages(session_id)
        response = get_agent_executor().invoke({
            "input": user_message,
            "chat_history": history,
        })

        output_text = response.get("output", "Toi khong co cau tra loi.")
        tool_frames = _extract_tool_frames(response.get("intermediate_steps", []))

        memory_manager.add_user_message(session_id, user_message)
        memory_manager.add_ai_message(session_id, output_text)

        return {
            "success": True,
            "response": output_text,
            "data": {"frames": tool_frames},
        }
    except Exception as exc:
        safe_message = _safe_error_message(exc)
        logger.error("Error in LLM Planner execution: %s", safe_message)
        if _is_prompt_limit_error(str(exc)):
            memory_manager.clear_session(session_id)
        return {
            "success": False,
            "response": f"Loi tac nhan: {safe_message}",
            "data": None,
        }


