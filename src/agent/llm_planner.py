import os
import logging
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from src.agent.tools import agent_tools
from src.agent.memory_manager import memory_manager

# Import both so the user can choose via ENV
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# System prompt based on Spatiotemporal Reasoning (STAR) framework and Advanced RAG
SYSTEM_PROMPT = """Bạn là Tác nhân Trí tuệ Nhân tạo (Agentic AI) chuyên về Tìm kiếm Đa phương thức (Multimodal Retrieval) và RAG (Retrieval-Augmented Generation) cho hệ thống AIC 2026.
Nhiệm vụ của bạn là giải quyết các bài toán KIS (Known-Item Search), AVS (Ad-hoc Video Search) và Conversational KIS.

Bạn được trang bị bộ công cụ (Tools) để truy xuất dữ liệu:
1. vector_search_tool: Tìm kiếm video/hình ảnh qua văn bản mô tả (Faiss/CLIP).
2. ocr_search_tool: Tìm chữ xuất hiện trên màn hình video (Elasticsearch).
3. asr_search_tool: Tìm lời thoại của nhân vật trong video (Elasticsearch).
4. temporal_search_tool: Tìm chuỗi hành động theo thời gian (TRAKE).
5. video_qa_tool: Trả lời câu hỏi trực quan chi tiết về khung hình (VQA). Tool này trả về trường 'answer'.

QUY TRÌNH TƯ DUY SPATIOTEMPORAL (STAR) & RAG:
1. Phân tích: Câu hỏi yêu cầu tìm kiếm tổng quan, hay có chứa văn bản/lời thoại? Câu hỏi có yếu tố thời gian trước/sau không?
2. Chọn Tool: Gọi công cụ phù hợp.
3. RAG Synthesis (Tổng hợp): 
   - Khi công cụ trả về kết quả JSON, tuyệt đối KHÔNG in ra chuỗi JSON thô hay các mảng dữ liệu.
   - Hãy trích xuất các thông tin quan trọng như 'video_key', 'frame_key', 'ocr_text', 'answer'.
   - Tổng hợp thành một câu trả lời tự nhiên, thân thiện bằng tiếng Việt.
   - Ví dụ: Nếu user hỏi "Có bao nhiêu người?", bạn gọi video_qa_tool, nhận kết quả {'answer': '2 people', 'video_key': 'L01_V01'}, bạn trả lời: "Dựa vào hình ảnh từ video L01_V01, có 2 người xuất hiện trong phân cảnh này."
   - Gợi ý người dùng cung cấp thêm manh mối (màu sắc, hành động) nếu kết quả chưa rõ ràng.
"""

def get_llm():
    """Khởi tạo LLM dựa trên biến môi trường."""
    if os.getenv("GOOGLE_API_KEY"):
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    elif os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    else:
        logger.warning("No GOOGLE_API_KEY or OPENAI_API_KEY found. Falling back to ChatOpenAI with dummy key for initialization.")
        return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key="dummy")

# Initialize LLM and tools
llm = get_llm()

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent and executor
agent = create_tool_calling_agent(llm, agent_tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=agent_tools, verbose=True)

def execute_chat_turn(session_id: str, user_message: str) -> Dict[str, Any]:
    """
    Thực thi một lượt chat với LLM Planner Agent.
    """
    try:
        # Lấy lịch sử hội thoại
        history = memory_manager.get_messages(session_id)
        
        # Chạy agent
        response = agent_executor.invoke({
            "input": user_message,
            "chat_history": history
        })
        
        output_text = response.get("output", "Tôi không có câu trả lời.")
        
        # Lưu vào memory_manager
        memory_manager.add_user_message(session_id, user_message)
        memory_manager.add_ai_message(session_id, output_text)
        
        return {
            "success": True,
            "response": output_text,
            "data": None
        }
    except Exception as e:
        logger.error(f"Error in LLM Planner execution: {str(e)}")
        # Xử lý lỗi an toàn nếu người dùng chưa cấu hình API Key
        memory_manager.add_user_message(session_id, user_message)
        error_msg = f"Lỗi tác nhân: {str(e)}. (Vui lòng kiểm tra lại API Key trong biến môi trường)"
        memory_manager.add_ai_message(session_id, error_msg)
        return {
            "success": False,
            "response": error_msg,
            "data": None
        }
