from langchain_core.tools import tool
from typing import List, Dict, Any
import logging

from src.services.user_service import (
    getImageDataSingleTextSearch,
    getTextSearchOCR,
    getTextSearchASR,
    GetImageDataTrakeSearch,
    getImageDataQAndASearch
)

logger = logging.getLogger(__name__)

@tool
def vector_search_tool(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Tìm kiếm semantic vector (Faiss + CLIP) dựa trên text query.
    Sử dụng tool này khi câu hỏi yêu cầu tìm kiếm nội dung hình ảnh, màu sắc, hành động chung chung trong video.
    Ví dụ: 'người đàn ông mặc áo đỏ đang chạy', 'chiếc xe màu xanh'.
    """
    logger.info(f"Agent called vector_search_tool with query: {query}")
    try:
        results = getImageDataSingleTextSearch(query, top_k)
        # Lược bỏ trường 'image' base64 dài dòng khi trả về cho LLM để tránh đầy context window
        for res in results:
            if 'image' in res:
                del res['image']
        return results
    except Exception as e:
        return [{"error": str(e)}]

@tool
def ocr_search_tool(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Tìm kiếm văn bản xuất hiện trên màn hình video (OCR - Optical Character Recognition).
    Sử dụng tool này khi người dùng tìm kiếm CÁC TỪ NGỮ CỤ THỂ, VĂN BẢN, BIỂN BÁO, LOGO có trong khung hình.
    Ví dụ: 'chữ trên bảng', 'biển số xe', 'dòng chữ tin tức'.
    """
    logger.info(f"Agent called ocr_search_tool with query: {query}")
    try:
        results = getTextSearchOCR(query, top_k)
        return results
    except Exception as e:
        return [{"error": str(e)}]

@tool
def asr_search_tool(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Tìm kiếm giọng nói/lời thoại trong video (ASR - Automatic Speech Recognition).
    Sử dụng tool này khi câu hỏi liên quan đến lời nói, hội thoại, người ta đang nói gì.
    Ví dụ: 'họ đang nói chuyện về gì', 'MC nói từ khóa'.
    """
    logger.info(f"Agent called asr_search_tool with query: {query}")
    try:
        results = getTextSearchASR(query, top_k)
        return results
    except Exception as e:
        return [{"error": str(e)}]

@tool
def temporal_search_tool(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Tìm kiếm chuỗi sự kiện theo thời gian (Temporal Search / TRAKE).
    Sử dụng tool này khi câu hỏi có yếu tố thời gian, trước/sau, hoặc một chuỗi các hành động liên tiếp.
    Ví dụ: 'người đàn ông chạy SAU ĐÓ ngã', 'trước tiên..., sau đó...'.
    """
    logger.info(f"Agent called temporal_search_tool with query: {query}")
    try:
        results = GetImageDataTrakeSearch(query, top_results=top_k)
        for res in results:
            if 'image' in res:
                del res['image']
        return results
    except Exception as e:
        return [{"error": str(e)}]

@tool
def video_qa_tool(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Tìm kiếm kết hợp trả lời câu hỏi trực quan (Video QA) sử dụng mô hình VQA.
    Sử dụng tool này khi câu hỏi yêu cầu giải thích chi tiết về một phân cảnh, 
    đếm số lượng, hoặc trả lời một câu hỏi cụ thể dựa trên hình ảnh.
    Ví dụ: 'Có bao nhiêu người trong khung cảnh này?', 'Người đàn ông đang cầm gì?'.
    Lưu ý: Tool này chạy chậm (Slow thinking), chỉ nên gọi khi thực sự cần thiết.
    """
    logger.info(f"Agent called video_qa_tool with query: {query}")
    try:
        results = getImageDataQAndASearch(query, top_k)
        for res in results:
            if 'image' in res:
                del res['image']
        return results
    except Exception as e:
        return [{"error": str(e)}]

# List of all tools to bind to the agent
agent_tools = [
    vector_search_tool,
    ocr_search_tool,
    asr_search_tool,
    temporal_search_tool,
    video_qa_tool
]
