"""DEPRECATED: User controller for search API endpoints (Flask).

This module contains the legacy Flask route controllers.
It is deprecated in favor of the new FastAPI routers in `src/api/routers/`.
"""

from flask import request, Response, json, Blueprint

from src.services.user_service import (
    getImageDataSingleTextSearch,
    getImageDataQAndASearch,
    getImageSearchById,
    GetImageDataTrakeSearch,
    getImageSearchByFile,
)

users = Blueprint("users", __name__)

# ---------------------------------------------------------------------------
# Default values for optional parameters
# ---------------------------------------------------------------------------
DEFAULT_TOPK: int = 100


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _error_response(message: str, status: int = 400) -> Response:
    """Build a standardised JSON error response.

    Args:
        message: Human-readable error description.
        status: HTTP status code (default 400).

    Returns:
        A Flask ``Response`` with the unified error schema.
    """
    payload = {
        "success": False,
        "data": None,
        "message": message,
    }
    return Response(
        response=json.dumps(payload, indent=2),
        status=status,
        mimetype="application/json",
    )


def _success_response(items: list) -> Response:
    """Build a standardised JSON success response.

    Args:
        items: List of result dicts to return.

    Returns:
        A Flask ``Response`` with the unified success schema.
    """
    payload = {
        "success": True,
        "data": {
            "items": items,
            "total_items": len(items),
        },
        "message": None,
    }
    return Response(
        response=json.dumps(payload, indent=2),
        status=200,
        mimetype="application/json",
    )


def _parse_topk(raw_value: str | int | None) -> int | None:
    """Validate and return *topk* as a positive integer.

    Args:
        raw_value: The raw value from the request payload.

    Returns:
        A validated positive integer, or ``None`` if the value is invalid.
    """
    if raw_value is None:
        return DEFAULT_TOPK
    try:
        value = int(raw_value)
        if value <= 0:
            return None
        return value
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@users.route('/singletextsearch', methods=["POST"])
def handle_single_text_search() -> Response:
    """Search keyframes by a single text query using CLIP similarity.

    Required JSON fields:
        query (str): The text search query.

    Optional JSON fields:
        topk (int): Number of results to return (default 100).

    Returns:
        JSON response with matching keyframe items or a 400 error.
    """
    data: dict | None = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _error_response("Request body must be a valid JSON object.")

    query: str | None = data.get("query")
    if not query or not isinstance(query, str) or not query.strip():
        return _error_response("Missing required field: query")

    topk: int | None = _parse_topk(data.get("topk"))
    if topk is None:
        return _error_response("topk must be a positive integer.")

    res = getImageDataSingleTextSearch(query.strip(), topk)
    return _success_response(res)


@users.route('/qnasearch', methods=["POST"])
def handle_qna_search() -> Response:
    """Search keyframes by text query and answer each result using VLM.

    Required JSON fields:
        query (str): The text search query.

    Optional JSON fields:
        topk (int): Number of results to return (default 100).

    Returns:
        JSON response with matching keyframe items (with VLM answers)
        or a 400 error.
    """
    data: dict | None = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _error_response("Request body must be a valid JSON object.")

    query: str | None = data.get("query")
    if not query or not isinstance(query, str) or not query.strip():
        return _error_response("Missing required field: query")

    topk: int | None = _parse_topk(data.get("topk"))
    if topk is None:
        return _error_response("topk must be a positive integer.")

    res = getImageDataQAndASearch(query.strip(), topk)
    return _success_response(res)


@users.route('/imagesearch', methods=["POST"])
def handle_image_search() -> Response:
    """Search keyframes by an uploaded image file or a Faiss index ID.

    Form fields:
        image (file, optional): An uploaded image file.
        faiss_index (str, optional): A Faiss index ID for similarity search.
        topk (int, optional): Number of results to return (default 100).

    At least one of ``image`` or ``faiss_index`` must be provided.

    Returns:
        JSON response with matching keyframe items or a 400 error.
    """
    topk: int | None = _parse_topk(request.form.get("topk"))
    if topk is None:
        return _error_response("topk must be a positive integer.")

    file = request.files.get("image")
    faiss_index: str | None = request.form.get("faiss_index", "default")

    if file and file.filename != '':
        res = getImageSearchByFile(file, topk)
    elif faiss_index and faiss_index not in ("default", "null"):
        try:
            faiss_index_int = int(faiss_index)
        except (TypeError, ValueError):
            return _error_response("faiss_index must be an integer.")
        res = getImageSearchById(faiss_index_int, topk)
    else:
        return _error_response(
            "Either an uploaded image file or a valid faiss_index must be "
            "provided."
        )

    return _success_response(res)


@users.route('/temporalsearch', methods=["POST"])
def handle_temporal_search() -> Response:
    """Search keyframes using temporal (multi-event) queries.

    Required JSON fields:
        query (list[dict]): A non-empty list of event objects, each
            containing a ``query`` string.

    Optional JSON fields:
        topk (int): Number of results to return (default 100).

    Returns:
        JSON response with matching keyframe items or a 400 error.
    """
    data: dict | None = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _error_response("Request body must be a valid JSON object.")

    query = data.get("query")
    if not query or not isinstance(query, list) or len(query) == 0:
        return _error_response(
            "Missing required field: query (must be a non-empty list of "
            "events)."
        )

    for item in query:
        if (
            not isinstance(item, dict)
            or not item.get("query")
            or not isinstance(item.get("query"), str)
            or not item.get("query").strip()
        ):
            return _error_response(
                "Each event in query list must have a non-empty 'query' "
                "string."
            )

    topk: int | None = _parse_topk(data.get("topk"))
    if topk is None:
        return _error_response("topk must be a positive integer.")

    res = GetImageDataTrakeSearch(query, top_results=topk)
    return _success_response(res)


@users.route('/ocrandodsearch', methods=["POST"])
def handle_ocr_and_od_search() -> Response:
    """Placeholder endpoint for OCR and object-detection search.

    Returns:
        A 501 Not Implemented response.
    """
    return _error_response("OCR/OD search is not implemented yet.", status=501)
