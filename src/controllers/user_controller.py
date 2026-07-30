from flask import request, Response, json, Blueprint
from src.services.user_service import getImageDataSingleTextSearch, getImageDataQAndASearch, getImageSearchById, GetImageDataTrakeSearch, getImageSearchByFile

users = Blueprint("users", __name__)

@users.route('/singletextsearch', methods = ["POST"])
def handle_single_text_search():
    data = request.get_json() 
    
    res = getImageDataSingleTextSearch(data["query"], data["topk"])

    response_data = {
        "success": True,
        "data": {
            "items": res,
            "total_items": len(res)
        }
    }
    return Response(
        response=json.dumps(response_data, indent=2),
        status=200,
        mimetype="application/json"
    )

@users.route('/qnasearch', methods = ["POST"])
def handle_qna_search():
    data = request.get_json()
    print(data)

    res = getImageDataQAndASearch(data["query"], data["topk"])
    response_data = {
        "success": True,
        "data": {
            "items": res,
            "total_items": len(res)
        }
    }
    return Response(
        response=json.dumps(response_data, indent=2),
        status=200,
        mimetype="application/json"
    )

@users.route('/imagesearch', methods = ["POST"])
def handle_image_search():
    try:
        topk_str = request.form.get("topk")
        if topk_str is None:
            topk = 100
        else:
            topk = int(topk_str)
            if topk <= 0:
                raise ValueError()
    except (TypeError, ValueError):
        return Response(
            response=json.dumps({"success": False, "message": "topk must be a positive integer.", "data": {"items": [], "total_items": 0}}),
            status=400,
            mimetype="application/json"
        )

    file = request.files.get("image")
    faiss_index = request.form.get("faiss_index", "default")
    
    res = []
    if file and file.filename != '':
        res = getImageSearchByFile(file, topk)
    elif faiss_index and faiss_index not in ("default", "null"):
        try:
            faiss_index_int = int(faiss_index)
        except (TypeError, ValueError):
            return Response(
                response=json.dumps({"success": False, "message": "faiss_index must be an integer.", "data": {"items": [], "total_items": 0}}),
                status=400,
                mimetype="application/json"
            )
        res = getImageSearchById(faiss_index_int, topk)
    else:
        return Response(
            response=json.dumps({"success": False, "message": "Either an uploaded image file or a valid faiss_index must be provided.", "data": {"items": [], "total_items": 0}}),
            status=400,
            mimetype="application/json"
        )

    response_data = {
        "success": True,
        "data": {
            "items": res,
            "total_items": len(res)
        }
    }
    return Response(
        response=json.dumps(response_data, indent=2),
        status=200,
        mimetype="application/json"
    )

@users.route('/trakesearch', methods=["POST"])
@users.route('/temporalsearch', methods=["POST"])
def handle_trake_search():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return Response(
            response=json.dumps({"success": False, "message": "Request must be a JSON object.", "data": {"items": [], "total_items": 0}}),
            status=400,
            mimetype="application/json"
        )
        
    query = data.get("query")
    if not query or not isinstance(query, list) or len(query) == 0:
        return Response(
            response=json.dumps({"success": False, "message": "Query must be a non-empty list of events.", "data": {"items": [], "total_items": 0}}),
            status=400,
            mimetype="application/json"
        )
        
    for item in query:
        if not isinstance(item, dict) or not item.get("query") or not isinstance(item.get("query"), str) or not item.get("query").strip():
            return Response(
                response=json.dumps({"success": False, "message": "Each event in query list must have a non-empty 'query' string.", "data": {"items": [], "total_items": 0}}),
                status=400,
                mimetype="application/json"
            )

    try:
        top_k = data.get("topk")
        if top_k is None:
            top_k = 100
        top_k = int(top_k)
        if top_k <= 0:
            raise ValueError()
    except (TypeError, ValueError):
        return Response(
            response=json.dumps({"success": False, "message": "topk must be a positive integer.", "data": {"items": [], "total_items": 0}}),
            status=400,
            mimetype="application/json"
        )
    
    res = GetImageDataTrakeSearch(query, top_results = top_k)
    response_data = {
        "success": True,
        "data": {
            "items": res,
            "total_items": len(res)
        }
    }
    return Response(
        response=json.dumps(response_data, indent=2),
        status=200,
        mimetype="application/json"
    )

@users.route('/ocrandodsearch', methods=["POST"])
def handle_ocr_and_od_search():
    # Explicit placeholder to prevent silent empty successful results
    response_data = {
        "success": False,
        "message": "OCR/OD search is not implemented yet.",
        "data": {
            "items": [],
            "total_items": 0
        }
    }
    return Response(
        response=json.dumps(response_data, indent=2),
        status=501,
        mimetype="application/json"
    )