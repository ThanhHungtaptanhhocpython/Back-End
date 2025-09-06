from flask import request, Response, json, Blueprint
from src.services.user_service import getImageDataSingleTextSearch, getImageDataQAndASearch, GetImageDataTrakeSearch

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
@users.route('/trakesearch', methods=["POST"])
def handle_trake_search():
    data = request.get_json()
    query = data.get("query")
    top_k = data.get("topk", 100) # Default to 100
   

    if not query:
        return Response(
            response=json.dumps({"success": False, "message": "Query parameter 'query' is required."}),
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