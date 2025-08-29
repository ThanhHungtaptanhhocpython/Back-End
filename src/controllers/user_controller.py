from flask import request, Response, json, Blueprint
from src.services.user_service import getImageData, getImageDataSingleTextSearch

users = Blueprint("users", __name__)

@users.route('/', methods = ["GET"])
def handle_home():

    res = getImageData()
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
@users.route('/singletextsearch', methods = ["POST"])
def handle_single_text_search():
    data = request.get_json() 
    
    res = getImageDataSingleTextSearch(data["clip"], data["clipv2"], data["query"], data["topk"])

    response_data = {
        "success": True,
        "data": {
            "items": res,
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

    res = getImageData()
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
