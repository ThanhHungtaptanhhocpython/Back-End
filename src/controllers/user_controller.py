from flask import request, Response, json, Blueprint
from src.services.user_service import getImageData

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
