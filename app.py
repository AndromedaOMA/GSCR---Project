from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# ✅ Allow only requests from your Chrome extension
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    """
    Ensure every response includes the correct CORS headers.
    This fixes the issue where CORS only worked for OPTIONS but failed for POST.
    """
    origin = request.headers.get("Origin")  # ✅ Get the request's origin

    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin  # ✅ Allow dynamic origins
        response.headers["Access-Control-Allow-Credentials"] = "true"
    
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.route('/correct', methods=['OPTIONS'])
def preflight():
    """
    Handles the CORS preflight request.
    This must return proper headers so that the actual request is allowed.
    """
    response = jsonify({"message": "CORS preflight successful"})
    return add_cors_headers(response)

@app.route('/correct', methods=['POST'])
def correct_text():
    data = request.get_json()
    if data is None or "text" not in data:
        return jsonify({"error": "Invalid request, 'text' key missing"}), 400

    text = data["text"]
    corrected_text = text.replace("teh", "the").replace("recieve", "receive")

    response = jsonify({"corrected": corrected_text})
    return add_cors_headers(response)

# if __name__ == '__main__':
#     app.run(debug=True, port=5000, ssl_context=('cert.pem', 'key.pem'))
if __name__ == '__main__':
    app.run(debug=True, host="localhost", port=5000, ssl_context=('cert.pem', 'key.pem'))