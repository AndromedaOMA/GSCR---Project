import sys
import pathlib
from flask import Flask, request, jsonify
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler

sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

from src.models import load_model, generate_corrections
from database.database import store_feedback, init_db
from src.active_learning import run_active_learning

DB_PATH = "feedback.db"
init_db(DB_PATH)

# Load model here once to avoid reloading on every request
model_path = "./t5-grammar-finetuned"
model, tokenizer = load_model(model_path)

# Schedule the active learning process to run every 3 days(number of days can be changed)
scheduler = BackgroundScheduler()
scheduler.add_job(
    run_active_learning,
    'interval',
    days=3,
    kwargs={"model": model, "tokenizer": tokenizer, "db_path": "feedback.db"}
)
scheduler.start()

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
    # corrected_text = text.replace("teh", "the").replace("recieve", "receive")
    suggestions = generate_corrections(model, tokenizer, text, num_suggestions=5)
    corrected = suggestions[0] if suggestions else text

    response = jsonify({
        "original": text,
        "corrected": corrected,
        "suggestions": suggestions
    })
    return add_cors_headers(response)

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json(force=True)
    orig        = data.get("original")
    suggestions = data.get("suggestions", [])
    chosen      = data.get("chosen")   # index sau textul selectat

    if not orig or chosen is None:
        return jsonify({"error": "Insufficient payload"}), 400

    # stochează în DB pentru antrenamente viitoare
    store_feedback(original=orig,
                   suggestions=suggestions,
                   chosen=chosen,
                   db_path="feedback.db")

    return add_cors_headers(jsonify({"status": "ok"}))
