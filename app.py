import os
import re
import sys
import pathlib
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
from transformers import (
    AutoTokenizer,
)

from src.wordnet.wordnet import get_related_forms

sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

from src.models import load_model, generate_corrections
from database.database import store_feedback, init_db
from src.active_learning import run_active_learning
from src.correct_word.levenshtein import recommend_corrected_word
from src.suggestion_ranker import rank_suggestions
from src.preprocess.teprolin_pipeline import teprolin_preprocess
from src.detection.detect import HFWrapperULMFiT


DB_PATH = "feedback.db"
init_db(DB_PATH)

# Load model here once to avoid reloading on every request
model_path = "./t5-grammar-finetuned"
model, tokenizer = load_model(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clf_model_path = os.path.join(pathlib.Path(__file__).parent, "src", "detection", "content", "trained_model_V2_2")
clf_tokenizer = AutoTokenizer.from_pretrained(clf_model_path)
clf_model = HFWrapperULMFiT.from_pretrained(str(clf_model_path)).to(device)
clf_model.eval()

# Schedule the active learning process to run every 3 days
scheduler = BackgroundScheduler()
scheduler.add_job(
    run_active_learning,
    'interval',
    days=3,
    kwargs={"model": model, "tokenizer": tokenizer, "db_path": "feedback.db"}
)
scheduler.start()

app = Flask(__name__)

CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    """
    Ensure every response includes the correct CORS headers.
    This fixes the issue where CORS only worked for OPTIONS but failed for POST.
    """
    origin = request.headers.get("Origin")

    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
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
    data = request.get_json(force=True)
    text = data.get("text")
    if data is None or "text" not in data:
        return jsonify({"error": "Invalid request, 'text' key missing"}), 400

    teprolin_result = teprolin_preprocess(text)
    corrected_text = teprolin_result["teprolin-result"]["text"]
    raw_suggestions = generate_corrections(model, tokenizer, corrected_text, num_suggestions=5)
    ranked = rank_suggestions(original=text, suggestions=raw_suggestions, metric="bleu")
    suggestions = [s for s, _ in ranked]
    corrected   = suggestions[0] if suggestions else corrected_text

    return add_cors_headers(jsonify({
        "original":    text,
        "corrected":   corrected,
        "suggestions": suggestions
    }))
    
@app.route('/check', methods=['POST'])
def check_and_correct_text():
    data = request.get_json(force=True)
    text = data.get("text")
    if data is None or "text" not in data:
        return jsonify({"error": "Invalid request, 'text' key missing"}), 400

    # Split input into sentences
    sentences = re.split(r'(?<=[\.\!\?])\s+', text)
    corrected_sentences = []

    for sent in sentences:
        result = clf_model(**clf_tokenizer(sent, return_tensors="pt", truncation=True, padding=True).to(device))
        is_correct = (torch.argmax(result['logits'], dim=1).item() == 0)
        if is_correct:
            corrected_sentences.append(sent)
        else:
            teprolin_result = teprolin_preprocess(sent)
            cleaned = teprolin_result["teprolin-result"]["text"]
            raw = generate_corrections(model, tokenizer, cleaned, num_suggestions=1)
            suggestion = raw[0] if raw else cleaned
            corrected_sentences.append(suggestion)

    corrected = "".join(corrected_sentences)
    return add_cors_headers(jsonify({"corrected": corrected}))

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json(force=True)
    orig        = data.get("original")
    suggestions = data.get("suggestions", [])
    chosen      = data.get("chosen")

    if not orig or chosen is None:
        return jsonify({"error": "Insufficient payload"}), 400

    store_feedback(original=orig,
                   suggestions=suggestions,
                   chosen=chosen,
                   db_path="feedback.db")

    return add_cors_headers(jsonify({"status": "ok"}))

@app.route('/word', methods=['POST'])
def correct_word():
    data = request.get_json(force=True)
    if not data or "word" not in data:
        return jsonify({"error": "Invalid request, 'word' key missing"}), 400

    word = data["word"]
    suggestions = recommend_corrected_word(word, num_suggestions=5)

    if len(suggestions) < 5:
        model_suggestions = generate_corrections(model, tokenizer, word, num_suggestions=9)
        extras = [s for s in model_suggestions if s not in suggestions]
        needed = 5 - len(suggestions)
        suggestions.extend(extras[:needed])

    return add_cors_headers(jsonify({
        "original":    word,
        "suggestions": suggestions
    }))


@app.route('/synonym', methods=['POST'])
def recommend_wordnet():
    data = request.get_json()
    if data is None or "word" not in data:
        return jsonify({"error": "Invalid request, 'word' key missing"}), 400

    word = data["word"]
    suggestions = get_related_forms(word)

    response = jsonify({
        "suggestions": suggestions
    })
    return add_cors_headers(response)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host="localhost", port=5001, ssl_context=('./SSL/cert.pem', './SSL/key.pem'))