"""
Flask Web Application for Parkinson's Disease Prediction System.
Serves interactive REST API endpoints and web interface on localhost.
"""

import os
import sys
import json
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.predict import ParkinsonsPredictor

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Global predictor instance
predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        predictor = ParkinsonsPredictor(models_dir="models")
    return predictor


@app.route("/")
def index():
    """Renders main web app interface."""
    return render_template("index.html")


@app.route("/outputs/<path:filename>")
def serve_output_image(filename):
    """Serves generated EDA and model evaluation plot PNGs."""
    return send_from_directory("outputs", filename)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST /api/predict
    Body: JSON patient features dict
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No input JSON data provided."}), 400

        pred_engine = get_predictor()
        result = pred_engine.predict_one(data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/api/samples", methods=["GET"])
def api_samples():
    """Returns sample patient profiles for quick UI loading."""
    healthy_sample = {
        "MDVP:Fo(Hz)": 197.076,
        "MDVP:Fhi(Hz)": 206.896,
        "MDVP:Flo(Hz)": 192.055,
        "MDVP:Jitter(%)": 0.00289,
        "MDVP:Jitter(Abs)": 0.00001,
        "MDVP:RAP": 0.00166,
        "MDVP:PPQ": 0.00168,
        "Jitter:DDP": 0.00498,
        "MDVP:Shimmer": 0.01098,
        "MDVP:Shimmer(dB)": 0.09700,
        "Shimmer:APQ3": 0.00563,
        "Shimmer:APQ5": 0.00680,
        "MDVP:APQ": 0.00802,
        "Shimmer:DDA": 0.01689,
        "NHR": 0.00339,
        "HNR": 26.77500,
        "RPDE": 0.422229,
        "DFA": 0.648633,
        "spread1": -7.348300,
        "spread2": 0.177551,
        "D2": 1.743867,
        "PPE": 0.085569
    }

    pd_high_risk_sample = {
        "MDVP:Fo(Hz)": 119.992,
        "MDVP:Fhi(Hz)": 157.302,
        "MDVP:Flo(Hz)": 74.997,
        "MDVP:Jitter(%)": 0.00784,
        "MDVP:Jitter(Abs)": 0.00007,
        "MDVP:RAP": 0.00370,
        "MDVP:PPQ": 0.00554,
        "Jitter:DDP": 0.01109,
        "MDVP:Shimmer": 0.04374,
        "MDVP:Shimmer(dB)": 0.42600,
        "Shimmer:APQ3": 0.02182,
        "Shimmer:APQ5": 0.03130,
        "MDVP:APQ": 0.02971,
        "Shimmer:DDA": 0.06545,
        "NHR": 0.02211,
        "HNR": 21.03300,
        "RPDE": 0.414783,
        "DFA": 0.815285,
        "spread1": -4.813031,
        "spread2": 0.266482,
        "D2": 2.301442,
        "PPE": 0.284654
    }

    pd_moderate_sample = {
        "MDVP:Fo(Hz)": 153.848,
        "MDVP:Fhi(Hz)": 165.738,
        "MDVP:Flo(Hz)": 125.688,
        "MDVP:Jitter(%)": 0.00450,
        "MDVP:Jitter(Abs)": 0.00003,
        "MDVP:RAP": 0.00230,
        "MDVP:PPQ": 0.00260,
        "Jitter:DDP": 0.00690,
        "MDVP:Shimmer": 0.02200,
        "MDVP:Shimmer(dB)": 0.21000,
        "Shimmer:APQ3": 0.01100,
        "Shimmer:APQ5": 0.01350,
        "MDVP:APQ": 0.01750,
        "Shimmer:DDA": 0.03300,
        "NHR": 0.01100,
        "HNR": 23.50000,
        "RPDE": 0.490000,
        "DFA": 0.710000,
        "spread1": -6.100000,
        "spread2": 0.200000,
        "D2": 2.050000,
        "PPE": 0.160000
    }

    return jsonify({
        "healthy": healthy_sample,
        "high_risk": pd_high_risk_sample,
        "moderate_risk": pd_moderate_sample
    })


@app.route("/api/metadata", methods=["GET"])
def api_metadata():
    """Returns trained model performance metadata and feature importances."""
    try:
        meta_path = "models/model_metadata.json"
        imp_path = "models/feature_importances.csv"

        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)

        importances = []
        if os.path.exists(imp_path):
            df_imp = pd.read_csv(imp_path)
            importances = df_imp.to_dict(orient="records")

        return jsonify({
            "metadata": metadata,
            "feature_importances": importances
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # Initialize predictor on startup
    try:
        get_predictor()
        logger.info("Predictor initialized successfully.")
    except Exception as err:
        logger.warning(f"Predictor initialization deferred: {err}")

    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Parkinson's Prediction Web App on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
