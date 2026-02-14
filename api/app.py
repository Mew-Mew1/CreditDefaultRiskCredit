from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os
import sys
from source.utils.logger import get_logger

app = Flask(__name__)
logger = get_logger(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),  # go up one level from 'api'
    'models',
    'best_model_pipeline.pkl'
)

# Attempt to load the model on startup
try:
    logger.info(f"Attempting to load model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully.")
except FileNotFoundError:
    logger.warning(f"Model file not found at {MODEL_PATH}. Prediction endpoints will fail.")
    model = None
except Exception as e:
    logger.error(f"An unexpected error occurred while loading the model: {str(e)}", exc_info=True)
    model = None

@app.route("/", methods=["GET"])
def home():
    # Render the HTML form instead of JSON
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health_check():
    status = "healthy" if model is not None else "degraded"
    logger.info(f"Health check requested. Status: {status}")
    return jsonify({"status": status, "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        logger.error("Prediction requested but model is not loaded.")
        return jsonify({'error': 'Model not loaded on server'}), 500

    try:
        data = request.get_json()
        if not data:
            logger.warning("Prediction requested with empty payload.")
            return jsonify({'error': 'No input data provided'}), 400
        
        logger.info("Processing prediction request...")
        df = pd.DataFrame([data])
        
        probability = float(model.predict_proba(df)[0][1])
        prediction = int(model.predict(df)[0])
        
        # --- NEW 3-TIER RISK LOGIC ---
        if probability < 0.33:
            risk_level = 'Minimal'
        elif probability < 0.66:
            risk_level = 'Moderate'
        else:
            risk_level = 'High'
        
        result = {
            'prediction': prediction,
            'probability_of_default': probability,
            'risk_level': risk_level
        }
        
        logger.info(f"Prediction successful: {result}")
        return jsonify(result)
        
    except KeyError as e:
        logger.error(f"Missing required feature in input data: {str(e)}")
        return jsonify({'error': f'Missing required feature: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error during prediction'}), 500

if __name__ == "__main__":
    logger.info("Starting Flask API server...")
    app.run(host="0.0.0.0", port=5000, debug=True)