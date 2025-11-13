from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback

app = Flask(__name__)

# Load model once at startup
model = joblib.load("likelihood_model.joblib")

# OPTIONAL: if you saved a separate list of columns
try:
    model_columns = joblib.load("model_columns.joblib")
except Exception:
    model_columns = None

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/score", methods=["POST"])
def score():
    try:
        payload = request.get_json(force=True)

        # Accept both {"records":[...]} and [...]
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and "records" in payload:
            records = payload["records"]
        else:
            return jsonify({
                "error": "Payload must be either an array of records or an object with a 'records' array"
            }), 400

        if not records:
            return jsonify({"error": "No records provided"}), 400

        df = pd.DataFrame(records)

        # If we stored column order, align to it
        if model_columns is not None:
            # add missing columns with 0
            missing = [c for c in model_columns if c not in df.columns]
            for c in missing:
                df[c] = 0
            # drop extra cols and reorder
            df = df[model_columns]

        proba = model.predict_proba(df)[:, 1]
        return jsonify({"scores": [float(x) for x in proba]})

    except Exception as e:
        # Print to Render logs for debugging
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
