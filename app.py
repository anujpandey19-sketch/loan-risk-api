from flask import Flask, request, jsonify
import joblib, pandas as pd

app = Flask(__name__)
model = joblib.load("likelihood_model.joblib")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/score", methods=["POST"])
def score():
    payload = request.get_json(force=True)
    # Expect: {"records":[{"DPD__c": ..., "MIA__c": ..., "ECL__c": ...}, ...]}
    df = pd.DataFrame(payload["records"])
    proba = model.predict_proba(df)[:,1]
    return jsonify({"scores": [float(x) for x in proba]})
