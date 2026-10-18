import math
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request
from sklearn.datasets import load_iris


app = Flask(__name__)

MODEL_PATH = Path(__file__).parent / "model.pkl"
FEATURES = ("sepal_length", "sepal_width", "petal_length", "petal_width")

if not MODEL_PATH.exists():
    raise FileNotFoundError("model.pkl not found. Run python train_model.py first.")

saved_model = joblib.load(MODEL_PATH)

# Support both an artifact dictionary and a model saved directly.
if isinstance(saved_model, dict) and "model" in saved_model:
    model = saved_model["model"]
    target_names = saved_model.get("target_names", load_iris().target_names.tolist())
    model_accuracy = saved_model.get("accuracy")
else:
    model = saved_model
    target_names = load_iris().target_names.tolist()
    model_accuracy = None


def validate_features(data):
    values = []

    for name in FEATURES:
        raw_value = data.get(name)

        if raw_value is None or str(raw_value).strip() == "":
            raise ValueError(f"Enter a value for {name.replace('_', ' ')}.")

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"{name.replace('_', ' ').capitalize()} must be a number.")

        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name.replace('_', ' ').capitalize()} must be greater than zero.")

        values.append(value)

    return values


@app.get("/")
def home():
    return render_template(
        "index.html",
        form_values={name: "" for name in FEATURES},
        prediction=None,
        error=None,
    )


@app.get("/health")
def health():
    return jsonify(
        status="healthy",
        model_loaded=model is not None,
        model_accuracy=model_accuracy,
    ), 200


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "GET":
        return home()

    is_api_request = request.is_json

    if is_api_request:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error="Send a JSON object containing all four measurements."), 400
    else:
        data = request.form

    try:
        values = validate_features(data)
        class_id = int(model.predict([values])[0])
        species = target_names[class_id]
    except ValueError as error:
        if is_api_request:
            return jsonify(error=str(error)), 400

        return render_template(
            "index.html",
            form_values=data,
            prediction=None,
            error=str(error),
        ), 400

    if is_api_request:
        return jsonify(
            prediction=species,
            class_id=class_id,
            input=dict(zip(FEATURES, values)),
        ), 200

    return render_template(
        "index.html",
        form_values=data,
        prediction=species,
        error=None,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)