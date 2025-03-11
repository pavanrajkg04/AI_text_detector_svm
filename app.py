import pickle
import pandas as pd
import logging
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# Set up logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 1️⃣ Load Training Data
try:
    logging.info("Training initiated.")
    train = pd.read_csv("/workspace/AI_text_detector_svm/DataSet/train_essays.csv").dropna()
    logging.info("Training data loaded successfully.")
except Exception as e:
    logging.error(f"Error loading training data: {e}")
    raise

# 2️⃣ Build & Tune the Model Pipeline
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train = vectorizer.fit_transform(train["text"])
y_train = train["generated"]

# Hyperparameter tuning with GridSearchCV
param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
grid_search = GridSearchCV(SVC(probability=True, class_weight="balanced"), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best Model
best_svc = grid_search.best_estimator_
model_pipeline = make_pipeline(vectorizer, best_svc)

logging.info("Model trained successfully with best parameters.")

# 3️⃣ Save Model (for quick loading)
try:
    with open("model.pkl", "wb") as model_file:
        pickle.dump(model_pipeline, model_file)
    logging.info("Model saved successfully.")
except Exception as e:
    logging.error(f"Error saving model: {e}")
    raise

# 4️⃣ Create Flask App
app = Flask(__name__)

# 5️⃣ Load Model
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

# 6️⃣ Serve the Webpage
@app.route("/")
def home():
    return render_template("index.html")

# 7️⃣ API Endpoint for Prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        text = data.get("text", "").strip()

        if not text:
            logging.warning("Empty text input received.")
            return jsonify({"error": "No text provided"}), 400

        probability = model.predict_proba([text])[0][1]
        prediction = "AI-generated" if probability > 0.5 else "Human-written"

        logging.info(f"Prediction made - Text: '{text}' | Prediction: {prediction} | Probability: {probability:.2f}")

        return jsonify({"text": text, "prediction": prediction, "probability": round(probability, 2)})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

# 8️⃣ Run Flask App
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
