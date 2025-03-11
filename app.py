import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# 1️⃣ Load Training Data
train = pd.read_csv("/workspace/LLM-Text-Detection/Actual-Datasets/train_essays.csv").dropna()

# 2️⃣ Build & Train the Model Pipeline
model_pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    SVC(probability=True, class_weight="balanced")
)

X_train, y_train = train['text'], train['generated']
model_pipeline.fit(X_train, y_train)

# 3️⃣ Save Model (for quick loading)
with open("model.pkl", "wb") as model_file:
    pickle.dump(model_pipeline, model_file)

# 4️⃣ Create Flask App
app = Flask(__name__)

# 5️⃣ Load Model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# 6️⃣ Serve the Webpage
@app.route('/')
def home():
    return render_template('index.html')

# 7️⃣ API Endpoint for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    probability = model.predict_proba([text])[0][1]
    prediction = "AI-generated" if probability > 0.5 else "Human-written"

    return jsonify({"text": text, "prediction": prediction, "probability": round(probability, 2)})

# 8️⃣ Run Flask App
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
