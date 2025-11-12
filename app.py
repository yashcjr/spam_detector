from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from src.train_sms_model import sms_model
from src.train_url_model import url_model
import os
import threading
import time

app = Flask(__name__, template_folder="templates")
CORS(app)  # <-- important for separate frontend

MODEL1_PATH = "models/msg_model.pkl"
MODEL2_PATH = "models/url_model.pkl"

# Train model if it doesn't exist
if not os.path.exists(MODEL1_PATH):
    sms_model()
if not os.path.exists(MODEL2_PATH):
    url_model()


# http://localhost:5000/predict

from src.predict import predict_spam

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")
    
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    if not text.strip():
        return jsonify({"result": "⚠️ Empty input"}), 200

    print(f"TEXT:: {text}")
    result = predict_spam(text)
    print(f"RESULT:: {result}")
    return jsonify({"result": result})


#this function will keep render alive for eternity
def keep_alive():
    while True:
        time.sleep(600)

if __name__ == "__main__":
    t1 = threading.Thread(target=app.run(debug=True, port=5000, debugger=True))
    t2 = threading.Thread(target=keep_alive)

    t1.start()
    t2.start()

    t1.join()
    t2.join()



# app.py

# MODEL1_PATH = "models/msg_model.pkl"
# MODEL2_PATH = "models/url_model.pkl"

# # Train model if it doesn't exist
# if not os.path.exists(MODEL1_PATH):
#     sms_model()
#     print("training model")
# if not os.path.exists(MODEL2_PATH):
#     url_model()
#     print("trainin another model")


# from src.predict import predict_spam


# if __name__ == "__main__":
#     text = input("Enter URL or message: ")
#     result = predict_spam(text)
#     print(result)



