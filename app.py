from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load model (relative path use karo)
model = load_model("models/image_classifier_model.h5")

class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']

        if not file:
            return jsonify({"error": "No file uploaded"})

        # Save temporary file
        filepath = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)

        # Preprocess image
        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        result = class_labels[class_index]

        if result == 'notumor':
            label = "No Tumor"
        else:
            label = f"Tumor: {result}"

        return jsonify({
            "result": label,
            "confidence": f"{confidence*100:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)