from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from numpy import argmax
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import os
from PIL import Image
import io
import base64

app = Flask(__name__, template_folder="templates")
CORS(app)  # Enable CORS for all routes

# Load the model once when the server starts
MODEL_PATH = 'final_model.h5'
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def process_image(image_file):
    """Process uploaded image for digit prediction"""
    img = Image.open(image_file)

    if img.mode != 'L':  # Convert to grayscale
        img = img.convert('L')

    img = img.resize((28, 28))
    img_array = img_to_array(img)
    img_array = img_array.reshape(1, 28, 28, 1)
    img_array = img_array.astype('float32') / 255.0

    return img_array

def process_base64_image(base64_string):
    """Process base64 encoded image (from camera capture)"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    image_bytes = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(image_bytes))

    if img.mode != 'L':
        img = img.convert('L')

    img = img.resize((28, 28))
    img_array = img_to_array(img)
    img_array = img_array.reshape(1, 28, 28, 1)
    img_array = img_array.astype('float32') / 255.0

    return img_array

@app.route('/predict', methods=['POST'])
def predict_digit():
    """API endpoint for digit prediction"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        if 'image' in request.files:  # File upload
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            img_array = process_image(file)

        elif request.json and 'imageData' in request.json:  # Base64 image
            base64_data = request.json['imageData']
            img_array = process_base64_image(base64_data)

        else:
            return jsonify({'error': 'No image data provided'}), 400

        # Make prediction
        prediction = model.predict(img_array)
        predicted_digit = int(argmax(prediction))
        confidence = float(prediction[0][predicted_digit])
        probabilities = {str(i): float(prediction[0][i]) for i in range(10)}

        return jsonify({
            'predicted_digit': predicted_digit,
            'confidence': confidence,
            'probabilities': probabilities,
            'success': True
        })

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/', methods=['GET'])
def home():
    """Serve the frontend"""
    return render_template("index.html")

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file '{MODEL_PATH}' not found!")
    app.run(debug=True, host='0.0.0.0', port=5000)
