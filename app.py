from flask import Flask, request, jsonify
from flask_cors import CORS
from numpy import argmax
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import os
import tempfile
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model once when the server starts
MODEL_PATH = 'final_model.h5'  # Update this path as needed
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def process_image(image_file):
    """Process uploaded image for digit prediction"""
    try:
        # Open the image
        img = Image.open(image_file)
        
        # Convert to grayscale if not already
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize to 28x28
        img = img.resize((28, 28))
        
        # Convert to array
        img_array = img_to_array(img)
        
        # Reshape into a single sample with 1 channel
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # Normalize pixel values
        img_array = img_array.astype('float32') / 255.0
        
        return img_array
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def process_base64_image(base64_string):
    """Process base64 encoded image (from camera capture)"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Create PIL Image from bytes
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale if not already
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize to 28x28
        img = img.resize((28, 28))
        
        # Convert to array
        img_array = img_to_array(img)
        
        # Reshape into a single sample with 1 channel
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # Normalize pixel values
        img_array = img_array.astype('float32') / 255.0
        
        return img_array
    except Exception as e:
        raise Exception(f"Error processing base64 image: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict_digit():
    """API endpoint for digit prediction"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Check if it's a file upload or base64 data
        if 'image' in request.files:
            # Handle file upload
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            img_array = process_image(file)
            
        elif 'imageData' in request.json:
            # Handle base64 encoded image (from camera)
            base64_data = request.json['imageData']
            img_array = process_base64_image(base64_data)
            
        else:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_digit = int(argmax(prediction))
        confidence = float(prediction[0][predicted_digit])
        
        # Get all probabilities for visualization
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
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Digit Recognition API',
        'endpoints': {
            '/predict': 'POST - Upload image for digit prediction',
            '/health': 'GET - Health check'
        }
    })

if __name__ == '__main__':
    # Make sure the model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file '{MODEL_PATH}' not found!")
        print("Please ensure your model file is in the correct location.")
    
    print("Starting Flask server...")
    print("Upload endpoint: http://localhost:5000/predict")
    app.run(debug=True, host='0.0.0.0', port=5000)
