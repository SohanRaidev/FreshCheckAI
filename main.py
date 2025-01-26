from flask import Flask, request, render_template, jsonify
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import UnidentifiedImageError

app = Flask(__name__)

# Ensure your 'uploads' folder exists or create it
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'  # For serving static files like images

# Load your trained model
model = load_model('/Users/sohanrai/Downloads/freshnessrecognition.keras')

# Define class labels for prediction
class_labels = {
    0: 'freshapple',
    1: 'freshbanana',
    2: 'freshorange',
    3: 'rottenapple',
    4: 'rottenbanana',
    5: 'rottenorange'
}

# Image preprocessing function
def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
    except UnidentifiedImageError:
        return None
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess and predict
    img = preprocess_image(filepath)
    if img is None:
        return jsonify({'error': 'Invalid image file'})

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    predicted_class = class_labels[class_index]

    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
