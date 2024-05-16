import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from pill_dictionary import PillDictionary

app = Flask(__name__, static_url_path='/static')

# Load the trained model
model = load_model('multi_model_classifier.h5')

# Load the pill dictionary
pill_dict = PillDictionary()

# Function to preprocess the input image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to make a prediction and display pill information
def predict_pill(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    predicted_pill = list(pill_dict.pill_info.keys())[predicted_class]

    # Display the image
    img = Image.open(image_path)
    img.show()

    # Display pill information
    info = pill_dict.get_pill_info(predicted_pill)
    return predicted_pill, info

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/aboutus.html', methods=['GET'])
def about_us():
    return render_template('aboutus.html')

@app.route('/upload.html', methods=['GET'])
def upload():
    return render_template('upload.html', result=None)

@app.route('/identify', methods=['POST'])
def identify_pill():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        # Save the file temporarily
        file_path = 'static/img/temp.jpg'
        file.save(file_path)

        # Example usage
        predicted_pill, info = predict_pill(file_path)

        # Return the prediction and information
        return render_template('result.html', predicted_pill=predicted_pill, info=info)

    @app.route('/dictionary')
    def pill_dictionary():
        # Example: Fetch information about a specific pill
        pill_name = request.args.get('pill_name')

        if pill_name:
            info = pill_dict.get_pill_info(pill_name)
            return render_template('dictionary.html', pill_name=pill_name, info=info)
        else:
            return "Pill name not provided"


if __name__ == '__main__':
    app.run(debug=True, port=1702)
