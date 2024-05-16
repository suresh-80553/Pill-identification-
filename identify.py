import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pill_dictionary import PillDictionary
from tkinter import Tk, filedialog

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
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted Pill: {predicted_pill}')

    # Display pill information
    info = pill_dict.get_pill_info(predicted_pill)
    print(f'\nBenefits: {info["benefits"]}\nTablets: {info["tablets"]}\nSideEffect: {info["side_effects"]}\nDosage: {info["dosage"]}')
    plt.show()

# Create a Tkinter root window (hidden)
root = Tk()
root.attributes('-topmost', True)
root.withdraw()

# Open a file dialog for image selection
file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

# Check if a file is selected
if file_path:
    # Example usage
    predict_pill(file_path)



