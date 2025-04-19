# Importing the required libraries
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from PIL import Image
import io

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
final_model = tf.keras.models.load_model(
    r"C:\Users\shriy\Downloads\CNN try 2\CNN try 2\model.h5"
)

# Function to format the image (resize and normalize)
def format_image(img):
    img = img.resize((28, 28))                     # Resize to 28x28
    img_array = np.array(img)                      # Convert to NumPy array
    return img_array

# Function to predict the digit using the CNN model
def predict_digit(model, img_array: np.ndarray) -> str:
    data = img_array.reshape(1, 28, 28, 1).astype('float32') / 255.0  # Reshape + normalize
    prediction = model.predict(data)                                 # Get prediction
    digit = np.argmax(prediction)                                    # Highest confidence class
    return str(digit)

# Bootup/root endpoint
@app.get("/")
def read_root():
    return {"message": "MNIST digit recognition API is live!"}

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()                                     # Read uploaded image file
    img = Image.open(io.BytesIO(contents)).convert('L')              # Convert to grayscale
    img_array = format_image(img)                                    # Format image
    digit = predict_digit(final_model, img_array)                    # Predict digit
    return {"Digit": digit}                                          # Return response
