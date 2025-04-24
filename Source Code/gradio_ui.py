import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model(r"D:\College\AIML\Semester 4\CNN Final\model.h5")

# Preprocess the uploaded image
def preprocess_image(img: Image.Image):
    img = img.convert("L")           # Convert to grayscale
    img = img.resize((28, 28))       # Resize to 28x28
    img_array = np.array(img)
    img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255.0
    return img_array

# Prediction function
def predict_digit(image):
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)
    return f"Predicted Digit: {digit} (Confidence: {confidence:.2f})"

# Gradio Interface
ui = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="pil", label="Upload a digit image"),
    outputs="text",
    title="Handwritten Digit Recognizer",
    description="Upload a 28x28 grayscale digit image or any handwritten digit. The model will predict the digit.",
    allow_flagging="never"
)

if __name__ == "__main__":
    ui.launch()
