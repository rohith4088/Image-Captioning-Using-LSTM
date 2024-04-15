from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from PIL import Image
import tensorflow
from tensorflow.keras.models import load_model

# Load the Keras model
model = load_model('model.h5')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Open the image file
            image = Image.open(file.stream)
            # Convert the image to a numpy array
            image_array = np.array(image)
            # Expand dimensions to match the model's expected input shape
            image_array = np.expand_dims(image_array, axis=0)
            # Use the model to generate a caption
            caption = model.predict(image_array)
            # Assuming the model returns a string
            # Note: The actual prediction might need further processing depending on your model's output
            return render_template('result.html', caption=caption)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
