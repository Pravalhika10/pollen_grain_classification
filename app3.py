from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

# Load the trained model
model = load_model('C:/Users/Pravalhika/model.h5', compile=False)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction.html')
def prediction():
    return render_template('prediction.html')

@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/logout.html')
def logout():
    return render_template('logout.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', f.filename)
        f.save(file_path)

        # Preprocess the image
        img = tf.keras.utils.load_img(file_path, target_size=(128, 128))
        x = tf.keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Make prediction
        pred = np.argmax(model.predict(x))
        op = ['anadenanthera', 'arecaceae', 'arrabidaea', 'cecropia', 'chromolaena',
              'combretum', 'croton', 'dipetryx', 'eucalipto', 'faramea', 'hyptis',
              'mabea', 'matayba', 'mimosa', 'myrcia', 'protium', 'qualea', 'schinus',
              'senegalia', 'serjania', 'syagrus', 'tridax', 'urochloa']

        result = op[pred]
        return render_template('prediction.html', pred=result)

    return "Please submit an image."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
