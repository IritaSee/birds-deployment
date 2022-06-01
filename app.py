from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow import expand_dims
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model = load_model('adam.h5')
class_dict = {0:'Cendrawasih Kuning Besar', 1:'Jalak Bali', 2:'Kakak Tua Putih Jambul Kuning', 3:'Kasuari', 4:'Lain-lain',
                   5:'Maleo', 6:'Merak Biru'}

def predict(img_path):
    loaded_img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(loaded_img) / 255.0
    img_array = expand_dims(img_array, [0])
    predicted = model.predict(img_array)
    predicted = predicted.argmax(axis=-1)
    y = " ".join(str(x) for x in predicted)#mengambil nilai terbesar
    y = int(y)
    res = class_dict[y]
    return res

def predict_array(img_path):
    loaded_img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(loaded_img) / 255.0
    img_array = expand_dims(img_array, [0])
    predicted = model.predict(img_array)
    predicted = predicted[0] * 100
    res = {}
    for idx,a in enumerate(class_dict.values()):
        res[a] = predicted[idx]
    return res

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            out_array = True
            if out_array:
                 prediction = predict_array(img_path)
            else:
                prediction = predict_array(img_path)
            return render_template('index.html', uploaded_image=image.filename, prediction=prediction, out_array=out_array)

    return render_template('index.html')

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)