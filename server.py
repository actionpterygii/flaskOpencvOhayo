from flask import Flask, request, render_template
from datetime import datetime
import numpy as np
import cv2
import os

app = Flask(__name__)

SAVE_DIR = "./static"

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/", methods = ["POST"])
def upload():

    stream = request.files['image'].stream
    img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
    im = cv2.imdecode(img_array, 1)

    dt_now = datetime.now().strftime("%Y_%m_%d%_H_%M_%S_")
    save_path = os.path.join(SAVE_DIR, dt_now + "in" + ".png")
    cv2.imwrite(save_path, im)

    cascade = cv2.CascadeClassifier('cascade.xml')
    faces = cascade.detectMultiScale(im, 1.1, 3) 

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x,y), (x+w,y+h), (255, 0, 0), thickness=6) 
    
    save_path = os.path.join(SAVE_DIR, dt_now + "ou" + ".png")
    cv2.imwrite(save_path, im)

    return render_template('index.html', im=os.listdir(SAVE_DIR)[-1])


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=80, threaded=True)