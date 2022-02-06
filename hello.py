from flask import Flask, request, jsonify
import base64
import cv2
from PIL import Image
import numpy as np
import re
from datetime import datetime
import time

app = Flask(__name__)

queue = []

@app.route("/", methods=['POST'])
def receive():
    print("asfasdasdf")
    
    filename = f'result{time.time()}'  # I assume you have a way of picking unique filenames
    with open(filename+'.txt', 'wb') as f:
        f.write(request.data)
    with open(filename+'.txt', 'r') as f:
        data_modified = re.sub('^data:image/.+;base64,', '', f.read())
        imgdata = base64.b64decode(data_modified)
        with open(filename+'.jpg', 'wb') as f2:
            f2.write(imgdata)
            queue.append(filename+'.jpg')
        
    return "<p>Hello, World!</p>"

@app.route("/", methods=['GET'])
def send():
    print("uiuioio")

    filename = queue.pop()
    img_encode = ""
    with open(filename, "rb") as f:
        img_encode = base64.b64encode(f.read()).decode('utf-8')

    return jsonify({'msg': 'success', 'size': [1280, 720], 'img': img_encode})



