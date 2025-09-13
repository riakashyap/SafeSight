import os
import sys
from tempfile import NamedTemporaryFile
import cv2
import numpy as np
from flask import Flask, flash, request, redirect, url_for, render_template,send_file, abort
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.block import C2f, Bottleneck, SPPF
from torch.nn.modules.container import Sequential
import json
#from waitress import serve


if getattr(sys, "frozen", False):
    # Running as a PyInstaller one-file exe
    BUNDLE_DIR = sys._MEIPASS  # temp extraction folder (read-only)
    BASE_DIR = os.getcwd()     # folder where the exe is located (read/write)
    TEMPLATE_DIR = os.path.join(BUNDLE_DIR, "templates")
    # keep static external for uploads
    STATIC_DIR = os.path.join(BASE_DIR, "static")
    MODELS_DIR = os.path.join(BUNDLE_DIR, "models")
else:
    # Running from source
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BUNDLE_DIR = BASE_DIR
    TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
    STATIC_DIR = os.path.join(BASE_DIR, "static")
    MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ensure external static/uploads exist for runtime writes
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.secret_key = "secret key"

# -------------------------------
# Config
# -------------------------------

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg","mp4","mov"}


def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower().lstrip(".") in ALLOWED_EXTENSIONS


model = YOLO(".src/yolov8s.onnx", task="detect")

print("model initialised")


@app.route("/")
def home():
    return render_template("homepage.html")

@app.route("/upload_page")
def index():
    return render_template("upload_page.html")

@app.route('/upload', methods=['POST'])
def upload():
   
    if 'file' not in request.files:
        return abort(400, "No file uploaded")
    f = request.files['file']
    if f.filename == '':
        return abort(400, "Empty filename")

    # Save uploaded file to a temporary file
    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.filename)[1]) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        result_list = model(tmp_path,save =True)
        processed_path = result_list[0].save_dir 
        #result_json = result_list[0].to_json()
        
        processed_video = os.path.basename(result_list[0].path)
        processed_path = os.path.join(processed_path,processed_video)
        print(processed_path)
        #print(json.dumps(result_json))
        

        return send_file(processed_path, mimetype='video/mp4', as_attachment=False)
    finally:
       
        pass

@app.after_request
def add_header(response):
    response.headers["X-UA-Compatible"] = "IE=Edge,chrome=1"
    response.headers["Cache-Control"] = "public, max-age=0"
    return response

# if _name_ == "_main_":
#     # from waitress import serve
#     # serve(app, host="127.0.0.1", port=5000)
#     app.run(host='127.0.0.1', port=5000, debug=True)

if __name__ == '__main__':
    # dev server, use a proper WSGI server in production
    print("let's run")
    #serve(app, host="127.0.0.1", port=8080)
    app.run(host='127.0.0.1', port=8080, debug=True)
    print("it ran")
