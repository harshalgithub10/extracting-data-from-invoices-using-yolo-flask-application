import os
from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import pytesseract
from PIL import Image
import cv2
import re
import json
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Define paths for YOLO files
YOLO_CFG_PATH = os.path.join("yolo", "/home/harshu/flask3/yolo/yolov4-custom.cfg")
YOLO_WEIGHTS_PATH = os.path.join("yolo", "/home/harshu/flask3/yolo/yolov4-custom_last.weights")
YOLO_CLASSES_PATH = os.path.join("yolo", "/home/harshu/flask3/yolo/obj.names")

# Load class names
with open(YOLO_CLASSES_PATH, "r") as f:
    classes = [line.strip() for line in f.readlines()]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_image(image_path):
    with Image.open(image_path) as img:
        text = pytesseract.image_to_string(img)
    return text

def extract_key_info(text):
    key_patterns = {
        "invoice_number": r"(?:Invoice\s*Number|Invoice\s*No\.?|Invoice\s*#|INV\s*No\.?)\s*[:\-]?\s*([\w-]+)",
        "invoice_date": r"(?:Invoice\s*Date|Date\s*of\s*Invoice|Date)\s*[:\-]?\s*([\d/.-]+)",
        "total_amount": r"(?:Total\s*Amount|Amount\s*Payable|Total|Net\s*Amount)\s*[:\-]?\s*([\d,]+\.?\d*)",
        "cgst": r"CGST\s*[:\-]?\s*([\d,]+\.?\d*)",
        "sgst": r"SGST\s*[:\-]?\s*([\d,]+\.?\d*)",
        "igst": r"IGST\s*[:\-]?\s*([\d,]+\.?\d*)",
        "grand_total": r"(?:Grand\s*Total|Total\s*Amount|Net\s*Total)\s*[:\-]?\s*([\d,]+\.?\d*)",
        "bill_from": r"Bill\s*From\s*[:\-]?\s*([\w\s,.-]+)",
        "ship_to": r"Ship\s*To\s*[:\-]?\s*([\w\s,.-]+)",
        "bill_to": r"Bill\s*To\s*[:\-]?\s*([\w\s,.-]+)"
    }
    extracted_info = {}
    for key, pattern in key_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted_info[key] = match.group(1)
    return extracted_info

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Object detection using YOLO
        yolo_output = detect_objects(filepath)

        # Extract text from detected regions
        extracted_text = extract_text_from_image(yolo_output)

        # Extract key information from text
        key_info = extract_key_info(extracted_text)
        key_info_json = json.dumps(key_info, indent=4)

        return render_template('result.html', extracted_text=extracted_text, key_info_json=key_info_json)
    return redirect(request.url)

def detect_objects(image_path):
    # YOLO object detection
    net = cv2.dnn.readNet(YOLO_WEIGHTS_PATH, YOLO_CFG_PATH)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    img = cv2.imread(image_path)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "detected_" + os.path.basename(image_path))
    cv2.imwrite(output_image_path, img)
    return output_image_path

if __name__ == '__main__':
    app.run(debug=True)

