# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run a Flask REST API exposing one or more YOLOv5s models
"""
import sys
sys.path.append('../../')
import argparse
import io

import torch
from flask import Flask, request
from PIL import Image
from models.common import DetectMultiBackend
from utils.torch_utils import select_device

app = Flask(__name__)
model = None

DETECTION_URL = '/v1/object-detection/species_43_exp/'


@app.route(DETECTION_URL, methods=['POST'])
def predict():
    if request.method != 'POST':
        return

    if request.files.get('image'):
        # Method 1
        # with request.files["image"] as f:
        #     im = Image.open(io.BytesIO(f.read()))

        # Method 2
        im_file = request.files['image']
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        results = model(im, augment=Flask, visualize=False)  # reduce size=320 for faster inference
        print(results)
        return results.pandas().xyxy[0].to_json(orient='records')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flask API exposing YOLOv5 model')
    parser.add_argument('--port', default=5000, type=int, help='port number')
    parser.add_argument('--weight', type=str, required=True, help='model(s) to run, i.e. --model yolov5n yolov5s')
    parser.add_argument('--data', type=str,help='data/coco128.yaml dataset.yaml')
    parser.add_argument('--device', type=str, required=True, help='cuda device, i.e. 0,1,2,3 or cpu')

    args = parser.parse_args()
    device = select_device(args.device)
    model = DetectMultiBackend(args.weight, device=device, dnn=False, data=args.data, fp16=False)


    app.run(host='0.0.0.0', port=args.port)  # debug=True causes Restarting with stat
