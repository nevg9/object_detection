# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Perform test request
"""

import pprint

import requests

DETECTION_URL = 'http://localhost:5000/v1/object-detection/species_43_exp/'
IMAGE = '/home/data/image_data/species_43/test/images/HBNR013X-NH0078-20200805-01945.jpg'

# Read image
with open(IMAGE, 'rb') as f:
    image_data = f.read()

response = requests.post(DETECTION_URL, files={'image': image_data}).json()

pprint.pprint(response)
