import numpy as np
from PIL import Image
import tensorflow as tf
import json

with open("plant_disease.json") as f:
    disease_data = json.load(f)

CLASS_NAMES = [d["name"] for d in disease_data]

def preprocess(image):
    image = image.convert("RGB").resize((160,160))
    img = np.array(image)/255.0
    return np.expand_dims(img, axis=0)

def predict(image, model):
    img = preprocess(image)
    pred = model.predict(img)[0]
    idx = int(np.argmax(pred))
    conf = float(pred[idx])

    disease = disease_data[idx]

    return disease["name"], conf, disease["cause"], disease["cure"]
