"""Quick sanity check: predict one random image from each class."""
import json
import os
import random

import numpy as np
import tensorflow as tf
from PIL import Image

DATA_DIR = os.path.join("data", "plantvillage")

model = tf.keras.models.load_model("models/crop_model.keras")
with open("models/class_labels.json", encoding="utf-8") as f:
    class_names = json.load(f)

height, width = model.input_shape[1], model.input_shape[2]

print(f"{'Actual class':<25} {'Predicted class':<25} {'Confidence':>10}  Result")
print("-" * 72)

for actual in class_names:
    folder = os.path.join(DATA_DIR, actual)
    file_name = random.choice(os.listdir(folder))
    image = Image.open(os.path.join(folder, file_name)).convert("RGB")
    image = image.resize((width, height))
    batch = np.expand_dims(np.array(image, dtype="float32"), axis=0)

    probabilities = model.predict(batch, verbose=0)[0]
    top_index = int(np.argmax(probabilities))
    predicted = class_names[top_index]
    confidence = probabilities[top_index] * 100
    result = "CORRECT" if predicted == actual else "WRONG"

    print(f"{actual:<25} {predicted:<25} {confidence:>9.1f}%  {result}")