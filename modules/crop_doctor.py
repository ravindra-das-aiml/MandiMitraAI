"""Crop Doctor: image-based crop disease screening.

This is a preliminary screening tool, not a confirmed expert diagnosis.
"""
import json

import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "models/crop_model.keras"
LABELS_PATH = "models/class_labels.json"
CONFIDENCE_THRESHOLD = 0.70  # below this, the result is shown as uncertain

# Cautious, general next steps. These are not pesticide recommendations.
ADVICE = {
    "healthy": "No disease detected by the model. Keep monitoring the crop regularly.",
    "Early_blight": (
        "Possible early blight. Remove badly affected leaves, avoid overhead watering, "
        "and consult your local agriculture officer or KVK before using any treatment."
    ),
    "Late_blight": (
        "Possible late blight. This can spread quickly in cool, humid weather. "
        "Contact your local agriculture officer or KVK soon for confirmation and treatment advice."
    ),
}


def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


def pretty_name(label):
    """'Tomato_Early_blight' -> 'Tomato - Early blight'"""
    crop, _, condition = label.replace("___", "_").partition("_")
    return f"{crop} - {condition.replace('_', ' ')}"


def advice_for(label):
    for key, text in ADVICE.items():
        if label.endswith(key):
            return text
    return "Consult your local agriculture officer for guidance."


def predict(model, class_names, image):
    """Return the top 3 predictions and whether the result is reliable."""
    height, width = model.input_shape[1], model.input_shape[2]
    image = image.convert("RGB").resize((width, height))
    batch = np.expand_dims(np.array(image, dtype="float32"), axis=0)

    probabilities = model.predict(batch, verbose=0)[0]
    order = np.argsort(probabilities)[::-1][:3]
    top3 = [(class_names[i], float(probabilities[i])) for i in order]
    reliable = top3[0][1] >= CONFIDENCE_THRESHOLD
    return top3, reliable