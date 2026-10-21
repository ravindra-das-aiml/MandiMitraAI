"""
Crop Doctor - MobileNetV2 transfer learning.
Folder layout:  data/plantvillage/<class_name>/*.jpg
Run:  python train_crop_model.py
Outputs: models/crop_model.keras, models/class_labels.json, reports/*
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "data/plantvillage"
IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42
EPOCHS_HEAD = 5
EPOCHS_FINE = 5

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ---- Load data: 70% train / 15% val / 15% test ----
full = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, image_size=IMG_SIZE, batch_size=BATCH,
    shuffle=True, seed=SEED, label_mode="int",
)
class_names = full.class_names
n_classes = len(class_names)
n_batches = tf.data.experimental.cardinality(full).numpy()

train_n = int(n_batches * 0.70)
val_n = int(n_batches * 0.15)
train_ds = full.take(train_n)
val_ds = full.skip(train_n).take(val_n)
test_ds = full.skip(train_n + val_n)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

with open("models/class_labels.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

# ---- Model ----
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(0.2),  # helps with real phone photos
])

base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)
base.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = augment(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ---- Stage 1: train head ----
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)

# ---- Stage 2: fine-tune top layers ----
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE)

# ---- Evaluate on held-out test set ----
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTEST ACCURACY: {test_acc:.4f}")

y_true, y_pred = [], []
for imgs, labels in test_ds:
    probs = model.predict(imgs, verbose=0)
    y_pred.extend(np.argmax(probs, axis=1))
    y_true.extend(labels.numpy())

report = classification_report(y_true, y_pred, target_names=class_names)
print(report)
with open("reports/classification_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Test accuracy: {test_acc:.4f}\n\n{report}")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 7))
plt.imshow(cm, cmap="Blues")
plt.xticks(range(n_classes), class_names, rotation=90)
plt.yticks(range(n_classes), class_names)
plt.colorbar()
plt.title("Confusion Matrix (test set)")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=150)

model.save("models/crop_model.keras")
print("Saved models/crop_model.keras")
