"""Train the chest X-ray CNN (MobileNetV2 transfer learning).

Dataset: Kaggle "Chest X-Ray Images (Pneumonia)" by Paul Mooney (Kermany et al.)
Put it in dataset/train, dataset/validation, dataset/test (NORMAL / PNEUMONIA).
Run from the project folder:  python model/train_model.py
"""
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dataset")
OUT = os.path.join(ROOT, "model", "xray_cnn_model.keras")
IMG, BATCH, SEED = (224, 224), 32, 42
tf.keras.utils.set_random_seed(SEED)


def count(split, cls):
    return len(os.listdir(os.path.join(DATA, split, cls)))


def load(split, **kw):
    return tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA, split), label_mode="binary", image_size=IMG,
        batch_size=BATCH, class_names=["NORMAL", "PNEUMONIA"], **kw)  # PNEUMONIA = 1


# The original Kaggle validation folder has only 16 images, so if it is that
# small we carve 20% out of the training folder for validation instead.
if count("validation", "NORMAL") + count("validation", "PNEUMONIA") >= 100:
    train_ds, val_ds = load("train", shuffle=True, seed=SEED), load("validation")
else:
    print("Validation folder too small -> using 20% of train for validation")
    train_ds = load("train", validation_split=0.2, subset="training", seed=SEED)
    val_ds = load("train", validation_split=0.2, subset="validation", seed=SEED)
test_ds = load("test", shuffle=False)

# the dataset is imbalanced (more pneumonia than normal) -> class weights
n0, n1 = count("train", "NORMAL"), count("train", "PNEUMONIA")
total = n0 + n1
class_weight = {0: total / (2 * n0), 1: total / (2 * n1)}
print("Class weights:", class_weight)

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])
base = tf.keras.applications.MobileNetV2(input_shape=IMG + (3,), include_top=False, weights="imagenet")
base.trainable = False

inputs = tf.keras.Input(shape=IMG + (3,))
x = augment(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)   # scales 0-255 -> -1..1 inside the model
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)   # P(pneumonia)
model = tf.keras.Model(inputs, outputs)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(OUT, monitor="val_loss", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
]
metrics = ["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]

print("\n--- Phase 1: training the new classification head ---")
model.compile(tf.keras.optimizers.Adam(1e-3), "binary_crossentropy", metrics=metrics)
model.fit(train_ds, validation_data=val_ds, epochs=6, class_weight=class_weight, callbacks=callbacks)

print("\n--- Phase 2: fine-tuning the last layers of MobileNetV2 ---")
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False
model.compile(tf.keras.optimizers.Adam(1e-5), "binary_crossentropy", metrics=metrics)
model.fit(train_ds, validation_data=val_ds, epochs=6, class_weight=class_weight, callbacks=callbacks)

print("\n--- Evaluation on the unseen TEST set ---")
model = tf.keras.models.load_model(OUT)
y_true = np.concatenate([y.numpy().ravel() for _, y in test_ds]).astype(int)
y_prob = model.predict(test_ds).ravel()
y_pred = (y_prob >= 0.5).astype(int)
report = classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"])
cm = confusion_matrix(y_true, y_pred)
print(report, "\nConfusion matrix:\n", cm)
with open(os.path.join(ROOT, "model", "test_results.txt"), "w") as f:
    f.write(report + "\nConfusion matrix:\n" + str(cm))
print("Model saved to", OUT)
