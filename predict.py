"""Image validation + CNN prediction used by the Flask app."""
import os
import numpy as np
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xray_cnn_model.keras")
IMG_SIZE = (224, 224)
_model = None


class ModelNotTrainedError(Exception):
    pass


def validate_xray(path):
    """Heuristic check that the picture looks like a chest X-ray.

    Chest X-rays are (1) greyscale, (2) have a wide range of tones, (3) are
    roughly left-right symmetric (two lungs, spine in the middle) and (4) are
    not extreme in shape. Colourful photos, screenshots, blank images and
    most random pictures fail at least one of these tests.
    """
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return False
    w, h = img.size
    if min(w, h) < 128 or not (0.6 <= w / h <= 1.8):
        return False

    small = img.resize((128, 128))
    sat = np.asarray(small.convert("HSV"), dtype=np.float32)[..., 1] / 255.0
    if (sat > 0.25).mean() > 0.08:          # too colourful
        return False

    gray = np.asarray(small.convert("L"), dtype=np.float32)
    if gray.std() < 25 or not (35 < gray.mean() < 200):   # blank / washed out
        return False

    flipped = gray[:, ::-1]                  # left-right symmetry
    corr = np.corrcoef(gray.ravel(), flipped.ravel())[0, 1]
    if np.isnan(corr) or corr < 0.30:
        return False

    # centre (spine/mediastinum) is brighter than the darker lung fields
    centre = gray[:, 54:74].mean()
    lungs = np.concatenate([gray[:, 20:50].ravel(), gray[:, 78:108].ravel()]).mean()
    return bool(centre > lungs * 0.95)


def _load():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise ModelNotTrainedError(
                "Model file not found. Run  python model/train_model.py  first.")
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def predict_xray(path):
    """Return class probabilities (in %) straight from the CNN's sigmoid output."""
    model = _load()
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    x = np.expand_dims(np.asarray(img, dtype=np.float32), 0)   # preprocessing is inside the model
    p_pneumonia = float(model.predict(x, verbose=0)[0][0])
    pneu, norm = round(p_pneumonia * 100, 1), round((1 - p_pneumonia) * 100, 1)
    label = "Pneumonia" if p_pneumonia >= 0.5 else "Normal"
    return {"label": label, "normal": norm, "pneumonia": pneu,
            "confidence": max(norm, pneu)}
