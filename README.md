# AI X-Ray Scanner System
M.Sc. AI & ML (Semester 3) project: a Flask web app that classifies chest X-rays as **Normal** or **Pneumonia** with a CNN (MobileNetV2 transfer learning).
For academic demonstration only. It is not a medical diagnosis.

## 1. Setup (VS Code terminal)
Use Python 3.10-3.12 (TensorFlow does not support newer versions yet).

    python -m venv venv
    venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
    pip install -r requirements.txt

## 2. Get the dataset
Download "Chest X-Ray Images (Pneumonia)" from Kaggle (paultimothymooney/chest-xray-pneumonia).
Copy its `train`, `val` and `test` folders into `dataset/` so you get
`dataset/train/NORMAL`, `dataset/train/PNEUMONIA`, `dataset/validation/...`, `dataset/test/...`
(rename `val` to `validation`).
The original validation set has only 16 images, so the training script automatically uses 20% of `train` for validation in that case.

## 3. Train the model (once)

    python model/train_model.py

This saves `model/xray_cnn_model.keras` and writes test accuracy, precision, recall and the confusion matrix to `model/test_results.txt` (use these in your report).
Training takes roughly 10-30 minutes on a CPU.

## 4. Run the app

    python app.py

Chrome opens at http://127.0.0.1:5000. Register an account, log in, then use Scan X-Ray.

## How it works
- `app.py`: routes, login/registration (SQLite, hashed passwords), sessions, protected pages.
- `model/predict.py`: image validation, then CNN probabilities (real sigmoid output).
- `model/train_model.py`: data pipeline, augmentation, class weights, two-phase training, test evaluation.
- Invalid images show "This image is invalid. Please use a chest X-ray image." and are never predicted.

## Known limitations (mention these in your viva)
- Image validation is rule-based (greyscale, tonal range, left-right symmetry, centre-vs-lung brightness). It is a filter, not a classifier, so it can make mistakes. A stronger version would train a small "X-ray / not X-ray" CNN.
- The model is trained on pediatric X-rays from a single source, so accuracy on adult or other-hospital images may be lower.
- Set a real `SECRET_KEY` environment variable if you deploy anywhere beyond your laptop.
