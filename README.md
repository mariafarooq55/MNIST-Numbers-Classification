# MNIST Numbers Classification

A Streamlit web app that recognizes handwritten digits from 0 to 9 using a pre-trained scikit-learn model based on the MNIST dataset.

## Features

- Draw a digit in the browser with a canvas
- Upload an image of a handwritten digit
- Preprocess the drawing/image to match the MNIST format
- Predict the digit and show confidence for each class

## Project files

- `app.py` — Streamlit application
- `mnist_model.pkl` — trained model used for inference
- `requirements.txt` — Python dependencies
- `Lab 3 - MNIST Digit Recognition (Colab).ipynb` — notebook used for training and experimentation

## Requirements

Python 3.9+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

From the project folder:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, typically:

```text
http://localhost:8501
```

## How it works

1. The app captures either a user-drawn digit or an uploaded image.
2. The image is converted to grayscale and normalized into a 28x28-style digit representation.
3. The trained model predicts the most likely digit.
4. The result is displayed with a per-class probability chart.

## Model details

The app loads a pre-trained classifier from `mnist_model.pkl` and uses `predict_proba()` to estimate class probabilities. The preprocessing step centers and scales the digit so it matches the format expected by the model.

## Notes

This project is intended for local use and demonstration. For a production-ready deployment, you may want to add model versioning, input validation, and improved error handling.
