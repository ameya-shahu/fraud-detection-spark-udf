# Fraud ID classification on Apache Spark

## 📌 Project Overview

This project integrates CNN-based image classification into Apache Spark workflows using Spark SQL UDFs. It enables batch inference directly within Spark, facilitating scalable, distributed image processing.

The project includes:
- A CNN model for ID image classification
- A **Spark-based** pipeline using UDFs for distributed inference
- A **Python-only** script for benchmarking and comparison

---

## 🚀 Environment Setup (Docker)

The project uses Docker to ensure a consistent environment.

### 🔧 Prerequisites
- Docker installed
- For GPU: NVIDIA Container Toolkit (`--gpus all`)
- For CPU-only: Remove `--gpus all` from commands

### 📦 Build Docker Image

```bash
docker build -t spark-cnn-runtime .
```

---

## ⚙️ Usage

### 1. 🔁 Batch Inference with Spark

Run distributed inference using Spark:

```bash
docker run --rm -it --gpus all \
  -v $(pwd):/app \
  spark-cnn-runtime \
  python /app/spark_infer.py [backend]
```

- `[backend]`: `pytorch` or `onnx` (defaults to `pytorch`)

Example:

```bash
docker run --rm -it --gpus all \
  -v $(pwd):/app \
  spark-cnn-runtime \
  python /app/spark_infer.py pytorch
```

### 2. 🧪 Standalone Inference (Python Only)

Run inference sequentially using plain Python:

```bash
docker run --rm -it --gpus all \
  -v $(pwd):/app \
  spark-cnn-runtime \
  python /app/python_infer.py
```

---

## 🧱 Project Structure

```
.
├── spark_infer.py              # Spark UDF + Inference pipeline
├── python_infer.py             # Python-only inference for benchmarking
├── model/
│   ├── id_classification_cnn_model.pth
│   └── id_classification_cnn_model.onnx
├── dataset/
│   ├── all_images/             # Folder of input images
│   └── image_data.csv          # CSV of image paths and labels
├── output_predictions/         # Spark results
├── Dockerfile
└── ...
```

---

## 🧠 How It Works

### ⚡ Spark UDF Architecture

- Loads image metadata into Spark DataFrame
- Registers UDF: `get_image_inference_udf(path, backend)`
- Lazily loads model once per executor
- Performs CNN inference on each image using PyTorch or ONNX
- Outputs structured results: class, confidence, probabilities

### 🐍 Python-Only Mode

- Loads model with PyTorch
- Loops through image CSV with pandas
- Applies inference sequentially
- Exports predictions and confusion matrix

---

## 🆚 Benchmarking: Spark vs Python

| Mode           | Description                   | Pros                             | Cons                    |
|----------------|-------------------------------|----------------------------------|-------------------------|
| **Spark (UDF)**| Parallel, distributed inference| Fast on large datasets, scalable | Startup overhead        |
| **Python**     | Sequential loop with PyTorch   | Simple to test/debug             | Slow for large datasets |

> In tests, Spark UDF mode with GPU showed significant speedup over plain Python (even with GPU) due to parallelism.

---

## 📊 Sample Performance Logs

### Spark UDF (PyTorch):

```
Prediction Details Query Time: 0.115s
Confusion Matrix Query Time: 0.162s
```

### Spark UDF (ONNX):

```
Prediction Details Query Time: 0.123s
Confusion Matrix Query Time: 0.090s
```

### Python:

```
Prediction Details: 13.83s
Confusion Matrix: 13.87s
```

---

## 📄 License

This project is licensed under the MIT License.

Feel free to use, modify, and distribute — just credit the authors 🙌

---

## ✍️ Authors

- [Ameya Shahu](https://github.com/ameya-shahu)
- [Ayushi Rajshekar]()
- [Harshit Sharma]()
- [Sarthak Rana]()

---
