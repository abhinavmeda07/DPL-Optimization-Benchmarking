# Deep Learning Model Optimization & Benchmarking Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-ff6f00.svg)](https://www.tensorflow.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-1.8%2B-brightgreen.svg)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end framework and benchmarking pipeline for deep learning model optimization, graph conversion, quantization, and resource profiling across edge and server environments.

---

## 📌 Project Overview
Deploying deep learning models to production environments requires balancing **inference speed (latency/FPS)**, **memory footprint (RAM/VRAM)**, **hardware utilization (CPU/GPU)**, and **storage size**. 

This repository implements a rigorous performance benchmark evaluating an image classification pipeline across four deployment stages:
1. **PyTorch Baseline (FP32)**: Standard eager-execution model instance.
2. **ONNX Runtime (FP32)**: Open Neural Network Exchange optimized computation graph.
3. **TensorFlow Lite Baseline (FP32)**: Mobile-optimized flatbuffer format.
4. **TensorFlow Lite Quantized (INT8)**: Post-Training Quantized (PTQ) model utilizing full integer arithmetic for compressed memory bandwidth and accelerated edge execution.

---

## 📊 Summary Performance Benchmarks (500 Test Samples)

| Stage | Latency (ms) | FPS | RAM Usage (MB) | CPU Usage (%) | GPU Mem (MB) | Storage Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch Baseline (FP32)** | ~12.45 | ~80.32 | 1420.50 | 45.2 | 184.20 | 16.80 |
| **ONNX Runtime (FP32)** | ~9.12 | ~109.65 | 1150.30 | 38.4 | 0.00 | 16.50 |
| **TFLite Baseline (FP32)** | ~10.80 | ~92.59 | 1110.20 | 40.1 | 0.00 | 16.20 |
| **TFLite Quantized (INT8)** | **~4.35** | **~229.88** | **940.10** | **22.5** | **0.00** | **4.45** |

---

## 🛠️ Tech Stack & Dependencies
* **Core Frameworks**: PyTorch, Torchvision, TensorFlow, Keras
* **Optimization Runtimes**: ONNX, ONNX Runtime, TensorFlow Lite Converter
* **Profiling & Metrics**: `psutil`, `time`, `GPUtil`
* **Visualization & Reporting**: Pandas, Matplotlib, Seaborn, Gradio

---

## 🚀 Quick Start / Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/abhinavmeda07/deep-learning-optimization-benchmark.git](https://github.com/abhinavmeda07/deep-learning-optimization-benchmark.git)
cd deep-learning-optimization-benchmark
pip install -r requirements.txt
