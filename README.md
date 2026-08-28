# 🚨 DeepVision Crowd Monitor

### AI-Powered Crowd Density Estimation & Overcrowding Detection

> An end-to-end Computer Vision system that estimates crowd density, visualizes high-risk areas, classifies crowd safety levels, and generates automated alerts.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv)](https://opencv.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Deployment-2496ED?logo=docker)](https://www.docker.com/)

---

## 🎯 Overview

**DeepVision Crowd Monitor** uses deep learning-based density estimation to analyze crowded scenes where traditional person detection can struggle because of heavy occlusion.

The system takes an image, estimates the crowd density using CNN models, generates a density heatmap, calculates the crowd count, and assigns a safety level based on configurable thresholds.

---

## ✨ Key Features

* 🧠 **Crowd Density Estimation** using CSRNet and lightweight CNN models
* 🔥 **Density Heatmaps** to visualize crowded regions
* 🚦 **4-Level Safety Classification** — Safe, Moderate, High Risk, Critical
* ⚡ **CUDA GPU Acceleration** for faster inference
* 🌐 **FastAPI Backend** for model serving
* 📊 **Streamlit Dashboard** for interactive analysis
* 🔔 **Twilio / SMTP Alerts** for threshold-based notifications
* 🐳 **Docker Support** for deployment

---

## 🏗️ How It Works

```text
Input Image
     ↓
Preprocessing
     ↓
Crowd Density Model
(CSRNet / MobileNetCSRNet / SimpleCNN)
     ↓
Density Map
     ↓
Crowd Count
     ↓
Safety Assessment
     ↓
Dashboard + Heatmap + Alerts
```

---

## 🧠 Model Performance

The project evaluates multiple approaches based on accuracy and inference speed.

| Model           |       MAE |      RMSE | Inference |
| --------------- | --------: | --------: | --------: |
| 🏆 **CSRNet**   | **49.27** | **72.03** |    150 ms |
| MobileNetCSRNet |    109.41 |    149.92 |     60 ms |
| SimpleCNN       |      87.5 |     124.8 |     40 ms |
| RandomForest    |     142.3 |     198.7 |    220 ms |

**CSRNet** was selected as the primary model because it provided the strongest crowd-counting performance in the evaluation.

---

## 📊 Dashboard

The Streamlit dashboard provides:

* Image upload and model selection
* Crowd-count estimation
* Density-map visualization
* Heatmap generation
* Safety-level classification
* Automated recommendations
* Downloadable analysis reports

---

## 🛠️ Tech Stack

**AI / ML:** PyTorch, CSRNet, MobileNetCSRNet, CNN
**Computer Vision:** OpenCV, NumPy, Pillow, SciPy
**Backend:** FastAPI, Uvicorn
**Dashboard:** Streamlit, Matplotlib, Pandas
**Deployment:** Docker, Nginx, CUDA
**Notifications:** Twilio, SMTP

---

## 📂 Project Structure

```text
deepvision-crowd-monitor/
│
├── backend/          # FastAPI backend
├── frontend/         # Streamlit dashboard
├── models/           # Crowd-counting models
├── preprocessing/    # Data preprocessing
├── results/          # Evaluation & visualizations
├── EDA/              # Exploratory analysis
├── src/              # Core source code
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Run Locally

### 1. Clone

```bash
git clone https://github.com/Ayush73raj/deepvision-crowd-monitor.git
cd deepvision-crowd-monitor
```

### 2. Install dependencies

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### 3. Start the application

Run the FastAPI backend and Streamlit frontend using the entry files provided in the repository.

---

## 🌍 Applications

The system can be adapted for:

* 🚉 Railway & Metro Stations
* 🎟️ Concerts & Stadiums
* 🛕 Religious Gatherings
* 🏙️ Smart-City Monitoring
* 🛍️ Retail & Shopping Malls
* 🎓 Campus Security

---

## 🔮 Future Improvements

* 🎥 Real-time CCTV/video-stream analysis
* 📹 Multi-camera crowd monitoring
* 📈 Predictive overcrowding alerts
* 📱 Mobile monitoring application
* 🤖 Edge deployment on Jetson/TPU devices
* ☁️ Cloud-based large-scale deployment

---

## 💡 What This Project Demonstrates

This project combines **Machine Learning + Computer Vision + Backend Development + Deployment** into one complete system.

> **From image processing and model training to API serving, visualization, GPU acceleration, and automated alerts.**

---

## 👨‍💻 Author

### Ayush Raj

**B.Tech — Electronics System Engineering**

Interested in **Computer Vision, Edge AI, Machine Learning and AI Deployment**.

[GitHub](https://github.com/Ayush73raj) • [LinkedIn](https://www.linkedin.com/)

---

⭐ **If you find this project useful, consider giving it a star!**

> **DeepVision Crowd Monitor — From crowd images to actionable safety intelligence.**
