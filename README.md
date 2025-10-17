# 🚗 Driver Drowsiness Detection 😴  
![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> ⚡ A real-time **AI-powered safety system** that detects driver drowsiness using **Deep Learning** and **Computer Vision**.  
> The project helps reduce road accidents by monitoring eye activity and triggering alerts when drowsiness is detected.

---

## 🧭 Table of Contents

- [🧠 Overview](#-overview)
- [✨ Features](#-features)
- [🔄 Project Workflow](#-project-workflow)
- [🧰 Technologies Used](#-technologies-used)
- [📊 Dataset](#-dataset)
- [🧩 Model Architecture](#-model-architecture)
- [▶️ How to Run](#️-how-to-run)
- [📈 Results](#-results)
- [🚀 Future Improvements](#-future-improvements)
- [👨‍💻 Contributors](#-contributors)
- [📄 License](#-license)

---

## 🧠 Overview

Drowsy driving is a major cause of road accidents worldwide.  
This project leverages **Convolutional Neural Networks (CNNs)** to classify whether a driver is **alert or drowsy** based on live camera input.  
It demonstrates the potential of AI in improving real-world safety systems.

> 💡 **Note:** This was my **Capstone Project** during my **Summer Internship at IIITDM Jabalpur**.  
> It was originally implemented and executed in **Google Colab** as per internship guidelines — therefore, the file `drowsiness.ipynb` is attached here.

---

## ✨ Features

✅ Real-time video stream analysis using **OpenCV**  
✅ Eye-state detection (Open / Closed)  
✅ Drowsiness alert with sound warnings  
✅ Lightweight CNN architecture using **TensorFlow**  
✅ Easy integration with embedded or IoT systems  

---

## 🔄 Project Workflow

1. 📸 **Data Collection:** Gather eye images (open & closed states)  
2. 🧹 **Preprocessing:** Resize, normalize & label data  
3. 🧠 **Model Building:** Build a CNN model using Keras/TensorFlow  
4. 🧪 **Training:** Train on labeled dataset to classify eye state  
5. 🎥 **Detection:** Monitor driver via webcam feed in real time  
6. 🚨 **Alert:** Trigger alarm when eyes stay closed for several frames  

---

## 🧰 Technologies Used

| Tool / Library | Purpose |
|-----------------|----------|
| 🐍 **Python 3.x** | Programming Language |
| 🧠 **TensorFlow / Keras** | Deep Learning Framework |
| 👁️ **OpenCV** | Real-time Video Processing |
| 📊 **NumPy, Pandas** | Data Handling |
| 📉 **Matplotlib** | Visualization |
| 🧪 **Scikit-learn** | Model Evaluation |

---

## 📊 Dataset

The model is trained on eye-state datasets (Open/Closed).  
You can use the [MrlEye Dataset](https://mrl.cs.vsb.cz/eyedataset) or any similar dataset.

Each image is preprocessed into grayscale and resized (24×24) for CNN input.

**Labels:**
- 👁️ `Open`
- 😴 `Closed`

---

## 🧩 Model Architecture

The CNN consists of:
- **Conv2D + MaxPooling2D** → for feature extraction  
- **Flatten + Dense layers** → for classification  
- **Sigmoid activation** → for binary output (Open / Closed)

> 🧾 Model achieves high accuracy in detecting eye states and is optimized for real-time inference.

---

## ▶️ How to Run

### 💻 Option 1 — Run on Google Colab  
📎 [**Open in Google Colab**](https://colab.research.google.com/drive/1Ev11FrgWc14bWhgyrJloSUX6_JBXjWkx)  
Follow the notebook and execute cells sequentially.

### 🖥️ Option 2 — Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/Driver-Drowsiness-Detection.git
cd Driver-Drowsiness-Detection

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook drowsiness.ipynb
```bash


