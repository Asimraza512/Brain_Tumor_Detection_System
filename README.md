# Brain_Tumor_Detection_System

📌 Project Overview

This project is a Brain Tumor Detection System that uses Deep Learning (VGG16 transfer learning) to classify MRI images into different categories. The model is deployed using a Flask web application with a simple frontend (HTML, CSS, JS) for user interaction.

It predicts whether a brain MRI scan contains a tumor and identifies its type with confidence score.


🚀 Features

🧠 Detects brain tumor from MRI images

🤖 Deep Learning model using VGG16 (Transfer Learning)

🌐 Web-based interface using Flask

📊 Shows prediction with confidence score

📁 Supports image upload from user

🔄 Automatic model download from Google Drive (if not available locally)




🧬 Tumor Classes

The model classifies MRI images into:

Glioma Tumor

Meningioma Tumor

Pituitary Tumor

No Tumor



⚙️ Tech Stack

🔹 Deep learning / AI

Python

TensorFlow / Keras

VGG16 (Pretrained Model)

NumPy

Scikit-learn

Matplotlib & Seaborn

🔹 Web Development

Flask (Backend)

HTML

CSS

JavaScript




🏗️ Model Architecture

Base Model: VGG16 (ImageNet weights)

Custom Layers:

Flatten layer

Dense (128 neurons, ReLU)

Dropout layers

Softmax output layer

Loss Function: Categorical Crossentropy

Optimizer: Adam (lr = 0.0001)



🚀 How to Run Project

python app.py


🧪 Model Training Summary

Image Size: 224x224

Batch Size: 12

Epochs: 10

Training includes image augmentation (brightness + contrast)

Evaluation using:

    Confusion Matrix
    Classification Report
    ROC Curve


📊 Results

Achieved good accuracy on MRI dataset

Model performs well in distinguishing tumor vs non-tumor cases

Confidence score provided for each prediction


🌐 Web App Workflow

User uploads MRI image

Flask backend receives image

Image is processed and resized (224x224)

Model predicts tumor type

Result + confidence shown on UI
