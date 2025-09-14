# AI-powered-disaster-relief-image-classification-using-DL
Disaster Image Classification using Deep Learning
📌 Project Overview

This project focuses on automated disaster image classification using the ASONAM17 / MEDIC dataset.
The goal is to classify images into 7 disaster types (earthquake, flood, hurricane, landslide, mild, none, severe).

We trained a ResNet-18 CNN with transfer learning and evaluated it on development/test splits.
The model was further deployed as a Flask web application for real-time image prediction.

🎯 Learning Objectives

To preprocess and organize disaster datasets for deep learning.

To train and evaluate a robust classification model (ResNet-18).

To visualize model performance using confusion matrix and classification reports.

To deploy the trained model into a simple web app for real-time predictions.

🚨 Problem Statement

Natural disasters create massive amounts of image data on social media.
Manually analyzing them for damage severity or disaster types is slow and inefficient.

The problem:
➡️ How can we automatically classify disaster-related images into meaningful categories to assist relief operations?

✅ Solution

Used pretrained ResNet-18 architecture.

Applied data augmentation (resized crops, flips, normalization).

Optimized with Adam optimizer, CrossEntropyLoss, and StepLR scheduler.

Evaluated using Precision, Recall, F1-score, and Confusion Matrix.

Deployed the trained model with Flask + HTML frontend.

🛠️ Tools and Technologies Used

Python, PyTorch, Torchvision

Pandas, NumPy, Matplotlib, Seaborn

Scikit-learn (metrics & evaluation)

Flask (deployment)

Google Colab + Google Drive (training & storage)

📂 Dataset

Dataset: ASONAM17 / MEDIC

Google Drive Link: ASONAM17_Damage_Image_Dataset
 (replace with your link)

Structure reorganized into 7 folders (one per class).

📊 Results

Macro F1 Score (Dev Set): ~0.22

Classification Report & Confusion Matrix were generated.

Flask app successfully predicts disaster type from uploaded image.

Confusion Matrix Example:

🚀 Deployment

Run locally or in Colab:

python app.py


Then open the URL displayed (or via flask_ngrok in Colab).

📑 Project Structure
disaster_classification_project/
│
├── requirements.txt           # library requirements
├── app.py                     # Flask web app
├── eval.py                    # Evaluation script
├── DisasterClassification.ipynb    # Main training notebook
├── outputs/                   # Training logs/history
├── README.md                  # This file

🔑 Improvisations Done

Reorganized dataset into 7-class folders automatically using scripts.

Added classification_report.csv export.

Generated confusion_matrix.png for PPT/reporting.

Built Flask app for easy demo of the trained model.

📜 License

This project is for educational purposes only
