# shadowfox-ml-vision
# ShadowFox ML Vision Internship - Image Tagging using TensorFlow

This project is part of my AI/ML internship with **ShadowFox**, focused on building an **Image Tagging model using TensorFlow**.

## 🔍 Project Description
This model classifies images into categories like **cats and dogs** using TensorFlow. A small custom dataset was used and deployed via a **Streamlit** interface.

## 🧠 Technologies Used
- Python
- TensorFlow
- Streamlit
- NumPy, PIL
- Git & GitHub

## 🎯 Key Features
- Custom image dataset loading and preprocessing
- CNN-based image classification model
- Live prediction via a Streamlit web app
- Project runs in virtual environment (`venv` excluded from repo)

## 🚀 How to Run
```bash
# Activate the virtual environment
.\venv\Scripts\activate

# Train the model
python train.py

# Launch the web app
streamlit run app.py
