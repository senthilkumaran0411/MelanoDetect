# 🧠 MelanoDetect: Skin Cancer Prediction Web App

**MelanoDetect** is a Flask-based web application that uses deep learning models to detect skin cancer from uploaded images. The app supports pre-trained models like EfficientNet and RainbowFlow to analyze skin lesions and provide accurate results, along with downloadable PDF reports.

---
![image](https://github.com/senthilkumaran0411/MelanoDetector/blob/a753fe0412d34abd346fd89ffb151c6103de4377/uploads/Screenshot%202025-07-31%20215154.png)

## 🚀 Features

- 🔍 Upload skin lesion images for analysis
- 🧠 Predict cancer type using AI models
- 📸 Capture image using webcam (optional)
- 📊 Generate PDF report of the prediction
- 💡 Check image quality using blur detection
- 🌐 Simple UI built with Flask and HTML/CSS

---

## 🗂️ Project Structure

MelanoDetect/
├── app.py # Main Flask application
├── efficientnetb0.h5 # EfficientNet model file
├── rainbowflow_model.h5 # RainbowFlow model file
├── inference_model.py # Handles prediction logic
├── pre-model.py # Preprocessing functions
├── webcam_predictor.py # Webcam capture and prediction
│
├── static/
│ └── style.css # Frontend styles
│
├── templates/
│ ├── index.html # Upload page
│ ├── result.html # Result display page
│ └── report_form.html # Form to generate report
│
├── uploads/
│ └── webcam_capture.jpg # Stores user-uploaded images
│
└── reportlab/ # Report generation modules


---

## 🧠 Models Used

- **EfficientNetB0**: Lightweight CNN for image classification.
- **RainbowFlow**: Advanced model fine-tuned for high-accuracy skin cancer detection.

Models are loaded using:

```python
from tensorflow.keras.models import load_model
model = load_model("efficientnetb0.h5")  # or rainbowflow_model.h5

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/senthilkumaran0411/MelanoDetector.git
cd MelanoDetect

2. Create a virtual environment
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies
bash
pip install -r requirements.txt
4. Run the app
bash
python app.py
Visit http://127.0.0.1:5000 in your browser to access the web interface.

📄 PDF Report Generation
Uses ReportLab to generate a downloadable PDF report after prediction.

The report includes:

Uploaded image

Prediction result

Confidence score

Date and time of analysis

📸 Webcam Support
Capture real-time skin images using a webcam

Predict directly from the captured image using webcam_predictor.py

Useful for quick detection without uploading files manually
