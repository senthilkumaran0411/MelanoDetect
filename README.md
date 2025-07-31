# 🧠 MelanoDetect: Skin Cancer Prediction Web App

**MelanoDetect** is a Flask-based web application that uses deep learning models to detect skin cancer from uploaded images. The app supports pre-trained models like EfficientNet and RainbowFlow to analyze skin lesions and provide accurate results, along with downloadable PDF reports.

---

![MelanoDetect Screenshot](https://github.com/senthilkumaran0411/MelanoDetector/blob/a753fe0412d34abd346fd89ffb151c6103de4377/uploads/Screenshot%202025-07-31%20215154.png)

## 🚀 Features

- 🔍 Upload skin lesion images for analysis
- 🧠 Predict cancer type using AI models
- 📸 Capture image using webcam (optional)
- 📊 Generate PDF report of the prediction
- 💡 Check image quality using blur detection
- 🌐 Simple UI built with Flask and HTML/CSS

---

## 🗂️ Project Structure

```
MelanoDetect/
├── app.py                    # Main Flask application
├── efficientnetb0.h5         # EfficientNet model file
├── rainbowflow_model.h5      # RainbowFlow model file
├── inference_model.py        # Handles prediction logic
├── pre-model.py              # Preprocessing functions
├── webcam_predictor.py       # Webcam capture and prediction
│
├── static/
│   └── style.css             # Frontend styles
│
├── templates/
│   ├── index.html            # Upload page
│   ├── result.html           # Result display page
│   └── report_form.html      # Form to generate report
│
├── uploads/
│   └── webcam_capture.jpg    # Stores user-uploaded images
│
└── reportlab/                # Report generation modules
```

---

## 🧠 Models Used

- **EfficientNetB0**: Lightweight CNN for image classification.
- **RainbowFlow**: Advanced model fine-tuned for high-accuracy skin cancer detection.

Models are loaded using:

```python
from tensorflow.keras.models import load_model
model = load_model("efficientnetb0.h5")  # or rainbowflow_model.h5
```

---

## ⚙️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/senthilkumaran0411/MelanoDetector.git
   ```

2. **Navigate to the Project Directory:**
   ```bash
   cd MelanoDetect
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```bash
   python app.py
   ```

---

## 📄 PDF Report Generation

Uses ReportLab to generate a downloadable PDF report after prediction.

The report includes:
- Uploaded image
- Prediction result  
- Confidence score
- Date and time of analysis

---

## 📸 Webcam Support

**Features:**
- Real-time skin image capture using webcam
- Direct prediction from captured images
- No manual file upload required

**Usage:**
```bash
python webcam_predictor.py
```

---

## 🙏 Thank You!

<div align="center">
  
  ![Typing Animation](https://readme-typing-svg.demolab.com?font=Fira+Code&size=24&duration=2000&pause=500&color=22D3EE&width=500&lines=Thank+You+for+Using+MelanoDetect!;Your+Feedback+Matters+❤️)
  
  <br>
  
  <img src="https://raw.githubusercontent.com/senthilkumaran0411/MelanoDetector/main/assets/melano-logo-spin.gif" width="150" alt="MelanoDetect Logo">
  
  <br><br>
  
  **Contact Us:**
  
  📧 **Email:** [senthilkumaran0411@gmail.com](mailto:senthilkumaran0411@gmail.com)  
  💻 **GitHub:** [senthilkumaran0411](https://github.com/senthilkumaran0411)
  
</div>
