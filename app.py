import os
import cv2
import numpy as np
import datetime
from flask import Flask, render_template, request, send_file, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.utils import ImageReader

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
BLUR_THRESHOLD = 50  # Adjust based on your needs

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def analyze_image(image_path):
    """Perform image analysis and return results"""
    try:
        # Example analysis - replace with your actual model inference
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Perform your actual analysis here
        # This is just a placeholder - replace with your model's inference
        result_data = {
            "prediction": "Melanoma",  # Example prediction
            "confidence": 0.85,        # Example confidence
            "features": {              # Example additional features
                "asymmetry": 0.7,
                "border": 0.8,
                "color": 0.9,
                "diameter": 5.2
            }
        }
        return result_data
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None

@app.route('/')
def home():
    session.clear()  # Clear previous session data
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('home'))
    
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload a JPG, JPEG, or PNG image.', 'error')
        return redirect(url_for('home'))
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read and check the image
        image = cv2.imread(filepath)
        if image is None:
            flash('Could not read the image file', 'error')
            return redirect(url_for('home'))
        
        # Check image quality
        blur_variance = check_blur(image)
        if blur_variance < BLUR_THRESHOLD:
            flash('Image is too blurry. Please upload a clearer image.', 'error')
            return redirect(url_for('home'))
        
        # Analyze the image
        result_data = analyze_image(filepath)
        if not result_data:
            flash('Image analysis failed', 'error')
            return redirect(url_for('home'))
        
        # Store results in session
        session['image_path'] = filepath
        session['prediction'] = result_data.get('prediction', 'Unknown')
        session['confidence'] = f"{float(result_data.get('confidence', 0)) * 100:.2f}%"
        session['sharpness'] = f"{blur_variance:.1f}"
        session['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session['analysis_data'] = result_data  # Store all analysis data
        
        return render_template('result.html',
                             prediction=session['prediction'],
                             confidence=session['confidence'],
                             sharpness=session['sharpness'],
                             timestamp=session['timestamp'])
    
    except Exception as e:
        print(f"Error processing image: {e}")
        flash('Error processing image. Please try again.', 'error')
        return redirect(url_for('home'))

@app.route('/report-form')
def report_form():
    if 'prediction' not in session:
        flash('No analysis data available. Please upload an image first.', 'error')
        return redirect(url_for('home'))
    return render_template('report_form.html')

@app.route('/download-report', methods=['POST'])
def download_report():
    if 'prediction' not in session:
        flash('No analysis data available', 'error')
        return redirect(url_for('home'))
    
    try:
        # Get form data
        patient_info = {
            'name': request.form.get('patient_name', 'Not provided'),
            'age': request.form.get('patient_age', 'Not provided'),
            'gender': request.form.get('patient_gender', 'Not provided'),
            'doctor': request.form.get('doctor_name', 'Not provided'),
            'notes': request.form.get('additional_notes', '')
        }

        # Create PDF buffer
        buffer = BytesIO()
        doc = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        
        # Set up styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=18,
            alignment=1,
            spaceAfter=20
        )
        
        # Add title and header
        title = Paragraph("Skin Cancer Analysis Report", title_style)
        title.wrapOn(doc, width-100, 50)
        title.drawOn(doc, 50, height - 50)
        
        # Add patient information
        doc.setFont("Helvetica-Bold", 12)
        doc.drawString(50, height - 100, "Patient Information:")
        
        patient_data = [
            ["Patient Name:", patient_info['name']],
            ["Age:", patient_info['age']],
            ["Gender:", patient_info['gender']],
            ["Referring Doctor:", patient_info['doctor']],
            ["Report Date:", session['timestamp']]
        ]
        
        patient_table = Table(patient_data, colWidths=[150, 300])
        patient_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        patient_table.wrapOn(doc, width, height)
        patient_table.drawOn(doc, 70, height - 180)
        
        # Add image if available
        if 'image_path' in session and os.path.exists(session['image_path']):
            try:
                img = Image.open(session['image_path'])
                img.thumbnail((300, 300))
                
                # Save to temporary bytes buffer
                img_bytes = BytesIO()
                img.save(img_bytes, format='JPEG')
                img_bytes.seek(0)
                
                # Add image to PDF
                doc.drawString(50, height - 220, "Analyzed Image:")
                pdf_img = RLImage(ImageReader(img_bytes), width=200, height=200)
                pdf_img.wrapOn(doc, width, height)
                pdf_img.drawOn(doc, 70, height - 430)
            except Exception as e:
                print(f"Error adding image to PDF: {e}")
        
        # Add analysis results
        doc.setFont("Helvetica-Bold", 12)
        doc.drawString(50, height - 450, "Analysis Results:")
        
        result_data = [
            ["Parameter", "Value"],
            ["Prediction", session['prediction']],
            ["Confidence Level", session['confidence']],
            ["Image Quality", session['sharpness']],
            ["Analysis Date", session['timestamp']]
        ]
        
        # Add additional analysis features if available
        if 'analysis_data' in session and 'features' in session['analysis_data']:
            features = session['analysis_data']['features']
            for key, value in features.items():
                result_data.append([f"{key.capitalize()} Score:", f"{float(value):.2f}"])
        
        result_table = Table(result_data, colWidths=[200, 200])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d0d0d0')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        result_table.wrapOn(doc, width, height)
        result_table.drawOn(doc, 50, height - 550)
        
        # Add notes section
        notes_style = ParagraphStyle(
            'NotesStyle',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#555555'),
            spaceBefore=20
        )
        
        notes_text = f"""
        <b>Clinical Notes:</b><br/>
        {patient_info['notes'] or 'No additional notes provided.'}
        """
        
        notes = Paragraph(notes_text, notes_style)
        notes.wrapOn(doc, width-100, 100)
        notes.drawOn(doc, 50, height - 650)
        
        # Add disclaimer
        disclaimer_style = ParagraphStyle(
            'DisclaimerStyle',
            parent=styles['Italic'],
            fontSize=8,
            textColor=colors.HexColor('#777777'),
            spaceBefore=10
        )
        
        disclaimer = Paragraph("""
        <b>Disclaimer:</b> This report is generated by an AI system and should be reviewed by a qualified dermatologist. 
        The results are not a definitive diagnosis. Clinical examination and additional tests may be required for confirmation.
        """, disclaimer_style)
        disclaimer.wrapOn(doc, width-100, 100)
        disclaimer.drawOn(doc, 50, 50)
        
        doc.save()
        buffer.seek(0)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Skin_Cancer_Report_{timestamp}.pdf"
        
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        print(f"Error generating PDF: {e}")
        flash('Error generating report. Please try again.', 'error')
        return redirect(url_for('report_form'))

if __name__ == '__main__':
    app.run(debug=True)