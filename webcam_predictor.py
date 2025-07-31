# webcam_predictor.py
import cv2
from inference_model import infer_image
import tempfile
import os

BLUR_THRESHOLD = 50  # Adjust based on testing

def check_blur(image):
    """Check image blur using Laplacian variance"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'c' to capture or 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Skin Cancer Detection", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        print("Exiting...")
        break
    elif key == ord('c'):
        blur_variance = check_blur(frame)
        if blur_variance < BLUR_THRESHOLD:
            print(f"Image too blurry (Variance: {blur_variance:.1f}). Please recapture.")
            continue

        # Save frame temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            cv2.imwrite(tmp_file.name, frame)
            image_path = tmp_file.name

        try:
            result = infer_image(image_path)
            result_text = f"Prediction: {result['prediction']} ({result['confidence']})"
            print(result_text)
        except RuntimeError as e:
            result_text = "Error during inference"

        # Show result overlay on the frame
        display_frame = frame.copy()
        cv2.putText(display_frame, result_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Sharpness: {blur_variance:.1f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Analysis Result", display_frame)
        cv2.waitKey(0)
        cv2.destroyWindow("Analysis Result")

        # Clean up temp image
        os.remove(image_path)

cap.release()
cv2.destroyAllWindows()
