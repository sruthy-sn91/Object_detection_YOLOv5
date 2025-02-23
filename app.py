# Monkey-patch for NumPy compatibility with YOLOv5 dependencies
import numpy as np
if not hasattr(np, '_no_nep50_warning'):
    np._no_nep50_warning = lambda: None

from flask import Flask, render_template, Response
import cv2
import torch

app = Flask(__name__)

# Check available device (MPS on Mac M2 Max or fallback to CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load YOLOv5 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device)
model.eval()

# Open the default camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Error: Could not open video capture. Ensure your camera is connected.")

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert frame to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run object detection
        results = model(img_rgb)
        results.render()  # Annotates the frame; rendered image is stored in results.ims

        # Convert the annotated frame to JPEG
        annotated_frame = results.ims[0]
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # Yield the output frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    # Render the main page
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Return the response generated along with the specific media type (mime type)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
