# Monkey-patch for NumPy compatibility with YOLOv5 dependencies
import numpy as np
if not hasattr(np, '_no_nep50_warning'):
    np._no_nep50_warning = lambda: None

import cv2
import torch

def main():
    # Check if MPS (Metal Performance Shaders) is available on your MacBook M2 Max
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load YOLOv5 model from PyTorch Hub (this downloads the model weights on first run)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device)
    model.eval()  # Set the model to evaluation mode

    # Open the default camera (0 is usually the built-in webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture. Check if your camera is connected.")
        return

    print("Press 'q' to quit the camera feed.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        # Convert the frame from BGR to RGB as required by the model
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection on the frame
        results = model(img_rgb)
        
        # Render the detection results (bounding boxes, labels) on the image
        results.render()  # results.ims now contains the annotated frame

        # Display the annotated frame using the updated attribute "ims"
        annotated_frame = results.ims[0]
        cv2.imshow("YOLOv5 Live Object Detection", annotated_frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
