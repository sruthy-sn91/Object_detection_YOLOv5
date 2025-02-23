Live Object Detection Web App

This project demonstrates a live object detection web application built using Flask and a pre-trained YOLOv5 model from PyTorch Hub. The app captures live video from your webcam, processes each frame with YOLOv5 to detect objects, and streams the annotated video via a web interface. The project is also Dockerized for ease of deployment.

Features

Real-Time Object Detection: Uses YOLOv5 to perform object detection on live camera feeds.
Web Interface: A simple Flask web app displays the live annotated video.
Dockerized: Containerized application for easy deployment and reproducibility.
Cross-Platform: Developed to run on macOS (e.g., MacBook M2 Max) and other platforms.

Requirements

Python 3.9 (or change the version in the Dockerfile/environment as needed)
Conda or venv for local environment management (optional)
Docker (for containerized deployment)
