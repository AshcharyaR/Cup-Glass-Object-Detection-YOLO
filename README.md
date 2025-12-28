YOLO Cup/Glass Object Detection
This project trains a YOLOv5 model to detect cups and glasses in images using a custom dataset from Roboflow. The notebook runs in Google Colab with GPU acceleration.
​

Features
Detects two classes: cup and glass.

Trained for 100 epochs on 416x416 images with batch size 32.

Achieves final validation mAP@50-95 of 0.592 (cup: 0.681, glass: 0.503).
​

Uses YOLOv5s pretrained weights fine-tuned on custom data.

Dataset
Source: Roboflow project "My-First-Project" version 2.

Train: 447 images, Valid: 40 images, Test images available.

Classes: 26 cups, 41 glasses in validation set.
​

Training Results
Metric	Value
mAP@50	0.835
mAP@50:95	0.592
Precision	0.808
Recall	0.788
Best weights	runs/train/exp/weights/best.pt (14.3 MB)
​
Setup Instructions
Clone YOLOv5 repository: !git clone https://github.com/ultralytics/yolov5.

Install dependencies: !pip install -qr requirements.txt.

Download dataset via Roboflow API (requires API key).
​

Training Command
text
!python train.py --img 416 --batch 32 --epochs 100 --data datasets/My-First-Project-2/data.yaml --weights yolov5s.pt --cache
Runs on Tesla T4 GPU with PyTorch 2.8.0+cu126.
​

Inference Examples
Test on dataset images: Successfully detects cups/glasses with conf 0.1.

Custom image example: !python detect.py --weights runs/train/exp/weights/best.pt --img 416 --conf 0.1 --source your_image.jpg.
​

Model Performance
The model shows steady improvement, reaching peak performance around epoch 99 with box loss ~0.014 and strong detection on test images like wine glasses and coffee cups.
