from ultralytics import YOLO
import time
from ultralytics import YOLO

model = YOLO("runs/pose/x_7_train/weights/best.pt")  # 可换成 yolov8s.pt, yolov8m.pt 等
results = model("datasets/ship-detect2-kp/train/images")  # 替换为实际图像路径
model.info(verbose=True)
