from ultralytics import YOLO

model = YOLO('/root/ultralytics/runs/detect/x_15_train/weights/last.pt')
results = model.train(save=True, resume=True)