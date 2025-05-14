from ultralytics import YOLO

# Load a model
model = YOLO('/graduate work/ultralytics-8.0/runs/detect/1_train/weights/best.pt')  # load an official model

# Predict with the model
results = model(source='/graduate work/ultralytics-8.0/datasets/ship-detect2/test/images',
                save=True, save_txt=True, name='1_test-2'
                )  # predict on an image

