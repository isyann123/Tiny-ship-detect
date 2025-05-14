
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
#model = YOLO("yolov8n.pt")  # load a pretrained model 不使用预训练权重，就注释这一行即可
# train
model.train(data='ship-detect2.yaml',  # 训练参数均可以重新设置
                        epochs=100,
                        save=True,
                        #dfl=0.01,
                        #box=0.01,

                        warmup_epochs=1.0,
                        close_mosaic=1,
                )
