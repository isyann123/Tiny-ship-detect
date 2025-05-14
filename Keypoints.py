from glob import glob
from pathlib import Path
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh, xyxy2xywhn

labels = glob("/datasets/ship-detect2/train/labels/*.txt", recursive=True)

root_dir = Path(labels[0]).parent.parent.parent
for label in labels:
    label = Path(label)
    split = label.parent.parent.name
    img_path = [(root_dir / split / "images" / label.with_suffix(sfx).name)
                for sfx in [".png", ".jpg", ".PNG", ".JPG"]]
    img_path = [pth for pth in img_path if pth.exists()][0]
    boxes = []
    points = []
    classes = []
    save_pth = root_dir / split / "labels_kp" / label.name
    save_pth.parent.mkdir(exist_ok=True)
    with open(label) as f:
        lines = f.readlines()
        for line in lines:
            splits = line.rstrip().split(" ")
            if not splits or splits[0] == '':  # 检查splits[0]是否为空字符串
                continue  # 如果是空字符串，则跳过当前循环
            cls_id = int(splits[0])
            box = splits[1:]
            if not box:
                with open(save_pth, "w") as f:
                    pass
                continue

            box = [float(pt) for pt in box]
            point = (box[0], box[1])
            points.append(point)
            boxes.append(box)
            classes.append(cls_id)

    with open(save_pth, "w") as f:
        for point, box, cls_id in zip(points, boxes, classes):
            f.writelines(f"{cls_id} {box[0]} {box[1]} {box[2]} {box[3]} {point[0]} {point[1]} 1 \n")