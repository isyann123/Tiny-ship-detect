import pandas as pd
import torch
from ultralytics import YOLO

def save_model_info_to_csv(model_path, curve_csv_path):
    # 加载模型
    model = YOLO(model_path)

    # 计算模型的总参数量
    total_parameters = sum(p.numel() for p in model.model.parameters()) / 1e6  # 转换为百万（M）

    # 读取 curve.csv 文件
    try:
        curve_data = pd.read_csv(curve_csv_path)
    except FileNotFoundError:
        print(f"Error: The file {curve_csv_path} does not exist.")
        return

    # 确保 'Parameters (M)' 列存在
    if 'Parameters (M)' not in curve_data.columns:
        curve_data['Parameters (M)'] = None

    # 假设模型信息对应于第一个 epoch
    curve_data.loc[0, 'Parameters (M)'] = total_parameters

    # 保存更新后的 curve.csv 文件
    curve_data.to_csv(curve_csv_path, index=False)
    print(f"Parameters 已成功添加到 {curve_csv_path}")

# 使用示例
model_path = 'runs/detect/1_train/weights/best.pt'  # 替换为你的模型路径
curve_csv_path = 'runs/detect/1_test2/labels_A/curve.csv'  # 替换为你的 curve.csv 文件路径
save_model_info_to_csv(model_path, curve_csv_path)