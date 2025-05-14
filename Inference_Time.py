import pandas as pd
import time
from ultralytics import YOLO

def save_inference_time_to_csv(model_path, curve_csv_path):
    # 加载模型
    model = YOLO(model_path)

    # 手动测量推理时间
    start_time = time.time()  # 开始时间
    results = model(source='datasets/ship-detect2/test/images',
                    save=True, save_txt=True, save_conf=True, name='1_test2')
    end_time = time.time()  # 结束时间

    # 计算推理时间（毫秒）
    inference_time_ms = (end_time - start_time) * 1000

    # 读取 curve.csv 文件
    try:
        curve_data = pd.read_csv(curve_csv_path)
    except FileNotFoundError:
        print(f"Error: The file {curve_csv_path} does not exist.")
        return

    # 确保 'Inference Time (ms)' 列存在
    if 'Inference Time (ms)' not in curve_data.columns:
        curve_data['Inference Time (ms)'] = None

    # 假设模型信息对应于第一个 epoch
    curve_data.loc[0, 'Inference Time (ms)'] = inference_time_ms

    # 保存更新后的 curve.csv 文件
    curve_data.to_csv(curve_csv_path, index=False)
    print(f"Inference Time 已成功添加到 {curve_csv_path}")

# 使用示例
model_path = 'runs/detect/1_train/weights/best.pt'  # 替换为你的模型路径
curve_csv_path = 'curve.csv'  # 替换为你的 curve.csv 文件路径
save_inference_time_to_csv(model_path, curve_csv_path)