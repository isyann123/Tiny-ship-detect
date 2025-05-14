import os
import pandas as pd
import torch
import time
import logging
from ultralytics import YOLO
from openpyxl import Workbook

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def init_value_csv(path):
    """初始化Value.csv文件结构"""
    if not os.path.exists(path):
        pd.DataFrame(columns=[
            'Inference Time (ms)', 'Parameters (M)','Precision','Recall','F1'
            'confidence'
        ]).to_csv(path, index=False)
        logger.info(f"已初始化 {path}")


def PredictWithInference(model_path, value_csv_path, test_images_dir, test_name, labels_dir):
    """
    执行模型推理并记录时间
    返回: 生成的labels目录路径
    """
    try:
        # 加载模型
        model = YOLO(model_path)
        logger.info(f"已加载模型: {model_path}")

        # 手动测量推理时间
        start_time = time.time()
        results = model(source=test_images_dir,
                        save=True, save_txt=True, save_conf=True, name=test_name)
        end_time = time.time()

        # 计算推理时间（毫秒）
        inference_time_ms = (end_time - start_time) * 1000
        logger.info(f"推理完成，耗时: {inference_time_ms:.2f} ms")

        # 等待结果生成
        max_wait = 30  # 最大等待时间(秒)
        while not os.path.exists(labels_dir) and max_wait > 0:
            time.sleep(1)
            max_wait -= 1

        if not os.path.exists(labels_dir):
            raise FileNotFoundError(f"推理结果未生成: {labels_dir}")

        # 读取或创建Value.csv
        value_data = pd.read_csv(value_csv_path) if os.path.exists(value_csv_path) else pd.DataFrame()

        # 更新推理时间
        value_data['Inference Time (ms)'] = [inference_time_ms]
        value_data.to_csv(value_csv_path, index=False)
        logger.info(f"已更新推理时间到 {value_csv_path}")

        return labels_dir

    except Exception as e:
        logger.error(f"推理过程中出错: {str(e)}")
        raise


def Confidence(folder_path, output_file, value_csv_path):
    """处理置信度数据"""
    try:
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"目录不存在: {folder_path}")

        # 获取所有txt文件
        txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        if not txt_files:
            raise ValueError(f"目录中没有找到txt文件: {folder_path}")

        # 读取并处理所有txt文件
        all_data = []
        for txt_file in txt_files:
            file_path = os.path.join(folder_path, txt_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, start=1):
                    if line.strip():  # 跳过空行
                        all_data.append({
                            '文件名': txt_file,
                            '行号': line_num,
                            '内容': line.strip()
                        })

        # 转换为DataFrame并处理数据
        df = pd.DataFrame(all_data)
        df[['类别', 'x_center', 'y_center', 'width', 'height', '置信度']] =df['内容'].str.split(' ', expand=True)

        # 保存Excel
        df[['类别', 'x_center', 'y_center', 'width', 'height', '置信度']].to_excel(
        output_file, index=False, engine='openpyxl')
        logger.info(f"置信度数据已保存到 {output_file}")

        # 更新Value.csv
        value_data = pd.read_csv(value_csv_path)
        value_data['confidence'] = df['置信度'].astype(float).mean()  # 使用平均置信度
        value_data.to_csv(value_csv_path, index=False)
        logger.info(f"置信度已更新到 {value_csv_path}")

    except Exception as e:
        logger.error(f"处理置信度时出错: {str(e)}")
        raise


import matplotlib.pyplot as plt
import numpy as np


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

def PRF1(train_csv_path, value_csv_path):
    """
    从训练日志读取metrics/precision(B)和metrics/recall(B)数据，
    计算F1分数，并绘制F1曲线、P曲线、R曲线和PR曲线

    参数:
        train_csv_path: 训练日志CSV文件路径
        value_csv_path: 输出Value.csv文件路径
    """
    try:
        # 1. 读取训练数据
        train_data = pd.read_csv(train_csv_path)

        # 检查必要列是否存在
        required_columns = ['   metrics/precision(B)', '      metrics/recall(B)']
        missing_cols = [col for col in required_columns if col not in train_data.columns]
        if missing_cols:
            raise ValueError(f"训练日志缺少必要列: {missing_cols}")


        # 2. 读取Value.csv
        if os.path.exists(value_csv_path):
            value_data = pd.read_csv(value_csv_path)
        else:
            value_data = pd.DataFrame()

            # 创建新的一行数据
        new_row = pd.DataFrame({
            'Precision': [train_data['   metrics/precision(B)'].astype(float).iloc[-1]],
            'Recall': [train_data['      metrics/recall(B)'].astype(float).iloc[-1]],
            'F1': [2 * (train_data['   metrics/precision(B)'].astype(float).iloc[-1] *
                        train_data['      metrics/recall(B)'].astype(float).iloc[-1]) /
                   (train_data['   metrics/precision(B)'].astype(float).iloc[-1] +
                    train_data['      metrics/recall(B)'].astype(float).iloc[-1] + 1e-9)]
        })

        # 追加新的数据行
        value_data = pd.concat([value_data, new_row], ignore_index=True)

        # 5. 绘制曲线
        output_dir = os.path.dirname(value_csv_path) or '.'
        epochs = np.arange(1, len(train_data) + 1)

        plt.figure(figsize=(15, 10))

        # F1曲线
        plt.subplot(2, 2, 1)
        plt.plot(epochs, value_data['F1'], 'b-', label='F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('F1 Score Curve')
        plt.grid(True)
        plt.legend()

        # Precision曲线
        plt.subplot(2, 2, 2)
        plt.plot(epochs, value_data['Precision'], 'g-', label='Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.title('Precision Curve')
        plt.grid(True)
        plt.legend()

        # Recall曲线
        plt.subplot(2, 2, 3)
        plt.plot(epochs, value_data['Recall'], 'r-', label='Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.title('Recall Curve')
        plt.grid(True)
        plt.legend()

        # PR曲线
        plt.subplot(2, 2, 4)
        plt.plot(value_data['Recall'], value_data['Precision'], 'm-', label='PR Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.legend()

        # 保存图像
        plt.tight_layout()
        curves_path = os.path.join(output_dir, 'performance_curves.png')
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"性能曲线已保存到 {curves_path}")

    except Exception as e:
        logging.error(f"PRF1处理过程中出错: {str(e)}", exc_info=True)
        raise

def Parameters(model_path, value_csv_path):
    """记录模型参数量"""
    try:
        model = YOLO(model_path)
        total_params = sum(p.numel() for p in model.model.parameters()) / 1e6  # 转换为百万

        value_data = pd.read_csv(value_csv_path)
        value_data['Parameters (M)'] = [total_params]
        value_data.to_csv(value_csv_path, index=False)

        logger.info(f"参数量记录成功: {total_params:.2f}M")

    except Exception as e:
        logger.error(f"记录参数量时出错: {str(e)}")
        raise


def main():
    try:
        # 初始化文件路径
        #选择模型
        model_name='2'
        pose=False

        # 初始化模型文件路径
        test_name = model_name + '_test'
        if not pose:
            test_dir = 'runs/detect/' + model_name + '_test'
            model_path = 'runs/detect/' + model_name + '_train/weights/best.pt'
            value_csv_path = 'runs/detect/' + model_name + '_train/Value.csv'#评估文件路径
            test_images_dir = 'datasets/ship-detect2/test/images'#test数据集路径
            train_csv_path = 'runs/detect/' + model_name + '_train/results.csv'#result.csv文件路径
            labels_dir = test_dir + '/labels_A'#confidence存放路径
        else:
            test_dir = 'runs/pose/' + model_name + '_test'
            model_path = 'runs/pose/' + model_name + '_train/weights/best.pt'
            value_csv_path = 'runs/detect/' + model_name + '_train/Value.csv'
            test_images_dir = 'datasets/ship-detect2-kp/test/images'
            train_csv_path = 'runs/detect/' + model_name + '_train/results.csv'
            labels_dir = test_dir + '/labels_A'

        # 初始化Value.csv
        init_value_csv(value_csv_path)

        # 1. 执行推理并获取结果路径
        PredictWithInference(
            model_path=model_path,
            value_csv_path=value_csv_path,
            test_images_dir=test_images_dir,
            test_name=test_name,
            labels_dir=labels_dir
        )

        # 2. 处理置信度数据
        Confidence(
            folder_path=labels_dir,
            output_file=os.path.join(os.path.dirname(labels_dir), 'ReadAll.xlsx'),
            value_csv_path=value_csv_path
        )

        # 3. 计算PRF1指标（如果训练日志存在）
        if os.path.exists(train_csv_path):
            PRF1(train_csv_path, value_csv_path)
        else:
            logger.warning(f"未找到训练日志: {train_csv_path}，跳过指标计算")

        # 4. 记录参数量
        Parameters(model_path, value_csv_path)

        logger.info("所有步骤已完成！结果保存在: {}".format(
            os.path.abspath(value_csv_path)))

    except Exception as e:
        logger.error(f"主程序运行失败: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()