import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_pr_curve_from_results(csv_path, output_dir=None, smooth_sigma=1):
    """
    从YOLOv8训练日志绘制PR曲线

    参数:
        csv_path: results.csv文件路径
        output_dir: 输出目录(默认为csv所在目录)
        smooth_sigma: PR曲线平滑系数(设为0禁用平滑)
    """
    try:
        # 1. 读取数据
        df = pd.read_csv(csv_path)

        # 2. 提取有效数据(去掉precision和recall为0的epoch)
        valid_data = df[(df['   metrics/precision(B)'] > 0) | (df['      metrics/recall(B)'] > 0)]
        if len(valid_data) < 5:
            print("警告: 有效数据点不足，可能模型尚未收敛")
            valid_data = df.iloc[-20:]  # 至少取最后20个epoch

        precision = valid_data['   metrics/precision(B)'].values
        recall = valid_data['      metrics/recall(B)'].values
        epochs = valid_data['                  epoch'].values

        # 3. 计算F1分数
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        best_idx = np.argmax(f1_scores)

        # 4. 平滑处理(可选)
        if smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter1d
            precision_sm = gaussian_filter1d(precision, sigma=smooth_sigma)
            recall_sm = gaussian_filter1d(recall, sigma=smooth_sigma)
        else:
            precision_sm, recall_sm = precision, recall

        # 5. 创建图形
        plt.figure(figsize=(10, 8))

        # 绘制PR曲线
        plt.plot(recall_sm, precision_sm, 'b-', linewidth=2, label='PR Curve')
        plt.scatter(recall[best_idx], precision[best_idx],
                    c='red', s=100, label=f'Best F1 ({f1_scores[best_idx]:.2f})')

        # 标记关键点
        for i, (r, p, e) in enumerate(zip(recall, precision, epochs)):
            if i % 10 == 0 or i == best_idx:  # 每10个epoch标记一次
                plt.annotate(f'e{e}', (r, p), textcoords="offset points",
                             xytext=(5, 5), ha='center', fontsize=8)

        # 图表装饰
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve (Epoch {epochs[0]}-{epochs[-1]})', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=10, loc='lower left')
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # 6. 保存图像
        if output_dir is None:
            output_dir = os.path.dirname(csv_path) or '.'
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, 'pr_curve.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"PR曲线已保存至: {output_path}")
        return output_path

    except Exception as e:
        print(f"绘图出错: {str(e)}")
        raise


# 使用示例
plot_pr_curve_from_results(
    csv_path='runs/detect/1_train/results.csv',
    output_dir='runs/detect/1_train/pr_curves',
    smooth_sigma=1  # 设为0可禁用平滑
)