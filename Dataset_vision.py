import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# 设置字体，解决中文乱码问题
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'  # 中文支持
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置 seaborn 风格
sns.set(style='whitegrid', palette='muted', font='Microsoft YaHei')

# 设置路径
label_dir = "datasets/ship-detect2/test/labels"
image_size = 1024  # 图像尺寸

# 容器
class_counts = {}
widths, heights, areas = [], [], []

# 遍历标签
for label_file in os.listdir(label_dir):
    if not label_file.endswith(".txt"):
        continue
    label_path = os.path.join(label_dir, label_file)
    if os.path.getsize(label_path) == 0:
        continue
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = map(float, parts)
            cls_id = int(cls)
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
            box_w = w * image_size
            box_h = h * image_size
            widths.append(box_w)
            heights.append(box_h)
            areas.append(box_w * box_h)

# 类别分布图
plt.figure(figsize=(7, 5))
plt.bar(class_counts.keys(), class_counts.values(), color=sns.color_palette("pastel"))
plt.xticks(list(class_counts.keys()), fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("类别 ID", fontsize=14)
plt.ylabel("目标数量", fontsize=14)
plt.title("目标类别分布", fontsize=16)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# 宽高散点图
plt.figure(figsize=(7, 6))
plt.scatter(widths, heights, alpha=0.6, s=12, color=sns.color_palette("dark")[0])
plt.xlabel("宽度 (像素)", fontsize=14)
plt.ylabel("高度 (像素)", fontsize=14)
plt.title("目标框宽高分布", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 面积分布图
plt.figure(figsize=(7, 6))
scaled_areas = [a / 10 for a in areas]  # 面积缩放
plt.hist(scaled_areas, bins=60, color=sns.color_palette("deep")[2], log=True, edgecolor='black')
plt.xlabel("面积 (像素²)", fontsize=14)

plt.ylabel("数量 (log)", fontsize=14)
plt.title("目标面积分布", fontsize=16)
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
