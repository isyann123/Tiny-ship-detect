import matplotlib.pyplot as plt
import numpy as np

# 示例数据，这里随机生成一些F1分数和confidence值
np.random.seed(42)
confidence = np.linspace(0, 1, 100)
F1_scores = np.random.rand(100)

# 绘制F1曲线
plt.figure(figsize=(10, 6))
plt.plot(confidence, F1_scores, label='F1 Curve', color='blue')

# 设置图形属性
plt.title('F1 Curve')
plt.xlabel('Confidence')
plt.ylabel('F1 Score')
plt.grid(True)
plt.legend()

# 显示图形
plt.show()
