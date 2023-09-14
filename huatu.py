import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('/home/ubuntu/tzy/yolov5s/runs/train/yolov5s_voc_baseline/results.csv')

data.columns = data.columns.str.strip()

# 提取数据列
box_loss = data['train/box_loss']
cls_loss = data['train/cls_loss']
obj_loss = data['train/obj_loss']

# 创建曲线图
plt.figure(figsize=(10, 6))
plt.plot(box_loss, label='Box Loss')
plt.plot(cls_loss, label='Cls Loss')
plt.plot(obj_loss, label='Obj Loss')

# 添加标注
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 显示图表
# plt.show()
plt.savefig('./1.jpg')
