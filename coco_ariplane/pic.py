import os
import random
import shutil

# 定义coco_airplane目录和子目录
coco_airplane_dir = "/home/ubuntu/yolov5-6.1/coco_ariplane"
train_images_dir = os.path.join(coco_airplane_dir, "images", "train2017")
test_images_dir = os.path.join(coco_airplane_dir, "images", "test2017")

# 创建test_images_dir目录
os.makedirs(test_images_dir, exist_ok=True)

# 获取train2017目录下所有的jpg文件列表
jpg_files = [f for f in os.listdir(train_images_dir) if f.endswith(".jpg")]

# 随机选取300张图片的文件名
random_selected_files = random.sample(jpg_files, 300)

# 复制选中的图片到test_images_dir目录
for filename in random_selected_files:
    train_image_path = os.path.join(train_images_dir, filename)
    test_image_path = os.path.join(test_images_dir, filename)

    # 复制图片文件
    shutil.copy(train_image_path, test_image_path)