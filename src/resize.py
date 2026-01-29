import os
import shutil
import random
import cv2

# Đường dẫn thư mục dữ liệu gốc
source_dir = "data"
# Thư mục sau khi chia và resize
output_dir = "dataset"

# Tên các lớp
classes = ["Calculus", "Data caries", "Gingivitis", "Tooth Discoloration", "Mouth Ulcer", "hypodontia"]

# Tỷ lệ chia
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Kích thước ảnh cho VGG16
img_size = (224, 224)

# Tạo thư mục chia dữ liệu
for split in ["train", "val", "test"]:
    for class_name in classes:
        os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

# Hàm đọc, resize và lưu ảnh
def process_and_save_image(src_path, dst_path):
    img = cv2.imread(src_path)
    if img is not None:
        # Resize
        img = cv2.resize(img, img_size)
        # Chuyển về RGB nếu cần
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        # Lưu lại
        cv2.imwrite(dst_path, img)

# Bắt đầu chia và xử lý ảnh
for class_name in classes:
    class_path = os.path.join(source_dir, class_name)
    images = os.listdir(class_path)
    images = [img for img in images if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, split_images in splits.items():
        for img in split_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(output_dir, split, class_name, img)
            process_and_save_image(src_path, dst_path)

print("✅ Đã chia dữ liệu và resize ảnh về 224x224 cho VGG16.")
