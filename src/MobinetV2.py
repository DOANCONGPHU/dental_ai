# import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator
# from keras.applications import MobileNetV2
# from keras.models import Model
# from keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# import numpy as np
# from sklearn.utils import class_weight

# # Đường dẫn dữ liệu
# train_dir = "../dataset/train"
# val_dir = "../dataset/val"
# test_dir = "../dataset/test"

# # Các tham số
# img_size = (224, 224)
# batch_size = 6
# num_classes = 6
# epochs = 25

# # Tăng cường dữ liệu (augment nhẹ hơn)
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     brightness_range=[0.8, 1.2],
# )

# val_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# # Load dữ liệu
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode="categorical"
# )

# val_generator = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode="categorical"
# )

# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode="categorical",
#     shuffle=False
# )

# # Tính class_weight
# class_weights = class_weight.compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(train_generator.classes),
#     y=train_generator.classes
# )
# class_weights_dict = dict(enumerate(class_weights))
# print("Class weights:", class_weights_dict)

# # Load MobileNetV2 và cho trainable tất cả layers
# base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')
# base_model.trainable = True  # <<< Bật trainable ngay từ đầu

# # Xây mô hình
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.3)(x)
# predictions = Dense(num_classes, activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=predictions)

# # Compile
# model.compile(
#     optimizer=Adam(learning_rate=0.0001),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # Callbacks
# earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
# checkpoint = ModelCheckpoint("best_mobilenetv2_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)

# # Train
# history = model.fit(
#     train_generator,
#     epochs=epochs,
#     validation_data=val_generator,
#     class_weight=class_weights_dict,
#     callbacks=[earlystop, checkpoint]
# )

# # Load lại model tốt nhất
# model = tf.keras.models.load_model("best_mobilenetv2_model.h5")

# # Đánh giá trên test
# import sklearn.metrics as metrics

# y_pred = model.predict(test_generator)
# y_pred_classes = np.argmax(y_pred, axis=1)
# y_true = test_generator.classes

# print("\n🎯 Ma trận nhầm lẫn:")
# print(metrics.confusion_matrix(y_true, y_pred_classes))

# print("\n🎯 Báo cáo phân loại:")
# print(metrics.classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))

# # Lưu mô hình cuối cùng
# model.save("oral_disease_model_mobilenetv2_final.h5")
# print("✅ 90 Model saved as oral_disease_model_mobilenetv2_final.h5")


# import os
# import cv2
# import numpy as np
# import random
# from tqdm import tqdm
# from sklearn.utils import class_weight
# from keras.preprocessing.image import ImageDataGenerator
# from keras.applications import MobileNetV2
# from keras.models import Model
# from keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import seaborn as sns
# import sklearn.metrics as metrics

# # ====== CẤU HÌNH ======
# original_train_dir = r"D:\DO_AN\dataset\train"
# val_dir = r"D:\DO_AN\dataset\val"
# test_dir = r"D:\DO_AN\dataset\test"
# balanced_train_dir = r"D:\DO_AN\dataset_balanced\train"

# target_count = 1964
# img_size = (224, 224)
# batch_size = 6
# num_classes = 6
# epochs = 25

# # ====== TĂNG CƯỜNG DỮ LIỆU CHO LỚP THIẾU ======
# augmenter = ImageDataGenerator(
#     rotation_range=20,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     brightness_range=[0.8, 1.2],
#     fill_mode='nearest'
# )

# def balance_dataset():
#     print("🚀 Bắt đầu tăng cường ảnh để cân bằng dữ liệu...")
#     os.makedirs(balanced_train_dir, exist_ok=True)

#     for class_name in os.listdir(original_train_dir):
#         input_path = os.path.join(original_train_dir, class_name)
#         output_path = os.path.join(balanced_train_dir, class_name)
#         os.makedirs(output_path, exist_ok=True)

#         images = [
#             f for f in os.listdir(input_path)
#             if f.lower().endswith(('.jpg', '.jpeg', '.png'))
#         ]

#         current_count = len(images)
#         needed = target_count - current_count

#         # Copy ảnh gốc trước
#         for f in images:
#             src = os.path.join(input_path, f)
#             dst = os.path.join(output_path, f)
#             if not os.path.exists(dst):
#                 os.system(f'copy "{src}" "{dst}" >nul')

#         if needed <= 0:
#             continue

#         print(f"🔧 Lớp '{class_name}': {current_count} ảnh → cần tăng thêm {needed}")
#         i = 0
#         pbar = tqdm(total=needed, desc=f"Tăng ảnh {class_name}")
#         while i < needed:
#             img_name = random.choice(images)
#             img_path = os.path.join(input_path, img_name)
#             img = cv2.imread(img_path)
#             if img is None:
#                 continue
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = cv2.resize(img, img_size)
#             img = np.expand_dims(img / 255.0, axis=0)

#             aug_iter = augmenter.flow(img, batch_size=1)
#             aug_img = next(aug_iter)[0]
#             aug_img_bgr = cv2.cvtColor((aug_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

#             save_path = os.path.join(output_path, f"aug_{i}.jpg")
#             cv2.imwrite(save_path, aug_img_bgr)
#             i += 1
#             pbar.update(1)
#         pbar.close()
#     print("✅ Đã hoàn tất cân bằng dữ liệu!\n")

# # ====== VẼ BIỂU ĐỒ TRAINING ======
# def plot_training_history(history):
#     acc = history.history['accuracy']
#     val_acc = history.history['val_accuracy']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     epochs_range = range(len(acc))

#     plt.figure(figsize=(12, 5))

#     plt.subplot(1, 2, 1)
#     plt.plot(epochs_range, acc, label='Train Accuracy')
#     plt.plot(epochs_range, val_acc, label='Val Accuracy')
#     plt.legend(loc='lower right')
#     plt.title('Training & Validation Accuracy')

#     plt.subplot(1, 2, 2)
#     plt.plot(epochs_range, loss, label='Train Loss')
#     plt.plot(epochs_range, val_loss, label='Val Loss')
#     plt.legend(loc='upper right')
#     plt.title('Training & Validation Loss')

#     plt.tight_layout()
#     plt.savefig("training_history.png")
#     plt.show()

# # ====== VẼ MA TRẬN NHẦM LẪN ======
# def plot_confusion_matrix(y_true, y_pred, class_names):
#     cm = metrics.confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=class_names,
#                 yticklabels=class_names)
#     plt.ylabel('Actual')
#     plt.xlabel('Predicted')
#     plt.title('Confusion Matrix')
#     plt.tight_layout()
#     plt.savefig("confusion_matrix.png")
#     plt.show()

# # ====== TĂNG CƯỜNG TRƯỚC ======
# balance_dataset()

# # ====== LOAD DỮ LIỆU SAU TĂNG CƯỜNG ======
# train_datagen = ImageDataGenerator(rescale=1./255)
# val_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     balanced_train_dir,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode="categorical"
# )

# val_generator = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode="categorical"
# )

# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode="categorical",
#     shuffle=False
# )

# # ====== TÍNH TRỌNG SỐ LỚP ======
# class_weights = class_weight.compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(train_generator.classes),
#     y=train_generator.classes
# )
# class_weights_dict = dict(enumerate(class_weights))
# print("📊 Class weights:", class_weights_dict)

# # ====== MÔ HÌNH MOBILENETV2 ======
# base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')
# base_model.trainable = True

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.3)(x)
# predictions = Dense(num_classes, activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=predictions)

# model.compile(
#     optimizer=Adam(learning_rate=0.0001),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # ====== TRAINING ======
# earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
# checkpoint = ModelCheckpoint("best_mobilenetv2_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)

# history = model.fit(
#     train_generator,
#     epochs=epochs,
#     validation_data=val_generator,
#     class_weight=class_weights_dict,
#     callbacks=[earlystop, checkpoint]
# )


# # ====== ĐÁNH GIÁ MÔ HÌNH ======
# model = tf.keras.models.load_model("best_mobilenetv2_model.h5")

# y_pred = model.predict(test_generator)
# y_pred_classes = np.argmax(y_pred, axis=1)
# y_true = test_generator.classes
# class_names = list(test_generator.class_indices.keys())

# # ====== IN BÁO CÁO 1 LẦN ======
# print("\n🎯 Ma trận nhầm lẫn:")
# print(metrics.confusion_matrix(y_true, y_pred_classes))

# print("\n🎯 Báo cáo phân loại:")
# print(metrics.classification_report(y_true, y_pred_classes, target_names=class_names))

# # ====== VẼ BIỂU ĐỒ ======
# plot_training_history(history)
# plot_confusion_matrix(y_true, y_pred_classes, class_names)

# # ====== LƯU MÔ HÌNH ======
# model.save("oral_disease_model_mobilenetv2_final.h5")
# print("✅ Mô hình đã được lưu vào: oral_disease_model_mobilenetv2_final.h5")

# /////////////////////////////////////////////////////


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import pickle
import zipfile

# Đường dẫn dữ liệu

model_dir = r"D:\DO_AN\model"
train_dir = r"D:\DO_AN\dataset\train"
val_dir = r"D:\DO_AN\dataset\val"
test_dir = r"D:\DO_AN\dataset\test"
# Kiểm tra thư mục
os.makedirs(model_dir, exist_ok=True)
for directory in [train_dir, val_dir, test_dir]:
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist")

# Các tham số
img_size = (224, 224)
batch_size = 6
num_classes = 6
epochs = 25

# Kiểm tra phân bố lớp
train_generator_temp = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)
print("Phân bố lớp trong tập huấn luyện:", Counter(train_generator_temp.classes))

# Tăng cường dữ liệu
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# Tính class_weight
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))
print("Class weights:", class_weights_dict)

# Load MobileNetV2 và đóng băng các tầng ban đầu
base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')
base_model.trainable = False

# Xây mô hình
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint(
    os.path.join(model_dir, "best_mobilenetv2_model.h5"),
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
checkpoint_periodic = ModelCheckpoint(
    os.path.join(model_dir, "mobilenetv2_epoch_{epoch:02d}.h5"),
    monitor='val_loss',
    save_best_only=False,
    verbose=1,
    period=5
)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)

# Train giai đoạn 1
history_phase1 = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    class_weight=class_weights_dict,
    callbacks=[earlystop, checkpoint, checkpoint_periodic, reduce_lr]
)

# Giai đoạn 2: Mở khóa toàn bộ mô hình để fine-tune
base_model.trainable = True
model.compile(
    optimizer=Adam(learning_rate=15e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train giai đoạn 2
history_phase2 = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    class_weight=class_weights_dict,
    callbacks=[earlystop, checkpoint, checkpoint_periodic, reduce_lr]
)

# Lưu lịch sử huấn luyện
with open(os.path.join(model_dir, 'mobilenetv2_history.pkl'), 'wb') as f:
    pickle.dump({'phase1': history_phase1.history, 'phase2': history_phase2.history}, f)

# Vẽ biểu đồ loss/accuracy
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_phase2.history['accuracy'], label='Train Accuracy')
plt.plot(history_phase2.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history_phase2.history['loss'], label='Train Loss')
plt.plot(history_phase2.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(model_dir, "training_plot_mobilenetv2.png"))
plt.show()

# Đánh giá trên tập kiểm tra
model = tf.keras.models.load_model(os.path.join(model_dir, "best_mobilenetv2_model.h5"))
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

print("\n🎯 Ma trận nhầm lẫn:")
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print(conf_matrix)

print("\n🎯 Báo cáo phân loại:")
print(classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))

# Vẽ ma trận nhầm lẫn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "confusion_matrix_mobilenetv2.png"))
plt.show()

# Lưu mô hình cuối cùng
model.save(os.path.join(model_dir, "oral_disease_model_mobilenetv2_tamthoi.h5"))
print(f"✅ Model saved as {os.path.join(model_dir, 'oral_disease_model_mobilenetv2_final.h5')}")