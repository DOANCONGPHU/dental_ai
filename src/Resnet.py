# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator
# from keras.applications import ResNet50
# from keras.models import Model
# from keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from sklearn.utils import class_weight
# from sklearn.metrics import confusion_matrix, classification_report

# # === 1. Cài đặt tham số ===
# train_dir = "D:/DO_AN/dataset/train"
# val_dir = "D:/DO_AN/dataset/val"
# test_dir = "D:/DO_AN/dataset/test"

# img_size = (224, 224)
# batch_size = 8
# num_classes = 6
# epochs = 10

# # === 2. Tiến hành augment dữ liệu ===
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

# train_generator = train_datagen.flow_from_directory(
#     train_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
# )
# val_generator = val_datagen.flow_from_directory(
#     val_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
# )
# test_generator = test_datagen.flow_from_directory(
#     test_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=False
# )

# # === 3. Tính class weights ===
# class_weights = class_weight.compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(train_generator.classes),
#     y=train_generator.classes
# )
# class_weights_dict = dict(enumerate(class_weights))

# # === 4. Tạo mô hình ResNet50 ===
# base_model = ResNet50(input_shape=img_size + (3,), include_top=False, weights='imagenet')
# base_model.trainable = True

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.3)(x)
# predictions = Dense(num_classes, activation='softmax')(x)
# model = Model(inputs=base_model.input, outputs=predictions)

# model.compile(
#     optimizer=Adam(learning_rate=1e-4),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # === 5. Callbacks ===
# earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
# checkpoint = ModelCheckpoint("best_resnet_model.h5", monitor='val_loss', save_best_only=True, verbose=1)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)

# # === 6. Huấn luyện mô hình ===
# history = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=epochs,
#     class_weight=class_weights_dict,
#     callbacks=[earlystop, checkpoint, reduce_lr]
# )

# # === 7. Vẽ biểu đồ loss/accuracy ===
# plt.figure(figsize=(12, 5))

# # Accuracy
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Val Accuracy')
# plt.title('Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

# # Loss
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Val Loss')
# plt.title('Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.tight_layout()
# plt.savefig("training_plot_resnet.png")
# plt.show()

# # === 8. Đánh giá trên test ===
# model = tf.keras.models.load_model("best_resnet_model.h5")
# y_pred = model.predict(test_generator)
# y_pred_classes = np.argmax(y_pred, axis=1)
# y_true = test_generator.classes

# print("\n\U0001f3af Ma trận nhầm lẫn:")
# conf_matrix = confusion_matrix(y_true, y_pred_classes)
# print(conf_matrix)

# print("\n\U0001f3af Báo cáo phân loại:")
# print(classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))

# # === 9. Vẽ ma trận nhầm lẫn ===
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.tight_layout()
# plt.savefig("confusion_matrix_resnet.png")
# plt.show()

# # === 10. Lưu mô hình cuối ===
# model.save("oral_disease_model_resnet_final.h5")
# print("\u2705 Model saved as oral_disease_model_resnet_final.h5")



import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
import pickle
from collections import Counter
import zipfile

# === 1. Cài đặt tham số ===
model_dir = "D:/DO_AN/models"
train_dir = "D:/DO_AN/dataset/train"
val_dir = "D:/DO_AN/dataset/val"
test_dir = "D:/DO_AN/dataset/test"

# Kiểm tra thư mục
os.makedirs(model_dir, exist_ok=True)
for directory in [train_dir, val_dir, test_dir]:
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist")

img_size = (224, 224)
batch_size = 8
num_classes = 6
epochs = 10

# === 2. Kiểm tra phân bố lớp ===
train_generator_temp = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
)
print("Phân bố lớp trong tập huấn luyện:", Counter(train_generator_temp.classes))

# === 3. Tiến hành augment dữ liệu ===
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
    train_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=False
)

# === 4. Tính class weights ===
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))

# === 5. Tạo mô hình ResNet50 ===
base_model = ResNet50(input_shape=img_size + (3,), include_top=False, weights='imagenet')

# Giai đoạn 1: Đóng băng các tầng của ResNet50
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === 6. Callbacks ===
earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint(
    os.path.join(model_dir, "best_resnet_model.h5"),
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
checkpoint_periodic = ModelCheckpoint(
    os.path.join(model_dir, "resnet_epoch_{epoch:02d}.h5"),
    monitor='val_loss',
    save_best_only=False,
    verbose=1,
    period=5
)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)

# === 7. Huấn luyện giai đoạn 1 ===
history_phase1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    class_weight=class_weights_dict,
    callbacks=[earlystop, checkpoint, checkpoint_periodic, reduce_lr]
)

# === 8. Giai đoạn 2: Mở khóa toàn bộ mô hình để fine-tune ===
base_model.trainable = True
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === 9. Huấn luyện giai đoạn 2 ===
history_phase2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    class_weight=class_weights_dict,
    callbacks=[earlystop, checkpoint, checkpoint_periodic, reduce_lr]
)

# === 10. Lưu lịch sử huấn luyện ===
with open(os.path.join(model_dir, 'resnet_history.pkl'), 'wb') as f:
    pickle.dump({'phase1': history_phase1.history, 'phase2': history_phase2.history}, f)

# === 11. Vẽ biểu đồ loss/accuracy ===
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
plt.savefig(os.path.join(model_dir, "training_plot_resnet.png"))
plt.show()

# === 12. Đánh giá trên tập kiểm tra ===
model = tf.keras.models.load_model(os.path.join(model_dir, "best_resnet_model.h5"))
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

print("\n\U0001f3af Ma trận nhầm lẫn:")
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print(conf_matrix)

print("\n\U0001f3af Báo cáo phân loại:")
print(classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))

# === 13. Vẽ ma trận nhầm lẫn ===
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "confusion_matrix_resnet.png"))
plt.show()

# === 14. Lưu mô hình cuối ===
model.save(os.path.join(model_dir, "oral_disease_model_resnet_final.h5"))
print(f"\u2705 Model saved as {os.path.join(model_dir, 'oral_disease_model_resnet_final.h5')}")