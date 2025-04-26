import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Đường dẫn tới thư mục chứa dữ liệu
DATA_DIR = "dataset"  # Sửa lại nếu khác
IMG_SIZE = (224, 224)  # Kích thước ảnh chuẩn của VGG16
BATCH_SIZE = 32
EPOCHS = 10

# Tạo datagen với augment cho train, không augment cho val
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

# Tạo generator cho train và validation
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Tải mô hình VGG16 pretrained, bỏ phần fully connected phía trên
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Đóng băng các lớp pretrained
for layer in base_model.layers:
    layer.trainable = False

# Thêm các lớp fully connected mới
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(6, activation='softmax')(x)  # 6 lớp

# Kết hợp lại thành mô hình cuối cùng
model = Model(inputs=base_model.input, outputs=predictions)

# Compile mô hình
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# Lưu mô hình
model.save("vgg16_teeth_disease_classifier.h5")
