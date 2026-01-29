*** Phân loại bệnh răng miệng từ ảnh sử dụng CNN & Transfer Learning
Giới thiệu

Dự án này xây dựng một hệ thống phân loại bệnh răng miệng từ ảnh dựa trên mô hình học sâu (Deep Learning), nhằm hỗ trợ quá trình chẩn đoán ban đầu.
Mô hình sử dụng Transfer Learning với MobileNetV2, giúp đạt hiệu quả tốt ngay cả khi tập dữ liệu không lớn.

1 Mô tả tập dữ liệu (Dataset)

- Các lớp bệnh

- Tập dữ liệu gồm 6 lớp bệnh răng miệng:
    Calculus (Cao răng)
    Caries (Sâu răng)
    Gingivitis (Viêm nướu)
    Tooth Discoloration (Nhiễm màu răng)
    Ulcers (Loét miệng)
    Hypodontia (Thiếu răng)

- Đặc điểm dữ liệu
    Loại dữ liệu: ảnh RGB
    Kích thước ảnh: chuẩn hóa về 224 × 224
    Số lượng dữ liệu: ở mức trung bình / hạn chế
    Phân bố dữ liệu giữa các lớp chưa cân bằng
- Tiền xử lý dữ liệu
    Resize ảnh
    Chuẩn hóa giá trị pixel
    Mã hóa nhãn (label encoding)
    Chia tập huấn luyện và tập kiểm tra

- Tăng cường dữ liệu (Data Augmentation)
    Áp dụng augmentation theo từng lớp nhằm giảm hiện tượng nhầm lẫn giữa các bệnh có đặc điểm hình ảnh tương đồng:
    Xoay ảnh (rotation)
    Phóng to/thu nhỏ (zoom)
    Lật ngang (horizontal flip)
    Điều chỉnh độ sáng

2 Mô hình sử dụng

- Kiến trúc chính
    MobileNetV2 (pre-trained trên ImageNet)
    Đóng vai trò là bộ trích xuất đặc trưng (feature extractor)
- Phần phân loại (Classifier)
    Global Average Pooling
    Các lớp Fully Connected
    Dropout để giảm overfitting
    Lớp Softmax cho 6 lớp đầu ra
- Chiến lược huấn luyện
    Sử dụng Transfer Learning
    Fine-tune một số layer phía trên của MobileNetV2
    Hàm mất mát: Categorical Cross-Entropy
    Bộ tối ưu: Adam

3 Tham số huấn luyện (Hyperparameters)

    Tham số	Mô tả
    Learning rate	Điều chỉnh nhỏ khi fine-tune
    Batch size	Thử nghiệm để đảm bảo ổn định
    Epoch	Huấn luyện đến khi hội tụ
    Dropout	Giảm hiện tượng overfitting
    
4 Kết quả & Đánh giá


- Chỉ số đánh giá
    Accuracy > 80%
    Confusion Matrix :
  <img width="800" height="600" alt="confusion_matrix_mobilenetv2" src="https://github.com/user-attachments/assets/3ae092c7-7297-4080-92ef-191be91605bc" />
  <img width="800" height="600" alt="confusion_matrix_resnet" src="https://github.com/user-attachments/assets/1494e9f5-ec59-4af6-a279-cccd9fd641fa" />
      
    Biểu đồ loss/accuracy theo epoch:
  <img width="1200" height="500" alt="training_plot_mobilenetv2" src="https://github.com/user-attachments/assets/b70d0ae8-60b9-4f0d-acbd-0d6942174c01" />
  <img width="1200" height="500" alt="training_plot_resnet" src="https://github.com/user-attachments/assets/99f5f866-8558-4c5a-a5cb-50ef9fa1dc53" />

- Nhận xét
    Transfer Learning giúp cải thiện độ chính xác rõ rệt so với huấn luyện từ đầu
    Data augmentation giúp mô hình tổng quát hóa tốt hơn
    Một số lớp bệnh có đặc điểm hình ảnh gần nhau vẫn còn nhầm lẫn
    (Có thể cập nhật độ chính xác cụ thể sau khi hoàn tất thử nghiệm)

4 Công nghệ sử dụng

    Python
    TensorFlow / Keras
    NumPy, Pandas
    Matplotlib
    Google Colab
    
5 Hướng phát triển

    Mở rộng và cân bằng tập dữ liệu
    Thử nghiệm các mô hình khác như EfficientNet, ResNet
    Áp dụng các kỹ thuật augmentation nâng cao
    Triển khai mô hình dưới dạng web hoặc mobile app
