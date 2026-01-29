# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io
# #ho tro bac si


# # Định nghĩa các lớp
# class_names = ["Calculus", "Data caries", "Gingivitis", "Mouth Ulcer", "Tooth Discoloration", "hypodontia"]

# # Tải mô hình
# @st.cache_resource
# def load_model():
#     try:
#         model = tf.keras.models.load_model("oral_disease_model_mobilenetv2_tamthoi.h5")
#         st.write("✅ Mô hình đã được tải thành công!")
#         return model
#     except Exception as e:
#         st.error(f"❌ Lỗi khi tải mô hình: {str(e)}")
#         return None

# model = load_model()
# if model is None:
#     st.stop()  # Dừng ứng dụng nếu không tải được mô hình

# # Hàm xử lý ảnh
# def preprocess_image(image):
#     try:
#         image = image.resize((224, 224))  # Resize ảnh về kích thước 224x224
#         image_array = np.array(image) / 255.0  # Chuẩn hóa giá trị pixel về [0, 1]
#         if image_array.shape[-1] == 4:  # Nếu ảnh có kênh alpha (RGBA), chuyển về RGB
#             image_array = image_array[..., :3]
#         image_array = np.expand_dims(image_array, axis=0)  # Thêm batch dimension
#         return image_array
#     except Exception as e:
#         st.error(f"❌ Lỗi khi xử lý ảnh: {str(e)}")
#         return None

# # Tiêu đề ứng dụng
# st.title("Dự đoán bệnh răng miệng")
# st.write("Tải lên một hình ảnh răng miệng để dự đoán bệnh. Mô hình hiện tại đạt độ chính xác **91%** trên tập kiểm tra.")

# # Tải ảnh từ người dùng
# uploaded_file = st.file_uploader("Chọn một hình ảnh...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     try:
#         # Đọc và hiển thị ảnh
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Hình ảnh đã tải lên", use_column_width=True)

#         # Xử lý ảnh và dự đoán
#         image_array = preprocess_image(image)
#         if image_array is None:
#             st.stop()

#         prediction = model.predict(image_array)
#         predicted_class = np.argmax(prediction, axis=1)[0]
#         confidence = prediction[0][predicted_class] * 100

#         # Hiển thị kết quả
#         st.write(f"**Dự đoán**: {class_names[predicted_class]}")
#         st.write(f"**Độ tin cậy**: {confidence:.2f}%")

#         # Hiển thị xác suất cho tất cả các lớp
#         st.write("**Xác suất chi tiết cho từng lớp**:")
#         for i, class_name in enumerate(class_names):
#             st.write(f"{class_name}: {prediction[0][i] * 100:.2f}%")

#         # Cảnh báo về nhầm lẫn
#         if class_names[predicted_class] in ["Calculus", "Gingivitis"]:
#             st.warning("Lưu ý: Mô hình có thể nhầm lẫn giữa Calculus và Gingivitis do đặc trưng hình ảnh tương tự. Vui lòng tham khảo ý kiến bác sĩ để xác nhận.")
#     except Exception as e:
#         st.error(f"❌ Lỗi khi dự đoán: {str(e)}")


##############################################

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import pandas as pd

# CSS tùy chỉnh để làm đẹp giao diện
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .title {
        color: #007bff;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        color: #333;
        font-size: 18px;
        text-align: center;
        margin-bottom: 20px;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ffeeba;
    }
    .success-box {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    </style>
""", unsafe_allow_html=True)

# Định nghĩa các lớp (bệnh lý bằng tiếng Việt)
class_names = ["Cao răng", "Sâu răng", "Viêm nướu", "Loét miệng", "Răng đổi màu", "Thiếu răng"]

# Tải mô hình
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("oral_disease_model_resnet_final.h5")
        st.markdown('<div class="success-box">✅ Mô hình đã được tải thành công!</div>', unsafe_allow_html=True)
        return model
    except Exception as e:
        st.error(f"❌ Lỗi khi tải mô hình: {str(e)}")
        return None

model = load_model()
if model is None:
    st.stop()

# Hàm xử lý ảnh
def preprocess_image(image):
    try:
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        if image_array.shape[-1] == 4:
            image_array = image_array[..., :3]
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý ảnh: {str(e)}")
        return None

# Sidebar với thông tin và hướng dẫn
with st.sidebar:
    st.image("https://idodesign.vn/wp-content/uploads/2023/07/mau-thiet-ke-logo-nha-khoa-dep-giup-xay-uy-tin-dung-niem-tin-1.jpeg", caption="Phòng Khám Nha Khoa AI", use_container_width=True)
    st.markdown("### Hướng dẫn sử dụng")
    st.markdown("""
    1. Tải lên ảnh răng miệng (jpg, jpeg, png).
    2. Xem kết quả chẩn đoán và độ tin cậy.
    3. Kiểm tra xác suất chi tiết và biểu đồ.
    4. Tham khảo ý kiến bác sĩ để xác nhận.
    """)
    st.markdown("### Thông tin mô hình")
    st.markdown("""
    - **Mô hình**: MobileNetV2
    - **Độ chính xác**: 91% (tập kiểm tra)
    - **Lớp**: 6 bệnh lý răng miệng
    - **Đầu vào**: Ảnh 224x224 RGB
    """)

# Tiêu đề ứng dụng
st.markdown('<div class="title">Hỗ trợ Chẩn đoán Bệnh Răng Miệng qua Ảnh</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ứng dụng AI hỗ trợ bác sĩ nha khoa phân tích hình ảnh nhanh chóng và chính xác</div>', unsafe_allow_html=True)

# Bố cục chính
with st.container():
    st.markdown("#### Tải lên hình ảnh")
    uploaded_file = st.file_uploader("Chọn ảnh răng miệng (jpg, jpeg, png)...", type=["jpg", "jpeg", "png"], help="Hỗ trợ các định dạng ảnh phổ biến")

    if uploaded_file is not None:
        try:
            # Hiển thị ảnh
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(image, caption="Hình ảnh đã tải lên", use_container_width=True)
            
            # Xử lý và dự đoán
            with st.spinner("Đang phân tích hình ảnh..."):
                image_array = preprocess_image(image)
                if image_array is None:
                    st.stop()
                
                prediction = model.predict(image_array)
                predicted_class = np.argmax(prediction, axis=1)[0]
                confidence = prediction[0][predicted_class] * 100

            # Hiển thị kết quả chính
            st.markdown("#### Kết quả chẩn đoán")
            st.markdown(f"**Bệnh lý**: {class_names[predicted_class]}")
            st.markdown(f"**Độ tin cậy**: {confidence:.2f}%")

            # Cảnh báo nhầm lẫn
            if class_names[predicted_class] in ["Cao răng", "Viêm nướu"]:
                st.markdown('<div class="warning-box">⚠️ Lưu ý: Có thể nhầm lẫn giữa Cao răng và Viêm nướu do đặc trưng hình ảnh tương tự. Vui lòng tham khảo ý kiến bác sĩ.</div>', unsafe_allow_html=True)

            # Biểu đồ xác suất
            st.markdown("#### Xác suất chi tiết")
            prob_df = pd.DataFrame({
                "Bệnh lý": class_names,
                "Xác suất (%)": [pred * 100 for pred in prediction[0]]
            })
            plt.figure(figsize=(8, 4))
            plt.bar(prob_df["Bệnh lý"], prob_df["Xác suất (%)"], color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6'])
            plt.title("Xác suất chẩn đoán cho từng bệnh lý")
            plt.xlabel("Bệnh lý")
            plt.ylabel("Xác suất (%)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)

            # Danh sách xác suất chi tiết
            with col2:
                st.markdown("**Danh sách xác suất**:")
                for i, class_name in enumerate(class_names):
                    st.markdown(f"- {class_name}: {prediction[0][i] * 100:.2f}%")
        
        except Exception as e:
            st.error(f"❌ Lỗi khi dự đoán: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**Phòng Khám Nha Khoa AI** | Phát triển bởi xAI | Dành cho bác sĩ nha khoa")
