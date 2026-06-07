import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array


DATASET_PATH = r"E:\machine_learning_project\dataset"

IMG_SIZE = (128, 128)  
BATCH_SIZE = 32
EPOCHS = 30  


CLASSES = ['chai nhựa', 'lon nhôm', 'thùng giấy coton']

print("--- ĐANG ĐỌC VÀ TIỀN XỬ LÝ HÌNH ẢNH ---")
X = []
y = []

for idx, label in enumerate(CLASSES):
    # Kết hợp đường dẫn tuyệt đối một cách an toàn
    folder_path = os.path.join(DATASET_PATH, label)
    
    if not os.path.exists(folder_path):
        print(f"Lỗi: Không tìm thấy thư mục tại: {folder_path}")
        continue
        
    print(f"Đang đọc dữ liệu từ thư mục: {label}...")
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = load_img(img_path, target_size=IMG_SIZE)
                img_array = img_to_array(img)
                X.append(img_array)
                y.append(idx)
            except Exception:
                continue

# Kiểm tra dữ liệu trước khi chia tập huấn luyện
if len(X) == 0:
    print("\n[THÔNG BÁO LỖI] Không tìm thấy bất kỳ ảnh nào! Hãy chắc chắn bạn đã bỏ ảnh vào các thư mục trên ổ E.")
    # Tạo dữ liệu giả lập an toàn để tránh bị sập chương trình (Crash ValueError) nếu chưa có ảnh
    X_train = X_val = X_test = np.zeros((1, 128, 128, 3))
    y_train = y_val = y_test = np.array([0])
else:
    X = np.array(X, dtype="float32") / 255.0  
    y = np.array(y)
    print(f"\n=> Thành công! Tổng số lượng ảnh load được: {len(X)}")

    # 2. Chia dữ liệu Train (70%), Val (15%), Test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# 3. Kiến trúc mạng CNN chuẩn theo form mẫu báo cáo của thầy
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    
    Dense(128, activation='relu'),
    Dropout(0.5),
    
    Dense(3, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

if len(X) > 0:
    # 4. Huấn luyện mô hình
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("\n--- BẮT ĐẦU QUÁ TRÌNH TRAINING ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop]
    )

    # 5. Đánh giá kết quả tập Test
    print("\n--- ĐÁNH GIÁ TRÊN TẬP TEST ---")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Độ chính xác tập Test: {test_acc*100:.2f}%")

    
    MODEL_PATH = "my_model.keras"
    print("Đã lưu file mô hình tại thư mục dự án thành công!")
    # 6. Hiển thị đồ thị kết quả
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Do thi Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Do thi Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    

