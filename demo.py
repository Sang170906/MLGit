import os
import cv2
import numpy as np
import tensorflow as tf
import sys  


# 1. ĐỊNH NGHĨA ĐƯỜNG DẪN VÀ THAM SỐ

MODEL_PATH = r"E:\machine_learning_project\my_model.keras"

IMG_SIZE = (128, 128)
CLASSES = ['chai nhựa', 'Lon nhôm', 'thùng giấy coton']

print("--- ĐANG TẢI MÔ HÌNH ĐÃ HUẤN LUYỆN ---")

# Kiểm tra nếu không thấy file mô hình
if not os.path.exists(MODEL_PATH):
    print(f"\n[LỖI CẦN SỬA]: Không tìm thấy file mô hình tại đường dẫn: {MODEL_PATH}")
    print("-" * 60)
    print("Dưới đây là các file đang có trong thư mục 'machine_learning_project' của bạn:")
    
    # Tự động quét và hiển thị các file trong thư mục để bạn biết tên file mô hình của mình
    thu_muc = r"E:\machine_learning_project"
    if os.path.exists(thu_muc):
        cac_file = os.listdir(thu_muc)
        co_file_mo_hinh = False
        for f in cac_file:
            if f.endswith('.h5') or f.endswith('.keras') or f.endswith('.pkl') or f.endswith('.h5model'):
                print(f" -> Tìm thấy file mô hình khả nghi: {f}  <-- HÃY THAY TÊN NÀY VÀO DÒNG MODEL_PATH")
                co_file_mo_hinh = True
        if not co_file_mo_hinh:
            print(" -> Cảnh báo: Không tìm thấy bất kỳ file .h5 hay .keras nào trong thư mục này!")
            print(" -> Bạn có chắc chắn là file 'train.py' trước đó đã chạy xong và lưu file mô hình chưa?")
    else:
        print(" -> Thư mục E:\machine_learning_project không tồn tại!")
        
    print("-" * 60)
    sys.exit()

# Nếu tìm thấy file thì tiến hành load mô hình
model = tf.keras.models.load_model(MODEL_PATH)
print("Tải mô hình thành công!")

# 2. HÀM MÔ PHỎNG LỆNH ĐIỀU KHIỂN ROBOT / BĂNG CHUYỀN
def dieu_khien_robot(nhan_vat_the):
    print(f"[ROBOT] Phát hiện: {nhan_vat_the}")
    if nhan_vat_the == 'chai nhựa':
        print("-> Lệnh: Kích hoạt tay kẹp số 1 - Đẩy chai nhựa vào giỏ A.")
    elif nhan_vat_the == 'Lon nhôm':
        print("-> Lệnh: Kích hoạt tay kẹp số 2 - Đẩy lon nhôm vào giỏ B.")
    elif nhan_vat_the == 'thùng giấy coton':
        print("-> Lệnh: Băng chuyền chạy thẳng - Đưa thùng giấy vào khu vực C.")
    print("-" * 40)

# 3. VÒNG LẶP MÔ PHỎNG NHẬN DIỆN QUA CAMERA
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở được Camera! Vui lòng kiểm tra lại kết nối.")
    sys.exit()

print("Bắt đầu mô phỏng hệ thống nhận diện. Nhấn 'q' trên bàn phím để thoát.")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Lỗi: Không thể nhận dữ liệu từ camera.")
        break
        
    img_processed = cv2.resize(frame, IMG_SIZE)
    img_processed = img_processed / 255.0  
    img_processed = np.expand_dims(img_processed, axis=0) 
    
    predictions = model.predict(img_processed, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100  
    
    label_predicted = CLASSES[class_idx]
    
    frame_count += 1
    if frame_count % 30 == 0:
        dieu_khien_robot(label_predicted)
        frame_count = 0
        
    text_display = f"Vat the: {label_predicted} ({confidence:.2f}%)"
    cv2.putText(frame, text_display, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("He thong phan loai tren bang chuyen - Robot AI", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Đã tắt hệ thống mô phỏng.")