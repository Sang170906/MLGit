Tệp Markdown (`README.md`) cho báo cáo dự án Machine Learning của bạn đã được khởi tạo thành công. 

Nội dung tệp đã được chuẩn hóa theo đúng cấu trúc Markdown chuẩn của một kho lưu trữ GitHub (Repository), tổng hợp đầy đủ thông tin từ báo cáo bao gồm: Thông tin sinh viên, mô tả bài toán, kiến trúc CNN chi tiết, thông số và kết quả huấn luyện (gồm bảng số liệu và cơ chế Early Stopping), nhận xét đánh giá, hướng dẫn cài đặt chạy mã nguồn mô phỏng cùng định hướng phát triển bài toán (Transfer Learning, Data Augmentation).

[file-tag: code-generated-file-0-1780844402401751210]

Dưới đây là nội dung chi tiết được ghi trong file để bạn có thể xem trước:

```markdown
# Hệ Thống Robot Tự Động Phân Loại Phế Liệu Sử Dụng Học Máy (CNN)

Dự án xây dựng hệ thống phân loại phế liệu tự động trên băng chuyền sử dụng mạng nơ-ron tích chập (CNN) nhằm thay thế các phương pháp phân loại thủ công, tăng năng suất và độ chính xác trong quản lý chất thải và tái chế.

---

## 📌 Thành Viên Thực Hiện
* **Họ và tên:** Cao Xuân Sang
* **Mã sinh viên:** 2421060188
* **Lớp:** DCCDROBOT69
* **Mã môn học:** 7090299
* **Giảng viên hướng dẫn:** Nguyễn Đức Khoát

---

## 🛠️ Tổng Quan Bài Toán & Kiến Trúc

### 1. Mô tả bài toán
Hệ thống nhận diện và phân loại phế liệu di chuyển trên băng chuyền thành **3 nhóm chính**:
1. `Chai nhựa`
2. `Lon nhôm`
3. `Thùng giấy carton`

* **Lý do sử dụng Học sâu (Deep Learning):** Do hình dạng, kích thước, góc nghiêng, nhãn mác của phế liệu rất đa dạng và điều kiện ánh sáng thực tế trên băng chuyền thay đổi liên tục. Các phương pháp xử lý ảnh truyền thống (ngưỡng màu, đường biên cố định) không đáp ứng được độ bền vững. Mạng CNN có khả năng tự động trích xuất đặc trưng hình học bất chấp sự dịch chuyển, xoay hoặc thay đổi độ sáng.
* **Yêu cầu hệ thống:**
  * **Đầu vào (Input):** Ảnh phế liệu từ tập dữ liệu sẵn có (chuẩn hóa về kích thước $128 \times 128$).
  * **Đầu ra (Output):** Phân loại chính xác 1 trong 3 nhóm phế liệu.
  * **Thời gian xử lý:** $< 0.5$ giây/ảnh (đảm bảo kịp tốc độ dịch chuyển của băng chuyền).

### 2. Kiến trúc mô hình (CNN Sequential)
Mô hình mạng CNN được xây dựng theo cấu trúc `Sequential` tuần tự gồm các tầng trích xuất đặc trưng và phân loại:

| Tầng (Layer) | Loại Layer | Cấu hình / Kích thước đầu ra |
| :--- | :--- | :--- |
| **Layer 1** | Conv2D | 32 filters, Kernel $3 \times 3$, Activation: `relu`, Input: `(128, 128, 3)` |
| **Layer 2** | MaxPooling2D | Pool size $2 \times 2$, Output: `(64, 64, 32)` |
| **Layer 3** | Conv2D | 64 filters, Kernel $3 \times 3$, Activation: `relu` |
| **Layer 4** | MaxPooling2D | Pool size $2 \times 2$, Output: `(30, 30, 64)` |
| **Layer 5** | Conv2D | 128 filters, Kernel $3 \times 3$, Activation: `relu` |
| **Layer 6** | MaxPooling2D | Pool size $2 \times 2$, Output: `(14, 14, 128)` |
| **Layer 7** | Flatten | Phẳng hóa thành vector 1 chiều: $14 \times 14 \times 128 = 25088$ |
| **Layer 8** | Dense | 128 neurons, Activation: `relu` |
| **Layer 9** | Dropout | Tỷ lệ $0.5$ (chống hiện tượng Overfitting) |
| **Layer 10** | Dense (Output) | 3 neurons (Chai nhựa, Lon nhôm, Thùng giấy), Activation: `softmax` |

* **Hàm mất mát (Loss function):** `sparse_categorical_crossentropy` (phù hợp nhãn dạng số nguyên không cần mã hóa một nóng).
* **Bộ tối ưu (Optimizer):** `Adam` với tỷ lệ học (learning rate) mặc định `0.001`.
* **Số Epoch tối đa:** 30 epochs.
* **Batch size:** 32.
* **Cơ chế dừng sớm (Early Stopping):** Theo dõi `val_loss`, dừng nếu không cải thiện sau **5 epochs liên tiếp** (`patience=5`), khôi phục trọng số tốt nhất (`restore_best_weights=True`).

---

## 📊 Tập Dữ Liệu & Kết Quả Huấn Luyện

### 1. Cấu trúc tập dữ liệu (Dataset)
Tổng số lượng ảnh tải vào hệ thống là **1295 ảnh**, được lưu trữ cục bộ tại đường dẫn `E:\machine_learning_project\dataset`. 
Dữ liệu được phân chia theo tỷ lệ cân bằng (`stratify=y`): **70% Train / 15% Validation / 15% Test**.

**Quy trình tiền xử lý:**
1. Quét ảnh tự động từ các thư mục con tương ứng với tên lớp, lọc bỏ các file không đúng định dạng ảnh (chỉ nhận `.png`, `.jpg`, `.jpeg`).
2. Định dạng lại kích thước ảnh (Resize) về kích thước chuẩn hóa `(128, 128)`.
3. Chuyển thành mảng NumPy, ép kiểu `float32` và chuẩn hóa giá trị pixel về đoạn `[0, 1]` bằng cách chia cho `255.0` nhằm tăng tốc độ hội tụ khi huấn luyện.

### 2. Kết quả huấn luyện
Nhờ cơ chế Early Stopping, mô hình tự động dừng huấn luyện tại **Epoch thứ 13** để tối ưu thời gian huấn luyện và tránh hiện tượng học vẹt.

| Tập dữ liệu (Dataset) | Độ chính xác (Accuracy) | Giá trị mất mát (Loss) |
| :--- | :---: | :---: |
| **Train** (Huấn luyện) | 85.30% | 0.3900 |
| **Validation** (Kiểm định) | 78.35% | 0.5194 |
| **Test** (Đánh giá độc lập) | **73.85%** | 0.6510 |

**Đánh giá & Nhận xét:**
* Độ chính xác tập Test đạt **73.85%**, mức Khá đối với mô hình CNN tự xây dựng từ đầu (Scratch) trên tập dữ liệu nhỏ (~1300 ảnh cho 3 lớp).
* **Ưu điểm:** Mô hình nhận diện rất tốt `Thùng giấy carton` do đặc trưng hình khối vuông vức và diện tích bề mặt lớn, dễ phân biệt.
* **Hạn chế:** Giữa `Chai nhựa` và `Lon nhôm` đôi khi xảy ra nhầm lẫn nhẹ do một số góc chụp phản chiếu ánh sáng giống nhau hoặc chai nhựa bị móp méo làm thay đổi hình dạng đặc trưng, hoặc do nhãn mác gây nhiễu đặc trưng.

---

## 💻 Hướng Dẫn Cài Đặt & Chạy Demo

### 1. Chuẩn bị môi trường
* Hệ điều hành: Windows / Linux / macOS
* Python phiên bản khuyến nghị: `3.10`
* Các thư viện cần thiết: `tensorflow`, `numpy`, `scikit-learn`, `matplotlib`

Cài đặt nhanh các thư viện bằng pip:
```bash
pip install tensorflow numpy scikit-learn matplotlib
