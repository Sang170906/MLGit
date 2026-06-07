# Hệ Thống Robot Tự Động Phân Loại Phế Liệu Sử Dụng Học Máy (CNN)

Dự án xây dựng hệ thống phân loại phế liệu tự động trên băng chuyền sử dụng mạng nơ-ron tích chập (CNN) nhằm thay thế các phương pháp phân loại thủ công, tăng năng suất và độ chính xác trong quản lý chất thải và tái chế

---

##  Thành Viên Thực Hiện
* **Họ và tên:** Cao Xuân Sang 
* **Mã sinh viên:** 2421060188 
***Lớp:** DCCDROBOT69
***Mã môn học:** 7090299
* **Giảng viên hướng dẫn:** Nguyễn Đức Khoát

---

## Tổng Quan Bài Toán & Kiến Trúc

### 1. Mô tả bài toán
Hệ thống nhận diện và phân loại phế liệu di chuyển trên băng chuyền thành **3 nhóm chính**:
1. `Chai nhựa` 
2. `Lon nhôm` 
3.`Thùng giấy carton` 

* **Lý do sử dụng Học sâu (Deep Learning):** Do hình dạng, kích thước, góc nghiêng, nhãn mác của phế liệu rất đa dạng và điều kiện ánh sáng thực tế trên băng chuyền thay đổi liên tục.Các phương pháp xử lý ảnh truyền thống dựa trên ngưỡng màu hoặc đường biên cố định không thể giải quyết triệt để.Mạng CNN có khả năng tự động trích xuất các đặc trưng hình học bất chấp sự dịch chuyển, xoay hoặc thay đổi độ sáng
* **Yêu cầu hệ thống:**
  ***Đầu vào (Input):** Ảnh phế liệu từ tập dữ liệu sẵn có, kích thước gốc được chuẩn hóa về 128x128
  * **Đầu ra (Output):** Phân loại chính xác 1 trong 3 nhóm phế liệu (chai nhựa, lon nhôm, thùng giấy coton).
  * **Thời gian xử lý:** \le 0.5 giây/ảnh để đảm bảo kịp tốc độ dịch chuyển của băng chuyền.

### 2. Kiến trúc mô hình (CNN Sequential)
Mô hình mạng CNN được xây dựng theo cấu trúc `Sequential` tuần tự gồm các tầng trích xuất đặc trưng nối tiếp các tầng phân loại tuyến tính:

| Tầng (Layer) | Loại Layer | Cấu hình / Kích thước đầu ra |
| :--- | :--- | :--- |
| **Layer 1** | Conv2D | 32 bộ lọc (filters), kích thước kernel 3x3, hàm kích hoạt `relu`.Nhận Input Shape `(128, 128, 3)` |
| **Layer 2** | MaxPooling2D | Kích thước pool 2x2, giúp giảm chiều không gian xuống còn `(64, 64, 32)`. |
| **Layer 3** | Conv2D | 64 bộ lọc, kích thước kernel 3x3, hàm kích hoạt `relu` |
| **Layer 4** | MaxPooling2D | Kích thước pool 2x2, giảm chiều xuống còn `(30, 30, 64)` |
| **Layer 5** | Conv2D |128 bộ lọc, kích thước kernel 3x3, hàm kích hoạt `relu`. |
| **Layer 6** | MaxPooling2D | Kích thước pool 2x2 giảm chiều xuống còn `(14, 14, 128)`. |
| **Layer 7** | Flatten | Phẳng hóa ma trận đặc trưng thành vector 1 chiều có kích thước 14x14x128 = 25088 phần tử. |
| **Layer 8** | Dense | Tầng kết nối đầy đủ (Fully Connected) gồm 128 nơ-ron, hàm kích hoạt `relu` |
| **Layer 9** | Dropout |Tỷ lệ ẩn `0.5` giúp chống học vẹt (overfitting). |
| **Layer 10** | Dense (Output) |3 nơ-ron đại diện cho 3 lớp phế liệu, hàm kích hoạt `softmax` để tính phân phối xác suất đầu ra. |

* **Hàm mất mát (Loss function):** `sparse_categorical_crossentropy` (phù hợp với nhãn dạng số nguyên không cần mã hóa một nóng)
* **Bộ tối ưu (Optimizer):** `Adam` với tỷ lệ học (learning rate) mặc định `0.001`
* **Số Epoch tối đa:** 30 epochs
* **Batch size:** 32
* **Biện pháp chống quá khớp (Early Stopping):** Theo dõi giá trị `val_loss`, nếu không cải thiện sau **5 epochs liên tiếp** (`patience=5`), hệ thống sẽ tự động dừng huấn luyện và khôi phục trọng số tốt nhất (`restore_best_weights=True`)

---

##  Tập Dữ Liệu & Kết Quả Huấn Luyện

### 1. Cấu trúc tập dữ liệu (Dataset)
]Tổng số lượng ảnh tải vào hệ thống là **1295 ảnh**, được lưu trữ cục bộ tại đường dẫn cố định trên máy: `E:\machine_learning_project\dataset`. Dữ liệu được phân tách theo tỷ lệ **70% Train / 15% Validation / 15% Test** bằng hàm `train_test_split` có cấu hình `stratify=y` nhằm đảm bảo tính cân bằng giữa các lớp dữ liệu trong mỗi tập

**Quy trình tiền xử lý:**
1. Quét ảnh tự động từ các thư mục con tương ứng với tên lớp
2.Lọc bỏ các file không đúng định dạng ảnh, chỉ chấp nhận đuôi `.png`, `.jpg`, `.jpeg`
3.Resize toàn bộ ảnh về kích thước chuẩn hóa `(128, 128)`
4. Chuyển đổi ảnh thành mảng Numpy, ép kiểu `float32` và chuẩn hóa giá trị pixel về đoạn `[0, 1]` bằng cách chia cho `255.0` nhằm tăng tốc độ hội tụ khi huấn luyện

### 2. Kết quả huấn luyện
Mô hình tự động dừng huấn luyện ở **Epoch thứ 13** nhờ cơ chế Early Stopping giúp tối ưu thời gian huấn luyện và tránh hiện tượng học vẹt

| Tập dữ liệu (Dataset) | Độ chính xác (Accuracy) | Giá trị mất mát (Loss) |

| **Train** (Huấn luyện) | 85.30% | 0.3900
| **Validation** (Kiểm định) | 78.35% | 0.5194 
| **Test** (Đánh giá độc lập) | **73.85%**  |  0.6510

**Đánh giá đồ thị từ Plot pane:**
***Đồ thị Accuracy:** Đường `Train Accuracy` và `Val Accuracy` có xu hướng đồng hành đi lên trong 8 epoch đầu tiên.Tuy nhiên từ epoch thứ 9 trở đi, khoảng cách giữa tập train và validation bắt đầu có sự phân kỳ nhẹ, cho thấy dấu hiệu mô hình bắt đầu bị bão hòa thông tin trích xuất trên tập dữ liệu hiện tại
* **Đồ thị Loss:** Giá trị `Loss` giảm mạnh từ mức ban đầu (>1.0) xuống dưới mức 0.6].Hàm `val_loss` đạt điểm tối ưu thấp nhất tại khoảng epoch thứ 8 trước khi có biến động đi ngang và tăng nhẹ, kích hoạt lệnh dừng sớm ở epoch 13 để giữ lại các trọng số tối ưu nhất

**Nhận xét kết quả:**
*Độ chính xác trên tập Test đạt **73.85%**. Đây là kết quả ở mức Khá đối với một mô hình CNN cơ bản tự xây dựng (Scratch) huấn luyện trên tập dữ liệu nhỏ (~1300 ảnh chia cho 3 lớp).
* **Ưu điểm:** Mô hình nhận diện tương đối chính xác `Thùng giấy carton` do đặc trưng hình khối và diện tích bề mặt lớn, dễ phân biệt
* **Hạn chế:** Giữa `Chai nhựa` và `Lon nhôm` đôi khi vẫn xảy ra sự nhầm lẫn nhẹ do một số góc chụp phản chiếu ánh sáng giống nhau hoặc nhãn mác gây nhiễu đặc trưng.

---

##  Mô Phỏng Dự Đoán (Demo Inference)

Vì hệ thống không sử dụng Webcam trực tiếp, phần Demo được thực hiện thông qua việc chạy dự đoán trên các hình ảnh ngẫu nhiên từ tập dữ liệu Test để kiểm tra khả năng nhận diện thực tế của Model sau huấn luyện.

* **Môi trường mô phỏng:** Python 3.10 - Thư viện TensorFlow & Matplotlib.
* **Dữ liệu đầu vào:** Các tệp ảnh phế liệu trong thư mục `Test` chưa qua huấn luyện
* **Quy trình thực hiện:** 1. Tải Model đã lưu (`my_model.keras`)
  2. Đưa ảnh vào hàm `model.predict()`.
  3.Hiển thị ảnh kèm theo nhãn dự đoán và nhãn thực tế để đối soát
* **Thông số khi chạy thực tế (Inference):**
  * **Tốc độ xử lý:** ~0.05 giây / 1 hình ảnh (Rất nhanh, đáp ứng tốt nếu đưa lên băng chuyền thực tế).
  * **Độ chính xác demo:** ~74% (Sát với kết quả đánh giá tập Test).
  ***Kết quả dự đoán chi tiết:** Đối với *Thùng giấy* nhận diện rất tốt nhờ đặc trưng hình khối vuông vức
    Đối với *Lon nhôm & Chai nhựa* nhận diện tốt ở điều kiện ảnh rõ nét, đôi khi nhầm lẫn nếu chai nhựa bị móp méo làm thay đổi hình dạng đặc trưng

---

##  Kết Luận & Hướng Phát Triển

### Đã đạt được:
* Xây dựng hoàn chỉnh mã nguồn tải dữ liệu tự động, tiền xử lý hình ảnh chuyên sâu, phân tách tập dữ liệu cân bằng khoa học.
* Thiết lập cấu trúc mạng CNN tầng bậc hoạt động ổn định, không bị lỗi sập chương trình, tích hợp thành công callback Early Stopping kiểm soát Overfitting.
* Mô hình được đóng gói thành công dưới định dạng chuẩn `my_model.keras` lưu trữ trực tiếp tại thư mục dự án.

### Hạn chế:
*Độ chính xác trên tập dữ liệu kiểm thử (73.85%) cần được cải thiện thêm để có thể đưa vào ứng dụng thực tế trong công nghiệp sản xuất (thường yêu cầu >90%).
* Kiến trúc mạng CNN tự thiết kế còn đơn giản, chưa tối ưu hết các tính chất sâu của ảnh.

### Hướng phát triển:
1. **Tăng cường dữ liệu (Data Augmentation):** Áp dụng kỹ thuật như xoay ảnh, lật ảnh, thay đổi độ tương phản trực tiếp trong quá trình Train để mô hình học được nhiều biến thể phế liệu hơn.
2. **Học chuyển giao (Transfer Learning):** Chuyển sang sử dụng phương pháp Học chuyển giao, ứng dụng các backbone mạnh mẽ đã được tiền huấn luyện trên tập ImageNet khổng lồ như `MobileNetV2` hoặc `ResNet50` để nâng cao vượt trội độ chính xác mà không tốn nhiều tài nguyên huấn luyện.

