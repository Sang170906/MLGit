# Hệ Thống Robot Tự Động Phân Loại Phế Liệu Sử Dụng Học Máy (CNN)

[cite_start]Dự án xây dựng hệ thống phân loại phế liệu tự động trên băng chuyền sử dụng mạng nơ-ron tích chập (CNN) nhằm thay thế các phương pháp phân loại thủ công, tăng năng suất và độ chính xác trong quản lý chất thải và tái chế[cite: 8, 9].

---

##  Thành Viên Thực Hiện
* [cite_start]**Họ và tên:** Cao Xuân Sang [cite: 3]
* [cite_start]**Mã sinh viên:** 2421060188 [cite: 4]
* [cite_start]**Lớp:** DCCDROBOT69 [cite: 5]
* [cite_start]**Mã môn học:** 7090299 [cite: 6]
* [cite_start]**Giảng viên hướng dẫn:** Nguyễn Đức Khoát [cite: 7]

---

##  Tổng Quan Bài Toán & Kiến Trúc

### 1. Mô tả bài toán
[cite_start]Hệ thống nhận diện và phân loại phế liệu di chuyển trên băng chuyền thành **3 nhóm chính**[cite: 9, 10]:
1. [cite_start]`Chai nhựa` [cite: 11]
2. [cite_start]`Lon nhôm` [cite: 12]
3. [cite_start]`Thùng giấy carton` [cite: 13]

* [cite_start]**Lý do sử dụng Học sâu (Deep Learning):** Do hình dạng, kích thước, góc nghiêng, nhãn mác của phế liệu rất đa dạng và điều kiện ánh sáng thực tế trên băng chuyền thay đổi liên tục[cite: 14, 15, 16]. [cite_start]Các phương pháp xử lý ảnh truyền thống dựa trên ngưỡng màu hoặc đường biên cố định không thể giải quyết triệt để[cite: 17]. [cite_start]Mạng CNN có khả năng tự động trích xuất các đặc trưng hình học bất chấp sự dịch chuyển, xoay hoặc thay đổi độ sáng[cite: 18].
* **Yêu cầu hệ thống:**
  * [cite_start]**Đầu vào (Input):** Ảnh phế liệu từ tập dữ liệu sẵn có, kích thước gốc được chuẩn hóa về $128 \times 128$[cite: 19, 20].
  * [cite_start]**Đầu ra (Output):** Phân loại chính xác 1 trong 3 nhóm phế liệu (chai nhựa, lon nhôm, thùng giấy coton)[cite: 21].
  * [cite_start]**Thời gian xử lý:** $\le 0.5$ giây/ảnh để đảm bảo kịp tốc độ dịch chuyển của băng chuyền[cite: 22].

### 2. Kiến trúc mô hình (CNN Sequential)
[cite_start]Mô hình mạng CNN được xây dựng theo cấu trúc `Sequential` tuần tự gồm các tầng trích xuất đặc trưng nối tiếp các tầng phân loại tuyến tính[cite: 32, 33]:

| Tầng (Layer) | Loại Layer | Cấu hình / Kích thước đầu ra |
| :--- | :--- | :--- |
| **Layer 1** | Conv2D | 32 bộ lọc (filters), kích thước kernel $3 \times 3$, hàm kích hoạt `relu`. [cite_start]Nhận Input Shape `(128, 128, 3)`[cite: 34]. |
| **Layer 2** | MaxPooling2D | [cite_start]Kích thước pool $2 \times 2$, giúp giảm chiều không gian xuống còn `(64, 64, 32)`[cite: 35]. |
| **Layer 3** | Conv2D | [cite_start]64 bộ lọc, kích thước kernel $3 \times 3$, hàm kích hoạt `relu`[cite: 36]. |
| **Layer 4** | MaxPooling2D | [cite_start]Kích thước pool $2 \times 2$, giảm chiều xuống còn `(30, 30, 64)`[cite: 37]. |
| **Layer 5** | Conv2D | [cite_start]128 bộ lọc, kích thước kernel $3 \times 3$, hàm kích hoạt `relu`[cite: 38]. |
| **Layer 6** | MaxPooling2D | [cite_start]Kích thước pool $2 \times 2$, giảm chiều xuống còn `(14, 14, 128)`[cite: 39]. |
| **Layer 7** | Flatten | [cite_start]Phẳng hóa ma trận đặc trưng thành vector 1 chiều có kích thước $14 \times 14 \times 128 = 25088$ phần tử[cite: 40]. |
| **Layer 8** | Dense | [cite_start]Tầng kết nối đầy đủ (Fully Connected) gồm 128 nơ-ron, hàm kích hoạt `relu`[cite: 41]. |
| **Layer 9** | Dropout | [cite_start]Tỷ lệ ẩn `0.5` giúp chống học vẹt (overfitting)[cite: 42]. |
| **Layer 10** | Dense (Output) | [cite_start]3 nơ-ron đại diện cho 3 lớp phế liệu, hàm kích hoạt `softmax` để tính phân phối xác suất đầu ra[cite: 43]. |

* [cite_start]**Hàm mất mát (Loss function):** `sparse_categorical_crossentropy` (phù hợp với nhãn dạng số nguyên không cần mã hóa một nóng)[cite: 44, 45].
* [cite_start]**Bộ tối ưu (Optimizer):** `Adam` với tỷ lệ học (learning rate) mặc định `0.001`[cite: 46].
* [cite_start]**Số Epoch tối đa:** 30 epochs[cite: 48].
* [cite_start]**Batch size:** 32[cite: 49].
* [cite_start]**Biện pháp chống quá khớp (Early Stopping):** Theo dõi giá trị `val_loss`, nếu không cải thiện sau **5 epochs liên tiếp** (`patience=5`), hệ thống sẽ tự động dừng huấn luyện và khôi phục trọng số tốt nhất (`restore_best_weights=True`)[cite: 47].

---

##  Tập Dữ Liệu & Kết Quả Huấn Luyện

### 1. Cấu trúc tập dữ liệu (Dataset)
[cite_start]Tổng số lượng ảnh tải vào hệ thống là **1295 ảnh**, được lưu trữ cục bộ tại đường dẫn cố định trên máy: `E:\machine_learning_project\dataset`[cite: 24, 31]. [cite_start]Dữ liệu được phân tách theo tỷ lệ **70% Train / 15% Validation / 15% Test** bằng hàm `train_test_split` có cấu hình `stratify=y` nhằm đảm bảo tính cân bằng giữa các lớp dữ liệu trong mỗi tập[cite: 30].

**Quy trình tiền xử lý:**
1. [cite_start]Quét ảnh tự động từ các thư mục con tương ứng với tên lớp[cite: 26].
2. [cite_start]Lọc bỏ các file không đúng định dạng ảnh, chỉ chấp nhận đuôi `.png`, `.jpg`, `.jpeg`[cite: 27].
3. [cite_start]Resize toàn bộ ảnh về kích thước chuẩn hóa `(128, 128)`[cite: 28].
4. [cite_start]Chuyển đổi ảnh thành mảng Numpy, ép kiểu `float32` và chuẩn hóa giá trị pixel về đoạn `[0, 1]` bằng cách chia cho `255.0` nhằm tăng tốc độ hội tụ khi huấn luyện[cite: 29].

### 2. Kết quả huấn luyện
[cite_start]Mô hình tự động dừng huấn luyện ở **Epoch thứ 13** nhờ cơ chế Early Stopping giúp tối ưu thời gian huấn luyện và tránh hiện tượng học vẹt[cite: 51].

| Tập dữ liệu (Dataset) | Độ chính xác (Accuracy) | Giá trị mất mát (Loss) |
| :--- | :---: | :---: |
| **Train** (Huấn luyện) | 85.30% | [cite_start]0.3900 [cite: 53] |
| **Validation** (Kiểm định) | 78.35% | [cite_start]0.5194 [cite: 53] |
| **Test** (Đánh giá độc lập) | **73.85%** | [cite_start]0.6510 [cite: 53] |

**Đánh giá đồ thị từ Plot pane:**
* [cite_start]**Đồ thị Accuracy:** Đường `Train Accuracy` và `Val Accuracy` có xu hướng đồng hành đi lên trong 8 epoch đầu tiên[cite: 56]. [cite_start]Tuy nhiên từ epoch thứ 9 trở đi, khoảng cách giữa tập train và validation bắt đầu có sự phân kỳ nhẹ, cho thấy dấu hiệu mô hình bắt đầu bị bão hòa thông tin trích xuất trên tập dữ liệu hiện tại[cite: 57].
* [cite_start]**Đồ thị Loss:** Giá trị `Loss` giảm mạnh từ mức ban đầu (>1.0) xuống dưới mức 0.6[cite: 58]. [cite_start]Hàm `val_loss` đạt điểm tối ưu thấp nhất tại khoảng epoch thứ 8 trước khi có biến động đi ngang và tăng nhẹ, kích hoạt lệnh dừng sớm ở epoch 13 để giữ lại các trọng số tối ưu nhất[cite: 59].

**Nhận xét kết quả:**
* [cite_start]Độ chính xác trên tập Test đạt **73.85%**[cite: 61]. [cite_start]Đây là kết quả ở mức Khá đối với một mô hình CNN cơ bản tự xây dựng (Scratch) huấn luyện trên tập dữ liệu nhỏ (~1300 ảnh chia cho 3 lớp)[cite: 61].
* [cite_start]**Ưu điểm:** Mô hình nhận diện tương đối chính xác `Thùng giấy carton` do đặc trưng hình khối và diện tích bề mặt lớn, dễ phân biệt[cite: 62].
* [cite_start]**Hạn chế:** Giữa `Chai nhựa` và `Lon nhôm` đôi khi vẫn xảy ra sự nhầm lẫn nhẹ do một số góc chụp phản chiếu ánh sáng giống nhau hoặc nhãn mác gây nhiễu đặc trưng[cite: 63].

---

##  Mô Phỏng Dự Đoán (Demo Inference)

[cite_start]Vì hệ thống không sử dụng Webcam trực tiếp, phần Demo được thực hiện thông qua việc chạy dự đoán trên các hình ảnh ngẫu nhiên từ tập dữ liệu Test để kiểm tra khả năng nhận diện thực tế của Model sau huấn luyện[cite: 65].

* [cite_start]**Môi trường mô phỏng:** Python 3.10 - Thư viện TensorFlow & Matplotlib[cite: 66].
* [cite_start]**Dữ liệu đầu vào:** Các tệp ảnh phế liệu trong thư mục `Test` chưa qua huấn luyện[cite: 67].
* [cite_start]**Quy trình thực hiện:** 1. Tải Model đã lưu (`my_model.keras`)[cite: 69].
  2. [cite_start]Đưa ảnh vào hàm `model.predict()`[cite: 70].
  3. [cite_start]Hiển thị ảnh kèm theo nhãn dự đoán và nhãn thực tế để đối soát[cite: 71].
* **Thông số khi chạy thực tế (Inference):**
  * [cite_start]**Tốc độ xử lý:** ~0.05 giây / 1 hình ảnh (Rất nhanh, đáp ứng tốt nếu đưa lên băng chuyền thực tế)[cite: 73].
  * [cite_start]**Độ chính xác demo:** ~74% (Sát với kết quả đánh giá tập Test)[cite: 74].
  * [cite_start]**Kết quả dự đoán chi tiết:** Đối với *Thùng giấy* nhận diện rất tốt nhờ đặc trưng hình khối vuông vức[cite: 76]. [cite_start]Đối với *Lon nhôm & Chai nhựa* nhận diện tốt ở điều kiện ảnh rõ nét, đôi khi nhầm lẫn nếu chai nhựa bị móp méo làm thay đổi hình dạng đặc trưng[cite: 77].

---

##  Kết Luận & Hướng Phát Triển

### Đã đạt được:
* [cite_start]Xây dựng hoàn chỉnh mã nguồn tải dữ liệu tự động, tiền xử lý hình ảnh chuyên sâu, phân tách tập dữ liệu cân bằng khoa học[cite: 80].
* [cite_start]Thiết lập cấu trúc mạng CNN tầng bậc hoạt động ổn định, không bị lỗi sập chương trình, tích hợp thành công callback Early Stopping kiểm soát Overfitting[cite: 81].
* [cite_start]Mô hình được đóng gói thành công dưới định dạng chuẩn `my_model.keras` lưu trữ trực tiếp tại thư mục dự án[cite: 82].

### Hạn chế:
* [cite_start]Độ chính xác trên tập dữ liệu kiểm thử (73.85%) cần được cải thiện thêm để có thể đưa vào ứng dụng thực tế trong công nghiệp sản xuất (thường yêu cầu >90%)[cite: 84].
* [cite_start]Kiến trúc mạng CNN tự thiết kế còn đơn giản, chưa tối ưu hết các tính chất sâu của ảnh[cite: 85].

### Hướng phát triển:
1. [cite_start]**Tăng cường dữ liệu (Data Augmentation):** Áp dụng kỹ thuật như xoay ảnh, lật ảnh, thay đổi độ tương phản trực tiếp trong quá trình Train để mô hình học được nhiều biến thể phế liệu hơn[cite: 87].
2. [cite_start]**Học chuyển giao (Transfer Learning):** Chuyển sang sử dụng phương pháp Học chuyển giao, ứng dụng các backbone mạnh mẽ đã được tiền huấn luyện trên tập ImageNet khổng lồ như `MobileNetV2` hoặc `ResNet50` để nâng cao vượt trội độ chính xác mà không tốn nhiều tài nguyên huấn luyện[cite: 88].

