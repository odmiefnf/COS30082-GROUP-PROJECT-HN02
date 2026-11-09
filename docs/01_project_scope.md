# Phần 1: Mục tiêu và phạm vi dự án

**Dự án:** Facial Recognition with Emotion & Liveness  
**Môn học:** COS30082 Applied Machine Learning  
**Nguồn yêu cầu:** mục 1 - Introduction, trang 1 của đề bài PDF.  
**Trạng thái:** Đã hoàn thành bản xác định yêu cầu; người dùng đã yêu cầu chuyển sang phần 2.

## 1. Mục tiêu

Xây dựng hệ thống chấm công bằng khuôn mặt cho doanh nghiệp, sử dụng mạng nơ-ron tích chập (CNN) để thực hiện xác minh khuôn mặt (face verification). Hệ thống phải tính đến việc đăng ký nhân viên mới, chống giả mạo và nhận diện cảm xúc.

Trọng tâm của dự án là áp dụng học máy và tự xây dựng hệ thống, thể hiện được cách mô hình được huấn luyện và sử dụng.

## 2. Làm rõ bài toán

| Khái niệm | Ý nghĩa | Ví dụ |
| --- | --- | --- |
| Face classification | Gán khuôn mặt vào một ID trong các danh tính đã biết khi huấn luyện; đề mô tả đây là bài toán closed-set. | Ảnh này thuộc An, Bình hay Chi? |
| Face verification | Xác định hai ảnh có thuộc cùng một người hay không; có thể gặp danh tính chưa xuất hiện khi huấn luyện. | Hai ảnh này có phải cùng một người không? |

Hướng giải quyết chính của dự án là **face verification**. Cách huấn luyện và so sánh các mô hình sẽ được xử lý ở phần 2.

## 3. Yêu cầu lấy trực tiếp từ phần 1

| Mã | Yêu cầu | Ý nghĩa khi triển khai |
| --- | --- | --- |
| P1-01 | Dùng CNN để thực hiện face verification. | Xây dựng mô hình phục vụ xác minh danh tính qua ảnh khuôn mặt. |
| P1-02 | Thiết kế hệ thống chấm công hoàn chỉnh cho doanh nghiệp. | Các thành phần cần hoạt động cùng nhau trong hệ thống cuối cùng. |
| P1-03 | Tính đến việc đăng ký face ID mới khi tuyển nhân viên. | Có chức năng bổ sung danh tính mới vào hệ thống. |
| P1-04 | Tích hợp module anti-spoofing. | Có thành phần phát hiện khuôn mặt giả mạo trong hệ thống. |
| P1-05 | Tích hợp module emotion detection. | Có thành phần nhận diện cảm xúc từ khuôn mặt. |
| P1-06 | Tự xây dựng hệ thống để áp dụng kiến thức học máy. | Việc thực hiện phải thể hiện công việc xây dựng và huấn luyện của nhóm. |
| P1-07 | Không dùng mô hình pretrained nguyên trạng mà không huấn luyện thêm. | Có thể tự huấn luyện, hoặc dùng transfer learning và fine-tune một phần mô hình. |

Các mã P1 ở trên dùng để theo dõi công việc; chúng không phải mã có sẵn trong đề.

## 4. Thiết kế ban đầu được đề xuất

Nội dung dưới đây là phương án tổ chức ban đầu để đáp ứng yêu cầu, không phải yêu cầu bổ sung từ đề bài.

| Thành phần | Trách nhiệm dự kiến |
| --- | --- |
| Đăng ký danh tính | Tiếp nhận ID/thông tin và ảnh khuôn mặt của nhân viên mới để tạo hồ sơ đăng ký. |
| Xác minh khuôn mặt | Dùng CNN để hỗ trợ so sánh khuôn mặt cần kiểm tra với dữ liệu tham chiếu. |
| Chống giả mạo | Phân tích dữ liệu khuôn mặt và trả về kết quả kiểm tra giả mạo. |
| Nhận diện cảm xúc | Phân tích biểu cảm khuôn mặt và trả về nhãn cảm xúc. |
| Tích hợp chấm công | Kết nối các thành phần trên thành luồng sử dụng của hệ thống. |

Ba tình huống dùng để định hướng thiết kế:

1. **Có nhân viên mới:** hệ thống có quy trình bổ sung danh tính và dữ liệu khuôn mặt của người đó.
2. **Cần xác minh một người:** thành phần xác minh đưa ra kết quả dựa trên khuôn mặt cần kiểm tra và dữ liệu tham chiếu.
3. **Sử dụng hệ thống tích hợp:** các thành phần xác minh, chống giả mạo và cảm xúc cùng tham gia vào hệ thống chấm công.

Chính sách ra quyết định chấm công và cách phối hợp chi tiết giữa các module sẽ được xác định khi triển khai. Phần 1 không quy định cảm xúc phải quyết định việc chấp nhận chấm công.

## 5. Những lựa chọn chưa chốt ở bước này

- Framework và kiến trúc CNN.
- Dữ liệu huấn luyện và cách tiền xử lý.
- Loss function, siêu tham số và lịch huấn luyện.
- Phép đo so khớp và ngưỡng quyết định.
- Cách lưu dữ liệu đăng ký và hình thức giao diện.

Các lựa chọn trên sẽ được giải quyết ở phần tương ứng, dựa trên yêu cầu của toàn đề và dữ liệu thực tế.

## 6. Kết quả của phần 1 và điểm dừng

- Đã xác định mục tiêu và hướng giải quyết chính bằng face verification.
- Đã ghi nhận đủ ba tình huống được nêu trong phần giới thiệu.
- Đã xác định ràng buộc sử dụng mô hình pretrained và quyền dùng transfer learning/fine-tuning.
- Đã phác thảo các thành phần cần xây dựng, tách biệt với yêu cầu gốc.

Phần 1 là bước xác định mục tiêu và phạm vi; đề không yêu cầu một chương trình hoặc một lượt huấn luyện riêng cho phần này. Tài liệu này là sản phẩm để người dùng xem trước khi chuyển sang phần 2.

**Bước tiếp theo sau khi người dùng xem và yêu cầu tiếp tục:** phần 2 - Face Recognition, bắt đầu từ dữ liệu đầu vào, face embedding và hai hướng huấn luyện.
