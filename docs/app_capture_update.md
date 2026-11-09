# Cập nhật bắt khuôn mặt và giao diện tiếng Anh

Ngày thực hiện: 27/10/2026. Cập nhật theo phản hồi của người dùng: khó giữ khuôn mặt được phát hiện và muốn toàn bộ giao diện bằng tiếng Anh.

## Nguyên nhân và thay đổi

Bắt khuôn mặt diễn ra trước liveness. Trước đây, bộ định vị dùng một Haar cascade trên ảnh rộng tối đa 640 pixel, `scaleFactor=1.1`, `minNeighbors=6`, kích thước tối thiểu 60 pixel ở ảnh đã thu nhỏ. Với khung rộng 960 pixel, điều này tương ứng khuôn mặt khoảng 90 pixel ở ảnh gốc. Khi không có mặt được phát hiện, các CNN không được chạy và mẫu đăng ký/chuỗi xác nhận bị reset. Liveness cũng là một mô hình riêng và có thể từ chối người thật dù định vị thành công.

Bộ mới trong `attendance_app/camera.py` dùng:

- Ảnh định vị rộng tối đa 800 pixel, `scaleFactor=1.08`, `minNeighbors=4`.
- Hai Haar cascade có sẵn: `frontalface_default` trên ảnh cân bằng histogram và `frontalface_alt2` trên ảnh CLAHE (clip limit 2, tile 8×8).
- Kích thước dò tối thiểu suy từ yêu cầu 80 pixel ở ảnh gốc.
- Loại box trùng có IoU từ 0,55; giữ các mặt riêng biệt để tiếp tục chặn nhiều người.

Không giữ lại box khi frame hiện tại mất mặt. CLAHE/cân bằng histogram chỉ áp dụng cho tìm mặt; ba CNN vẫn nhận cùng crop từ RGB gốc. Điều kiện chất lượng, ngưỡng identity, ngưỡng liveness, checkpoint và yêu cầu ba mẫu/ba khung liên tiếp không đổi. Thay đổi detector có thể thay đổi vị trí crop, nên vẫn cần kiểm tra camera riêng trước khi kết luận về chất lượng mô hình.

Toàn bộ cửa sổ chính và đối chiếu thủ công đã chuyển sang tiếng Anh, gồm lỗi kiểm tra đầu vào, trạng thái, cột nguồn và nhãn biểu cảm. Tên hồ sơ tiếng Việt do người dùng nhập vẫn được hỗ trợ và không bị dịch.

## Kết quả kiểm tra

Toàn bộ **97/97 bài kiểm thử đạt trong 70,1 giây**, gồm 5 bài mới về bộ định vị. Kiểm tra bổ sung xác nhận: hai cascade thấy cùng mặt chỉ sinh một crop RGB gốc; cascade phụ có thể khôi phục một lần dò bị bỏ sót; hai mặt kể cả chồng một phần vẫn bị chặn; không tái sử dụng box cũ khi mất mặt; tọa độ được đưa về đúng kích thước ảnh gốc. Nhật ký: `results/app_check_20261027/english_detector_tests.log`.

Lượt webcam ngắn so sánh **cùng 90 khung hình** trong khoảng 7,03 giây:

| Kết quả | Bộ cũ | Bộ mới |
| --- | ---: | ---: |
| Không bắt được mặt | 67 | 39 |
| Bắt được một mặt đủ điều kiện chất lượng | 23 | 51 |
| Qua liveness | 11 | 37 |
| Không qua liveness | 12 | 14 |
| Thời gian xử lý trung bình mỗi pipeline | 10,38 ms | 29,75 ms |

Đây là một lượt thử thuận tiện trên webcam, không có nhãn tấn công/người thật độc lập, không phải benchmark accuracy, FAR/FRR hoặc đánh giá chống giả mạo. Các khung liên tiếp có tương quan. Hai pipeline chạy lần lượt trên cùng frame, nên thời gian không được hiểu là FPS toàn bộ GUI. Bộ mới cần thêm tính toán để tăng số mặt tìm được. Không lưu ảnh/video; không tạo hồ sơ hoặc ghi điểm danh. Số liệu và checkpoint SHA-256: `results/app_check_20261027/detector_comparison.json`. Công cụ tái lập: `python -m tools.check_camera_detection --frames 90 --seconds 20 --output results/new_camera_comparison.json`; đóng các ứng dụng đang giữ webcam trước khi chạy.

Đã mở bản GUI mới và kiểm tra trực tiếp: hình camera hiện, có box mặt, `LIVENESS: Passed`, biểu cảm tiếng Anh và nút `Register from camera`. Chưa hoàn tất thử nghiệm đăng ký/điểm danh/đối chiếu thủ công với người dùng. Không nâng các kết quả này thành xác nhận ứng dụng ổn định trên mọi góc mặt hay điều kiện ánh sáng.

## Cách dùng bản mới

Mở `start_attendance.cmd`. Nhập `Session`, bấm `Start camera`, giữ một người trong khung. Khi `LIVENESS` là `Passed`, nhập tên và mã sinh viên rồi bấm `Register from camera`. Xoay nhẹ để lấy ba mẫu; sau đăng ký, tiếp tục nhìn camera để xác nhận điểm danh. `Stop` dừng và nhả camera. `Manual review` chỉ bật khi có hồ sơ và khuôn mặt qua liveness nhưng danh tính chưa chắc chắn.

Không có box: xem góc mặt, khoảng cách và ánh sáng. Có box nhưng `Not passed`: mô hình liveness đang từ chối crop; không tự hạ ngưỡng để cho qua. Nhiều người có mặt: yêu cầu từng người tiến vào khung.

Tài liệu OpenCV: [CascadeClassifier](https://docs.opencv.org/4.13.0/d1/de5/classcv_1_1CascadeClassifier.html), [CLAHE và cân bằng histogram](https://docs.opencv.org/4.13.0/d5/daf/tutorial_py_histogram_equalization.html).

Báo cáo Word phần 8 ghi nhận mốc trước lần cập nhật app này; ghi chú hiện tại mô tả thay đổi mới. Các bảng AUC/ACER/F1 của mô hình trong báo cáo không bị thay đổi bởi việc chỉnh định vị.
