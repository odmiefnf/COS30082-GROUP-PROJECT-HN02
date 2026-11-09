# Giai đoạn 2: Lịch sử điểm danh và xuất CSV

Hoàn tất ngày **30/10/2026**. Giao diện giữ tiếng Anh. Tính năng xem dữ liệu đã lưu, hoạt động khi camera tắt và không phụ thuộc việc nạp CNN thành công.

## Cách dùng

1. Mở app bằng `start_attendance.cmd`, bấm **History / CSV** ở thanh trên.
2. Trong **Session**, chọn **All sessions** hoặc một buổi cụ thể. Các buổi trong danh sách có tiền tố `Session:` để phân biệt với lựa chọn xem tất cả.
3. Trong **Source**, chọn **All sources**, **Automatic**, **Manual** hoặc **Unknown (legacy)**. Có thể nhập một phần tên/MSSV hoặc tên buổi vào **Search name or session**.
4. Dòng tổng hợp cho biết số buổi và số bản ghi đang hiển thị, kèm số Automatic/Manual/Unknown. Đây là số bản ghi đã có; không suy ra người vắng hoặc đi muộn vì app chưa có danh sách lớp và giờ bắt đầu.
5. Chọn dòng rồi bấm **View details** hoặc nhấp đúp để xem nguồn, người đối chiếu, lý do và thời gian.
6. Bấm **Refresh** để lấy bản chụp dữ liệu mới. Lịch sử không tự chạy lại nhận diện; dữ liệu xuất là bản chụp của các dòng đang hiển thị khi bấm xuất.
7. Bấm **Export displayed rows…**, chọn nơi lưu tệp `.csv`. Nếu bộ lọc không còn dòng nào, nút xuất được khóa. Đóng cửa sổ bằng **Close**.

Camera không cần bật để xem/xuất. Việc mở lịch sử hoặc lọc không thay đổi tên buổi đang nhập trên cửa sổ chính. Bản ghi cũ chưa có nguồn xác nhận được hiển thị **Unknown**, không tự gán là Automatic.

## Nội dung CSV

CSV dùng UTF-8 có BOM để mở tiếng Việt trong Excel. Các cột gồm:

`session, person_id, name, recorded_local, recorded_utc, source, source_code, reviewer, reason, observed_utc, reviewed_utc`

`person_id` là ID nội bộ của hồ sơ; tên/MSSV người dùng đã nhập nằm trong `name`. `recorded_local` dùng UTC+07:00, kèm thời gian UTC gốc để đối chiếu. `source` là Automatic/Manual/Unknown; `source_code` là mã tương ứng automatic/manual_review/legacy_unrecorded. Dữ liệu người đối chiếu/lý do được giữ cho bản ghi thủ công.

CSV không chứa embedding, ảnh/video, hoặc biểu cảm. Các ô văn bản bắt đầu như công thức bảng tính được thêm dấu nháy đơn để giữ dạng văn bản; dữ liệu trong database không bị sửa. Bộ xuất dùng CSV quoting để giữ đúng dấu phẩy/dấu nháy trong tên và lý do. Chỉ ghi đè tệp CSV có sẵn sau xác nhận ở hộp Save của người dùng; không cho dùng database làm tệp xuất.

## Phạm vi và kiểm tra

`attendance_app/history.py` mở SQLite bằng chế độ chỉ đọc và lấy một snapshot nhất quán. Không tạo database nếu đường dẫn chưa có và không chạy migration khi xem dữ liệu cũ. Việc xuất chỉ dùng snapshot đã lọc, không ghi điểm danh. Nếu refresh lỗi, app giữ snapshot trước và báo lỗi rõ; nhấn Refresh lại sau khi khắc phục.

Tám bài kiểm tra mới đạt, gồm provenance, bộ lọc/tổng hợp, database cũ/thiếu/rỗng, không sửa database, snapshot trước/sau khi có bản ghi mới, UTF-8/CSV quoting, công thức văn bản và bảo vệ tệp có sẵn. Lượt toàn bộ đạt **134/134 bài trong 61,737 giây**, nhật ký `results/app_upgrade_20261030/phase2_full_tests.log`.

Đã kiểm tra trực quan cửa sổ chính, bảng lịch sử, các nút và bộ lọc Automatic bằng hồ sơ mô phỏng. CSV được kiểm tra bằng dữ liệu tạm trong unit tests. Chưa xuất dữ liệu thật của người dùng qua giao diện; không mở webcam hoặc sửa database thật trong giai đoạn này.

Xem bố cục độc lập bằng:

```powershell
.\.venv-gpu\Scripts\python.exe -m attendance_app.gui --preview-history
```

Cửa sổ mang nhãn **PREVIEW DATA** và chỉ chứa hồ sơ giả lập. [Lộ trình](app_upgrade_roadmap.md): dừng sau giai đoạn 2 để người dùng xem; giai đoạn 3 quản lý hồ sơ/nhập ảnh tham chiếu chưa bắt đầu.
