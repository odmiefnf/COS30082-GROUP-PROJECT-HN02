# Bước 3: Profiles và nhập ảnh tham chiếu

Hoàn tất ngày **30/10/2026**. Giao diện giữ tiếng Anh. Mỗi người có **một hồ sơ** và tối đa bốn nhóm appearance; mỗi lượt nhập ảnh bổ sung đúng ba mẫu. Với trường hợp đeo/bỏ kính, thêm nhóm `Without glasses` vào hồ sơ đã có để giữ cùng ID và lịch sử điểm danh.

## Mở và quản lý hồ sơ

1. Mở `start_attendance.cmd`, chờ nạp mô hình. Nếu camera đang chạy, bấm **Stop**.
2. Bấm **Profiles…** trong vùng **People & registration**. Camera chỉ khởi động lại sau khi đóng cửa sổ này.
3. Tìm theo tên/mã sinh viên rồi chọn người. Bảng bên phải hiện từng appearance và số mẫu, không hiện ảnh gốc vì app không lưu ảnh.
4. Để đổi tên, sửa ô tên rồi bấm **Rename selected person**. ID, mẫu và các dòng điểm danh được giữ; tên hiển thị khi xem lại lịch sử dùng tên mới. Tệp CSV đã xuất trước đó vẫn là snapshot cũ.
5. Để bỏ một nhóm mẫu sai, chọn nhóm rồi bấm **Remove selected look…** và đọc xác nhận. App giữ ít nhất một nhóm cho mỗi người, không xóa hồ sơ hay bản ghi điểm danh.

Trước khi đổi tên hoặc bỏ nhóm mẫu, app sao lưu **toàn bộ database** vào thư mục `backups` cạnh database đang dùng. Mặc định là `data/attendance/backups/profiles_<UTC>_<random>.sqlite3`. Nếu sao lưu lỗi, thao tác bị hủy. Đường dẫn bản sao lưu hiện trong thông báo kết quả; bản sao cũng chứa embedding và lịch sử như database gốc. Chưa có nút tự khôi phục; khi cần khôi phục phải đóng app và giữ riêng database hiện tại trước khi thay bằng bản sao phù hợp.

## Thêm ảnh cho người đã đăng ký

1. Trong **Profiles…**, tự chọn đúng người; app không tự chọn hồ sơ thay bạn.
2. Bấm **Import another appearance…**. Nhập nhãn khác các nhóm hiện có, ví dụ `Without glasses`.
3. Bấm **Choose 3 photos…**, chọn đúng ba ảnh khác nhau của cùng người và cùng ngoại hình. Có thể giữ ảnh trong thư mục như `faces_db/Kiet/without_glasses`; app không tự quét/nhập cả thư mục.
4. Bấm **Check photos**. Đọc kết quả từng ảnh; nhấp đúp một dòng để xem đầy đủ lý do bị từ chối. Bước này chưa lưu mẫu.
5. Nếu cả ba ảnh hợp lệ, tự đối chiếu đúng người và quyền sử dụng ảnh rồi đánh dấu **I verified that all 3 photos…**. Bấm **Import checked photos** trong vòng 120 giây sau kiểm tra; quá hạn cần kiểm tra lại.
6. Đóng **Profiles…**, bấm **Start camera** và thử đeo/bỏ kính. Việc nhập thành công chưa bảo đảm mô hình luôn nhận đúng; cần kiểm tra thực tế riêng.

## Tạo người mới từ ảnh

Trong **Profiles…**, bấm **New person from photos…**, nhập tên kèm mã sinh viên và thực hiện chọn/kiểm tra/xác nhận ba ảnh tương tự. Ba mẫu tạo nhóm ban đầu **Original**. Nếu đã có hồ sơ, dùng **Import another appearance…** để tránh tạo cùng người thành hai danh tính. App chặn tên trùng và ảnh nhận thành hồ sơ khác hoặc kết quả ambiguous theo mô hình hiện tại; các kiểm tra này vẫn có sai số.

## Điều kiện ảnh và cách xử lý từ chối

| Kiểm tra | Yêu cầu / cách xử lý |
| --- | --- |
| Tệp | JPG/JPEG, PNG, BMP hoặc WebP; mỗi ảnh tối đa 50 MB và 50 megapixel |
| Khuôn mặt | Đúng một mặt được detector tìm thấy, vùng mặt ít nhất 80×80 sau khi ảnh lớn được thu nhỏ tối đa 1600×1600; dùng ảnh cận mặt rõ, hạn chế nền có người khác |
| Chất lượng | Laplacian variance ít nhất 20, độ sáng trung bình 20–235 trên crop; chọn ảnh rõ, đủ sáng và tránh phản chiếu kính |
| Chống giả mạo | Crop qua CNN PAD và ngưỡng đang cấu hình; nếu bị từ chối hãy dùng ảnh khác hoặc đăng ký bằng camera, không hạ ngưỡng để bỏ qua |
| Khác ảnh | Ba ảnh không được có cùng pixel sau xoay EXIF/chuyển RGB, kể cả đổi tên tệp |
| Nhất quán | Ba embedding phải đủ giống nhau theo ngưỡng identity hiện tại; cùng người nhưng góc/ngoại hình quá khác có thể bị từ chối |
| Hồ sơ | Chặn tên/nhãn trùng, tối đa bốn nhóm, kiểm tra lại hồ sơ và encoder khi lưu |

Nếu một ảnh hoặc kiểm tra cả nhóm không đạt, **không mẫu nào trong lượt đó được lưu**. Kết quả kiểm tra gắn với mã thao tác, token một lần và bản embedding đã kiểm tra trong RAM. Thay lựa chọn ảnh/hủy/đóng biểu mẫu làm yêu cầu cũ mất hiệu lực. Import ảnh cần detector **Haar**; chế độ **ROI** không xác nhận được số khuôn mặt nên nút nhập ảnh bị khóa.

## Liên hệ với Lab 7 và giới hạn

Luồng này giống Lab 7 ở việc dùng ảnh tham chiếu riêng cho từng người. Encoder identity đã huấn luyện tạo embedding rồi lưu vào SQLite; **không huấn luyện lại CNN, không thêm ảnh của bạn vào tập train gốc**. App chỉ đọc các tệp bạn chọn và lưu vector, không chép/lưu ảnh gốc hoặc video. Các appearance được so khớp riêng trong cùng người để giữ mẫu có/không kính.

Kiểm tra PAD trên ảnh tĩnh **không chứng minh người đang hiện diện**. Nhập ảnh chỉ quản lý hồ sơ, không tạo điểm danh. Việc người vận hành xác nhận danh tính cũng không phải kết quả mô hình tự nhận đúng. Camera vẫn phải qua liveness và identity để ghi Automatic; Manual review giữ nguồn riêng.

## Đã kiểm tra

- **148/148 kiểm thử** trong 62,329 giây, gồm 14 kiểm tra mới về quản lý hồ sơ/import: sao lưu trước thay đổi, giữ ID/lịch sử, từ chối nhóm cuối/sai người, không ghi một phần, ảnh trùng/không rõ/nhiều mặt/PAD fail, token/xác nhận/hết hạn, encoder không khớp và hồ sơ thay đổi giữa kiểm tra/lưu. Log: `results/app_upgrade_20261030/phase3_full_tests.log`.
- Đã xem cửa sổ hồ sơ và cả hai biểu mẫu nhập ảnh bằng **dữ liệu giả lập**, không mở camera/database thực. Lệnh: `.\.venv-gpu\Scripts\python.exe -m attendance_app.gui --preview-profiles`.
- Đã chạy `python -m tools.check_reference_import` với **encoder, PAD và detector thật**, ảnh LCC train và database tạm. Sáu nhóm đầu bị từ chối do chất lượng/PAD; nhóm thứ bảy nhập được. Đã kiểm tra tạo một người và thêm một appearance, tổng sáu mẫu, không điểm danh. Nhóm thứ hai cố ý lặp cùng ba ảnh để kiểm tra giao dịch; không thể dùng làm bằng chứng cải thiện có/không kính. Chi tiết: `results/app_upgrade_20261030/reference_import_integration.json`.
- Chưa nhập ảnh cá nhân của người dùng hoặc sửa database đang dùng. Mô hình/ngưỡng giữ nguyên. Chưa chạy đánh giá webcam có/không kính trong bước này; báo cáo Word và bước 4–5 chờ lượt tiếp theo.
