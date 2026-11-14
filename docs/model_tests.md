# Kiểm thử và so sánh mô hình trong app

Mở app như bình thường bằng `start_attendance.cmd`, dừng camera điểm danh rồi bấm
**Model tests…**, cạnh **Diagnostics…**. Màn hình này so sánh **hai model chống giả
mạo (PAD)**, không đánh giá danh tính hay biểu cảm và không tự đổi model mặc định.

## Chạy một lượt thử

1. Chọn **Camera index** ở màn hình chính trước khi mở Model tests. Khi dialog
   đang mở, camera điểm danh, Test camera và thao tác quản lý hồ sơ bị khóa để
   không tranh thiết bị với công cụ kiểm thử. History / CSV vẫn xem được.
2. **App baseline** là checkpoint liveness đang cấu hình cho app. Bấm **Browse…**
   chọn **Candidate** là checkpoint thử nghiệm của dự án. Cả hai phải là model đã
   hoàn tất huấn luyện từ đầu, có ngưỡng hiệu chỉnh bằng validation. Model thiếu
   điều kiện này bị từ chối trước khi camera mở.
3. Chọn đúng kịch bản: người thật đeo/không đeo kính, ảnh tĩnh trên điện thoại,
   video ghi sẵn hoặc ảnh in. Với tấn công, để mặt thật ngoài khung. Nhãn kịch bản
   do người vận hành cung cấp; phần mềm không tự chứng minh nhãn này là đúng.
4. Chọn thời gian đo, mặc định 15 giây, tối đa 120 giây. Chỉ đánh dấu xác nhận
   khi người tham gia đồng ý, đã sẵn sàng và vật liệu trình diễn đúng kịch bản.
   Bấm **Start comparison** để mở camera. Có 10 giây căn khung không tính điểm,
   sau đó mới bắt đầu đo.
5. Xem preview riêng của phép thử. Hai model nhận cùng crop RGB gốc sau các điều
   kiện một mặt/kích thước/chất lượng của app. Ngưỡng giữ nguyên; không có tối ưu
   hoặc hiệu chỉnh từ lượt camera. Bấm **Cancel test**, Q/Esc trong preview hoặc
   đóng dialog để hủy. Khi đang chạy, việc đóng chờ tiến trình nhả camera; một
   lượt đo dở được ghi rõ **Cancelled / Partial trial**. Hủy trong lúc tải model
   không mở camera và không tạo kết quả đo.

## Đọc kết quả

- **Captured**: số khung trong khoảng đo; không gồm các khung căn chỉnh.
- **One face**: khung có một phát hiện mặt, kể cả mặt quá nhỏ hoặc chất lượng kém.
- **Scored crops**: crop qua điều kiện đầu vào và được cả hai model chấm.
- **Accepted / Rejected**: model cho qua là live / không cho qua.
- **Correct / Wrong**: đối chiếu quyết định với kịch bản người vận hành đã chọn.
  Với người thật, accepted là đúng; với ảnh/video giả, rejected là đúng.
- **Accepted %**: accepted chia cho scored crops. Không có crop hợp lệ thì hiển
  thị dấu gạch, không tự cho kết quả 0% hay 100%.

No-face, nhiều mặt, mặt quá nhỏ và chất lượng kém không được coi là model PAD
đã từ chối ảnh giả. Các khung trong một lượt có tương quan; không báo chúng là
những người hoặc những lượt tấn công độc lập. Không suy ra AUC, ACER hay khả năng
bảo vệ quần thể từ một kịch bản ngắn. Tính năng này hỗ trợ đo lỗi, không tự khắc
phục giới hạn chống ảnh điện thoại của model hiện tại.

## Báo cáo và dữ liệu

Mỗi lượt tự lưu JSON mới dưới `results/model_tests/<UTC-timestamp>_<id>.json`.
**Open report…** đọc được báo cáo paired PAD do công cụ
`python -m tools.check_liveness_camera` tạo trước đây, gồm các lượt trong
`results/pad_webcam_20261109/`. Đây là ảnh chụp kết quả lịch sử; không tự mở các
checkpoint khi xem báo cáo.
Nếu báo cáo cũ chưa có cờ hủy, giao diện ghi **Cancellation not recorded** và CSV
ghi `not-recorded`; không suy đoán rằng lượt đó đã hoàn tất.

**Export JSON…** giữ nguyên báo cáo; **Export CSV…** xuất hai dòng baseline và
candidate, kèm ngưỡng, SHA256, kịch bản, số khung, trạng thái hủy và phạm vi phép
đo. Chọn tên file mới: xuất báo cáo không ghi đè bằng chứng đã có. Trường văn bản
CSV được xử lý để không chạy công thức khi mở trong spreadsheet.

Công cụ không lưu ảnh/video, không đọc gallery và không ghi điểm danh. Các đường
dẫn checkpoint, hash và số liệu tổng hợp có trong báo cáo. App không thay ngưỡng,
checkpoint, tên buổi hay dữ liệu cá nhân từ thao tác này. Nếu thư mục dự án đang
được OneDrive đồng bộ, các báo cáo cũng theo cấu hình đồng bộ đó.

## Xem bố cục không dùng dữ liệu hoặc camera

```powershell
.\.venv-gpu\Scripts\python.exe -m attendance_app.gui --preview-model-tests
```

Preview khóa chọn file, chạy thử và xuất báo cáo; không tải model, đọc gallery hay
mở camera. Kiểm thử tự động dùng frame/model giả lập để kiểm tra logic, không
được tính là kết quả nhận diện hoặc chống giả mạo thật.
