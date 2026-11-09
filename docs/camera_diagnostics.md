# Bước 4: Camera và Diagnostics

Hoàn tất ngày **30/10/2026**. Thêm lựa chọn camera khi dừng, thử một khung hình, chẩn đoán từng thành phần và hướng dẫn lấy mẫu rõ hơn. Giao diện tiếng Anh; mô hình, ngưỡng identity/PAD và quy tắc điểm danh giữ nguyên.

## Chọn và thử camera

1. Mở `start_attendance.cmd`. Camera không tự bật.
2. Khi camera đang tắt, chọn **Camera index**. Thử `0` trước; nếu có webcam khác, thử `1` hoặc chỉ số phù hợp. Đây là chỉ số OpenCV, không phải tên thiết bị; số có thể thay đổi khi cắm/rút camera. App không tự quét/mở mọi thiết bị.
3. Bấm **Test camera**. App mở đúng thiết bị đã chọn, đọc **một khung hình** rồi nhả thiết bị. Kết quả như `Camera 0 · Ready · 640×480` cho biết thiết bị trả được hình; không chứng minh mô hình nhận đúng người hay liveness đạt. Không hiện/lưu ảnh từ lượt thử, không gọi CNN hoặc database để điểm danh.
4. Bấm **Start camera** sau khi các mô hình sẵn sàng. Muốn chuyển thiết bị, bấm **Stop** trước; lựa chọn và nút thử bị khóa khi đang mở/chạy camera hoặc thử thiết bị. Quản lý Profiles cũng cần camera tắt.
5. Nếu lỗi, đọc thông báo hoặc mở **Diagnostics…**. Đóng ứng dụng khác đang giữ webcam, kiểm tra chỉ số và quyền camera Windows rồi thử lại. App không tự đổi quyền hệ thống.

Menu có chỉ số `0–9` và giá trị khởi động. Nếu cần chỉ số khác trong `0–99`, mở app bằng tham số, ví dụ:

```powershell
.\.venv-gpu\Scripts\python.exe -m attendance_app.gui --camera 10
```

Lựa chọn áp dụng cho lần chạy hiện tại; chưa lưu thành thiết lập cho lần mở sau. Test camera chạy ngoài luồng Tk để cửa sổ vẫn phản hồi. OpenCV/driver có thể chậm ở lệnh mở/đọc; chưa có bảo đảm timeout ở mức driver. Đóng app trong lúc thử sẽ chờ tác vụ nhả thiết bị.

## Đọc Diagnostics

Nút **Diagnostics…** dùng được cả lúc mô hình chưa sẵn sàng. Cửa sổ có thanh cuộn và hiển thị:

- Thiết bị suy luận, chỉ số/trạng thái camera và kết quả thử gần nhất.
- **Python / Torch / OpenCV**, **Identity encoder**, **Liveness model**, **Expression model**, **Face locator**, **Profile database**: `Not checked`, `Loading`, `Ready` hoặc `Error`.
- Thành phần lỗi, chi tiết đường dẫn/lý do và hướng xử lý. Các bước chưa chạy sau lỗi vẫn là `Not checked`, không bị ghi thành `Ready`.
- Trạng thái khung hình mới nhất, độ rõ, độ sáng và PAD score/ngưỡng nếu đã chạy đến bước đó. Đây là số chẩn đoán trong RAM, không phải xác suất nhận diện chính xác hay chứng nhận liveness.

**Retry model setup** chỉ bật sau lỗi khởi tạo và khi camera tắt. Khôi phục tệp ở đường dẫn đang cấu hình hoặc sửa JSON identity cho đúng rồi bấm Retry; không cần đóng cửa sổ app. Nếu muốn đổi chính tham số `--liveness-checkpoint`, `--emotion-checkpoint` hoặc `--device`, đóng/mở app với tham số đúng. App chờ worker cũ kết thúc trước khi tạo worker mới; không chạy hai worker mô hình/database cùng lúc. Không thay encoder của gallery đang có bằng một checkpoint khác chỉ để bỏ qua lỗi fingerprint.

## Hướng dẫn dưới khung camera

| Hiện trạng | Hướng xử lý |
| --- | --- |
| Không thấy mặt | Nhìn camera, để toàn bộ mặt gần giữa khung, thêm ánh sáng từ phía trước |
| Nhiều mặt | Chỉ một người; kiểm tra mặt trên poster/màn hình phía sau |
| Mặt quá nhỏ/ngoài khung | Tiến gần hơn, giữ trán và cằm trong hình |
| Ảnh tối | Thêm ánh sáng phía trước, tránh cửa sổ sáng phía sau |
| Ảnh quá sáng | Giảm chiếu trực tiếp/phản chiếu trên mặt và kính |
| Ảnh mờ | Giữ yên, lau ống kính, chờ camera lấy nét |
| Liveness Not passed | Đã thấy mặt nhưng PAD chưa đạt; dùng camera thật, chỉnh ánh sáng/góc để giảm chói kính |
| Unknown khi thay đổi kính | Thêm appearance vào cùng hồ sơ đã có; không đăng ký thành người thứ hai |
| Ambiguous | Mặt rõ hơn/góc nhìn thẳng; chưa coi một điểm gần giống là xác nhận danh tính |

**Not checked** nghĩa là chưa chạy đến bước đó, chẳng hạn chất lượng ảnh không đạt nên identity/PAD chưa được gọi. **Not passed** là PAD đã chạy và không đạt; **Unavailable** là thiếu ngưỡng được hiệu chỉnh. Không đổi ngưỡng để làm đẹp kết quả. Cách nhận diện vẫn cần ba khung hợp lệ liên tiếp để ghi Automatic; ảnh tĩnh/đối chiếu thủ công giữ luồng riêng.

Ngưỡng chất lượng vẫn là mặt ít nhất 80×80, Laplacian variance ít nhất 20 và độ sáng trung bình 20–235. Chỉ bổ sung nguyên nhân và số chẩn đoán; các CNN tiếp tục nhận cùng crop RGB gốc.

## Đã kiểm tra và giới hạn

- **159/159 kiểm thử trong 57,702 giây**, gồm 11 kiểm tra mới về chỉ số thiết bị, mở/đọc/nhả camera lỗi, frame không hợp lệ, không trả/lưu pixel từ probe, nguyên nhân chất lượng, giữ nguyên crop và gán đúng lỗi từng thành phần. Log: `results/app_upgrade_20261030/phase4_full_tests.log`.
- Đã xem giao diện chính và Diagnostics trên máy hiện tại; chỉnh bố cục để vùng kết quả và bảng điểm danh còn hiển thị sau khi thêm thanh thiết bị. Lệnh preview: `python -m attendance_app.gui --preview-diagnostics`; không nạp mô hình/database/camera.
- Đã mở app thật, nạp ba CNN cùng detector/database hiện có. Nút **Test camera** thực tế trên camera `0` trả **640×480**, nhả thiết bị và bật lại các nút. Không bấm Start camera, không đăng ký hay ghi điểm danh trong lượt này.
- Đã tạo cấu hình identity thiếu checkpoint trong `tmp`, kiểm tra app báo **Identity Error**, chưa tạo database; sửa riêng bản cấu hình tạm rồi bấm **Retry model setup**, tất cả thành phần trở lại **Ready**. Dữ liệu người dùng và checkpoint gốc không được thay trong thử lỗi này.
- Chưa xác nhận webcam thứ hai, rút cáp giữa buổi hoặc mọi driver Windows bằng thiết bị thật; các đường lỗi này được kiểm tra bằng mock. Lượt probe không phải bài đánh giá nhận diện/PAD với người thật có/không kính.

Mốc kết quả ở `results/app_upgrade_20261030/phase4_status.json`. Dừng sau bước 4 để người dùng xem; đánh giá thực tế và đồng bộ báo cáo Word thuộc bước 5.
