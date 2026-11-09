# Phần 7: Đối chiếu thủ công có lưu vết

Ngày thực hiện: **27/10/2026**. Tính năng được tích hợp vào GUI hiện có, không cần tải dataset hoặc huấn luyện thêm. Mở ứng dụng bằng `start_attendance.cmd` như phần 6.

Giao diện hiện dùng tiếng Anh; tên nút bên dưới đã được cập nhật cho khớp. Xem [cập nhật bắt mặt và giao diện](app_capture_update.md) về thay đổi detector, lượt kiểm thử mới và giới hạn thử webcam.

## Yêu cầu của đề và tính năng được chọn

Mục 7, trang 4 của đề khuyến khích mỗi sinh viên triển khai một tính năng sáng tạo liên quan đến hệ thống điểm danh khuôn mặt, có đóng góp đáng kể. Tính năng cần được trình diễn rõ trong video và giải thích trong báo cáo cá nhân.

Tính năng được chọn cho lượt này là **đối chiếu thủ công có lưu vết khi nhận diện chưa chắc chắn**. Đây là đề xuất triển khai cho bạn xem; chưa gán tính năng hoặc tên tác giả cho thành viên nào của nhóm. Nhóm vẫn cần phân công đóng góp thực tế của từng người khi làm báo cáo.

## Vấn đề được giải quyết

Baseline triplet có AUC validation 0,859969 nhưng tại ngưỡng FPR cặp gần 1%, TPR chỉ 9,73%. Hệ thống có thể không nhận ra người đã đăng ký. Hạ ngưỡng cho dễ nhận ra có thể tăng nhận nhầm; một luồng đối chiếu của người vận hành cho phép xử lý trường hợp này mà vẫn ghi rõ nguồn quyết định.

Khi khuôn mặt có **liveness đạt** và identity là **unknown hoặc ambiguous**, người vận hành có thể xem crop, tự chọn hồ sơ và ghi nhận sau khi đã đối chiếu trực tiếp. Bản ghi giữ nguyên điểm cosine, lưu nguồn `manual_review`, tên người đối chiếu và lý do. Không đổi kết quả dự đoán của CNN thành “đúng”, không huấn luyện lại từ quyết định thủ công, không cộng quyết định này vào accuracy/AUC của mô hình.

## Cách dùng trong GUI

1. Đã có ít nhất một hồ sơ đăng ký từ phần 6. Nhập buổi học vào **Session** và bấm **Start camera**.
2. Để một người trước camera, đủ sáng và rõ. Khi **Liveness** hiển thị **Passed** nhưng tên chưa nhận ra hoặc kết quả còn mơ hồ, nút **Manual review** được bật.
3. Bấm nút này. Cửa sổ cho thấy crop đã quan sát; ảnh chỉ giữ tạm trong RAM. **Không hồ sơ nào được chọn sẵn.**
4. Giữ nguyên người trước camera. Người vận hành đối chiếu trực tiếp, tự chọn hồ sơ ở **Profile you checked**, nhập tên mình ở **Reviewer name** và lý do ở **Reason for confirmation**, rồi đánh dấu **I have checked the person against the selected profile in person.**
5. Bấm **Record manual attendance** trong thời hạn 30 giây. Nếu không đủ căn cứ, bấm **Cancel**; không cần chọn tên để cho qua.
6. Bảng điểm danh có cột **Source**, hiển thị **Automatic** hoặc **Manual**. Nhấp đúp một dòng để xem người đối chiếu và lý do. Mỗi người vẫn chỉ có một dòng mỗi buổi.

Nếu một bản ghi đã được xác nhận tự động, duyệt thủ công không ghi đè hay đổi nhãn nguồn của dòng đó. Hồ sơ mới chưa có trong danh sách phải đăng ký; đối chiếu thủ công không tự tạo hồ sơ hoặc embedding.

Để xem riêng bố cục cửa sổ mà không mở camera, nạp mô hình hay ghi database:

```powershell
.\.venv-gpu\Scripts\python.exe -m attendance_app.gui --preview-review
```

Chế độ này dùng ảnh placeholder và hồ sơ mô phỏng. Nút ghi nhận không lưu dữ liệu. Dữ liệu mô phỏng không được báo cáo như một lần điểm danh thật.

## Điều kiện của một lần đối chiếu

| Điều kiện | Cách xử lý |
| --- | --- |
| Liveness chưa đạt hoặc thiếu ngưỡng | Không mở/không xác nhận yêu cầu |
| Không có hồ sơ, không có quan sát mới, hoặc identity đã chắc chắn | Không mở yêu cầu thủ công |
| Yêu cầu quá 30 giây | Vô hiệu hóa; phải hủy và lấy lại quan sát mới |
| Khung hợp lệ mới nhất quá 1,5 giây | Không ghi nhận |
| Mất mặt, nhiều mặt, ảnh không đạt chất lượng, hoặc liveness không đạt trong lúc đối chiếu | Hủy tính hợp lệ của bằng chứng |
| Đặc trưng khuôn mặt đổi vượt ngưỡng liên tục của encoder | Vô hiệu hóa yêu cầu |
| Buổi học thay đổi hoặc mã yêu cầu không khớp/đã dùng | Không ghi nhận |
| Chưa tự chọn hồ sơ, thiếu tên/lý do, chưa đánh dấu đối chiếu | Yêu cầu hoàn thiện; không tự xác nhận |
| Cửa sổ đối chiếu vẫn mở, kể cả đã hết hạn | Tạm dừng ghi nhận tự động để tránh đua giữa hai luồng quyết định |

Sau khi hủy/hoàn tất, luồng tự động bắt đầu lại chuỗi khung liên tiếp. Khi dừng camera/đóng ứng dụng, crop và yêu cầu đang đối chiếu được giải phóng.

Kiểm tra “cùng khuôn mặt qua các khung” dùng chính encoder hiện có, nên vẫn có giới hạn chất lượng của mô hình. Nó không thay thế người đối chiếu và không chứng nhận chống thay người/giả mạo trong thực tế.

## Lưu dữ liệu và khả năng truy vết

`attendance_audit` là bảng bổ sung trong cùng database với `attendance`. Khi ghi nhận, cả dòng điểm danh và nguồn xác nhận được ghi trong **một transaction**; lỗi ở bước audit làm rollback dòng điểm danh. Lưu:

- Buổi học, ID hồ sơ, nguồn `automatic` hoặc `manual_review`.
- Tên người đối chiếu/lý do cho nguồn thủ công; không tự điền cho nguồn tự động.
- Mã yêu cầu duy nhất, thời điểm crop được quan sát và thời điểm ghi nhận, có múi giờ.
- Điểm cosine và liveness được giữ trong dòng attendance; quyết định thủ công không nâng điểm lên 1.

Không lưu ảnh/video, crop, biểu cảm hoặc embedding bổ sung từ lần duyệt. Bản ghi cũ trước khi có audit được giữ nguyên và hiển thị **Unknown** ở cột Source, không tự gắn tên người duyệt hoặc giả định có bằng chứng. Khi mở database khác encoder, kiểm tra fingerprint diễn ra trước khi thêm bảng mới.

Tên người vận hành được nhập tay trong bản demo; **chưa có đăng nhập/phân quyền hay nhật ký chống sửa**. Không mô tả tính năng như một hệ thống phê duyệt quản trị đã bảo mật hoàn chỉnh. Database mặc định nằm trong thư mục OneDrive của dự án, có thể chịu đồng bộ OneDrive như đã nêu ở phần 6; dùng `--database` nếu muốn chọn nơi khác.

## Kiểm tra đã thực hiện

`tests/test_part7.py` có **13 bài kiểm tra**: điểm dưới ngưỡng vẫn được giữ nguyên khi duyệt; bắt buộc lựa chọn/tên/lý do/xác nhận; không duyệt spoof hoặc thiếu ngưỡng; hết hạn/khung cũ; đổi người/buổi; mất mặt/nhiều mặt; chống dùng lại mã; tạm dừng auto khi đang duyệt; không đổi nhãn bản ghi tự động; không đăng ký và duyệt đồng thời; rollback khi audit lỗi; giữ dữ liệu và nguồn không xác định của phiên bản cũ.

Nhóm 14 kiểm tra phần 6 cũng đã chạy lại đạt. Nhật ký tổng thể được ghi ở `results/section7_full_tests.log`; kết quả tích hợp mô hình thật ở `results/manual_review_integration.json`.

Ở mốc phần 7, lượt toàn bộ đạt **92/92 bài trong 50,0 giây**. Trạng thái lịch sử được lưu ở `results/section7_status.json`. Sau cập nhật bắt mặt/tiếng Anh, lượt mới đạt **97/97 bài trong 70,1 giây**, nhật ký `results/app_check_20261027/english_detector_tests.log`. Những kiểm tra này xác nhận hành vi phần mềm; không thay cho thử đầy đủ với webcam/người vận hành thật.

Đã kiểm tra trực quan cửa sổ chính và hộp đối chiếu bằng chế độ preview với placeholder/hồ sơ mô phỏng. Bố cục hiển thị đủ lựa chọn, tên/lý do, đồng hồ, nút xác nhận và cột Source; không mở camera hoặc ghi bản điểm danh thật trong bước kiểm tra preview. Sau đó đã mở webcam và quan sát bắt mặt/liveness; chưa xác nhận trọn quy trình đăng ký và đối chiếu của người vận hành thật.

Trong kiểm tra tích hợp thật, dùng ảnh train của ID `n000003` và ba CNN đã có. Mô hình trả **unknown**, điểm cosine với hồ sơ đã đăng ký là **0,842151**, dưới ngưỡng **0,869540**. Thao tác đối chiếu được **mô phỏng bằng nhãn dataset**, lưu được nguồn `manual_review`, tên/lý do của tác nhân kiểm tra, trong khi dự đoán mô hình sau đó vẫn **unknown**. Ca liveness không đạt bị chặn. SHA256 của cả ba checkpoint không thay đổi. Worker thật nạp đủ ba mô hình và đạt trạng thái sẵn sàng mà camera vẫn tắt.

Đây là kiểm tra đường đi phần mềm trên ảnh train, **không phải thử nghiệm người vận hành thật, webcam, FAR/FRR hoặc độ chính xác nhận diện**. Chưa ghi nhận sự cải thiện định lượng nào về AUC/TPR hoặc hiệu quả của người duyệt. Cần thử thực tế với người đã/chưa đăng ký và người vận hành trước khi báo cáo hiệu quả ngoài kiểm tra phần mềm.

## Gợi ý trình diễn và giải thích cá nhân

Một đoạn demo khoảng 60–90 giây có thể gồm: chỉ ra một người đã đăng ký nhưng được nhận diện chưa chắc chắn; mở đối chiếu; tự chọn đúng hồ sơ, nhập tên/lý do và xác nhận; xem cột Nguồn và chi tiết; sau đó minh họa yêu cầu bị vô hiệu khi mất khuôn mặt hoặc hết hạn. Chỉ quay kết quả thật; nếu mô hình nhận ra chắc chắn thì chọn một trường hợp kiểm thử phù hợp, không sửa ngưỡng hoặc giả kết quả.

Trong báo cáo cá nhân, giải thích vấn đề recall thấp, cách giữ quyết định tự động/thủ công tách biệt, các điều kiện hủy, transaction audit, kết quả kiểm tra và giới hạn. Bổ sung công việc thực tế của người thực hiện; không dùng số bản ghi duyệt thủ công để tuyên bố CNN chính xác hơn.

## Mã nguồn

| File | Vai trò |
| --- | --- |
| attendance_app/review.py | Quan sát mới, yêu cầu 30 giây, ràng buộc khuôn mặt/buổi, xác nhận/hủy |
| attendance_app/review_dialog.py | Crop trong RAM, lựa chọn hồ sơ trống mặc định, tên/lý do, đồng hồ và xác nhận |
| attendance_app/service.py | Kết nối liveness/identity với duyệt, giữ auto tạm dừng khi đang đối chiếu |
| attendance_app/store.py | Audit cùng transaction, đọc nguồn/actor/lý do, tương thích dữ liệu cũ |
| attendance_app/gui.py | Nút đối chiếu, cột Nguồn, xem chi tiết và chế độ preview |
| tools/check_manual_review.py | Kiểm tra tích hợp ba CNN thật, reviewer mô phỏng, database tạm |

Mốc phần 7 là triển khai và kiểm tra tính năng đối chiếu. Sau đó đã hoàn thành báo cáo nhóm phần 8 và cập nhật bắt mặt/giao diện tiếng Anh; video phần 9 chưa bắt đầu.
