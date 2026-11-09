# Phần 9: Checklist video nhóm

Chuẩn bị ngày **30/10/2026**, căn cứ mục 7–11 của đề. Đây là kế hoạch quay/kiểm tra; **chưa khẳng định video đã được quay hoặc nộp**. Thông tin nhóm/thành viên vẫn để bổ sung theo yêu cầu; chưa gán tác giả hoặc đóng góp cho bất kỳ ai.

## Các điều kiện trong đề

- [ ] Video dài **5–10 phút**, trình diễn toàn bộ chức năng của dự án.
- [ ] Mỗi thành viên có thời lượng trình bày tương đương; **khuôn mặt người đang nói phải nhìn rõ**.
- [ ] Quay **một lần liên tục**, không cắt, ghép, biên tập hoặc hậu kỳ. Tập trước rồi quay lại toàn bộ lượt nếu cần; không ráp các đoạn thành một video.
- [ ] Nhóm tự đưa video lên nền tảng phù hợp như YouTube/OneDrive và nộp **chỉ đường dẫn** qua Canvas; liên kết phải mở được cho người chấm.
- [ ] Mỗi tính năng sáng tạo được chọn phải liên quan đến hệ thống điểm danh, được trình diễn rõ và giải thích trong báo cáo cá nhân của người thực sự đóng góp.

Đề yêu cầu người xem truy cập được video, không quy định phải công khai cho toàn Internet. Nhóm tự chọn quyền truy cập phù hợp và kiểm tra bằng tài khoản/trình duyệt không phải chủ sở hữu. Không tự tải lên, đổi quyền chia sẻ hoặc nộp Canvas bằng checklist này.

## Trước khi bấm quay

- [ ] Điền số nhóm, họ tên/mã sinh viên và phân công **đóng góp thực tế**, thống nhất người nói phần nào. Chia tổng thời gian nói cho số thành viên; để trống mục đóng góp chưa được xác nhận.
- [ ] App nạp đủ mô hình, **Diagnostics…** không còn lỗi khởi tạo; đã thử đúng **Camera index**. Đọc [hướng dẫn camera](camera_diagnostics.md).
- [ ] Dùng database/session demo riêng để việc đăng ký, thêm appearance hoặc điểm danh khi quay không sửa buổi học đang dùng. Tham số `--database` và **Session** cho phép tách buổi demo; chuẩn bị đúng hồ sơ trước hoặc đăng ký rõ trong video.
- [ ] Kiểm tra màn hình/chữ đủ đọc và âm thanh rõ. Tắt nội dung riêng không liên quan trong khung quay. Chỉ dùng ảnh/vật mẫu đã được người liên quan đồng ý.
- [ ] Bố trí để vừa thấy app vừa thấy mặt người nói. Nếu phần mềm quay và app tranh cùng webcam, thử trước; có thể dùng điện thoại/camera sẵn có để quay màn hình và người nói trong cùng một lượt. Không cần mua thêm thiết bị chỉ để thử một cách bố trí.
- [ ] Chuẩn bị người đã đăng ký, người chưa đăng ký và ảnh giả mạo đang có. Ghi đúng ảnh tĩnh trên điện thoại hay video phát lại, ảnh in màu hay đen trắng. Điều kiện chưa thử thì nói là chưa thử.
- [ ] Có bảng kết quả thật và đường dẫn bằng chứng: identity ROC/AUC, PAD baseline/current với phạm vi đánh giá, FER2013, kiểm thử phần mềm và lượt webcam đã hoàn tất. Không lấy preview hoặc dữ liệu mô phỏng làm bằng chứng người thật.
- [ ] Tập một lượt khoảng 7–9 phút để có khoảng đệm. Đây là gợi ý tổ chức, không thay giới hạn 5–10 phút của đề.

Ví dụ mở app với database mới dành cho quay, tại `grp_project`:

```powershell
.\.venv-gpu\Scripts\python.exe -m attendance_app.gui --database data/attendance/demo_recording.sqlite3 --session COS30082-video-demo
```

Database mới chưa có hồ sơ; cần đăng ký mẫu hợp lệ trước hoặc trong phần trình diễn. Không chép đè database đang dùng. Video nộp bài có ghi hình do nhóm chủ động thực hiện; điều này khác với công cụ đánh giá chỉ lưu số liệu và không lưu ảnh/video.

## Kịch bản tham khảo khoảng 8 phút

Các mốc là gợi ý cho nội dung. Sau khi biết số thành viên, chia lại để thời gian nói tương đương; không tự gán tên người trình bày.

| Khoảng thời gian | Nội dung có thể trình bày |
| --- | --- |
| 0:00–0:45 | Nhóm, mục tiêu và luồng webcam → định vị/chất lượng → PAD → identity/expression → điểm danh |
| 0:45–2:00 | Hai cách huấn luyện identity, cosine/Euclidean, ROC/AUC và lý do chọn triplet; nêu trùng dữ liệu và TPR thấp ở ngưỡng nghiêm |
| 2:00–3:30 | App thật: **Start camera**, tên người đã đăng ký, **Liveness**, **Expression**; đăng ký người mới bằng **Register from camera**, giữ đủ ba mẫu và chỉ ra một dòng mỗi người/buổi |
| 3:30–4:30 | PAD với vật mẫu thực tế đang có; nêu đúng loại tấn công và kết quả quan sát, gồm cả thất bại. Nêu phương pháp train PAD và biểu cảm |
| 4:30–6:00 | Tính năng sáng tạo/phần mở rộng có đóng góp thực tế: **Add appearance** có/không kính, **Manual review** khi đủ điều kiện; chỉ rõ nguồn **Automatic/Manual**, không gọi lựa chọn thủ công là mô hình tự nhận đúng |
| 6:00–7:15 | **Profiles…** xem nhóm mẫu/nhập ba ảnh, **History / CSV**, **Camera index / Test camera / Diagnostics…**; chỉ trình diễn thao tác có thật, tôn trọng các nút bị khóa và xác nhận |
| 7:15–8:00 | Kết quả webcam đã đo, điều kiện còn pending, giới hạn, đóng góp của từng người và kết luận |

Nếu thao tác nhận diện/liveness không đạt trong lượt quay, mô tả đúng tình huống và cách xử lý. Không dùng clip cũ, hạ ngưỡng, chọn tay tên rồi mô tả như nhận diện Automatic, hoặc thay trạng thái trên màn hình để tạo kết quả mong muốn. **Manual review** chỉ xuất hiện khi có mặt live hợp lệ nhưng identity unknown/ambiguous; không ép tạo tình huống bằng cách bỏ các điều kiện của app.

## Kiểm tra bản quay nguyên vẹn

- [ ] Thời lượng nằm trong 5–10 phút; không có đoạn dừng/ghép/hậu kỳ.
- [ ] Đủ thời lượng và khuôn mặt từng người nói; xem lại cả phần âm thanh.
- [ ] Người xem đọc được tên nút, kết quả và phân biệt bản chạy thật/preview, automatic/manual, dữ liệu train/phép thử webcam.
- [ ] Không tuyên bố các frame tương quan là số lượt độc lập, 100% accuracy, FAR/FRR bảo đảm hoặc chống mọi ảnh giả.
- [ ] Thông tin nói trong video khớp báo cáo cuối và mô hình đang dùng. Haar có sẵn được tách rõ khỏi ba CNN do nhóm huấn luyện.
- [ ] Nếu phải quay lại, quay lại một lượt hoàn chỉnh; giữ riêng các lần tập, không ghép vào bản nộp.
- [ ] Nhóm tự kiểm tra liên kết truy cập sau khi tải lên và nộp đúng nơi trong Canvas.

## Các đầu việc nộp khác

- [ ] **Một thành viên đại diện** nộp source code, group report và liên kết video.
- [ ] Group report **không quá 12 trang**, có Methodology, Results và Discussion. Sau khi điền thông tin nhóm hoặc cập nhật kết quả, kiểm tra lại số trang của bản cuối.
- [ ] **Mỗi sinh viên** nộp individual report **không quá 5 trang**: đóng góp, phương pháp, thách thức/cách giải quyết; tính năng sáng tạo nếu có phải đúng đóng góp thực tế.
- [ ] **Mỗi sinh viên** nộp peer evaluation form.
- [ ] Chuẩn bị giải thích mã, loss, dữ liệu, chọn mô hình/ngưỡng, sai số và đóng góp trong **phỏng vấn tutorial tuần 12**. Đề không ghi ngày nộp cụ thể; xem mốc được công bố trong môn/Canvas.

Checklist này không chứng minh đã quay, tải lên hoặc hoàn tất các đầu việc cá nhân. Duy trì trạng thái còn chờ thay vì tự điền tên thành viên, đóng góp hay kết quả chưa được xác nhận.
