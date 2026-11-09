# Bước 5: Kết quả đánh giá hiện có

Cập nhật **30/10/2026**. Chỉ ghi kết quả từ JSON `completed`. Đã có **một lượt đeo kính và một lượt ảnh tĩnh trên điện thoại**, với nhãn do người tham gia xác nhận; mỗi điều kiện có n=1 lượt. Các khung liên tiếp tương quan, không phải ước lượng accuracy/FAR/FRR độc lập. Các tình huống còn chờ không được tính là đạt hoặc không có lỗi.

| Điều kiện | Lượt hoàn tất | Khung hình | Một mặt | PAD đạt / đã chấm | Identity đúng / đã chạy | Identity khớp ảnh giả / đã chạy | Điểm danh đúng tạm | Điểm danh ảnh giả tạm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Đeo kính | 1 | 312 | 243 | 232 / 243 | 175 / 232 | Không áp dụng | 1 | Không áp dụng |
| Ảnh tĩnh trên điện thoại | 1 | 431 | 240 | 237 / 237 | Không áp dụng | 0 / 237 | Không áp dụng | 0 |
| Không kính | Chưa sẵn sàng | — | — | — | — | Không áp dụng | — | Không áp dụng |
| Người chưa đăng ký | Chưa thực hiện | — | — | — | Không áp dụng | Không áp dụng | — | Không áp dụng |
| Video ghi sẵn trên màn hình | Chưa thực hiện | — | — | — | Không áp dụng | — | Không áp dụng | — |
| Ảnh in | Chưa thực hiện | — | — | — | Không áp dụng | — | Không áp dụng | — |

"Đúng danh tính" chỉ dùng cho người thật đã xác nhận hồ sơ. Với ảnh giả, cần đọc riêng việc PAD cho qua, identity nhận khớp và điểm danh chấp nhận; không gọi nhận đúng ảnh giả là thành công nhận diện.

## Lượt đeo kính

Người tham gia xác nhận đang đeo kính và hồ sơ `kiet` là của mình. Gallery lúc thử chỉ có **một người, một appearance**. Dùng camera 0, CPU bốn thread, yêu cầu 960×720 nhưng thiết bị trả 960×540. Đo 15 giây, loại tám khung warm-up khỏi mẫu số. Không thay ngưỡng hoặc gallery giữa lượt.

- Tổng 312 khung: detector báo một mặt ở 243 khung, không thấy mặt ở 27 khung và nhiều mặt ở 42 khung. Không lưu hình để xác định nguyên nhân; không suy ra có nhiều người thực sự từ riêng thông báo detector.
- Cả 243 crop một mặt qua chất lượng; PAD đạt 232, không đạt 11. Tỷ lệ PAD đạt là **95,47% trên 243 crop được chấm**, hoặc **74,36% trên toàn bộ 312 khung**.
- Identity chạy ở 232 crop qua PAD: nhận đúng hồ sơ 175 lần, unknown 57 lần, ambiguous 0 và nhận thành hồ sơ khác 0. Tỷ lệ đúng là **75,43% trên 232 probe identity**, hoặc **56,09% trên toàn bộ 312 khung**. Đây là tỷ lệ frame tương quan của lượt này, không phải accuracy triển khai.
- Quy tắc tự động tạo một bản ghi đúng trong database tạm; lần ghi đầu ở khoảng **2,812 giây**. Không có bản ghi cho người khác. Thời gian này thuộc một lượt, không phải thời gian đáp ứng bảo đảm.
- Expression được suy luận ở 232 crop, nhưng không có nhãn biểu cảm từ người chấm nên không tính accuracy biểu cảm webcam.

Một gallery một người không chứng minh khả năng phân biệt nhiều người hoặc an toàn trước kẻ giả mạo. Ánh sáng và khoảng cách chưa được đo/kiểm soát; chưa có đối chứng không kính hoặc người lạ. Lượt ảnh điện thoại bên dưới chỉ là một vật mẫu tấn công. Không so lượt đeo kính với các lần webcam cũ để kết luận mức cải thiện do một thay đổi cụ thể.

Nguồn bất biến: [glasses_01.json](../results/app_upgrade_20261030/evaluation/trials/glasses_01.json). Các tệp lưu số đếm, mẫu số, trạng thái, kích thước khung và fingerprint/ngưỡng mô hình; không chứa ảnh/video hoặc embedding webcam. Công cụ chỉ mở database gốc bằng SQLite `mode=ro`; mọi ghi nhận tự động nằm trong bản sao tạm đã được dọn.

## Lượt ảnh tĩnh trên điện thoại

Người dùng xác nhận trình chiếu **ảnh tĩnh của chính mình, tương ứng hồ sơ `kiet`, trên điện thoại**, và được hướng dẫn giữ mặt thật ngoài khung. Trường `presentation_type` là `phone-still-photo`; tên điều kiện kỹ thuật `screen-replay` trong JSON không có nghĩa đã thử video ghi sẵn. Gallery vẫn chỉ có **một người, một appearance**; cùng mô hình/ngưỡng và pipeline app. Camera 0 trả 960×540; thời gian đo **15,016 giây**, không tính tám khung warm-up.

- Tổng **431 khung**: detector báo một mặt ở 240 khung, không thấy mặt ở 190 và nhiều mặt ở 1. Ba khung một mặt bị loại do mờ/chói; còn **237 crop** đạt chất lượng và được PAD chấm. Các khung chưa chấm PAD không được tính thành PAD chặn thành công.
- **PAD cho qua cả 237/237 crop ảnh giả**, không từ chối crop nào: **100% trên các crop được chấm**, hoặc **54,99% trên toàn bộ 431 khung**. Đây là kết quả frame của một lần trình chiếu, không phải APCER độc lập trên nhiều vật mẫu. **PAD đã không chặn được mẫu tấn công này.**
- Identity chạy trên 237 crop: **unknown 237, khớp hồ sơ đích 0, khớp hồ sơ khác 0, ambiguous 0**. Không có bản ghi điểm danh trong bản sao tạm; số chấp nhận ảnh giả ở luồng điểm danh là **0 trong một lượt thử**.
- Kết quả cuối bị chặn bởi **identity**, không phải PAD. Không thể dùng việc không điểm danh để kết luận liveness đã phân biệt được người thật và ảnh trên điện thoại. Ngược lại, việc PAD cho qua mẫu này là giới hạn cần giữ rõ trong báo cáo và khi trình diễn.
- Expression được suy luận ở 237 crop, nhưng không dùng cho quyết định điểm danh hoặc tính accuracy webcam. Không có ảnh/video được lưu và không sửa gallery hay bản ghi điểm danh gốc.

Nguồn bất biến: [phone_photo_01.json](../results/app_upgrade_20261030/evaluation/trials/phone_photo_01.json). Chưa thử video ghi sẵn, ảnh in hay người chưa đăng ký; chưa thể suy ra tỷ lệ chấp nhận tấn công trong thực tế từ n=1. Lượt không kính hiện chờ người dùng sẵn sàng.

## Kiểm thử phần mềm và tài liệu

Lượt toàn bộ đạt **171/171 kiểm thử trong 72,486 giây**, gồm 12 kiểm tra mới cho nhãn thực nghiệm, mẫu số, số không áp dụng là null, snapshot không sửa nguồn và nhả camera khi lỗi. Đây là kiểm tra hành vi phần mềm, không phải 171 người hoặc 171 lượt webcam.

Báo cáo tiếng Anh đã bổ sung cả lượt đeo kính và ảnh tĩnh trên điện thoại, kết xuất thành **12 trang** và hoàn tất kiểm tra trực quan toàn bộ trang, giữ trống thông tin nhóm. Abstract và Discussion nêu rõ PAD không chặn được mẫu ảnh điện thoại dù identity đã ngăn điểm danh. Ảnh QA cuối ở `tmp/group_report/render_phase5_phone_final`; kết quả QA và SHA-256 do `results/app_upgrade_20261030/phase5_report_status.json` ghi nhận. Báo cáo tách PAD baseline/regression trên test đã dùng lại, kết quả từng gate webcam và các điều kiện còn thiếu; chưa có bản sửa mô hình chống giả mạo trong bước này. [Quy trình thử tiếp](phase5_evaluation_protocol.md) và [checklist video nhóm](09_video_checklist.md) đã chuẩn bị; chưa quay/nộp video hoặc viết thay báo cáo cá nhân cho từng thành viên.
