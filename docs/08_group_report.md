# Phần 8: Báo cáo nhóm

**Bản ngày 14/11/2026** bổ sung kết quả tìm kiếm trong danh sách 1:N, thử nghiệm PAD vùng cắt rộng 2× và bằng chứng cho chức năng Model Tests. Nhóm đánh giá trên ảnh dữ liệu và ghi rõ giới hạn đối với webcam và người mới. Trạng thái tài liệu nằm tại `results/readiness_20261114/report_status.json`. Các mốc 30/10 bên dưới là lịch sử.

Đã tạo `reports/COS30082_Group_Report.docx` bằng tiếng Anh theo lựa chọn của người dùng. Bản cuối đã bổ sung ảnh điện thoại, có **12 trang**, gồm cả trang đầu và tài liệu tham khảo, và hoàn tất kiểm tra trực quan toàn bộ trang; đáp ứng giới hạn không quá 12 trang trong đề. Tên nhóm, họ tên và mã sinh viên để trống để điền sau.

**Cập nhật 30/10/2026:** đã đồng bộ giao diện tiếng Anh, detector hiện tại, nhiều appearance, nhập ảnh tham chiếu, lịch sử/CSV và chẩn đoán camera/mô hình. Báo cáo phân biệt PAD baseline với checkpoint mặc định mới, regression trên tập test đã dùng lại và kết quả validation không cải thiện đồng đều. Bản cuối có 171 kiểm thử, một lượt webcam đeo kính và một lượt **ảnh tĩnh trên điện thoại** có nhãn do người dùng xác nhận; đã kết xuất và QA đủ 12 trang. Trạng thái/SHA-256 cuối ở `results/app_upgrade_20261030/phase5_report_status.json`. [Kết quả bước 5](phase5_evaluation_results.md). Bản baseline ngày 27/10 được giữ ở `reports/archive/COS30082_Group_Report_baseline_20261027.docx`.

Lượt ảnh điện thoại có 431 khung trong 15,016 giây, 237 crop được PAD chấm và **cả 237 đều được cho qua**. Identity trả unknown ở 237 probe, khớp ảnh giả 0 và không có điểm danh trong bản sao tạm. Báo cáo đã nêu rõ **PAD không chặn được mẫu này; identity mới là gate ngăn điểm danh**, thay vì diễn giải 0 điểm danh như chống giả mạo thành công. Không gọi ảnh tĩnh là video replay; thử video ghi sẵn vẫn còn chờ. Mỗi điều kiện chỉ n=1, gallery một người/một appearance, không có ảnh/video lưu lại hoặc ghi nhận vào database gốc.

## Nội dung

- Abstract và Introduction: mục tiêu, phạm vi và kết quả chính.
- Methodology: sơ đồ tổng thể; dữ liệu và cách lọc trùng; kiến trúc CNN; loss; hyperparameter; chọn checkpoint và hiệu chỉnh ngưỡng của classification, triplet, liveness và expression.
- Results: so sánh hai phương pháp identity, ROC/AUC và điểm vận hành; APCER/BPCER/ACER của liveness; accuracy, macro-F1 và kết quả từng lớp của FER2013; bằng chứng kiểm tra tích hợp và phần mềm.
- Discussion: lý do chọn triplet, giới hạn so sánh chi phí tính toán, trùng dữ liệu identity, TPR thấp ở ngưỡng nghiêm, lỗi chống giả mạo, lớp biểu cảm yếu và giới hạn kiểm thử camera.
- Conclusion, tám tài liệu tham khảo và các đường dẫn bằng chứng để tái lập.

Các bảng và biểu đồ lấy số liệu từ JSON kết quả đã lưu. Công cụ tạo báo cáo xác minh SHA-256 của năm checkpoint (hai identity, PAD baseline/current, expression) và file cặp verification trước khi dựng tài liệu. Báo cáo phân biệt kiểm tra phần mềm bằng ảnh train với đánh giá độc lập; không mô tả đối chiếu thủ công là cải thiện độ chính xác của CNN. Bộ định vị Haar có sẵn và lựa chọn ROI được nêu rõ, bên cạnh ba CNN được huấn luyện từ đầu.

## Kiểm tra tài liệu

Đã kết xuất bản cuối có cả hai lượt webcam thành **12 trang** và kiểm tra trực quan toàn bộ trang, bao gồm công thức, bảng, biểu đồ, tài liệu tham khảo và đánh số trang. Abstract và Discussion đều nêu rõ PAD cho ảnh điện thoại qua và identity mới ngăn điểm danh; không mô tả đây là bản đã sửa lỗi chống giả mạo. PDF và ảnh kết xuất chỉ dùng kiểm tra nội bộ trong `tmp/group_report/render_phase5_phone_final`, không phải tệp nộp được yêu cầu ở lượt này. Không thay đổi mô hình, ngưỡng hoặc gallery người dùng trong khi viết báo cáo/đo webcam.

Nguồn dựng: `tools/build_group_report.py`. Công cụ kết xuất kiểm tra: `tools/render_group_report.py`. Trạng thái và SHA-256 bản cập nhật: `results/app_upgrade_20261030/phase5_report_status.json`; tiến độ đánh giá: `results/app_upgrade_20261030/phase5_status.json`. `results/section8_status.json` giữ mốc báo cáo baseline lịch sử.

## Việc còn lại sau khi xem

Điền thông tin nhóm/thành viên vào trang đầu và kiểm tra lại giới hạn 12 trang sau khi chỉnh sửa. Nếu sửa bằng mã nguồn, bổ sung thông tin trong công cụ dựng trước khi chạy lại để tránh mất thay đổi thủ công. Đã có một lượt người thật đeo kính và một lượt ảnh tĩnh trên điện thoại; người dùng chưa sẵn sàng lượt không kính. Người chưa đăng ký, video ghi sẵn và ảnh in còn chờ; báo cáo cần giữ đúng trạng thái hiện có.

**Phần 9 chưa quay/nộp video**; đã có [checklist chuẩn bị](09_video_checklist.md). Bước 5 nâng cấp app chưa hoàn tất đánh giá các điều kiện còn chờ.
