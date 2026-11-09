# Lộ trình hoàn thiện app

Cập nhật **30/10/2026**. Triển khai từng giai đoạn, gửi kết quả để người dùng xem sau mỗi giai đoạn rồi mới tiếp tục. Tài liệu giải thích bằng tiếng Việt; giao diện giữ tiếng Anh. Các giai đoạn dưới đây là nâng cấp app, không đổi số mục trong đề: **mục 9 của đề vẫn là video nhóm**.

## Phạm vi theo đề

Đề yêu cầu hai cách identity classification và triplet cùng ROC/AUC, mô hình liveness và biểu cảm do nhóm huấn luyện, webcam nhận diện/đăng ký và hiển thị kết quả theo thời gian thực. So sánh cosine/Euclidean và tính năng sáng tạo được khuyến khích. Các CNN và so sánh identity đã có; ưu tiên hiện tại là hoàn thiện luồng sử dụng và bằng chứng kiểm tra. Thêm chức năng quản lý không thay cho kiểm chứng chất lượng mô hình.

| Giai đoạn | Kết quả cần có | Tiêu chí xem xét | Trạng thái |
| --- | --- | --- | --- |
| 1. Ổn định lấy mẫu và Add appearance | Yêu cầu gắn với mặt lúc mở, biểu mẫu 30 giây, lấy mẫu tối đa 30 giây; hiện tiến độ/lý do reset; bắt lỗi OpenCV để có thể thử lại | Mất mặt, nhiều mặt, liveness fail, gián đoạn, đổi người/buổi hoặc hết hạn làm vô hiệu bằng chứng; không dùng lại yêu cầu; cần ba mẫu hợp lệ trước khi lưu | Hoàn tất mã, 126/126 kiểm thử và xem bố cục bằng preview; dừng để người dùng xem |
| 2. Session history và Export CSV | Xem các buổi đã lưu khi camera tắt, số bản ghi, thời gian và nguồn Automatic/Manual; xuất bản chụp dữ liệu ra CSV | Không ghi lại điểm danh khi xem/xuất; giữ nguồn xác nhận; UTF-8 phù hợp Excel, xử lý ô văn bản có thể bị hiểu là công thức; không xuất embedding/ảnh | Hoàn tất, 134/134 kiểm thử và xem giao diện bằng dữ liệu mô phỏng; dừng để người dùng xem |
| 3. Hồ sơ, appearance và ảnh tham chiếu | Duyệt người/appearance/số mẫu; đổi tên, bỏ nhóm mẫu có bản sao lưu; Import reference photos cho người mới hoặc thêm appearance như Lab 7 | Ba ảnh khác nhau cho mỗi nhóm, một mặt/ảnh, chất lượng/PAD/identity; kiểm tra rồi xác nhận trong 120 giây; giữ ID/lịch sử, không tạo điểm danh | Hoàn tất, **148/148 kiểm thử**, xem preview và kiểm tra mô hình thật với database tạm; chưa nhập ảnh cá nhân; dừng để người dùng xem |
| 4. Camera và chẩn đoán | Chọn chỉ số camera khi tắt, thử một frame; Diagnostics cho từng thành phần và Retry model setup; hướng dẫn theo nguyên nhân chất lượng | Khóa thao tác thiết bị trong lúc mở/chạy/thử; nhả camera sau thử/lỗi; chỉ rõ thành phần lỗi, giữ bước chưa chạy là Not checked; không đổi ngưỡng | Hoàn tất, **159/159 kiểm thử**; camera 0 trả 640×480; đã thử lỗi cấu hình và phục hồi qua GUI; dừng để người dùng xem |
| 5. Đánh giá và đồng bộ tài liệu | Thử có/không kính, người chưa đăng ký, ảnh điện thoại/video ghi sẵn và ảnh in; cập nhật báo cáo nhóm/checklist video | Tách kiểm thử phần mềm khỏi tỷ lệ frame và FAR/FRR; tách PAD cho ảnh giả qua khỏi identity/điểm danh; Word tối đa 12 trang, ghi rõ chưa thử | Đang thực hiện: **171/171 kiểm thử**, n=1 lượt đeo kính và n=1 lượt ảnh tĩnh điện thoại; PAD cho qua 237/237 crop ảnh giả, identity chặn nên 0 điểm danh tạm; báo cáo đã đồng bộ và QA đủ 12 trang; chờ không kính/người lạ/video/ảnh in; chưa quay video |

Import ảnh tham chiếu ở giai đoạn 3 là tạo embedding/template bằng encoder đã có, **không phải huấn luyện lại CNN**. Kiểm tra liveness trên ảnh tĩnh không tự chứng minh người đó đang hiện diện trước camera; nhập hồ sơ không được tạo bản điểm danh. Cần giữ riêng các appearance của cùng người, đối chiếu fingerprint encoder và không biến thư mục ảnh thành nhiều danh tính mới cho một người.

## Giai đoạn 1 đã hoàn tất

`People & registration` phân biệt **Register from camera** cho người mới và **Add appearance** cho người đã có hồ sơ nhưng thay đổi ngoại hình, chẳng hạn bỏ kính. Biểu mẫu không chọn sẵn người; phải nhập nhãn và tự xác nhận hồ sơ. Yêu cầu có mã riêng, gắn với quan sát lúc mở và hết hạn sau 30 giây. Sau **Begin capture**, app có thêm tối đa 30 giây để lấy ba mẫu, hiển thị số mẫu và thời gian còn lại. Các trạng thái khung hình truyền thông tin hợp lệ/lý do hủy thống nhất tới giao diện; không lấy crop cũ làm bằng chứng khi camera mất mặt.

Lượt tổng thể cuối đạt **126/126 bài trong 65,221 giây**, ghi tại `results/app_upgrade_20261030/full_tests.log`; tám bài mới kiểm tra đổi người khi biểu mẫu mở, yêu cầu hết hạn/cũ/sai mã/dùng lại, hủy đúng yêu cầu, hết hạn khi liveness fail hoặc mất mặt, trạng thái review trên khung lỗi và camera OpenCV lỗi có thể thử lại. Đã xem trực tiếp cửa sổ chính và biểu mẫu với dữ liệu giả lập bằng `--preview-appearance`: chữ tiếng Anh, lựa chọn không điền sẵn, đồng hồ và nút bị khóa khi hết hạn. Preview không nạp mô hình, không mở camera và không tạo database. Đây là kiểm tra phần mềm/bố cục, chưa phải một buổi đăng ký webcam có/không kính với người thật. Xem mã ở `attendance_app/service.py`, `camera.py`, `gui.py`, `appearance_dialog.py` và các kiểm tra `tests/test_capture_reliability.py`, `tests/test_appearance_service.py`.

## Giới hạn cần giữ khi đánh giá

FPR trên cặp verification không phải FAR bảo đảm cho gallery nhiều người. Tập identity gốc có 667 ảnh validation trùng train. Haar là bộ định vị có sẵn, cần trình bày rõ khi giải thích quy định mô hình huấn luyện; ROI là lựa chọn căn thủ công. Liveness và identity vẫn có sai số; không hạ ngưỡng để làm đẹp demo hoặc coi việc người vận hành chọn hồ sơ là mô hình nhận đúng.

[Báo cáo nhóm](08_group_report.md) đã được đồng bộ với app, lượt đeo kính và ảnh tĩnh điện thoại; bản Word cuối có 12 trang và đã được kiểm tra trực quan toàn bộ trang. Bản baseline được giữ riêng. [Hướng dẫn dùng app](06_user_interface.md) và [ghi chú liveness](liveness_glasses_diagnosis.md) mô tả cấu hình hiện tại. Bước 1–4 đã hoàn tất; bước 5 đang chờ các điều kiện thử còn thiếu.

## Giai đoạn 2 đã hoàn tất

Đã thêm **History / CSV**, lọc buổi/nguồn/tên, số bản ghi, xem chi tiết và xuất đúng snapshot đang hiển thị. Lịch sử đọc SQLite không tạo/migrate database; dùng được khi camera tắt. Bản ghi cũ thiếu audit giữ nguồn Unknown. CSV có BOM/quoting và bảo vệ công thức văn bản, giữ nguồn xác nhận nhưng không xuất ảnh/embedding. Lượt kiểm thử tại mốc này đạt **134/134 trong 61,737 giây**; đã xem giao diện và thử lọc Automatic bằng dữ liệu giả lập. [Hướng dẫn bước 2](attendance_history.md).

## Giai đoạn 3 đã hoàn tất

Đã thêm **Profiles…** để tìm người, xem từng nhóm/số mẫu, đổi tên và bỏ một nhóm sau xác nhận. Đổi tên/bỏ nhóm tạo bản sao lưu SQLite trước khi sửa, giữ ID và lịch sử; không cho bỏ nhóm cuối. Luồng **New person from photos…** và **Import another appearance…** tách rõ người mới/người đã có. Ba ảnh phải qua kiểm tra, không trùng pixel, nhất quán identity và không nhận thành người khác; cần xác nhận riêng trước khi lưu. Mã thao tác/token có thời hạn 120 giây; kiểm tra lại encoder và hồ sơ lúc lưu, không ghi một phần hay tạo điểm danh.

Lượt tổng thể đạt **148/148 trong 62,329 giây**; đã xem cửa sổ/biểu mẫu bằng preview. Kiểm tra tích hợp dùng mô hình thật, ảnh LCC train và database tạm nhập được một nhóm sau sáu nhóm bị từ chối; thêm nhóm thứ hai lặp đầu vào chỉ để kiểm tra giao dịch. Đây không phải benchmark hay kiểm chứng có/không kính. **Chưa nhập ảnh cá nhân**, không sửa database người dùng hoặc huấn luyện lại mô hình. [Hướng dẫn bước 3](profile_management.md); kết quả ở `results/app_upgrade_20261030/phase3_status.json`. Dừng để người dùng xem trước khi sang bước 4.

## Giai đoạn 4 đã hoàn tất

Đã thêm **Camera index**, **Test camera**, **Diagnostics…**, **Retry model setup** và hướng dẫn riêng cho mặt quá nhỏ/nhiều mặt/tối/chói/mờ/PAD chưa đạt. Nút thử đọc một frame rồi nhả, không gọi mô hình/database để điểm danh. Trạng thái Not checked tách khỏi PAD Not passed; không sửa checkpoint/ngưỡng/gate. Bố cục giữ vùng kết quả và bảng điểm danh hiển thị.

Lượt tổng thể đạt **159/159 trong 57,702 giây**. Đã mở app thật và thử camera 0 được 640×480, nút hoạt động lại sau khi nhả. Đã tạo lỗi đường dẫn checkpoint trong cấu hình tạm, xem đúng Identity Error và phục hồi toàn bộ thành phần qua Retry sau khi sửa tệp tạm. Chưa chạy Start camera/đăng ký/điểm danh hay đánh giá kính trong lượt này. [Hướng dẫn bước 4](camera_diagnostics.md); kết quả ở `results/app_upgrade_20261030/phase4_status.json`. Dừng để người dùng xem trước bước 5.

## Giai đoạn 5 đang thực hiện

Đã có công cụ đo theo nhãn do người dùng xác nhận, chạy pipeline app trên bản sao gallery tạm, không ghi hồ sơ/điểm danh gốc hoặc lưu ảnh/video. Đạt **171/171 kiểm thử trong 72,486 giây**. Lượt đeo kính 15 giây có 312 khung, PAD đạt 232/243 crop, identity đúng 175/232 probe và một bản ghi đúng trong bản sao tạm. Gallery một người/một appearance; không coi đây là accuracy/FAR/FRR triển khai.

Đã hoàn tất thêm **một lượt ảnh tĩnh của người dùng trên điện thoại**, 431 khung trong 15,016 giây; 240 khung một mặt và 237 crop qua chất lượng. **PAD cho cả 237/237 crop ảnh giả qua, không chặn được mẫu này**. Identity trả unknown ở toàn bộ 237 probe, không nhận khớp hồ sơ nào nên không tạo điểm danh trong bản sao tạm. Không coi 0 điểm danh là bằng chứng PAD chống giả mạo thành công. Đây là ảnh tĩnh, **chưa phải thử video ghi sẵn**.

Báo cáo tiếng Anh đã bổ sung kết quả ảnh điện thoại, giữ trống thông tin nhóm và hoàn tất QA trực quan **12/12 trang**; ảnh QA cuối ở `tmp/group_report/render_phase5_phone_final`. Báo cáo nêu rõ lỗi PAD ở Abstract và Discussion; chưa sửa mô hình/ngưỡng trong bước này. Có [quy trình thử](phase5_evaluation_protocol.md), [kết quả hiện có](phase5_evaluation_results.md) và [checklist video](09_video_checklist.md). Người dùng chưa sẵn sàng lượt không kính; người chưa đăng ký, video ghi sẵn và ảnh in còn chờ. Mỗi điều kiện đã đo chỉ có n=1 với gallery một người/một appearance; chưa đánh dấu toàn bộ bước 5 hoàn tất, chưa quay video hoặc tự gán đóng góp cá nhân. Trạng thái ở `results/app_upgrade_20261030/phase5_status.json`.
