# Phần 6: Webcam, đăng ký và giao diện điểm danh

**Bổ sung giai đoạn 4 ngày 30/10/2026:** chọn **Camera index** khi dừng, **Test camera** thử một frame, **Diagnostics…** chỉ rõ lỗi từng thành phần và **Retry model setup** sau khi sửa cấu hình. Hướng dẫn phân biệt tối/chói/mờ và bước chưa kiểm tra. [Hướng dẫn camera và chẩn đoán](camera_diagnostics.md).

**Bổ sung giai đoạn 3 ngày 30/10/2026:** khi camera tắt, **Profiles…** cho tìm/đổi tên hồ sơ, bỏ nhóm mẫu sau sao lưu và nhập ba ảnh tham chiếu cho người mới hoặc thêm appearance. [Hướng dẫn quản lý hồ sơ và nhập ảnh](profile_management.md).

**Bổ sung giai đoạn 2 ngày 30/10/2026:** nút **History / CSV** cho xem các buổi đã lưu, lọc tên/nguồn và xuất các dòng đang hiển thị. Dùng được khi camera tắt. [Hướng dẫn lịch sử và CSV](attendance_history.md).

Ngày thực hiện: **27/10/2026**. Mã phần này dùng ba CNN đã huấn luyện: identity, LCC-FASD liveness và FER2013 biểu cảm. Hai phương pháp identity được huấn luyện từ đầu và đánh giá trước khi chọn cấu hình cho ứng dụng. Kết quả thật nằm trong [báo cáo identity](06_identity_training_results.md), `results/comparison/comparison.json` và `results/identity_app_config.json`.

**Bổ sung phần 7:** GUI hiện có nút **Manual review** cho unknown/ambiguous có liveness đạt, cột **Source** và chi tiết người duyệt/lý do. [Hướng dẫn tính năng](07_innovative_feature.md). Quyết định thủ công không làm thay đổi dự đoán hoặc accuracy/AUC của CNN.

**Cập nhật 27/10/2026:** giao diện đã chuyển sang tiếng Anh và bộ bắt mặt được chỉnh để giảm bỏ sót. [Chi tiết thay đổi và kết quả kiểm tra](app_capture_update.md). Phần giải thích trong tài liệu này vẫn dùng tiếng Việt; tên nút khớp với app tiếng Anh.

**Cập nhật 30/10/2026:** bước 1–4 của [lộ trình nâng cấp app](app_upgrade_roadmap.md) đã hoàn tất. Lượt cuối đạt 159/159 kiểm thử; camera 0 thật trả 640×480 ở lượt Test camera. Đánh giá có/không kính và đồng bộ báo cáo thuộc bước 5.

**Liveness mặc định hiện tại:** app nạp `results/liveness_camera_matched/best_model.pt`. Đây là CNN cùng kiến trúc, huấn luyện lại từ đầu 10 epoch với tăng cường ánh sáng và che một phần vùng trên khuôn mặt; ngưỡng chọn trên validation theo mức chấp nhận spoof đối chiếu baseline. Checkpoint `results/liveness_lcc/best_model.pt` và số đo baseline trước đây được giữ nguyên. Các lượt so sánh webcam còn nhỏ và chưa phải kiểm chứng độ chính xác với kính; tập test cũ đã được dùng lại, không phải benchmark mới. [Chi tiết chẩn đoán, mô hình/ngưỡng và giới hạn](liveness_glasses_diagnosis.md).

## Yêu cầu của phần 6

Theo mục 6, trang 4 của đề: dùng webcam, tự nhận diện khi người xuất hiện; hiển thị tên người đã đăng ký; gợi ý và cho đăng ký người chưa được nhận ra; đồng thời hiển thị liveness và biểu cảm. GUI được đề xuất để tương tác. Phần này triển khai GUI Tkinter, hiện dùng tiếng Anh, và lưu hồ sơ trên máy. Không thay nhiệm vụ nhận diện bằng bảng phân loại 4.000 tên trong dataset: người dùng tự đăng ký tên/mã sinh viên và embedding để so khớp với camera.

## Mở ứng dụng trên máy hiện tại

Cách nhanh: nhấp đúp **`start_attendance.cmd`** trong thư mục `grp_project`.

Hoặc mở PowerShell ở `grp_project`:

```powershell
.\.venv-gpu\Scripts\python.exe -m attendance_app.gui
```

Để quay lại checkpoint liveness baseline khi cần đối chiếu:

```powershell
.\.venv-gpu\Scripts\python.exe -m attendance_app.gui --liveness-checkpoint results/liveness_lcc/best_model.pt
```

Ứng dụng mặc định suy luận trên CPU với 4 thread để GPU có thể dành cho huấn luyện và để giảm tài nguyên khi demo. Môi trường GPU vẫn chạy được chế độ CPU. Muốn dùng GPU khi demo:

```powershell
.\.venv-gpu\Scripts\python.exe -m attendance_app.gui --device cuda
```

Camera chỉ được mở sau khi bấm **Start camera**. Chế độ xem bố cục hoàn toàn không nạp mô hình, không mở camera, không tạo database:

```powershell
.\.venv-gpu\Scripts\python.exe -m attendance_app.gui --preview
```

Trên máy khác, tạo Python 3.11 venv rồi cài `requirements-webcam.txt`. Các checkpoint và `results/identity_app_config.json` phải được chép sang, trong đó liveness mặc định mới cần có tại `results/liveness_camera_matched/best_model.pt`; trường `checkpoint` trong JSON identity cần trỏ đến checkpoint thật trên máy đó. Nếu muốn so sánh baseline, chép thêm `results/liveness_lcc/best_model.pt` và dùng lệnh đối chiếu bên trên. Khi đổi vị trí, chỉ sửa đường dẫn; SHA256 phải giữ khớp. Chạy bằng `.venv/Scripts/python.exe` thay cho `.venv-gpu` nếu cài CPU.

## Thử theo thứ tự này

1. Khi camera tắt, chọn **Camera index** và có thể bấm **Test camera** để kiểm tra một frame; không tạo điểm danh. Nhập tên buổi học vào **Session**, ví dụ `COS30082-lopA-2026-10-30`. Bấm **Start camera** khi các mô hình đã sẵn sàng. Nếu lỗi, mở **Diagnostics…**.
2. Một người đứng trước camera, nhìn thẳng, đủ sáng và giữ mặt rõ. Khung sẽ xuất hiện quanh mặt.
3. Khi **Liveness** hiển thị **Passed** và chưa nhận ra, app hướng dẫn phân biệt người mới và người đã có hồ sơ. Chỉ với người mới, nhập tên kèm mã sinh viên ở vùng **People & registration**, bấm **Register from camera**. Nếu đã có hồ sơ nhưng thay đổi ngoại hình, dùng **Add appearance** theo hướng dẫn bên dưới, không đăng ký lại thành một người khác.
4. Giữ một người trong khung, xoay mặt nhẹ. Hệ thống lấy ba crop khác pixel, cách nhau ít nhất 0,45 giây, đều qua liveness và đủ giống nhau theo ngưỡng identity. Có thể bấm **Cancel capture**.
5. Sau khi đăng ký thành công, tiếp tục nhìn camera. Khi cùng một hồ sơ được nhận diện ở ba khung hợp lệ liên tiếp, tên hiện trong bảng **Present this session**.
6. Rời khung rồi quay lại: một hồ sơ chỉ có một dòng trong mỗi buổi. Dừng camera, đổi tên buổi và bắt đầu lại để tạo buổi khác.
7. Bấm **Stop** hoặc đóng cửa sổ để nhả camera. Hồ sơ và bản ghi vẫn còn khi mở lại.

Nếu có hai người trong hình, ứng dụng yêu cầu chỉ một người; không chọn tùy ý khuôn mặt lớn nhất. Nếu không nhận ra người đã đăng ký, hãy thử ánh sáng/góc mặt tương tự; không hạ ngưỡng chỉ để đạt demo mà không đánh giá lại.

## Thêm appearance cho người đã đăng ký

1. Để đúng người trước camera, chờ **Liveness: Passed**, rồi bấm **Add appearance**. Cần có ít nhất một hồ sơ; không mở đồng thời với lấy mẫu hoặc **Manual review**.
2. Giữ nguyên người trong khung. Ở **Existing profile**, tự chọn đúng hồ sơ; app không chọn sẵn theo dự đoán. Nhập **Appearance label**, ví dụ `Without glasses`, rồi đánh dấu xác nhận đã đối chiếu đúng người.
3. Biểu mẫu có thời hạn **30 giây**, gắn với khuôn mặt lúc mở. Mất mặt, nhiều mặt, liveness/chất lượng không đạt, đổi người, gián đoạn quá lâu, đổi buổi hoặc hết hạn làm yêu cầu không còn hợp lệ. App hiển thị lý do và khóa **Begin capture**; bấm **Cancel** rồi bắt đầu lại với quan sát mới.
4. Bấm **Begin capture** khi bằng chứng còn hợp lệ. App có tối đa **30 giây** để lấy đủ ba mẫu live khác nhau, hiển thị **Appearance sample N/3** và thời gian còn lại. Giữ nguyên người, chỉ xoay nhẹ; xem thông báo nếu mẫu bị reset và dùng **Cancel capture** khi cần.
5. Chỉ sau khi đủ mẫu app mới lưu appearance vào **cùng hồ sơ**, giữ mẫu gốc riêng. Tối đa bốn appearance gồm cả Original; nhãn không được trùng. Sau đó thử đeo/bỏ kính để xem nhận diện thực tế; việc lưu mẫu không bảo đảm mô hình luôn nhận đúng.

Không lưu ảnh/video trong luồng này. Người vận hành xác nhận danh tính là một thao tác đăng ký có chủ ý, không phải bằng chứng mô hình tự nhận đúng và không tự tạo bản điểm danh. Để xem bố cục biểu mẫu bằng dữ liệu mô phỏng, không camera/mô hình/database:

```powershell
.\.venv-gpu\Scripts\python.exe -m attendance_app.gui --preview-appearance
```

Để dùng ảnh có sẵn theo cách tổ chức của Lab 7, dừng camera rồi mở **Profiles…** → chọn người → **Import another appearance…**, hoặc **New person from photos…** cho người mới. Chọn đúng ba ảnh, bấm **Check photos**, tự đối chiếu danh tính rồi xác nhận **Import checked photos**. App không tự quét thư mục, không huấn luyện lại CNN và không tạo điểm danh từ ảnh. [Điều kiện ảnh và hướng dẫn đầy đủ](profile_management.md).

## Luồng xử lý

```text
Webcam BGR -> RGB -> định vị một mặt -> kiểm tra kích thước/độ rõ/ánh sáng
  -> cùng một crop RGB -> CNN biểu cảm -> hiển thị nhãn (không quyết định điểm danh)
                      -> CNN liveness
       -> chưa đạt/thiếu ngưỡng: hiển thị trạng thái, dừng nhận diện/đăng ký
       -> đạt: CNN identity embedding
            -> so với centroid hồ sơ đã đăng ký
                 -> đạt ngưỡng, đủ cách biệt: hiển thị tên
                 -> dưới ngưỡng: gợi ý đăng ký
                 -> hai điểm quá gần: yêu cầu thử lại
            -> cùng tên qua ba khung liên tiếp: ghi một lần / người / buổi
```

Biểu cảm chỉ mô tả nhãn FER2013 nhìn thấy; không suy luận trạng thái tâm lý, không thay đổi kết quả ghi nhận và không được lưu vào bảng điểm danh. Từ ngày 08/11/2026, biểu cảm được phân loại trên crop đạt chất lượng cả khi PAD không đạt; identity, đăng ký và điểm danh vẫn bị PAD chặn. Khi CNN biểu cảm lỗi, GUI hiển thị **Unavailable** và Diagnostics ghi nguyên nhân; identity/liveness vẫn hoạt động.

## Định vị mặt

Mặc định dùng **Haar cascade cổ điển có sẵn trong OpenCV** để tạo bounding box. Đây là thành phần định vị có trọng số được cung cấp sẵn; không dùng nó để tính embedding, phân loại danh tính, liveness hay biểu cảm. Ba CNN của bài được khởi tạo và huấn luyện từ đầu. Cần trình bày rõ sự khác biệt này trong báo cáo; không ghi rằng toàn bộ thành phần định vị cũng tự huấn luyện.

Bản cập nhật kết hợp cascade `default` trên ảnh cân bằng histogram với cascade `alt2` trên ảnh CLAHE, xét ảnh rộng tối đa 800 pixel, `scaleFactor=1.08`, `minNeighbors=4`, và kích thước mặt tối thiểu tương ứng khoảng 80 pixel trên khung gốc. Các phát hiện chồng lấp có IoU từ 0,55 được gộp để tránh đếm cùng mặt hai lần; không giữ khung mặt cũ khi khung hiện tại không phát hiện. CNN vẫn nhận crop RGB gốc, không nhận ảnh tăng tương phản dành cho detector. Riêng bước chỉnh detector giữ nguyên ngưỡng chất lượng, liveness, identity và checkpoint; việc đổi liveness mặc định sau đó được giải thích trong ghi chú đầu tài liệu.

Nếu cách hiểu quy định của giảng viên không cho dùng cả detector có sẵn, chế độ ROI tránh thành phần đó nhưng người dùng phải tự căn mặt và hệ thống không phát hiện được nhiều mặt ngoài ROI:

```powershell
.\.venv-gpu\Scripts\python.exe -m attendance_app.gui --detector roi
.\.venv-gpu\Scripts\python.exe -m attendance_app.gui --detector roi --roi 250 120 300 300
```

ROI là chế độ căn thủ công; không báo nó là face detector. Tham số camera khác: `--camera 1`. Tọa độ ROI là pixel trên khung RGB gốc, chưa lật gương.

## Hồ sơ và ngưỡng

Database mặc định: **`data/attendance/profiles.sqlite3`**. SQLite lưu tên, ID nội bộ ngẫu nhiên, thời gian UTC, ba embedding/sample và bản ghi theo buổi. Tên tiếng Việt được chuẩn hóa NFC; tên trùng không được ghi đè. Dùng tên kèm mã sinh viên để phân biệt người trùng tên. Không tạo tên đường dẫn từ tên người dùng; không lưu ảnh/video camera.

Thư mục dự án hiện nằm trong OneDrive. Ứng dụng không tự gửi dữ liệu qua mạng; nếu OneDrive đang đồng bộ thư mục dự án, database cũng có thể được đồng bộ theo cấu hình đó. Có thể chọn database ngoài OneDrive bằng `--database "$env:LOCALAPPDATA\COS30082\attendance.sqlite3"` khi chạy PowerShell.

Mỗi hồ sơ được so khớp bằng cosine giữa embedding probe và centroid đã chuẩn hóa L2. Ứng dụng kiểm tra SHA256 checkpoint và database, không tái sử dụng embedding cũ khi đổi encoder. Nếu đổi mô hình, dùng database mới với `--database data/attendance/new_encoder.sqlite3` và đăng ký lại.

`tools.calibrate_identity_app` chọn mô hình có AUC cosine cao nhất trong hai mô hình; trên chính tập validation, chọn ngưỡng có TPR cao nhất với **FPR của cặp ≤1%**. Đây là ngưỡng thử nghiệm in-sample, không phải chứng nhận FAR ngoài thực tế. So khớp 1:N với nhiều hồ sơ có thể tăng xác suất nhận nhầm; cần đo lại bằng người chưa đăng ký trên webcam. Điều kiện cách biệt với điểm thứ hai mặc định 0,05 là quy tắc thận trọng bổ sung, chưa được hiệu chỉnh bằng tập webcam.

Tập gốc có 667 ảnh verification validation trùng pixel train; kết quả AUC và ngưỡng phải đi kèm giới hạn này. Không tự lọc file cặp gốc để cải thiện số đo. Liveness LCC-FASD và biểu cảm FER2013 cũng có khác biệt miền dữ liệu với webcam; điểm softmax không phải xác suất đảm bảo đúng ngoài môi trường đó.

## Kiểm tra đã thực hiện và phần cần thử bằng webcam

Các bài kiểm tra phần mềm dùng ảnh tổng hợp để kiểm tra: liveness chặn danh tính/đăng ký; nhận diện unknown/ambiguous; đăng ký ba mẫu khác nhau; đổi người/liveness fail làm reset mẫu; lưu UTF-8 và khớp fingerprint; ba khung cùng người; chống ghi trùng theo buổi; không dùng biểu cảm để quyết định; no-face/multi-face/chất lượng ảnh; sự kiện điều khiển không bị các frame làm mất. Xem `tests/test_part6.py`, `tests/test_identity_cache.py` và `tests/test_identity_calibration.py`.

Ở mốc phần 6, `unittest discover` đạt **79/79 bài trong 53,3 giây**; riêng phần 6 có 14 bài, gồm mô phỏng đọc camera lỗi và dừng/nhả thiết bị. Nhật ký: `results/section6_full_tests.log`. Một lượt trước có timeout tiến trình con của bài liveness cũ; chạy lại riêng 11 bài liveness và lượt toàn bộ cuối đều đạt. Trạng thái lịch sử: `results/section6_status.json`. Sau cập nhật bắt mặt/tiếng Anh, lượt mới đạt **97/97 bài trong 70,1 giây**, nhật ký `results/app_check_20261027/english_detector_tests.log`.

Đã mở GUI chế độ preview và kiểm tra bố cục qua công cụ desktop. `tools.check_attendance_models` kiểm tra offline ba checkpoint thật và dùng database tạm; hồ sơ tham chiếu lấy từ ID dataset chỉ dành cho kiểm tra phần mềm. Không coi ảnh train được dùng ở bước này là benchmark độc lập hay đăng ký webcam thành công. Kết quả lưu `results/attendance_integration.json`.

Lượt offline gồm 24 ảnh, 18 ảnh qua liveness và thực thi cả identity/biểu cảm. Có **một ảnh của ID chưa đăng ký bị nhận nhầm** thành hồ sơ đã có; ảnh đó không tạo bản ghi vì chưa đủ chuỗi khung liên tiếp. DB tạm có một bản ghi của ID đã đăng ký và được xóa khi kết thúc kiểm tra. Những kết quả này xác nhận đường đi của phần mềm và bộc lộ giới hạn mô hình; không dùng để báo FAR/accuracy triển khai.

**Đã mở webcam và quan sát khuôn mặt/liveness**, nhưng chưa xác nhận đầy đủ việc đăng ký, điểm danh, đối chiếu thủ công, độ chính xác tên hoặc khả năng chặn ảnh in/màn hình với người thật. Trong cùng 90 khung của một lượt webcam ngắn 7,031 giây, detector cũ bắt được mặt ở 23 khung và có 11 khung liveness đạt; bản mới tương ứng là 51 và 37 khung. Đây là kiểm tra thuận tiện trên một đoạn ngắn, không phải benchmark accuracy/FAR/FRR. Chỉ số **processing FPS** của GUI gồm đọc camera, định vị mặt và các CNN ở worker; không bao gồm đầy đủ thời gian màn hình/UI.

Nếu không mở được camera, ứng dụng giữ trạng thái tắt, báo lỗi ngay trong cửa sổ và cho thử lại. Kiểm tra camera index, ứng dụng khác đang giữ camera và quyền Windows bằng thao tác của người dùng. Không tự thay đổi quyền hệ thống.

## Mã nguồn chính

| File | Vai trò |
| --- | --- |
| attendance_app/gui.py | GUI, worker riêng, hàng đợi giới hạn frame, Start/Stop/đóng nhả camera |
| attendance_app/camera.py | Haar/ROI, một khuôn mặt, kiểm tra ảnh, crop RGB dùng chung |
| attendance_app/recognition.py | Nạp encoder thật, kiểm tra fingerprint, embedding, cosine open-set |
| attendance_app/store.py | SQLite transaction, tên Unicode, mẫu, ghi nhận duy nhất theo buổi |
| attendance_app/service.py | Liveness gate, đăng ký có chủ ý, theo dõi nhận diện qua các frame |
| tools/calibrate_identity_app.py | Chọn encoder/ngưỡng dựa trên đánh giá gốc, ghi cấu hình |
| tools/check_attendance_models.py | Kiểm tra offline với ba mô hình thật và DB tạm |

Mốc phần 6 hoàn thiện GUI cơ bản. Sau đó đã bổ sung đối chiếu thủ công ở phần 7, viết báo cáo nhóm phần 8 và cập nhật bắt mặt/giao diện tiếng Anh theo yêu cầu; video phần 9 chưa bắt đầu.
