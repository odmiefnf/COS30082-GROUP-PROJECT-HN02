# COS30082 GROUP PROJECT HN02

Facial Recognition with Emotion & Liveness — group project repository.

This repository contains source code, trained checkpoints, the group report and
aggregate evaluation evidence. Raw datasets, personal face profiles and attendance
databases are excluded. Install `requirements-webcam.txt` in a Python 3.11
environment, then run `python -m attendance_app.gui`; a new local gallery starts
empty. See [submission readiness](docs/submission_readiness.md) for the setup,
results and remaining demonstration tasks.

**Rà soát theo đề ngày 08/11/2026:** đã đối chiếu đủ 12 mục của đề; **202 kiểm thử đạt**, cấu hình identity chạy được khi chuyển thư mục, và ba CNN thật khởi tạo đầy đủ với database demo rỗng. Cảm xúc hiện được hiển thị trên crop đạt chất lượng dù PAD không đạt; identity, đăng ký và điểm danh vẫn yêu cầu PAD đạt. Camera chỉ có một màn lớn, khung vàng khi chưa nhận ra người/PAD chưa đạt và xanh lá khi nhận diện hợp lệ. [Bảng yêu cầu và việc còn thiếu](docs/submission_readiness.md), [chuẩn bị phỏng vấn và dàn ý báo cáo cá nhân](docs/interview_preparation.md), [đóng gói mã để chấm](docs/submission_package.md).

Đã chạy thêm một thí nghiệm PAD 20 epoch nhưng không đạt tiêu chí validation đặt trước, nên **không thay mô hình đang dùng**. Thử webcam mới được để lại theo yêu cầu người dùng. Lỗi ảnh điện thoại vượt PAD trong phép thử cũ vẫn là giới hạn thực tế; xem [thí nghiệm PAD](docs/pad_regularization_experiment.md). Các số đo và mốc kiểm thử tháng 10 bên dưới là lịch sử, không phải kết quả đánh giá mới.

Dự án COS30082 được thực hiện từng phần. **Báo cáo nhóm tiếng Anh cập nhật ngày 08/11/2026 có 12 trang, đã kiểm tra trực quan toàn bộ trang**, khớp luồng app hiện tại và giữ riêng kết quả lịch sử của một lượt đeo kính, một lượt ảnh tĩnh trên điện thoại. Abstract và Discussion nêu rõ PAD không chặn được mẫu ảnh điện thoại dù identity đã ngăn điểm danh; thí nghiệm PAD bổ sung không được chọn vào app. Thông tin nhóm/thành viên để trống theo yêu cầu. [Tệp Word báo cáo nhóm](reports/COS30082_Group_Report.docx), [ghi chú phần 8](docs/08_group_report.md) và [checklist video](docs/09_video_checklist.md). Chưa quay/nộp video phần 9.

**Lộ trình app ngày 30/10/2026:** [kế hoạch năm giai đoạn](docs/app_upgrade_roadmap.md). Bước 1–4 đã hoàn tất: lấy mẫu/Add appearance, **History / CSV**, **Profiles…**/nhập ảnh như Lab 7 và **Camera index / Test camera / Diagnostics…**. Bước 5 có **171/171 kiểm thử**, một lượt đeo kính và một lượt ảnh tĩnh trên điện thoại, mỗi điều kiện n=1 với gallery một người/một appearance. **PAD cho qua 237/237 crop ảnh điện thoại; identity trả unknown nên không có điểm danh tạm. Đây là lỗi chặn ảnh giả của PAD, dù luồng điểm danh chưa chấp nhận mẫu này.** Còn chờ không kính/người chưa đăng ký/video ghi sẵn/ảnh in. [Kết quả hiện có](docs/phase5_evaluation_results.md), [quy trình đánh giá](docs/phase5_evaluation_protocol.md), [camera/chẩn đoán](docs/camera_diagnostics.md), [hồ sơ/nhập ảnh](docs/profile_management.md), [lịch sử/CSV](docs/attendance_history.md). Không lưu ảnh/video, không sửa gallery/điểm danh người dùng hoặc đổi mô hình/ngưỡng trong các lượt đo này; không coi tỷ lệ frame là FAR/FRR độc lập.

**Trạng thái ngày 27/10/2026:** hai identity CNN khởi tạo từ đầu đã chạy đủ 10 epoch trên 380.638 ảnh / 4.000 ID, RGB 64×64. AUC trên đủ 8.805 cặp gốc: classification **0,825601**, triplet **0,859969**. App dùng triplet. Có 667 ảnh verification validation trùng train; đây không phải test verification độc lập. Ở ngưỡng FPR cặp gần 1%, TPR chỉ **9,73%**, còn bỏ sót nhiều. [Kết quả và giới hạn](docs/06_identity_training_results.md). Phần 3 có AUC=0,823907/ACER=26,05%; phần 4 accuracy=58,36%/macro-F1=0,4984 trên các tập đã lọc. Đã kiểm tra offline ba mô hình thật và mở webcam để quan sát bắt mặt/liveness; chưa xác nhận trọn luồng đăng ký, điểm danh và đối chiếu thủ công với người thật. Downloader CelebA-Spoof đã hủy, không tự chạy lại.

**Mở demo:** nhấp đúp `start_attendance.cmd`, hoặc chạy `.\.venv-gpu\Scripts\python.exe -m attendance_app.gui` tại `grp_project`, rồi bấm **Start camera**. Giao diện hiện dùng tiếng Anh; hướng dẫn giải thích bằng tiếng Việt. [Hướng dẫn đăng ký và chạy phần 6](docs/06_user_interface.md). Camera không tự bật khi mở cửa sổ. Hồ sơ mặc định ở `data/attendance/profiles.sqlite3`; không lưu ảnh/video camera, biểu cảm không quyết định ghi nhận.

**Phần 7:** khi liveness đạt nhưng identity là unknown/ambiguous, nút **Manual review** cho người vận hành tự chọn hồ sơ, nhập tên/lý do và xác nhận trong 30 giây. Bảng điểm danh có cột **Source**, hiển thị **Automatic**/**Manual**; nhấp đúp để xem chi tiết. Điểm số, checkpoint và AUC/TPR của mô hình giữ nguyên. [Cách dùng và giới hạn](docs/07_innovative_feature.md). Xem bố cục không dùng camera bằng `python -m attendance_app.gui --preview-review`.

**Cập nhật bắt mặt và tiếng Anh:** dùng hai Haar cascade cùng chuẩn hóa tương phản để giảm bỏ sót; riêng lượt chỉnh detector này giữ nguyên mô hình và ngưỡng liveness/identity. Lượt kiểm thử tại mốc đó đạt **97/97 bài trong 70,1 giây**. Trên cùng 90 khung webcam trong một lượt ngắn, số khung bắt được mặt tăng từ 23 lên 51 và số khung liveness đạt tăng từ 11 lên 37; đây là kiểm tra thuận tiện, không phải benchmark độ chính xác. [Chi tiết thay đổi, kiểm tra và giới hạn](docs/app_capture_update.md).

**Liveness mặc định hiện tại:** app dùng `results/liveness_camera_matched/best_model.pt`, CNN khởi tạo từ đầu và huấn luyện 10 epoch với tăng cường ánh sáng/che một phần vùng trên khuôn mặt. Ngưỡng được chọn bằng validation, có đối chiếu mức chấp nhận spoof của baseline. Checkpoint cũ `results/liveness_lcc/best_model.pt` vẫn được giữ; các số đo phần 3 ở trên là kết quả baseline lịch sử. Kiểm tra lại tập test cũ và một số crop webcam cho thấy hướng cải thiện nhưng không phải benchmark mới hoặc chứng nhận accuracy với kính. [Chẩn đoán, lựa chọn mô hình/ngưỡng và giới hạn](docs/liveness_glasses_diagnosis.md).

- [Phần 1: mục tiêu và phạm vi](docs/01_project_scope.md)
- [Phần 2: cách triển khai và chạy](docs/02_face_recognition.md)
- [Phần 3: liveness, tải dữ liệu và tích hợp](docs/03_anti_spoofing.md)
- [Phần 3: kết quả huấn luyện trên LCC-FASD thật](docs/03_lcc_fasd_results.md)
- [Phần 4: FER2013, mô hình và cách chạy webcam](docs/04_emotion_recognition.md)
- [Phần 4: kết quả huấn luyện trên FER2013 thật](docs/04_fer2013_results.md)
- [Phần 5: dữ liệu danh tính gốc và kiểm tra chất lượng](docs/05_identity_dataset.md)
- [Phần 6: GUI, webcam, đăng ký và điểm danh](docs/06_user_interface.md)
- [Phần 7: đối chiếu thủ công có lưu vết](docs/07_innovative_feature.md)
- [Phần 8: báo cáo nhóm tiếng Anh, 12 trang](docs/08_group_report.md)
- [Cập nhật bắt mặt và giao diện tiếng Anh](docs/app_capture_update.md)
- [Liveness với kính: chẩn đoán và checkpoint demo hiện tại](docs/liveness_glasses_diagnosis.md)
- [Lộ trình nâng cấp app và trạng thái từng giai đoạn](docs/app_upgrade_roadmap.md)
- [Lịch sử buổi và xuất CSV](docs/attendance_history.md)
- [Kết quả thật hai mô hình identity](docs/06_identity_training_results.md)
- [Các bộ dữ liệu đã chọn](docs/dataset_choices.md)

## Chuẩn bị môi trường

Chạy PowerShell tại thư mục `grp_project`. Các lệnh dưới đây tạo môi trường riêng cho dự án; không sử dụng `.venv` bị hỏng ở thư mục cha.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Mã được kiểm tra trên Python 3.11.9, PyTorch 2.5.1 CPU, NumPy 2.4.6 và Pillow 12.3.0. Các phiên bản này được ghi trong `requirements.txt`; không cần torchvision, scikit-learn hoặc matplotlib để chạy phần 2.

Demo webcam phần 4 dùng dependency tùy chọn `requirements-webcam.txt` (OpenCV 4.13.0.92), không cần khi chỉ train/evaluate bằng ảnh.

Môi trường đã có sẵn được dùng để kiểm tra trên máy hiện tại là `C:\Users\kiet1\OneDrive\Desktop\cos30082\lab4\.venv\Scripts\python.exe`. Mã nguồn của dự án không phụ thuộc đường dẫn đó; môi trường riêng cài đúng requirements cũng có thể chạy.

## Kiểm tra dữ liệu

Dataset gốc đã được nhập tại `data/face_recognition`. Ảnh/cặp được giữ nguyên; xem `data/identity_dataset_status.json` và phần 5 về cảnh báo ảnh trùng. Có thể chạy lại kiểm tra cấu trúc như sau:

Đặt dữ liệu theo cấu trúc trong tài liệu phần 2, rồi chạy:

```powershell
.\.venv\Scripts\python.exe -m face_verification.check_data --data-root data/face_recognition
```

Lệnh trả mã lỗi 2 và mô tả các thành phần còn thiếu nếu dữ liệu chưa sẵn sàng. Không tự tải bộ dữ liệu hoặc tạo kết quả giả.

Các công cụ phần 5: `face_verification.import_data` nhập ZIP/thư mục mà không ghi đè tệp khác nội dung; `face_verification.audit_archive` giải mã/hash trực tiếp trong ZIP; `face_verification.audit_data` kiểm tra trực tiếp thư mục khi cần. File `verification_pairs_val.txt` gốc không bị viết lại hoặc lọc.

## Huấn luyện và so sánh

```powershell
.\.venv\Scripts\python.exe -m face_verification.train --method classification --data-root data/face_recognition --output-dir results/classification_new --epochs 10 --image-size 64 --batch-size 256 --evaluate-test
.\.venv\Scripts\python.exe -m face_verification.train --method triplet --data-root data/face_recognition --output-dir results/triplet_new --epochs 10 --image-size 64 --batch-size 256
.\.venv\Scripts\python.exe -m face_verification.evaluate --checkpoint results/classification_new/best_model.pt --checkpoint results/triplet_new/best_model.pt --data-root data/face_recognition --output-dir results/comparison_new
```

Không cần huấn luyện lại để mở demo trên máy hiện tại. Ví dụ trên dùng output mới để giữ checkpoint đã hiệu chỉnh. Công thức thực nghiệm GPU và cache của lượt đã hoàn tất nằm trong báo cáo kết quả identity; CPU sẽ chậm hơn trên bộ lớn.

Chỉ dùng `--evaluate-test` nếu có `classification_data/test_data`. Nếu thiếu test split, bỏ cờ này; nó không bắt buộc để huấn luyện và tính AUC verification. Test classification được đánh giá sau khi chọn checkpoint, không dùng để chọn mô hình.

## Kiểm tra mã

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Các kiểm tra tạo ảnh hình học trong thư mục tạm, thực hiện bước huấn luyện của cả hai phương pháp và đánh giá luồng xử lý. Đây là kiểm tra phần mềm, không đo chất lượng nhận diện khuôn mặt. Dữ liệu và checkpoint kiểm tra được xóa cùng thư mục tạm.

## Phần 3: chống giả mạo

Tải thủ công [LCC-FASD trên Kaggle](https://www.kaggle.com/datasets/faber24/lcc-fasd), lưu ZIP thành `data/downloads/lcc-fasd.zip`, rồi giải nén để có:

```text
data/LCC_FASD/
  LCC_FASD_training/real/ và spoof/
  LCC_FASD_development/real/ và spoof/
  LCC_FASD_evaluation/real/ và spoof/
```

Adapter cũng chấp nhận một lớp thư mục `LCC_FASD` lồng bên trong. Sau khi giải nén, chạy tại `grp_project`:

```powershell
python -m anti_spoofing.prepare_lcc_fasd --data-root data/LCC_FASD --output-dir data/lcc_fasd_manifests
python -m anti_spoofing.check_data --manifest-dir data/lcc_fasd_manifests --crop-mode full --inspect-images 5
python -m anti_spoofing.train --train-manifest data/lcc_fasd_manifests/train.csv --val-manifest data/lcc_fasd_manifests/val.csv --output-dir results/liveness_lcc --crop-mode full --cache-images
python -m anti_spoofing.evaluate --checkpoint results/liveness_lcc/best_model.pt --manifest data/lcc_fasd_manifests/test.csv --output-dir results/liveness_lcc_test
```

LCC-FASD tập trung tấn công qua màn hình; cần kiểm thử ảnh in riêng. Dữ liệu mất cân bằng nên xem ACER/APCER/BPCER/AUC cùng accuracy.

Adapter giữ development; loại 202 ảnh train theo 12 mã ID trùng development, rồi loại thêm 4 ảnh train có nội dung trùng validation. Còn **8.093 ảnh train, 2.948 validation**. Evaluation loại 58 ảnh trùng train/validation, còn **7.522 ảnh**. Toàn bộ loại trừ được ghi trong manifest summary. Evaluation không có ID: báo cáo ghi rõ chưa xác minh được tách danh tính. Xem [phần 3](docs/03_anti_spoofing.md) về giao thức và giới hạn.

Luồng `attendance_pipeline.py` chặn xác minh danh tính nếu không đạt liveness. Mã CelebA-Spoof cũ được giữ để tham khảo; các lệnh hướng dẫn hiện dùng LCC-FASD.

## Phần 4: biểu cảm với FER2013

Đã tải/giải nén đủ 35.887 ảnh FER2013. Sau khi lọc trùng pixel/nhãn mâu thuẫn, dùng 27.170 Training để train, 3.562 PublicTest để validation, 3.271 PrivateTest để đánh giá cuối. Nhãn cố định: angry, disgust, fear, happy, sad, surprise, neutral. Mô hình khởi tạo từ đầu và train 15 epoch, chọn bằng macro-F1 validation. Test đã lọc đạt **58,36% accuracy, 0,4984 macro-F1, 50,91% balanced accuracy**; fear/disgust còn yếu. API một crop trong RAM median khoảng 1,11 ms trên CPU, không gồm camera hoặc các module khác. [Báo cáo chi tiết](docs/04_fer2013_results.md).

```powershell
python -m emotion_recognition.infer --checkpoint results/emotion_fer/best_model.pt --image ANH_KHUON_MAT_DA_CROP
python -m pip install -r requirements-webcam.txt
python -m emotion_recognition.webcam --checkpoint results/emotion_fer/best_model.pt --camera 0
```

Camera chỉ bật khi người dùng chạy lệnh demo; căn mặt vào ROI cố định, Q/Esc thoát. Có thể thêm `--liveness-checkpoint results/liveness_lcc/best_model.pt`. Mốc triển khai phần 4 chưa thử camera vật lý; sau đó app điểm danh đã được mở webcam và quan sát bắt mặt/liveness, xem bản cập nhật bên trên.

API `AttendanceVerificationPipeline(..., emotion_recognizer=...)` nhận cùng crop RGB; emotion chỉ để hiển thị, không quyết định chấp nhận chấm công. App phần 6 dùng cùng nguyên tắc này cho gallery nhiều người. Kiểm tra offline 24 ảnh với ba checkpoint thật có 18 crop qua liveness và chạy identity/biểu cảm; có một ảnh chưa đăng ký bị nhận nhầm, không được ghi nhận do chưa đủ chuỗi khung liên tiếp. Đây là kiểm tra phần mềm trên ảnh train, không phải benchmark độc lập. Kết quả đầy đủ: `results/attendance_integration.json`. Cần thử đầy đủ với người đã/chưa đăng ký và tấn công ảnh in/màn hình qua webcam.
