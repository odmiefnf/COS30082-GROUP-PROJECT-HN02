# Phần 5: Dữ liệu nhận diện danh tính

**Cập nhật 27/10/2026:** phần dữ liệu dưới đây ghi lại mốc hoàn thành ngày 24/10. Hai mô hình identity hiện đã huấn luyện thật đủ 10 epoch và được tích hợp vào GUI phần 6. Xem [kết quả mới](06_identity_training_results.md) và [hướng dẫn GUI](06_user_interface.md).

**Trạng thái ngày 24/10/2026: đã hoàn thành bước dữ liệu, `ready_with_warnings`.** Đã nhập đủ 465.738 tệp, giải mã kiểm tra đủ 465.735 ảnh, kiểm tra cấu trúc trên đĩa đạt và SHA256 file cặp khớp ZIP. Chưa huấn luyện mô hình identity thật hoặc tính AUC identity. Trạng thái tổng hợp: `data/identity_dataset_status.json`; kết quả cấu trúc: [05_data_status.json](05_data_status.json).

## Yêu cầu từ đề và phạm vi phần này

Nguồn: mục 5, trang 3–4 trong [đề bài](<../Facial Recognition with Emotion and Liveness.pdf>). Dataset danh tính được lấy từ [cuộc thi Kaggle gốc](https://www.kaggle.com/competitions/11-785-fall-20-homework-2-part-2/data).

| Thành phần | Vai trò đúng theo đề |
| --- | --- |
| classification_data/train_data/<ID>/ | Cập nhật trọng số cho cả classification-based và metric learning |
| classification_data/val_data/<ID>/ | Kiểm tra classification, chọn checkpoint classification; không cần cho metric learning |
| classification_data/test_data/<ID>/ | Đánh giá classification sau chọn mô hình; không dùng train hoặc chọn mô hình |
| verification_data/ | Các ảnh để xác minh hai ảnh cùng người hay khác người |
| verification_pairs_val.txt | Hai đường dẫn ảnh và nhãn 1=cùng người, 0=khác người; tính ROC/AUC trên chính danh sách này |

LCC-FASD và FER2013 là các nguồn riêng cho liveness/biểu cảm; không thay bộ danh tính hoặc file cặp gốc. Cùng một ID xuất hiện ở train/val/test classification là bình thường trong bài toán closed-set; ảnh trùng pixel giữa các tập mới cần được báo riêng.

## Nguồn và tải xuống

Ngày 24/10/2026, người dùng đăng nhập Kaggle và tự chấp nhận nội quy cuộc thi. Tệp được tải qua phiên trình duyệt đã có quyền; không xuất cookie, lưu mật khẩu hoặc tạo API key mới.

```text
ZIP trong dự án:
C:\Users\kiet1\OneDrive\Desktop\cos30082\grp_project\data\downloads\identity_kaggle_2020.zip

Thư mục dữ liệu sau nhập:
C:\Users\kiet1\OneDrive\Desktop\cos30082\grp_project\data\face_recognition
```

ZIP có **1.445.337.624 byte**, chứa **465.738 tệp**, tổng dữ liệu **1.429.423.145 byte**. Đây là tổng byte dữ liệu, chưa tính overhead hệ thống tệp. Đã giữ bản tải gốc trong Downloads; không xóa tệp người dùng.

| Nhóm tệp trong ZIP | Số tệp |
| --- | ---: |
| classification_data/train_data | 380.638 |
| classification_data/val_data | 8.000 |
| classification_data/test_data | 8.000 |
| verification_data | 69.097 |
| verification_pairs_val.txt | 1 |
| verification_pairs_test.txt | 1 |
| hw2p2_sample_submission.csv | 1 |

Mẫu ảnh gốc được kiểm tra có kích thước 64×64. Kích thước đầu vào CNN được cấu hình riêng khi huấn luyện; mặc định mã phần 2 là RGB 112×112. Khi so sánh hai phương pháp cần dùng cùng cấu hình đầu vào.

## Nhập dữ liệu mà không thay file gốc

`face_verification.import_data` nhận ZIP hoặc thư mục đã giải nén. Công cụ:

- Giữ nguyên byte ảnh, ID và file cặp; không tự tạo nhãn, sinh cặp hoặc viết lại đường dẫn trong file cặp.
- Kiểm tra đường dẫn ZIP, chặn traversal/drive/ADS, symlink và đường dẫn trùng trên Windows.
- Đọc mỗi entry qua ZipExtFile để kiểm tra CRC; ghi qua tệp tạm rồi mới đưa vào đích.
- Không ghi đè tệp có sẵn với nội dung khác. Có thể chạy lại khi các tệp đích đã có đúng nội dung.
- Giữ wrapper/root gốc khi cần để các tiền tố trong đường dẫn cặp vẫn dùng được; root thật được in trong kết quả và lưu ở `import_status.json`.
- Cache kiểm tra thư mục cha để xử lý hàng trăm nghìn ảnh mà vẫn giữ ranh giới thư mục đầu ra.

```powershell
python -m face_verification.import_data --source data/downloads/identity_kaggle_2020.zip --output-dir data/face_recognition
Get-Content data/face_recognition/import_status.json | ConvertFrom-Json
```

`status=extracting` chưa phải dữ liệu sẵn sàng. Chỉ sau khi đủ tệp và có root hợp lệ, trạng thái mới chuyển `imported_not_yet_audited`. Nhật ký thực tế: `data/logs/identity_import.log`.

Sau khi đối chiếu toàn bộ kết quả ngày 24/10, trạng thái lượt này đã được cập nhật thành `imported_audited_with_warnings`. `data/identity_dataset_status.json` là trạng thái tổng hợp hiện tại; các cảnh báo trùng ảnh vẫn được giữ nguyên.

## Kiểm tra dữ liệu

**Kết quả kiểm tra ZIP:** giải mã đủ 465.735 ảnh, không có ảnh lỗi. Mỗi split classification có 4.000 ID; val/test đều không có ID ngoài train. Tất cả 4.000 ID train có ít nhất hai ảnh để tạo positive cho triplet. File cặp validation có **8.805 hàng** (4.315 cùng người, 4.490 khác người), dùng 9.355 ảnh khác nhau; không có cặp lặp, cặp nhãn mâu thuẫn hoặc self-pair.

Có **8.775 nhóm pixel trùng qua các nhóm dữ liệu**, không phát hiện pixel giống hệt mang hai ID classification khác nhau. Trong 9.355 ảnh thuộc cặp validation, **667 ảnh trùng pixel với train classification**, 27 ảnh trùng val classification, 26 ảnh trùng test classification. Đây là cảnh báo dữ liệu gốc, không phải lỗi do nhập ZIP. Không tự thay file cặp hoặc xóa ảnh. Khi huấn luyện cần quyết định công khai dùng nguyên protocol của đề hay thêm một protocol train đã làm sạch riêng; dù cách nào, không gọi AUC validation là test độc lập.

Để tránh đọc lại gần nửa triệu tệp nhỏ, công cụ `audit_archive` kiểm tra trực tiếp dữ liệu trong ZIP song song với giải nén:

```powershell
python -m face_verification.audit_archive --archive data/downloads/identity_kaggle_2020.zip --output-dir data/identity_archive_audit
python -m face_verification.check_data --data-root data/face_recognition --output docs/05_data_status.json
```

Kết quả archive gồm `audit.json`, `image_hashes.csv` và `duplicates.json`. Kiểm tra:

1. Số ảnh và số ID từng split, điều kiện triplet ít nhất hai ID có hai ảnh mỗi ID.
2. Giải mã mọi ảnh, xoay EXIF đúng như loader, đổi RGB; ghi hash file và hash pixel gốc kèm kích thước.
3. Ảnh giống hệt qua split hoặc nằm dưới nhiều ID; chỉ báo, không tự xóa ảnh khỏi dataset gốc.
4. Cặp xác minh có đủ hai nhãn, mọi ảnh được tham chiếu tồn tại, cặp bị lặp/nhãn mâu thuẫn và self-pair.
5. SHA256 ZIP/file cặp để đối chiếu sau giải nén.

Auditor không tự nhận diện người; hash chỉ phát hiện ảnh giống hệt, không chứng minh ảnh khác nhau không cùng người hoặc video. `ready_with_warnings` nghĩa là cấu trúc có thể chạy nhưng phải đọc cảnh báo trước khi gọi kết quả là đánh giá độc lập.

Có thể kiểm tra trực tiếp thư mục hoặc chạy lại về sau:

```powershell
python -m face_verification.audit_data --data-root data/face_recognition --output data/identity_audit_from_files.json
```

Mặc định lệnh trên đọc/giải mã tất cả ảnh classification và verification, kể cả ảnh verification chưa có trong cặp validation; có thể mất thời gian. `--structure-only` bỏ giải mã và không được coi là kiểm tra pixel đầy đủ.

## File cặp và AUC cần giữ đúng

Không lọc hoặc thay `verification_pairs_val.txt` bằng cặp tự tạo nếu vẫn báo kết quả theo đề. Giữ đủ hàng và nhãn gốc; ghi rõ nếu có ảnh trùng hoặc cặp lặp làm giới hạn kết luận. Trường hợp phải làm một giao thức sạch khác, dùng thư mục/file khác và báo cáo riêng.

`verification_pairs_test.txt` và `hw2p2_sample_submission.csv` phục vụ luồng nộp Kaggle. Các số 0 trong submission mẫu **không phải nhãn ground truth** để tính AUC. Chỉ dùng nhãn có trong `verification_pairs_val.txt` cho AUC được yêu cầu tại bước này.

Evaluator dùng cosine similarity hoặc âm Euclidean distance để ROC có quy tắc “điểm cao hơn = cùng người hơn”. Ngưỡng chọn từ cặp validation là ngưỡng in-sample; không gọi đó là kết quả test độc lập.

## Liên hệ lại phần 2 và bước sau

Sau khi kiểm tra dữ liệu thật đạt, hai phương pháp đã có sẵn lệnh:

```powershell
python -m face_verification.train --method classification --data-root data/face_recognition --output-dir results/classification --epochs 10 --evaluate-test
python -m face_verification.train --method triplet --data-root data/face_recognition --output-dir results/triplet --epochs 10
python -m face_verification.evaluate --checkpoint results/classification/best_model.pt --checkpoint results/triplet/best_model.pt --data-root data/face_recognition --output-dir results/comparison
```

Đây là các lệnh cho thí nghiệm phần 2; việc có dữ liệu không tự tạo ra checkpoint hoặc AUC thật. Hiện môi trường `lab4/.venv` dùng PyTorch CPU. Máy có RTX 3060 Laptop 6 GiB được phát hiện bằng nvidia-smi, nhưng cần môi trường PyTorch CUDA tương thích riêng nếu chọn chạy GPU cho bộ lớn này; chưa cài/thay môi trường GPU ở phần 5.

Điểm dừng tại ngày hoàn thành phần 5 là tải/nhập/kiểm tra bộ danh tính gốc, ghi trạng thái và giới hạn. Huấn luyện identity và GUI đã được thực hiện tiếp vào ngày 27/10 ở phần 6, không thay file cặp gốc.
