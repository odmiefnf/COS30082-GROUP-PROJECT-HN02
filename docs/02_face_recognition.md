# Phần 2: Face Recognition

**Nguồn:** mục 2, 2.1, 2.2, 2.3 của đề bài PDF. Cấu trúc dữ liệu tuân theo mục 5 để có thể chạy thí nghiệm phần 2.  
**Trạng thái cập nhật 27/10/2026:** hai baseline đã huấn luyện từ đầu đủ 10 epoch trên 380.638 ảnh, RGB 64×64. AUC gốc: classification 0,825601; triplet 0,859969. App phần 6 dùng triplet. [Kết quả đầy đủ](06_identity_training_results.md). Xem [phần 5](05_identity_dataset.md) về ảnh trùng và phạm vi đánh giá; ngưỡng ứng dụng chặt còn bỏ sót nhiều.  
**Phạm vi:** tài liệu này chỉ nói về face verification; liveness/biểu cảm đã có tài liệu và checkpoint riêng ở phần 3/4.

## 1. Yêu cầu và phần đã triển khai

| Yêu cầu | Cách đáp ứng trong mã |
| --- | --- |
| Nhận hai ảnh và tạo điểm số so khớp | `FaceVerifier` và lệnh `face_verification.verify` trả cosine similarity và Euclidean distance. |
| Huấn luyện CNN tạo embedding gọn | `FaceEncoder` tạo vector 128 chiều mặc định, chuẩn hóa L2. |
| Classification-based | CNN + lớp phân loại danh tính, huấn luyện bằng cross-entropy. |
| Metric learning | CNN huấn luyện bằng triplet margin loss trên anchor/positive/negative. |
| So sánh hai cách học | Evaluator nhận nhiều checkpoint và xuất bảng so sánh trên cùng các cặp ảnh. |
| Đánh giá bằng ROC/AUC | Tính ROC từ điểm liên tục, gộp điểm bằng nhau đúng cách và tính AUC bằng quy tắc hình thang. |
| Nghiên cứu lựa chọn metric | Chọn cosine làm phép đo chính ban đầu trên embedding chuẩn hóa, đồng thời triển khai Euclidean để đối chiếu. |

Triển khai cả hai hướng học là yêu cầu bắt buộc. So sánh cả hai phép đo là phần đề khuyến khích; mã này đã hỗ trợ cả hai.

## 2. Quyết định triển khai

Các thông số dưới đây là baseline được đề xuất để bắt đầu, không phải thông số do đề quy định và chưa được khẳng định là tối ưu.

| Thành phần | Cấu hình ban đầu |
| --- | --- |
| Đầu vào | Ảnh khuôn mặt RGB, resize 112 x 112 |
| Chuẩn hóa pixel | Chuyển sang float32 trong [-1, 1] |
| CNN dùng chung cho hai cách học | 4 khối convolution với số kênh 16, 32, 64, 128; mỗi khối có 2 convolution + BatchNorm + ReLU, convolution đầu giảm kích thước |
| Tạo embedding | Adaptive average pooling 2 x 2, flatten, linear sang 128 chiều |
| Embedding để verification | Chuẩn hóa L2 thành vector đơn vị |
| Classification-based | Linear head theo số ID; cross-entropy nhận logits, không thêm softmax trước loss |
| Metric learning | TripletMarginLoss, khoảng cách L2, margin 0.2 |
| Optimizer | Adam, learning rate 0.001 |
| Huấn luyện mặc định | 10 epoch, batch size 32, seed 42 |
| Khởi tạo | Từ đầu, không sử dụng pretrained |
| Augmentation baseline | Tắt cho cả hai phương pháp để giữ cùng tiền xử lý |

Ảnh đầu vào cần là ảnh khuôn mặt/crop phù hợp. Phần 2 chưa có bộ phát hiện và căn chỉnh khuôn mặt; không có bước tự động nhận diện khuôn mặt trong ảnh toàn cảnh.

Classification sử dụng đặc trưng trước lớp phân loại làm embedding. Metric learning tạo positive là một ảnh khác cùng ID, còn negative thuộc ID khác. Mỗi epoch lấy mẫu lại các triplet theo seed và epoch; baseline chưa dùng hard-negative mining.

Khi giữ nguyên seed và kích thước embedding, hai phương pháp bắt đầu với cùng khởi tạo encoder. Chúng dùng cùng kiến trúc và cấu hình baseline; cross-entropy và triplet loss tối ưu những mục tiêu khác nhau. Classification có validation theo ID; triplet baseline chọn checkpoint theo training loss. Khi phân tích kết quả cần nêu rõ khác biệt này.

## 3. Dữ liệu cần có

```text
data/face_recognition/
  classification_data/
    train_data/
      person_001/*.jpg
      person_002/*.jpg
    val_data/
      person_001/*.jpg
      person_002/*.jpg
    test_data/
      person_001/*.jpg
      person_002/*.jpg
  verification_data/
    ...images...
  verification_pairs_val.txt
```

- Classification và metric learning chỉ cập nhật trọng số bằng `train_data`.
- Baseline triplet cần ít nhất 2 ID có tối thiểu 2 ảnh mỗi ID. Các ID chỉ có một ảnh vẫn có thể cung cấp negative.
- `val_data` và `test_data` dùng cho bài toán classification theo mô tả của đề; không được đưa vào luồng huấn luyện triplet.
- Khi validation/test classification có ID không có trong train, chương trình ghi nhận và loại chúng khỏi phép đo closed-set; số ID và ảnh bị loại được ghi trong metadata. Không tự đổi thứ tự nhãn của train.
- Các cặp verification dùng để tính ROC/AUC cho cả hai phương pháp. Cần có cả nhãn 0 và nhãn 1.

Ví dụ một dòng trong file cặp:

```text
verification_data/person_001/a.jpg verification_data/person_001/b.jpg 1
verification_data/person_001/a.jpg verification_data/person_002/a.jpg 0
```

Nhãn `1` là cùng người; `0` là khác người. Parser hỗ trợ khoảng trắng/tab, dòng trống, chú thích, header tùy chọn và đường dẫn có dấu cách nếu đặt trong dấu ngoặc kép. Đường dẫn sai hoặc khớp nhiều ảnh đều báo lỗi.

Các bộ dữ liệu được ghi tại [dataset_choices.md](dataset_choices.md): LCC-FASD cho chống giả mạo, FER2013 cho biểu cảm. Dữ liệu danh tính/file cặp gốc đã có tại `data/face_recognition` từ ngày 24/10/2026, xem phần 5.

## 4. Chạy từng bước

Các lệnh dưới đây chạy tại thư mục `grp_project` trong môi trường đã cài `requirements.txt`.

### Kiểm tra cấu trúc dữ liệu

```powershell
python -m face_verification.check_data --data-root data/face_recognition
```

### Huấn luyện classification-based

```powershell
python -m face_verification.train --method classification --data-root data/face_recognition --output-dir results/classification --epochs 10 --batch-size 32 --image-size 112 --embedding-dim 128 --seed 42
```

Nếu có test split, thêm `--evaluate-test` để đánh giá accuracy closed-set một lần sau khi khôi phục checkpoint tốt nhất. Test không tham gia chọn checkpoint. Validation có ID phù hợp thì chọn checkpoint theo validation loss; thiếu validation phù hợp thì ghi rõ dùng training loss.

### Huấn luyện metric learning

```powershell
python -m face_verification.train --method triplet --data-root data/face_recognition --output-dir results/triplet --epochs 10 --batch-size 32 --image-size 112 --embedding-dim 128 --seed 42
```

Triplet chọn checkpoint theo training loss. `--evaluate-test` không áp dụng cho phương pháp này.

### So sánh trên cặp verification

```powershell
python -m face_verification.evaluate --checkpoint results/classification/best_model.pt --checkpoint results/triplet/best_model.pt --data-root data/face_recognition --output-dir results/comparison
```

Có thể dùng `--pairs DUONG_DAN_FILE` nếu danh sách cặp nằm ở vị trí khác. `--device cpu` ép chạy CPU; mặc định tự chọn CUDA nếu có, nếu không dùng CPU.

### Xác minh một cặp ảnh

```powershell
python -m face_verification.verify --checkpoint results/classification/best_model.pt --image1 data/face_recognition/verification_data/person_001/a.jpg --image2 data/face_recognition/verification_data/person_001/b.jpg
```

Lệnh trên trả điểm số. Để nhận quyết định cùng/khác người, thêm `--metric cosine --threshold GIA_TRI_DA_HIEU_CHINH`, hoặc dùng `--metric euclidean` với ngưỡng khoảng cách tương ứng. Không có ngưỡng mặc định được đoán sẵn.

`--max-batches 1` trong lệnh train chỉ dùng kiểm tra nhanh phần mềm. Checkpoint từ lượt chạy này được đánh dấu `smoke_test`; không dùng chúng làm kết quả hoàn chỉnh của thí nghiệm.

## 5. ROC, AUC và ngưỡng

| Phép đo | Cách quyết định | Điểm đưa vào ROC/AUC |
| --- | --- | --- |
| Cosine similarity | Cùng người nếu similarity >= threshold | similarity |
| Euclidean distance | Cùng người nếu distance <= threshold | -distance |

ROC/AUC nhận điểm liên tục trước khi so với ngưỡng. Nhãn dương là 1 (cùng người).

Với hai vector đã chuẩn hóa L2, có quan hệ:

```text
euclidean_distance^2 = 2 - 2 * cosine_similarity
```

Vì vậy, cosine và Euclidean xếp hạng các cặp tương đương; AUC của cùng một encoder dự kiến bằng nhau, trừ sai số số học/điểm trùng. Không nên coi hai hàng metric là hai mô hình độc lập hay khẳng định metric này nhận diện tốt hơn chỉ từ các embedding đơn vị đó.

Baseline dùng cosine làm phép đo chính vì có thể tính bằng tích vô hướng của embedding đơn vị và dễ diễn giải chiều điểm. Euclidean được giữ để kiểm tra và đối chiếu theo đề. Kết luận mô hình nào tốt hơn cần dựa trên kết quả classification-based so với metric learning trên dữ liệu thật.

Evaluator đề xuất ngưỡng hữu hạn tối đa hóa `TPR - FPR` trên các cặp validation. Metadata luôn ghi `validation-calibrated / in-sample`: accuracy/TPR/FPR tại ngưỡng này không phải kết quả test độc lập. Chỉ chọn checkpoint và hiệu chỉnh ngưỡng trên tập phù hợp; khi có test độc lập thì giữ nguyên ngưỡng để đánh giá.

## 6. Các file kết quả sau khi chạy dữ liệu thật

Mỗi phương pháp huấn luyện tạo:

- `best_model.pt`: encoder được chọn; checkpoint classification còn chứa trọng số head.
- `last_model.pt`: encoder sau epoch cuối.
- `config.json`: cấu hình, nhãn ID, dữ liệu và cách chọn checkpoint.
- `train_history.csv`: loss, accuracy classification khi có và số mẫu mỗi epoch.
- `classification_test.json`: chỉ tạo khi dùng `--evaluate-test` với classification.

Luồng đánh giá tạo:

- `comparison.csv` và `comparison.json`: so sánh các checkpoint và hai metric.
- Mỗi checkpoint có `pair_scores.csv`, `roc_points.csv`, `roc_curve.svg`, `metadata.json`.
- Metadata ghi nguồn checkpoint, cấu hình, số cặp, định nghĩa điểm, scope của ngưỡng và hash để theo dõi nguồn gốc lượt đánh giá.

Các lệnh sử dụng đúng thư mục output đã chỉ định; chạy lại cùng đường dẫn sẽ ghi lại các kết quả ở đó. Khi muốn giữ các thí nghiệm riêng, dùng output-dir khác cho từng lượt.

## 7. Kiểm tra đã chuẩn bị và điều kiện hoàn tất

**Kết quả kiểm tra ngày 20/10/2026:** cả 13 bài kiểm tra đều đạt trên môi trường CPU đã nêu trong README. Bộ kiểm tra hoàn chỉnh chạy trong khoảng 15 giây; thời gian này không phải thời gian huấn luyện dữ liệu thật.

```powershell
python -m unittest discover -s tests -v
```

Các kiểm tra dùng ảnh hình học tổng hợp trong thư mục tạm, không phải ảnh khuôn mặt: tiền xử lý, nhãn validation, tính hợp lệ của triplet, lưu/khôi phục encoder, AUC với điểm trùng, chiều điểm Euclidean, huấn luyện cả hai cách và xuất kết quả đánh giá trên các ID không có trong train.

Chẩn đoán cũ [02_data_status.json](02_data_status.json) ghi tình trạng thiếu dữ liệu ngày 20/10. Trạng thái mới là `ready` về cấu trúc ở [05_data_status.json](05_data_status.json), kết hợp cảnh báo chất lượng trong `data/identity_dataset_status.json`. Kiểm tra cấu trúc không thay thế việc xem ảnh trùng.

Phần triển khai mã có thể xem và chạy kiểm tra ngay. Để hoàn tất thí nghiệm phần 2 còn cần:

1. Có bộ ảnh theo ID và danh sách cặp verification gốc, kiểm tra cấu trúc/nhãn và các split.
2. Huấn luyện hai phương pháp trên dữ liệu thật và lưu cấu hình, lịch sử.
3. Tính ROC/AUC trên `verification_pairs_val.txt`, so sánh và lựa chọn phương án dựa trên kết quả thật.

Chưa có kết quả AUC hoặc checkpoint identity thật để đưa vào báo cáo. Dữ liệu đã có từ phần 5, nhưng vẫn cần chạy cả hai thí nghiệm và ghi rõ protocol/cảnh báo trùng ảnh trước khi kết luận.

## 8. Nguồn kỹ thuật

- [FaceNet, CVPR 2015](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf): cơ sở tham khảo về học face embedding và triplet; CNN và loss trong baseline này không phải bản tái lập đầy đủ FaceNet.
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html): loss phân loại nhận logits.
- [PyTorch TripletMarginLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html): loss bộ ba với margin và khoảng cách Lp.
- [scikit-learn ROC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html): định nghĩa ROC từ nhãn và điểm liên tục. Dự án tự tính ROC bằng NumPy và kiểm tra tie handling; không cần cài scikit-learn.
