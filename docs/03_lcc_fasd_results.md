# Kết quả phần 3 — LCC-FASD

Báo cáo tạo lúc 23/10/2026 16:13 UTC+7.

Đã tải dữ liệu thật, kiểm tra ảnh, huấn luyện CNN từ đầu và đánh giá bằng ngưỡng validation. Đây là baseline nghiên cứu; chưa phải hệ thống điểm danh hoàn chỉnh.

## Dữ liệu và kiểm tra chất lượng

- Nguồn: [Kaggle LCC-FASD v1](https://www.kaggle.com/datasets/faber24/lcc-fasd).
- ZIP: 5,196,000,412 byte; SHA256 `040532c10f1dbc1a957ed521b7e9a5190e11fcb07d4184de5cc212a8900b0eee`.
- Đã giải nén và kiểm tra CRC 18,827 ảnh; đọc/resize thành công 18,827/18,827 ảnh, không có ảnh lỗi.
- Ảnh gốc vẫn còn trong `data/LCC_FASD`; các loại trừ chỉ áp dụng vào manifest.

| Tập sử dụng | Ảnh thật | Ảnh giả | Tổng | Mã ID biết được |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,193 | 6,900 | 8,093 | 106 |
| Validation | 405 | 2,543 | 2,948 | 25 |
| Evaluation đã lọc | 256 | 7,266 | 7,522 | Không có ID |

Loại 202 ảnh train theo các mã ID trùng development; loại thêm 4 ảnh train trùng byte với validation. Evaluation loại 58 ảnh trùng byte với train/validation. Validation được giữ nguyên. Test không quyết định ảnh nào được giữ để huấn luyện. Danh sách và lý do nằm trong `data/lcc_fasd_manifests/split_summary.json` và `source_inventory.json`.

**Đây là giao thức đã điều chỉnh và tập evaluation con, không phải kết quả benchmark nguyên bản.** Evaluation ẩn danh: không xác minh được tách người hoặc video; chỉ kiểm tra ảnh không trùng đường dẫn/hash với train/validation.

## Cấu hình huấn luyện

CNN bốn khối convolution 16/32/64/128 kênh, đặc trưng 128 chiều và head hai lớp; trọng số khởi tạo ngẫu nhiên, không pretrained. RGB 112×112, bilinear resize, chuẩn hóa [-1,1], dùng toàn bộ ảnh crop (`crop_mode=full`).

Cross-entropy không trọng số, Adam lr=0.001, batch=32, seed=42, 10 epoch đầy đủ, không giới hạn batch, không augmentation. Thiết bị cpu, 4 CPU threads. Cache tensor chỉ giảm đọc/giải mã, không thay đổi pixel. Thời gian huấn luyện và hiệu chỉnh, không gồm bước đọc/hash/nạp cache: 3.5 phút.

Chọn **epoch 1** theo validation loss thấp nhất; sau đó hiệu chỉnh ngưỡng trên toàn bộ 2,948 validation. Ngưỡng live: **0.072996063**. Evaluation dùng nguyên ngưỡng đó, không chọn lại mô hình hoặc ngưỡng trên test.

| Epoch | Train loss | Validation loss | Train accuracy (argmax) | Validation accuracy (argmax) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.34537 | 0.33744 | 86.16% | 85.92% |
| 2 | 0.29694 | 0.36609 | 88.15% | 85.21% |
| 3 | 0.26040 | 0.44570 | 89.78% | 80.60% |
| 4 | 0.21325 | 0.40310 | 91.19% | 79.34% |
| 5 | 0.18013 | 0.43750 | 92.99% | 82.84% |
| 6 | 0.14582 | 0.48059 | 94.34% | 84.84% |
| 7 | 0.12167 | 0.80305 | 95.29% | 78.63% |
| 8 | 0.09464 | 0.84815 | 96.53% | 74.73% |
| 9 | 0.07780 | 0.78345 | 97.17% | 81.24% |
| 10 | 0.07623 | 0.75799 | 97.17% | 82.43% |

Validation loss tăng sau epoch tốt nhất trong khi train loss tiếp tục giảm, cho thấy overfitting. Báo cáo dùng checkpoint được chọn bằng validation, không dùng epoch cuối. Các cải tiến về augmentation, cân bằng lớp hoặc regularization cần được so sánh bằng validation trong một thí nghiệm sau.

## Kết quả tại ngưỡng validation

| Chỉ số | Validation, dùng để hiệu chỉnh | Evaluation đã lọc |
| --- | ---: | ---: |
| AUC | 0.829557 | 0.823907 |
| accuracy | 68.93% | 76.97% |
| APCER | 34.41% | 22.80% |
| BPCER | 10.12% | 29.30% |
| ACER | 22.27% | 26.05% |

APCER là tỷ lệ giả được chấp nhận nhầm; BPCER là tỷ lệ người thật bị từ chối; ACER là trung bình hai lỗi. Validation tại ngưỡng vừa chọn là in-sample.

| Nhãn thật | Dự đoán giả | Dự đoán thật |
| --- | ---: | ---: |
| Giả | 5,609 | 1,657 |
| Thật | 75 | 181 |

Nếu luôn đoán giả, accuracy vẫn đạt 96.60%, nhưng BPCER=100%, ACER=50%, AUC=0.5. Vì vậy cần đọc AUC và cả hai loại lỗi, không kết luận từ accuracy riêng.

[Đường ROC](../results/liveness_lcc_test/roc_curve.svg) · [Metrics và provenance](../results/liveness_lcc_test/metrics.json) · [Predictions](../results/liveness_lcc_test/predictions.csv)

## Tệp dùng tiếp và phạm vi còn lại

- Checkpoint đã chọn và hiệu chỉnh: `results/liveness_lcc/best_model.pt`.
- Lịch sử: `results/liveness_lcc/history.csv`; nhật ký: `training.log`; cấu hình và hash: `config.json`.
- Module suy luận: `python -m anti_spoofing.infer --checkpoint results/liveness_lcc/best_model.pt --image DUONG_DAN_ANH_CROP --face-crop`.
- `AttendanceVerificationPipeline` kiểm tra liveness trước, chỉ gọi identity khi live đạt ngưỡng. Mã tích hợp đã có kiểm thử phần mềm; chưa đánh giá tích hợp bằng mô hình identity thật vì thiếu dữ liệu/checkpoint phần 2.
- LCC-FASD tập trung giả mạo qua màn hình; chưa có kết quả trên ảnh in riêng hoặc webcam thực tế.
- Chưa có GUI, đăng ký người lạ hoặc dữ liệu điểm danh. Chưa bắt đầu phần 4 về cảm xúc.

Độ khái quát giữa người, video và thiết bị vẫn chưa được chứng minh; không dùng các chỉ số này để khẳng định hệ thống an toàn khi triển khai thực tế.

## Kiểm tra CLI trên ảnh thật

Lấy ảnh đầu tiên của mỗi lớp trong manifest evaluation, không chọn ảnh theo kết quả. Hai ví dụ này kiểm tra CLI, không dùng để ước lượng chất lượng mô hình.

| Nhãn thật | Tệp | Điểm live | Quyết định |
| --- | --- | ---: | --- |
| live | real_0.png | 0.583725 | live |
| spoof | spoof_1000.png | 0.059204 | spoof |

[Kết quả CLI](../results/liveness_lcc_test/inference_examples.json)
