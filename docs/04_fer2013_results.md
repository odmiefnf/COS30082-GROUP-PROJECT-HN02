# Kết quả phần 4 — FER2013

Báo cáo tạo lúc 23/10/2026 16:58 UTC+7.

Đã tải FER2013, tự huấn luyện CNN phân loại bảy biểu cảm nhìn thấy và đánh giá bằng PrivateTest đã lọc. Đây là nhãn biểu cảm của dataset, không phải đo trạng thái nội tâm. Kết quả chỉ để hiển thị, không quyết định chấp nhận chấm công.

## Dữ liệu và kiểm tra

Nguồn: [Kaggle FER2013 v1](https://www.kaggle.com/datasets/msambare/fer2013). Tải và kiểm tra CRC đủ 35,887 ảnh; bước chuẩn bị giải mã đủ 35,887 ảnh 48×48. SHA256 ZIP: `9f3422c7a45b676c2965ef4fc5f3099932aa98fd13eaaa2c254f620a9894a6e9`.

| Tập | Nhóm tệp gốc | Số gốc | Sau lọc |
| --- | --- | ---: | ---: |
| Train | Training_ | 28,709 | 27,170 |
| Validation | PublicTest_ | 3,589 | 3,562 |
| Test cuối | PrivateTest_ | 3,589 | 3,271 |

Loại các nhóm cùng pixel nhưng nhãn mâu thuẫn; giữ một bản cho ảnh trùng trong từng split. Tiếp đó loại train trùng validation và test trùng train/validation. Không xóa ảnh gốc. Tất cả loại trừ được lưu trong `data/fer2013_manifests/excluded_images.json`. Test không thay đổi danh sách train/validation.

**Giao thức đã điều chỉnh: không so trực tiếp với benchmark FER2013 nguyên bản.** Không có ID được xác minh nên chưa chứng minh được tách người/video; chỉ kiểm tra tách đường dẫn/pixel đã giải mã.

## Phương pháp

EmotionCNN-v1 khởi tạo ngẫu nhiên, không pretrained: ba khối hai convolution với kênh 16/32/64, BatchNorm/ReLU/MaxPool, dropout2d 0,1/0,15/0,2; adaptive pool 3×3, FC128, dropout 0,4, head bảy logits. Grayscale 48×48, chuẩn hóa [-1,1].

Huấn luyện 15 epoch đầy đủ bằng Adam lr=0.001, weight decay=0.0001, batch=64, seed=42. Cross-entropy có trọng số căn bậc hai nghịch tần suất lớp từ train. Train được lật ngang xác suất 0,5 và dịch tối đa 2 pixel; validation/test không augmentation. Scheduler giảm lr theo macro-F1 validation. Cache tensor trong RAM không đổi pixel.

Chọn **epoch 15** bằng macro-F1 validation cao nhất, hòa thì loss thấp nhất. Thiết bị cpu, 4 threads; thời gian gồm nạp dữ liệu và train: 11.1 phút. Không dùng test để chọn checkpoint.

## Kết quả

| Chỉ số | Validation dùng chọn mô hình | PrivateTest đã lọc |
| --- | ---: | ---: |
| Accuracy | 57.72% | 58.36% |
| Macro-F1 | 52.18% | 49.84% |
| Balanced accuracy | 52.97% | 50.91% |

Validation chịu ảnh hưởng của việc chọn mô hình; kết quả test được tính sau khi đã khóa checkpoint. Macro-F1 là trung bình F1 của bảy lớp; balanced accuracy là trung bình recall. Lớp ít ảnh như disgust cần được đọc riêng, không chỉ xem accuracy tổng.

| Lớp | Số ảnh test | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| angry | 452 | 48.65% | 47.79% | 48.21% |
| disgust | 38 | 18.60% | 21.05% | 19.75% |
| fear | 479 | 41.40% | 18.58% | 25.65% |
| happy | 845 | 77.24% | 87.57% | 82.09% |
| sad | 574 | 44.95% | 53.48% | 48.85% |
| surprise | 287 | 64.50% | 68.99% | 66.67% |
| neutral | 596 | 56.52% | 58.89% | 57.68% |

Baseline luôn đoán lớp phổ biến nhất đạt accuracy 25.83%. Mức này chỉ là tham chiếu từ phân bố nhãn, không phải một mô hình đã được chọn bằng test.

### Confusion matrix

Hàng là nhãn thật; cột là dự đoán.

| Thật / dự đoán | angry | disgust | fear | happy | sad | surprise | neutral |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| angry | 216 | 15 | 24 | 41 | 90 | 7 | 59 |
| disgust | 21 | 8 | 5 | 1 | 2 | 1 | 0 |
| fear | 91 | 11 | 89 | 44 | 124 | 63 | 57 |
| happy | 15 | 0 | 12 | 740 | 25 | 19 | 34 |
| sad | 55 | 6 | 31 | 56 | 307 | 9 | 110 |
| surprise | 11 | 1 | 32 | 27 | 8 | 198 | 10 |
| neutral | 35 | 2 | 22 | 49 | 127 | 10 | 351 |

### Diễn biến huấn luyện

| Epoch | Train loss có trọng số | Val loss không trọng số | Val accuracy | Val macro-F1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.8088 | 1.6634 | 36.30% | 24.11% |
| 2 | 1.6297 | 1.4124 | 46.04% | 36.51% |
| 3 | 1.4984 | 1.3135 | 48.82% | 37.61% |
| 4 | 1.4349 | 1.2774 | 51.66% | 44.13% |
| 5 | 1.3941 | 1.2402 | 51.43% | 43.45% |
| 6 | 1.3695 | 1.2065 | 53.37% | 44.90% |
| 7 | 1.3474 | 1.1882 | 54.80% | 46.42% |
| 8 | 1.3306 | 1.1826 | 54.04% | 46.79% |
| 9 | 1.3069 | 1.1679 | 55.14% | 48.04% |
| 10 | 1.2941 | 1.1676 | 54.49% | 47.82% |
| 11 | 1.2796 | 1.1365 | 56.09% | 48.92% |
| 12 | 1.2736 | 1.1307 | 56.40% | 49.40% |
| 13 | 1.2648 | 1.1179 | 57.02% | 51.35% |
| 14 | 1.2454 | 1.1054 | 57.94% | 51.12% |
| 15 | 1.2404 | 1.1122 | 57.72% | 52.18% |

## Suy luận và kiểm tra thời gian

Đã đối chiếu API một crop với kết quả batch trên ảnh đầu tiên của mỗi lớp: bảy kết quả khớp, không chọn ví dụ theo việc dự đoán đúng. Trên CPU 4 threads, 10 warmup + 100 lượt đo: median **1.11 ms**, p95 **1.37 ms** cho một crop RGB đã nằm trong RAM.

Phép đo gồm chuyển grayscale/resize + CNN + softmax và tạo dictionary. **Không gồm** camera, định vị mặt, liveness, identity, đọc tệp hoặc giao diện; không phải FPS đo bằng webcam thật. Không tự bật camera trong quá trình làm phần này.

## Tích hợp và tệp bàn giao

- `results/emotion_fer/best_model.pt`: checkpoint đã chọn; config có nhãn, recipe, provenance và hash mã.
- `results/emotion_fer_test/metrics.json`, `predictions.csv`, `confusion_matrix.csv`, `inference_check.json`.
- `emotion_recognition.webcam`: demo camera ROI cố định, người dùng căn mặt; Q/Esc thoát. Có thể thêm checkpoint liveness để bỏ qua biểu cảm nếu không live. Chưa tự phát hiện/theo dõi nhiều người.
- `AttendanceVerificationPipeline`: cả ba module dùng cùng crop; emotion chỉ để hiển thị, lỗi emotion không tự làm thay đổi quyết định liveness/identity.
- Mã tích hợp và xử lý frame đã có kiểm thử; chưa đo camera thật và chưa thử đầy đủ ba mô hình thật do phần 2 còn thiếu dữ liệu/checkpoint identity.

## Giới hạn và điểm dừng

Biểu cảm ngoài đời khác ảnh FER2013 đã căn mặt; nhãn có nhiễu, lớp mất cân bằng, mô hình có thể nhầm. Chưa có đo chất lượng đa dạng ánh sáng/góc mặt/thiết bị webcam. Kết quả này là baseline phân loại biểu cảm, không xác định trạng thái nội tâm hoặc năng lực làm việc.

Đã hoàn thành bước triển khai, huấn luyện và đánh giá baseline phần 4. Dừng để người dùng xem; chưa chuyển sang phần 5 hoặc GUI/đăng ký/chấm công phần 6.
