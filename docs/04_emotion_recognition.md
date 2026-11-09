# Phần 4: Phân loại biểu cảm khuôn mặt với FER2013

## Yêu cầu và phạm vi

**Trạng thái 23/10/2026:** đã train đủ 15 epoch và chọn epoch 15. PrivateTest đã lọc đạt accuracy 58,36%, macro-F1 0,4984, balanced accuracy 50,91%. Xem [báo cáo kết quả](04_fer2013_results.md). Tổng 54 kiểm thử phần mềm đã đạt; có kiểm tra API theo frame và lỗi encoding tiếng Việt trên Windows. Chưa đo camera vật lý.

Mục 4 của đề yêu cầu nhận diện cảm xúc/biểu cảm theo thời gian thực, tự huấn luyện bằng dữ liệu công khai và tích hợp với các thành phần trước. Hiển thị biểu tượng là tùy chọn.

Triển khai hiện tại phân loại **biểu cảm nhìn thấy theo nhãn FER2013**, không xác định trạng thái nội tâm. Nhãn biểu cảm chỉ để hiển thị, không quyết định có chấp nhận chấm công hay không. Phần này có CNN, chuẩn bị dữ liệu, train/evaluate, API ảnh crop, tích hợp pipeline và demo webcam dùng ROI căn mặt thủ công. Chưa xây GUI/đăng ký danh tính của phần 6.

## Dataset và quy ước nhãn

Nguồn đã được người dùng chọn: [Kaggle FER2013, msambare, v1](https://www.kaggle.com/datasets/msambare/fer2013). Tải bằng `tools/download_fer2013.py`:

- ZIP: `data/downloads/fer2013.zip`, 63.252.113 byte.
- Giải nén: `data/FER2013`, đủ 35.887 ảnh, 56.510.189 byte.
- Trạng thái/CRC và SHA256: `data/fer2013_download_status.json`.
- Ảnh gốc 48×48, được chuyển sang grayscale; mô hình nhận float32 `[N,1,48,48]`, chuẩn hóa [-1,1].

Không lấy nhãn bằng thứ tự bảng chữ cái của thư mục. Dùng đúng thứ tự cố định:

| ID | Thư mục | Hiển thị tiếng Việt |
| ---: | --- | --- |
| 0 | angry | Giận dữ |
| 1 | disgust | Ghê tởm |
| 2 | fear | Sợ hãi |
| 3 | happy | Vui |
| 4 | sad | Buồn |
| 5 | surprise | Ngạc nhiên |
| 6 | neutral | Trung tính |

Checkpoint lưu thứ tự này; loader từ chối checkpoint có thứ tự nhãn khác.

## Giao thức dữ liệu và ảnh trùng

Bản Kaggle có hai thư mục `train`/`test`, nhưng tên tệp giữ lại nguồn gốc:

| Tiền tố tên tệp | Dùng cho | Số ảnh gốc | Số ảnh sau lọc |
| --- | --- | ---: | ---: |
| Training_ | Train | 28.709 | 27.170 |
| PublicTest_ | Validation | 3.589 | 3.562 |
| PrivateTest_ | Test cuối | 3.589 | 3.271 |

Không chia ngẫu nhiên toàn bộ 35.887 ảnh. Chuẩn bị dữ liệu đọc mọi ảnh, kiểm tra kích thước và hash SHA256 của **pixel grayscale đã giải mã**, không chỉ hash JPEG. Các quy tắc cố định, không dựa trên kết quả mô hình:

1. Trong mỗi split, giữ một ảnh khi cùng pixel/cùng nhãn; loại toàn bộ nhóm pixel có nhãn mâu thuẫn.
2. Giữ validation đã làm sạch; loại khỏi train ảnh có pixel trong validation.
3. Loại khỏi test ảnh có pixel trong train/validation. Test không thay đổi danh sách ảnh train/validation.
4. Giữ ảnh gốc; ghi mọi ảnh bị loại và lý do ở `data/fer2013_manifests/excluded_images.json`.
5. Train/evaluate kiểm tra lại hash thật so với manifest để phát hiện dữ liệu bị thay đổi.

Train loại 107 ảnh thuộc nhóm nhãn mâu thuẫn, 1.168 bản trùng trong train và 264 ảnh trùng validation. Validation loại 2 ảnh nhãn mâu thuẫn và 25 bản trùng. Test loại 4 ảnh nhãn mâu thuẫn, 15 bản trùng và 299 ảnh trùng train/validation.

**Giao thức đã điều chỉnh**, kết quả không phải điểm benchmark FER2013 nguyên bản. Không có ID được xác minh: không khẳng định các split tách biệt theo người hoặc video.

## Mô hình và huấn luyện

- `EmotionCNN-v1`, khởi tạo ngẫu nhiên; không tải pretrained weights.
- Ba khối, mỗi khối hai Conv3×3 + BatchNorm + ReLU, MaxPool; kênh 16/32/64. Dropout2d lần lượt 0,1/0,15/0,2.
- AdaptiveAvgPool 3×3, Linear 576→128, ReLU, dropout 0,4, Linear 128→7 logits.
- Adam, lr 0,001; weight decay 0,0001; batch 64; seed 42; 15 epoch đầy đủ.
- Cross-entropy có trọng số lớp `sqrt(N/(7*n_class))`, chuẩn hóa trung bình bằng 1; tính riêng từ train. Validation loss là cross-entropy không trọng số.
- Tăng cường **chỉ train**: lật ngang xác suất 0,5; dịch tối đa 2 pixel với padding lặp biên. Validation/test không augmentation.
- Cache tensor đã tiền xử lý trong CPU RAM để không đọc lại JPEG mỗi epoch.
- Chọn checkpoint bằng **macro-F1 validation cao nhất**, hòa thì validation loss thấp nhất. ReduceLROnPlateau theo macro-F1, factor 0,5, patience 3.
- Không đọc test trong quá trình train/chọn mô hình. Không dùng cảm xúc làm nhãn danh tính hoặc nhãn live/spoof.

## Lệnh chạy

Chạy tại `grp_project` trong môi trường đã cài `requirements.txt`:

```powershell
python tools/download_fer2013.py
python -m emotion_recognition.prepare --data-root data/FER2013 --output-dir data/fer2013_manifests
python -m emotion_recognition.train --train-manifest data/fer2013_manifests/train.csv --val-manifest data/fer2013_manifests/val.csv --output-dir results/emotion_fer --epochs 15 --batch-size 64 --device auto --num-threads 4
python -m emotion_recognition.evaluate --checkpoint results/emotion_fer/best_model.pt --manifest data/fer2013_manifests/test.csv --output-dir results/emotion_fer_test --device auto
python -m emotion_recognition.infer --checkpoint results/emotion_fer/best_model.pt --image ANH_KHUON_MAT_DA_CROP
```

Train xuất best/last checkpoint, history và config với provenance nguồn. Evaluate xuất predictions có đủ bảy điểm softmax, confusion matrix và metrics accuracy/macro-F1/balanced accuracy cùng precision/recall/F1 từng lớp. Macro-F1 cho mỗi lớp trọng số bằng nhau; undefined precision/recall được tính là 0. Điểm softmax không phải độ chắc chắn đã hiệu chỉnh.

## Tích hợp liveness và identity

`AttendanceVerificationPipeline` nhận thêm `emotion_recognizer` tùy chọn. Cả ba nhận cùng crop RGB; emotion tự chuyển sang grayscale 48×48 bên trong.

```text
Crop RGB -> liveness
  Không đạt / thiếu ngưỡng -> dừng; không gọi emotion hoặc identity
  Đạt -> dự đoán biểu cảm để hiển thị
       -> xác minh identity
       -> chấp nhận chỉ theo liveness + identity
```

Nếu biểu cảm dự đoán là angry/sad/... hoặc module biểu cảm lỗi, điều đó không tự làm thay đổi quyết định identity/liveness. Lỗi biểu cảm được ghi trong `emotion_error` và có thể hiển thị trạng thái không khả dụng.

Khi đã có checkpoint identity thật của phần 2:

```powershell
python attendance_pipeline.py --liveness-checkpoint results/liveness_lcc/best_model.pt --identity-checkpoint results/classification/best_model.pt --emotion-checkpoint results/emotion_fer/best_model.pt --probe ANH_CROP --reference ANH_THAM_CHIEU --identity-threshold NGUONG_ID_TU_VALIDATION --face-crop
```

**Cập nhật 27/10/2026:** đã có hai checkpoint identity thật và AUC gốc, xem [kết quả](06_identity_training_results.md). GUI phần 6 đã tích hợp cả ba mô hình và được kiểm tra offline; chưa xác nhận webcam với người thật. Xem [hướng dẫn chạy](06_user_interface.md).

## Demo webcam tùy chọn

```powershell
python -m pip install -r requirements-webcam.txt
python -m emotion_recognition.webcam --checkpoint results/emotion_fer/best_model.pt --camera 0
```

Căn một khuôn mặt vào khung cố định; `--roi X Y W H` đặt khung theo pixel. Đây là demo **ROI thủ công**, chưa tự phát hiện/theo dõi nhiều khuôn mặt. Không có detector pretrained bổ sung. Nhấn Q/Esc để thoát; chương trình giải phóng camera khi kết thúc. Không tự lưu ảnh/video hoặc tạo bản ghi chấm công.

Có thể thêm `--liveness-checkpoint results/liveness_lcc/best_model.pt` để hiển thị trạng thái PAD cùng biểu cảm. Từ bản ngày 08/11/2026, biểu cảm vẫn được phân loại trên crop đạt chất lượng khi PAD không đạt; PAD tiếp tục chặn identity và điểm danh trong app tích hợp. Đây vẫn là baseline liveness với tỷ lệ lỗi đã báo ở phần 3; cần kiểm thử webcam thực tế. Đã cài OpenCV 4.13.0.92 trong môi trường đang dùng, không thay NumPy/PyTorch. Camera không được tự bật trong quá trình làm phần này.

## Điểm dừng

Đã lưu báo cáo riêng `docs/04_fer2013_results.md`, checkpoint ở `results/emotion_fer/best_model.pt` và kết quả ở `results/emotion_fer_test`. Mốc phần 4 đã được người dùng xem và cho tiếp tục; dữ liệu phần 5 hiện được trình bày ở [tài liệu riêng](05_identity_dataset.md). GUI/đăng ký/chấm công phần 6 chưa xây.
