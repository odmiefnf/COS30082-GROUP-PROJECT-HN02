# Phần 3: Anti-Spoofing / Liveness Detection

**Cập nhật 23/10/2026:** đã tải/giải nén và đọc thành công đủ 18.827 ảnh LCC-FASD; huấn luyện CNN từ đầu đủ 10 epoch, chọn epoch 1 bằng validation loss, hiệu chỉnh ngưỡng 0,072996063 trên validation. Evaluation đã lọc có AUC=0,823907; ACER=26,05%. Xem [báo cáo kết quả](03_lcc_fasd_results.md) về chỉ số, overfitting và giới hạn. Mã tích hợp đã có; chưa kiểm tra cùng mô hình identity thật do thiếu dữ liệu/checkpoint phần 2. Phần 4 hiện có [tài liệu riêng](04_emotion_recognition.md).

## 1. Mức độ phù hợp

[Bài gốc LCC-FASD](https://pureportal.spbu.ru/files/51584200/CSIT2019_Grishkin_Timoshenko.pdf) mô tả **18.827 ảnh**: 1.942 thật, 16.885 giả, với 83 thiết bị ghi và 82 thiết bị hiển thị. Ảnh đã crop quanh mặt, có vùng bao quanh. Bộ dữ liệu tập trung ảnh/video được phát trên **màn hình** rồi chụp lại, phù hợp baseline CNN chống giả mạo từ một ảnh. Chất lượng chống **ảnh in** và mặt nạ chưa được chứng minh; cần kiểm thử ảnh in riêng khi demo và nêu rõ giới hạn trong báo cáo.

Đã đối chiếu đủ metadata công khai của [bản Kaggle người dùng cung cấp](https://www.kaggle.com/datasets/faber24/lcc-fasd):

Chi tiết phép kiểm tra trước khi tải được lưu tại [lcc_fasd_metadata_audit.json](lcc_fasd_metadata_audit.json); đây là bản kiểm tra metadata tên/kích thước ở thời điểm đó.

| Tập gốc | Thật | Giả | Tổng |
| --- | ---: | ---: | ---: |
| Training | 1.223 | 7.076 | 8.299 |
| Development | 405 | 2.543 | 2.948 |
| Evaluation | 314 | 7.266 | 7.580 |
| Tổng | 1.942 | 16.885 | 18.827 |

Tổng kích thước tệp **5.189.883.545 byte = 5,19 GB = 4,83 GiB dữ liệu**, không phải kích thước ZIP. CelebA-Spoof trước đây có 71,711 GiB nén và 73,070 GiB giải nén theo archive đã kiểm tra. Về số ảnh, LCC-FASD ít hơn khoảng 33 lần so với 625.537 ảnh CelebA-Spoof. Lượt CPU hiện tại mất khoảng 3,5 phút cho 10 epoch và hiệu chỉnh, chưa tính tải, chuẩn bị/hash và nạp cache; thời gian máy khác có thể khác.

LCC-FASD chỉ thay dataset phần 3. Phần 2 vẫn cần ảnh theo danh tính và file cặp xác minh của đề; phần 4 vẫn dùng FER2013.

## 2. Tải và kiểm tra dữ liệu

Người dùng đã yêu cầu tải tự động. Công cụ `tools/download_lcc_fasd.py` lấy đúng phiên bản Kaggle v1, giữ ZIP để tiếp tục nếu mạng gián đoạn, kiểm tra danh sách và CRC khi giải nén. ZIP có 5.196.000.412 byte. Trạng thái riêng của LCC-FASD:

```powershell
Get-Content data/lcc_fasd_download_status.json | ConvertFrom-Json
```

Chỉ `status=ready` và `dataset_ready=true` xác nhận tải/giải nén hoàn tất. Khi cần tự chạy lại công cụ có thể dùng `python tools/download_lcc_fasd.py`; không chạy hai lượt cùng lúc. Sau đây là phương án tải thủ công dự phòng:

Mở [Kaggle LCC-FASD](https://www.kaggle.com/datasets/faber24/lcc-fasd), chọn **Download**, đăng nhập nếu Kaggle yêu cầu. Lưu ZIP với tên `lcc-fasd.zip` tại:

```text
C:\Users\kiet1\OneDrive\Desktop\cos30082\grp_project\data\downloads\lcc-fasd.zip
```

Giải nén để có:

```text
data/LCC_FASD/
  LCC_FASD_training/real/ và spoof/
  LCC_FASD_development/real/ và spoof/
  LCC_FASD_evaluation/real/ và spoof/
```

Adapter chấp nhận root trực tiếp hoặc một lớp `LCC_FASD` bên trong. Nó cũng hỗ trợ evaluation có ảnh `real_*.png`/`spoof_*.png` trực tiếp trong thư mục evaluation. Không cần đổi tên ảnh hoặc tạo bbox sidecar.

`data/download_status.json` là lịch sử downloader **CelebA-Spoof**, không thể dùng làm tiến độ LCC-FASD. Lượt tải CelebA tự động trước đã hủy và các phần tải dở của lượt đó đã xóa theo yêu cầu; không tự khởi động lại.

## 3. Nhãn và xử lý ảnh

| Lớp / thư mục | source_label do adapter quy ước | Nhãn mô hình |
| --- | ---: | ---: |
| real / LIVE | 0 | 1 |
| spoof / SPOOF | 1 | 0 |

Tên thư mục xác định lớp; `source_label` là quy ước số do adapter ghi lại, không phải một annotation số có sẵn của bản Kaggle. Nhãn mô hình bằng `1 - source_label`, logits `[spoof, live]`, điểm live bằng `softmax(logits)[1]`.

Huấn luyện LCC-FASD phải có **`--crop-mode full`**: ảnh đã crop, không có sidecar `*_BB.txt` của CelebA. Chuyển RGB, resize 112x112, chuẩn hóa float32 CHW trong [-1,1]. Suy luận đọc crop mode từ checkpoint. ROI camera cần có vùng bao quanh tương tự ảnh huấn luyện; OpenCV BGR phải đổi thành RGB.

## 4. Giao thức dữ liệu

Không chia ngẫu nhiên từng ảnh vì nhiều ảnh xuất phát từ cùng video.

1. Đọc training, development và evaluation gốc; development dùng làm validation.
2. Trích mã từ `_idNN_` trong train/development, không lấy từng tên ảnh làm danh tính hoặc thêm tiền tố split để che trùng mã.
3. Metadata bản Kaggle có **12 mã trùng** giữa train và development: `0, 5, 9, 12, 29, 39, 42, 43, 48, 117, 122, 123`. Adapter loại toàn bộ ảnh train mang những mã này, giữ nguyên development. Loại **202 ảnh**, tạm còn **8.097 ảnh train**. Đây là biện pháp bảo thủ theo mã tên ảnh, không phải xác nhận bằng hình ảnh rằng mọi mã trùng đều cùng người. Kiểm tra dữ liệu thật phát hiện thêm **4 ảnh train trùng byte với validation**: loại khỏi train, cuối cùng còn **8.093 ảnh** (1.193 live, 6.900 spoof). Validation giữ **2.948 ảnh** (405 live, 2.543 spoof).
4. Evaluation có **58 ảnh trùng byte với train/validation**: loại khỏi evaluation, còn **7.522 ảnh** (256 live, 7.266 spoof). Chỉ dùng hash nội dung cho quy tắc chất lượng dữ liệu này; không dùng dự đoán hoặc kết quả mô hình. Test không quyết định ảnh nào được giữ để huấn luyện. Tên `real_0.png`/`spoof_0.png` không có ID: để `subject_id` rỗng và `subject_id_source=unavailable`, không suy ra người từ số thứ tự ảnh.
5. Manifest và `split_summary.json` ghi số lượng thực tế, ID và ảnh train bị loại, hash và giao thức đã điều chỉnh. Không báo cáo là giữ nguyên hoàn toàn protocol gốc.
6. Train/validation phải có ID, không trùng mã/đường dẫn. Bước chuẩn bị dùng SHA256 để thực hiện các loại trừ công khai ở trên. Khi train/evaluate vẫn kiểm tra lại hash; nếu tiếp tục phát hiện trùng sẽ báo lỗi thay vì bỏ qua. `source_inventory.json` lưu mọi ảnh gốc với trạng thái giữ/loại và lý do; không xóa ảnh gốc khỏi dataset.
7. Chọn checkpoint bằng validation loss thấp nhất, sau đó hiệu chỉnh ngưỡng bằng toàn bộ validation. Không dùng evaluation để cập nhật trọng số, chọn mô hình hoặc ngưỡng.
8. Đánh giá evaluation bằng ngưỡng đã lưu, kiểm tra đường dẫn và SHA256 so với train/validation. Báo cáo ghi rõ **chưa xác minh được tách danh tính evaluation**. Hash chỉ phát hiện trùng theo byte, không chứng minh ảnh khác nhau không cùng người/video.

Do lọc train/evaluation và thiếu ID evaluation, kết quả thuộc giao thức đã điều chỉnh và tập evaluation con, không phải phép tái lập đầy đủ benchmark bài gốc. Bài gốc đề nghị DET/EER; baseline hiện có ROC/AUC và các tỷ lệ lỗi dưới đây, chưa xuất DET/EER.

## 5. Mô hình và lệnh chạy

CNN hai lớp khởi tạo từ đầu: bốn khối convolution 16/32/64/128 kênh, đặc trưng 128 chiều, linear head hai lớp. Adam, learning rate 0.001, batch size 32, 10 epoch, seed 42. Baseline dùng cross-entropy không trọng số, không augmentation. Cờ `--cache-images` giữ tensor train/validation đã xử lý trong CPU RAM để tránh giải mã lại mỗi epoch; pixel và thứ tự huấn luyện không đổi, cần khoảng 1,55 GiB RAM cho bộ này. Có thể so sánh class weighting dựa riêng trên train sau khi có baseline; giữ nguyên validation/evaluation.

Chạy tại `grp_project`, trong môi trường cài `requirements.txt`:

```powershell
python -m anti_spoofing.prepare_lcc_fasd --data-root data/LCC_FASD --output-dir data/lcc_fasd_manifests
python -m anti_spoofing.check_data --manifest-dir data/lcc_fasd_manifests --crop-mode full --inspect-images 5
python -m anti_spoofing.train --train-manifest data/lcc_fasd_manifests/train.csv --val-manifest data/lcc_fasd_manifests/val.csv --output-dir results/liveness_lcc --crop-mode full --epochs 10 --device auto --cache-images
python -m anti_spoofing.evaluate --checkpoint results/liveness_lcc/best_model.pt --manifest data/lcc_fasd_manifests/test.csv --output-dir results/liveness_lcc_test --device auto
```

`--max-batches 1` chỉ kiểm tra chạy thử: giới hạn train/validation trong epoch, nhưng hiệu chỉnh ngưỡng cuối vẫn dùng toàn bộ validation. Checkpoint được đánh dấu smoke test. Các kiểm tra bằng ảnh tổng hợp không phải kết quả chất lượng chống giả mạo.

## 6. Đánh giá

Dữ liệu gốc khoảng 89,7% giả; evaluation gốc khoảng 95,9% giả, tập evaluation con sau lọc khoảng 96,6% giả. Mô hình luôn đoán giả có accuracy cao nhưng từ chối mọi người thật. Báo đồng thời:

| Chỉ số | Ý nghĩa |
| --- | --- |
| AUC | Phân biệt live/spoof bằng điểm P(live) |
| APCER | Tỷ lệ ảnh giả bị chấp nhận là thật |
| BPCER | Tỷ lệ ảnh thật bị từ chối |
| ACER | Trung bình APCER và BPCER |
| Accuracy | Tỷ lệ đúng tại ngưỡng validation đã lưu |

Train tạo `best_model.pt`, `last_model.pt`, `history.csv`, `config.json`, `calibration.json`. Best chứa ngưỡng validation; last chưa hiệu chỉnh. Ngưỡng tối đa hóa validation Youden J, tương đương tối thiểu hóa ACER. Chỉ số trên validation tại ngưỡng vừa chọn là in-sample.

Evaluate tạo `predictions.csv`, `metrics.json`, `roc_points.csv`, `roc_curve.svg`, kèm provenance và giới hạn kiểm tra ID. Không tự hiệu chỉnh ngưỡng trên tập đánh giá.

## 7. Tích hợp với phần 2

`AttendanceVerificationPipeline` dùng cùng một crop RGB cho hai mô hình: kiểm tra liveness trước, chỉ xác minh danh tính nếu live đạt ngưỡng. Chỉ chấp nhận khi cả hai đều đạt. Thiếu ngưỡng sẽ từ chối quyết định chấp nhận.

Khi có checkpoint thật:

```powershell
python -m anti_spoofing.infer --checkpoint results/liveness_lcc/best_model.pt --image DUONG_DAN_ANH_CROP --face-crop
python attendance_pipeline.py --liveness-checkpoint results/liveness_lcc/best_model.pt --identity-checkpoint results/classification/best_model.pt --probe DUONG_DAN_ANH_CROP --reference ANH_THAM_CHIEU --identity-threshold NGUONG_TU_VALIDATION --face-crop
```

Phần 4 hiện có demo webcam ROI thủ công và classifier biểu cảm trong tài liệu riêng; GUI đầy đủ, đăng ký người lạ và cơ sở dữ liệu điểm danh chưa xây. Mã CelebA `anti_spoofing.prepare` và downloader cũ được giữ để tham khảo, không tự chạy lại.
