# Liveness khi đeo kính

Ngày 27/10/2026, người dùng xác nhận: **vẫn có khung khuôn mặt, nhưng Liveness báo Not passed khi đeo kính**. Vì vậy đây là phản hồi về bộ phân loại liveness sau bước định vị, không phải chỉ do bộ Haar mất mặt.

## Giả thuyết và giới hạn

Gọng kính, che khuất và phản xạ có thể làm thay đổi các tín hiệu ảnh mà mô hình dùng. Đây là giả thuyết cho trường hợp người dùng; chưa có thí nghiệm có kiểm soát để xác nhận nguyên nhân chính. Nghiên cứu [Face Anti-Spoofing with Human Material Perception](https://arxiv.org/abs/2007.02157) mô tả việc các mô hình PAD sử dụng tín hiệu vật liệu, kết cấu và phản xạ; bài đó không chứng minh nguyên nhân hay giải pháp cho checkpoint hiện tại.

Baseline LCC-FASD được huấn luyện không augmentation, không weight decay, chọn epoch 1 theo validation loss. Training loss tiếp tục giảm trong khi validation loss tăng, phù hợp dấu hiệu overfit. Baseline tại ngưỡng 0,072996 có validation APCER 34,41% / BPCER 10,12% / ACER 22,27%; test đã lọc có APCER 22,80% / BPCER 29,30% / ACER 26,05%. Đây là số liệu trên các split LCC cụ thể, không phải tỷ lệ lỗi riêng khi đeo kính trên webcam.

## Bản ứng viên độc lập

Giữ nguyên `results/liveness_lcc/best_model.pt`. Tạo ứng viên ở `results/liveness_camera_aug` với cùng kiến trúc LivenessCNN RGB112, khởi tạo ngẫu nhiên, 10 epoch đầy đủ trên 8.093 ảnh train, validation 2.948 ảnh; seed42, batch64, Adam LR0,0005, weight decay0,0001, GPU RTX3060Laptop. Không dùng webcam làm dữ liệu huấn luyện.

Augmentation chỉ áp dụng khi train: flip ngang p0,5; brightness/contrast 0,75–1,25; patch trong suốt sáng/tối vùng mặt trên p0,2, diện tích 1–6%, opacity0,25–0,6. Hai lớp thật/giả có cùng phân phối augmentation. Patch là mô phỏng che khuất đơn giản, không phải ảnh kính chân thực hoặc dữ liệu kính có nhãn. Ý tưởng giảm nhạy với che khuất có liên quan đến [Random Erasing](https://arxiv.org/abs/1708.04896), nhưng không suy ra đảm bảo cải thiện liveness khi đeo kính.

Tensors gốc/cache không bị sửa. Validation/test và tiền xử lý inference không được augmentation. Checkpoint ứng viên chọn bằng ACER validation thấp nhất, validation loss phá hòa; ngưỡng ứng viên được hiệu chỉnh từ validation của chính mô hình, không lấy điểm webcam hoặc test để hạ ngưỡng.

## Kiểm tra mã

Lượt cuối **105/105 kiểm thử đạt trong 77,378 giây**. Bốn bài augmentation kiểm tra không đổi dữ liệu nguồn/cache, tái lập với seed, hình dạng/dtype/range và phạm vi patch. Hai bài training mới xác nhận augmentation không chạy ở validation và chọn/calibrate từ đầy đủ validation. Hai bài hiệu chỉnh bổ sung kiểm tra giới hạn APCER và điểm trùng không bị tách sai. Nhật ký cuối: `results/app_check_20261027/liveness_glasses_final_tests.log` (lượt trước103bài ở `liveness_camera_full_tests.log`).

## Thử camera có kiểm soát

`tools/check_liveness_camera.py` so sánh hai mô hình trên cùng crop RGB hiện tại, giữ ngưỡng validation riêng, không lưu ảnh/video và không ghi hồ sơ/điểm danh. Có thể ghi điều kiện `glasses`, `no-glasses`, `screen-replay` hoặc `print`. Nhãn điều kiện do người thực hiện cung cấp; các khung liên tiếp không độc lập, nên số khung Passed không được báo cáo như accuracy/FAR/FRR.

Nên thử cùng ánh sáng/góc mặt, đeo và bỏ kính; sau đó thử ảnh in và màn hình riêng. Không duy trì quyết định Passed từ khung trước khi khung mới thất bại. Một bản cải thiện phải được xét cùng khả năng nhận ảnh giả, không chỉ tăng số lần chấp nhận người thật.

## Kết quả thí nghiệm

Ứng viên đã hoàn tất 10 epoch, chọn epoch10 bằng validation, AUC validation0,8745 / AUC test0,9291 (baseline0,8296 /0,8239). Tại ngưỡng ACER validation thấp nhất của ứng viên (0,040205), APCER test giảm còn6,88% nhưng BPCER test29,69%, chưa giảm bỏ sót người thật so với baseline29,30%.

Lượt webcam đầu khi người dùng báo đang đeo kính gồm240frame/8,84giây: 106crop được chấm;77frame không tìm thấy mặt;57frame có nhiều box nên bị chặn. Trên cùng106crop, baselinePassed102, ứng viên ngưỡng0,040205Passed90. Vì vậy ứng viên ở điểm vận hành đó không được thay vào app. Không quy tất cả frame bị chặn thành lỗi kính hoặc lỗi liveness.

Một điểm vận hành thứ hai được tính **chỉ từ validation**, giữ APCER validation không vượt baseline34,41%, rồi chọn TPR cao nhất. Giữ nguyên trọng số epoch10, lưu checkpoint dẫn xuất riêng `results/liveness_camera_matched/best_model.pt`, threshold0,00214453. Điểm này có APCER validation33,94% / BPCER13,58%; BPCER validation vẫn cao hơn baseline10,12%. Trên dự đoán test đã lưu của cùng trọng số, APCER19,10% / BPCER11,33% / ACER15,22%, so với baseline22,80% /29,30% /26,05%.

Ngưỡng thấp hơn không có nghĩa tự động tốt hơn; đây là thang điểm của mô hình mới và một lựa chọn đánh đổi được hiệu chỉnh bằng validation. Split test cũ được dùng kiểm tra hồi quy, không phải tập triển khai độc lập mới và không được dùng chọn ngưỡng. Dữ liệu test ẩn danh không xác minh được tách danh tính. Không dùng điểm webcam để hiệu chỉnh hai ngưỡng.

Lượt webcam thứ hai khi người dùng xác nhận đeo kính thu240frame/8,22giây, chỉ20crop đủ điều kiện để chấm;220frame không tìm thấy mặt. Trên cùng20crop đó, baselinePassed15/20 và bản hiệu chỉnhPassed20/20. Đây là mẫu nhỏ với nhiều khung không được chấm, không được diễn giải như100%accuracy hoặc một kiểm chứng kính đầy đủ. Không có lượt bỏ kính đối chứng tương ứng, cũng chưa có thử ảnh in/màn hình trực tiếp trong hai lượt này.

## Bản đang dùng trong app

Đã chọn `results/liveness_camera_matched/best_model.pt` cho demo mặc định của `attendance_app.gui`. File cấu hình ghi nhận checkpoint/SHA256: `results/liveness_app_config.json`. Bản cũ và cả điểm vận hành ACER của ứng viên vẫn giữ riêng. Không thay đổi encoder identity, biểu cảm hay hồ sơ đã đăng ký.

Đã mở app thật, bật camera và quan sát người đeo kính: có box khuôn mặt, `LIVENESS: Passed`, hồ sơ đã có được nhận ra. Đây là kiểm tra trạng thái trực tiếp, không phải đánh giá độ chính xác trên nhiều người. Agent không thực hiện đăng ký thêm hồ sơ.

Kiểm tra tích hợp bằng24ảnh train với ba CNN và DB tạm:21crop qua liveness/chạy identity+expression,2probe chưa đăng ký được so khớp nhầm với gallery,1dòng điểm danh trong DBtạm. Những probe này cho thấy giới hạn identity vẫn còn dù liveness dễ dùng hơn. Không dùng lượt này để tính độ chính xác hoặc xác nhận triển khai an toàn. Kết quả: `results/liveness_camera_matched/attendance_integration.json`.

Quay về liveness baseline khi cần:

```powershell
.\.venv-gpu\Scripts\python.exe -m attendance_app.gui --liveness-checkpoint results/liveness_lcc/best_model.pt
```

Trạng thái thí nghiệm và quyết định: `results/liveness_camera_aug/experiment_status.json`. Ảnh/video của các lượt so sánh không được lưu; bản gốc được giữ nguyên. Báo cáo Word phần8 vẫn ghi mốc baseline trước các lần chỉnh app và thí nghiệm kính; cần dùng ghi chú này để phân biệt kết quả mới.
