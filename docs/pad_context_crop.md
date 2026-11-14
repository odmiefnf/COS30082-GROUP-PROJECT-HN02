# Thử vùng cắt rộng cho chống giả mạo

Vùng cắt thử nghiệm là hình vuông tâm tại khung mặt, cạnh bằng hai lần
cạnh dài nhất của khung, giới hạn trong ảnh gốc. App lấy vùng này từ khung
camera gốc cho checkpoint đã khai báo `context_crop`. Nhận diện danh tính,
biểu cảm và kiểm tra chất lượng dùng vùng mặt ban đầu. Model Tests tự chọn
vùng theo checkpoint và ghi chính sách cắt cho từng model.

## So sánh trên cùng ảnh LCC-FASD

Hai CNN khởi tạo từ đầu và chạy 10 epoch, dùng cùng ảnh nguồn, bộ phát
hiện, kiểm tra chất lượng và công thức huấn luyện. Mỗi bản chọn ngưỡng từ
305 ảnh validation và giữ nguyên ngưỡng khi kiểm tra. Mỗi bản có 825 ảnh
train và 271 ảnh test được chấm, gồm 137 ảnh thật và 134 ảnh giả.

| Trên 271 ảnh được chấm | Cắt sát mặt | Vùng rộng 2× |
| --- | ---: | ---: |
| Ảnh giả lọt qua (APCER) | 22/134 (16,42%) | 24/134 (17,91%) |
| Ảnh thật bị từ chối (BPCER) | 64/137 (46,72%) | 51/137 (37,23%) |
| ACER | 31,57% | 27,57% |
| AUC | 0,7742 | 0,8089 |

Vùng rộng giảm nhầm người thật và làm lọt thêm hai ảnh giả. 229/500 ảnh
test được lấy mẫu bị loại trước khâu chấm. Ảnh nguồn vốn đã được cắt,
không có ảnh gốc rộng hơn để tái tạo viền điện thoại hoặc tờ giấy bên ngoài
khung. Đây là tập con của LCC-FASD đã dùng trong dự án. Tập evaluation thiếu
nhãn loại tấn công và thông tin danh tính đủ để kiểm tra tách người. Chưa có
kết quả webcam mới cho hai bản này. App mặc định vẫn dùng checkpoint cũ.

Tái lập từ manifest LCC-FASD đã chuẩn bị. Lệnh tạo hai checkpoint cục bộ
và kết quả tổng hợp. Chọn tên thư mục mới khi chạy lại.

```powershell
python -m tools.compare_pad_context --output-dir results/pad_context_repeat --device auto
```

Để xem kết quả mẫu, mở
[metrics.json](../results/pad_context_20261114/metrics.json). Muốn thử ở
webcam, chọn checkpoint trong `work/context-2x/model/` bằng nút **Model
Tests…**. Báo cáo tách theo tình huống ảnh tĩnh trên điện thoại, video phát
lại, ảnh in và người thật có hoặc không kính. Nên chia lượt đo theo người
và thiết bị khi có thêm người tham gia. Báo cáo tổng hợp ảnh tĩnh cũ vẫn
giữ nguyên phạm vi và kết quả lịch sử.

`tools.capture_pad_data capture --context-crop` dùng cửa sổ thu mẫu riêng.
Q/Esc trong màn căn mặt hủy phiên trước khi lưu ảnh. Các phiên vùng rộng
được tách bằng `split --context-crop` rồi huấn luyện bằng `--context-crop
--crop-mode full`. Thư mục ảnh thu và các checkpoint huấn luyện bằng ảnh cá
nhân ở ngoài gói nộp. Ý tưởng giữ vùng quanh mặt được tham khảo từ
[mô tả FaceGuard AI](https://github.com/henryhoang207/Facial-Recognition---COS30082/blob/a2266a909adbffc3657d7e1275c30db83e650d1e/README.md#L106-L120).
