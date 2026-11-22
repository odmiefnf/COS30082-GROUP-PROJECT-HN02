# Đánh giá nhận diện một người trong nhiều hồ sơ (1:N)

App so ảnh khuôn mặt với tất cả hồ sơ đã đăng ký. Phép đo này kiểm tra cả
người đã đăng ký và người không có trong danh sách, với cùng bộ mã hóa Triplet,
điểm cosine và khoảng cách tối thiểu 0,05 giữa hai hồ sơ đứng đầu của app.

## Cách đánh giá

- Dùng ảnh trong `classification_data/test_data` của bộ dữ liệu bài tập. Chỉ mục
  kiểm tra 465.735 ảnh trong ZIP loại 176 ảnh test có pixel RGB trùng ảnh khác
  trong ZIP. Kiểm tra mã SHA-256 của ZIP và mỗi ảnh đã chọn.
- Dùng seed 42 để chọn hai nhóm danh tính không giao nhau. Mỗi nhóm có 20 người
  trong danh sách, mỗi người một ảnh đăng ký và một ảnh truy vấn khác. Thêm 300
  ảnh từ 300 người ngoài danh sách làm truy vấn người lạ cho mỗi nhóm.
- Chọn ngưỡng trên nhóm hiệu chỉnh với mục tiêu tối đa 1% truy vấn người lạ bị
  nhận nhầm. Trong các ngưỡng đáp ứng mục tiêu, ưu tiên nhận đúng nhiều người
  đã đăng ký hơn. Nếu bằng nhau, ưu tiên ít nhận nhầm người lạ hơn rồi ngưỡng
  cosine cao hơn. Áp dụng nguyên ngưỡng này lên nhóm kiểm tra.

## Kết quả trên nhóm kiểm tra

| Ngưỡng cosine | Nhận đúng người đã đăng ký | Gán nhầm sang hồ sơ khác | Từ chối / mơ hồ đối với người đã đăng ký | Người lạ bị gán nhầm vào hồ sơ |
| --- | ---: | ---: | ---: | ---: |
| App hiện tại, 0,869540 | 2/20 (10%) | 0/20 | 16/20 từ chối, 2/20 mơ hồ | 25/300 (8,33%) |
| Hiệu chỉnh 1:N, 1,0 | 0/20 | 0/20 | 20/20 từ chối | 0/300 |

Ở nhóm hiệu chỉnh, ngưỡng app nhận đúng 3/20 người và gán nhầm 11/300 người
lạ. Với mục tiêu 1%, không có ngưỡng nào vừa đạt giới hạn vừa nhận đúng được
người đã đăng ký ở nhóm hiệu chỉnh. Ngưỡng 1,0 dẫn đến từ chối toàn bộ truy
vấn đã đo. Kết quả này không đủ để thay ngưỡng app.

Cả 640 danh tính trong thí nghiệm đều có ảnh trong tập huấn luyện của bộ mã
hóa. “Người lạ” ở đây là người không đăng ký trong danh sách mô phỏng. Mẫu
số là các lượt truy vấn trên ảnh test, với 20 hồ sơ một ảnh. Phép đo chưa
kiểm chứng khả năng nhận người chưa từng xuất hiện trong huấn luyện, webcam,
nhiều ảnh đăng ký mỗi người hay kết quả sau bộ lọc chống giả mạo.

Kết quả tổng hợp và mã kiểm tra dữ liệu được lưu ở
[`results/identity_open_set_20261114/metrics.json`](../results/identity_open_set_20261114/metrics.json).
Chạy lệnh dưới đây từ thư mục dự án với ZIP và chỉ mục kiểm tra ảnh đã chuẩn bị.

```powershell
python -m tools.evaluate_identity_open_set --output results/identity_open_set_repeat/metrics.json
```

Lệnh sau tạo chỉ mục kiểm tra ảnh từ ZIP gốc.

```powershell
python -m face_verification.audit_archive --archive data/downloads/identity_kaggle_2020.zip --output-dir data/identity_archive_audit
```
