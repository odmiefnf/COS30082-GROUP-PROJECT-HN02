# Các bộ dữ liệu của dự án

## Lựa chọn hiện tại, ngày 27/10/2026

| Thành phần | Bộ dữ liệu | Nguồn | Trạng thái |
| --- | --- | --- | --- |
| Phần 3: chống giả mạo | **LCC-FASD**, thay CelebA-Spoof để nhẹ hơn | [Kaggle faber24/lcc-fasd](https://www.kaggle.com/datasets/faber24/lcc-fasd) | Đã tải/giải nén/đọc đủ ảnh, train 10 epoch; evaluation đã lọc AUC=0,823907, ACER=26,05%. [Báo cáo](03_lcc_fasd_results.md). |
| Phần 4: biểu cảm | FER2013 v1 | [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013) | Đã tải/giải nén/đọc 35.887 ảnh, train CNN từ đầu 15 epoch; PrivateTest đã lọc accuracy=58,36%, macro-F1=0,4984. [Báo cáo](04_fer2013_results.md). |
| Phần 2/5: xác minh danh tính | Bộ ảnh và cặp gốc của đề | [Kaggle homework 2 part 2](https://www.kaggle.com/competitions/11-785-fall-20-homework-2-part-2/data) | Đã nhập/kiểm tra 465.738 tệp, giải mã 465.735 ảnh; 4.000 ID train và 8.805 cặp val. Hai mô hình train đủ 10 epoch; AUC classification 0,825601, triplet 0,859969. Có ảnh trùng và recall thấp ở ngưỡng chặt. [Kết quả](06_identity_training_results.md). |

LCC-FASD có 18.827 ảnh (1.942 thật, 16.885 giả), tổng 5.189.883.545 byte theo metadata: **5,19 GB / 4,83 GiB dữ liệu**, không phải dung lượng ZIP. [Bài gốc](https://pureportal.spbu.ru/files/51584200/CSIT2019_Grishkin_Timoshenko.pdf) cho biết bộ này tập trung giả mạo qua màn hình; cần kiểm thử ảnh in riêng.

Adapter `anti_spoofing.prepare_lcc_fasd` loại 202 ảnh train theo 12 mã ID trùng development, rồi loại 4 ảnh train trùng nội dung validation. Giữ nguyên 2.948 validation; train còn 8.093 ảnh. Evaluation loại 58 ảnh trùng nội dung train/validation, còn 7.522 ảnh. Các ảnh gốc vẫn được giữ và mọi loại trừ được ghi trong manifest summary/inventory. Evaluation không có ID trong tên ảnh nên chưa xác minh được tách danh tính. Đây là giao thức đã điều chỉnh, không báo cáo như benchmark nguyên bản.

Lưu ZIP thủ công thành `data/downloads/lcc-fasd.zip`, giải nén tại `data/LCC_FASD`. Xem [hướng dẫn phần 3](03_anti_spoofing.md) về cấu trúc và lệnh chạy `--crop-mode full`.

LCC-FASD phục vụ nhãn thật/giả và FER2013 phục vụ cảm xúc. Chúng không thay bộ dữ liệu danh tính hoặc `verification_pairs_val.txt` của phần 2. Cặp tự tạo không được báo cáo như kết quả trên file cặp gốc của đề.

## Lựa chọn cũ: CelebA-Spoof

Nguồn: [kho chính thức CelebA-Spoof](https://github.com/ZhangYuanhan-AI/CelebA-Spoof). Người dùng đã yêu cầu hủy lượt tải tự động; 48 phần tải dở và cache của lượt đó được xóa ngày 21/10. Không tự khởi động lại. `data/download_status.json` ghi lượt CelebA, không phải LCC-FASD.

Mã CelebA được giữ để tham khảo; kế hoạch chính phần 3 hiện dùng LCC-FASD. Các module huấn luyện không tự tải dataset hoặc mô hình pretrained.
