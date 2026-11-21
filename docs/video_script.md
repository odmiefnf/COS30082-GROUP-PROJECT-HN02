# Kịch bản video demo (khoảng 8 phút 30 giây, 6 người nói)

Soạn ngày 21/11/2026 cho web app và các mô hình hiện tại. Đề yêu cầu: 5–10 phút, **một lượt quay liên tục** không cắt ghép, mỗi người nói thời lượng tương đương và **lộ mặt khi nói**. Điều kiện chi tiết ở [09_video_checklist.md](09_video_checklist.md).

Thứ tự "Người nói 1–6" chỉ là chỗ trống: nhóm tự gán tên theo **đóng góp thật**. Phần tính năng sáng tạo (đoạn 6) phải do chính người làm tính năng đó trình bày; nếu nhiều người có tính năng riêng, chia đoạn 6 hoặc rút bớt các đoạn khác. Lời thoại tiếng Anh để nói trước người chấm; có thể diễn đạt lại bằng lời của mình, nhưng giữ đúng con số.

## Phân công người nói (nhóm điền)

| Đoạn | Nội dung | Thời gian | Người nói |
| --- | --- | --- | --- |
| 1 | Giới thiệu và luồng hệ thống | 0:00–1:25 | |
| 2 | Nhận diện khuôn mặt (classification, triplet, ArcFace) | 1:25–2:50 | |
| 3 | Đăng ký người mới và điểm danh | 2:50–4:15 | |
| 4 | Chống giả mạo (liveness) | 4:15–5:40 | |
| 5 | Nhận diện cảm xúc và nhiều người | 5:40–7:05 | |
| 6 | Tính năng mở rộng và kết luận | 7:05–8:30 | |

Mỗi người một đoạn, khoảng 1 phút 25 giây. Ai nói về phần mình làm chính thì dùng "I"; ai nói về phần chung của nhóm thì dùng "we" và không nhận là việc riêng của mình. Người đăng ký trực tiếp ở đoạn 3 là người nói 3, nên **không** đăng ký sẵn người này.

## Chuẩn bị trước khi quay

- Mở app với database demo riêng: `.\.venv-gpu\Scripts\python.exe -m attendance_app.web --open --database data/attendance/demo_recording.sqlite3 --session COS30082-video-demo`.
- Trước khi quay, đăng ký sẵn **năm** người nói (đoạn 3 sẽ đăng ký người thứ sáu trực tiếp trong video). Người đăng ký trong video nên đeo kính để đoạn 6 có thể thêm appearance không kính.
- Mỗi người nói đứng trước webcam của app khi nói: khung Live hiện mặt người đó, nên quay màn hình là đủ để thấy mặt. Nếu đang ở trang không có camera (Profiles, Attendance), người nói vẫn phải lộ mặt: dùng điện thoại quay cả màn hình và người nói, hoặc giữ một cửa sổ Live nhỏ cạnh trang đang trình bày.
- Đồ dùng: điện thoại có **ảnh** và **video** mặt một thành viên (đã đồng ý), **ảnh in** nếu nhóm đã thử bằng `tools\record_print_attack.cmd`. Chỉ nói kết quả của loại tấn công đã thật sự thử.
- Mở sẵn hình ROC và bảng kết quả trong báo cáo nhóm (`reports/COS30082_Group_Report.docx`) ở một cửa sổ khác.
- Tập trọn một lượt trước. Nếu quay hỏng, quay lại **cả lượt**, không ghép.

## Kịch bản

### 1. Người nói 1 — Giới thiệu và luồng hệ thống (0:00–1:25)

Màn hình: trang **Live**, camera đang tắt, rồi bấm **Start camera**.

> "Hello, we are Group HN02. Our project is a face attendance system that recognises registered people from a webcam, checks that the face is live, and shows their facial expression. The pipeline is: YuNet locates faces, we check size, sharpness and brightness, then a liveness model blocks spoof faces, an identity model matches the face against registered people, and an expression model shows the emotion. Attendance is recorded only after three consecutive live, recognised frames, and only once per person per session."

Đứng trước camera để app nhận ra mình: chỉ vào tên, **Liveness: Passed**, nhãn biểu cảm có icon.

> "All three networks start from ImageNet weights and are fine-tuned on the task data, which the brief allows. We also trained from-scratch baselines for comparison. YuNet is a pretrained face locator only; it does not do recognition, liveness or emotion."

### 2. Người nói 2 — Nhận diện khuôn mặt (1:25–2:50)

Màn hình: hình ROC trong báo cáo, sau đó trang **Settings & Info**.

> "For face verification we implemented both required approaches: classification with cross-entropy over 4,000 training identities, and metric learning with triplet loss, margin 0.2. Each was trained twice: a compact CNN from scratch, and a ResNet-18 fine-tuned from ImageNet. On all 8,805 validation pairs, the cosine AUC is 0.826 and 0.860 for the compact models, and 0.914 and 0.915 for ResNet-18."

> "Cosine similarity and Euclidean distance give the same AUC on unit-length embeddings, because squared Euclidean distance equals two minus two times cosine. As an additional approach we trained ArcFace, which adds an angular margin to the classification head. It has the best AUC, 0.915, and at a 1% false-positive rate it accepts 33.0% of genuine pairs, against 26.6% for classification and 12.4% for triplet, so the app uses ArcFace with a cosine threshold of 0.509."

> "This is still a weak operating point: about two thirds of genuine validation pairs are rejected, which is why we register many poses per person."

### 3. Người nói 3 — Đăng ký người mới và điểm danh (2:50–4:15)

Màn hình: người nói 3 (chưa đăng ký) đứng trước camera → nhãn vàng/unknown → trang **Register**.

> "When someone is not recognised, the app shows them as unknown and prompts registration. Registration captures ten poses: front, left, right, up and down. Each shot is liveness-checked on its own frame; shots marked Not live are not stored, and at least three live shots are needed."

Chụp đủ 10 ảnh (phím Space), nhập tên, lưu. Quay lại **Live**: app nhận ra tên.

> "Registration stores embeddings, so a new employee does not require retraining the classifier. Now I'm recognised, and after three stable frames my attendance is recorded."

Mở trang **Attendance**: một dòng cho mỗi người trong buổi, nguồn Automatic.

### 4. Người nói 4 — Chống giả mạo (4:15–5:40)

Màn hình: **Live**, đưa ảnh điện thoại, video điện thoại (và ảnh in nếu đã thử) trước camera; mặt thật để ngoài khung. Người thật phải rời khung **ít nhất 2 giây** trước khi đưa ảnh giả: app giữ tên và Passed cho mặt vừa điểm danh đến khi mặt đó vắng quá 2 giây.

> "For anti-spoofing we fine-tuned MobileNetV3 on the LCC-FASD dataset. In live use, that first model accepted a phone photo in a dark room, so we recorded our own webcam sessions — real faces, phone photos and phone videos — and added them to training, split by whole recording session."

Cho thấy liveness không đạt (đỏ), không ghi tên, không ghi điểm danh.

> "On held-out webcam sessions the app model accepted 0 of 52 phone-photo crops and 0 of 116 phone-video crops, and passed all 96 genuine crops; the earlier model accepted 29 and 67. The trade-off is on LCC-FASD, where its error rate rose from 5.7% to 8.9%. These webcam results come from one person, one room and one day, so they are not a general attack success rate."

Chỉ nói câu về ảnh in nếu đã có kết quả trong `results/liveness_mobilenetv3_webcam_20261120_on_print/`.

### 5. Người nói 5 — Nhận diện cảm xúc và nhiều người (5:40–7:05)

Màn hình: **Live**, thử vài biểu cảm (vui, ngạc nhiên, buồn), rồi một người khác bước vào khung.

> "The expression model is MobileNetV3 fine-tuned on FER2013 with seven labels, shown with an icon in real time. On the filtered test set it reaches 64.2% accuracy and macro-F1 0.628, compared with 58.4% and 0.498 for our from-scratch CNN. Fear and sad remain the hardest classes. The expression is shown for feedback only; it never affects attendance."

> "With several people in view, the largest face is the main person: only they are liveness-checked, show an expression and can be recorded. Other faces get a name tag — green if registered, yellow if not — but are never recorded."

### 6. Người nói 6 — Tính năng sáng tạo và kết luận (7:05–8:30)

Màn hình: **Register → Add appearance**, sau đó **Profiles**, **Manual review** nếu có tình huống hợp lệ.

> "Our extensions focus on real attendance problems. Add appearance stores extra templates for an existing person, for example without glasses, and refuses shots that clearly match someone else. Profiles can be renamed, given reference photos, or deleted after a database backup. When a live person is unknown or ambiguous, an operator can do a manual review; it is recorded with the reviewer and reason, and marked Manual, not Automatic. Finally, once a person is confirmed, the app keeps their name and liveness result until they leave the frame for two seconds, so one bad frame no longer drops them; it never adds a second record."

Thêm appearance không kính cho người đã đăng ký ở đoạn 3. Rời khung ít nhất 2 giây, rồi quay lại không đeo kính để app nhận ra từ đầu; nếu bỏ kính ngay trước camera thì tên còn hiện chỉ vì khoá tên, không chứng minh appearance có tác dụng.

> "To conclude: the system meets the brief end to end, and our main limits are the low identity recall at strict thresholds and anti-spoofing evidence from a single room; in a brighter room the liveness model can reject a genuine face, and printed photos are untested. Thank you."

## Nếu có sự cố khi quay

- Không nhận ra người đã đăng ký: nói đúng như vậy, thử lại với ánh sáng/góc mặt khác; không hạ ngưỡng, không chọn tay rồi gọi là Automatic.
- Liveness chặn nhầm mặt thật: nói đây là lỗi BPCER và giải thích; không giấu.
- Ảnh giả lọt qua: nói thật, chỉ ra identity hoặc chuỗi ba khung có chặn điểm danh hay không.
