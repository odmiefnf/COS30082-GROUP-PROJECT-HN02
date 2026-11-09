# Bước 5: Quy trình đánh giá webcam

Ngày chuẩn bị: **30/10/2026**. Tài liệu này quy định cách thu và diễn giải bằng chứng, không phải bảng kết quả đã đạt. Kết quả chỉ được bổ sung từ từng JSON thực tế có trạng thái `completed`; lượt thiếu người tham gia hoặc vật mẫu giữ là **chưa thực hiện**.

Mục tiêu là kiểm tra app hiện tại với người đã đăng ký khi có/không kính, người chưa đăng ký và ảnh giả mạo. Đây là phép thử thuận tiện ở quy mô nhỏ, không thay thế ROC/AUC trên 8.805 cặp gốc hoặc một đánh giá triển khai độc lập nhiều người/thiết bị.

## Chuẩn bị cùng người tham gia

1. Chỉ bắt đầu khi người tham gia đã sẵn sàng và tự xác nhận điều kiện. Với người đã đăng ký, xác nhận **đúng tên hồ sơ hiện có**, ví dụ `kiet` chỉ khi chính người dùng xác nhận. Không suy ra danh tính, kính, ảnh thật/giả hoặc tình trạng đăng ký từ dự đoán của CNN.
2. Để camera ngang tầm mắt, đặt máy chắc chắn, dùng ánh sáng phía trước và khoảng cách thoải mái đủ thấy toàn bộ mặt. Giữ vị trí/ánh sáng tương tự cho lượt có/không kính; ghi mọi thay đổi vào ghi chú. Nếu tháo kính bất tiện, có thể để lượt đó chờ thay vì ép thực hiện.
3. Bấm **Stop** trong app và đóng ứng dụng đang giữ webcam trước khi chạy công cụ đo. **Test camera** chỉ xác nhận một frame, không được tính như lượt nhận diện đầy đủ. Kiểm tra chỉ số thiết bị nếu có nhiều camera.
4. Ghi camera, ánh sáng, khoảng cách ước lượng, ngày/giờ, tên/ID lượt, người xác nhận nhãn và danh sách appearance đang có của hồ sơ mục tiêu. Giữ gallery/reference cố định giữa các lượt so sánh. Không thêm appearance hay thay checkpoint giữa chừng rồi gộp thành cùng một cấu hình.
5. Thống nhất trước giới hạn mỗi lượt, chẳng hạn tối đa 15 giây. Khuyến nghị **ít nhất ba lần thử riêng cho mỗi điều kiện**, khởi động một lượt mới và để người/vật mẫu rời rồi vào lại khung. Đây là khuyến nghị của quy trình, **không phải số lượt do đề bắt buộc**. Nhiều lượt cùng một người/camera vẫn chưa đại diện cho nhiều người hoặc thiết bị độc lập.

## Các điều kiện cần thu

| Điều kiện | Nhãn và đối chứng do người thực hiện cung cấp | Trạng thái cần theo dõi |
| --- | --- | --- |
| Có kính | `glasses`, tên chính xác của hồ sơ người thật | Đã có [lượt glasses_01](../results/app_upgrade_20261030/evaluation/trials/glasses_01.json); chưa đủ lượt lặp và đối chứng không kính |
| Không kính | `no-glasses`, **cùng hồ sơ** và cấu hình như có kính | Chờ người tham gia xác nhận sẵn sàng; không dùng số liệu có kính thay thế |
| Người chưa đăng ký | `unknown`, người tự xác nhận chưa có hồ sơ trong gallery đang thử | Chờ người tham gia phù hợp; không coi một người đã có hồ sơ nhưng bị Unknown là ground truth chưa đăng ký |
| Ảnh trên điện thoại | `screen-replay` và bắt buộc khai báo `phone-still-photo` hoặc `recorded-video` | Chờ vật mẫu; không gọi ảnh tĩnh trên màn hình là video replay |
| Ảnh in | `print`, `printed-photo`, ảnh của hồ sơ mục tiêu và loại bản in | Chờ bản in phù hợp; không có máy in thì giữ pending, không cần mua để hoàn tất lượt hiện có |

Với ảnh màn hình/bản in, người thực hiện tự chỉ rõ ảnh thuộc hồ sơ mục tiêu nào và xác nhận có thể sử dụng ảnh đó. Giữ người thật ngoài khung, chỉ đưa vật mẫu trước camera. Ghi thiết bị màn hình, độ sáng, ảnh tĩnh hay video, màu/đen trắng, giấy và kích thước tương đối nếu biết. Không suy ra loại tấn công từ việc PAD đạt hay không đạt.

Khi dùng `screen-replay`, công cụ bắt buộc `--presentation-type phone-still-photo` cho ảnh tĩnh trên điện thoại hoặc `--presentation-type recorded-video` cho video đã ghi. Với `print`, dùng `--presentation-type printed-photo`. Đây là nhãn do người thực hiện xác nhận, không suy ra từ camera. Các điều kiện chưa có bằng chứng đều là **pending**, không được ghi thành không lỗi/0% chấp nhận giả.

## Chạy công cụ đo

`tools/evaluate_webcam_trial.py` dùng đúng các mô hình và `FrameProcessor` của app. Công cụ đọc gallery gốc ở chế độ chỉ đọc, tạo bản sao tạm để chạy quy tắc điểm danh rồi dọn bản sao. Lịch sử hoặc hồ sơ đang dùng không được ghi thêm. Không gọi luồng đăng ký hay Manual review trong lượt đánh giá tự động này.

Ví dụ cho người đã tự xác nhận hồ sơ `kiet` và đang đeo kính, chạy tại `grp_project`:

```powershell
.\.venv-gpu\Scripts\python.exe -m tools.evaluate_webcam_trial --condition glasses --expected-profile "kiet" --camera 0 --seconds 15 --max-frames 1500 --output results/app_upgrade_20261030/evaluation/trials/glasses_02.json
```

Thay `--condition` thành `no-glasses` cho đối chứng. Điều kiện `unknown` **không truyền `--expected-profile`**. Mỗi lần dùng tên tệp mới; công cụ không ghi đè bằng chứng cũ. Lưu các lượt thật trong **`results/app_upgrade_20261030/evaluation/trials/`** để báo cáo tự đọc đúng nguồn; không đặt JSON mô phỏng/unit-test trong thư mục bằng chứng này.

Ví dụ cho ảnh tĩnh của hồ sơ đã xác nhận trên điện thoại:

```powershell
.\.venv-gpu\Scripts\python.exe -m tools.evaluate_webcam_trial --condition screen-replay --presentation-type phone-still-photo --expected-profile "kiet" --camera 0 --seconds 15 --max-frames 1500 --output results/app_upgrade_20261030/evaluation/trials/phone_photo_01.json
```

Chỉ đổi thành `recorded-video` nếu thực sự đang phát video đã ghi. Với ảnh in, dùng `--condition print --presentation-type printed-photo` và tên tệp mới. `screen-replay`/`print` vẫn cần tên chính xác của hồ sơ mục tiêu. Có thể bổ sung `--environment`, `--notes` và `--distance-cm` để ghi ánh sáng/bố trí/khoảng cách do người thực hiện xác nhận; chưa đo khoảng cách thì không điền một giá trị đoán.

Công cụ có phần khởi động camera loại khỏi mẫu số, rồi báo `CAPTURE_STARTED`. Giữ đúng điều kiện đã khai báo trong lúc đo, chỉ xoay mặt nhẹ như cách dùng app thông thường. Lượt dừng khi đạt thời lượng **hoặc** giới hạn frame, vì vậy luôn báo thời gian/frame thực tế trong JSON. `--seconds 15` không tự chứng minh đã đo đủ 15 giây. Driver OpenCV có thể chậm ở lệnh mở/đọc, nên giới hạn thời lượng không phải cam kết timeout ở mức driver.

## Dữ liệu được giữ

Chỉ giữ số đếm, trạng thái, tỷ lệ mô tả, thời gian, cấu hình/hash/ngưỡng mô hình, thông tin điều kiện do người thực hiện nhập và kết quả điểm danh mô phỏng của bản sao tạm. **Không lưu ảnh/video, crop hoặc embedding camera vào báo cáo JSON.** Các frame chỉ xử lý trong RAM. Tên hồ sơ trong bằng chứng vẫn là thông tin nhận diện; chỉ dùng đúng tên đã được xác nhận cho lượt thử.

Không điều chỉnh ngưỡng identity/PAD, chọn lại checkpoint hoặc thêm mẫu dựa trên kết quả các lượt này rồi báo chúng như test độc lập của cấu hình mới. Nếu cần sửa hệ thống, giữ nguyên bằng chứng thất bại, đặt tên phiên bản mới và thu một loạt đối chứng mới. Đánh giá expression accuracy cần nhãn biểu cảm độc lập; số frame chạy expression không chứng minh dự đoán biểu cảm đúng, càng không suy ra trạng thái tâm lý.

Bản ngày 08/11/2026 chạy expression trên crop đạt chất lượng dù PAD không đạt; identity và mọi thao tác lưu vẫn yêu cầu PAD đạt. Hai lượt tháng 10 được giữ nguyên là bằng chứng của phiên bản lúc đo. Người dùng đã yêu cầu để các lượt webcam mới thử sau; không tái gán kết quả lịch sử cho luồng mới hoặc gọi kiểm thử phần mềm là một lượt người thật mới.

## Đọc kết quả đúng mẫu số

| Mức đo | Nên báo | Không được suy diễn |
| --- | --- | --- |
| Một lượt người thật đã đăng ký | Có tạo dòng mô phỏng đúng người không, có dòng sai người không, thời gian đến dòng đầu, tổng thời gian | Một thành công không chứng minh luôn nhận đúng khi đeo/bỏ kính |
| Một lượt người chưa đăng ký | Có bất kỳ dòng mô phỏng cho hồ sơ trong gallery không; nếu có thì lượt đó chấp nhận sai | Một lượt không có điểm danh chưa đủ chứng minh gallery FAR bằng 0 |
| Một lượt ảnh giả mạo | PAD có qua không; có nhận thành mục tiêu/bất kỳ hồ sơ không; có điểm danh mô phỏng không | Không được đánh đồng chặn ở detector/quality với PAD đã phân loại đúng |
| Các frame trong một lượt | Một mặt/tổng frame, quality/tổng frame, PAD Passed/crop đã chấm và Passed/tổng frame, identity đúng/probe đã chấm và đúng/tổng frame | Frame liên tiếp tương quan; các tỷ lệ này không phải số lượt độc lập, accuracy dân số hay FAR/FRR được bảo đảm |
| Nhiều lượt riêng | Số lượt đã hoàn tất và số lượt chấp nhận đúng/sai theo từng điều kiện, kèm số người/camera | Không gộp điều kiện thiếu thử thành số không, không trộn dữ liệu trước/sau đổi cấu hình |

`simulated_attendance` trong JSON là điểm danh **trong bản sao tạm**, không phải ghi nhận hiện diện thật. Với lượt genuine, một dòng đúng người là kết quả thành công ở mức lượt; ghi riêng mọi dòng sai người. Với unknown/attack, bất kỳ dòng nào được chấp nhận là lỗi của lượt. `first_record_seconds` là thời gian tới **dòng đầu tiên bất kỳ**, nên phải đối chiếu dòng đó đúng/sai trước khi gọi là thời gian nhận đúng.

Nếu không có crop qua PAD hoặc không có probe identity, tỷ lệ tương ứng là `null`/**không đủ mẫu số**, không phải 0%. Báo cả số frame no face/nhiều mặt/quality fail. Một lượt có lỗi camera hoặc mô hình là `not_completed`, giữ lý do; không tính là một lượt từ chối người/ảnh thành công. Việc không phát hiện mặt hoặc không đạt chất lượng chỉ chứng minh luồng đã chặn ở bước đó trong lượt ấy, không chứng minh PAD nhận đúng giả mạo.

## Sổ tổng hợp và việc còn chờ

Mỗi dòng sổ lượt thử cần có: mã lượt, condition, tên hồ sơ mục tiêu (nếu có), nhãn người xác nhận, loại ảnh màn hình/bản in, ánh sáng/khoảng cách, cấu hình gallery, JSON bằng chứng, trạng thái hoàn tất và kết quả mức lượt. Không đoán kết quả từ tên tệp hoặc từ kết quả mong muốn.

Đối chiếu có/không kính, người chưa đăng ký, ảnh điện thoại và ảnh in phải giữ thành các mục riêng. Một lượt 15 giây được người dùng xác nhận chỉ hoàn thành chính lượt đó; các đối chứng và lần lặp chưa thu vẫn chờ. Báo cáo tiếng Anh phải nêu số người/số lượt thực tế và giới hạn này, không biến số kiểm thử phần mềm đạt thành accuracy webcam.

Video nhóm là đầu việc nộp bài riêng: nhóm chủ động quay theo [checklist video](09_video_checklist.md). Quy trình đo không tự ghi hoặc tải video lên mạng.
