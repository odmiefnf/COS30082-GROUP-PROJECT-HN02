"""Write the review artifact for the completed FER2013 experiment."""
from __future__ import annotations
import csv
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]

def read(relative):
    return json.loads((PROJECT/relative).read_text(encoding="utf-8-sig"))

def main():
    splits=read("data/fer2013_manifests/split_summary.json")
    download=read("data/fer2013_download_status.json")
    config=read("results/emotion_fer/config.json"); training=config["training"]
    test=read("results/emotion_fer_test/metrics.json")
    runtime=read("results/emotion_fer_test/inference_check.json")
    metrics=test["metrics"]; validation=training["validation_metrics"]
    if test["training_smoke_test"] or test["test_used_for_model_selection"]:
        raise ValueError("This report requires completed training and untouched test selection.")
    with (PROJECT/"results/emotion_fer/history.csv").open(encoding="utf-8",newline="") as handle:
        history=list(csv.DictReader(handle))
    started=datetime.fromisoformat(training["started_at_utc"])
    completed=datetime.fromisoformat(training["completed_at_utc"])
    date=datetime.now(timezone(timedelta(hours=7))).strftime("%d/%m/%Y %H:%M UTC+7")
    lines=["# Kết quả phần 4 — FER2013", "", f"Báo cáo tạo lúc {date}.", "",
           "Đã tải FER2013, tự huấn luyện CNN phân loại bảy biểu cảm nhìn thấy và đánh giá bằng PrivateTest đã lọc. "
           "Đây là nhãn biểu cảm của dataset, không phải đo trạng thái nội tâm. Kết quả chỉ để hiển thị, không quyết định chấp nhận chấm công.", "",
           "## Dữ liệu và kiểm tra", "",
           "Nguồn: [Kaggle FER2013 v1](https://www.kaggle.com/datasets/msambare/fer2013). "
           f"Tải và kiểm tra CRC đủ {download['images_verified']:,} ảnh; bước chuẩn bị giải mã đủ {splits['images_decoded']:,} ảnh 48×48. "
           f"SHA256 ZIP: `{download['archive_sha256']}`.", "",
           "| Tập | Nhóm tệp gốc | Số gốc | Sau lọc |", "| --- | --- | ---: | ---: |"]
    for key,name,source in (("train","Train","Training_"),("val","Validation","PublicTest_"),("test","Test cuối","PrivateTest_")):
        lines.append(f"| {name} | {source} | {splits['original_splits'][key]:,} | {splits['splits'][key]['samples']:,} |")
    lines += ["", "Loại các nhóm cùng pixel nhưng nhãn mâu thuẫn; giữ một bản cho ảnh trùng trong từng split. "
              "Tiếp đó loại train trùng validation và test trùng train/validation. Không xóa ảnh gốc. "
              "Tất cả loại trừ được lưu trong `data/fer2013_manifests/excluded_images.json`. Test không thay đổi danh sách train/validation.", "",
              "**Giao thức đã điều chỉnh: không so trực tiếp với benchmark FER2013 nguyên bản.** "
              "Không có ID được xác minh nên chưa chứng minh được tách người/video; chỉ kiểm tra tách đường dẫn/pixel đã giải mã.", "",
              "## Phương pháp", "",
              "EmotionCNN-v1 khởi tạo ngẫu nhiên, không pretrained: ba khối hai convolution với kênh 16/32/64, "
              "BatchNorm/ReLU/MaxPool, dropout2d 0,1/0,15/0,2; adaptive pool 3×3, FC128, dropout 0,4, head bảy logits. "
              "Grayscale 48×48, chuẩn hóa [-1,1].", "",
              f"Huấn luyện {training['completed_epochs']} epoch đầy đủ bằng Adam lr={training['learning_rate']}, "
              f"weight decay={training['weight_decay']}, batch={training['batch_size']}, seed={training['seed']}. "
              "Cross-entropy có trọng số căn bậc hai nghịch tần suất lớp từ train. "
              "Train được lật ngang xác suất 0,5 và dịch tối đa 2 pixel; validation/test không augmentation. "
              "Scheduler giảm lr theo macro-F1 validation. Cache tensor trong RAM không đổi pixel.", "",
              f"Chọn **epoch {training['epoch']}** bằng macro-F1 validation cao nhất, hòa thì loss thấp nhất. "
              f"Thiết bị {training['device']}, {training['num_threads']} threads; thời gian gồm nạp dữ liệu và train: "
              f"{(completed-started).total_seconds()/60:.1f} phút. Không dùng test để chọn checkpoint.", "",
              "## Kết quả", "", "| Chỉ số | Validation dùng chọn mô hình | PrivateTest đã lọc |",
              "| --- | ---: | ---: |"]
    for key,name in (("accuracy","Accuracy"),("macro_f1","Macro-F1"),("balanced_accuracy","Balanced accuracy")):
        lines.append(f"| {name} | {100*validation[key]:.2f}% | {100*metrics[key]:.2f}% |")
    lines += ["", "Validation chịu ảnh hưởng của việc chọn mô hình; kết quả test được tính sau khi đã khóa checkpoint. "
              "Macro-F1 là trung bình F1 của bảy lớp; balanced accuracy là trung bình recall. "
              "Lớp ít ảnh như disgust cần được đọc riêng, không chỉ xem accuracy tổng.", "",
              "| Lớp | Số ảnh test | Precision | Recall | F1 |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in metrics["per_class"]:
        lines.append(f"| {row['label']} | {row['support']} | {row['precision']*100:.2f}% | {row['recall']*100:.2f}% | {row['f1']*100:.2f}% |")
    majority=max(splits["splits"]["test"]["class_counts"].values())/metrics["sample_count"]
    lines += ["",f"Baseline luôn đoán lớp phổ biến nhất đạt accuracy {majority*100:.2f}%. "
              "Mức này chỉ là tham chiếu từ phân bố nhãn, không phải một mô hình đã được chọn bằng test.", "",
              "### Confusion matrix", "", "Hàng là nhãn thật; cột là dự đoán.", "",
              "| Thật / dự đoán | "+" | ".join(metrics["labels"])+" |",
              "| --- |"+" ---: |"*7]
    for name,row in zip(metrics["labels"],metrics["confusion_matrix"]): lines.append("| "+name+" | "+" | ".join(str(value) for value in row)+" |")
    lines += ["", "### Diễn biến huấn luyện", "",
              "| Epoch | Train loss có trọng số | Val loss không trọng số | Val accuracy | Val macro-F1 |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for row in history:
        lines.append(f"| {row['epoch']} | {float(row['train_loss']):.4f} | {float(row['val_loss']):.4f} | "
                     f"{float(row['val_accuracy'])*100:.2f}% | {float(row['val_macro_f1'])*100:.2f}% |")
    lines += ["", "## Suy luận và kiểm tra thời gian", "",
              f"Đã đối chiếu API một crop với kết quả batch trên ảnh đầu tiên của mỗi lớp: bảy kết quả khớp, "
              f"không chọn ví dụ theo việc dự đoán đúng. Trên CPU {runtime['threads']} threads, "
              f"10 warmup + {runtime['measured_calls']} lượt đo: median **{runtime['median_ms']:.2f} ms**, "
              f"p95 **{runtime['p95_ms']:.2f} ms** cho một crop RGB đã nằm trong RAM.", "",
              "Phép đo gồm chuyển grayscale/resize + CNN + softmax và tạo dictionary. **Không gồm** camera, định vị mặt, "
              "liveness, identity, đọc tệp hoặc giao diện; không phải FPS đo bằng webcam thật. "
              "Không tự bật camera trong quá trình làm phần này.", "",
              "## Tích hợp và tệp bàn giao", "",
              "- `results/emotion_fer/best_model.pt`: checkpoint đã chọn; config có nhãn, recipe, provenance và hash mã.",
              "- `results/emotion_fer_test/metrics.json`, `predictions.csv`, `confusion_matrix.csv`, `inference_check.json`.",
              "- `emotion_recognition.webcam`: demo camera ROI cố định, người dùng căn mặt; Q/Esc thoát. "
              "Có thể thêm checkpoint liveness để bỏ qua biểu cảm nếu không live. Chưa tự phát hiện/theo dõi nhiều người.",
              "- `AttendanceVerificationPipeline`: cả ba module dùng cùng crop; emotion chỉ để hiển thị, "
              "lỗi emotion không tự làm thay đổi quyết định liveness/identity.",
              "- Mã tích hợp và xử lý frame đã có kiểm thử; chưa đo camera thật và chưa thử đầy đủ ba mô hình thật "
              "do phần 2 còn thiếu dữ liệu/checkpoint identity.", "",
              "## Giới hạn và điểm dừng", "",
              "Biểu cảm ngoài đời khác ảnh FER2013 đã căn mặt; nhãn có nhiễu, lớp mất cân bằng, "
              "mô hình có thể nhầm. Chưa có đo chất lượng đa dạng ánh sáng/góc mặt/thiết bị webcam. "
              "Kết quả này là baseline phân loại biểu cảm, không xác định trạng thái nội tâm hoặc năng lực làm việc.", "",
              "Đã hoàn thành bước triển khai, huấn luyện và đánh giá baseline phần 4. Dừng để người dùng xem; "
              "chưa chuyển sang phần 5 hoặc GUI/đăng ký/chấm công phần 6.", ""]
    target=PROJECT/"docs/04_fer2013_results.md"
    target.write_text("\n".join(lines),encoding="utf-8")
    print(target)
    return 0

if __name__=="__main__": raise SystemExit(main())
