"""Build a Vietnamese review report from the completed, unchanged LCC run."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def read_json(relative: str) -> dict:
    return json.loads((PROJECT / relative).read_text(encoding="utf-8-sig"))


def main() -> int:
    download = read_json("data/lcc_fasd_download_status.json")
    images = read_json("data/lcc_fasd_image_check.json")
    splits = read_json("data/lcc_fasd_manifests/split_summary.json")
    config = read_json("results/liveness_lcc/config.json")
    calibration = read_json("results/liveness_lcc/calibration.json")
    evaluation = read_json("results/liveness_lcc_test/metrics.json")
    training = config["training"]
    if training.get("smoke_test") or evaluation["training_smoke_test"]:
        raise ValueError("A smoke-test checkpoint cannot produce this completed-training report.")
    if evaluation["threshold_retuned_on_evaluation"]:
        raise ValueError("Evaluation threshold was retuned; the documented protocol was not followed.")
    with (PROJECT / "results/liveness_lcc/history.csv").open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    actual = splits["actual_splits"]
    validation = calibration["validation_metrics"]
    test = evaluation["metrics"]
    started = datetime.fromisoformat(training["started_at_utc"])
    completed = datetime.fromisoformat(training["completed_at_utc"])
    now = datetime.now(timezone(timedelta(hours=7))).strftime("%d/%m/%Y %H:%M UTC+7")
    lines = [
        "# Kết quả phần 3 — LCC-FASD", "", f"Báo cáo tạo lúc {now}.", "",
        "Đã tải dữ liệu thật, kiểm tra ảnh, huấn luyện CNN từ đầu và đánh giá bằng ngưỡng validation. "
        "Đây là baseline nghiên cứu; chưa phải hệ thống điểm danh hoàn chỉnh.", "",
        "## Dữ liệu và kiểm tra chất lượng", "",
        f"- Nguồn: [Kaggle LCC-FASD v1](https://www.kaggle.com/datasets/faber24/lcc-fasd).",
        f"- ZIP: {download['downloaded_bytes']:,} byte; SHA256 `{download['archive_sha256']}`.",
        f"- Đã giải nén và kiểm tra CRC {download['files_verified']:,} ảnh; "
        f"đọc/resize thành công {images['decoded_successfully']:,}/{images['images']:,} ảnh, không có ảnh lỗi.",
        "- Ảnh gốc vẫn còn trong `data/LCC_FASD`; các loại trừ chỉ áp dụng vào manifest.", "",
        "| Tập sử dụng | Ảnh thật | Ảnh giả | Tổng | Mã ID biết được |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, name in (("train", "Train"), ("val", "Validation"), ("test", "Evaluation đã lọc")):
        row = actual[key]
        identity = str(row["subjects"]) if key != "test" else "Không có ID"
        lines.append(f"| {name} | {row['live']:,} | {row['spoof']:,} | {row['samples']:,} | {identity} |")
    lines += [
        "",
        f"Loại {splits['excluded_train_overlap']['samples']} ảnh train theo các mã ID trùng development; "
        f"loại thêm {splits['excluded_train_content_overlap']['samples']} ảnh train trùng byte với validation. "
        f"Evaluation loại {splits['excluded_test_content_overlap']['samples']} ảnh trùng byte với train/validation. "
        "Validation được giữ nguyên. Test không quyết định ảnh nào được giữ để huấn luyện. "
        "Danh sách và lý do nằm trong `data/lcc_fasd_manifests/split_summary.json` và `source_inventory.json`.", "",
        "**Đây là giao thức đã điều chỉnh và tập evaluation con, không phải kết quả benchmark nguyên bản.** "
        "Evaluation ẩn danh: không xác minh được tách người hoặc video; chỉ kiểm tra ảnh không trùng đường dẫn/hash với train/validation.", "",
        "## Cấu hình huấn luyện", "",
        "CNN bốn khối convolution 16/32/64/128 kênh, đặc trưng 128 chiều và head hai lớp; "
        "trọng số khởi tạo ngẫu nhiên, không pretrained. RGB 112×112, bilinear resize, chuẩn hóa [-1,1], "
        "dùng toàn bộ ảnh crop (`crop_mode=full`).", "",
        f"Cross-entropy không trọng số, Adam lr={training['learning_rate']}, batch={training['batch_size']}, "
        f"seed={training['seed']}, {training['completed_epochs']} epoch đầy đủ, không giới hạn batch, không augmentation. "
        f"Thiết bị {training['device']}, {training['num_threads']} CPU threads. "
        f"Cache tensor chỉ giảm đọc/giải mã, không thay đổi pixel. Thời gian huấn luyện và hiệu chỉnh, không gồm bước đọc/hash/nạp cache: "
        f"{(completed-started).total_seconds()/60:.1f} phút.", "",
        f"Chọn **epoch {training['best_epoch']}** theo validation loss thấp nhất; sau đó hiệu chỉnh ngưỡng trên "
        f"toàn bộ {calibration['validation_samples_used']:,} validation. Ngưỡng live: **{evaluation['threshold']:.9f}**. "
        "Evaluation dùng nguyên ngưỡng đó, không chọn lại mô hình hoặc ngưỡng trên test.", "",
        "| Epoch | Train loss | Validation loss | Train accuracy (argmax) | Validation accuracy (argmax) |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in history:
        lines.append(f"| {row['epoch']} | {float(row['train_loss']):.5f} | {float(row['val_loss']):.5f} | "
                     f"{float(row['train_accuracy'])*100:.2f}% | {float(row['val_accuracy'])*100:.2f}% |")
    if float(history[-1]["val_loss"]) > training["best_val_loss"]:
        lines += ["", "Validation loss tăng sau epoch tốt nhất trong khi train loss tiếp tục giảm, "
                  "cho thấy overfitting. Báo cáo dùng checkpoint được chọn bằng validation, không dùng "
                  "epoch cuối. Các cải tiến về augmentation, cân bằng lớp hoặc regularization cần được "
                  "so sánh bằng validation trong một thí nghiệm sau."]
    lines += [
        "", "## Kết quả tại ngưỡng validation", "",
        "| Chỉ số | Validation, dùng để hiệu chỉnh | Evaluation đã lọc |",
        "| --- | ---: | ---: |",
    ]
    for key in ("AUC", "accuracy", "APCER", "BPCER", "ACER"):
        left, right = validation[key], test[key]
        a, b = (f"{left:.6f}", f"{right:.6f}") if key == "AUC" else (f"{left*100:.2f}%", f"{right*100:.2f}%")
        lines.append(f"| {key} | {a} | {b} |")
    confusion = test["confusion"]
    lines += [
        "", "APCER là tỷ lệ giả được chấp nhận nhầm; BPCER là tỷ lệ người thật bị từ chối; "
        "ACER là trung bình hai lỗi. Validation tại ngưỡng vừa chọn là in-sample.", "",
        "| Nhãn thật | Dự đoán giả | Dự đoán thật |",
        "| --- | ---: | ---: |",
        f"| Giả | {confusion['true_spoof']:,} | {confusion['spoof_accepted_live']:,} |",
        f"| Thật | {confusion['live_rejected_spoof']:,} | {confusion['true_live']:,} |", "",
        f"Nếu luôn đoán giả, accuracy vẫn đạt {100*test['spoof_count']/test['sample_count']:.2f}%, "
        "nhưng BPCER=100%, ACER=50%, AUC=0.5. Vì vậy cần đọc AUC và cả hai loại lỗi, không kết luận từ accuracy riêng.", "",
        "[Đường ROC](../results/liveness_lcc_test/roc_curve.svg) · "
        "[Metrics và provenance](../results/liveness_lcc_test/metrics.json) · "
        "[Predictions](../results/liveness_lcc_test/predictions.csv)", "",
        "## Tệp dùng tiếp và phạm vi còn lại", "",
        "- Checkpoint đã chọn và hiệu chỉnh: `results/liveness_lcc/best_model.pt`.",
        "- Lịch sử: `results/liveness_lcc/history.csv`; nhật ký: `training.log`; cấu hình và hash: `config.json`.",
        "- Module suy luận: `python -m anti_spoofing.infer --checkpoint results/liveness_lcc/best_model.pt "
        "--image DUONG_DAN_ANH_CROP --face-crop`.",
        "- `AttendanceVerificationPipeline` kiểm tra liveness trước, chỉ gọi identity khi live đạt ngưỡng. "
        "Mã tích hợp đã có kiểm thử phần mềm; chưa đánh giá tích hợp bằng mô hình identity thật vì thiếu dữ liệu/checkpoint phần 2.",
        "- LCC-FASD tập trung giả mạo qua màn hình; chưa có kết quả trên ảnh in riêng hoặc webcam thực tế.",
        "- Chưa có GUI, đăng ký người lạ hoặc dữ liệu điểm danh. Chưa bắt đầu phần 4 về cảm xúc.", "",
        "Độ khái quát giữa người, video và thiết bị vẫn chưa được chứng minh; không dùng các chỉ số này "
        "để khẳng định hệ thống an toàn khi triển khai thực tế.", "",
    ]
    examples_path = PROJECT / "results/liveness_lcc_test/inference_examples.json"
    if examples_path.is_file():
        examples = json.loads(examples_path.read_text(encoding="utf-8"))["examples"]
        lines += ["## Kiểm tra CLI trên ảnh thật", "",
                  "Lấy ảnh đầu tiên của mỗi lớp trong manifest evaluation, không chọn ảnh theo kết quả. "
                  "Hai ví dụ này kiểm tra CLI, không dùng để ước lượng chất lượng mô hình.", "",
                  "| Nhãn thật | Tệp | Điểm live | Quyết định |",
                  "| --- | --- | ---: | --- |"]
        for row in examples:
            result = row["prediction"]
            lines.append(f"| {row['ground_truth']} | {Path(row['image_path']).name} | "
                         f"{result['live_score']:.6f} | {result['status']} |")
        lines += ["", "[Kết quả CLI](../results/liveness_lcc_test/inference_examples.json)", ""]
    target = PROJECT / "docs/03_lcc_fasd_results.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
