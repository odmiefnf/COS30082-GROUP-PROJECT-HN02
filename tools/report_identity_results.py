"""Write a reviewable report from the completed identity experiments."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def main():
    project = Path(__file__).resolve().parents[1]
    comparison = json.loads((project/"results/comparison/comparison.json").read_text(encoding="utf-8"))
    app = json.loads((project/"results/identity_app_config.json").read_text(encoding="utf-8"))
    test = json.loads((project/"results/classification/classification_test.json").read_text(encoding="utf-8"))
    rows = []
    for model in comparison["models"]:
        training = model["checkpoint_config"]["training"]
        history_file = Path(model["checkpoint"]).parent/"train_history.csv"
        with history_file.open(encoding="utf-8", newline="") as stream:
            history = list(csv.DictReader(stream))
        if len(history) != 10 or any(int(r["train_samples"]) != 380638 for r in history):
            raise ValueError("Report expects ten uncapped epochs using every training anchor.")
        if model["pair_count"] != 8805 or model["checkpoint_smoke_test"] is not False:
            raise ValueError("Report requires the original complete pair file and uncapped training.")
        metrics = {m["metric"]: m for m in model["metrics"]}
        rows.append({"method": model["method"], "selected_epoch": training["epoch"],
                     "selection": training["checkpoint_selection"], "epochs_completed": len(history),
                     "cosine_auc": metrics["cosine_similarity"]["auc"],
                     "euclidean_auc": metrics["euclidean_distance"]["auc"],
                     "checkpoint_sha256": model["checkpoint_sha256"]})
    table = "\n".join(f"| {r['method']} | {r['selected_epoch']} | {r['selection']} | {r['cosine_auc']:.6f} | {r['euclidean_auc']:.6f} |" for r in rows)
    text = f"""# Bổ sung phần 2: Huấn luyện identity và chọn mô hình cho phần 6

Ngày hoàn tất: **27/10/2026**. Đây là kết quả huấn luyện trên dữ liệu danh tính thật, không phải kết quả của các fixture kiểm tra phần mềm.

## Giao thức

- Nguồn: dataset cuộc thi gốc đã kiểm tra ở [phần 5](05_identity_dataset.md).
- Cả hai phương pháp khởi tạo CNN từ đầu; không nạp trọng số pretrained. Cùng FaceEncoder 4 block, embedding 128 chiều, RGB **64×64**, chuẩn hóa [-1,1], không augmentation, seed 42.
- Dùng 380.638 ảnh train / 4.000 ID, đủ **10 epoch**, batch 256, **1.487 batch/epoch**. Không dùng `--max-batches`; mọi epoch có đủ 380.638 anchor. Triplet dùng ba ảnh mỗi anchor; cùng số epoch không đồng nghĩa cùng lượng tính toán.
- Classification: head Linear(128,4000), CrossEntropyLoss, feature thô để tạo logits; chọn checkpoint theo validation loss trên 8.000 val. Đánh giá 8.000 classification test sau chọn checkpoint.
- Metric learning: TripletMarginLoss(margin=0.2,p=2), embedding L2; positive khác ảnh cùng ID, negative khác ID; lấy triplet ngẫu nhiên theo seed/epoch/anchor. Không dùng hard mining. Chọn checkpoint theo train loss, không dùng các cặp verification để cập nhật trọng số/chọn epoch.
- Adam lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0, không scheduler. Môi trường riêng `.venv-gpu`, PyTorch 2.5.1+cu124, RTX 3060 Laptop 6 GiB. Tham khảo [bản CUDA chính thức của PyTorch](https://pytorch.org/get-started/previous-versions/#v251).
- Cố định seed và bật deterministic/warn_only; PyTorch cảnh báo adaptive_avg_pool2d_backward_cuda không bảo đảm deterministic. Không tuyên bố bitwise reproducibility trên GPU.
- Đầu vào 64×64 dùng kích thước mẫu gốc để giảm chi phí. Cache uint8 NCHW được tạo trực tiếp từ ZIP gốc với EXIF/RGB/PIL bilinear đúng như loader. Kiểm tra SHA256 cache và danh sách ảnh khi nạp. Không sửa dataset/file cặp. Nếu thay ảnh trên đĩa, cache cũ vẫn đại diện ZIP đã ghi; cần tạo cache mới cho thí nghiệm khác.

## Kết quả thật

ROC/AUC dùng **đủ {comparison['pair_count']:,} cặp gốc**, 4.315 cùng người và 4.490 khác người; 9.355 ảnh riêng biệt. Cùng cặp và preprocessing cho hai encoder.

| Phương pháp | Epoch được chọn | Tiêu chí chọn epoch | AUC cosine | AUC âm Euclidean |
| --- | ---: | --- | ---: | ---: |
{table}

Classification test closed-set: **{test['metrics']['accuracy']:.4%} accuracy**, loss {test['metrics']['loss']:.6f}, đủ {test['metrics']['samples']:,} ảnh, không dùng test chọn checkpoint. Phân loại đúng một trong 4.000 ID khác với xác minh một cặp; không diễn giải hai chỉ số như cùng một đại lượng.

Với embedding L2, khoảng cách Euclidean và cosine cho thứ tự tương đương về toán học; AUC bằng nhau không phải hai bằng chứng độc lập. Xem `results/comparison/comparison.csv`, JSON và `roc_curve.svg` trong mỗi thư mục mô hình để có ROC, điểm từng cặp và SHA256.

## Chọn ngưỡng cho ứng dụng

Ứng dụng chọn **{app['method']}**, AUC validation cosine **{app['selected_model_validation_auc']:.6f}**. Ngưỡng cosine **{app['threshold']:.9f}**, chọn điểm TPR cao nhất có FPR cặp validation không quá 1%: TPR **{app['observed_validation_pair_tpr']:.4%}**, FPR **{app['observed_validation_pair_fpr']:.4%}**. Cấu hình lưu ở `results/identity_app_config.json`, gắn SHA256 đúng checkpoint. Đây là ngưỡng bổ sung cho demo; ngưỡng Youden của evaluator vẫn được lưu riêng để so sánh, không bị ghi đè.

**Hạn chế chất lượng đáng kể:** ở ngưỡng chặt này, TPR chỉ {app['observed_validation_pair_tpr']:.2%}, tương ứng bỏ sót khoảng {1-app['observed_validation_pair_tpr']:.2%} cặp cùng người trên validation. AUC cao hơn không có nghĩa ứng dụng đạt accuracy 86%. Đây là baseline nghiên cứu, cần cải thiện đặc trưng/metric learning và hiệu chỉnh bằng webcam trước khi dùng thực tế.

Mô hình, ngưỡng được chọn và đánh giá trên cùng tập verification validation: đây là số đo in-sample. **667 ảnh thuộc các cặp validation trùng pixel train**; dataset còn có ảnh trùng qua các split khác. Dùng nguyên giao thức đề để báo AUC và công bố giới hạn, không gọi là test verification độc lập. FPR của cặp cũng không bảo đảm FAR khi tìm trong gallery nhiều người hoặc trên webcam khác miền.

## Tái lập

Chạy ở `grp_project`:

```powershell
.\\.venv-gpu\\Scripts\\python.exe -m face_verification.image_cache --archive data/downloads/identity_kaggle_2020.zip --output-dir data/identity_cache64 --image-size 64
.\\.venv-gpu\\Scripts\\python.exe -m face_verification.train --method classification --data-root data/face_recognition --output-dir results/classification --epochs 10 --batch-size 256 --image-size 64 --device cuda --num-threads 4 --image-cache data/identity_cache64 --evaluate-test
.\\.venv-gpu\\Scripts\\python.exe -m face_verification.train --method triplet --data-root data/face_recognition --output-dir results/triplet --epochs 10 --batch-size 256 --image-size 64 --device cuda --num-threads 4 --image-cache data/identity_cache64
.\\.venv-gpu\\Scripts\\python.exe -m face_verification.evaluate --checkpoint results/classification/best_model.pt --checkpoint results/triplet/best_model.pt --data-root data/face_recognition --output-dir results/comparison --device cuda --batch-size 256
.\\.venv-gpu\\Scripts\\python.exe -m tools.calibrate_identity_app
.\\.venv-gpu\\Scripts\\python.exe -m tools.check_attendance_models
```

Cache đã được tạo trên máy hiện tại; bỏ lệnh tạo lại nếu `manifest.json` đã có. Để thử nghiệm mới, dùng thư mục output mới để giữ kết quả đã báo cáo. Nhật ký hiện tại: `data/logs/identity_classification_train.log`, `identity_triplet_train.log`, `identity_comparison.log`; môi trường: `results/gpu_environment.txt`.

Đây là baseline cho bài học; random triplet không hard mining và thời lượng 10 epoch còn hạn chế. Chất lượng webcam, FAR/FRR gallery và đăng ký thực tế cần đo riêng; không suy ra từ kiểm tra offline trên ảnh train. Xem [hướng dẫn GUI phần 6](06_user_interface.md).
"""
    (project/"docs/06_identity_training_results.md").write_text(text, encoding="utf-8")
    (project/"results/identity_experiment_summary.json").write_text(json.dumps({"models": rows,
        "classification_test_accuracy": test["metrics"]["accuracy"], "app": app}, indent=2), encoding="utf-8")
    print("Saved docs/06_identity_training_results.md and results/identity_experiment_summary.json")


if __name__ == "__main__":
    main()
