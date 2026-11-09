"""Check trained crop inference against batch output and time the CPU crop API."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import statistics
import sys
import time
from PIL import Image
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from emotion_recognition.data import CLASSES
from emotion_recognition.infer import EmotionRecognizer
from anti_spoofing.evaluate import _sha256

def main():
    torch.set_num_threads(4)
    checkpoint=PROJECT/"results/emotion_fer/best_model.pt"
    model_hash=_sha256(checkpoint)
    recognizer=EmotionRecognizer(checkpoint,"cpu")
    directory=PROJECT/"results/emotion_fer_test"
    with (directory/"predictions.csv").open(encoding="utf-8",newline="") as handle:
        rows=list(csv.DictReader(handle))
    examples=[]
    for label in CLASSES:
        row=next(row for row in rows if row["true_label"]==label)
        result=recognizer.predict(row["image_path"])
        difference=max(abs(result["scores"][name]-float(row[name])) for name in CLASSES)
        if result["label"]!=row["predicted_label"] or difference>1e-4:
            raise ValueError("Single-crop and batch predictions disagree.")
        examples.append({"ground_truth":label,"image_path":row["image_path"],
                         "prediction":result,"maximum_score_difference_from_batch":difference})
    with Image.open(examples[0]["image_path"]) as source:
        image=source.convert("RGB")
    for _ in range(10): recognizer.predict_crop(image)
    durations=[]
    for _ in range(100):
        started=time.perf_counter(); recognizer.predict_crop(image)
        durations.append(time.perf_counter()-started)
    if _sha256(checkpoint)!=model_hash:
        raise ValueError("Inference changed the checkpoint.")
    ordered=sorted(durations)
    report={"checkpoint_sha256":model_hash,"device":"cpu","threads":4,"warmup_calls":10,"measured_calls":100,
            "median_ms":1000*statistics.median(durations),"p95_ms":1000*ordered[94],
            "mean_ms":1000*statistics.mean(durations),"crop_api_fps_equivalent":1/statistics.mean(durations),
            "scope":"One cached RGB face crop: Pillow grayscale/resize + CNN + softmax/JSON dictionary; excludes camera, localization, liveness, identity, GUI and disk I/O",
            "physical_webcam_tested":False,"batch_consistency_verified":True,
            "example_selection":"First retained PrivateTest image per class, not chosen for prediction correctness",
            "examples":examples}
    (directory/"inference_check.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({key:value for key,value in report.items() if key!="examples"},indent=2))
    return 0

if __name__=="__main__": raise SystemExit(main())
