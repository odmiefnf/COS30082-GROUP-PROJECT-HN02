"""Evaluate FER PrivateTest once without changing the chosen checkpoint."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from anti_spoofing.evaluate import _device,_sha256,_write_json
from .data import CLASSES,EmotionDataset
from .metrics import classification_metrics
from .model import EmotionCNN

def load_model(checkpoint,device="cpu"):
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if not isinstance(payload,dict) or payload.get("task")!="facial_expression":
        raise ValueError("Expected an emotion checkpoint from this project.")
    config=payload.get("config",{})
    if config.get("classes")!=list(CLASSES) or config.get("image_size")!=48 or config.get("channels")!=1 or config.get("architecture")!="EmotionCNN-v1":
        raise ValueError("Checkpoint architecture/preprocessing/label order is incompatible.")
    model=EmotionCNN(); model.load_state_dict(payload["model_state"],strict=True)
    model.to(_device(device)).eval()
    return model,payload

def predict(model,dataset,device="cpu",batch_size=128):
    labels=[]; scores=[]
    model.eval()
    with torch.inference_mode():
        for images,targets in DataLoader(dataset,batch_size=batch_size,shuffle=False):
            logits=model(images.to(device))
            if not torch.isfinite(logits).all():
                raise ValueError("Emotion model returned nonfinite logits.")
            labels.extend(targets.tolist()); scores.extend(logits.double().softmax(1).cpu().tolist())
    return labels,scores

def evaluate(args):
    device=_device(args.device)
    if device.type=="cpu": torch.set_num_threads(args.num_threads)
    model,payload=load_model(args.checkpoint,device)
    dataset=EmotionDataset(args.manifest)
    references=payload.get("training",{}).get("provenance",{})
    if not all(isinstance(references.get(split),list) and references[split] for split in ("train","val")):
        raise ValueError("Checkpoint lacks train/validation provenance.")
    seen_pixels={row["pixel_sha256"] for records in references.values() for row in records}
    seen_paths={row["image_path"] for records in references.values() for row in records}
    overlap=[row["image_path"] for row in dataset.records if row["pixel_sha256"] in seen_pixels or row["image_path"] in seen_paths]
    if overlap or {row["source_split"] for row in dataset.records}!={"PrivateTest"}:
        raise ValueError("Evaluation must be PrivateTest, with paths/pixels absent from train/validation.")
    labels,scores=predict(model,dataset,device,args.batch_size)
    predicted=[max(range(7),key=lambda index:row[index]) for row in scores]
    metrics=classification_metrics(labels,predicted)
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    with (output/"predictions.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.writer(handle); writer.writerow(["image_path","true_label","predicted_label","confidence",*CLASSES])
        for record,label,pred,score in zip(dataset.records,labels,predicted,scores):
            writer.writerow([record["image_path"],CLASSES[label],CLASSES[pred],max(score),*score])
    with (output/"confusion_matrix.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.writer(handle); writer.writerow(["true/predicted",*CLASSES])
        for label,row in zip(CLASSES,metrics["confusion_matrix"]): writer.writerow([label,*row])
    report={"metrics":metrics,"checkpoint":str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256":_sha256(args.checkpoint),"manifest_sha256":_sha256(args.manifest),
            "training_smoke_test":payload["training"]["smoke_test"],
            "selected_epoch":payload["training"]["epoch"],"evaluation_samples":len(dataset),
            "evaluation_scope":"Filtered PrivateTest; disjoint paths and decoded pixels, subject separation unverified",
            "test_used_for_model_selection":False,"checkpoint_changed":False,
            "subject_disjoint_verified":False,"device":str(device)}
    _write_json(output/"metrics.json",report)
    print(json.dumps(report,indent=2),flush=True)
    return report

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","manifest","output-dir"): parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    parser.add_argument("--batch-size",type=int,default=128); parser.add_argument("--num-threads",type=int,default=4)
    args=parser.parse_args(argv)
    if args.batch_size<1 or args.num_threads<1: parser.error("Batch size and threads must be positive.")
    try: evaluate(args)
    except (OSError,ValueError,RuntimeError,KeyError) as exc: parser.exit(2,f"Emotion evaluation failed: {exc}\n")
    return 0

if __name__=="__main__": raise SystemExit(main())
