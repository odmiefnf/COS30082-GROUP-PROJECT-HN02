"""Recover FER official usage prefixes and clean decoded-image duplicates."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from .data import CLASSES,COLUMNS,SOURCE_SPLITS,source_pixels,validate_splits

def prepare(data_root: str | Path,output_dir: str | Path) -> dict:
    root=Path(data_root).expanduser().resolve()
    raw={split:[] for split in SOURCE_SPLITS}
    for folder in ("train","test"):
        directory=root/folder
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing FER folder: {directory}")
        if {entry.name for entry in directory.iterdir() if entry.is_dir()}!=set(CLASSES):
            raise ValueError(f"Expected exactly seven FER class folders under {directory}")
        for label,expression in enumerate(CLASSES):
            for image in sorted((directory/expression).iterdir()):
                if not image.is_file() or image.suffix.lower() not in (".jpg",".jpeg",".png"):
                    raise ValueError(f"Unexpected FER entry: {image}")
                path=image.resolve()
                if not path.is_relative_to(root):
                    raise ValueError("FER image escapes dataset directory.")
                prefix=image.name.split("_",1)[0]
                split=next((key for key,value in SOURCE_SPLITS.items() if value==prefix),None)
                if split is None or (split=="train")!=(folder=="train"):
                    raise ValueError(f"Cannot recover the official usage split: {image}")
                _,file_hash,pixel_hash=source_pixels(path)
                raw[split].append({"image_path":str(path),"label":label,"source_split":prefix,
                                   "file_sha256":file_hash,"pixel_sha256":pixel_hash})
    excluded=[]; cleaned={}
    for split,rows in raw.items():
        groups={}
        for row in rows:
            groups.setdefault(row["pixel_sha256"],[]).append(row)
        retained=[]
        for group in groups.values():
            if len({row["label"] for row in group})>1:
                excluded.extend({**row,"split":split,"reason":"conflicting labels for identical decoded pixels within split"} for row in group)
            else:
                retained.append(group[0])
                excluded.extend({**row,"split":split,"reason":"duplicate decoded image within split"} for row in group[1:])
        cleaned[split]=retained
    # PublicTest determines validation. PrivateTest never changes train/val inclusion.
    val_hashes={row["pixel_sha256"] for row in cleaned["val"]}
    train=[]
    for row in cleaned["train"]:
        if row["pixel_sha256"] in val_hashes:
            excluded.append({**row,"split":"train","reason":"training pixels already in validation"})
        else:
            train.append(row)
    cleaned["train"]=train
    development=val_hashes|{row["pixel_sha256"] for row in train}
    test=[]
    for row in cleaned["test"]:
        if row["pixel_sha256"] in development:
            excluded.append({**row,"split":"test","reason":"test pixels already in train/validation"})
        else:
            test.append(row)
    cleaned["test"]=test
    summary=validate_splits(cleaned)
    output=Path(output_dir).expanduser().resolve(); output.mkdir(parents=True,exist_ok=True)
    for split,rows in cleaned.items():
        with (output/f"{split}.csv").open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=COLUMNS); writer.writeheader()
            writer.writerows(sorted(rows,key=lambda row:row["image_path"]))
    reasons=sorted({row["reason"] for row in excluded})
    summary.update(dataset="FER2013",dataset_root=str(root),labels=list(CLASSES),
                   original_splits={split:len(rows) for split,rows in raw.items()},
                   images_decoded=sum(len(rows) for rows in raw.values()),
                   excluded_counts={split:{reason:sum(row["split"]==split and row["reason"]==reason for row in excluded) for reason in reasons} for split in raw},
                   protocol_modified=True,test_used_to_select_training_images=False,
                   protocol="Training/PublicTest/PrivateTest prefixes; within-split duplicates/conflicts removed; train filtered against val; test filtered against development")
    (output/"split_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    (output/"excluded_images.json").write_text(json.dumps(excluded,indent=2)+"\n",encoding="utf-8")
    return summary

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        summary=prepare(args.data_root,args.output_dir)
    except (OSError,ValueError) as exc:
        parser.exit(2,f"FER preparation failed: {exc}\n")
    print(json.dumps(summary,indent=2),flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
