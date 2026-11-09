"""Fixed FER2013 labels, grayscale preprocessing and verified image manifests."""
from __future__ import annotations
import csv
import hashlib
import io
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

CLASSES = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
VIETNAMESE = ("Giận dữ", "Ghê tởm", "Sợ hãi", "Vui", "Buồn", "Ngạc nhiên", "Trung tính")
SOURCE_SPLITS = {"train":"Training", "val":"PublicTest", "test":"PrivateTest"}
COLUMNS = ("image_path", "label", "source_split", "file_sha256", "pixel_sha256")

def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def source_pixels(path: str | Path) -> tuple[np.ndarray, str, str]:
    encoded=Path(path).read_bytes()
    with Image.open(io.BytesIO(encoded)) as source:
        if source.size != (48,48):
            raise ValueError(f"FER2013 source must be 48x48: {path}")
        pixels=np.asarray(source.convert("L"),dtype=np.uint8).copy()
    return pixels,sha256(encoded),sha256(pixels.tobytes())

def preprocess_emotion_image(image: Image.Image | np.ndarray) -> torch.Tensor:
    if isinstance(image,np.ndarray):
        if image.dtype!=np.uint8 or not (image.ndim==2 or image.ndim==3 and image.shape[2]==3):
            raise ValueError("Image array must be grayscale/RGB uint8.")
        image=Image.fromarray(image)
    if not isinstance(image,Image.Image):
        raise TypeError("Expected a Pillow image or grayscale/RGB uint8 array.")
    image=image.convert("L").resize((48,48),Image.Resampling.BILINEAR)
    pixels=np.asarray(image,dtype=np.float32)/127.5-1.0
    return torch.from_numpy(pixels[None].copy())

def load_manifest(path: str | Path) -> list[dict]:
    path=Path(path).expanduser().resolve()
    rows=[]; seen=set(); pixels=set()
    with path.open(encoding="utf-8-sig",newline="") as handle:
        reader=csv.DictReader(handle)
        if reader.fieldnames!=list(COLUMNS):
            raise ValueError(f"Unexpected emotion manifest schema: {path}")
        for row in reader:
            if None in row or any(row.get(key) is None for key in COLUMNS):
                raise ValueError("Malformed emotion manifest row.")
            if row["label"] not in tuple(str(index) for index in range(7)):
                raise ValueError("Emotion label must be 0..6 in fixed FER order.")
            label=int(row["label"])
            image=Path(row["image_path"]).expanduser().resolve()
            if not image.is_file() or image.parent.name!=CLASSES[label]:
                raise ValueError(f"Missing image or class-directory mismatch: {image}")
            if row["source_split"] not in SOURCE_SPLITS.values() or not image.name.startswith(row["source_split"]+"_"):
                raise ValueError(f"Source split conflicts with filename: {image}")
            for key in ("file_sha256","pixel_sha256"):
                if len(row[key])!=64 or any(character not in "0123456789abcdef" for character in row[key]):
                    raise ValueError(f"Invalid {key} for {image}")
            key=os.path.normcase(str(image))
            if key in seen or row["pixel_sha256"] in pixels:
                raise ValueError("Manifest contains duplicate paths or decoded images.")
            seen.add(key); pixels.add(row["pixel_sha256"])
            rows.append({**row,"image_path":str(image),"label":label})
    if not rows:
        raise ValueError("Empty emotion manifest.")
    return rows

def validate_splits(splits: dict[str,list[dict]]) -> dict:
    paths=set(); pixels=set(); summary={}
    for name,rows in splits.items():
        if name not in SOURCE_SPLITS or {row["source_split"] for row in rows}!={SOURCE_SPLITS[name]}:
            raise ValueError(f"Wrong official source group for {name}")
        if {row["label"] for row in rows}!=set(range(7)):
            raise ValueError(f"{name} must contain all seven expression classes.")
        for row in rows:
            key=os.path.normcase(row["image_path"])
            if key in paths or row["pixel_sha256"] in pixels:
                raise ValueError("Emotion splits overlap by path or decoded pixels.")
            paths.add(key); pixels.add(row["pixel_sha256"])
        summary[name]={"samples":len(rows),"class_counts":{label:sum(row["label"]==index for row in rows) for index,label in enumerate(CLASSES)}}
    return {"splits":summary,"disjoint_paths":True,"disjoint_decoded_pixels":True,
            "subject_disjoint_verified":False,"identity_note":"FER image folders have no verified subject IDs."}

class EmotionDataset(Dataset):
    """Cache verified float32 1x48x48 inputs once; training augments batches later."""
    def __init__(self,manifest: str | Path):
        self.manifest_path=Path(manifest).expanduser().resolve()
        self.records=load_manifest(self.manifest_path)
        images=[]
        for index,row in enumerate(self.records,1):
            pixels,file_hash,pixel_hash=source_pixels(row["image_path"])
            if file_hash!=row["file_sha256"] or pixel_hash!=row["pixel_sha256"]:
                raise ValueError(f"Image changed after preparation: {row['image_path']}")
            images.append(preprocess_emotion_image(pixels))
            if index%10000==0:
                print(f"Loaded {index}/{len(self.records)} images from {self.manifest_path.name}",flush=True)
        self.images=torch.stack(images)
        self.labels=torch.tensor([row["label"] for row in self.records],dtype=torch.long)
    def __len__(self):
        return len(self.records)
    def __getitem__(self,index):
        return self.images[index],self.labels[index]
