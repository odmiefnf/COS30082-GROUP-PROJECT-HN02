"""Classify FER facial-expression categories on an already cropped face."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from PIL import Image
import torch
from anti_spoofing.evaluate import _device
from .data import CLASSES,VIETNAMESE,preprocess_emotion_image
from .evaluate import load_model

class EmotionRecognizer:
    def __init__(self,checkpoint,device="cpu"):
        self.device=_device(device); self.checkpoint=Path(checkpoint).resolve()
        self.model,self.payload=load_model(checkpoint,self.device)
    def predict_crop(self,image):
        tensor=preprocess_emotion_image(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits=self.model(tensor)
            if not torch.isfinite(logits).all(): raise ValueError("Nonfinite expression logits.")
            scores=logits.double().softmax(1)[0].cpu().tolist()
        index=max(range(7),key=lambda label:scores[label])
        return {"label":CLASSES[index],"label_vi":VIETNAMESE[index],"class_index":index,
                "confidence":scores[index],"scores":dict(zip(CLASSES,scores)),
                "meaning":"FER facial-expression category; not a measurement of internal emotional state",
                "used_for_attendance_acceptance":False}
    def predict(self,image):
        if isinstance(image,(str,Path)):
            with Image.open(image) as source: return self.predict_crop(source)
        return self.predict_crop(image)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True); parser.add_argument("--image",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda","auto"),default="auto"); parser.add_argument("--num-threads",type=int,default=4)
    args=parser.parse_args(argv)
    if args.num_threads<1: parser.error("Threads must be positive.")
    if _device(args.device).type=="cpu": torch.set_num_threads(args.num_threads)
    try: result=EmotionRecognizer(args.checkpoint,args.device).predict(args.image)
    except (OSError,ValueError,RuntimeError,KeyError) as exc: parser.exit(2,f"Expression inference failed: {exc}\n")
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return 0

if __name__=="__main__": raise SystemExit(main())
