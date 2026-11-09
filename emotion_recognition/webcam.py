"""Optional webcam expression demo with a manually aligned, fixed face ROI."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from anti_spoofing.data import crop_liveness_image
from .infer import EmotionRecognizer

def analyze_frame(rgb_frame,roi,recognizer,liveness=None):
    """Display expression and liveness independently on the same RGB crop."""
    if not isinstance(rgb_frame,np.ndarray) or rgb_frame.dtype!=np.uint8 or rgb_frame.ndim!=3 or rgb_frame.shape[2]!=3:
        raise ValueError("Frame must be RGB uint8 HxWx3.")
    crop=crop_liveness_image(rgb_frame,bbox=roi)
    live=None if liveness is None else liveness.predict_crop(crop)
    expression=recognizer.predict_crop(crop)
    return {"roi":list(roi),"crop_size":list(crop.size),"liveness":live,"expression":expression,
            "attendance_decision":None}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--liveness-checkpoint",type=Path)
    parser.add_argument("--camera",type=int,default=0)
    parser.add_argument("--roi",type=int,nargs=4,metavar=("X","Y","W","H"))
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    parser.add_argument("--num-threads",type=int,default=4)
    args=parser.parse_args(argv)
    if args.num_threads<1 or args.camera<0: parser.error("Positive threads and nonnegative camera index required.")
    try:
        import cv2
    except ImportError:
        parser.exit(2,"Install optional webcam dependencies: pip install -r requirements-webcam.txt\n")
    torch.set_num_threads(args.num_threads)
    recognizer=EmotionRecognizer(args.checkpoint,args.device)
    liveness=None
    if args.liveness_checkpoint is not None:
        from anti_spoofing.infer import LivenessDetector
        liveness=LivenessDetector(args.liveness_checkpoint,device=args.device)
    camera=cv2.VideoCapture(args.camera)
    try:
        if not camera.isOpened(): raise RuntimeError("Cannot open camera. Check camera index and Windows camera permission.")
        while True:
            started=time.perf_counter()
            success,bgr=camera.read()
            if not success: raise RuntimeError("Camera stopped returning frames.")
            height,width=bgr.shape[:2]
            side=int(min(height,width)*.65)
            roi=args.roi or ((width-side)//2,(height-side)//2,side,side)
            result=analyze_frame(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),roi,recognizer,liveness)
            expression=result["expression"]
            live=result["liveness"]
            if expression is not None:
                text=f"{expression['label']} {expression['confidence']:.2f}"; color=(0,220,0)
            else:
                text="Expression unavailable"; color=(0,0,230)
            if live is not None and live.get("is_live") is not True:
                color=(0,190,250)
            x,y,w,h=roi
            cv2.rectangle(bgr,(x,y),(x+w,y+h),color,2)
            cv2.putText(bgr,text,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2)
            live_text=("Liveness: not checked" if live is None else
                       "Liveness: passed" if live.get("is_live") is True else
                       "Liveness: unavailable" if live.get("is_live") is None else "Liveness: not passed")
            cv2.putText(bgr,live_text,(12,54),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.putText(bgr,"Align face in box | FER expression | Q / Esc: quit",(12,height-15),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            fps=1/max(time.perf_counter()-started,1e-9)
            cv2.putText(bgr,f"Processing: {fps:.1f} FPS",(12,78),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.imshow("FER2013 expression demo",bgr)
            if cv2.waitKey(1)&0xff in (ord("q"),27): break
    except (OSError,ValueError,RuntimeError) as exc:
        parser.exit(2,f"Webcam demo failed: {exc}\n")
    finally:
        camera.release(); cv2.destroyAllWindows()
    return 0

if __name__=="__main__": raise SystemExit(main())
