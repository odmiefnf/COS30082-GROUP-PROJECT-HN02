"""Train a fresh FER CNN; select by validation macro-F1, never PrivateTest."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from anti_spoofing.evaluate import _device,_sha256,_write_json
from .data import CLASSES,EmotionDataset,validate_splits
from .metrics import classification_metrics
from .model import EmotionCNN

def augment(images):
    """Independent horizontal flips and integer shifts up to 2px, train only."""
    count=images.shape[0]; device=images.device
    flip=torch.rand(count,1,1,1,device=device)<.5
    images=torch.where(flip,images.flip(-1),images)
    padded=F.pad(images,(2,2,2,2),mode="replicate")
    offsets=torch.randint(0,5,(count,2),device=device)
    rows=torch.arange(48,device=device)[None,:,None]+offsets[:,0,None,None]
    columns=torch.arange(48,device=device)[None,None,:]+offsets[:,1,None,None]
    return padded[torch.arange(count,device=device)[:,None,None],0,rows,columns][:,None]

def epoch(model,loader,device,criterion,optimizer=None):
    model.train(optimizer is not None)
    total_loss=0.; labels=[]; predicted=[]
    with torch.set_grad_enabled(optimizer is not None):
        for images,targets in loader:
            images=images.to(device); targets=targets.to(device)
            if optimizer is not None:
                images=augment(images); optimizer.zero_grad(set_to_none=True)
            logits=model(images)
            loss=criterion(logits,targets)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError("Nonfinite emotion logits/loss.")
            if optimizer is not None:
                loss.backward(); optimizer.step()
            total_loss+=float(loss.detach())*len(targets)
            labels.extend(targets.cpu().tolist()); predicted.extend(logits.argmax(1).detach().cpu().tolist())
    return {"loss":total_loss/len(labels),**classification_metrics(labels,predicted)}

def train(args):
    if args.epochs<1 or args.batch_size<1 or args.num_threads<1 or not 0<=args.seed<2**32:
        raise ValueError("Invalid epoch, batch, threads or seed.")
    if not math.isfinite(args.learning_rate) or args.learning_rate<=0: raise ValueError("Invalid learning rate.")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.benchmark=False
    device=_device(args.device)
    if device.type=="cpu": torch.set_num_threads(args.num_threads)
    started=datetime.now(timezone.utc).isoformat()
    training=EmotionDataset(args.train_manifest); validation=EmotionDataset(args.val_manifest)
    split_check=validate_splits({"train":training.records,"val":validation.records})
    counts=torch.bincount(training.labels,minlength=7).float()
    weights=(counts.sum()/(7*counts)).sqrt(); weights=weights/weights.mean()
    model=EmotionCNN().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=.5,patience=3)
    train_loss=nn.CrossEntropyLoss(weight=weights.to(device))
    val_loss=nn.CrossEntropyLoss()
    generator=torch.Generator().manual_seed(args.seed)
    train_loader=DataLoader(training,batch_size=args.batch_size,shuffle=True,generator=generator)
    val_loader=DataLoader(validation,batch_size=args.batch_size,shuffle=False)
    config={"architecture":"EmotionCNN-v1","classes":list(CLASSES),"channels":1,"image_size":48,
            "normalization":"[-1,1]","pretrained":False}
    provenance={"train":training.records,"val":validation.records}
    recipe={"started_at_utc":started,"device":str(device),"num_threads":args.num_threads,
            "seed":args.seed,"epochs_requested":args.epochs,"batch_size":args.batch_size,
            "learning_rate":args.learning_rate,"weight_decay":1e-4,"optimizer":"Adam",
            "scheduler":"ReduceLROnPlateau, val macro-F1, factor=.5, patience=3",
            "class_weighting":"normalized sqrt(N/(7*train_class_count))", "class_weights":weights.tolist(),
            "loss":"weighted cross-entropy in training; unweighted cross-entropy in validation",
            "augmentation":{"horizontal_flip_probability":.5,"translation_pixels":2,"validation_augmented":False},
            "initialization":"random_from_scratch","smoke_test":False,"test_data_read":False,
            "selection":"maximum validation macro-F1, minimum validation loss as tie-breaker",
            "provenance":provenance,"split_check":split_check,
            "source_sha256":{name:_sha256(Path(__file__).parent/name) for name in ("data.py","model.py","train.py","metrics.py")},
            "versions":{"torch":str(torch.__version__),"numpy":str(np.__version__)}}
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    best=(-1.,float("-inf")); best_epoch=0
    def save(path,metadata):
        torch.save({"task":"facial_expression","format_version":1,"config":config,
                    "model_state":{key:value.detach().cpu().clone() for key,value in model.state_dict().items()},
                    "training":{**recipe,**metadata}},path)
    columns=("epoch","train_loss","val_loss","train_accuracy","val_accuracy","val_macro_f1","val_balanced_accuracy","learning_rate","best_epoch")
    print(f"Training FER CNN from scratch: {len(training)} train, {len(validation)} validation; {device}.",flush=True)
    with (output/"history.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=columns); writer.writeheader()
        for index in range(1,args.epochs+1):
            rate=optimizer.param_groups[0]["lr"]
            t=epoch(model,train_loader,device,train_loss,optimizer)
            v=epoch(model,val_loader,device,val_loss)
            quality=(v["macro_f1"],-v["loss"])
            improved=quality>best
            if improved: best=quality; best_epoch=index
            metadata={"epoch":index,"best_epoch":best_epoch,"validation_metrics":v,
                      "train_samples_per_epoch":len(training),"validation_samples_per_epoch":len(validation)}
            if improved: save(output/"best_model.pt",metadata)
            save(output/"last_model.pt",metadata)
            writer.writerow({"epoch":index,"train_loss":t["loss"],"val_loss":v["loss"],
                             "train_accuracy":t["accuracy"],"val_accuracy":v["accuracy"],
                             "val_macro_f1":v["macro_f1"],"val_balanced_accuracy":v["balanced_accuracy"],
                             "learning_rate":rate,"best_epoch":best_epoch}); handle.flush()
            scheduler.step(v["macro_f1"])
            print(f"Epoch {index}/{args.epochs}: train_loss={t['loss']:.4f}, val_loss={v['loss']:.4f}, "
                  f"val_accuracy={v['accuracy']:.4f}, val_macro_f1={v['macro_f1']:.4f}; best={best_epoch}",flush=True)
    payload=torch.load(output/"best_model.pt",map_location="cpu",weights_only=True)
    payload["training"].update(completed_epochs=args.epochs,completed_at_utc=datetime.now(timezone.utc).isoformat())
    torch.save(payload,output/"best_model.pt")
    _write_json(output/"config.json",{"model":config,"training":payload["training"],"best_checkpoint":str((output/"best_model.pt").resolve())})
    print(f"Saved best epoch {best_epoch}: {output/'best_model.pt'}",flush=True)
    return output/"best_model.pt"

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("train-manifest","val-manifest","output-dir"): parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=15); parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--learning-rate",type=float,default=.001); parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--num-threads",type=int,default=4); parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    try: train(parser.parse_args(argv))
    except (OSError,ValueError,RuntimeError) as exc: parser.exit(2,f"Emotion training failed: {exc}\n")
    return 0

if __name__=="__main__": raise SystemExit(main())
