"""Meaningful FER protocol, model and integration tests using synthetic images."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock
import numpy as np
from PIL import Image
import torch
from attendance_pipeline import AttendanceVerificationPipeline
from emotion_recognition.data import CLASSES,EmotionDataset,load_manifest,preprocess_emotion_image,validate_splits
from emotion_recognition.prepare import prepare
from emotion_recognition.metrics import classification_metrics
from emotion_recognition.model import EmotionCNN
from emotion_recognition.train import augment,main as train_main
from emotion_recognition.evaluate import main as evaluate_main,load_model
from emotion_recognition.infer import EmotionRecognizer
from emotion_recognition.webcam import analyze_frame

def fixture(root):
    for split_index,(folder,prefix) in enumerate((("train","Training"),("test","PublicTest"),("test","PrivateTest"))):
        for label,name in enumerate(CLASSES):
            directory=root/folder/name; directory.mkdir(parents=True,exist_ok=True)
            for frame in range(3):
                pixels=np.random.default_rng(split_index*1000+label*10+frame).integers(0,256,(48,48),dtype=np.uint8)
                Image.fromarray(pixels).save(directory/f"{prefix}_{label}_{frame}.png")

class FERChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary=tempfile.TemporaryDirectory(prefix="fer_synthetic_")
        self.base=Path(self.temporary.name); self.root=self.base/"FER2013"; self.manifests=self.base/"manifests"
        fixture(self.root)
    def tearDown(self): self.temporary.cleanup()
    def test_preprocessing_fixed_label_order_and_invalid_input(self):
        self.assertEqual(CLASSES[4:],("sad","surprise","neutral"))
        self.assertEqual(tuple(preprocess_emotion_image(np.zeros((48,48),dtype=np.uint8)).shape),(1,48,48))
        self.assertTrue(torch.all(preprocess_emotion_image(Image.new("RGB",(80,60),"white"))==1))
        self.assertTrue(torch.all(preprocess_emotion_image(np.zeros((48,48),dtype=np.uint8))==-1))
        with self.assertRaises(ValueError): preprocess_emotion_image(np.zeros((48,48),dtype=np.float32))
    def test_official_prefix_splits_and_label_folders(self):
        summary=prepare(self.root,self.manifests)
        self.assertEqual(summary["original_splits"],{"train":21,"val":21,"test":21})
        rows={name:load_manifest(self.manifests/f"{name}.csv") for name in ("train","val","test")}
        self.assertTrue(validate_splits(rows)["disjoint_decoded_pixels"])
        self.assertFalse(summary["subject_disjoint_verified"])
        for split,records in rows.items():
            self.assertEqual({row["label"] for row in records},set(range(7)))
            self.assertTrue(all(Path(row["image_path"]).parent.name==CLASSES[row["label"]] for row in records))
    def test_duplicate_conflict_cleanup_and_test_never_selects_training_images(self):
        source=self.root/"train/angry/Training_0_0.png"
        (self.root/"train/angry/Training_repeat.png").write_bytes(source.read_bytes())
        (self.root/"train/happy/Training_conflict.png").write_bytes(source.read_bytes())
        summary=prepare(self.root,self.manifests)
        self.assertEqual(summary["excluded_counts"]["train"]["conflicting labels for identical decoded pixels within split"],3)
        before={row["image_path"] for row in load_manifest(self.manifests/"train.csv")}
        retained=self.root/"train/angry/Training_0_1.png"
        (self.root/"test/angry/PrivateTest_copy.png").write_bytes(retained.read_bytes())
        summary=prepare(self.root,self.manifests)
        self.assertEqual(before,{row["image_path"] for row in load_manifest(self.manifests/"train.csv")})
        self.assertEqual(summary["excluded_counts"]["test"]["test pixels already in train/validation"],1)
        self.assertFalse(summary["test_used_to_select_training_images"])
    def test_training_content_is_filtered_against_validation(self):
        original=self.root/"train/sad/Training_4_0.png"
        (self.root/"test/sad/PublicTest_copy.png").write_bytes(original.read_bytes())
        summary=prepare(self.root,self.manifests)
        self.assertEqual(summary["excluded_counts"]["train"]["training pixels already in validation"],1)
        self.assertNotIn(str(original.resolve()),{row["image_path"] for row in load_manifest(self.manifests/"train.csv")})
    def test_missing_split_prefix_and_changed_image_are_rejected(self):
        invalid=self.root/"test/angry/unknown_0.png"
        Image.new("L",(48,48),12).save(invalid)
        with self.assertRaises(ValueError): prepare(self.root,self.manifests)
        invalid.unlink(); prepare(self.root,self.manifests)
        Image.new("L",(48,48),12).save(self.root/"train/angry/Training_0_0.png")
        with self.assertRaisesRegex(ValueError,"changed"): EmotionDataset(self.manifests/"train.csv")
    def test_macro_metrics_account_for_unpredicted_classes(self):
        perfect=classification_metrics(list(range(7)),list(range(7)))
        self.assertEqual(perfect["macro_f1"],1.)
        majority=classification_metrics(list(range(7)),[3]*7)
        self.assertAlmostEqual(majority["accuracy"],1/7)
        self.assertAlmostEqual(majority["macro_f1"],1/28)
        self.assertAlmostEqual(majority["balanced_accuracy"],1/7)
        with self.assertRaises(ValueError): classification_metrics([0],[7])
    def test_augmentation_preserves_range_shape_and_is_reproducible(self):
        images=torch.linspace(-1,1,48*48).reshape(1,1,48,48).repeat(4,1,1,1)
        torch.manual_seed(42); first=augment(images)
        torch.manual_seed(42); second=augment(images)
        self.assertEqual(first.shape,images.shape); self.assertTrue(torch.equal(first,second))
        self.assertTrue(torch.isfinite(first).all()); self.assertLessEqual(float(first.max()),1.)
        self.assertGreaterEqual(float(first.min()),-1.); self.assertFalse(torch.equal(first,images))
    def test_model_logits_and_gradient_are_valid(self):
        model=EmotionCNN(); logits=model(torch.randn(3,1,48,48))
        self.assertEqual(tuple(logits.shape),(3,7))
        torch.nn.functional.cross_entropy(logits,torch.tensor([0,3,6])).backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
    def test_full_train_evaluate_and_infer_keep_fixed_labels_and_holdout(self):
        prepare(self.root,self.manifests); output=self.base/"model"; evaluation=self.base/"evaluation"
        self.assertEqual(train_main(["--train-manifest",str(self.manifests/"train.csv"),"--val-manifest",str(self.manifests/"val.csv"),"--output-dir",str(output),"--epochs","1","--batch-size","7","--device","cpu","--num-threads","2"]),0)
        checkpoint=output/"best_model.pt"; before=checkpoint.read_bytes()
        common=["--checkpoint",str(checkpoint),"--output-dir",str(evaluation),"--device","cpu","--num-threads","2"]
        self.assertEqual(evaluate_main([*common,"--manifest",str(self.manifests/"test.csv")]),0)
        self.assertEqual(checkpoint.read_bytes(),before)
        report=json.loads((evaluation/"metrics.json").read_text())
        self.assertEqual(report["metrics"]["sample_count"],21); self.assertFalse(report["test_used_for_model_selection"])
        with self.assertRaises(SystemExit): evaluate_main([*common,"--manifest",str(self.manifests/"val.csv")])
        recognizer=EmotionRecognizer(checkpoint)
        sample=Image.new("RGB",(48,48),(50,80,120)); a=recognizer.predict_crop(sample); b=recognizer.predict_crop(np.asarray(sample))
        self.assertEqual(a["scores"],b["scores"]); self.assertAlmostEqual(sum(a["scores"].values()),1.)
        self.assertFalse(a["used_for_attendance_acceptance"])
        environment={**os.environ,"PYTHONIOENCODING":"cp1252"}
        process=subprocess.run([sys.executable,"-m","emotion_recognition.infer","--checkpoint",str(checkpoint),
                                "--image",str(self.root/"test/angry/PrivateTest_0_0.png"),"--device","cpu"],
                               cwd=Path(__file__).resolve().parents[1],env=environment,capture_output=True,check=True)
        cli=json.loads(process.stdout.decode("utf-8"))
        self.assertIn(cli["label"],CLASSES); self.assertIsInstance(cli["label_vi"],str)
        payload=torch.load(checkpoint,weights_only=True); payload["config"]["classes"]=sorted(CLASSES); torch.save(payload,self.base/"wrong.pt")
        with self.assertRaises(ValueError): load_model(self.base/"wrong.pt")

class EmotionGateChecks(unittest.TestCase):
    def test_frame_roi_is_shared_and_expression_is_independent_of_liveness(self):
        frame=np.zeros((80,100,3),dtype=np.uint8); frame[10:50,20:60]=(25,75,125)
        recognizer=Mock(); recognizer.predict_crop.return_value={"label":"happy"}
        liveness=Mock(); liveness.predict_crop.return_value={"is_live":True}
        result=analyze_frame(frame,(20,10,40,40),recognizer,liveness)
        crop=recognizer.predict_crop.call_args.args[0]
        self.assertIs(liveness.predict_crop.call_args.args[0],crop)
        self.assertTrue(np.all(np.asarray(crop)==(25,75,125)))
        self.assertIsNone(result["attendance_decision"])
        for is_live in (False,None):
            recognizer.reset_mock(); liveness.predict_crop.return_value={"is_live":is_live}
            result=analyze_frame(frame,(20,10,40,40),recognizer,liveness)
            self.assertEqual(result["expression"],{"label":"happy"})
            self.assertIs(result["liveness"]["is_live"],is_live)
            recognizer.predict_crop.assert_called_once()
            self.assertIs(liveness.predict_crop.call_args.args[0],recognizer.predict_crop.call_args.args[0])
    def test_spoof_skips_identity_but_keeps_expression_display(self):
        for is_live in (False,None):
            detector=Mock(); crop=Image.new("RGB",(48,48)); detector.prepare_face.return_value=crop
            detector.predict_crop.return_value={"is_live":is_live}
            identity=Mock(); expression=Mock(); expression.predict_crop.return_value={"label":"happy"}
            result=AttendanceVerificationPipeline(detector,identity,expression).verify("probe","reference")
            self.assertFalse(result["accepted"]); self.assertEqual(result["emotion"],{"label":"happy"})
            expression.predict_crop.assert_called_once_with(crop); identity.verify_images.assert_not_called()
            expression.predict_crop.side_effect=RuntimeError("FER unavailable")
            result=AttendanceVerificationPipeline(detector,identity,expression).verify("probe","reference")
            self.assertFalse(result["accepted"]); self.assertIsNone(result["emotion"])
            self.assertIn("emotion_error",result); identity.verify_images.assert_not_called()
    def test_expression_is_descriptive_and_same_crop_is_used(self):
        crop=Image.new("RGB",(48,48)); detector=Mock(); detector.prepare_face.return_value=crop; detector.predict_crop.return_value={"is_live":True}
        identity=Mock(); identity.verify_images.return_value={"same_person":True}
        expression=Mock(); expression.predict_crop.return_value={"label":"angry"}
        pipeline=AttendanceVerificationPipeline(detector,identity,expression)
        self.assertTrue(pipeline.verify("probe",crop)["accepted"])
        self.assertIs(expression.predict_crop.call_args.args[0],crop)
        expression.predict_crop.side_effect=RuntimeError("Expression unavailable")
        result=pipeline.verify("probe",crop)
        self.assertTrue(result["accepted"]); self.assertIn("emotion_error",result)

if __name__=="__main__": unittest.main()
