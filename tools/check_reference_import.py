"""Bounded integration check on local training fixtures, never the user's gallery.

This checks software wiring with the real models. It is not a PAD/identity
benchmark: the photos come from training data and the second group repeats
the first group's inputs deliberately to check the add-appearance transaction.
"""
from collections import defaultdict
import csv
import json
from pathlib import Path
import tempfile

import torch

from anti_spoofing.infer import LivenessDetector
from attendance_app.camera import FaceLocator
from attendance_app.recognition import FaceRecognizer
from attendance_app.reference_import import ReferenceImporter
from attendance_app.store import ProfileStore


def main():
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(4)
    recognizer = FaceRecognizer(root/"results/identity_app_config.json")
    live = LivenessDetector(root/"results/liveness_camera_matched/best_model.pt")
    locator = FaceLocator()
    groups = defaultdict(list)
    with (root/"data/lcc_fasd_manifests/train.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["label"] == "1":
                path = Path(row["image_path"])
                groups[path.stem.rsplit("_", 1)[0]].append(str(path))
    report = {"scope": "Software integration using LCC training fixtures; not an accuracy benchmark",
              "production_database_used": False, "attempts": [], "success": False}
    temporary_root = root/"tmp"
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
        store = ProfileStore(Path(directory)/"fixture.sqlite3", recognizer.fingerprint, recognizer.dimension)
        try:
            importer = ReferenceImporter(store, recognizer, live, locator)
            for prefix, files in list((p, f) for p, f in groups.items() if len(f) >= 3)[:12]:
                selected = files[:3]
                operation = "fixture-new-person-"+prefix
                checked = importer.prepare(operation_id=operation, files=selected, name="Dataset fixture")
                report["attempts"].append({"capture_group": prefix, "valid": checked["valid"],
                                            "items": checked["items"], "message": checked["message"]})
                if not checked["valid"]:
                    continue
                assert not store.profiles(), "Check must not write a profile"
                result = importer.commit(operation_id=operation, token=checked["token"], confirmed=True)
                assert store.profiles()[0]["sample_count"] == 3
                second_operation = "fixture-add-appearance-"+prefix
                second = importer.prepare(operation_id=second_operation, files=selected,
                    person_id=result["person_id"], label="Repeated fixture check")
                assert second["valid"], second["message"]
                importer.commit(operation_id=second_operation, token=second["token"], confirmed=True)
                person = store.profiles()[0]
                assert person["sample_count"] == 6 and len(person["appearances"]) == 2
                assert store.connection.execute("SELECT COUNT(*) FROM attendance").fetchone()[0] == 0
                report.update(success=True, final_profile_count=1, final_sample_count=6,
                              final_group_count=2, attendance_count=0,
                              second_group_inputs="Same three training photos; transaction check only")
                break
        finally:
            store.close()
    destination = root/"results/app_upgrade_20261030/reference_import_integration.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"success": report["success"], "attempts": len(report["attempts"]),
                      "report": str(destination)}))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
