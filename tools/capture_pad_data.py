"""Collect labelled webcam PAD crops through the app's own face pipeline, then build splits.

capture: one operator-declared session (live person or one attack type). Save RGB
crops after the app's face-count and quality checks. Use --context-crop for the 2x crop.
split:   hold out whole sessions for validation. Context splits use only context
webcam sessions and are written to splits/context-2x/.

Saved face images are personal data: they stay under data/ (git-ignored) and must only be
captured with the participant's consent.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import time

PRESENTATIONS = ("live-person", "phone-still-photo", "recorded-video", "printed-photo", "laptop-screen")
COLUMNS = ("image_path", "label", "subject_id", "source_label", "subject_id_source")
PROJECT = Path(__file__).resolve().parents[1]


class SaveGate:
    """FrameProcessor service stand-in that keeps every Nth accepted crop."""

    def __init__(self, directory: Path, every: int):
        self.directory, self.every = directory, every
        self.accepted, self.saved = 0, []

    def reset_tracking(self, reason=None):
        pass

    def analyze(self, crop, *, context_crop=None):
        self.accepted += 1
        if (self.accepted - 1) % self.every == 0:
            path = self.directory / f"crop_{len(self.saved):04d}.png"
            (crop if context_crop is None else context_crop).save(path)
            self.saved.append(path)
        return {"status": "saved"}


def capture(args) -> Path | None:
    import cv2
    from attendance_app.camera import FaceLocator, FrameProcessor

    label = 1 if args.presentation == "live-person" else 0
    context_crop = getattr(args, "context_crop", False)
    session = f"cam_{args.presentation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    directory = args.root / ("live" if label else "spoof") / session
    gate = SaveGate(directory, args.every)
    processor = FrameProcessor(FaceLocator(), gate, pad_context=context_crop)
    camera = cv2.VideoCapture(args.camera)
    window = "PAD data capture"
    counts: dict[str, int] = defaultdict(int)
    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not camera.isOpened():
            raise RuntimeError("Cannot open the webcam.")
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 960, 720)
        for phase, seconds in (("setup", args.setup_seconds), ("capture", args.seconds)):
            if phase == "capture":
                directory.mkdir(parents=True, exist_ok=False)
            started = time.monotonic()
            while time.monotonic() - started < seconds:
                ok, bgr = camera.read()
                if not ok:
                    raise RuntimeError("Camera stopped returning frames.")
                display = bgr.copy()
                left = seconds - (time.monotonic() - started)
                if phase == "capture":
                    result = processor.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    counts[result["status"]] += 1
                    for x, y, w, h in result.get("boxes", []):
                        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    text = f"RECORDING {args.presentation} | saved {len(gate.saved)} | {left:.0f} s | Q stop"
                else:
                    text = f"Get ready: {args.presentation} | starts in {left:.0f} s | Q cancel"
                cv2.putText(display, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 255), 2)
                cv2.imshow(window, display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    if phase == "setup":
                        return None
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    meta = {"session": session, "presentation": args.presentation, "label": label, "note": args.note,
            "frame_statuses": dict(counts), "saved_crops": len(gate.saved), "every": args.every,
            "crop_policy": "context-2x" if context_crop else "tight",
            "captured_at": datetime.now().astimezone().isoformat()}
    (directory / "session.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    manifest = args.root / "manifest.csv"
    new = not manifest.exists()
    with manifest.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(COLUMNS)
        for path in gate.saved:
            writer.writerow((str(path.resolve()), label, session, 1 - label, "provided"))
    print(json.dumps(meta, indent=2), flush=True)
    return directory


def split_sessions(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Hold out the latest session of every presentation type that has at least two sessions."""
    sessions: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        presentation = row["subject_id"].split("_")[1]
        if row["subject_id"] not in sessions[presentation]:
            sessions[presentation].append(row["subject_id"])
    held_out = {sorted(names)[-1] for names in sessions.values() if len(names) >= 2}
    train = [row for row in rows if row["subject_id"] not in held_out]
    val = [row for row in rows if row["subject_id"] in held_out]
    if {row["label"] for row in val} != {"0", "1"}:
        raise ValueError("Validation needs live and spoof sessions: capture at least two sessions "
                         "of live-person and two of one attack type.")
    return train, val


def split(args) -> None:
    context_crop = getattr(args, "context_crop", False)
    crop_policy = "context-2x" if context_crop else "tight"
    root = args.root.resolve()
    with (root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        webcam = []
        policies = {}
        for row in csv.DictReader(handle):
            image_path = root / row["image_path"]
            metadata = image_path.parent / "session.json"
            if metadata not in policies:
                policy = "tight"
                if metadata.is_file():
                    policy = json.loads(metadata.read_text(encoding="utf-8")).get("crop_policy", policy)
                if policy not in ("tight", "context-2x"):
                    raise ValueError(f"Unrecognized crop policy in {metadata}: {policy!r}")
                policies[metadata] = policy
            if policies[metadata] == crop_policy:
                row["image_path"] = str(image_path)
                webcam.append(row)
    train, val = split_sessions(webcam)
    output = root / "splits"
    base = []
    if context_crop:
        output /= crop_policy
    else:
        with args.base_train.open(encoding="utf-8-sig", newline="") as handle:
            base = [{column: row.get(column, "") for column in COLUMNS} for row in csv.DictReader(handle)]
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.csv", base + train), ("val.csv", val), ("webcam_train.csv", train)):
        with (output / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    summary = {name: {"rows": len(rows), "live": sum(r["label"] == "1" for r in rows),
                      "sessions": len({r["subject_id"] for r in rows})}
               for name, rows in (("webcam_train", train), ("webcam_val", val))}
    print(json.dumps(summary, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=PROJECT / "data/pad_webcam")
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--presentation", choices=PRESENTATIONS, required=True)
    cap.add_argument("--note", default="", help="e.g. glasses, lamp off, max brightness")
    cap.add_argument("--seconds", type=float, default=20.)
    cap.add_argument("--setup-seconds", type=float, default=5.)
    cap.add_argument("--every", type=int, default=2, help="Keep every Nth accepted crop.")
    cap.add_argument("--camera", type=int, default=0)
    cap.add_argument("--context-crop", action="store_true", help="Save a 2x face crop with surrounding context.")
    spl = sub.add_parser("split")
    spl.add_argument("--base-train", type=Path, default=PROJECT / "data/lcc_fasd_manifests/train.csv",
                     help="Base LCC training manifest for tight-crop splits.")
    spl.add_argument("--context-crop", action="store_true", help="Split 2x context webcam sessions only.")
    args = parser.parse_args(argv)
    if args.command == "capture":
        if not 0 < args.seconds <= 120 or args.every < 1:
            parser.error("Use 0 < --seconds <= 120 and --every >= 1.")
        if capture(args) is None:
            print("Capture cancelled during setup.", flush=True)
    else:
        split(args)


if __name__ == "__main__":
    main()
