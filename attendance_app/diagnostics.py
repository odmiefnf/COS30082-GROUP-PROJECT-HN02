"""Actionable English guidance, separate from recognition decisions."""

COMPONENTS = {
    "Runtime": ("Python / Torch / OpenCV", "Use the project Python environment and requirements-webcam.txt."),
    "Identity": ("Identity encoder", "Check the identity configuration, checkpoint path and matching SHA256. Do not mix encoders with an existing gallery."),
    "Liveness": ("Liveness model", "Check the checkpoint path and its validation-calibrated threshold; use a completed training checkpoint."),
    "Expression": ("Expression model", "Check the expression checkpoint path and completed training metadata."),
    "Detector": ("Face locator", "Check the Haar cascade files in OpenCV, or the configured manual ROI."),
    "Profiles": ("Profile database", "Check the database path, write access and encoder compatibility. Keep a backup before changing databases."),
}


def frame_guidance(result):
    status = result["status"]
    if status == "no_face":
        return "Look toward the camera, keep your whole face near the center and light your face from the front."
    if status == "multiple_faces":
        return "Keep only one person in view; check for faces on posters or screens behind you."
    if status == "face_too_small_or_outside":
        return "Move closer and keep your forehead and chin inside the frame."
    if status == "image_quality":
        reasons = result.get("quality_reasons", [])
        hints = []
        if "dark" in reasons:
            hints.append("Add light in front of your face; avoid a bright window behind you.")
        if "bright" in reasons:
            hints.append("Reduce direct light or screen glare on your face and glasses.")
        if "blur" in reasons:
            hints.append("Hold still, clean the lens and allow the camera to focus.")
        return " ".join(hints) or "Hold still and adjust the lighting until your face is clear."
    if status == "spoof":
        return "Face detected; liveness did not pass. Use the live camera, adjust front lighting and tilt slightly to reduce glasses glare."
    if status == "liveness_unavailable":
        return "Open Diagnostics to check the liveness model and calibrated threshold."
    if status == "unknown":
        return "Already registered with glasses? Add a separate appearance to the same profile; do not create a second person."
    if status == "ambiguous":
        return "Keep one person in view and try a clearer, front-facing angle. A close match alone is not a confirmed identity."
    if status == "attendance_confirmed":
        return "Attendance is already stored for this session. You may step away."
    if status in ("known", "registered", "appearance_added"):
        return "Keep looking at the camera; automatic attendance needs consistent recognition across three valid frames."
    return "Keep the same person in view, move only slightly and follow the capture progress above."


def frame_details(result):
    """Diagnostic measurements only; never change a gate or persist a face."""
    parts = ["Frame: "+result["status"].replace("_", " ")]
    if "sharpness" in result:
        parts.append(f"Sharpness: {result['sharpness']:.1f} (minimum 20)")
    if "brightness" in result:
        parts.append(f"Brightness: {result['brightness']:.1f} (20–235)")
    live = result.get("liveness") or {}
    if live.get("live_score") is not None and live.get("threshold") is not None:
        parts.append(f"PAD score: {live['live_score']:.6g}; threshold: {live['threshold']:.6g} (not a confidence guarantee)")
    if result.get("expression_error"):
        parts.append("Expression error: "+result["expression_error"])
    return "\n".join(parts)
