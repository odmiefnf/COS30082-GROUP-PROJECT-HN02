"""Build the English group report from recorded project evidence using the document runtime."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.mathtext import math_to_image
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "tmp/group_report"
ASSETS = WORK / "assets"
OUTPUT = ROOT / "reports/COS30082_Group_Report.docx"
BLUE, ORANGE = "#234f78", "#b2592b"


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def camera_trial_text():
    """Pool every completed repetition, with denominators and ground truth roles."""
    conditions = {"glasses": "glasses", "no-glasses": "no glasses", "unknown": "unregistered person",
                  "phone-still-photo": "phone still photo", "recorded-video": "recorded video replay",
                  "print": "printed photo", "screen-unspecified": "screen presentation of unspecified type"}
    grouped = {key: [] for key in conditions}
    for path in sorted((ROOT/"results/app_upgrade_20261030/evaluation/trials").glob("*.json")):
        trial = json.loads(path.read_text(encoding="utf-8"))
        assert trial["schema_version"] == 1, f"Unsupported trial schema: {path}"
        condition = trial["operator_ground_truth"]["condition"]
        if condition == "screen-replay":
            presentation = trial["operator_ground_truth"].get("presentation_type")
            assert presentation in (None, "phone-still-photo", "recorded-video"), f"Unsupported screen presentation: {path}"
            condition = presentation or "screen-unspecified"
        assert condition in grouped, f"Unsupported trial condition: {path}"
        if trial["status"] != "completed":
            continue
        assert trial["production_database_read_only"] is True
        assert trial["profiles_or_attendance_written_to_production"] is False
        assert trial["images_or_video_saved"] is False and trial["thresholds_retuned"] is False
        grouped[condition].append(trial)
    if not any(grouped.values()):
        return "Webcam evaluation is pending operator-labelled trials; glasses, no glasses, unregistered-person, phone-still-photo, recorded-video and printed-photo trials have not been completed."
    summaries = []
    gallery_sizes, appearances = set(), set()
    for condition, trials in grouped.items():
        if not trials:
            continue
        total = lambda key: sum(trial["counts"][key] for trial in trials)
        duration = sum(trial["duration_seconds"] for trial in trials)
        text = (f"{conditions[condition]} ({len(trials)} trial{'s' if len(trials) != 1 else ''}, {duration:.1f}s): "
                f"{total('all_frames')} frames, {total('exactly_one_face')} with one face, "
                f"{sum(trial['frame_statuses'].get('no_face', 0) for trial in trials)} no-face, "
                f"{sum(trial['frame_statuses'].get('multiple_faces', 0) for trial in trials)} multiple-face, "
                f"{total('quality_pass')} quality-pass, ")
        pad = (f"{total('pad_pass')}/{total('pad_scored')}" if total('pad_scored') else "not evaluated (no scored crops)")
        if condition in ("glasses", "no-glasses"):
            scratch = sum(trial["simulated_attendance"]["correct_record_count"] > 0 for trial in trials)
            scored = (f"{total('identity_correct')}/{total('identity_scored')} scored" if total('identity_scored')
                      else "not evaluated (no scored probes)")
            all_frames = (f"{total('identity_correct')}/{total('all_frames')} all" if total('all_frames') else "no measured frames")
            text += (f"PAD {pad}, correct identity {scored} ({all_frames}), "
                     f"wrong {total('identity_wrong')}, scratch target record in {scratch}/{len(trials)} trials")
        else:
            key = "unknown_matched" if condition == "unknown" else "attack_any_matched"
            record_key = "unknown_accept_record_count" if condition == "unknown" else "attack_accept_record_count"
            scratch = sum(trial["simulated_attendance"][record_key] > 0 for trial in trials)
            matched = f"{total(key)}/{total('identity_scored')} scored" if total('identity_scored') else "not evaluated (no scored probes)"
            attack_pad = (f"PAD accepted {pad} attack crops" if total('pad_scored')
                          else "PAD not evaluated (no scored attack crops)")
            text += ((f"PAD {pad}" if condition == "unknown" else attack_pad)
                     + f", gallery match {matched}, "
                     f"scratch acceptance in {scratch}/{len(trials)} trials")
            if condition != "unknown" and total('pad_pass') > 0 and not scratch:
                if (total('pad_pass') == total('pad_scored') and total('identity_scored') > 0
                        and total('identity_unknown') == total('identity_scored')):
                    text += "; all scored attack crops passed PAD, a PAD failure in this declared trial set; every identity probe was unknown, so no scratch record was created"
                else:
                    text += "; PAD accepted attack crops despite zero scratch attendance in this trial set"
        summaries.append(text)
        for trial in trials:
            gallery_sizes.add(trial["gallery_profile_count"])
            appearances.update(trial["gallery_appearance_counts"])
    missing = [label for condition, label in conditions.items() if condition != "screen-unspecified" and not grouped[condition]]
    gallery = ", ".join(str(value) for value in sorted(gallery_sizes))
    looks = ", ".join(str(value) for value in sorted(appearances))
    return ("Historical operator-labelled webcam captures: " + "; ".join(summaries) + ". "
            + ("Not completed: " + ", ".join(missing) + ". " if missing else "")
            + f"Gallery sizes: {gallery} profile(s); appearance counts per profile: {looks}. "
            + ("A one-person gallery cannot establish impostor safety. " if gallery_sizes == {1} else "")
            + "Lighting and distance were not controlled. No images were saved, so no-face or multiple-face causes cannot be visually confirmed. "
            + "Scratch records never entered production. These trials precede the current capture/UI and independent-expression updates. Correlated frames do not establish independent accuracy, FAR or FRR.")


def software_evidence():
    """Only describe a suite after its saved count agrees with a successful log."""
    suite = read_json("results/readiness_20261108/status.json")["software_tests"]
    assert suite["passed"] > 0
    log = (ROOT/suite["log"]).read_text(encoding="utf-8-sig")
    match = re.search(r"Ran (\d+) tests in ([\d.]+)s", log)
    assert match and log.rstrip().endswith("OK")
    assert int(match[1]) == suite["passed"] and np.isclose(float(match[2]), suite["duration_seconds"])
    return {"passed": suite["passed"], "total": suite["passed"],
            "seconds": suite["duration_seconds"], "log": suite["log"]}


def evidence():
    identity = read_json("results/comparison/comparison.json")
    app = read_json("results/identity_app_config.json")
    app_checkpoint = Path(app["checkpoint"])
    if not app_checkpoint.is_absolute():
        app_checkpoint = ROOT/"results"/app_checkpoint
    assert app["checkpoint_sha256"] == digest(app_checkpoint)
    live = read_json("results/liveness_lcc_test/metrics.json")
    expression = read_json("results/emotion_fer_test/metrics.json")
    current_live = read_json("results/liveness_camera_matched/test_regression_metrics.json")
    current_calibration = read_json("results/liveness_camera_matched/calibration.json")
    current_app = read_json("results/liveness_app_config.json")
    assert identity["pair_count"] == 8805
    assert identity["pairs_file_sha256"] == digest(ROOT / "data/face_recognition/verification_pairs_val.txt")
    for model in [*identity["models"], live, expression]:
        assert model["checkpoint_sha256"] == digest(model["checkpoint"])
    assert live["metrics"]["sample_count"] == 7522
    assert expression["metrics"]["sample_count"] == 3271
    cm = np.array(expression["metrics"]["confusion_matrix"])
    assert int(cm.sum()) == 3271
    assert np.isclose(np.trace(cm) / cm.sum(), expression["metrics"]["accuracy"])
    assert np.isclose(1657 / 7266, live["metrics"]["APCER"])
    assert np.isclose(75 / 256, live["metrics"]["BPCER"])
    assert current_app["checkpoint_sha256"] == digest(ROOT/"results"/current_app["checkpoint"])
    assert current_live["metrics"]["sample_count"] == 7522
    assert np.isclose(current_live["metrics"]["APCER"], 1388 / 7266)
    assert np.isclose(current_live["metrics"]["BPCER"], 29 / 256)
    assert current_live["metrics"]["threshold"] == current_calibration["threshold"]
    assert current_live["threshold_retuned_on_test"] is False
    assert current_calibration["test_data_used"] is False and current_calibration["camera_data_used"] is False
    phone = read_json("results/app_upgrade_20261030/evaluation/trials/phone_photo_01.json")
    assert phone["status"] == "completed" and phone["operator_ground_truth"]["presentation_type"] == "phone-still-photo"
    assert phone["counts"]["pad_pass"] == phone["counts"]["pad_scored"] == phone["counts"]["identity_unknown"] == 237
    assert phone["counts"]["attack_any_matched"] == phone["simulated_attendance"]["attack_accept_record_count"] == 0
    return identity, app, live, expression, current_live


def figures(identity, app, expression):
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "black", "text.color": "black",
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.15), gridspec_kw={"width_ratios": [1.15, 1]})
    operating = {}
    for model, color, linestyle in zip(identity["models"], (BLUE, ORANGE), ("-", "--")):
        curve = [r for r in rows(model["artifacts"]["roc_points_csv"]) if r["metric"] == "cosine_similarity"]
        x = np.array([float(r["fpr"]) for r in curve])
        y = np.array([float(r["tpr"]) for r in curve])
        t = np.array([float(r["raw_threshold"]) for r in curve])
        auc = next(m["auc"] for m in model["metrics"] if m["metric"] == "cosine_similarity")
        assert np.isclose(np.trapezoid(y, x), auc)
        eligible = [i for i in range(len(x)) if np.isfinite(t[i]) and x[i] <= .01 and -1 <= t[i] <= 1]
        best = max(eligible, key=lambda i: (y[i], t[i]))
        operating[model["method"]] = {"fpr": float(x[best]), "tpr": float(y[best]), "threshold": float(t[best])}
        label = f"{model['method'].capitalize()}  AUC {auc:.4f}"
        for axis in axes:
            axis.plot(x, y, linestyle=linestyle, color=color, lw=1.8, label=label)
            axis.grid(alpha=.22)
            axis.set_xlabel("False positive rate")
            axis.set_ylabel("True positive rate")
    axes[0].plot([0, 1], [0, 1], color="#888888", linestyle=":", lw=1)
    axes[0].set(xlim=(0, 1), ylim=(0, 1), title="All verification pairs")
    axes[1].set(xlim=(0, .05), ylim=(0, .5), title="Low false positive region")
    axes[1].axvline(.01, color="#777777", lw=1, linestyle=":")
    axes[1].scatter([app["observed_validation_pair_fpr"]], [app["observed_validation_pair_tpr"]],
                    marker="*", s=75, color=ORANGE, zorder=4)
    axes[1].annotate("App point\nTPR 9.73%", (.0098, .0973), xytext=(.021, .07), fontsize=9,
                     arrowprops={"arrowstyle": "->", "color": "#555555", "lw": .8})
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, -.01), fontsize=9)
    fig.tight_layout(rect=(0, .095, 1, 1), w_pad=1.2)
    fig.savefig(ASSETS / "identity_roc.png", dpi=250)
    fig.savefig(ASSETS / "identity_roc.svg")
    plt.close(fig)

    per_class = expression["metrics"]["per_class"]
    fig, ax = plt.subplots(figsize=(6.85, 2.25))
    scores = [r["f1"] for r in per_class]
    bars = ax.bar(np.arange(7), scores, color=[ORANGE if s < .3 else BLUE for s in scores], width=.62)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 score")
    ax.set_xticks(np.arange(7), [f"{r['label']}\nn={r['support']}" for r in per_class], fontsize=9)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x()+bar.get_width()/2, score+.025, f"{score:.3f}", ha="center", fontsize=9)
    fig.tight_layout(pad=.6)
    fig.savefig(ASSETS / "expression_f1.png", dpi=250)
    fig.savefig(ASSETS / "expression_f1.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.85, 2.15))
    ax.set(xlim=(0, 10), ylim=(0, 4))
    ax.axis("off")
    boxes = [(0.1, 1.65, 1.65, .9, "RGB face crop\nQuality passed"),
             (2.2, 1.65, 1.7, .9, "Liveness CNN\nGate on score"),
             (4.4, 2.5, 2.2, .9, "Identity encoder\nGallery matching"),
             (4.4, .75, 2.2, .9, "Expression CNN\nDisplay only"),
             (7.2, 2.5, 2.65, .9, "Known or reviewed\nAttendance record"),
             (7.2, .75, 2.65, .9, "Unknown or ambiguous\nRegister or review")]
    for x, y, w, h, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.06,rounding_size=.08",
                                   facecolor="#f0f4f7", edgecolor="#6f8496", linewidth=1))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=9)
    for start, end in [((1.82, 2.1), (2.1, 2.1)), ((3.98, 2.1), (4.28, 2.95)),
                       ((.92, 1.57), (.92, 1.2)), ((.92, 1.2), (4.28, 1.2)),
                       ((6.68, 2.95), (7.08, 2.95)),
                       ((6.68, 2.95), (7.08, 1.2))]:
        ax.annotate("", end, start, arrowprops={"arrowstyle": "->", "color": "#415365", "lw": 1.1})
    ax.text(2.2, .22, "Failed or uncalibrated liveness blocks identity, registration and review.", fontsize=9)
    fig.tight_layout(pad=.4)
    fig.savefig(ASSETS / "system_flow.png", dpi=250)
    plt.close(fig)
    # Raster math avoids symbol substitution in cross-platform OMML import.
    math_to_image(r"$L_{\mathrm{triplet}}=\max(0,\Vert z_a-z_p\Vert_2-\Vert z_a-z_n\Vert_2+0.2)$",
                  ASSETS/"triplet_loss.png", prop=FontProperties(size=12), dpi=350, color="black")
    math_to_image(r"$s_{\mathrm{cos}}=u^{\mathsf{T}}v,\qquad d^2=2-2s_{\mathrm{cos}}$",
                  ASSETS/"similarity_identity.png", prop=FontProperties(size=12), dpi=350, color="black")
    return operating


def math_run(text):
    r = OxmlElement("m:r")
    t = OxmlElement("m:t")
    t.text = text
    r.append(t)
    return r


def math_script(base, script, kind="sub"):
    node = OxmlElement("m:sSub" if kind == "sub" else "m:sSup")
    e, s = OxmlElement("m:e"), OxmlElement("m:sub" if kind == "sub" else "m:sup")
    e.append(math_run(base)); s.append(math_run(script))
    node.extend([e, s])
    return node


def math_norm_difference(left, right):
    node = OxmlElement("m:sSub")
    e, sub = OxmlElement("m:e"), OxmlElement("m:sub")
    delimiter, properties, contents = OxmlElement("m:d"), OxmlElement("m:dPr"), OxmlElement("m:e")
    for key in ("begChr", "endChr"):
        value = OxmlElement("m:"+key); value.set(qn("m:val"), "‖"); properties.append(value)
    contents.extend([math_script("z", left), math_run(" − "), math_script("z", right)])
    delimiter.extend([properties, contents]); e.append(delimiter)
    sub.append(math_run("2")); node.extend([e, sub])
    return node


class Report:
    def __init__(self):
        self.doc = Document()
        section = self.doc.sections[0]
        section.page_width, section.page_height = Inches(8.5), Inches(11)
        section.left_margin = section.right_margin = Inches(.8)
        section.top_margin = Inches(.65); section.bottom_margin = Inches(.65)
        section.header_distance = section.footer_distance = Inches(.25)
        for name in ["Normal", "Title", "Subtitle", "Heading 1", "Heading 2", "Heading 3", "Caption"]:
            style = self.doc.styles[name]
            style.font.name = "Calibri"
            style.font.color.rgb = RGBColor(0, 0, 0)
            style.font.size = Pt(11)
            style.paragraph_format.space_after = Pt(6)
            style.paragraph_format.line_spacing = 1.06
            style.paragraph_format.widow_control = True
            rpr = style.element.get_or_add_rPr()
            color = rpr.find(qn("w:color"))
            if color is not None:
                for attr in list(color.attrib):
                    if attr != qn("w:val"):
                        del color.attrib[attr]
            for border in list(style.element.iter(qn("w:pBdr"))):
                border.getparent().remove(border)
        for name, size in [("Title", 22), ("Heading 1", 17), ("Heading 2", 14), ("Heading 3", 12)]:
            self.doc.styles[name].font.size = Pt(size)
            self.doc.styles[name].font.bold = True
            self.doc.styles[name].paragraph_format.keep_with_next = True
            self.doc.styles[name].paragraph_format.space_before = Pt(9 if name != "Title" else 0)
        self.doc.styles["Caption"].font.size = Pt(9)
        self.doc.styles["Caption"].font.italic = False
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        footer.add_run("COS30082 Group report    Page ")
        for field, fallback in [("PAGE", "1"), ("NUMPAGES", "12")]:
            if field == "NUMPAGES":
                footer.add_run(" of ")
            element = OxmlElement("w:fldSimple"); element.set(qn("w:instr"), field)
            run = OxmlElement("w:r"); text = OxmlElement("w:t"); text.text = fallback
            run.append(text); element.append(run); footer._p.append(element)
        for run in footer.runs:
            run.font.size = Pt(9)
        self.doc.core_properties.title = "Facial recognition attendance with expression and liveness"
        self.doc.core_properties.subject = "COS30082 group project report"
        self.doc.core_properties.author = ""
        self.doc.core_properties.keywords = "face verification, liveness, facial expression, attendance"

    def p(self, text, style=None, size=None):
        p = self.doc.add_paragraph(text, style)
        if size is not None:
            for run in p.runs:
                run.font.size = Pt(size)
        return p

    def head(self, text, level=2, page=False):
        p = self.doc.add_heading(text, level)
        p.paragraph_format.page_break_before = page
        return p

    def equation(self, filename, description):
        p = self.doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(7)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        path = ASSETS/filename
        with Image.open(path) as source:
            width = source.width / source.info["dpi"][0]
        shape = p.add_run().add_picture(str(path), width=Inches(min(width, 6.8)))
        shape._inline.docPr.set("descr", description)

    def table(self, headers, records, widths, number, title):
        self.p(f"Table {number}  {title}", "Caption")
        table = self.doc.add_table(rows=1, cols=len(headers))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        for column, width in zip(table.columns, widths):
            column.width = Inches(width)
        for cell, text in zip(table.rows[0].cells, headers):
            cell.text = text
        repeat = OxmlElement("w:tblHeader"); table.rows[0]._tr.get_or_add_trPr().append(repeat)
        for record in records:
            cells = table.add_row().cells
            for cell, text in zip(cells, record):
                cell.text = str(text)
        for row_index, row in enumerate(table.rows):
            no_split = OxmlElement("w:cantSplit"); row._tr.get_or_add_trPr().append(no_split)
            for column_index, (cell, width) in enumerate(zip(row.cells, widths)):
                cell.width = Inches(width)
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                properties = cell._tc.get_or_add_tcPr()
                shading = OxmlElement("w:shd")
                shading.set(qn("w:fill"), "263F57" if row_index == 0 else "F1F4F7" if row_index % 2 == 0 else "FFFFFF")
                properties.append(shading)
                margins = OxmlElement("w:tcMar")
                for side, value in [("top", "65"), ("bottom", "65"), ("left", "80"), ("right", "80")]:
                    margin = OxmlElement("w:"+side); margin.set(qn("w:w"), value); margin.set(qn("w:type"), "dxa"); margins.append(margin)
                properties.append(margins)
                borders = OxmlElement("w:tcBorders")
                for side in ("top", "bottom", "left", "right"):
                    border = OxmlElement("w:"+side)
                    for key, value in [("val", "single"), ("sz", "5"), ("color", "D9D9D9")]:
                        border.set(qn("w:"+key), value)
                    borders.append(border)
                properties.append(borders)
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(1)
                    p.paragraph_format.line_spacing = 1.02
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT if column_index == 0 else WD_ALIGN_PARAGRAPH.CENTER
                    for run in p.runs:
                        run.font.size = Pt(10)
                        run.font.bold = row_index == 0
                        run.font.color.rgb = RGBColor(255, 255, 255) if row_index == 0 else RGBColor(0, 0, 0)
        self.p("").paragraph_format.space_after = Pt(1)

    def picture(self, name, caption, alt):
        p = self.doc.add_paragraph()
        p.paragraph_format.keep_with_next = True
        shape = p.add_run().add_picture(str(ASSETS/name), width=Inches(6.85))
        shape._inline.docPr.set("descr", alt)
        self.p(caption, "Caption")

    def reference(self, number, text, url=None):
        p = self.p(f"[{number}] {text}", size=10)
        p.paragraph_format.space_after = Pt(8)
        if url:
            p.add_run(" ")
            rel = self.doc.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
            link = OxmlElement("w:hyperlink"); link.set(qn("r:id"), rel)
            run = OxmlElement("w:r"); props = OxmlElement("w:rPr")
            color = OxmlElement("w:color"); color.set(qn("w:val"), "234F78"); props.append(color)
            run.append(props); token = OxmlElement("w:t"); token.text = "Source"; run.append(token)
            link.append(run); p._p.append(link)


def build():
    identity, app, live, expression, current_live = evidence()
    operating = figures(identity, app, expression)
    live_m, exp_m, current_live_m = live["metrics"], expression["metrics"], current_live["metrics"]
    software = software_evidence()
    report = Report()
    P, H, T = report.p, report.head, report.table

    # Page 1
    P("Facial recognition attendance with expression and liveness", "Title")
    P("COS30082 Applied Machine Learning Group Report", "Subtitle")
    P("8 November 2026", size=10)
    P("Group name or number   __________________________________________")
    P("Members and student IDs   ________________________________________")
    P("______________________________________________________________")
    H("Abstract")
    P("We developed an attendance prototype that combines face verification, passive liveness detection and facial expression classification. Two compact identity CNNs were trained from scratch using classification and triplet metric learning, then compared on the original 8,805 verification pairs. Triplet achieved the higher validation AUC, 0.859969 versus 0.825601, and supplies the application's embeddings. The current liveness model achieved AUC 0.929084 and ACER 15.22% in regression on reused LCC FASD test predictions. Expression classification achieved 58.36% accuracy and macro F1 0.4984 on filtered FER2013 PrivateTest.")
    P("The English webcam interface supports registration, separate appearance templates and manual review with recorded provenance. Identity validation includes 667 training pixel duplicates and TPR is only 9.73% near a 1% pair FPR. An operator-labelled phone still photo passed PAD on all 237 scored crops; identity rejection prevented attendance in that trial. Convenience camera trials, a single-person gallery and passing software tests do not establish deployment accuracy or security; independent gallery and human-review studies remain pending.")
    H("Introduction")
    P("The project brief requires compact CNN embeddings, both classification and metric learning, ROC and AUC evaluation, a trained anti spoofing module, expression recognition and an attendance interface [1]. Our design separates the prediction tasks from the attendance decision. Registration adds a person's name and embeddings without retraining the identity classifier. Liveness gates identity processing; expression labels are descriptive outputs and never determine attendance.")
    P("This report explains the data protocol, architectures, optimisation and model selection, followed by measured results and their practical limits. The main design decision is to retain a strict identity operating point and provide explicit human review for uncertain cases, while recording that a human decision is distinct from a correct CNN prediction.")

    # Page 2
    H("Methodology", level=1, page=True)
    H("System framework and data protocol")
    P("Capture converts BGR to RGB on its own thread and keeps only the latest frame. A separate inference worker locates one face and rejects small, out of frame, blurred or poorly exposed crops. PAD and expression use the same original RGB crop; expression remains available when PAD fails, while identity and all attendance or enrolment actions require passed PAD. Tk renders independently. Stale observations cannot authorise writes. Figure 1 shows the logical branches; model calls within the inference worker remain sequential.")
    report.picture("system_flow.png", "Figure 1  Logical processing and attendance flow", "A quality-passed RGB crop feeds PAD and expression separately. Passed PAD permits identity matching, registration, explicit review and attendance; expression is display only.")
    T(["Task", "Training", "Validation", "Final evaluation"], [
        ["Identity images [2]", "380,638 / 4,000 IDs", "8,000 classification", "8,000 classification"],
        ["Identity verification [2]", "No pair training", "8,805 original pairs", "No labelled independent set"],
        ["LCC FASD [4,5]", "8,093", "2,948", "7,522 filtered"],
        ["FER2013 [6,7]", "27,170", "3,562 PublicTest", "3,271 PrivateTest"]
    ], [1.3, 1.7, 1.6, 2.25], 1, "Data actually used in the experiments")
    P("The identity dataset contains 465,735 images and the required pair file has 4,315 same identity and 4,490 different identity trials using 9,355 distinct images. ZIP CRC, decoding, file hashes and decoded pixel hashes were audited. The original pair bytes and labels were preserved. All 4,000 training IDs have at least two images, enabling distinct positive examples for triplet sampling.")
    P("Pixel duplication was detected across dataset roles: 667 images appearing in verification validation match classification training pixels. Validation AUC is therefore reported under the assignment protocol, not as an independent verification test. Hash separation also cannot prove subject separation. LCC FASD and FER2013 use explicit filtered protocols described below; their scores are not directly comparable to results on the original unfiltered benchmarks.")

    # Page 3
    H("Face verification models", page=True)
    P("The two identity methods share the same randomly initialised FaceEncoder, RGB 64 by 64 input and 128 dimensional projection. Images are EXIF oriented, converted to RGB, bilinearly resized and mapped to [-1,1]. A block contains a 3 by 3 convolution with stride 2, BatchNorm and ReLU, followed by a stride 1 convolution, BatchNorm and ReLU. Convolutions have padding 1 and no bias. Table 2 also shows the backbone resolution used by liveness.")
    T(["Stage", "Output channels", "Identity spatial size", "Liveness spatial size"], [
        ["Input", "3", "64 by 64", "112 by 112"],
        ["Block 1", "16", "32 by 32", "56 by 56"],
        ["Block 2", "32", "16 by 16", "28 by 28"],
        ["Block 3", "64", "8 by 8", "14 by 14"],
        ["Block 4", "128", "4 by 4", "7 by 7"],
        ["Adaptive average pool", "128", "2 by 2", "2 by 2"],
        ["Flatten and linear", "128 features", "512 to 128", "512 to 128"]
    ], [1.85, 1.15, 1.9, 1.95], 2, "Shared backbone architecture")
    H("Classification and metric objectives", level=3)
    P("The classification model attaches a Linear(128,4000) identity head to raw projection features and minimises mean cross entropy over the identity logits. Softmax is handled by the loss during training. At verification time, the head is discarded and each 128 dimensional feature is L2 normalised. This representation can be matched to newly registered identities rather than being restricted to the training classifier's labels.")
    P("The metric model uses an L2 normalised embedding and one triplet for every eligible anchor per epoch. The positive is a different image with the same training ID; the negative has a different ID. Sampling is deterministic for the seed, epoch and anchor. The loss below uses unsquared L2 distances, margin 0.2 and batch mean reduction. This is a compact random triplet baseline inspired by embedding based verification [3], not a reproduction of FaceNet's architecture or online mining strategy.")
    report.equation("triplet_loss.png", "Triplet loss is max of zero and the anchor-positive L2 distance minus anchor-negative L2 distance plus margin 0.2.")
    P("Identity labels supply both classification targets and triplet relationships. Neither method uses verification pairs to update network weights. No data augmentation is applied to either identity model, keeping preprocessing and augmentation consistent for the comparison.")

    # Page 4
    H("Identity training and similarity selection", page=True)
    T(["Setting", "Classification", "Triplet metric learning"], [
        ["Initialisation and seed", "From scratch / 42", "From scratch / 42"],
        ["Epochs and batch size", "10 full epochs / 256", "10 full epochs / 256 anchors"],
        ["Examples per epoch", "380,638", "380,638 anchors, three views each"],
        ["Optimizer", "Adam", "Adam"],
        ["Learning rate", "0.001, fixed", "0.001, fixed"],
        ["Adam configuration", "betas 0.9 and 0.999, epsilon 1e-8", "Same"],
        ["Weight decay", "0", "0"],
        ["Epoch selection", "Minimum classification validation loss", "Minimum triplet training loss"],
        ["Selected epoch", "9", "10"]
    ], [1.85, 2.5, 2.5], 3, "Identity experiment configuration")
    P("Both runs use a GeForce RTX 3060 Laptop GPU with 6 GiB memory, PyTorch 2.5.1 with CUDA 12.4, and four CPU compute threads. Every epoch contains 1,487 batches with no cap. A checksum verified uint8 cache of the source ZIP reduces repeated small file reads without altering the PIL preprocessing. The methods share architecture, seed and anchor count, but triplets process three images per anchor, so they are not matched for total computation. Deterministic settings were requested; PyTorch warned that adaptive pooling backward on CUDA is not guaranteed deterministic.")
    P("We compare cosine similarity and Euclidean distance on unit embeddings. Cosine needs a dot product and has a bounded scale; Euclidean distance offers an intuitive geometric distance consistent with embedding learning [3]. Their mathematical relationship means they should rank unit embedding pairs equivalently, apart from numerical ties:")
    report.equation("similarity_identity.png", "For unit embeddings u and v, cosine is their dot product and squared Euclidean distance equals two minus twice cosine.")
    P("ROC treats label 1 as the same identity, using cosine or negative Euclidean distance so larger scores mean a stronger match. AUC is computed on all original verification validation pairs. The evaluation tool also saves Youden thresholds. For the application we separately choose the highest TPR among finite cosine thresholds with validation pair FPR at most 1%, then use the model with higher validation AUC. This selection is in sample. Classification test images are evaluated once after epoch selection; they are not used to choose the checkpoint.")

    # Page 5
    H("Liveness model and evaluation protocol", page=True)
    P("LCC FASD targets replay attacks across recording and display devices [4]; we use the Kaggle distribution [5]. Its 18,827 images comprise 1,942 genuine and 16,885 spoof images. We preserve the development split, remove 202 training images with 12 filename derived subject codes overlapping development, and remove four further training images duplicated in validation. Evaluation excludes 58 byte duplicates of training or validation. The resulting training, validation and evaluation sizes are 8,093, 2,948 and 7,522.")
    P("The retained training split has 1,193 genuine and 6,900 spoof images; validation has 405 and 2,543; evaluation has 256 and 7,266. Training and validation subject codes are separated under the filename convention. Evaluation filenames are anonymous, so subject or video separation cannot be verified. Original files are retained and exclusions are recorded in manifests. Evaluation does not determine which training images are kept, and it is not used for model or threshold selection.")
    P("LivenessCNN reuses only the backbone architecture in Table 2, with RGB 112 by 112 input, raw 128 dimensional projection and a Linear(128,2) head. Its own weights are randomly initialised; no identity checkpoint is loaded. The two logit classes are spoof 0 and live 1, reversing the source dataset's label convention. Supplied face centred crops are used in full, bilinearly resized and normalised to [-1,1].")
    T(["Training choice", "Historical baseline", "Current app model"], [
        ["Loss", "Unweighted cross entropy", "Unweighted cross entropy"],
        ["Adam learning rate", "0.001", "0.0005"],
        ["Epochs / batch", "10 / 32", "10 / 64"],
        ["Regularisation", "None; weight decay 0", "Flip, light and upper-face patch; decay 0.0001"],
        ["Seed / device", "42 / CPU, four threads", "42 / RTX 3060 Laptop"],
        ["Selected checkpoint", "Epoch 1; validation loss", "Epoch 10; validation ACER"],
        ["Validation threshold", "0.072996063; Youden J", "0.002144532; APCER constrained"]
    ], [1.75, 2.5, 2.6], 4, "Historical and current liveness configurations")
    P("The current CNN is independently retrained from scratch with horizontal flipping, brightness and contrast changes, and small translucent upper-face patches; validation and inference are not augmented. Patches simulate occlusion, not realistic labelled glasses. After checkpoint selection, the current threshold maximises validation TPR subject to APCER not exceeding the baseline validation APCER. No camera or test scores choose this threshold. APCER measures spoof acceptance, BPCER genuine rejection, and ACER their mean; AUC measures ranking.")

    # Page 6
    H("Facial expression model and training", page=True)
    P("We use the FER2013 distribution [7], associated with the facial expression challenge described by Goodfellow et al. [6]. Filename prefixes recover Training, PublicTest and PrivateTest roles. All 35,887 images were decoded. Within each split, identical decoded pixels with conflicting labels are removed, while consistent duplicates retain one copy. Training copies of validation pixels and final test copies of retained development pixels are then excluded. This leaves 27,170 training, 3,562 validation and 3,271 test images.")
    P("The procedure changes the original benchmark population and verifies image path and decoded pixel separation, not subject independence. The seven fixed labels are angry, disgust, fear, happy, sad, surprise and neutral. Inputs are grayscale 48 by 48 tensors normalised to [-1,1]. EmotionCNN is trained from scratch; it classifies visible FER expression categories and is not validated to infer internal emotional state, mental health or engagement.")
    T(["Stage", "Operations", "Output"], [
        ["Block 1", "Two 3 by 3 convolutions, BN, ReLU; max pool; dropout 0.10", "16 by 24 by 24"],
        ["Block 2", "Same structure; dropout 0.15", "32 by 12 by 12"],
        ["Block 3", "Same structure; dropout 0.20", "64 by 6 by 6"],
        ["Pooling and head", "Adaptive pool 3 by 3; flatten; Linear 576 to 128; ReLU; dropout 0.40", "128 features"],
        ["Output", "Linear 128 to 7", "Seven logits"]
    ], [1.25, 3.95, 1.65], 5, "Expression CNN architecture")
    P("Training uses weighted cross entropy with class weights proportional to the square root of inverse training class frequency, normalised to mean one. The disgust class receives weight 2.5193 versus 0.5737 for happy. Validation uses unweighted cross entropy. Adam uses learning rate 0.001 and weight decay 0.0001, batch 64, seed 42 and four CPU threads for 15 full epochs. Training augmentation is horizontal flipping with probability 0.5 and integer translations up to two pixels with replicated borders; validation and test have no augmentation.")
    P("A ReduceLROnPlateau scheduler monitors validation macro F1 with factor 0.5 and patience 3; the recorded learning rate remained 0.001 throughout this run. The checkpoint with maximum validation macro F1 is selected, breaking ties by lower validation loss. Epoch 15 was selected. PrivateTest is evaluated only after selection. Macro F1 averages per class F1, while balanced accuracy averages per class recall, preventing the largest class from dominating the report.")

    # Page 7
    H("Application and manual review", page=True)
    P("The locator combines two supplied OpenCV Haar cascades [8], using histogram equalisation and CLAHE, scale factor 1.08 and four neighbours on images at most 800 pixels wide. Overlapping duplicates are suppressed, retaining an actual detected box. Exactly one face must remain, with crop size at least 80 by 80, grayscale mean 20 to 235 and Laplacian variance at least 20. The CNNs receive original RGB pixels. Haar is pretrained preprocessing, separate from the three CNNs trained from scratch. Manual ROI avoids Haar but cannot detect extra people outside its region.")
    P("Camera registration collects three distinct live crops at least 0.45 seconds apart and rejects inconsistent embeddings. SQLite profiles are tied to the encoder SHA256. Each person can retain up to four appearance centroids, including separate looks with and without glasses. Matching uses the maximum cosine within each person, then compares different people: threshold 0.869540211 and runner-up gap 0.05. The gap is heuristic. Three consecutive valid frames confirm attendance; SQLite permits one record per person per session.")
    P("Profiles supports search, renaming and appearance removal with a database backup; the final appearance cannot be removed. Reference import checks three distinct photos for one face, image quality, PAD and consistency, followed by explicit operator confirmation within 120 seconds. It adds embeddings, not CNN training or attendance. PAD on a static photo cannot establish live presence. Adding an appearance preserves the person's ID and attendance history.")
    H("Human review with recorded provenance", level=3)
    P("For an uncertain live face, review requires an explicit profile choice, reviewer name, reason and confirmation; none is preselected. A ticket expires after 30 seconds and needs a live observation within 1.5 seconds, the same session and embedding continuity. Lost, multiple or poor quality faces, failed PAD or changed embeddings invalidate it. Automatic recording pauses while review remains open. Attendance and audit metadata commit atomically, preserving scores and existing automatic provenance. Manual decisions do not update the encoder or templates; older unaudited rows retain unknown source.")
    P("One enlarged preview uses yellow outlines for unrecognised faces and green for a known live profile; fixed result slots prevent resizing. Camera analyses older than one second are discarded before enrolment or automatic attendance writes. History and CSV retain sources; camera testing and Diagnostics support setup recovery. Guidance separates face, quality and PAD failures.")
    P("Expression is display only. The app saves no camera images, review crops or expression labels. Names, embeddings and attendance metadata remain sensitive. Reviewer names are freely entered; authentication, operator roles and tamper resistant audit storage are not implemented.")

    # Page 8
    H("Results", level=1, page=True)
    H("Identity comparison and operating point")
    T(["Method", "Selected epoch", "Cosine AUC", "Negative L2 AUC"], [
        [m["method"].capitalize(), m["checkpoint_config"]["training"]["epoch"],
         f"{next(v['auc'] for v in m['metrics'] if v['metric']=='cosine_similarity'):.6f}",
         f"{next(v['auc'] for v in m['metrics'] if v['metric']=='euclidean_distance'):.6f}"]
        for m in identity["models"]
    ], [2.15, 1.35, 1.65, 1.7], 6, "Verification on all 8,805 original validation pairs")
    report.picture("identity_roc.png", "Figure 2  Identity ROC curves and the application's low FPR operating region", "Triplet AUC is 0.859969 and classification AUC is 0.825601; the selected triplet point has TPR 9.73 percent at pair FPR 0.98 percent.")
    P("Triplet improves validation AUC by 0.034368 over classification under this protocol. Cosine and negative Euclidean AUC coincide as expected from unit embedding geometry. Triplet is selected for the app; the corresponding cosine threshold is 0.869540211. At that point, 420 of 4,315 same identity pairs are accepted and 44 of 4,490 different identity pairs are falsely accepted: TPR 9.73%, FPR 0.98% and false rejection 90.27%. Classification reaches TPR 8.23% at FPR 0.96% under the same constraint.")
    P("The classification head, restored from epoch 9, achieves 39.4625% closed set test accuracy on 8,000 images. Its validation loss was 3.221723 at selection. This measures recognition among 4,000 training identities and is distinct from verification AUC or new person recognition. Neither the AUC nor the strict operating point is an unbiased independent test: both model comparison and calibration use verification validation, with the known train pixel overlap.")

    # Page 9
    H("Liveness and expression results", page=True)
    T(["Liveness measure", "Historical baseline", "Current app regression"], [
        ["AUC", f"{live_m['AUC']:.6f}", f"{current_live_m['AUC']:.6f}"],
        ["Accuracy", f"{100*live_m['accuracy']:.2f}%", f"{100*current_live_m['accuracy']:.2f}%"],
        ["APCER", f"{100*live_m['APCER']:.2f}%", f"{100*current_live_m['APCER']:.2f}%"],
        ["BPCER", f"{100*live_m['BPCER']:.2f}%", f"{100*current_live_m['BPCER']:.2f}%"],
        ["ACER", f"{100*live_m['ACER']:.2f}%", f"{100*current_live_m['ACER']:.2f}%"]
    ], [2.35, 2.25, 2.25], 7, "LCC FASD evaluation and regression on the same 7,522 images")
    P("The current operating point accepts 227 of 256 genuine images and 1,388 of 7,266 spoof images. These are regression results on reused original test predictions, not a new independent test or a webcam guarantee. Selection used validation only. Validation BPCER is worse than the baseline, 13.58% versus 10.12%, while APCER is 33.94% versus 34.41%; improvement is not universal. Baseline training loss decreases while validation loss rises after epoch 1, consistent with overfitting. Dataset imbalance makes accuracy alone misleading.")
    T(["Expression measure", "Filtered PrivateTest"], [
        ["Accuracy", f"{100*exp_m['accuracy']:.2f}%"], ["Macro F1", f"{exp_m['macro_f1']:.4f}"],
        ["Balanced accuracy", f"{100*exp_m['balanced_accuracy']:.2f}%"]
    ], [3.8, 3.05], 8, "Expression evaluation after checkpoint selection")
    report.picture("expression_f1.png", "Figure 3  Expression F1 and test support for each class", "Happy F1 is 0.821, while disgust and fear F1 are 0.198 and 0.256. Supports range from 38 to 845 images.")
    P("Happy has the strongest F1, 0.8209, followed by surprise, 0.6667. Disgust has only 38 test examples and F1 0.1975; fear has F1 0.2565 and recall 18.58%. Weighting and augmentation have not resolved these weaknesses. Validation macro F1 is 0.5218 versus 0.4984 on PrivateTest. The filtered protocol and unknown subject overlap limit comparisons with published FER2013 benchmarks.")

    # Page 10
    H("Integration and verification evidence", page=True)
    P(f"The current readiness suite passes {software['passed']} tests in {software['seconds']:.3f} seconds. It covers training fixtures, recognition gates, independent expression display, appearance capture/import, database integrity, history/CSV, device recovery, stable single-view rendering and review. These functional checks do not estimate recognition accuracy.")
    T(["Check", "Observed outcome", "Interpretation"], [
        ["Original verification protocol", "8,805 pairs; SHA256 unchanged", "Assignment pair file preserved"],
        ["Current software suite", f"{software['passed']} of {software['total']} passed", "Functional checks, not model quality"],
        ["Historical offline flow", "21 of 24 images passed PAD", "Train photos; not a camera benchmark"],
        ["Unregistered probes", "Two false gallery matches", "Identity errors remain"],
        ["Manual review integration", "Unknown before and after review", "Human decision stored separately"],
        ["Device and setup checks", "Camera 0 returned 640 by 480; setup retry recovered", "One-frame device check, not PAD accuracy"]
    ], [2.0, 2.15, 2.7], 9, "Implementation evidence and its scope")
    P("Historical offline integration used 24 training photos and labelled scratch profiles: 21 passed PAD, two unregistered probes matched the gallery, and one scratch attendance row was created. These are not held out accuracy results. Reference import creates templates without attendance in a separate temporary-database check.")
    P("Scripted review records provenance for ID n000003 at cosine 0.842151, below threshold 0.869540. The model remains unknown, checkpoint hashes remain unchanged and failed PAD blocks review. The actor is a software fixture; no human-review accuracy or productivity gain is claimed.")
    P(camera_trial_text())
    P("Expression-only CPU timing used four threads, ten warmups and 100 calls: median 1.11 ms, 95th percentile 1.37 ms. Capture, localisation, PAD, identity and display are excluded; this is not end to end webcam FPS.")

    # Page 11
    H("Discussion", level=1, page=True)
    H("Model selection and deployment limits")
    P("The evidence supports selecting triplet as the stronger of the two tested identity baselines by validation ranking and the low FPR operating point. It does not support calling the system 86% accurate in deployment. At the chosen threshold, most positive verification pairs are rejected. Random triplet loss reduction alone does not establish separation among difficult impostors. Hard triplet mining and stronger augmentation remain untested [3].")
    P("The comparison controls architecture, input resolution, initial seed and anchor epochs. It does not equalise compute, and it uses one run per method. No uncertainty intervals or repeated seed estimates are available. Verification validation is used for model choice and threshold calibration, and contains exact training pixel duplicates. Future work should retain the required original result and add an explicitly separate cleaned training protocol and an independent gallery test, rather than silently filtering the assignment's pair list.")
    P("Current PAD regression APCER/BPCER are 19.10%/11.33% on reused test data, with validation genuine rejection worse than the baseline. In the historical phone still-photo trial, all 237 scored attack crops passed PAD; identity returned unknown and prevented scratch attendance, so zero attendance does not show effective PAD. A 20-epoch regularised PAD run selected epoch 6 by validation ACER; APCER/BPCER/ACER were 16.67%/22.96%/19.82%, failing predeclared ACER below 19.34% and BPCER at most 13.58% limits. It was not promoted; no candidate test or camera evaluation informed selection. Separate participant, device and printed/displayed attack tests remain necessary; crop-domain shift and dependent errors prevent extrapolating offline scores to webcam security.")
    P("Expression classification is strongest for happy and weak for fear and disgust. Test support and class confusion must accompany aggregate accuracy. The model has not been validated for internal feelings, mental health, work attitude or student engagement. Keeping expression output outside the attendance rule avoids introducing an unsupported condition for presence recording.")
    H("Practical improvements and compliance", level=3)
    P("Manual review makes uncertain decisions explicit and traceable without disguising them as correct model predictions. Its current freely entered reviewer name is not authentication. Production use would need operator access control, protected logs, data retention rules and a study of reviewer error and effort. Embeddings remain biometric data, even though camera images are not saved. The default project database is in a OneDrive folder and may be synchronised by that service; a different database path can be configured.")
    P("The supplied Haar cascade is a separate pretrained preprocessing component [8]. A self trained locator would remove ambiguity about the brief's restriction, while manual ROI mode avoids that dependency at the cost of automatic localisation. Remaining validation should measure whole system latency and gallery false acceptance and rejection across people, sessions and devices, and compare automatic only operation with operator assisted operation using transparent decision labels.")

    # Page 12 -- cap includes references and all front matter.
    H("Conclusion", level=1, page=True)
    P("The project trains both required identity approaches, a liveness CNN and an expression CNN, and integrates them with registration, appearance templates and attendance. Triplet is the selected identity baseline. Leakage, weak low FPR recall, spoof acceptance and limited camera evidence prevent deployment claims. Manual review preserves scores and records operator provenance. Profile management, history export and device diagnostics improve operation; independent gallery and attack evaluation remains necessary.")
    H("References")
    report.reference(1, "COS30082 Applied Machine Learning. Facial Recognition with Emotion and Liveness. Project brief supplied with the assignment, pp. 1 to 6.")
    report.reference(2, "Kaggle. 11 785 Fall 20 Homework 2 Part 2. Identity dataset and verification pairs. Accessed 27 October 2026.", "https://www.kaggle.com/competitions/11-785-fall-20-homework-2-part-2/data")
    report.reference(3, "Schroff, F., Kalenichenko, D. and Philbin, J. (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. CVPR. arXiv 1503.03832.", "https://arxiv.org/abs/1503.03832")
    report.reference(4, "Timoshenko, D., Simonchik, K., Shutov, V., Zhelezneva, P. and Grishkin, V. (2019). Large Crowdcollected Facial Anti-Spoofing Dataset. CSIT.", "https://pureportal.spbu.ru/files/51584200/CSIT2019_Grishkin_Timoshenko.pdf")
    report.reference(5, "Kaggle contributor faber24. LCC FASD dataset distribution, version 1. Accessed 27 October 2026.", "https://www.kaggle.com/datasets/faber24/lcc-fasd")
    report.reference(6, "Goodfellow, I. J. et al. (2013). Challenges in Representation Learning: A report on three machine learning contests. arXiv 1307.0414.", "https://arxiv.org/abs/1307.0414")
    report.reference(7, "Kaggle contributor msambare. FER2013 dataset distribution, version 1. Accessed 27 October 2026.", "https://www.kaggle.com/datasets/msambare/fer2013")
    report.reference(8, "OpenCV. Cascade Classifier. OpenCV 4.13.0 documentation. Accessed 27 October 2026.", "https://docs.opencv.org/4.13.0/db/d28/tutorial_cascade_classifier.html")
    H("Reproduction evidence", level=3)
    P("Configurations, hashes and split audits accompany the checkpoints. Main evidence is in comparison/comparison.json, identity_app_config.json, liveness_lcc_test/metrics.json, liveness_camera_matched/test_regression_metrics.json and calibration.json, emotion_fer_test/metrics.json, and readiness_20261108/status.json. Historical webcam trials remain under app_upgrade_20261030/evaluation. Paths are relative to results. The README records preparation, training and evaluation commands.", size=10)

    OUTPUT.parent.mkdir(exist_ok=True)
    report.doc.save(OUTPUT)
    metadata = {"output": str(OUTPUT), "language": "English", "group_details": "left blank as requested",
                "page_limit": 12, "intended_pages": 12, "operating_points": operating,
                "source_files": {p: digest(ROOT/p) for p in ["results/comparison/comparison.json",
                    "results/identity_app_config.json", "results/liveness_lcc_test/metrics.json",
                    "results/emotion_fer_test/metrics.json", "results/manual_review_integration.json",
                    "results/liveness_app_config.json", "results/liveness_camera_matched/test_regression_metrics.json",
                    "results/liveness_camera_matched/calibration.json", "results/app_upgrade_20261030/phase4_status.json"]}}
    additional_sources = ["results/readiness_20261108/status.json", software["log"],
                          "results/readiness_20261108/attendance_integration.json",
                          "results/readiness_20261108/model_startup.json",
                          "results/readiness_20261108/relocation_check.json",
                          "results/liveness_regularized_20261108/experiment_plan.json",
                          "results/liveness_regularized_20261108/experiment_status.json",
                          "results/liveness_regularized_20261108/calibration.json",
                          "attendance_app/gui.py", "attendance_app/camera.py",
                          "attendance_app/service.py", "attendance_app/video_stream.py",
                          "attendance_pipeline.py", "emotion_recognition/webcam.py"]
    additional_sources.extend(str(path.relative_to(ROOT)) for path in
        sorted((ROOT/"results/app_upgrade_20261030/evaluation/trials").glob("*.json")))
    metadata["source_files"].update({path: digest(ROOT/path) for path in additional_sources if (ROOT/path).is_file()})
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK/"build_evidence.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    build()
