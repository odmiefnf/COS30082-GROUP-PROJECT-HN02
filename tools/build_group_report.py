"""Build the English group report from recorded project evidence using the document runtime."""
from __future__ import annotations

import csv
from collections import defaultdict
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
BLUE, ORANGE, GREEN = "#234f78", "#b2592b", "#2e7d4f"
COMPARISON = "results/comparison_identity_20261121/comparison.json"
GALLERY_NEW = "results/identity_open_set_20261121_arcface_resnet18/metrics.json"
GALLERY_OLD = "results/identity_open_set_20261120_classification_resnet18/metrics.json"
PAD_SCRATCH = "results/liveness_lcc_test/metrics.json"
PAD_LCC = "results/liveness_mobilenetv3_20261120"
PAD_APP = "results/liveness_mobilenetv3_webcam_20261120"
EXPRESSION_OLD = "results/emotion_fer_test/metrics.json"
EXPRESSION_NEW = "results/emotion_fer_mnv3_test/metrics.json"
PIPELINE = "results/app_update_20261120/pipeline_metrics.json"
TESTS_LOG = "results/readiness_20261125/full_tests.log"


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


def software_evidence():
    """Report the completed suite from its saved log."""
    log = (ROOT/TESTS_LOG).read_text(encoding="utf-8-sig")
    match = re.search(r"Ran (\d+) tests in ([\d.]+)s", log)
    assert match and log.rstrip().endswith("OK")
    return {"passed": int(match[1]), "seconds": float(match[2]), "log": TESTS_LOG}


def webcam_sessions(directory):
    """Accepted-as-live crops per recorded webcam session, at the model's own threshold."""
    threshold = read_json(f"{directory}/metrics.json")["metrics"]["threshold"]
    accepted, total = defaultdict(int), defaultdict(int)
    for row in rows(ROOT/directory/"predictions.csv"):
        session = next(p for p in row["image_path"].replace("\\", "/").split("/") if p.startswith("cam_"))
        total[session] += 1
        accepted[session] += float(row["live_score"]) >= threshold
    return {session: (accepted[session], total[session]) for session in total}


def model_name(model):
    backbone = model["checkpoint_config"]["training"].get("backbone", "compact")
    method = {"arcface": "ArcFace"}.get(model["method"], model["method"])
    return f"{'ResNet-18' if backbone == 'resnet18' else 'Compact'} {method}"


def evidence():
    identity = read_json(COMPARISON)
    assert identity["pair_count"] == 8805 and len(identity["models"]) == 5
    assert identity["pairs_file_sha256"] == digest(ROOT/"data/face_recognition/verification_pairs_val.txt")
    for model in identity["models"]:
        assert model["checkpoint_sha256"] == digest(model["checkpoint"])
    app = read_json("results/identity_app_config.json")
    assert app["checkpoint"] == "arcface_resnet18/best_model.pt"
    assert app["checkpoint_sha256"] == digest(ROOT/"results"/app["checkpoint"])
    gallery = read_json(GALLERY_NEW)
    assert gallery["checkpoint_sha256"] == app["checkpoint_sha256"]
    pad_app = read_json("results/liveness_app_config.json")
    assert pad_app["checkpoint"] == "liveness_mobilenetv3_webcam_20261120/best_model.pt"
    assert pad_app["checkpoint_sha256"] == digest(ROOT/"results"/pad_app["checkpoint"])
    pads = {name: read_json(path)["metrics"] for name, path in
            (("scratch", PAD_SCRATCH), ("lcc", f"{PAD_LCC}_test/metrics.json"), ("app", f"{PAD_APP}_lcc_test/metrics.json"))}
    assert all(m["sample_count"] == 7522 for m in pads.values())
    assert read_json(f"{PAD_APP}_on_test/metrics.json")["metrics"]["sample_count"] == 264
    expressions = {name: read_json(path)["metrics"] for name, path in (("old", EXPRESSION_OLD), ("new", EXPRESSION_NEW))}
    for metrics in expressions.values():
        cm = np.array(metrics["confusion_matrix"])
        assert int(cm.sum()) == 3271 and np.isclose(np.trace(cm) / cm.sum(), metrics["accuracy"])
    return identity, app, gallery, pads, expressions


def figures(identity, app, expressions):
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "black", "text.color": "black",
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.25), gridspec_kw={"width_ratios": [1.15, 1]})
    operating = {}
    for model in identity["models"]:
        name = model_name(model)
        color = {"classification": BLUE, "triplet": ORANGE}.get(model["method"], GREEN)
        linestyle = "-" if name.startswith("ResNet") else ":"
        curve = [r for r in rows(model["artifacts"]["roc_points_csv"]) if r["metric"] == "cosine_similarity"]
        x = np.array([float(r["fpr"]) for r in curve])
        y = np.array([float(r["tpr"]) for r in curve])
        t = np.array([float(r["raw_threshold"]) for r in curve])
        auc = next(m["auc"] for m in model["metrics"] if m["metric"] == "cosine_similarity")
        assert np.isclose(np.trapezoid(y, x), auc)
        eligible = [i for i in range(len(x)) if np.isfinite(t[i]) and x[i] <= .01 and -1 <= t[i] <= 1]
        best = max(eligible, key=lambda i: (y[i], t[i]))
        operating[name] = {"fpr": float(x[best]), "tpr": float(y[best]), "threshold": float(t[best]), "auc": auc,
                           "epoch": model["checkpoint_config"]["training"]["epoch"]}
        for axis in axes:
            axis.plot(x, y, linestyle=linestyle, color=color, lw=1.7, label=f"{name}  AUC {auc:.4f}")
            axis.grid(alpha=.22)
            axis.set_xlabel("False positive rate")
            axis.set_ylabel("True positive rate")
    axes[0].plot([0, 1], [0, 1], color="#888888", linestyle=":", lw=1)
    axes[0].set(xlim=(0, 1), ylim=(0, 1), title="All verification pairs")
    axes[1].set(xlim=(0, .05), ylim=(0, .6), title="Low false positive region")
    axes[1].axvline(.01, color="#777777", lw=1, linestyle=":")
    fpr, tpr = app["observed_validation_pair_fpr"], app["observed_validation_pair_tpr"]
    axes[1].scatter([fpr], [tpr], marker="*", s=80, color=GREEN, zorder=4)
    axes[1].annotate(f"App point\nTPR {tpr:.2%}", (fpr, tpr), xytext=(.03, .03), fontsize=9,
                     arrowprops={"arrowstyle": "->", "color": "#555555", "lw": .8})
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, -.01), fontsize=7.6)
    fig.tight_layout(rect=(0, .15, 1, 1), w_pad=1.2)
    fig.savefig(ASSETS / "identity_roc.png", dpi=250)
    plt.close(fig)

    old, new = (expressions[k]["per_class"] for k in ("old", "new"))
    fig, ax = plt.subplots(figsize=(6.85, 2.15))
    index = np.arange(7)
    for offset, records, color, label in ((-.19, old, "#9fb3c8", "EmotionCNN (scratch)"), (.19, new, BLUE, "MobileNetV3 (app)")):
        bars = ax.bar(index + offset, [r["f1"] for r in records], width=.36, color=color, label=label)
        for bar, record in zip(bars, records):
            ax.text(bar.get_x()+bar.get_width()/2, record["f1"]+.02, f"{record['f1']:.2f}", ha="center", fontsize=7.5)
    ax.set_ylim(0, 1.25)  # headroom keeps the legend clear of the tallest bars
    ax.set_ylabel("F1 score")
    ax.set_xticks(index, [f"{r['label']}\nn={r['support']}" for r in new], fontsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=.5)
    fig.savefig(ASSETS / "expression_f1.png", dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.85, 2.15))
    ax.set(xlim=(0, 10), ylim=(0, 4))
    ax.axis("off")
    boxes = [(0.1, 1.65, 1.65, .9, "YuNet faces\nQuality passed"),
             (2.2, 1.65, 1.7, .9, "Liveness CNN\nMain face only"),
             (4.4, 2.5, 2.2, .9, "Identity encoder\nGallery matching"),
             (4.4, .75, 2.2, .9, "Expression CNN\nDisplay only"),
             (7.2, 2.5, 2.65, .9, "Known live main face\nAttendance record"),
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
    ax.text(.1, .22, "Failed liveness blocks identity actions; other faces only receive name tags and are never recorded.", fontsize=9)
    fig.tight_layout(pad=.4)
    fig.savefig(ASSETS / "system_flow.png", dpi=250)
    plt.close(fig)
    # Raster math avoids symbol substitution in cross-platform OMML import.
    math_to_image(r"$L_{\mathrm{triplet}}=\max(0,\Vert z_a-z_p\Vert_2-\Vert z_a-z_n\Vert_2+0.2)$",
                  ASSETS/"triplet_loss.png", prop=FontProperties(size=12), dpi=350, color="black")
    math_to_image(r"$L_{\mathrm{arc}}=-\log\frac{e^{s\cos(\theta_y+m)}}{e^{s\cos(\theta_y+m)}+\sum_{j\neq y}e^{s\cos\theta_j}},\quad s=32,\ m=0.3$",
                  ASSETS/"arcface_loss.png", prop=FontProperties(size=12), dpi=350, color="black")
    math_to_image(r"$s_{\mathrm{cos}}=u^{\mathsf{T}}v,\qquad d^2=2-2s_{\mathrm{cos}}$",
                  ASSETS/"similarity_identity.png", prop=FontProperties(size=12), dpi=350, color="black")
    return operating


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
        p.paragraph_format.space_after = Pt(6)
        if url:
            p.add_run(" ")
            rel = self.doc.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
            link = OxmlElement("w:hyperlink"); link.set(qn("r:id"), rel)
            run = OxmlElement("w:r"); props = OxmlElement("w:rPr")
            color = OxmlElement("w:color"); color.set(qn("w:val"), "234F78"); props.append(color)
            run.append(props); token = OxmlElement("w:t"); token.text = "Source"; run.append(token)
            link.append(run); p._p.append(link)

    def link_line(self, label, url, size=9, space_after=4):
        p = self.p(f"{label} ", size=size)
        p.paragraph_format.space_after = Pt(space_after)
        rel = self.doc.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
        link = OxmlElement("w:hyperlink"); link.set(qn("r:id"), rel)
        run = OxmlElement("w:r"); props = OxmlElement("w:rPr")
        color = OxmlElement("w:color"); color.set(qn("w:val"), "234F78"); props.append(color)
        underline = OxmlElement("w:u"); underline.set(qn("w:val"), "single"); props.append(underline)
        sz = OxmlElement("w:sz"); sz.set(qn("w:val"), str(size * 2)); props.append(sz)
        run.append(props); token = OxmlElement("w:t"); token.text = url; run.append(token)
        link.append(run); p._p.append(link)
        return p


def build():
    identity, app, gallery, pads, expressions = evidence()
    operating = figures(identity, app, expressions)
    ops = {name: operating[name] for name in ("Compact classification", "Compact triplet",
                                                "ResNet-18 classification", "ResNet-18 triplet", "ResNet-18 ArcFace")}
    software = software_evidence()
    pipeline = read_json(PIPELINE)
    detect, latency = pipeline["detector_one_face_counts"], pipeline["latency_ms"]
    g_new, g_old = gallery["splits"]["test"], read_json(GALLERY_OLD)["splits"]["test"]
    gapp, gcal = g_new["current_app_threshold"], g_new["gallery_calibrated_threshold"]
    gold, gold_cal = g_old["current_app_threshold"], g_old["gallery_calibrated_threshold"]
    arc_test = read_json("results/arcface_resnet18/classification_test.json")["metrics"]["accuracy"]
    old_on, new_on = webcam_sessions(f"{PAD_LCC}_on_test"), webcam_sessions(f"{PAD_APP}_on_test")
    old_val, new_val = webcam_sessions(f"{PAD_LCC}_on_webcam_val"), webcam_sessions(f"{PAD_APP}_on_webcam_val_insample")
    session = lambda table, kind: next(v for k, v in table.items() if f"_{kind}_" in k)
    attack_new = sum(session(new_on, kind)[0] for kind in ("phone-still-photo", "recorded-video"))
    attack_total = sum(session(new_on, kind)[1] for kind in ("phone-still-photo", "recorded-video"))
    live_new = session(new_on, "live-person")
    sc, lc, ap = pads["scratch"], pads["lcc"], pads["app"]
    eo, en = expressions["old"], expressions["new"]
    rc, rt, ra = ops["ResNet-18 classification"], ops["ResNet-18 triplet"], ops["ResNet-18 ArcFace"]
    cc, ct = ops["Compact classification"], ops["Compact triplet"]
    accepted = lambda op, n: round(op["tpr"] * n)
    report = Report()
    P, H, T = report.p, report.head, report.table

    # Page 1
    P("Facial recognition attendance with expression and liveness", "Title")
    P("COS30082 Applied Machine Learning Group Report", "Subtitle")
    P("21 November 2026", size=10)
    P("Group HN02", size=11)
    P("Nguyen Van Tuan Kiet - 104772099     Huynh Quoc Hieu - 104848408", size=8)
    P("Nguyen Phuc Lam - 105006540     Bùi Minh Hiếu - 105006582", size=8)
    P("Nguyễn Huy Hoàng Việt - 104992509     Vũ Bá Hiển - 104995634", size=8)
    report.link_line("Demonstration video:", "https://youtu.be/j6G3C0v6YX0", size=9, space_after=2)
    report.link_line("Source code repository:", "https://github.com/odmiefnf/COS30082-GROUP-PROJECT-HN02", size=9, space_after=4)
    H("Abstract")
    P(f"We developed an attendance prototype that combines face verification, passive liveness detection and facial expression classification. Identity encoders were trained with classification and triplet objectives, first as compact CNNs from scratch and then as partially fine-tuned ImageNet ResNet-18 networks, and compared on the original 8,805 verification pairs. Fine-tuning raised validation AUC from {cc['auc']:.3f} and {ct['auc']:.3f} to {rc['auc']:.3f} and {rt['auc']:.3f}, and an additional ArcFace margin objective reached {ra['auc']:.3f}. The ArcFace encoder accepts {ra['tpr']:.2%} of same identity pairs at {ra['fpr']:.2%} pair FPR, against {rc['tpr']:.2%} for ResNet-18 classification and {ct['tpr']:.2%} for the earliest application model, and supplies the application's embeddings.")
    P(f"Liveness uses a partially fine-tuned MobileNetV3 trained on LCC FASD plus our own webcam recordings. On held-out webcam sessions it accepted {attack_new} of {attack_total} phone photo and video crops and passed {live_new[0]} of {live_new[1]} genuine crops; its LCC FASD ACER is {ap['ACER']:.2%}, against {lc['ACER']:.2%} without webcam data. A fine-tuned MobileNetV3 expression model reached {en['accuracy']:.2%} accuracy and macro F1 {en['macro_f1']:.4f} on filtered FER2013 PrivateTest, up from {eo['accuracy']:.2%} and {eo['macro_f1']:.4f}. Identity validation contains training pixel duplicates and the webcam evidence comes from one person, room and day, so these results do not establish deployment accuracy or security.")
    H("Introduction")
    P("The project brief requires CNN embeddings trained with classification and metric learning, ROC and AUC evaluation, a trained anti spoofing module, expression recognition and an attendance interface [1]. Pretrained networks may only be used with further training; our ImageNet initialised models are partially fine-tuned on the task data, and the from-scratch baselines are kept for comparison. Registration adds a person's name and embeddings without retraining. Liveness gates identity processing; expression labels are descriptive outputs and never determine attendance.")
    P("This report explains the data protocols, architectures, optimisation and model selection, followed by measured results and their practical limits. Each model replacement was adopted only after it improved held-out evidence. The main design decisions are a strict identity operating point, explicit human review for uncertain cases, and recording attendance only for a live main face.")

    # Page 2
    H("Methodology", level=1, page=True)
    H("System framework and data protocol")
    P("Capture converts BGR to RGB on its own thread and keeps only the latest frame. An inference worker detects faces with the pretrained YuNet detector [8] and rejects small, out of frame, blurred or poorly exposed crops. The largest face is the main face: liveness and expression use its original RGB crop, and identity, enrolment and attendance require passed liveness. Up to nine further faces are only identified for name tags and are never recorded. A localhost web page renders the stream; stale observations cannot authorise writes. Figure 1 shows the logical branches.")
    report.picture("system_flow.png", "Figure 1  Logical processing and attendance flow", "Detected, quality-passed faces feed liveness for the main face and expression separately. Passed liveness permits identity matching, registration, review and attendance; other faces only receive name tags.")
    T(["Task", "Training", "Validation", "Final evaluation"], [
        ["Identity images [2]", "380,638 / 4,000 IDs", "8,000 classification", "8,000 classification"],
        ["Identity verification [2]", "No pair training", "8,805 original pairs", "No labelled independent set"],
        ["LCC FASD [4,5]", "8,093", "2,948", "7,522 filtered"],
        ["Webcam PAD (ours)", "872 crops, 8 sessions", "211 crops, 3 sessions", "264 crops, 3 sessions"],
        ["FER2013 [6,7]", "27,170", "3,562 PublicTest", "3,271 PrivateTest"]
    ], [1.3, 1.7, 1.6, 2.25], 1, "Data actually used in the experiments")
    P("The identity dataset contains 465,735 images and the required pair file has 4,315 same identity and 4,490 different identity trials using 9,355 distinct images. ZIP CRC, decoding, file hashes and decoded pixel hashes were audited, and the original pair bytes and labels were preserved. All 4,000 training IDs have at least two images, enabling distinct positives for triplet sampling.")
    P("Pixel duplication was detected across dataset roles: 667 images appearing in verification validation match classification training pixels. Validation AUC is therefore reported under the assignment protocol, not as an independent verification test. LCC FASD and FER2013 use the filtered protocols described below, and webcam crops are split by whole recording session; their scores are not comparable to results on the original benchmarks.")

    # Page 3
    H("Face verification models", page=True)
    P("Two identity backbones were trained with each objective, all on RGB 64 by 64 input with a 128 dimensional embedding. The compact FaceEncoder is randomly initialised: four blocks, each a stride 2 and a stride 1 3 by 3 convolution with BatchNorm and ReLU, then adaptive pooling and a linear projection. The ResNet-18 encoder starts from ImageNet weights [10,12]; its max pool is removed so a 64 by 64 face still leaves a 4 by 4 map, the stem convolution stays frozen, and all residual stages and a new projection are trained. Inputs mapped to [−1,1] are converted to ImageNet normalisation inside the model.")
    T(["Stage", "Compact CNN", "ResNet-18 encoder"], [
        ["Initialisation", "Random", "ImageNet, stem frozen"],
        ["Feature stages", "4 blocks, 16 to 128 channels", "Stem and 4 residual stages, 64 to 512 channels"],
        ["Final feature map", "4 by 4 by 128, pooled to 2 by 2", "4 by 4 by 512, global average pool"],
        ["Projection", "Linear 512 to 128", "Linear 512 to 128"],
        ["Parameters", "0.36 million", "11.24 million (11.23 million trained)"]
    ], [1.6, 2.4, 2.85], 2, "Identity encoder architectures")
    H("Classification and metric objectives", level=3)
    P("The classification model attaches a Linear(128,4000) identity head to raw projection features and minimises mean cross entropy over the identity logits. At verification time the head is discarded and each 128 dimensional feature is L2 normalised, so newly registered people can be matched without retraining the classifier.")
    P("As an additional approach, the ArcFace variant [13] replaces the linear head on the ResNet-18 encoder with 4,000 L2 normalised class weights and adds an angular margin m to the true class before scaling the cosines by s. Validation and test accuracy use the plain scaled cosines, so the margin only shapes training; all other settings match ResNet-18 classification.")
    report.equation("arcface_loss.png", "ArcFace loss: cross entropy over scaled cosines, with margin 0.3 added to the angle of the true class and scale 32.")
    P("The metric model uses an L2 normalised embedding and one triplet for every eligible anchor per epoch. The positive is a different image with the same training ID; the negative has a different ID. Sampling is deterministic for the seed, epoch and anchor. The loss below uses unsquared L2 distances, margin 0.2 and batch mean reduction. This is a random triplet baseline inspired by embedding based verification [3], not a reproduction of FaceNet's architecture or online mining.")
    report.equation("triplet_loss.png", "Triplet loss is max of zero and the anchor-positive L2 distance minus anchor-negative L2 distance plus margin 0.2.")
    P("Identity labels supply both classification targets and triplet relationships; neither method uses verification pairs to update weights. Compact runs use no augmentation. ResNet runs add random horizontal flips, a cosine learning rate schedule and mixed precision, applied identically to both objectives so that each backbone pair remains a controlled comparison.")

    # Page 4
    H("Identity training and similarity selection", page=True)
    T(["Setting", "Compact CNN, both objectives", "ResNet-18, all three objectives"], [
        ["Initialisation and seed", "From scratch / 42", "ImageNet, stem frozen / 42"],
        ["Epochs and batch size", "10 full epochs / 256", "10 full epochs / 256"],
        ["Optimizer and rate", "Adam, 0.001 fixed", "Adam, 0.0003 cosine to zero"],
        ["Augmentation and precision", "None / FP32", "Horizontal flip 0.5 / mixed FP16"],
        ["Epoch selection", "Classification: validation loss; triplet: training loss", "Same"],
        ["Selected epoch (cls / triplet / ArcFace)", f"{cc['epoch']} / {ct['epoch']}", f"{rc['epoch']} / {rt['epoch']} / {ra['epoch']}"]
    ], [1.85, 2.5, 2.5], 3, "Identity experiment configuration")
    P("All runs use a GeForce RTX 3060 Laptop GPU with 6 GiB memory, PyTorch 2.5.1 with CUDA 12.4 and a checksum verified uint8 cache of the source ZIP. Every epoch contains 1,487 batches with no cap. Triplets process three images per anchor, so the objectives are not matched for total computation, and each configuration was run once.")
    P("We compare cosine similarity and Euclidean distance on unit embeddings. Cosine needs a dot product and has a bounded scale; Euclidean distance offers an intuitive geometric distance consistent with embedding learning [3]. Their relationship means they rank unit embedding pairs identically, apart from numerical ties:")
    report.equation("similarity_identity.png", "For unit embeddings u and v, cosine is their dot product and squared Euclidean distance equals two minus twice cosine.")
    P("ROC treats label 1 as the same identity, using cosine or negative Euclidean distance so larger scores mean a stronger match. AUC is computed on all original verification validation pairs. For the application we choose, for each model, the highest TPR among finite cosine thresholds with validation pair FPR at most 1%. We deploy the model with the best TPR at that point; the three ResNet AUCs differ by less than 0.001 while their low FPR recall differs strongly. This selection is in sample. Classification test images are evaluated once after epoch selection and are not used to choose a checkpoint.")

    # Page 5
    H("Liveness model and evaluation protocol", page=True)
    P("LCC FASD targets replay attacks across recording and display devices [4]; we use the Kaggle distribution [5]. We preserve the development split, remove 202 training images with 12 filename derived subject codes overlapping development and four duplicated in validation, and exclude 58 evaluation byte duplicates. The resulting training, validation and evaluation sizes are 8,093 (1,193 genuine), 2,948 (405) and 7,522 (256). Evaluation filenames are anonymous, so subject separation cannot be verified there.")
    P("Our webcam set was recorded with the application's own detector and quality checks and stores the 1.4 times square crop that the deployed model receives; a wider crop keeps screen edges in view [9]. Eight genuine sessions vary glasses and room lighting; attack sessions show a still photo and a recorded video of the same person on a phone. Whole sessions are split: for each presentation type the latest session is test and the previous one validation. Two dark close-range attack sessions were excluded after visual review showed the operator's real face rather than the screen, so no dark attack remains in training.")
    T(["Training choice", "Scratch baseline", "MobileNetV3, LCC", "MobileNetV3, LCC + webcam (app)"], [
        ["Network", "LivenessCNN, random", "MobileNetV3-Large, ImageNet", "Same, 4 blocks frozen"],
        ["Input", "112, supplied crop", "224, 1.4 times face crop", "224, 1.4 times face crop"],
        ["Optimizer", "Adam 0.001", "AdamW 0.0003, cosine", "AdamW 0.0003, cosine"],
        ["Epochs / batch", "10 / 32", "10 / 64", "10 / 64"],
        ["Sampling", "Shuffled", "Class balanced", "Class balanced, webcam x8"],
        ["Augmentation", "None", "Flip, light, upper-face patch", "Flip, light, upper-face patch"],
        ["Selected epoch", "1, validation loss", "6, validation ACER", "4, validation ACER"],
        ["Validation threshold", f"{sc['threshold']:.4f}", f"{lc['threshold']:.4f}", f"{ap['threshold']:.4f}"]
    ], [1.4, 1.6, 1.8, 2.05], 4, "Liveness configurations")
    P("MobileNetV3-Large [11] replaced the from-scratch network because the scratch models accepted phone presentations in camera trials. Thresholds maximise validation Youden J; no test or camera test data choose them. The app model's validation set adds the three webcam validation sessions to LCC FASD validation. APCER measures spoof acceptance, BPCER genuine rejection and ACER their mean; AUC measures ranking.")

    # Page 6
    H("Facial expression model and training", page=True)
    P("We use the FER2013 distribution [7], associated with the facial expression challenge described by Goodfellow et al. [6]. Filename prefixes recover Training, PublicTest and PrivateTest roles. All 35,887 images were decoded. Within each split, identical decoded pixels with conflicting labels are removed, while consistent duplicates retain one copy. Training copies of validation pixels and test copies of retained development pixels are then excluded, leaving 27,170 training, 3,562 validation and 3,271 test images.")
    P("The procedure verifies path and decoded pixel separation, not subject independence. The seven fixed labels are angry, disgust, fear, happy, sad, surprise and neutral, and inputs are grayscale 48 by 48 tensors normalised to [-1,1]. The baseline EmotionCNN is trained from scratch. The deployed model wraps ImageNet MobileNetV3-Large [11]: inside the network the face is resized to 112 by 112, repeated to three channels and ImageNet normalised, so data loading and application preprocessing are unchanged.")
    T(["Setting", "EmotionCNN baseline", "MobileNetV3 (app)"], [
        ["Initialisation", "Random", "ImageNet, first 4 blocks frozen"],
        ["Network input", "1 by 48 by 48", "48 resized to 112, three channels"],
        ["Feature extractor", "Three double-convolution blocks, 16 to 64 channels", "MobileNetV3-Large features"],
        ["Head", "Linear 576 to 128, dropout 0.40, Linear 128 to 7", "Linear 960 to 1280, dropout 0.2, Linear 1280 to 7"],
        ["Parameters", "0.15 million", "4.21 million"],
        ["Learning rate / device", "0.001 / CPU, four threads", "0.0003 / RTX 3060 Laptop"],
        ["Selected epoch", "15", "11"]
    ], [1.5, 2.6, 2.75], 5, "Expression model configurations")
    P("Both runs use weighted cross entropy with class weights proportional to the square root of inverse training frequency, normalised to mean one; disgust receives 2.5193 versus 0.5737 for happy. Validation uses unweighted cross entropy. Adam uses weight decay 0.0001, batch 64 and seed 42 for 15 epochs. Training augmentation is horizontal flipping with probability 0.5 and integer translations up to two pixels; validation and test are not augmented.")
    P("A ReduceLROnPlateau scheduler monitors validation macro F1 with factor 0.5 and patience 3; it never reduced the learning rate in either run. The checkpoint with maximum validation macro F1 is selected, breaking ties by lower validation loss, and PrivateTest is evaluated only after selection. Macro F1 averages per class F1 and balanced accuracy averages per class recall, so the largest class cannot dominate.")

    # Page 7
    H("Application and manual review", page=True)
    P(f"YuNet [8] (OpenCV Zoo 2023mar, score 0.6) replaced two Haar cascades. Its tighter boxes become squares of 0.91 times the longer side, the median Haar to YuNet ratio on 751 matched LCC FASD frames, so the networks keep their earlier crop geometry. On 300 genuine frames rotated by 35 degrees, Haar found exactly one face in {detect['rotated_35']['haar']} and YuNet in {detect['rotated_35']['yunet']}. Boxes lying mostly inside another are merged. Each scored crop must be at least 80 by 80 with grayscale mean 20 to 235 and Laplacian variance at least 20. YuNet is pretrained preprocessing, not one of the trained networks.")
    P(f"Guided registration captures ten poses whenever one face is visible; each shot is liveness checked on its own frame and only passing shots are saved, at least three. The same flow adds an appearance (with glasses, without glasses or with mask) to an existing profile; the operator confirms the person, because a new look can score closer to another profile than to its own. Profiles are tied to the encoder SHA256 and keep up to four appearance centroids. Matching takes the maximum cosine within each person, threshold {app['threshold']:.6f} and runner-up gap 0.05. Liveness is averaged over at most five frames within 0.85 seconds; three consecutive live, recognised frames record attendance once per person and session. The confirmed main face then keeps its name and liveness result while each new frame's embedding still matches the previous frame; the lock ends after two seconds out of view or when another face or registered person replaces it, and a frame-to-frame cosine of at least 0.72 is required. With several faces, each confirmed person keeps a lock of their own, so a stranger stepping closer does not take their name. Brief liveness or identity misses no longer drop a confirmed person, but a photo of that same person swapped in without a two second gap keeps the label, although it cannot add a record.")
    P("Profiles supports search, renaming, appearance removal and deletion of a person with their attendance, each after a database backup. Reference import saves 3 to 10 photos straight into a profile: it only locates the face and skips photos without one, with no liveness, quality, consistency or duplicate-identity checks, so the operator verifies recognition on the camera and deletes a wrong profile. Attendance still needs the live camera check. Adding an appearance preserves the person's ID and attendance history.")
    H("Human review with recorded provenance", level=3)
    P("For an uncertain live face, review requires an explicit profile choice, reviewer name, reason and confirmation; none is preselected. A ticket expires after 30 seconds and needs a live observation within 1.5 seconds, the same session and embedding continuity. Lost, multiple or poor quality faces, failed liveness or changed embeddings invalidate it. Attendance and audit metadata commit atomically, and manual decisions never update the encoder or templates.")
    P("The live view draws corner brackets, green for a recognised live main face or a registered bystander and yellow otherwise, with name tags, using analyses at most 0.5 seconds old. Analyses older than one second are discarded before writes. History export, camera tests and diagnostics support operation. Expression is display only; reviewer names are freely entered, so production use would need authentication and protected audit storage.")

    # Page 8
    H("Results", level=1, page=True)
    H("Identity comparison and operating point")
    T(["Encoder and objective", "Selected epoch", "Cosine AUC", "TPR at FPR at most 1%"], [
        [name, op["epoch"], f"{op['auc']:.6f}", f"{op['tpr']:.2%}"] for name, op in ops.items()
    ], [2.25, 1.3, 1.5, 1.8], 6, "Verification on all 8,805 original validation pairs")
    report.picture("identity_roc.png", "Figure 2  Identity ROC curves and the application's low FPR operating point", f"ResNet-18 encoders reach AUC {rc['auc']:.3f}, {rt['auc']:.3f} and {ra['auc']:.3f} (ArcFace), compact encoders {cc['auc']:.3f} and {ct['auc']:.3f}; the selected ArcFace point has TPR {ra['tpr']:.2%} at FPR {ra['fpr']:.2%}.")
    P(f"ImageNet initialisation, the deeper encoder and flip augmentation raise AUC by {rc['auc']-cc['auc']:.3f} for classification and {rt['auc']-ct['auc']:.3f} for triplet. Negative Euclidean AUC equals cosine AUC for every model. The ArcFace margin adds only {ra['auc']-rc['auc']:.4f} AUC but helps most where the app operates: at pair FPR {ra['fpr']:.2%} it accepts {accepted(ra, 4315):,} of 4,315 same identity pairs, against {accepted(rc, 4315):,} for ResNet-18 classification, {accepted(rt, 4315)} for ResNet-18 triplet and {accepted(ct, 4315)} for the earliest compact triplet application model. The application therefore uses ArcFace with cosine threshold {ra['threshold']:.6f}. Its head reaches {arc_test:.2%} closed set accuracy on 8,000 images of training identities, against 60.00% for ResNet-18 classification and 39.46% for the compact model.")
    P(f"A gallery search test enrols {gapp['known_probes']} identities and probes each with a different image. At each model's pair threshold the ArcFace encoder finds {gapp['known_correct']}/{gapp['known_probes']} known probes and misidentifies {gapp['unknown_false_identifications']}/{gapp['unknown_probes']} outsiders, against {gold['known_correct']}/{gold['known_probes']} and {gold['unknown_false_identifications']}/{gold['unknown_probes']} for ResNet-18 classification. A threshold calibrated on separate identities to limit outsider matches to 1% accepts {gcal['known_correct']}/{gcal['known_probes']} known and {gcal['unknown_false_identifications']}/{gcal['unknown_probes']} outsider probes, against {gold_cal['known_correct']}/{gold_cal['known_probes']} and {gold_cal['unknown_false_identifications']}/{gold_cal['unknown_probes']}. With 20 known probes these differences are within a few people, and all identities appeared in encoder training.")

    # Page 9
    H("Liveness and expression results", page=True)
    T(["LCC FASD measure", "Scratch baseline", "MobileNetV3, LCC", "MobileNetV3 + webcam (app)"], [
        ["AUC", f"{sc['AUC']:.4f}", f"{lc['AUC']:.4f}", f"{ap['AUC']:.4f}"],
        ["APCER", f"{sc['APCER']:.2%}", f"{lc['APCER']:.2%}", f"{ap['APCER']:.2%}"],
        ["BPCER", f"{sc['BPCER']:.2%}", f"{lc['BPCER']:.2%}", f"{ap['BPCER']:.2%}"],
        ["ACER", f"{sc['ACER']:.2%}", f"{lc['ACER']:.2%}", f"{ap['ACER']:.2%}"]
    ], [1.75, 1.6, 1.6, 1.9], 7, "Liveness on the same 7,522 LCC FASD evaluation images")
    P(f"Fine-tuning MobileNetV3 reduces LCC FASD ACER from {sc['ACER']:.2%} to {lc['ACER']:.2%}. Adding webcam data raises it to {ap['ACER']:.2%}, mainly through genuine rejection, which grows from {lc['BPCER']:.2%} to {ap['BPCER']:.2%} of the 256 genuine images; spoof acceptance is almost unchanged. The model has become specialised to our camera, a trade accepted because the camera result below decides the classroom deployment.")
    T(["Expression measure", "EmotionCNN baseline", "MobileNetV3 (app)"], [
        ["Accuracy", f"{eo['accuracy']:.2%}", f"{en['accuracy']:.2%}"],
        ["Macro F1", f"{eo['macro_f1']:.4f}", f"{en['macro_f1']:.4f}"],
        ["Balanced accuracy", f"{eo['balanced_accuracy']:.2%}", f"{en['balanced_accuracy']:.2%}"]
    ], [2.45, 2.2, 2.2], 8, "Expression evaluation on filtered PrivateTest after checkpoint selection")
    report.picture("expression_f1.png", "Figure 3  Expression F1 per class for both models, with test support", "MobileNetV3 improves every class F1; disgust rises from 0.20 to 0.66 and fear from 0.26 to 0.46.")
    old_f1 = {r["label"]: r["f1"] for r in eo["per_class"]}
    new_f1 = {r["label"]: r["f1"] for r in en["per_class"]}
    P(f"Every class improves. The weakest classes gain most: disgust, with only 38 test images, rises from F1 {old_f1['disgust']:.3f} to {new_f1['disgust']:.3f} and fear from {old_f1['fear']:.3f} to {new_f1['fear']:.3f}, while happy remains strongest at {new_f1['happy']:.3f}. Validation macro F1 at the selected epoch is 0.632, close to the test value. The filtered protocol limits comparison with published FER2013 benchmarks.")

    # Page 10
    H("Integration and webcam evidence", page=True)
    P(f"The software suite passes {software['passed']} tests in {software['seconds']:.1f} seconds. It covers training fixtures, recognition gates, multi-face scoring, guided registration, appearance and profile management, history export, device recovery, rendering and review. These functional checks do not estimate recognition accuracy.")
    P("During live testing the LCC trained MobileNetV3 scored a phone photo of a registered student in a dark room at 0.85 against its threshold of 0.041, and attendance was recorded. This motivated the webcam recordings. Table 9 compares that model with the deployed one on whole held-out sessions; validation rows were used by the deployed model for epoch and threshold selection and are therefore in sample.")
    T(["Webcam session", "Previous app model", "Deployed model"], [
        ["Test: genuine, dark room, no glasses (passed)", "{}/{}".format(*session(old_on, "live-person")), "{}/{}".format(*session(new_on, "live-person"))],
        ["Test: phone photo, bright, tilted (accepted)", "{}/{}".format(*session(old_on, "phone-still-photo")), "{}/{}".format(*session(new_on, "phone-still-photo"))],
        ["Test: phone video, bright (accepted)", "{}/{}".format(*session(old_on, "recorded-video")), "{}/{}".format(*session(new_on, "recorded-video"))],
        ["Validation: phone photo, dark (accepted)", "{}/{}".format(*session(old_val, "phone-still-photo")), "{}/{}".format(*session(new_val, "phone-still-photo"))],
        ["Validation: phone video, dark (accepted)", "{}/{}".format(*session(old_val, "recorded-video")), "{}/{}".format(*session(new_val, "recorded-video"))]
    ], [3.45, 1.7, 1.7], 9, "Liveness on recorded webcam sessions; crops passed or accepted as live")
    P("The deployed model blocks every held-out attack crop while passing every genuine crop, and the dark attack that succeeded live is blocked in validation. The evidence is narrow: one person, room and day, the same photo and video content in training and test sessions, and correlated frames within a session. A final live check of the application with a phone photo and a phone video accepted neither (observed, not counted). Printed photos, other phones and other people remain untested, so this is not an attack population error rate. In a later live check in a brighter office, the deployed model also rejected the registered user's genuine face with scores near zero, so the webcam data overfits its single room; the application can be started with the LCC-only model in such rooms, which restores genuine access but not phone attack protection.")
    P(f"Median single call latency on the RTX 3060 laptop is {latency['yunet_detect_960x720_cpu']:.0f} ms for YuNet on a 960 by 720 frame (CPU), {latency['liveness']:.0f} ms for liveness, {latency['identity_embed']:.0f} ms for identity and {latency['expression']:.0f} ms for expression. Each additional face only adds identity matching. Capture, drawing and streaming are excluded, so these are not end to end frame rates.")

    # Page 11
    H("Discussion", level=1, page=True)
    H("Model selection and deployment limits")
    P(f"The evidence supports ImageNet fine-tuning for all three tasks and ResNet-18 ArcFace as the deployed identity encoder. It does not support a deployment accuracy claim: at the selected point about {1-ra['tpr']:.0%} of same identity pairs are still rejected, and the gallery test finds {gapp['known_correct']} of {gapp['known_probes']} known people. Appearance templates and manual review compensate in operation. A trial on 23 November showed that compressed phone photos make weak templates: within each photo-imported profile the photos agreed at a mean cosine of only 0.50 to 0.57, against 0.94 for a camera registration, and five of six pairs of different people scored 0.52 to 0.68, above the threshold. Camera registration is therefore preferred; photo import is a fallback whose result the operator tests on the camera. Hard negative mining and larger input resolution remain untested, and ArcFace's scale and margin were not tuned.")
    P("Each comparison controls data, input resolution and seed within a backbone, but it does not equalise compute and uses one run per configuration, so no uncertainty intervals are available. Verification validation is used for model choice and threshold calibration and contains exact training pixel duplicates. An explicitly separate cleaned protocol and an independent gallery test would strengthen the identity results without silently filtering the assignment's pair list.")
    P("The liveness history shows why camera evidence matters. Fine-tuning made LCC FASD results excellent, yet the model accepted a phone photo on our webcam. A small set of in-domain recordings fixed that case at the cost of more genuine rejections on LCC FASD. Broader recordings, with several people, devices, printed photos and dark attacks kept in training, are the next step, followed by a fresh session-level test.")
    P("Expression classification improved most for the weakest classes, but fear and sad still have F1 near 0.5. The model has not been validated for internal feelings, mental health, work attitude or student engagement. Keeping expression outside the attendance rule avoids an unsupported condition for recording presence.")
    H("Practical improvements and compliance", level=3)
    P("Only the main face is liveness checked and recorded; other faces receive name tags without attendance, so a registered person's photo held in the background cannot be recorded. Manual review makes uncertain decisions explicit and traceable, but its free text reviewer name is not authentication. Embeddings remain biometric data even though camera images are not saved, and the default database lives in a synchronised OneDrive folder. YuNet is pretrained localisation, separate from the trained networks; ImageNet initialised networks are fine-tuned on the task data, consistent with the brief's restriction on pretrained models.")

    # Page 12 -- cap includes references and all front matter.
    H("Conclusion", level=1, page=True)
    P("The project trains both required identity approaches plus ArcFace, a liveness network and an expression network, and integrates them with guided registration, appearance templates, manual review and multi-face display. Fine-tuned ResNet-18 and MobileNetV3 models clearly outperform the from-scratch baselines on held-out data, and in-domain webcam recordings closed a phone photo attack seen in live use. Pixel overlap in identity validation, limited low FPR recall and single-room camera evidence still prevent deployment claims.")
    H("References")
    report.reference(1, "COS30082 Applied Machine Learning. Facial Recognition with Emotion and Liveness. Project brief supplied with the assignment, pp. 1 to 6.")
    report.reference(2, "Kaggle. 11 785 Fall 20 Homework 2 Part 2. Identity dataset and verification pairs. Accessed 27 October 2026.", "https://www.kaggle.com/competitions/11-785-fall-20-homework-2-part-2/data")
    report.reference(3, "Schroff, F., Kalenichenko, D. and Philbin, J. (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. CVPR. arXiv 1503.03832.", "https://arxiv.org/abs/1503.03832")
    report.reference(4, "Timoshenko, D. et al. (2019). Large Crowdcollected Facial Anti-Spoofing Dataset. CSIT.", "https://pureportal.spbu.ru/files/51584200/CSIT2019_Grishkin_Timoshenko.pdf")
    report.reference(5, "Kaggle contributor faber24. LCC FASD dataset distribution, version 1. Accessed 27 October 2026.", "https://www.kaggle.com/datasets/faber24/lcc-fasd")
    report.reference(6, "Goodfellow, I. J. et al. (2013). Challenges in Representation Learning: A report on three machine learning contests. arXiv 1307.0414.", "https://arxiv.org/abs/1307.0414")
    report.reference(7, "Kaggle contributor msambare. FER2013 dataset distribution, version 1. Accessed 27 October 2026.", "https://www.kaggle.com/datasets/msambare/fer2013")
    report.reference(8, "Wu, W., Peng, H. and Yu, S. (2023). YuNet: A Tiny Millisecond-level Face Detector. Machine Intelligence Research. Model from OpenCV Zoo, accessed 20 November 2026.", "https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet")
    report.reference(9, "FaceGuard AI. Anti-spoofing input crop design. GitHub commit a2266a9. Accessed 14 November 2026.", "https://github.com/henryhoang207/Facial-Recognition---COS30082/blob/a2266a909adbffc3657d7e1275c30db83e650d1e/README.md#L106-L120")
    report.reference(10, "He, K., Zhang, X., Ren, S. and Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR. arXiv 1512.03385.", "https://arxiv.org/abs/1512.03385")
    report.reference(11, "Howard, A. et al. (2019). Searching for MobileNetV3. ICCV. arXiv 1905.02244.", "https://arxiv.org/abs/1905.02244")
    report.reference(12, "Russakovsky, O. et al. (2015). ImageNet Large Scale Visual Recognition Challenge. IJCV. arXiv 1409.0575.", "https://arxiv.org/abs/1409.0575")
    report.reference(13, "Deng, J., Guo, J., Xue, N. and Zafeiriou, S. (2019). ArcFace: Additive Angular Margin Loss for Deep Face Recognition. CVPR. arXiv 1801.07698.", "https://arxiv.org/abs/1801.07698")
    H("Reproduction evidence", level=3)
    P("The results directory holds every evaluation cited here: comparison_identity_20261121, arcface_resnet18, the identity_open_set folders, the liveness_mobilenetv3 and liveness_mobilenetv3_webcam folders with their test reports, emotion_fer_mnv3_test, app_update_20261120 and the test log in readiness_20261121. The README links the reproduction commands.", size=10)

    OUTPUT.parent.mkdir(exist_ok=True)
    report.doc.save(OUTPUT)
    sources = [COMPARISON, GALLERY_NEW, GALLERY_OLD, PAD_SCRATCH, f"{PAD_LCC}_test/metrics.json",
               f"{PAD_APP}_lcc_test/metrics.json", f"{PAD_APP}_on_test/metrics.json", f"{PAD_LCC}_on_test/metrics.json",
               EXPRESSION_OLD, EXPRESSION_NEW, PIPELINE, TESTS_LOG, "results/identity_app_config.json",
               "results/liveness_app_config.json", "attendance_app/gui.py", "attendance_app/camera.py",
               "attendance_app/service.py", "attendance_app/web.py"]
    metadata = {"output": str(OUTPUT), "language": "English", "group_details": "Group HN02 with six supplied members",
                "page_limit": 12, "intended_pages": 12, "operating_points": operating,
                "source_files": {p: digest(ROOT/p) for p in sources if (ROOT/p).is_file()}}
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK/"build_evidence.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    build()
