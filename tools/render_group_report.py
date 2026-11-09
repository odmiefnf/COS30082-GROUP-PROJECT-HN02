"""Run the canonical DOCX renderer with a synchronous Windows LibreOffice launcher."""
from __future__ import annotations

import importlib.util
import argparse
import os
from pathlib import Path
import sys
import tempfile

from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SKILL = Path("C:/Users/kiet1/.codex/plugins/cache/openai-primary-runtime/documents/26.1105.12137/skills/documents")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"tmp/group_report/render_final")
    args = parser.parse_args()
    source = ROOT/"reports/COS30082_Group_Report.docx"
    destination = args.output_dir.resolve()
    temporary = ROOT/"tmp/group_report/lo_temp"
    temporary.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temporary)
    spec = importlib.util.spec_from_file_location("canonical_docx_renderer", SKILL/"render_docx.py")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    if sys.platform == "win32":
        office = ROOT/"tmp/report_tools/libreoffice/SourceDir/LibreOffice/program/soffice.com"
        if not office.is_file():
            raise FileNotFoundError("The local LibreOffice renderer helper is missing.")
        renderer._resolve_soffice = lambda: str(office)
    sys.argv = [str(SKILL/"render_docx.py"), str(source), "--output_dir", str(destination), "--emit_pdf", "--verbose"]
    try:
        renderer.main()
    except OSError as exc:
        # A Windows profile cleanup race does not invalidate complete page renders.
        if getattr(exc, "winerror", None) != 145:
            raise
        pdf = destination/(source.stem+".pdf")
        pages = list(destination.glob("page[0-9]*-*.png")) or list(destination.glob("page-*.png"))
        if not pdf.is_file() or pdf.stat().st_mtime < source.stat().st_mtime or len(pages) != len(PdfReader(pdf).pages):
            raise
        for path in pages:
            if path.stat().st_mtime < source.stat().st_mtime:
                raise RuntimeError("A page image predates the current report source.") from exc
            with Image.open(path) as image:
                image.verify()
            number = int(path.stem.rsplit("-", 1)[-1])
            if path.name != f"page-{number}.png":
                os.replace(path, destination/f"page-{number}.png")
        print("Pages fully rendered; temporary LibreOffice profile cleanup deferred.")


if __name__ == "__main__":
    main()
