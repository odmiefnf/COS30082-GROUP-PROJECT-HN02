"""Read-only attendance snapshots and explicit, spreadsheet-safe CSV export.

No model, image or embedding data is needed to browse recorded attendance.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
import sqlite3


LOCAL_TIMEZONE = timezone(timedelta(hours=7))
SOURCE_LABELS = {"automatic": "Automatic", "manual_review": "Manual", "legacy_unrecorded": "Unknown"}
CSV_COLUMNS = ("session", "person_id", "name", "recorded_local", "recorded_utc", "source", "source_code",
               "reviewer", "reason", "observed_utc", "reviewed_utc")


def read_attendance_snapshot(database):
    """Read a consistent snapshot without creating or migrating a database."""
    path = Path(database).resolve()
    if not path.is_file():
        raise FileNotFoundError("No attendance database yet. Record attendance first, then refresh.")
    connection = sqlite3.connect(path.as_uri()+"?mode=ro", uri=True, timeout=2.)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        has_audit = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='attendance_audit'").fetchone() is not None
        audit_fields = ("d.source,d.reviewer,d.reason,d.observed_utc,d.reviewed_utc" if has_audit else
                        "NULL AS source,NULL AS reviewer,NULL AS reason,NULL AS observed_utc,NULL AS reviewed_utc")
        audit_join = ("LEFT JOIN attendance_audit d ON d.session=a.session AND d.person_id=a.person_id "
                      if has_audit else "")
        query = ("SELECT a.session,a.person_id,p.name,a.recorded_utc,"+audit_fields+
                 " FROM attendance a JOIN profiles p ON p.id=a.person_id "+audit_join+
                 "ORDER BY a.recorded_utc DESC,a.session,p.name,a.person_id")
        rows = [dict(row) for row in connection.execute(query)]
        for row in rows:
            if row["source"] not in ("automatic", "manual_review"):
                row["source"] = "legacy_unrecorded"
        return rows
    finally:
        connection.close()


def filter_attendance(rows, session=None, source=None, search=""):
    needle = search.strip().casefold()
    return [dict(row) for row in rows
            if (session is None or row["session"] == session)
            and (source is None or row["source"] == source)
            and (not needle or needle in row["name"].casefold() or needle in row["session"].casefold())]


def summarize_attendance(rows):
    return {"sessions": len({row["session"] for row in rows}), "records": len(rows),
            **{code: sum(row["source"] == code for row in rows) for code in SOURCE_LABELS}}


def local_timestamp(value):
    """Return an explicit UTC+07:00 timestamp; preserve malformed legacy text."""
    if not value:
        return ""
    try:
        moment = datetime.fromisoformat(value)
        if moment.tzinfo is None:
            return value  # Do not invent a timezone for a legacy value.
        return moment.astimezone(LOCAL_TIMEZONE).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return str(value)


def spreadsheet_text(value):
    """Keep untrusted text literal when a CSV is opened in a spreadsheet."""
    text = "" if value is None else str(value)
    probe = text
    while probe and (probe[0].isspace() or probe[0] == "\ufeff"):
        probe = probe[1:]
    if text.startswith(("\t", "\r", "\n")) or probe.startswith(("=", "+", "-", "@")):
        return "'"+text
    return text


def export_attendance_csv(destination, rows, *, overwrite=False):
    """Export only the supplied snapshot; never read/write attendance records.

    Existing output files are protected unless the caller explicitly allows an
    overwrite, such as after the user's Save dialog confirmation.
    """
    destination = Path(destination)
    if destination.suffix.casefold() != ".csv":
        raise ValueError("Choose a destination with the .csv extension.")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    count = 0
    for row in rows:
        source = row.get("source", "legacy_unrecorded")
        record = {key: row.get(key) for key in CSV_COLUMNS}
        record.update(recorded_local=local_timestamp(row.get("recorded_utc")),
                      source=SOURCE_LABELS.get(source, "Unknown"),
                      source_code=source if source in SOURCE_LABELS else "legacy_unrecorded")
        writer.writerow({key: spreadsheet_text(record[key]) for key in CSV_COLUMNS})
        count += 1
    with destination.open("w" if overwrite else "x", encoding="utf-8-sig", newline="") as handle:
        handle.write(buffer.getvalue())
    return count
