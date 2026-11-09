"""Transactional local profiles and one attendance record per person/session."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import unicodedata
import uuid

import numpy as np


def clean_text(value, field="Name"):
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text.")
    value = unicodedata.normalize("NFC", value).strip()
    if not 1 <= len(value) <= 80 or any(unicodedata.category(c).startswith("C") for c in value):
        raise ValueError(f"{field} must contain 1–80 printable characters.")
    return value


def unit_vector(value, dimension):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (dimension,) or not np.isfinite(array).all():
        raise ValueError("Invalid face embedding shape or values.")
    norm = float(np.linalg.norm(array))
    if norm < 1e-8:
        raise ValueError("Face embedding must have nonzero norm.")
    return array / norm


class ProfileStore:
    """Embeddings are tied to one exact checkpoint; raw camera images are not saved."""
    def __init__(self, path, encoder_sha256, dimension):
        if len(encoder_sha256) != 64 or any(c not in "0123456789abcdef" for c in encoder_sha256):
            raise ValueError("Invalid encoder fingerprint.")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("Invalid embedding dimension.")
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self.connection = sqlite3.connect(self.path, timeout=10)
        try:
            self.connection.execute("PRAGMA foreign_keys=ON")
            expected = {"schema": "1", "encoder_sha256": encoder_sha256, "dimension": str(dimension)}
            if self.connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'").fetchone():
                present = dict(self.connection.execute("SELECT key,value FROM metadata"))
                if present and present != expected:
                    raise ValueError("Profile database belongs to another encoder. Use a new database and register again.")
            with self.connection:
                self.connection.executescript("""
                    BEGIN;
                    CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS profiles (
                        id TEXT PRIMARY KEY, name TEXT NOT NULL, name_key TEXT NOT NULL UNIQUE,
                        created_utc TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS appearances (
                        id TEXT PRIMARY KEY,
                        person_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                        label TEXT NOT NULL, label_key TEXT NOT NULL, created_utc TEXT NOT NULL,
                        UNIQUE(person_id,label_key));
                    CREATE TABLE IF NOT EXISTS samples (
                        person_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                        embedding BLOB NOT NULL,
                        appearance_id TEXT REFERENCES appearances(id) ON DELETE CASCADE);
                    CREATE TABLE IF NOT EXISTS attendance (
                        session TEXT NOT NULL, person_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                        recorded_utc TEXT NOT NULL, cosine REAL NOT NULL, live_score REAL NOT NULL,
                        PRIMARY KEY(session, person_id));
                    CREATE TABLE IF NOT EXISTS attendance_audit (
                        session TEXT NOT NULL, person_id TEXT NOT NULL,
                        source TEXT NOT NULL CHECK(source IN ('automatic','manual_review')),
                        reviewer TEXT, reason TEXT, review_id TEXT UNIQUE,
                        observed_utc TEXT NOT NULL, reviewed_utc TEXT NOT NULL,
                        PRIMARY KEY(session, person_id),
                        FOREIGN KEY(session,person_id) REFERENCES attendance(session,person_id) ON DELETE CASCADE);
                """)
                sample_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(samples)")}
                if "appearance_id" not in sample_columns:
                    self.connection.execute(
                        "ALTER TABLE samples ADD COLUMN appearance_id TEXT REFERENCES appearances(id) ON DELETE CASCADE")
                expected = {"schema": "1", "encoder_sha256": encoder_sha256, "dimension": str(dimension)}
                present = dict(self.connection.execute("SELECT key,value FROM metadata"))
                if present and present != expected:
                    raise ValueError("Profile database belongs to another encoder. Use a new database and register again.")
                if not present:
                    self.connection.executemany("INSERT INTO metadata VALUES (?,?)", expected.items())
        except Exception:
            self.close()
            raise

    def close(self):
        self.connection.close()

    def profiles(self):
        rows = self.connection.execute("SELECT id,name,created_utc FROM profiles ORDER BY name_key").fetchall()
        result = []
        for person_id, name, created in rows:
            blobs = self.connection.execute(
                "SELECT embedding,appearance_id FROM samples WHERE person_id=? ORDER BY rowid", (person_id,)).fetchall()
            if not blobs:
                raise ValueError("Profile has no embeddings.")
            groups = {}
            for blob, appearance_id in blobs:
                groups.setdefault(appearance_id, []).append(unit_vector(np.frombuffer(blob, dtype="<f4"), self.dimension))
            appearances = []
            if None in groups:
                original = groups.pop(None)
                appearances.append({"id": "original:"+person_id, "label": "Original",
                                    "embedding": unit_vector(np.mean(original, axis=0), self.dimension),
                                    "sample_count": len(original)})
            variants = self.connection.execute(
                "SELECT id,label FROM appearances WHERE person_id=? ORDER BY created_utc,id", (person_id,)).fetchall()
            for appearance_id, label in variants:
                samples = groups.pop(appearance_id, [])
                if not samples:
                    raise ValueError("Appearance has no embeddings.")
                appearances.append({"id": appearance_id, "label": label,
                                    "embedding": unit_vector(np.mean(samples, axis=0), self.dimension),
                                    "sample_count": len(samples)})
            if groups:
                raise ValueError("Profile contains an invalid appearance group.")
            # Old consumers keep the original centroid. New matching explicitly
            # uses the separately stored appearance centroids instead of mixing
            # glasses/no-glasses samples into one diluted template.
            centroid = appearances[0]["embedding"]
            result.append({"id": person_id, "name": name, "created_utc": created,
                           "sample_count": len(blobs), "embedding": centroid, "appearances": appearances})
        return result

    def enroll(self, name, embeddings):
        name = clean_text(name)
        embeddings = [unit_vector(value, self.dimension) for value in embeddings]
        if len(embeddings) < 3:
            raise ValueError("Registration requires at least three live face samples.")
        unit_vector(np.mean(embeddings, axis=0), self.dimension)
        person_id = uuid.uuid4().hex
        try:
            with self.connection:
                self.connection.execute("INSERT INTO profiles VALUES (?,?,?,?)",
                                        (person_id, name, name.casefold(), datetime.now(timezone.utc).isoformat()))
                self.connection.executemany("INSERT INTO samples(person_id,embedding) VALUES (?,?)",
                    [(person_id, value.astype("<f4").tobytes()) for value in embeddings])
        except sqlite3.IntegrityError as exc:
            raise ValueError("This name is already registered. Use a name plus student ID.") from exc
        return person_id

    def add_appearance(self, person_id, label, embeddings):
        """Add a separate template without replacing the person's original samples.

        A person has at most four appearance groups, including the original
        NULL group from older databases. Labels are unique per person after
        Unicode normalization/case folding. No camera images are stored.
        """
        label = clean_text(label, "Appearance label")
        vectors = [unit_vector(value, self.dimension) for value in embeddings]
        if len(vectors) < 3:
            raise ValueError("An appearance requires at least three live face samples.")
        unit_vector(np.mean(vectors, axis=0), self.dimension)
        appearance_id = uuid.uuid4().hex
        with self.connection:
            # Reserve the writer before checking the cap and label so two
            # concurrent additions cannot both pass the four-template limit.
            if not self.connection.in_transaction:
                self.connection.execute("BEGIN IMMEDIATE")
            if not self.connection.execute("SELECT 1 FROM profiles WHERE id=?", (person_id,)).fetchone():
                raise ValueError("The selected profile no longer exists.")
            has_original = self.connection.execute(
                "SELECT 1 FROM samples WHERE person_id=? AND appearance_id IS NULL LIMIT 1", (person_id,)).fetchone() is not None
            if (has_original and label.casefold() == "original") or self.connection.execute(
                    "SELECT 1 FROM appearances WHERE person_id=? AND label_key=?", (person_id, label.casefold())).fetchone():
                raise ValueError("This appearance label is already used for this person.")
            count = self.connection.execute("SELECT COUNT(*) FROM appearances WHERE person_id=?", (person_id,)).fetchone()[0]
            if count + int(has_original) >= 4:
                raise ValueError("A person can have at most four appearances, including the original.")
            self.connection.execute("INSERT INTO appearances VALUES (?,?,?,?,?)",
                (appearance_id, person_id, label, label.casefold(), datetime.now(timezone.utc).isoformat()))
            self.connection.executemany("INSERT INTO samples(person_id,embedding,appearance_id) VALUES (?,?,?)",
                [(person_id, value.astype("<f4").tobytes(), appearance_id) for value in vectors])
        return appearance_id

    def _backup_before_change(self):
        """Back up committed data while the caller holds the writer reservation."""
        directory = self.path.parent/"backups"
        directory.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        destination = directory/f"profiles_{stamp}_{uuid.uuid4().hex[:8]}.sqlite3"
        source = sqlite3.connect(self.path.as_uri()+"?mode=ro", uri=True, timeout=10)
        target = sqlite3.connect(destination)
        try:
            source.backup(target)
        finally:
            source.close()
            target.close()
        return str(destination)

    def rename_profile(self, person_id, name):
        name = clean_text(name)
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            row = self.connection.execute("SELECT name FROM profiles WHERE id=?", (person_id,)).fetchone()
            if row is None:
                raise ValueError("The selected profile no longer exists.")
            if row[0] == name:
                return None
            if self.connection.execute("SELECT 1 FROM profiles WHERE name_key=? AND id<>?", (name.casefold(), person_id)).fetchone():
                raise ValueError("This name is already registered. Add a student ID.")
            backup = self._backup_before_change()
            self.connection.execute("UPDATE profiles SET name=?,name_key=? WHERE id=?", (name, name.casefold(), person_id))
        return backup

    def remove_appearance(self, person_id, appearance_id):
        """Remove one template group after backup; keep the profile and attendance."""
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            person = next((p for p in self.profiles() if p["id"] == person_id), None)
            if person is None or not any(a["id"] == appearance_id for a in person["appearances"]):
                raise ValueError("Select a valid appearance belonging to this profile.")
            if len(person["appearances"]) <= 1:
                raise ValueError("Keep at least one appearance for this person.")
            backup = self._backup_before_change()
            if appearance_id == "original:"+person_id:
                self.connection.execute("DELETE FROM samples WHERE person_id=? AND appearance_id IS NULL", (person_id,))
            else:
                self.connection.execute("DELETE FROM appearances WHERE id=? AND person_id=?", (appearance_id, person_id))
        return backup

    def record_attendance(self, session, person_id, cosine, live_score):
        return self._record(session, person_id, cosine, live_score, "automatic")

    def record_manual_attendance(self, session, person_id, cosine, live_score, *, reviewer, reason,
                                 review_id, observed_utc):
        reviewer, reason = clean_text(reviewer, "Reviewer"), clean_text(reason, "Reason")
        if not isinstance(review_id, str) or not 16 <= len(review_id) <= 100:
            raise ValueError("Invalid review ID.")
        try:
            observed = datetime.fromisoformat(observed_utc)
            if observed.tzinfo is None:
                raise ValueError("Observation timestamp needs a timezone.")
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid observation timestamp.") from exc
        return self._record(session, person_id, cosine, live_score, "manual_review",
                            reviewer, reason, review_id, observed.isoformat())

    def _record(self, session, person_id, cosine, live_score, source,
                reviewer=None, reason=None, review_id=None, observed_utc=None):
        session = clean_text(session, "Session")
        if not np.isfinite([cosine, live_score]).all() or not -1 <= cosine <= 1 or not 0 <= live_score <= 1:
            raise ValueError("Invalid attendance scores.")
        recorded_utc = datetime.now(timezone.utc).isoformat()
        with self.connection:
            cursor = self.connection.execute("INSERT OR IGNORE INTO attendance VALUES (?,?,?,?,?)",
                (session, person_id, recorded_utc, float(cosine), float(live_score)))
            if cursor.rowcount == 1:
                self.connection.execute("INSERT INTO attendance_audit VALUES (?,?,?,?,?,?,?,?)",
                    (session, person_id, source, reviewer, reason, review_id,
                     observed_utc or recorded_utc, recorded_utc))
        return cursor.rowcount == 1

    def attendance(self, session):
        return [{"person_id": person_id, "name": name, "recorded_utc": recorded, "cosine": cosine,
                 "live_score": live, "source": source or "legacy_unrecorded", "reviewer": reviewer,
                 "reason": reason, "observed_utc": observed, "reviewed_utc": reviewed}
                for person_id, name, recorded, cosine, live, source, reviewer, reason, observed, reviewed in self.connection.execute(
                    "SELECT p.id,p.name,a.recorded_utc,a.cosine,a.live_score,d.source,d.reviewer,d.reason,d.observed_utc,d.reviewed_utc "
                    "FROM attendance a JOIN profiles p ON p.id=a.person_id "
                    "LEFT JOIN attendance_audit d ON d.session=a.session AND d.person_id=a.person_id "
                    "WHERE a.session=? ORDER BY a.recorded_utc DESC", (session,))]
