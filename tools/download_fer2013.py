"""Download and CRC-check the public msambare/fer2013 Kaggle v1 archive."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
import urllib.request
import zipfile
import zlib

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
ARCHIVE = DATA / "downloads/fer2013.zip"
DESTINATION = DATA / "FER2013"
URL = "https://www.kaggle.com/api/v1/datasets/download/msambare/fer2013?datasetVersionNumber=1"
ZIP_BYTES, IMAGE_BYTES, IMAGE_COUNT = 63_252_113, 56_510_189, 35_887

def main() -> int:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    status_path = DATA / "fer2013_download_status.json"
    started = time.monotonic()
    def report(state, **extra):
        value = {"dataset":"msambare/fer2013", "version":1, "status":state,
                 "dataset_ready":state == "ready", "source":URL, "archive":str(ARCHIVE),
                 "destination":str(DESTINATION), "process_id":os.getpid(),
                 "elapsed_seconds":round(time.monotonic()-started,1), **extra}
        status_path.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(value),flush=True)
    try:
        if shutil.disk_usage(DATA).free < 512 * 1024**2:
            raise OSError("Need at least 512 MiB free disk space.")
        report("downloading")
        if not ARCHIVE.exists():
            temporary = ARCHIVE.with_suffix(".zip.part")
            request = urllib.request.Request(URL, headers={"User-Agent":"FER2013 academic project downloader"})
            with urllib.request.urlopen(request,timeout=90) as response, temporary.open("wb") as target:
                if "zip" not in response.headers.get("Content-Type", "").lower():
                    raise ValueError("Kaggle did not return a ZIP.")
                shutil.copyfileobj(response,target,length=1024*1024)
            if temporary.stat().st_size != ZIP_BYTES:
                raise ValueError("Unexpected ZIP size for the verified dataset version.")
            temporary.replace(ARCHIVE)
        if ARCHIVE.stat().st_size != ZIP_BYTES:
            raise ValueError("Existing ZIP has an unexpected size; preserved.")
        digest=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
        with zipfile.ZipFile(ARCHIVE) as archive:
            files=[entry for entry in archive.infolist() if not entry.is_dir()]
            if len(files)!=IMAGE_COUNT or sum(entry.file_size for entry in files)!=IMAGE_BYTES:
                raise ValueError("ZIP inventory differs from the verified Kaggle v1 metadata.")
            seen=set()
            for index,entry in enumerate(files,1):
                path=PurePosixPath(entry.filename.replace("\\","/"))
                if (path.is_absolute() or ".." in path.parts or len(path.parts)!=3
                        or path.parts[0] not in ("train","test") or any(":" in part for part in path.parts)
                        or stat.S_ISLNK(entry.external_attr>>16)):
                    raise ValueError(f"Unsafe/unexpected ZIP path: {entry.filename}")
                target=(DESTINATION/Path(*path.parts)).resolve()
                if not target.is_relative_to(DESTINATION.resolve()):
                    raise ValueError("ZIP entry escapes FER2013 directory.")
                key=os.path.normcase(str(target))
                if key in seen:
                    raise ValueError("Duplicate ZIP destination.")
                seen.add(key)
                if target.exists():
                    if target.stat().st_size!=entry.file_size or zlib.crc32(target.read_bytes())&0xffffffff!=entry.CRC:
                        raise ValueError(f"Existing file differs from ZIP; preserved: {target}")
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    temporary=target.with_name(target.name+".extracting")
                    with archive.open(entry) as source,temporary.open("wb") as output:
                        shutil.copyfileobj(source,output)
                    temporary.replace(target)
                if index%5000==0:
                    report("extracting",images_verified=index,images_total=IMAGE_COUNT)
        report("ready",archive_sha256=digest,images_verified=IMAGE_COUNT,
               downloaded_bytes=ZIP_BYTES,extracted_bytes=IMAGE_BYTES)
        return 0
    except Exception as exc:
        report("failed",error=f"{type(exc).__name__}: {exc}")
        raise

if __name__ == "__main__":
    raise SystemExit(main())
