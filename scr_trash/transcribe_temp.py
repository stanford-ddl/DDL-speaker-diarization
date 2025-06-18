#!/usr/bin/env python3

from __future__ import annotations
import os, time, pathlib, concurrent.futures as cf
import requests
from dotenv import load_dotenv

# Load credentials
load_dotenv()
AAI_KEY = os.getenv("ASSEMBLYAI_API_KEY")
if not AAI_KEY:
    raise RuntimeError("Missing ASSEMBLYAI_API_KEY in .env")

HEADERS = {"authorization": AAI_KEY}
SESSION = requests.Session()

# Upload helper
def _stream(fp, chunk=5 * 1024 * 1024):
    while True:
        buf = fp.read(chunk)
        if not buf:
            break
        yield buf

def _upload(path: pathlib.Path, max_retry=3) -> str:
    for attempt in range(1, max_retry + 1):
        try:
            with path.open("rb") as f:
                r = SESSION.post(
                    "https://api.assemblyai.com/v2/upload",
                    headers=HEADERS,
                    data=_stream(f),
                    timeout=300
                )
            r.raise_for_status()
            return r.json()["upload_url"]
        except Exception:
            if attempt == max_retry:
                raise
            time.sleep(2 * attempt)

def _start_job(upload_url: str) -> str:
    r = SESSION.post(
        "https://api.assemblyai.com/v2/transcript",
        headers=HEADERS,
        json={"audio_url": upload_url}
    )
    r.raise_for_status()
    return r.json()["id"]

def _poll_job(job_id: str) -> dict:
    r = SESSION.get(f"https://api.assemblyai.com/v2/transcript/{job_id}",
                    headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

# Public batch transcription
def batch_transcribe(files: list[pathlib.Path], max_workers=8) -> list[str | None]:
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        upload_urls = list(ex.map(_upload, files))

    job_ids = [_start_job(u) for u in upload_urls]
    finished: dict[str, str | None] = {}

    while len(finished) < len(job_ids):
        for jid in job_ids:
            if jid in finished:
                continue
            try:
                info = _poll_job(jid)
                if info["status"] == "completed":
                    finished[jid] = info.get("text", "")
                elif info["status"] == "failed":
                    finished[jid] = None
            except requests.exceptions.RequestException:
                pass
        time.sleep(4)

    return [finished[j] for j in job_ids]
