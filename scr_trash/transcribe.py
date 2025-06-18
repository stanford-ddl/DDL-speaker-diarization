#!/usr/bin/env python3

from __future__ import annotations
import pathlib
import concurrent.futures as cf
import whisper
from typing import List, Optional

# Load Whisper model once (you can change "base" to "small", "medium", or "large")
WHISPER_MODEL = whisper.load_model("base")


def transcribe_file(path: pathlib.Path) -> Optional[str]:
    """Transcribe a single file using Whisper."""
    try:
        result = WHISPER_MODEL.transcribe(str(path))
        return result.get("text", "")
    except Exception as e:
        print(f"[ERROR] Failed to transcribe {path.name}: {e}")
        return None


def batch_transcribe(files: List[pathlib.Path], max_workers: int = 8) -> List[Optional[str]]:
    """Batch transcribe a list of audio files using Whisper."""
    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(transcribe_file, files))
