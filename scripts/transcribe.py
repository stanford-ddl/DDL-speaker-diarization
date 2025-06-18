import whisper
from typing import List
from pathlib import Path

def load_model(model_size: str = "base"):
    return whisper.load_model(model_size)

def transcribe_segments(files: List[Path], model) -> List[str]:
    results = []
    for file in files:
        try:
            result = model.transcribe(str(file))
            results.append(result.get("text", ""))
        except Exception as e:
            print(f"Error transcribing {file}: {e}")
            results.append("")
    return results
