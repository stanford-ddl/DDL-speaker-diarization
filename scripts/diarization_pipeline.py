import sys, os, shutil, subprocess, json
from pathlib import Path
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transcribe_with_rttm import (
    load_rttm,
    transcribe_full_audio,
    save_transcript
)

WHISPER_MODEL_NAME = "base"


def cleanup_previous(video: Path):
    stem = video.stem
    paths_to_remove = [
        f"exp_video/rttm/{stem}.rttm",
        f"exp_video/json/vad.json",
        f"exp_video/json/subseg.json",
        "results/diarization_result.json",
    ]
    for p in paths_to_remove:
        path = Path(p)
        if path.exists():
            path.unlink()


def run_3d_speaker_diarization(video_path: Path):
    result = subprocess.run([
        "bash",
        "egs/3dspeaker/speaker-diarization/run_video_local.sh",
        str(video_path)
    ])
    assert result.returncode == 0, "3D-Speaker diarization failed"


def match_multiple_speakers(transcribed_segments, rttm_segments):
    """
    Match each RTTM segment to Whisper segments it overlaps with.
    Return one output line per speaker segment (even if overlapping).
    """
    labeled = []
    for rttm in rttm_segments:
        spk = rttm['label']
        for seg in transcribed_segments:
            overlap = min(rttm['end'], seg['end']) - max(rttm['start'], seg['start'])
            if overlap > 0:
                labeled.append({
                    "start": seg['start'],
                    "end": seg['end'],
                    "text": seg['text'],
                    "speaker": spk
                })
    matched_times = {(l['start'], l['end']) for l in labeled}
    for seg in transcribed_segments:
        if (seg['start'], seg['end']) not in matched_times:
            labeled.append({
                "start": seg['start'],
                "end": seg['end'],
                "text": seg['text'],
                "speaker": "Unknown"
            })
    labeled.sort(key=lambda x: x['start'])
    return labeled


def main(video_path: str):
    video = Path(video_path)
    base_name = video.stem

    print("Cleaning up old outputs")
    cleanup_previous(video)

    print("Running 3D-Speaker diarization")
    run_3d_speaker_diarization(video)

    rttm_path = Path(f"exp_video/rttm/{base_name}.rttm")
    assert rttm_path.exists(), f"Expected RTTM at {rttm_path}"

    print("Transcribing full audio with Whisper")
    transcribed_segments = transcribe_full_audio(video, WHISPER_MODEL_NAME)

    print("Assigning overlapping speaker labels using RTTM segments")
    rttm_segments = load_rttm(rttm_path)
    labeled_segments = match_multiple_speakers(transcribed_segments, rttm_segments)

    output_file = Path("results/full_transcript.txt")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    save_transcript(output_file, labeled_segments)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diarization_pipeline.py path/to/video.mp4")
    else:
        main(sys.argv[1])