from __future__ import annotations
import os, sys, json, subprocess, time
from pathlib import Path
from pydub import AudioSegment
from scr_trash.transcribe import batch_transcribe
from visual_diarization import extract_frames, detect_faces, save_face_tracks
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") or sys.exit("Missing HF_TOKEN in .env")

# Diarization using 3D-Speaker
def run_diarization(wav: Path, out_dir: Path) -> Path:
    out_dir.mkdir(exist_ok=True)
    subprocess.run([
        "python", "speakerlab/bin/infer_diarization.py",
        "--wav", str(wav),
        "--out_dir", str(out_dir),
        "--out_type", "json",
        "--include_overlap",
        "--hf_access_token", HF_TOKEN,
    ], check=True)
    return next(out_dir.glob("*.json"))

# Load diarization segments from JSON
def load_segments(json_file: Path) -> list[dict]:
    with json_file.open() as f:
        return list(json.load(f).values())

# Slice audio into speaker segments
MIN_MS = 400

def slice_wavs(wav: Path, segments: list[dict], tgt: Path):
    audio = AudioSegment.from_wav(wav)
    files, kept = [], []
    for i, seg in enumerate(segments):
        s_ms = int(seg["start"] * 1000)
        e_ms = int(seg["stop"] * 1000)
        if e_ms - s_ms < MIN_MS:
            continue
        out = tgt / f"spk{seg['speaker']}_{i}.wav"
        audio[s_ms:e_ms].export(out, format="wav")
        files.append(out)
        kept.append(seg)
    return files, kept

# Merge audio diarization with visual face tracks using weighted confidence
def merge_audio_visual(audio_segments: list[dict], face_tracks: dict[int, list[dict]]) -> list[dict]:
    frame_rate = 1
    audio_confidence = 0.6
    visual_confidence = 0.4
    merged = []

    for seg in audio_segments:
        start_frame = int(seg["start"] * frame_rate)
        end_frame = int(seg["stop"] * frame_rate)
        face_count = defaultdict(int)

        for f in range(start_frame, end_frame + 1):
            for face in face_tracks.get(f, []):
                face_count[face["face_id"]] += 1

        if face_count:
            best_face = max(face_count, key=face_count.get)
            total_frames = sum(face_count.values())
            visual_score = face_count[best_face] / total_frames
            confidence = audio_confidence + visual_confidence * visual_score
            seg["visual_id"] = best_face
            seg["confidence"] = round(confidence, 3)
        else:
            seg["visual_id"] = "unknown"
            seg["confidence"] = audio_confidence

        merged.append(seg)
    return merged

# Main pipeline
def main(mp4_path: str):
    mp4 = Path(mp4_path).expanduser()
    if not mp4.exists():
        sys.exit(f"File not found: {mp4}")

    print("Starting full diarization pipeline")
    print(f"Input file: {mp4.name}\n")

    root = Path("data/segments") / mp4.stem
    root.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract WAV
    print("1) Extracting audio from MP4 …")
    wav = root / f"{mp4.stem}.wav"
    subprocess.run(["ffmpeg", "-i", str(mp4), "-ac", "1", "-ar", "16000", str(wav)], check=True)

    # Step 2: Audio diarization
    dia_cache = root / "dia_cache"
    print("2) Running audio diarization …"); t0 = time.time()
    diar_json = run_diarization(wav, dia_cache)
    base_segments = load_segments(diar_json)
    print(f"Found {len(base_segments)} speaker segments ({time.time()-t0:.1f}s)\n")

    # Step 3: Slice + Transcribe
    print("3) Slicing audio and transcribing …")
    seg_files, segments = slice_wavs(wav, base_segments, root)
    print(f"  Kept {len(seg_files)} slices (> {MIN_MS} ms)")

    print("Sending to AssemblyAI for transcription …")
    transcripts = batch_transcribe(seg_files, max_workers=12)

    for seg, txt, wav_f in zip(segments, transcripts, seg_files):
        seg["text"] = txt
        seg["wav"] = wav_f.name
        if txt:
            (wav_f.with_suffix(".txt")).write_text(txt, encoding="utf-8")

    # Step 4: Visual diarization
    print("4) Running visual diarization …")
    frames_dir = Path("data/frames") / mp4.stem
    vis_json = root / f"{mp4.stem}.faces.json"
    extract_frames(mp4, frames_dir)
    face_tracks = detect_faces(frames_dir, threshold=0.915)
    save_face_tracks(face_tracks, vis_json)

    # Final: Merge audio + visual
    merged = merge_audio_visual(segments, face_tracks)
    out_json = root / f"{mp4.stem}.merged.json"
    out_json.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Saved merged AV diarization to {out_json}\n")

    print("Pipeline complete! All results stored in:")
    print(f"{root.resolve()}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python diarization_pipeline.py /path/to/input.mp4")
    main(sys.argv[1])