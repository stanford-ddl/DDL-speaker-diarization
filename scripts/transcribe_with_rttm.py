import whisper
import json
import argparse
from pathlib import Path


def load_rttm(rttm_path):
    segments = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                segments.append({
                    "start": float(parts[3]),
                    "end": float(parts[3]) + float(parts[4]),
                    "label": parts[7]
                })
    return segments


def find_best_matching_speaker(start, end, rttm_segments):
    """Return the first speaker whose RTTM segment overlaps with this range."""
    best_match = None
    best_overlap = 0
    for seg in rttm_segments:
        overlap = min(end, seg["end"]) - max(start, seg["start"])
        if overlap > 0 and overlap > best_overlap:
            best_match = seg["label"]
            best_overlap = overlap
    return best_match or "Unknown"


def transcribe_full_audio(audio_path, model_name="base"):
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path), verbose=True)
    return result['segments']


def assign_speakers_one_per_line(transcribed_segments, rttm_segments):
    speaker_labeled = []
    for seg in transcribed_segments:
        speaker = find_best_matching_speaker(seg['start'], seg['end'], rttm_segments)
        speaker_labeled.append({
            "start": seg['start'],
            "end": seg['end'],
            "text": seg['text'],
            "speaker": speaker
        })
    return speaker_labeled


def save_transcript(output_path, labeled_segments):
    with open(output_path, "w") as f:
        for seg in labeled_segments:
            f.write(f"[{seg['start']:.2f} - {seg['end']:.2f}] Speaker {seg['speaker']}: {seg['text']}\n")
    print(f"Speaker-labeled transcript saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to input audio or video file")
    parser.add_argument("--rttm", required=True, help="Path to RTTM file")
    parser.add_argument("--output", default="results/full_transcript.txt", help="Output path")
    parser.add_argument("--model", default="base", help="Whisper model size (base/medium/etc)")
    args = parser.parse_args()

    rttm_segments = load_rttm(args.rttm)
    transcribed_segments = transcribe_full_audio(args.audio, args.model)
    labeled_segments = assign_speakers_one_per_line(transcribed_segments, rttm_segments)
    save_transcript(args.output, labeled_segments)


if __name__ == "__main__":
    main()