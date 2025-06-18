from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import argparse
from pathlib import Path

def run_visual_diarization(video_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    diarizer = pipeline(
        task=Tasks.speaker_diarization,
        model='damo/speech_ner_spk-diarization_multimodal',
    )

    result = diarizer(audio=video_path)
    output_file = output_dir / "diarization_result.json"

    with output_file.open("w") as f:
        import json
        json.dump(result, f, indent=2)

    print(f"Saved visual diarization results to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=Path, help="Path to input MP4 video")
    parser.add_argument("output_dir", type=Path, help="Directory to save diarization output")
    args = parser.parse_args()

    run_visual_diarization(args.video_path, args.output_dir)
