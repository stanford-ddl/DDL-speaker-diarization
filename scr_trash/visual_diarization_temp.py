import cv2
import face_recognition
import numpy as np
import json, time, argparse
from pathlib import Path
from collections import Counter

FPS = 1
DETECT_EVERY_N = 2
LOG_EVERY = 100

def extract_frames(video_path: Path, out_dir: Path):
    cap = cv2.VideoCapture(str(video_path))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    count = saved = 0
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🎞 Extracting frames …")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if int(count % (frame_rate // FPS)) == 0:
            out_file = out_dir / f"frame_{saved:05d}.jpg"
            cv2.imwrite(str(out_file), frame)
            saved += 1
            if saved % LOG_EVERY == 0:
                print(f"{saved} frames saved")
        count += 1
    cap.release()
    print(f"{saved} total frames saved to {out_dir}\n")

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def match_face_to_db(face_db, encoding, threshold=0.913):
    for fid, db_enc in face_db.items():
        if cosine_sim(encoding, db_enc) > threshold:
            return fid
    return None

def detect_faces(frames_dir: Path, threshold=0.915):
    face_db = {}
    face_tracks = {}
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    next_id = 0
    last_faces = []

    print(f"Detecting & tracking faces (threshold={threshold})")
    for i, img_path in enumerate(frame_paths):
        img = face_recognition.load_image_file(str(img_path))

        if i % DETECT_EVERY_N == 0:
            boxes = face_recognition.face_locations(img)
            encs = face_recognition.face_encodings(img, known_face_locations=boxes)
            frame_faces = []

            for box, enc in zip(boxes, encs):
                fid = match_face_to_db(face_db, enc, threshold)
                if fid is None:
                    fid = f"face_{next_id}"
                    face_db[fid] = enc
                    next_id += 1
                frame_faces.append({"face_id": fid, "bbox": box})

            last_faces = frame_faces
        else:
            frame_faces = last_faces

        face_tracks[i] = frame_faces
        if i % LOG_EVERY == 0 and i > 0:
            print(f"  Processed {i} frames")

    print(f"Tracked {len(face_db)} unique faces across {len(frame_paths)} frames\n")

    summary = Counter()
    for f in face_tracks.values():
        for face in f:
            summary[face["face_id"]] += 1
    for fid, count in sorted(summary.items()):
        print(f"   {fid}: {count} frames")

    return face_tracks

def save_face_tracks(face_tracks, out_file: Path):
    out_file.write_text(json.dumps(face_tracks, indent=2))
    print(f"Saved face tracks to {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Run visual diarization on a video file.")
    parser.add_argument("video_path", type=Path, help="Path to input MP4 file")
    parser.add_argument("frames_dir", type=Path, help="Directory to save extracted frames")
    parser.add_argument("output_json", type=Path, help="Path to output JSON file for face tracks")
    args = parser.parse_args()

    t0 = time.time()
    extract_frames(args.video_path, args.frames_dir)
    tracks = detect_faces(args.frames_dir, threshold=0.915)
    save_face_tracks(tracks, args.output_json)
    print(f"Done in {time.time() - t0:.1f} sec")

if __name__ == "__main__":
    main()