import os
import tempfile

def process_video(video_path: str, max_frames: int = 5) -> dict:
    """Process video file — extracts frames and audio for Gemma 4 analysis."""
    try:
        import cv2
        import numpy as np
        return _process_with_cv2(video_path, max_frames)
    except ImportError:
        # cv2 not available on this deployment — return basic metadata
        return {
            "selected_frames": [],
            "flicker_score": 0,
            "motion_score": 0,
            "audio_transcript": "",
            "duration_seconds": 0,
            "frame_summary": "Video processing not available on this deployment. Submit text description for analysis.",
            "temp_dir": tempfile.mkdtemp()
        }
    except Exception as e:
        return {
            "selected_frames": [],
            "flicker_score": 0,
            "motion_score": 0,
            "audio_transcript": "",
            "duration_seconds": 0,
            "frame_summary": f"Video processing error: {str(e)}",
            "temp_dir": tempfile.mkdtemp()
        }

def _process_with_cv2(video_path: str, max_frames: int) -> dict:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    max_analysis = int(min(30, duration) * fps)

    frames, brightness_values = [], []
    temp_dir = tempfile.mkdtemp()
    sample_interval = max(1, int(fps))
    frame_idx = 0

    while cap.isOpened() and frame_idx < max_analysis:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        brightness_values.append(brightness)
        frames.append((frame_idx, brightness, frame.copy()))
        frame_idx += sample_interval

    cap.release()

    if not frames:
        return {"selected_frames": [], "flicker_score": 0, "motion_score": 0,
                "audio_transcript": "", "duration_seconds": 0,
                "frame_summary": "No frames extracted.", "temp_dir": temp_dir}

    brightness_array = np.array(brightness_values)
    flicker_score = min(1.0, float(np.std(brightness_array) / 30.0))

    motion_scores = []
    for i in range(1, len(frames)):
        diff = cv2.absdiff(
            cv2.cvtColor(frames[i-1][2], cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(frames[i][2], cv2.COLOR_BGR2GRAY)
        )
        motion_scores.append(float(np.mean(diff)))

    motion_score = min(1.0, float(np.mean(motion_scores)) / 20.0) if motion_scores else 0.0

    selected = {0, len(frames)-1,
                int(np.argmax(brightness_values)),
                int(np.argmin(brightness_values))}

    frame_paths = []
    for idx in sorted(selected)[:max_frames]:
        if idx < len(frames):
            path = os.path.join(temp_dir, f"frame_{idx:04d}.jpg")
            cv2.imwrite(path, frames[idx][2], [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_paths.append(path)

    parts = [f"Video: {duration:.1f}s."]
    if flicker_score > 0.3:
        parts.append(f"Significant flickering (variance {flicker_score:.2f}). Brightness {min(brightness_values):.0f}–{max(brightness_values):.0f}.")
    elif flicker_score > 0.1:
        parts.append(f"Mild brightness variation (variance {flicker_score:.2f}).")
    else:
        parts.append("Stable brightness throughout.")
    if motion_score > 0.4:
        parts.append(f"High motion ({motion_score:.2f}).")

    return {
        "selected_frames": frame_paths,
        "flicker_score": round(flicker_score, 3),
        "motion_score": round(motion_score, 3),
        "audio_transcript": "",
        "duration_seconds": round(duration, 1),
        "frame_summary": " ".join(parts),
        "temp_dir": temp_dir
    }

def cleanup_temp_files(video_result: dict):
    import shutil
    temp_dir = video_result.get("temp_dir")
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)