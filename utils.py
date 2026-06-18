import argparse
import random
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm


VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}

SPLITS = ("train", "val", "test")
LABELS = ("real", "fake")


@dataclass(frozen=True)
class VideoItem:
    path: Path
    label: str
    source: str


@dataclass(frozen=True)
class ExtractedFrame:
    temp_path: Path
    label: str
    video_id: str
    frame_index: int


@dataclass
class Stats:
    video_counts: Dict[str, int] = field(default_factory=lambda: {"real": 0, "fake": 0})
    image_counts: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            split: {"real": 0, "fake": 0}
            for split in SPLITS
        }
    )
    skipped_no_face: int = 0
    failed_videos: int = 0


class SCRFDDetector:
    """Small wrapper around InsightFace SCRFD ONNX inference."""

    def __init__(
        self,
        model_path: str,
        det_size: Tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        prefer_gpu: bool = True,
    ) -> None:
        self.model_path = str(model_path)
        self.det_size = det_size
        self.det_thresh = det_thresh

        try:
            import onnxruntime as ort
            from insightface.model_zoo import get_model
        except ImportError as exc:
            raise RuntimeError(
                "Missing SCRFD dependencies. Install with: "
                "pip install insightface onnxruntime-gpu opencv-python tqdm numpy"
            ) from exc

        available = set(ort.get_available_providers())
        providers: List[str] = []
        if prefer_gpu and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        if providers[0] != "CUDAExecutionProvider":
            print("[WARN] CUDAExecutionProvider is unavailable. SCRFD will run on CPU.")

        model_file = Path(self.model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"SCRFD model not found: {model_file}")

        self.model = get_model(str(model_file), providers=providers)
        ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1
        self.model.prepare(ctx_id=ctx_id, input_size=self.det_size, det_thresh=self.det_thresh)
        print(f"[INFO] Loaded SCRFD model with provider: {providers[0]}")

    def detect(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return face boxes as Nx5 arrays: x1, y1, x2, y2, score."""
        bboxes, _ = self.model.detect(image_bgr, max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0:
            return np.empty((0, 5), dtype=np.float32)
        return np.asarray(bboxes, dtype=np.float32)

    def largest_face(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        bboxes = self.detect(image_bgr)
        if len(bboxes) == 0:
            return None

        widths = np.maximum(0.0, bboxes[:, 2] - bboxes[:, 0])
        heights = np.maximum(0.0, bboxes[:, 3] - bboxes[:, 1])
        areas = widths * heights
        return bboxes[int(np.argmax(areas))]


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input_root", required=True, type=Path, help="Root folder of the source videos")
    parser.add_argument("--output_root", required=True, type=Path, help="Folder to write cropped face images")
    parser.add_argument("--scrfd_model", required=True, type=Path, help="Path to SCRFD ONNX model")
    parser.add_argument("--img_size", default=380, type=int, help="Output face crop size in pixels")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for frame-level split")
    return parser


def find_videos(folder: Path) -> List[Path]:
    if not folder.exists():
        print(f"[WARN] Missing folder, skipped: {folder}")
        return []

    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def collect_videos(input_root: Path, real_dirs: Sequence[str], fake_dirs: Sequence[str]) -> List[VideoItem]:
    items: List[VideoItem] = []
    for dirname in real_dirs:
        for path in find_videos(input_root / dirname):
            items.append(VideoItem(path=path, label="real", source=dirname))

    for dirname in fake_dirs:
        for path in find_videos(input_root / dirname):
            items.append(VideoItem(path=path, label="fake", source=dirname))

    return items


def ensure_output_dirs(output_root: Path) -> None:
    for split in SPLITS:
        for label in LABELS:
            (output_root / split / label).mkdir(parents=True, exist_ok=True)


def evenly_spaced_frame_indices(total_frames: int, num_frames: int) -> List[int]:
    if total_frames <= 0 or num_frames <= 0:
        return []

    count = min(total_frames, num_frames)
    # Endpoint sampling spreads frames over the full video instead of taking a leading burst.
    indices = np.linspace(0, total_frames - 1, num=count, dtype=np.int64)
    return sorted(set(int(idx) for idx in indices))


def crop_with_margin(image_bgr: np.ndarray, bbox: np.ndarray, margin: float = 0.2) -> Optional[np.ndarray]:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox[:4].astype(float)
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    if box_w <= 1 or box_h <= 1:
        return None

    pad_x = box_w * margin
    pad_y = box_h * margin
    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(width, int(round(x2 + pad_x)))
    y2 = min(height, int(round(y2 + pad_y)))

    if x2 <= x1 or y2 <= y1:
        return None
    return image_bgr[y1:y2, x1:x2]


def safe_video_id(video_path: Path, input_root: Path) -> str:
    try:
        rel = video_path.relative_to(input_root)
    except ValueError:
        rel = video_path.name

    rel_no_suffix = str(Path(rel).with_suffix(""))
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", rel_no_suffix).strip("_")


def write_face(
    face_bgr: np.ndarray,
    output_path: Path,
    img_size: int,
) -> bool:
    resized = cv2.resize(face_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return bool(cv2.imwrite(str(output_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95]))


def final_frame_path(
    output_root: Path,
    dataset_name: str,
    split: str,
    label: str,
    video_id: str,
    frame_index: int,
) -> Path:
    filename = f"{dataset_name}_{split}_{label}_{video_id}_frame{frame_index:06d}.jpg"
    return output_root / split / label / filename


def process_video(
    item: VideoItem,
    input_root: Path,
    temp_dir: Path,
    num_frames: int,
    detector: SCRFDDetector,
    img_size: int,
    stats: Stats,
) -> List[ExtractedFrame]:
    extracted: List[ExtractedFrame] = []
    cap = cv2.VideoCapture(str(item.path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {item.path}")
        stats.failed_videos += 1
        return extracted

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = evenly_spaced_frame_indices(total_frames, num_frames)
    if not frame_indices:
        print(f"[WARN] Empty or unreadable video: {item.path}")
        stats.failed_videos += 1
        cap.release()
        return extracted

    video_id = safe_video_id(item.path, input_root)
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            stats.skipped_no_face += 1
            continue

        bbox = detector.largest_face(frame)
        if bbox is None:
            stats.skipped_no_face += 1
            continue

        face = crop_with_margin(frame, bbox, margin=0.2)
        if face is None:
            stats.skipped_no_face += 1
            continue

        temp_filename = f"{item.label}_{video_id}_frame{frame_index:06d}.jpg"
        temp_path = temp_dir / temp_filename
        if write_face(face, temp_path, img_size):
            extracted.append(
                ExtractedFrame(
                    temp_path=temp_path,
                    label=item.label,
                    video_id=video_id,
                    frame_index=frame_index,
                )
            )

    cap.release()
    return extracted


def split_extracted_frames(frames: Sequence[ExtractedFrame], seed: int) -> Dict[str, List[ExtractedFrame]]:
    rng = random.Random(seed)
    shuffled = sorted(frames, key=lambda frame: str(frame.temp_path))
    rng.shuffle(shuffled)

    n_frames = len(shuffled)
    n_train = int(n_frames * 0.8)
    n_val = int(n_frames * 0.1)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def move_frames_to_splits(
    frames_by_split: Dict[str, List[ExtractedFrame]],
    output_root: Path,
    dataset_name: str,
    stats: Stats,
) -> None:
    for split in SPLITS:
        for frame in tqdm(frames_by_split[split], desc=f"{dataset_name} move {split}", unit="frame"):
            output_path = final_frame_path(
                output_root=output_root,
                dataset_name=dataset_name,
                split=split,
                label=frame.label,
                video_id=frame.video_id,
                frame_index=frame.frame_index,
            )
            shutil.move(str(frame.temp_path), str(output_path))
            stats.image_counts[split][frame.label] += 1


def run_preprocess(
    dataset_name: str,
    input_root: Path,
    output_root: Path,
    scrfd_model: Path,
    img_size: int,
    seed: int,
    real_dirs: Sequence[str],
    fake_dirs: Sequence[str],
    frames_per_real_video: int,
    frames_per_fake_video: int,
) -> None:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if img_size <= 0:
        raise ValueError("--img_size must be a positive integer")

    ensure_output_dirs(output_root)

    items = collect_videos(input_root, real_dirs, fake_dirs)
    if not items:
        raise RuntimeError(f"No videos found under: {input_root}")

    stats = Stats()
    stats.video_counts["real"] = sum(1 for item in items if item.label == "real")
    stats.video_counts["fake"] = sum(1 for item in items if item.label == "fake")

    detector = SCRFDDetector(str(scrfd_model))
    temp_dir = Path(tempfile.mkdtemp(prefix=".frame_split_", dir=str(output_root)))
    extracted_frames: List[ExtractedFrame] = []

    try:
        for label in LABELS:
            label_items = sorted((item for item in items if item.label == label), key=lambda item: str(item.path))
            frames_per_video = frames_per_real_video if label == "real" else frames_per_fake_video
            desc = f"{dataset_name} extract {label}"
            for item in tqdm(label_items, desc=desc, unit="video"):
                extracted_frames.extend(
                    process_video(
                        item=item,
                        input_root=input_root,
                        temp_dir=temp_dir,
                        num_frames=frames_per_video,
                        detector=detector,
                        img_size=img_size,
                        stats=stats,
                    )
                )

        frame_splits = split_extracted_frames(extracted_frames, seed=seed)
        move_frames_to_splits(frame_splits, output_root, dataset_name, stats)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    print_summary(stats)


def print_summary(stats: Stats) -> None:
    print("\n========== Preprocess summary ==========")
    print(f"Videos real: {stats.video_counts['real']}")
    print(f"Videos fake: {stats.video_counts['fake']}")

    print("\nSaved face images after frame-level split:")
    for split in SPLITS:
        print(
            f"  {split}: real={stats.image_counts[split]['real']}, "
            f"fake={stats.image_counts[split]['fake']}"
        )

    print(f"\nSkipped frames without detected face/readable crop: {stats.skipped_no_face}")
    print(f"Failed videos: {stats.failed_videos}")
    print("========================================")
