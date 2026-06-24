# Deepfake Preprocessing with SCRFD

This project extracts evenly distributed video frames, detects the largest face in each frame with SCRFD, crops the face with a 20% margin, and saves flat `.jpg` face crops for Deepfake Detection training.

## Install

```bash
pip install -r requirements.txt
```

On Kaggle, choose a GPU runtime before installing. On Windows, install a CUDA/cuDNN version compatible with your `onnxruntime-gpu` package.

## Download SCRFD ONNX

Download an SCRFD ONNX model such as `scrfd_10g_bnkps.onnx` from the InsightFace model zoo, then pass its local path with `--scrfd_model`.

Example path:

```text
models/scrfd_10g_bnkps.onnx
```

## FaceForensics++ Preprocess

Expected input:

```text
root_folder/
├── original/
├── Deepfakes/
├── Face2Face/
├── FaceSwap/
└── NeuralTextures/
```

Run:

```bash
python preprocess_ffpp_scrfd.py ^
  --input_root "D:/datasets/FF++" ^
  --output_root "D:/datasets/ffpp_faces_scrfd" ^
  --scrfd_model "models/scrfd_10g_bnkps.onnx" ^
  --img_size 224 ^
  --seed 42
```

Linux/Kaggle:

```bash
python preprocess_ffpp_scrfd.py \
  --input_root "/kaggle/input/ffpp" \
  --output_root "/kaggle/working/ffpp_faces_scrfd" \
  --scrfd_model "models/scrfd_10g_bnkps.onnx" \
  --img_size 224 \
  --seed 42
```

FF++ settings:

- `original`: real videos, 32 sampled frames per video.
- `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`: fake videos, 32 sampled frames per video.
- Videos are sorted inside each source folder and split before frame extraction: the first 720 videos go to train, the next 140 go to val, and the remaining videos go to test. This keeps matching FF++ videos aligned across `original`, `Deepfakes`, `Face2Face`, `FaceSwap`, and `NeuralTextures`.

## CelebDF-v2 Preprocess

Expected input:

```text
celeb_root_folder/
├── Celeb-real/
├── Celeb-synthesis/
└── YouTube-real/
```

Run:

```bash
python preprocess_celebdf_scrfd.py ^
  --input_root "D:/datasets/CelebDF-v2" ^
  --output_root "D:/datasets/celebdf_faces_scrfd" ^
  --scrfd_model "models/scrfd_10g_bnkps.onnx" ^
  --img_size 224 ^
  --seed 42
```

Linux/Kaggle:

```bash
python preprocess_celebdf_scrfd.py \
  --input_root "/kaggle/input/celebdf-v2" \
  --output_root "/kaggle/working/celebdf_faces_scrfd" \
  --scrfd_model "models/scrfd_10g_bnkps.onnx" \
  --img_size 224 \
  --seed 42
```

CelebDF-v2 settings:

- `Celeb-real`, `YouTube-real`: real videos, 128 sampled frames per video.
- `Celeb-synthesis`: fake videos, 21 sampled frames per video.

## Output Structure

For FaceForensics++, output is grouped by split first and then by original source folder:

```text
output_root/
|__ train/
|   |__ original/
|   |__ Deepfakes/
|   |__ Face2Face/
|   |__ FaceSwap/
|   |__ NeuralTextures/
|__ val/
|   |__ original/
|   |__ Deepfakes/
|   |__ Face2Face/
|   |__ FaceSwap/
|   |__ NeuralTextures/
|__ test/
    |__ original/
    |__ Deepfakes/
    |__ Face2Face/
    |__ FaceSwap/
    |__ NeuralTextures/
```

CelebDF-v2 keeps the previous frame-level split behavior:

```text
output_root/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

The output folders contain only face crop images, without per-video subfolders. File names include dataset name, split, source or label, source video id, and frame index.

Example:

```text
ffpp_train_Deepfakes_Deepfakes_001_002_sample000005_frame000128.jpg
```

At the end, each script prints:

- Number of real/fake videos.
- Number of saved real/fake face images in each frame-level split.
- Number of skipped frames where no face/readable crop was detected.
- Number of failed videos.
