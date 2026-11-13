from pathlib import Path
import re
import subprocess

import modal

APP_NAME = "ganstego-train-coco"
DATASET_VOLUME = "stego-volume"          
CHECKPOINT_VOLUME = "ganstego-checkpoints"
COCO_MOUNT = "/root/coco/train2017"
CHECKPOINT_MOUNT = "/root/steganography_model"
MINIMUM_IMAGES = 10_500                  # 10,000 train + 500 validation
TOTAL_EPOCHS = 110

app = modal.App(APP_NAME)
dataset_volume = modal.Volume.from_name(DATASET_VOLUME)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)

runtime_image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime")
    .entrypoint([])
    .pip_install(
        "lpips==0.1.4",
        "torchmetrics==1.7.3",
        "reedsolo==1.7.0",
        "tqdm==4.67.1",
        "pillow==11.2.1",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir("ganstego", "/root/ganstego", copy=True)
    .add_local_file("train.py", "/root/train.py", copy=True)
)


def _checkpoint_epoch(path: Path) -> int:
    match = re.search(r"model_latest_(\d+)\.pth$", path.name)
    return int(match.group(1)) if match else -1


def _latest_checkpoint(root: Path):
    checkpoints = list(root.glob("model_latest_*.pth"))
    return max(checkpoints, key=_checkpoint_epoch, default=None)


@app.function(
    image=runtime_image,
    gpu="A10",
    cpu=4,
    memory=32768,
    timeout=60 * 60 * 24,
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=30.0),
    volumes={
        "/root/coco": dataset_volume.with_mount_options(read_only=True, sub_path="/coco"),
        CHECKPOINT_MOUNT: checkpoint_volume,
    },
)
def train() -> None:
    dataset_path = Path(COCO_MOUNT)
    if not dataset_path.is_dir():
        raise RuntimeError("Images are missing at {}".format(dataset_path))

    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_count = sum(
        1 for path in dataset_path.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    print("Cover images available: {}".format(image_count))
    if image_count < MINIMUM_IMAGES:
        raise RuntimeError(
            "Need at least {} images, found {}".format(MINIMUM_IMAGES, image_count))

    command = [
        "python", "/root/train.py",
        "--data", COCO_MOUNT,
        "--save-path", CHECKPOINT_MOUNT,
        "--epochs", str(TOTAL_EPOCHS),
        "--batch-size", "2",
        "--seed", "42",
    ]

    checkpoint_path = _latest_checkpoint(Path(CHECKPOINT_MOUNT))
    if checkpoint_path is not None:
        command += ["--resume", str(checkpoint_path)]
        print("Resuming from {}".format(checkpoint_path))
    else:
        print("No checkpoint found - training from scratch")
    subprocess.run(command, cwd="/root", check=True)


@app.local_entrypoint()
def main() -> None:
    call = train.spawn()
