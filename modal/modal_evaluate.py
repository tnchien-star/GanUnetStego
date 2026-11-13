import csv
import json
from pathlib import Path

import modal

APP_NAME = "ganstego-evaluate"
CHECKPOINT_PATH = "/root/checkpoints/ganstego_epoch96.pth"

DATASETS = {
    "coco": "eval_data/coco_val2017_first1000",
    "imagenet": "eval_data/imagenet_one_per_class_1000",
}

app = modal.App(APP_NAME)

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
    .add_local_file("checkpoints/ganstego_epoch96.pth", CHECKPOINT_PATH, copy=True)
)

for _dataset_id, _local_dir in DATASETS.items():
    if Path(_local_dir).is_dir():
        runtime_image = runtime_image.add_local_dir(
            _local_dir, "/root/eval/{}".format(_dataset_id), copy=True)


@app.function(image=runtime_image, gpu="A10", cpu=4, memory=32768, timeout=60 * 60 * 6)
def evaluate_dataset(dataset_id: str):
    import sys

    sys.path.insert(0, "/root")
    import torch
    from ganstego.checkpoint import load_models
    from ganstego.evaluation import evaluate_folder, format_table

    device = torch.device("cuda")
    encoder, decoder = load_models(CHECKPOINT_PATH, device=device, strict=True)
    rows = evaluate_folder(
        "/root/eval/{}".format(dataset_id),
        encoder,
        decoder,
        attacks="all",
        seed=42,
        device=device,
        dataset_id=dataset_id,
        checkpoint_name="ganstego_epoch96.pth",
    )
    print(format_table(rows))
    return rows


@app.local_entrypoint()
def main() -> None:
    calls = {dataset_id: evaluate_dataset.spawn(dataset_id) for dataset_id in DATASETS}
    for dataset_id, call in calls.items():
        rows = call.get()
        output_dir = Path("results_{}".format(dataset_id))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        with (output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("Saved {}".format(output_dir.resolve()))
