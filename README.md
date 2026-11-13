# GAN-Stego: Robust GAN-U-Net Image Steganography with Depthwise Separable Convolutions

## 1. Hide a message in one image and read it back

```bash
python demo.py --image samples/coco_000000000139.jpg \
               --message "GAN-Stego epoch 96 release test"
```

## 2. Attack the stego image before extraction

```bash
python demo.py --image samples/coco_000000000285.jpg --attack jpeg --param 90
python demo.py --image samples/coco_000000000285.jpg --attack rot  --param 10
python demo.py --image samples/coco_000000000285.jpg --attack crop --param 0.90
```

`--attack` accepts `clean`, `jpeg` (quality), `noise` (sigma), `crop`
(keep-ratio), `bc` (brightness/contrast factor), `blur` (sigma) and `rot`
(degrees).

## 3. Run the evaluation protocol on a folder

```bash
python evaluate.py --data samples --attacks clean,jpeg_q90,crop_0.90,rot_10
```
Useful flags: `--attacks all|quick|clean|<comma-separated ids>`, `--limit N`,
`--seed 42`, `--output-dir results/my_run`.

The attack identifiers are listed in `ganstego/evaluation.py::EVAL_ATTACKS`.

## 4. Train from scratch

```bash
python train.py --data data/coco/train2017 --epochs 110 --seed 42
```

This trains at 400x400, the resolution the paper evaluates at.

To train at 256x256 instead - one single configuration, 2.4x cheaper per epoch:

```bash
python train.py --data data/coco/train2017 --image-size 256 \n                --decoder-warmup-epochs 0 --epochs 110 --seed 42
```

or `scripts/train_256.sh` / `scripts/train_256.ps1`.

A model trained at 256x256 must also be EVALUATED at 256x256, and its numbers
are not comparable with Tables I-IV. [docs/REPRODUCE.md](docs/REPRODUCE.md)
explains why in one short section, with measurements.
