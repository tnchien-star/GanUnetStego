import csv
import json
import os
import numpy as np
import torch
from . import config
from .attacks import AttackSimulator
from .coding import rs_decode_bits, rs_encode_bits
from .data import build_transform, list_images, to_tensor
from .metrics import per_image_psnr

EVAL_ATTACKS = [
    ("clean", "Clean", "clean", 0),
    ("jpeg_q90", "JPEG Q = 90", "jpeg", 90),
    ("jpeg_q70", "JPEG Q = 70", "jpeg", 70),
    ("jpeg_q50", "JPEG Q = 50", "jpeg", 50),
    ("noise_0.01", "Gaussian noise 0.01", "noise", 0.01),
    ("noise_0.02", "Gaussian noise 0.02", "noise", 0.02),
    ("noise_0.05", "Gaussian noise 0.05", "noise", 0.05),
    ("crop_0.90", "Cropping 90%", "crop", 0.90),
    ("crop_0.60", "Cropping 60%", "crop", 0.60),
    ("crop_0.30", "Cropping 30%", "crop", 0.30),
    ("bc_1.0", "Brightness / contrast 1.0", "bc", 1.0),
    ("bc_1.2", "Brightness / contrast 1.2", "bc", 1.2),
    ("bc_1.3", "Brightness / contrast 1.3", "bc", 1.3),
    ("blur_0.5", "Gaussian blur 0.5", "blur", 0.5),
    ("blur_1.5", "Gaussian blur 1.5", "blur", 1.5),
    ("blur_2.0", "Gaussian blur 2.0", "blur", 2.0),
    ("rot_5", "Rotation 5 deg", "rot", 5),
    ("rot_10", "Rotation 10 deg", "rot", 10),
    ("rot_15", "Rotation 15 deg", "rot", 15),
]

QUICK_ATTACK_IDS = ("clean", "jpeg_q90", "jpeg_q70")

METRIC_NAMES = (
    "psnr", "ssim", "lpips_vgg", "coded_bit_accuracy",
    "rs_payload_accuracy", "rs_exact_recovery", "rs_decode_success",
)


def apply_single_attack(simulator, x, name, parameter):
    if name == "clean":
        return x
    if name == "jpeg":
        return simulator.real_jpeg_attack(x, parameter)
    if name == "noise":
        return simulator.add_gaussian_noise(x, float(parameter))
    if name == "crop":
        return simulator.center_crop_keep_ratio(x, float(parameter))
    if name == "bc":
        return simulator.brightness_contrast(x, float(parameter))
    if name == "blur":
        return simulator.gaussian_blur(x, kernel_size=5, sigma=float(parameter))
    if name == "rot":
        return simulator.fixed_rotation(x, float(parameter))
    raise ValueError("No attack '{}'".format(name))


def select_attacks(selection="all"):
    if selection in (None, "", "all"):
        return list(EVAL_ATTACKS)
    if selection == "quick":
        return [row for row in EVAL_ATTACKS if row[0] in QUICK_ATTACK_IDS]
    if selection == "clean":
        return [row for row in EVAL_ATTACKS if row[0] == "clean"]

    wanted = {token.strip() for token in selection.split(",") if token.strip()}
    chosen = [row for row in EVAL_ATTACKS if row[0] in wanted]
    unknown = wanted - {row[0] for row in chosen}
    if unknown:
        raise ValueError("No attack: {}".format(", ".join(sorted(unknown))))
    return chosen


def _load_lpips(device):
    try:
        import lpips
    except ImportError:
        print("LPIPS not installed")
        return None
    try:
        return lpips.LPIPS(net="vgg").to(device).eval()
    except Exception as error:
        print("LPIPS failed to initialize ({}).".format(error))
        return None


def _load_ssim(device):
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
    except ImportError:
        print("'torchmetrics' not installed")
        return None
    return StructuralSimilarityIndexMeasure(data_range=2.0).to(device)


# Evaluate robustness on an image folder
def evaluate_folder(image_dir, encoder, decoder, attacks="all", limit=None,
                    seed=42, device=None, dataset_id=None, progress=True,
                    checkpoint_name=""):
    device = config.DEVICE if device is None else device
    attack_list = select_attacks(attacks) if isinstance(attacks, str) else list(attacks)

    image_paths = list_images(image_dir, sort=True)
    if limit:
        image_paths = image_paths[:limit]

    simulator = AttackSimulator(device=device)
    transform = build_transform()
    lpips_fn = _load_lpips(device)
    ssim_fn = _load_ssim(device)

    values = {row[0]: {metric: [] for metric in METRIC_NAMES} for row in attack_list}

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    iterator = image_paths
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(image_paths, desc="evaluate:{}".format(dataset_id or "images"))
        except ImportError:
            pass

    with torch.inference_mode():
        for image_path in iterator:

            payload = np.random.randint(0, 2, config.PAYLOAD_BITS).astype(np.float32)
            coded = rs_encode_bits(payload, config.N_SYMBOLS)
            message = torch.from_numpy(coded).unsqueeze(0).to(device)

            cover = to_tensor(image_path, transform, device)
            stego = encoder(cover, message)

            for attack_id, _, attack_name, attack_param in attack_list:
                attacked = (stego if attack_name == "clean"
                            else apply_single_attack(simulator, stego, attack_name, attack_param))

                _, quantized = decoder(attacked)
                predicted = quantized[0, :config.SECRET_LENGTH].cpu().numpy()

                current = values[attack_id]

                current["psnr"].append(float(per_image_psnr(attacked, cover)[0]))
                current["ssim"].append(
                    float(ssim_fn(attacked.float(), cover.float())) if ssim_fn else float("nan"))
                current["lpips_vgg"].append(
                    float(lpips_fn(attacked, cover).view(-1)[0]) if lpips_fn else float("nan"))

                current["coded_bit_accuracy"].append(float((predicted == coded).mean()))

                decoded, decode_ok = rs_decode_bits(
                    predicted, config.N_SYMBOLS, payload_bits=config.PAYLOAD_BITS)
                payload_accuracy = float((decoded[:config.PAYLOAD_BITS] == payload).mean())
                current["rs_payload_accuracy"].append(payload_accuracy)

                current["rs_exact_recovery"].append(float(payload_accuracy == 1.0))
                current["rs_decode_success"].append(float(bool(decode_ok)))

    rows = []
    for attack_id, attack_label, _, _ in attack_list:
        row = {
            "dataset": dataset_id or os.path.basename(os.path.normpath(image_dir)),
            "attack": attack_id,
            "attack_label": attack_label,
            "image_count": len(image_paths),
            "image_size": config.IMAGE_SIZE,
            "raw_payload_bits": config.PAYLOAD_BITS,
            "rs_coded_bits": config.SECRET_LENGTH,
            "rs_configuration": "RS(n={},k={},t={} bytes)".format(
                config.K_BYTES + config.N_SYMBOLS, config.K_BYTES, config.N_SYMBOLS // 2),
            "checkpoint": checkpoint_name,
            "lpips_backbone": "VGG",
            "seed": seed,
        }
        for metric, metric_values in values[attack_id].items():
            row["{}_mean".format(metric)] = float(np.mean(metric_values))
            row["{}_std".format(metric)] = float(np.std(metric_values))

        row["rs_exact_recovery_count"] = int(sum(values[attack_id]["rs_exact_recovery"]))
        rows.append(row)
    return rows


def write_results(rows, csv_path=None, json_path=None):
    if csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    if json_path:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)


def format_table(rows):
    header = "{:<28}{:>10}{:>10}{:>10}{:>12}{:>12}{:>10}".format(
        "Attack", "PSNR", "SSIM", "LPIPS", "CodedAcc", "PayloadAcc", "Exact")
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append("{:<28}{:>10.2f}{:>10.4f}{:>10.4f}{:>12.4f}{:>12.4f}{:>10.3f}".format(
            row["attack_label"],
            row["psnr_mean"],
            row["ssim_mean"],
            row["lpips_vgg_mean"],
            row["coded_bit_accuracy_mean"],
            row["rs_payload_accuracy_mean"],
            row["rs_exact_recovery_mean"],
        ))
    return "\n".join(lines)
