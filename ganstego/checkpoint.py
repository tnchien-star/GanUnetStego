import glob
import os
import re
import torch
from . import config
from .models import Decoder, Discriminator, Encoder

# Find the checkpoint file like model_latest_<number>.pth
def find_latest_checkpoint(directory=None):
    directory = config.SAVE_PATH if directory is None else directory
    candidates = glob.glob(os.path.join(directory, "model_latest_*.pth"))
    if not candidates:
        return ""

    return max(
        candidates,
        key=lambda path: int(re.search(r"model_latest_(\d+)\.pth", path).group(1)),
    )


def describe_checkpoint(checkpoint):
    epoch = int(checkpoint.get("epoch", -1))
    return (
        "variant={} | epoch index={} (displayed epoch {}) | "
        "image_size={} | payload={} bits | coded={} bits".format(
            checkpoint.get("model_variant", "unknown"),
            epoch,
            epoch + 1,          
            checkpoint.get("image_size", "?"),
            checkpoint.get("payload_bits", "?"),
            checkpoint.get("coded_bits", "?"),
        )
    )

# Load the trained Encoder and Decoder from  checkpoint
def load_models(checkpoint_path, device=None, strict=True, with_discriminator=False,
                verbose=True):
    device = config.DEVICE if device is None else device

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if verbose:
        print("Checkpoint {} -> {}".format(checkpoint_path, describe_checkpoint(checkpoint)))

    stored_payload = int(checkpoint.get("payload_bits", config.PAYLOAD_BITS))
    if stored_payload != config.PAYLOAD_BITS:
        print("This checkpoint was not the same payload size as the current config")

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    encoder.load_state_dict(checkpoint["enc"], strict=strict)
    decoder.load_state_dict(checkpoint["dec"], strict=strict)
    encoder.eval()
    decoder.eval()

    if not with_discriminator:
        return encoder, decoder

    discriminator = Discriminator().to(device)
    if "disc" in checkpoint:
        discriminator.load_state_dict(checkpoint["disc"], strict=strict)
    discriminator.eval()
    return encoder, decoder, discriminator

# Copy a checkpoint and drop everything only training needs
def export_inference_checkpoint(source_path, target_path):
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    slim = {
        key: checkpoint[key]
        for key in ("enc", "dec", "disc", "epoch", "model_variant",
                    "image_size", "payload_bits", "coded_bits",
                    "targeted_robustness", "ramp", "jpeg_q_min", "attack_weights")
        if key in checkpoint
    }
    torch.save(slim, target_path)
    return target_path
