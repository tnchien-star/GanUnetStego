import argparse
import os
import numpy as np
import torch
from ganstego import config
from ganstego.attacks import AttackSimulator
from ganstego.checkpoint import load_models
from ganstego.data import build_transform, to_pil, to_tensor
from ganstego.evaluation import apply_single_attack
from ganstego.metrics import per_image_psnr, set_seed
from ganstego.pipeline import bytes_to_payload, embed, extract, payload_to_bytes, random_payload

DEFAULT_CHECKPOINT = os.path.join("checkpoints", "ganstego_epoch96.pth")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=str, required=True, help="the cover photo")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--message", type=str, default=None,
                        help="text to hide, up to 56 characters; random bits if omitted")
    parser.add_argument("--attack", type=str, default="clean",
                        choices=["clean", "jpeg", "noise", "crop", "bc", "blur", "rot"],
                        help="damage applied before reading the message back")
    parser.add_argument("--param", type=float, default=90,
                        help="strength of the attack, see the list above")
    parser.add_argument("--output-dir", type=str, default="demo_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else config.DEVICE

    encoder, decoder = load_models(args.checkpoint, device=device, strict=True)
    simulator = AttackSimulator(device=device)

    cover = to_tensor(args.image, build_transform(), device)

    if args.message is not None:
        payload = bytes_to_payload(args.message.encode("utf-8"))
    else:
        payload = random_payload()

    stego, _ = embed(encoder, cover, payload, device=device)

    psnr = float(per_image_psnr(stego, cover)[0])
    print("Stego: PSNR {:.2f} dB vs the cover".format(psnr))

    if args.attack == "clean":
        attacked = stego
        print("Attack   : none")
    else:
        attacked = apply_single_attack(simulator, stego, args.attack, args.param)
        print("Attack   : {} ({})".format(args.attack, args.param))

    recovered, decode_ok = extract(decoder, attacked, device=device)

    payload_accuracy = float((recovered[:config.PAYLOAD_BITS] == payload).mean())
    exact = bool(np.array_equal(recovered[:config.PAYLOAD_BITS], payload))

    print("RS decode success : {}".format(decode_ok))
    print("Payload-bit accuracy  : {:.4f} ({}/{} bits)".format(
        payload_accuracy, int(payload_accuracy * config.PAYLOAD_BITS), config.PAYLOAD_BITS))
    print("Exact recovery    : {}".format(exact))

    if args.message is not None:
        text = payload_to_bytes(recovered).rstrip(b"\x00")
        try:
            print("Recovered message : {!r}".format(text.decode("utf-8")))
        except UnicodeDecodeError:
            print("Recovered message : {!r}".format(text))

    os.makedirs(args.output_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(args.image))[0]

    to_pil(stego).save(os.path.join(args.output_dir, stem + "_stego.png"))
    if args.attack != "clean":
        to_pil(attacked).save(os.path.join(args.output_dir, stem + "_attacked.png"))

    residual = torch.clamp((stego - cover) * 10.0, -1.0, 1.0)
    to_pil(residual).save(os.path.join(args.output_dir, stem + "_residual.png"))

    print("Images saved to   : {}".format(os.path.abspath(args.output_dir)))


if __name__ == "__main__":
    main()
