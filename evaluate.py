import argparse
import os
import torch
from ganstego import config
from ganstego.checkpoint import load_models
from ganstego.evaluation import evaluate_folder, format_table, write_results

DEFAULT_CHECKPOINT = os.path.join("checkpoints", "ganstego_epoch96.pth")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=str, required=True, help="folder of cover images")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--attacks", type=str, default="all", help="'all', 'quick', 'clean'")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N photos")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="save results.csv and results.json")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--payload-bits", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else config.DEVICE

    if args.image_size is not None or args.payload_bits is not None:
        config.configure(image_size=args.image_size, payload_bits=args.payload_bits)

    encoder, decoder = load_models(args.checkpoint, device=device, strict=True)

    rows = evaluate_folder(
        args.data,
        encoder,
        decoder,
        attacks=args.attacks,
        limit=args.limit,
        seed=args.seed,
        device=device,
        dataset_id=args.dataset_name,
        checkpoint_name=os.path.basename(args.checkpoint),
    )

    print()
    print(format_table(rows))

    if args.output_dir:
        csv_path = os.path.join(args.output_dir, "results.csv")
        json_path = os.path.join(args.output_dir, "results.json")
        write_results(rows, csv_path=csv_path, json_path=json_path)
        print("\nSaved: {}\n       {}".format(csv_path, json_path))


if __name__ == "__main__":
    main()
