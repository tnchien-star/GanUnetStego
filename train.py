import argparse
from torch.utils.data import DataLoader
from ganstego import config
from ganstego.data import SteganographyDataset, build_transform
from ganstego.metrics import set_seed
from ganstego.training import train_model


def parse_args():
    """Describe the command line options."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=str, default=config.DATA_PATH)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save-path", type=str, default=config.SAVE_PATH, help="model_latest_<number>.pth")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--max-images", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--legacy-eot", action="store_true")
    parser.add_argument("--reset-optimizers", action="store_true")

    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--payload-bits", type=int, default=None)
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--decoder-warmup-epochs", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    if args.deterministic:
        import torch
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print("cuDNN set to deterministic mode")

    if args.image_size is not None or args.payload_bits is not None:
        config.configure(image_size=args.image_size, payload_bits=args.payload_bits)

    if args.decoder_warmup_epochs is not None:
        config.JPEG_DECODER_WARMUP_EPOCHS = int(args.decoder_warmup_epochs)
        print("decoder-only warm-up: {} epoch(s)".format(config.JPEG_DECODER_WARMUP_EPOCHS))

    print(config.describe_configuration())
    print("-" * 50)

    transform = build_transform()
    train_dataset = SteganographyDataset(args.data, transform, max_images=args.max_images)
    val_dataset = SteganographyDataset(args.data, transform, validation=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    train_model(
        train_loader,
        val_loader,
        epochs=args.epochs,
        resume_path=args.resume,
        targeted_robustness=not args.legacy_eot,
        reset_optimizers=args.reset_optimizers,
        save_path=args.save_path,
        warm_start=args.warm_start,
    )


if __name__ == "__main__":
    main()
