import os
import re
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from . import config
from .attacks import AttackSimulator
from .checkpoint import find_latest_checkpoint
from .coding import rs_exact_recovery, rs_payload_accuracy
from .metrics import calculate_psnr, robust_bit_loss
from .models import Decoder, Discriminator, Encoder

ENCODER_NEW_MODULES = ("secret_dense", "secret_context_norm", "secret_context_proj")
DECODER_NEW_MODULES = ("synchronizer", "global_head", "robust_gate")

# Initialize LPIPS and SSIM
def _init_perceptual_losses(device):
    lpips_loss_fn = None
    try:
        import lpips as lpips_module
        lpips_loss_fn = lpips_module.LPIPS(net="vgg").to(device)
        lpips_loss_fn.eval()
        for parameter in lpips_loss_fn.parameters():
            parameter.requires_grad_(False)
        print("LPIPS (VGG) initialised")
    except Exception as error:
        print("LPIPS unavailable ({})".format(error))

    ssim_loss_fn = None
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim_loss_fn = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        print("SSIM initialised")
    except Exception as error:
        print("SSIM unavailable ({})".format(error))

    return lpips_loss_fn, ssim_loss_fn


def train_model(train_loader, val_loader, epochs=config.EPOCHS, resume_path="",
                targeted_robustness=True, reset_optimizers=False,
                save_path=config.SAVE_PATH, device=config.DEVICE,
                warm_start=False):
    print("Starting training ...")
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    discriminator = Discriminator().to(device)
    attack_simulator = AttackSimulator(device=device)

    lpips_loss_fn, ssim_loss_fn = _init_perceptual_losses(device)

    if not resume_path:
        resume_path = find_latest_checkpoint(save_path)
        if resume_path:
            print("auto detected checkpoint: {}".format(resume_path))
    resuming = bool(resume_path) and os.path.exists(resume_path)

    # learning rates 
    fine_tuning = targeted_robustness and resuming
    if fine_tuning:
        legacy_lr, new_lr = config.LEGACY_LR, config.NEW_MODULE_LR
    else:
        legacy_lr = new_lr = config.CURRICULUM_LR
    print("learning rates: pretrained blocks {:g}, new modules {:g}{}".format(
        legacy_lr, new_lr, " (fine-tuning)" if fine_tuning else " (from scratch)"))

    # optimizers 
    enc_new = [p for n, p in encoder.named_parameters() if n.startswith(ENCODER_NEW_MODULES)]
    enc_legacy = [p for n, p in encoder.named_parameters() if not n.startswith(ENCODER_NEW_MODULES)]
    dec_new = [p for n, p in decoder.named_parameters() if n.startswith(DECODER_NEW_MODULES)]
    dec_legacy = [p for n, p in decoder.named_parameters() if not n.startswith(DECODER_NEW_MODULES)]

    enc_opt = optim.Adam([
        {"params": enc_legacy, "lr": legacy_lr},
        {"params": enc_new, "lr": new_lr},
    ])
    dec_opt = optim.Adam([
        {"params": dec_legacy, "lr": legacy_lr},
        {"params": dec_new, "lr": new_lr},
    ])
    disc_opt = optim.Adam(discriminator.parameters(), lr=legacy_lr)

    scaler_d = torch.amp.GradScaler("cuda", enabled=config.USE_AMP)
    scaler_g = torch.amp.GradScaler("cuda", enabled=config.USE_AMP)
    amp_device = "cuda" if config.USE_AMP else "cpu"

    def _decode_logits_in_chunks(model, x, chunk_bs=config.DEC_CHUNK_BS):
        if chunk_bs is None or x.size(0) <= chunk_bs:
            logits, _ = model(x)
            return logits
        outputs = []
        for start in range(0, x.size(0), chunk_bs):
            logits, _ = model(x[start:start + chunk_bs])
            outputs.append(logits)
        return torch.cat(outputs, dim=0)

    def _decode_with_theta_in_chunks(model, x, chunk_bs=config.DEC_CHUNK_BS):
        if chunk_bs is None or x.size(0) <= chunk_bs:
            logits, _, theta = model(x, return_theta=True)
            return logits, theta
        logit_list, theta_list = [], []
        for start in range(0, x.size(0), chunk_bs):
            logits, _, theta = model(x[start:start + chunk_bs], return_theta=True)
            logit_list.append(logits)
            theta_list.append(theta)
        return torch.cat(logit_list, dim=0), torch.cat(theta_list, dim=0)

    start_epoch = 0
    severe_ramp = 0.0                        
    jpeg_q_min = config.JPEG_Q_START         # current JPEG difficulty
    jpeg_ft_epochs_done = 0                  
    attack_weights = {
        "jpeg": config.JPEG_ATTACK_WEIGHT_INIT,
        "crop": config.CROP_ATTACK_WEIGHT_INIT,
        "rot": config.ROT_ATTACK_WEIGHT_INIT,
    }

    if resume_path and os.path.exists(resume_path):
        print("{} checkpoint: {}".format(
            "warm-starting from" if warm_start else "resuming from", resume_path))
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

        if not warm_start:
            severe_ramp = float(checkpoint.get("ramp", 0.0))
            jpeg_q_min = float(checkpoint.get("jpeg_q_min", config.JPEG_Q_START))
            jpeg_ft_epochs_done = int(checkpoint.get("jpeg_ft_epochs_done", 0))

            saved_attack_weights = checkpoint.get("attack_weights")
            if isinstance(saved_attack_weights, dict):
                for key in attack_weights:
                    if key in saved_attack_weights:
                        attack_weights[key] = float(saved_attack_weights[key])

        if "enc" in checkpoint:
            encoder.load_state_dict(checkpoint["enc"], strict=False)
        if "dec" in checkpoint:
            decoder.load_state_dict(checkpoint["dec"], strict=False)
        if "disc" in checkpoint:
            discriminator.load_state_dict(checkpoint["disc"])

        if "epoch" in checkpoint and not warm_start:
            start_epoch = checkpoint["epoch"] + 1

        if "enc_opt" in checkpoint and not reset_optimizers and not warm_start:
            try:
                enc_opt.load_state_dict(checkpoint["enc_opt"])
                dec_opt.load_state_dict(checkpoint["dec_opt"])
                disc_opt.load_state_dict(checkpoint["disc_opt"])
            except (ValueError, KeyError) as error:
                print("optimizer groups not match ({}), starting Adam from scratch".format(error))
        elif not warm_start:
            match = re.search(r"model_latest_(\d+)", resume_path)
            if match and start_epoch == 0:
                start_epoch = int(match.group(1)) + 1
    elif resume_path:
        print("checkpoint '{}' not found, training from scratch".format(resume_path))

    attack_simulator.set_jpeg_q_min(jpeg_q_min)

    if start_epoch >= epochs:
        epochs = start_epoch + 50

    weights = dict(config.LOSS_WEIGHTS)

    for epoch in range(start_epoch, epochs):
        encoder.train()
        decoder.train()
        discriminator.train()

        decoder_only_warmup = (
            targeted_robustness and jpeg_ft_epochs_done < config.JPEG_DECODER_WARMUP_EPOCHS
        )
        for parameter in encoder.parameters():
            parameter.requires_grad_(not decoder_only_warmup)

        if epoch < config.CURRICULUM_NONE_UNTIL:
            severity = "none"
        elif epoch < config.CURRICULUM_LIGHT_UNTIL:
            severity = "light"
        elif epoch < config.CURRICULUM_MODERATE_UNTIL:
            severity = "moderate"
        else:
            severity = "severe"

        ramp = severe_ramp if severity == "severe" else 1.0

        if targeted_robustness:
            attack_lambda = config.TRANSROBUST_W
        elif severity == "severe":
            attack_lambda = config.ATK_W_MIN + (config.ATK_W_MAX - config.ATK_W_MIN) * ramp
        else:
            attack_lambda = config.ATK_W_MAX

        metrics = {key: 0.0 for key in (
            "acc", "atk_acc", "jpeg_acc", "crop_acc", "rot_acc",
            "cover_loss", "lpips", "psnr", "ssim", "res", "logit")}

        phase_label = ("targeted" if targeted_robustness else
                       ("severe t={:.2f}".format(ramp) if severity == "severe" else severity))
        progress = tqdm(train_loader, desc="Ep {} ({})".format(epoch + 1, phase_label))

        for step, (images, payloads, messages) in enumerate(progress):
            images, messages = images.to(device), messages.to(device)

            disc_opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                stego_for_disc = encoder(images, messages)

            with torch.amp.autocast(amp_device, enabled=config.USE_AMP):
                real_loss = F.binary_cross_entropy_with_logits(
                    discriminator(images), torch.ones(images.size(0), 1, device=device))
                fake_loss = F.binary_cross_entropy_with_logits(
                    discriminator(stego_for_disc.detach()),
                    torch.zeros(images.size(0), 1, device=device))
                d_loss = real_loss + fake_loss

            scaler_d.scale(d_loss).backward()
            scaler_d.step(disc_opt)
            scaler_d.update()

            enc_opt.zero_grad(set_to_none=True)
            dec_opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(amp_device, enabled=config.USE_AMP):
                stego = encoder(images, messages)

                noisy_stego = stego + torch.randn_like(stego) * 0.005
                if targeted_robustness:
                    extracted_raw, extracted_quant, clean_theta = decoder(
                        noisy_stego, return_theta=True)
                else:
                    extracted_raw, extracted_quant = decoder(noisy_stego)
                    clean_theta = None

                # losses 
                cover_loss = F.mse_loss(stego, images)          
                loss_lpips = (lpips_loss_fn(stego, images).mean() if lpips_loss_fn is not None
                              else torch.tensor(0.0, device=device))
                loss_ssim = (1 - ssim_loss_fn(stego, images) if ssim_loss_fn is not None
                             else torch.tensor(0.0, device=device))

                msg_loss = F.binary_cross_entropy_with_logits(extracted_raw, messages)

                adv_loss = F.binary_cross_entropy_with_logits(
                    discriminator(stego), torch.ones(images.size(0), 1, device=device))

                consistency_loss = torch.tensor(0.0, device=device)
                sync_loss = torch.tensor(0.0, device=device)

                if targeted_robustness:
                    views = attack_simulator.targeted_transrobust_views(stego)

                    raw_jpeg_diff, theta_jd = _decode_with_theta_in_chunks(
                        decoder, views["jpeg_diff"][0])
                    raw_jpeg_ste, theta_js = _decode_with_theta_in_chunks(
                        decoder, views["jpeg_ste"][0])
                    raw_jpeg_real, theta_jr = _decode_with_theta_in_chunks(
                        decoder, views["jpeg_real"][0])
                    raw_crop, theta_crop = _decode_with_theta_in_chunks(
                        decoder, views["crop"][0])
                    raw_rot, theta_rot = _decode_with_theta_in_chunks(
                        decoder, views["rotation"][0])

                    jpeg_loss = (
                        config.JPEG_STE_WEIGHT * robust_bit_loss(raw_jpeg_ste, messages) +
                        config.JPEG_DIFF_WEIGHT * robust_bit_loss(raw_jpeg_diff, messages)
                    )
                    crop_loss = robust_bit_loss(raw_crop, messages)
                    rot_loss = robust_bit_loss(raw_rot, messages)

                    final_attack_loss = (
                        attack_weights["jpeg"] * jpeg_loss +
                        attack_weights["crop"] * crop_loss +
                        attack_weights["rot"] * rot_loss
                    )

                    teacher_probability = torch.sigmoid(extracted_raw.detach())
                    consistency_loss = torch.stack([
                        F.mse_loss(torch.sigmoid(raw_jpeg_diff), teacher_probability),
                        F.mse_loss(torch.sigmoid(raw_jpeg_ste), teacher_probability),
                        F.mse_loss(torch.sigmoid(raw_jpeg_real), teacher_probability),
                        F.mse_loss(torch.sigmoid(raw_crop), teacher_probability),
                        F.mse_loss(torch.sigmoid(raw_rot), teacher_probability),
                    ]).mean()

                    identity = attack_simulator.identity_theta(
                        images.size(0), images.device, clean_theta.dtype)
                    sync_loss = torch.stack([
                        F.mse_loss(clean_theta.float(), identity.float()),
                        F.mse_loss(theta_jd.float(), views["jpeg_diff"][1].float()),
                        F.mse_loss(theta_js.float(), views["jpeg_ste"][1].float()),
                        F.mse_loss(theta_jr.float(), views["jpeg_real"][1].float()),
                        F.mse_loss(theta_crop.float(), views["crop"][1].float()),
                        F.mse_loss(theta_rot.float(), views["rotation"][1].float()),
                    ]).mean()

                    jpeg_acc_step = ((torch.sigmoid(raw_jpeg_real) > 0.5).float()
                                     == messages).float().mean().item()
                    crop_acc_step = ((torch.sigmoid(raw_crop) > 0.5).float()
                                     == messages).float().mean().item()
                    rot_acc_step = ((torch.sigmoid(raw_rot) > 0.5).float()
                                    == messages).float().mean().item()
                    batch_attack_acc = (jpeg_acc_step + crop_acc_step + rot_acc_step) / 3.0

                elif severity == "none":
                    attacked_quant = extracted_quant
                    final_attack_loss = msg_loss
                    batch_attack_acc = (attacked_quant == messages).float().mean().item()
                    jpeg_acc_step = crop_acc_step = rot_acc_step = batch_attack_acc

                else:
                    if severity == "light":
                        eot_views = min(1, config.EOT_VIEWS)
                    elif severity == "moderate":
                        eot_views = min(2, config.EOT_VIEWS)
                    else:
                        eot_views = min(2 if ramp < 0.5 else 3, config.EOT_VIEWS)

                    diff_losses, real_losses = [], []

                    attacked_diff0, attacked_real0 = attack_simulator.combined_attack_pair(
                        stego, severity, ramp=ramp)
                    raw_diff0 = _decode_logits_in_chunks(decoder, attacked_diff0)
                    raw_real0 = _decode_logits_in_chunks(decoder, attacked_real0)
                    diff_losses.append(F.binary_cross_entropy_with_logits(raw_diff0, messages))
                    real_losses.append(F.binary_cross_entropy_with_logits(raw_real0, messages))
                    attacked_quant = (torch.sigmoid(raw_real0) > 0.5).float()

                    for _ in range(eot_views - 1):
                        attacked_diff, attacked_real = attack_simulator.combined_attack_pair(
                            stego, severity, ramp=ramp)
                        raw_diff = _decode_logits_in_chunks(decoder, attacked_diff)
                        raw_real = _decode_logits_in_chunks(decoder, attacked_real)
                        diff_losses.append(
                            F.binary_cross_entropy_with_logits(raw_diff, messages))
                        real_losses.append(
                            F.binary_cross_entropy_with_logits(raw_real, messages))

                    final_attack_loss = (
                        (1.0 - config.REAL_JPEG_WEIGHT) * torch.stack(diff_losses).mean() +
                        config.REAL_JPEG_WEIGHT * torch.stack(real_losses).mean()
                    )
                    batch_attack_acc = (attacked_quant == messages).float().mean().item()
                    jpeg_acc_step = crop_acc_step = rot_acc_step = batch_attack_acc

                total_loss = (weights["cover"] * cover_loss +
                              weights["lpips"] * loss_lpips +
                              weights["ssim"] * loss_ssim +
                              weights["message"] * msg_loss +
                              attack_lambda * final_attack_loss +
                              config.CONSISTENCY_W * consistency_loss +
                              config.SYNC_W * sync_loss +
                              weights["adv"] * adv_loss)

            scaler_g.scale(total_loss).backward()

            if not decoder_only_warmup:
                scaler_g.unscale_(enc_opt)
            scaler_g.unscale_(dec_opt)

            if not decoder_only_warmup:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)

            if not decoder_only_warmup:
                scaler_g.step(enc_opt)
            scaler_g.step(dec_opt)
            scaler_g.update()

            current_psnr = calculate_psnr(images, stego).item()
            if ssim_loss_fn is not None:
                with torch.no_grad():
                    current_ssim = ssim_loss_fn(stego.float(), images.float()).item()
            else:
                current_ssim = 0.0

            metrics["acc"] += (extracted_quant == messages).float().mean().item()
            metrics["atk_acc"] += batch_attack_acc
            metrics["jpeg_acc"] += jpeg_acc_step
            metrics["crop_acc"] += crop_acc_step
            metrics["rot_acc"] += rot_acc_step
            metrics["cover_loss"] += cover_loss.item()
            metrics["lpips"] += loss_lpips.item()
            metrics["psnr"] += current_psnr
            metrics["ssim"] += current_ssim

            with torch.no_grad():
                metrics["res"] += (stego - images).float().abs().mean().item()
                metrics["logit"] += extracted_raw.float().abs().mean().item()

            done = step + 1
            progress.set_postfix({
                "Acc": "{:.2f}".format(metrics["acc"] / done),
                "Atk": "{:.2f}".format(metrics["atk_acc"] / done),
                "J/C/R": "{:.2f}/{:.2f}/{:.2f}".format(
                    metrics["jpeg_acc"] / done,
                    metrics["crop_acc"] / done,
                    metrics["rot_acc"] / done),
                "LPIPS": "{:.3f}".format(metrics["lpips"] / done),
                "PSNR": "{:.2f}".format(metrics["psnr"] / done),
                "SSIM": "{:.3f}".format(metrics["ssim"] / done),
                "Res": "{:.4f}".format(metrics["res"] / done),
                "Lgt": "{:.2f}".format(metrics["logit"] / done),
            })

        val_acc = val_atk = val_pay = val_psnr = val_ssim = val_lpips = 0.0
        val_jpeg = val_crop = val_rot = 0.0

        jpeg_fixed = {quality: {"bit": 0.0, "exact": 0.0, "decode": 0.0}
                      for quality in (90, 70, 50)}

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for images, payloads, messages in val_loader:
                images, messages = images.to(device), messages.to(device)
                stego = encoder(images, messages)

                if targeted_robustness:
                    views = attack_simulator.targeted_transrobust_views(stego)
                    _, quant_jpeg = decoder(views["jpeg_real"][0])
                    _, quant_crop = decoder(views["crop"][0])
                    _, quant_rot = decoder(views["rotation"][0])

                    attack_quants = [quant_jpeg, quant_crop, quant_rot]
                    attack_accuracies = [(q == messages).float().mean().item()
                                         for q in attack_quants]
                    val_jpeg += attack_accuracies[0]
                    val_crop += attack_accuracies[1]
                    val_rot += attack_accuracies[2]
                    val_atk += sum(attack_accuracies) / 3.0

                    val_pay += float(np.mean([
                        rs_payload_accuracy(q, payloads)[0] for q in attack_quants]))
                    val_atk_quant = quant_jpeg

                    for quality, statistics in jpeg_fixed.items():
                        fixed_attack = attack_simulator.real_jpeg_attack(stego, quality)
                        _, fixed_quant = decoder(fixed_attack)
                        statistics["bit"] += (fixed_quant == messages).float().mean().item()
                        exact_rate, decode_rate = rs_exact_recovery(fixed_quant, payloads)
                        statistics["exact"] += exact_rate
                        statistics["decode"] += decode_rate
                elif severity == "none":
                    attacked = stego
                else:
                    attacked = attack_simulator.combined_attack(
                        stego, severity, jpeg_mode="real", ramp=ramp)

                _, val_quant = decoder(stego)

                if not targeted_robustness:
                    _, val_atk_quant = decoder(attacked)
                    payload_accuracy, _ = rs_payload_accuracy(val_atk_quant, payloads)
                    val_pay += payload_accuracy
                    val_atk += (val_atk_quant == messages).float().mean().item()

                val_acc += (val_quant == messages).float().mean().item()
                val_psnr += calculate_psnr(images, stego).item()
                if lpips_loss_fn is not None:
                    val_lpips += lpips_loss_fn(stego, images).mean().item()
                if ssim_loss_fn is not None:
                    val_ssim += ssim_loss_fn(stego.float(), images.float()).item()

        n_val = max(1, len(val_loader))
        print("Ep {}: Val Acc {:.3f} | Atk Acc {:.3f} | RS-Payload {:.3f} | "
              "PSNR {:.2f} dB | SSIM {:.4f} | LPIPS {:.4f}".format(
                  epoch + 1, val_acc / n_val, val_atk / n_val, val_pay / n_val,
                  val_psnr / n_val, val_ssim / n_val, val_lpips / n_val))

        if targeted_robustness:
            print("JPEG={:.3f} | Crop={:.3f} | Rotation={:.3f}".format(
                val_jpeg / n_val, val_crop / n_val, val_rot / n_val))
            for quality in (90, 70, 50):
                statistics = jpeg_fixed[quality]
                print("[jpeg-Q{}] coded={:.4f} | RS-exact={:.4f} | RS-decode={:.4f}".format(
                    quality, statistics["bit"] / n_val,
                    statistics["exact"] / n_val, statistics["decode"] / n_val))

        n_train = max(1, len(train_loader))
        print("|stego-cover|={:.5f} | |logit|={:.3f} | AMP scale={:.0f} | "
              "atk_w={:.1f}{}".format(
                  metrics["res"] / n_train, metrics["logit"] / n_train,
                  scaler_g.get_scale(), attack_lambda,
                  " | ramp={:.2f}".format(ramp)
                  if severity == "severe" and not targeted_robustness else ""))

        if epoch >= config.CURRICULUM_NONE_UNTIL + 3 and (val_acc / n_val) < 0.60:
            print("Clean validation accuracy below 0.60 - the model may be collapsing."
                  "Resume from the last good checkpoint.")

        if targeted_robustness:
            q90_accuracy = jpeg_fixed[90]["bit"] / n_val
            q70_accuracy = jpeg_fixed[70]["bit"] / n_val
            previous_q_min = jpeg_q_min
            if jpeg_q_min >= 85.0 and q90_accuracy >= config.JPEG_Q90_GATE:
                jpeg_q_min = 70.0
            elif jpeg_q_min >= 70.0 and q70_accuracy >= config.JPEG_Q70_GATE:
                jpeg_q_min = 50.0
            attack_simulator.set_jpeg_q_min(jpeg_q_min)
            if jpeg_q_min != previous_q_min:
                print("jpeg gate passed: Qmin {:.0f} -> {:.0f}".format(previous_q_min, jpeg_q_min))

            current_accuracy = {
                "jpeg": val_jpeg / n_val,
                "crop": val_crop / n_val,
                "rot": val_rot / n_val,
            }
            raw = {key: max(0.02, 1.0 - value) for key, value in current_accuracy.items()}
            raw_sum = sum(raw.values())
            target_weights = {key: value / raw_sum for key, value in raw.items()}
            target_weights = {
                key: min(config.ATTACK_WEIGHT_MAX, max(config.ATTACK_WEIGHT_MIN, value))
                for key, value in target_weights.items()
            }
            target_sum = sum(target_weights.values())
            target_weights = {key: value / target_sum for key, value in target_weights.items()}

            for key in attack_weights:
                attack_weights[key] = (
                    config.ATTACK_WEIGHT_EMA * attack_weights[key] +
                    (1.0 - config.ATTACK_WEIGHT_EMA) * target_weights[key]
                )
            weight_sum = sum(attack_weights.values())
            attack_weights = {key: value / weight_sum for key, value in attack_weights.items()}
            print("jpeg={:.3f}, crop={:.3f}, rotation={:.3f}".format(
                attack_weights["jpeg"], attack_weights["crop"], attack_weights["rot"]))

            if decoder_only_warmup:
                jpeg_ft_epochs_done += 1

        if severity == "severe" and not targeted_robustness:
            current_attack_accuracy = val_atk / n_val
            if current_attack_accuracy >= config.RAMP_GATE_ATK_ACC:
                # One tenth of the way per epoch, so at least 10 epochs to 1.0.
                new_ramp = min(1.0, severe_ramp + 1.0 / config.SEVERE_RAMP_EPOCHS)
                print(" Val AtkAcc {:.3f} >= {:.2f} -> ramp {:.2f} -> {:.2f}".format(
                    current_attack_accuracy, config.RAMP_GATE_ATK_ACC, severe_ramp, new_ramp))
                severe_ramp = new_ramp
            else:
                print(" Val AtkAcc {:.3f} < {:.2f} -> hold ramp {:.2f}".format(
                    current_attack_accuracy, config.RAMP_GATE_ATK_ACC, severe_ramp))

        os.makedirs(save_path, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "ramp": severe_ramp,
            "jpeg_q_min": jpeg_q_min,
            "jpeg_ft_epochs_done": jpeg_ft_epochs_done,
            "attack_weights": attack_weights,
            "model_variant": "v6-jpegfocus",
            "image_size": config.IMAGE_SIZE,
            "payload_bits": config.PAYLOAD_BITS,
            "coded_bits": config.SECRET_LENGTH,
            "targeted_robustness": targeted_robustness,
            "enc": encoder.state_dict(),
            "dec": decoder.state_dict(),
            "disc": discriminator.state_dict(),
            "enc_opt": enc_opt.state_dict(),
            "dec_opt": dec_opt.state_dict(),
            "disc_opt": disc_opt.state_dict(),
        }, os.path.join(save_path, "model_latest_{}.pth".format(epoch)))

    return encoder, decoder, discriminator
