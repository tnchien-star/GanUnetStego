import io
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from . import config

class DiffJPEG(nn.Module):

    def __init__(self, height=config.IMAGE_SIZE, width=config.IMAGE_SIZE,
                 differentiable=True, quality=80):
        super().__init__()
        self.differentiable = differentiable
        self.quality = quality
        self.height = height
        self.width = width

        self.register_buffer("dct_matrix", self._get_dct_matrix())

        self.register_buffer("std_y_table", torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ], dtype=torch.float32))

        self.register_buffer("std_c_table", torch.tensor([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
        ], dtype=torch.float32))

    @staticmethod
    def _get_dct_matrix():
        size = 8
        dct_mat = np.zeros((size, size))
        for k in range(size):          # k = frequency
            for n in range(size):      # n = pixel
                alpha = np.sqrt(1 / size) if k == 0 else np.sqrt(2 / size)
                dct_mat[k, n] = alpha * np.cos(np.pi * (2 * n + 1) * k / (2 * size))
        return torch.from_numpy(dct_mat).float()

    # Generate JPEG quantization tables for Y and Cb/Cr based on the quality value
    def get_quality_factors(self, quality):
        
        if quality < 50:
            scale = 5000.0 / quality
        else:
            scale = 200.0 - quality * 2
        q_y = (self.std_y_table * scale + 50) / 100
        q_c = (self.std_c_table * scale + 50) / 100
        return torch.clamp(q_y, min=1, max=255), torch.clamp(q_c, min=1, max=255)

    # Convert an RGB image from [-1, 1] to JPEG-style YCbCr in [0, 255].
    def rgb_to_ycbcr_jpeg(self, image):
        image = (image + 1) / 2.0 * 255.0
        matrix = torch.tensor([[0.299, 0.587, 0.114],
                               [-0.168736, -0.331264, 0.5],
                               [0.5, -0.418688, -0.081312]],
                              device=image.device).view(3, 3, 1, 1)
        bias = torch.tensor([0, 128, 128], device=image.device).view(1, 3, 1, 1)
        return F.conv2d(image, matrix, padding=0) + bias

    # Convert a JPEG-style YCbCr image in [0, 255] back to RGB in [-1, 1]
    def ycbcr_to_rgb_jpeg(self, image):
        bias = torch.tensor([0, 128, 128], device=image.device).view(1, 3, 1, 1)
        image = image - bias
        matrix = torch.tensor([[1.0, 0.0, 1.402],
                               [1.0, -0.344136, -0.714136],
                               [1.0, 1.772, 0.0]],
                              device=image.device).view(3, 3, 1, 1)
        result = F.conv2d(image, matrix)
        return (result / 255.0) * 2.0 - 1.0

    # Round values while preserving a usable gradient for backpropagation
    @staticmethod
    def diff_round(x):        
        return x + (torch.round(x) - x).detach()

    # Split the image into 8×8 blocks and apply a 2D DCT to each block
    def dct_transform(self, x):
        batch, channels, height, width = x.shape

        # If the size is not a multiple of 8, mirror a few pixels at the edge.
        h_pad = (8 - height % 8) % 8
        w_pad = (8 - width % 8) % 8
        if h_pad > 0 or w_pad > 0:
            x = F.pad(x, (0, w_pad, 0, h_pad), mode="reflect")
            batch, channels, height, width = x.shape

        x_reshaped = x.view(batch, channels, height // 8, 8, width // 8, 8)
        x_blocks = x_reshaped.permute(0, 1, 2, 4, 3, 5)

        coefficients = torch.matmul(self.dct_matrix, x_blocks)
        return torch.matmul(coefficients, self.dct_matrix.t())

    # Apply the inverse 2D DCT and reconstruct the image from 8×8 blocks
    def idct_transform(self, coefficients):
        x_blocks = torch.matmul(self.dct_matrix.t(), coefficients)
        x_blocks = torch.matmul(x_blocks, self.dct_matrix)
        batch, channels, h_blocks, w_blocks, _, _ = x_blocks.shape
        x_reshaped = x_blocks.permute(0, 1, 2, 4, 3, 5)
        restored = x_reshaped.reshape(batch, channels, h_blocks * 8, w_blocks * 8)
        return restored[:, :, :self.height, :self.width]

    # Apply differentiable JPEG compression using DCT-based quantization at the specified quality
    def forward(self, x, quality=None):
        quality = self.quality if quality is None else quality
        q_y, q_c = self.get_quality_factors(quality)

        # Reshape the 8x8 tables 
        q_y, q_c = q_y.view(1, 1, 1, 1, 8, 8), q_c.view(1, 1, 1, 1, 8, 8)

        ycbcr = self.rgb_to_ycbcr_jpeg(x) - 128.0
        dct_coefficients = self.dct_transform(ycbcr)

        y_quant = self.diff_round(dct_coefficients[:, 0:1] / q_y) * q_y
        c_quant = self.diff_round(dct_coefficients[:, 1:3] / q_c) * q_c

        dct_quantized = torch.cat([y_quant, c_quant], dim=1)
        return self.ycbcr_to_rgb_jpeg(self.idct_transform(dct_quantized) + 128.0)


# The attack simulator
class AttackSimulator:
    
    def __init__(self, attack_prob=1.0, device=config.DEVICE):
        self.attack_prob = attack_prob

        # Build the differentiable JPEG at the same size as the training images.
        self.jpeg_module = DiffJPEG(
            height=config.IMAGE_SIZE, width=config.IMAGE_SIZE, differentiable=True
        ).to(device)

        self.jpeg_q_min = config.JPEG_Q_START

    # --- JPEG ------
    # Set the minimum JPEG quality used during training, clamped to the range [1, 100]
    def set_jpeg_q_min(self, quality):
        self.jpeg_q_min = float(max(1.0, min(100.0, quality)))

    # Sample a random JPEG quality between the current minimum and 100
    def sample_jpeg_quality(self):
        return random.uniform(self.jpeg_q_min, 100.0)

    # Apply differentiable JPEG compression while preserving gradients to the Encoder
    def differentiable_jpeg(self, image, quality=80):
        return self.jpeg_module(image, quality)

    # Apply real JPEG compression using Pillow
    def real_jpeg_attack(self, x, quality=50, randomize_subsampling=False):
        images_out = []

        x_cpu = x.detach().float().cpu()
        quality = int(max(1, min(100, round(float(quality)))))

        for index in range(x_cpu.size(0)):        
            array = ((x_cpu[index].numpy().transpose(1, 2, 0) + 1.0) / 2.0 * 255.0)
            array = array.astype(np.uint8)

            buffer = io.BytesIO()
            save_kwargs = {"format": "JPEG", "quality": quality}
            if randomize_subsampling:
                save_kwargs["subsampling"] = random.choice([0, 2])
            Image.fromarray(array).save(buffer, **save_kwargs)

            buffer.seek(0)
            decoded = np.array(Image.open(buffer)).astype(np.float32) / 255.0 * 2.0 - 1.0
            images_out.append(torch.from_numpy(decoded.transpose(2, 0, 1)))

        return torch.stack(images_out, dim=0).to(x.device)

    # Use real JPEG 
    def straight_through_real_jpeg(self, image, quality=80):
        jpeg_diff = self.differentiable_jpeg(image, quality)
        jpeg_real = self.real_jpeg_attack(image, quality, randomize_subsampling=True)
        return jpeg_real + (jpeg_diff - jpeg_diff.detach())

    # Add Gaussian noise and clamp to [-1, 1]
    def add_gaussian_noise(self, image, std=0.05):
        return torch.clamp(image + torch.randn_like(image) * std, -1.0, 1.0)

    # Randomly drop 
    def pixel_dropout(self, image, p=0.05):
        if p <= 0:
            return image
        mask = (torch.rand_like(image[:, :1, :, :]) > p).float()
        return image * mask

    # Apply Gaussian blur and clamp to [-1, 1]
    def gaussian_blur(self, x, kernel_size=5, sigma=1.5):
        if kernel_size % 2 == 0:
            kernel_size += 1        

        axis = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1,
                            device=x.device, dtype=torch.float32)

        grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")

        kernel = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2.0 * (sigma ** 2) + 1e-8))

        kernel = kernel / (kernel.sum() + 1e-8)

        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(x.size(1), 1, 1, 1)

        pad = kernel_size // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return torch.clamp(F.conv2d(x_pad, kernel, groups=x.size(1)), -1.0, 1.0)

    # cropping 
    def center_crop_keep_ratio(self, x, keep_ratio=0.8):
        ratio = float(keep_ratio)
        _, _, height, width = x.shape
        new_h, new_w = max(1, int(height * ratio)), max(1, int(width * ratio))

        if new_h >= height or new_w >= width:
            return x

        top = (height - new_h) // 2
        left = (width - new_w) // 2
        cropped = x[:, :, top:top + new_h, left:left + new_w]

        return F.interpolate(cropped, size=(height, width),
                             mode="bilinear", align_corners=False)

    # rotation 
    def fixed_rotation(self, x, angle_deg=30.0):
        angle_rad = math.radians(float(angle_deg))

        theta = torch.tensor(
            [[[math.cos(angle_rad), -math.sin(angle_rad), 0.0],
              [math.sin(angle_rad), math.cos(angle_rad), 0.0]]],
            device=x.device, dtype=torch.float32,
        ).expand(x.size(0), -1, -1)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear",
                             padding_mode="reflection", align_corners=False)

    @staticmethod
    def identity_theta(batch_size, device, dtype=torch.float32):
        
        theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                             device=device, dtype=dtype)
        return theta.unsqueeze(0).expand(batch_size, -1, -1).clone()

    # brightness / contrast 
    def brightness_contrast(self, x, intensity=1.0):
        
        image = (x + 1.0) / 2.0                        
        mean = image.mean(dim=[2, 3], keepdim=True)    
        image = float(intensity) * (image - mean) + mean
        return torch.clamp(image * 2.0 - 1.0, -1.0, 1.0)

    # Generate  JPEG, crop, and rotation attack views in the fine-tuning stage
    def targeted_transrobust_views(self, image):        
        batch = image.size(0)
        identity = self.identity_theta(batch, image.device, image.dtype)

        jpeg_q = self.sample_jpeg_quality()
        keep = random.choices([0.90, 0.80, 0.70, 0.60],
                              weights=[0.40, 0.30, 0.20, 0.10])[0]
        angle_abs = random.choices([5.0, 10.0, 15.0],
                                   weights=[0.50, 0.35, 0.15])[0]
        angle = angle_abs if random.random() < 0.5 else -angle_abs   # both directions

        crop_theta = identity.clone()
        crop_theta[:, 0, 0] = 1.0 / keep
        crop_theta[:, 1, 1] = 1.0 / keep

        inverse = math.radians(-angle)
        rot_theta = identity.clone()
        rot_theta[:, 0, 0] = math.cos(inverse)
        rot_theta[:, 0, 1] = -math.sin(inverse)
        rot_theta[:, 1, 0] = math.sin(inverse)
        rot_theta[:, 1, 1] = math.cos(inverse)

        return {
            "jpeg_diff": (self.differentiable_jpeg(image, jpeg_q), identity, jpeg_q),
            "jpeg_ste": (self.straight_through_real_jpeg(image, jpeg_q), identity, jpeg_q),
            "jpeg_real": (self.real_jpeg_attack(image.detach(), jpeg_q), identity, jpeg_q),
            "crop": (self.center_crop_keep_ratio(image, keep), crop_theta, keep),
            "rotation": (self.fixed_rotation(image, angle), rot_theta, angle),
        }

    @staticmethod
    def _lerp(a, b, t):
        return a + (b - a) * float(t)

    # Linearly interpolate between two lists of weights
    @classmethod
    def _lerp_weights(cls, w0, w1, t):
        return [cls._lerp(a, b, t) for a, b in zip(w0, w1)]

    # Randomly select a crop 
    def _pick_keep_ratio(self, severity, ramp=1.0):
        if severity == "severe":
            weights = self._lerp_weights([0.45, 0.35, 0.20, 0.00],
                                         [0.20, 0.30, 0.35, 0.15], ramp)
            return random.choices([0.90, 0.70, 0.60, 0.50], weights=weights)[0]
        if severity == "moderate":
            return random.choices([0.90, 0.60], weights=[0.45, 0.55])[0]
        return random.choices([0.95, 0.90], weights=[0.50, 0.50])[0]

    # Randomly select a rotation angle
    def _pick_rot_deg(self, severity, ramp=1.0):
        if severity == "severe":
            weights = self._lerp_weights([0.20, 0.50, 0.30, 0.00],
                                         [0.10, 0.30, 0.45, 0.15], ramp)
            return random.choices([0, 5, 10, 15], weights=weights)[0]
        if severity == "moderate":
            return random.choices([0, 5, 10], weights=[0.20, 0.50, 0.30])[0]
        return random.choices([0, 2, 5], weights=[0.35, 0.35, 0.30])[0]

    # Randomly select JPEG quality
    def _pick_jpeg_q(self, severity, ramp=1.0):
        if severity == "severe":
            weights = self._lerp_weights([0.50, 0.40, 0.10],
                                         [0.20, 0.45, 0.35], ramp)
            return random.choices([70, 50, 40], weights=weights)[0]
        if severity == "moderate":
            return random.choices([90, 70, 50], weights=[0.25, 0.45, 0.30])[0]
        return random.choices([95, 90, 80], weights=[0.40, 0.40, 0.20])[0]

    # Generate a random attack plan
    def sample_attack_plan(self, severity="moderate", ramp=1.0):
        if severity == "none":
            return []
        ramp = min(1.0, max(0.0, float(ramp)))     

        if severity == "severe":
            settings = {
                "n": 3 if ramp < 0.5 else 4,
                "noise": (self._lerp(0.01, 0.03, ramp), self._lerp(0.03, 0.05, ramp)),
                "blur": (self._lerp(0.80, 1.20, ramp), self._lerp(1.20, 1.80, ramp)),
                "drop": (self._lerp(0.01, 0.03, ramp), self._lerp(0.03, 0.05, ramp)),
                "bc": (1.0, self._lerp(1.20, 1.30, ramp)),
            }
        else:
            presets = {
                "light": {"n": 2, "noise": (0.005, 0.01), "blur": (0.5, 0.8),
                          "drop": (0.00, 0.01), "bc": (1.0, 1.10)},
                "moderate": {"n": 3, "noise": (0.01, 0.03), "blur": (0.8, 1.2),
                             "drop": (0.01, 0.03), "bc": (1.0, 1.20)},
            }
            settings = presets.get(severity, presets["moderate"])

        plan = []
        if severity == "severe":
            p_force = 0.30 + 0.50 * ramp
            if random.random() < p_force:
                plan.append(("crop", self._pick_keep_ratio(severity, ramp)))
            if random.random() < p_force:
                angle = self._pick_rot_deg(severity, ramp)
                if angle > 0:                       
                    plan.append(("rot", angle))
            if random.random() < p_force:
                plan.append(("jpeg", self._pick_jpeg_q(severity, ramp)))

        candidates = [
            ("jpeg", self._pick_jpeg_q(severity, ramp)),
            ("noise", random.uniform(*settings["noise"])),
            ("blur", random.uniform(*settings["blur"])),
            ("drop", random.uniform(*settings["drop"])),
            ("crop", self._pick_keep_ratio(severity, ramp)),
            ("rot", self._pick_rot_deg(severity, ramp)),
            ("bc", random.uniform(*settings["bc"])),
        ]
        random.shuffle(candidates)      

        used = {name for name, _ in plan}
        needed = max(0, settings["n"] - len(plan))
        for name, parameter in candidates:
            if needed <= 0:
                break
            if name in used:            # never use the same attack twice
                continue
            if name == "rot" and float(parameter) == 0.0:
                continue                # 0 degree rotation is not an attack
            plan.append((name, parameter))
            used.add(name)
            needed -= 1

        return plan

    # Apply the attacks
    def apply_attack_plan(self, image, plan, jpeg_mode="diff"):
        x = image
        for name, parameter in plan:
            if name == "jpeg":
                x = (self.real_jpeg_attack(x, parameter) if jpeg_mode == "real"
                     else self.differentiable_jpeg(x, parameter))
            elif name == "noise":
                x = self.add_gaussian_noise(x, float(parameter))
            elif name == "blur":
                x = self.gaussian_blur(x, kernel_size=5, sigma=float(parameter))
            elif name == "drop":
                x = self.pixel_dropout(x, float(parameter))
            elif name == "crop":
                x = self.center_crop_keep_ratio(x, float(parameter))
            elif name == "rot":
                x = self.fixed_rotation(x, float(parameter))
            elif name == "bc":
                x = self.brightness_contrast(x, float(parameter))
        return torch.clamp(x, -1.0, 1.0)

    # generate random and apply a combined attack 
    def combined_attack(self, image, severity="moderate", jpeg_mode="diff", ramp=1.0):
        if severity == "none":
            return image
        if random.random() > self.attack_prob:
            return image                 
        plan = self.sample_attack_plan(severity, ramp=ramp)
        return self.apply_attack_plan(image, plan, jpeg_mode=jpeg_mode)

    # Apply the same attack plan 
    def combined_attack_pair(self, image, severity="moderate", ramp=1.0):
        if severity == "none" or random.random() > self.attack_prob:
            return image, image
        plan = self.sample_attack_plan(severity, ramp=ramp)
        attacked_diff = self.apply_attack_plan(image, plan, jpeg_mode="diff")
        attacked_real = self.apply_attack_plan(image.detach(), plan, jpeg_mode="real")
        return attacked_diff, attacked_real

EnhancedAttackSimulator = AttackSimulator
