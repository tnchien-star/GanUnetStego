import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config

# Generate a spatial attention mask
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()      

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

# Generate channel attention weights
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)   
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(batch, channels))
        max_out = self.fc(self.max_pool(x).view(batch, channels))
        return self.sigmoid(avg_out + max_out).view(batch, channels, 1, 1)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU(inplace=True)      

    def forward(self, x):
        return self.relu(self.norm(self.pointwise(self.depthwise(x))))


class SpatialSynchronizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
        )

        self.theta_head = nn.Linear(128, 6)
        nn.init.zeros_(self.theta_head.weight)
        nn.init.zeros_(self.theta_head.bias)

    def forward(self, image):
        delta = 0.75 * torch.tanh(self.theta_head(self.localization(image)))

        identity = image.new_tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        theta = (delta + identity).view(-1, 2, 3)
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        aligned = F.grid_sample(image, grid, mode="bilinear",
                                padding_mode="reflection", align_corners=False)
        return aligned, theta


# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(nn.Conv2d(4, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), DepthwiseSeparableConv(64, 64), nn.Dropout(0.1))
        self.ca1, self.sa1 = ChannelAttention(64), SpatialAttention()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(DepthwiseSeparableConv(64, 128), DepthwiseSeparableConv(128, 128), nn.Dropout(0.1))
        self.ca2, self.sa2 = ChannelAttention(128), SpatialAttention()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Sequential(DepthwiseSeparableConv(128, 256), DepthwiseSeparableConv(256, 256), nn.Dropout(0.1))
        self.ca3, self.sa3 = ChannelAttention(256), SpatialAttention()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = nn.Sequential(DepthwiseSeparableConv(256, 512), DepthwiseSeparableConv(512, 512))

        self.secret_dense = nn.Linear(config.MAX_CODED_BITS, 16 * 8 * 8)
        self.secret_context_norm = nn.GroupNorm(8, 16)
        self.secret_context_proj = nn.Conv2d(16, 512, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.secret_dense.weight, nonlinearity="relu")
        nn.init.zeros_(self.secret_dense.bias)
        nn.init.normal_(self.secret_context_proj.weight, mean=0.0, std=1e-3)

        self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), nn.Conv2d(512, 256, kernel_size=3, padding=1))
        self.dec3 = nn.Sequential(DepthwiseSeparableConv(512, 256), DepthwiseSeparableConv(256, 256), nn.Dropout(0.1))
        self.upconv2 = nn.Sequential( nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), nn.Conv2d(256, 128, kernel_size=3, padding=1))
        self.dec2 = nn.Sequential(DepthwiseSeparableConv(256, 128), DepthwiseSeparableConv(128, 128), nn.Dropout(0.1))
        self.upconv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), nn.Conv2d(128, 64, kernel_size=3, padding=1))
        self.dec1 = nn.Sequential(DepthwiseSeparableConv(128, 64), DepthwiseSeparableConv(64, 64))

        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, image, secret):
        batch_size = image.size(0)

        secret_padded = torch.zeros(batch_size, config.HEIGHT * config.WIDTH, device=secret.device)
        secret_padded[:, :config.SECRET_LENGTH] = secret
        secret_tensor = secret_padded.view(batch_size, 1, config.HEIGHT, config.WIDTH)
        secret_tensor = F.interpolate(secret_tensor.float(), size=(image.size(2), image.size(3)), mode="nearest")

        combined = torch.cat([image, secret_tensor], dim=1)
        e1 = self.enc1(combined)
        e1 = e1 * self.ca1(e1) * self.sa1(e1)     
        e2 = self.enc2(self.pool1(e1))
        e2 = e2 * self.ca2(e2) * self.sa2(e2)
        e3 = self.enc3(self.pool2(e2))
        e3 = e3 * self.ca3(e3) * self.sa3(e3)
        e4 = self.enc4(self.pool3(e3))

        secret_fixed = secret.new_zeros(batch_size, config.MAX_CODED_BITS)
        n_secret = min(secret.size(1), config.MAX_CODED_BITS)
        secret_fixed[:, :n_secret] = secret[:, :n_secret] - 0.5
        secret_ctx = self.secret_dense(secret_fixed).view(batch_size, 16, 8, 8)
        secret_ctx = F.relu(self.secret_context_norm(secret_ctx), inplace=True)
        secret_ctx = F.interpolate(secret_ctx, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        e4 = e4 + self.secret_context_proj(secret_ctx)

        d3 = self.dec3(torch.cat([self.upconv3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))

        residual = self.tanh(self.final_conv(d1))
        return torch.clamp(image + config.EMBED_SCALE * residual, -1.0, 1.0)


# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.synchronizer = SpatialSynchronizer()

        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), DepthwiseSeparableConv(64, 64))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(DepthwiseSeparableConv(64, 128), DepthwiseSeparableConv(128, 128))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Sequential(DepthwiseSeparableConv(128, 256), DepthwiseSeparableConv(256, 256))
        self.pool3 = nn.MaxPool2d(2, 2)

        self.self_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(256)

        self.enc4 = nn.Sequential(DepthwiseSeparableConv(256, 512), DepthwiseSeparableConv(512, 512))

        self.global_head = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(64 * 4 * 4, config.MAX_CODED_BITS),
        )
        nn.init.normal_(self.global_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.global_head[-1].bias)

        self.robust_gate = nn.Parameter(torch.tensor(-2.0))

        self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), nn.Conv2d(512, 256, kernel_size=3, padding=1))
        self.dec3 = nn.Sequential(DepthwiseSeparableConv(512, 256), DepthwiseSeparableConv(256, 256))
        self.upconv2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), nn.Conv2d(256, 128, kernel_size=3, padding=1))
        self.dec2 = nn.Sequential(DepthwiseSeparableConv(256, 128), DepthwiseSeparableConv(128, 128))
        self.upconv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), nn.Conv2d(128, 64, kernel_size=3, padding=1))
        self.dec1 = nn.Sequential(DepthwiseSeparableConv(128, 64), DepthwiseSeparableConv(64, 64))

        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, stego_image, return_theta=False):
        aligned_image, theta = self.synchronizer(stego_image)

        e1 = self.enc1(aligned_image)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        batch, channels, height, width = e3.shape
        attn_h, attn_w = min(height, config.ATTENTION_SIDE), min(width, config.ATTENTION_SIDE)
        e3_small = F.adaptive_avg_pool2d(e3, (attn_h, attn_w))

        e3_flat = e3_small.flatten(2).transpose(1, 2)
        e3_attn, _ = self.self_attention(e3_flat, e3_flat, e3_flat, need_weights=False)

        e3_attn = self.norm(e3_attn + e3_flat).transpose(1, 2).view(
            batch, channels, attn_h, attn_w)
        e3 = e3 + F.interpolate(e3_attn, size=(height, width), mode="bilinear", align_corners=False)

        e4 = self.enc4(self.pool3(e3))
        global_logits = self.global_head(e4)

        d3 = self.dec3(torch.cat([self.upconv3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))

        x = self.final_conv(d1)
        x = F.interpolate(x, size=(config.HEIGHT, config.WIDTH), mode="bilinear")
        spatial_logits = x.view(stego_image.size(0), -1)[:, :config.SECRET_LENGTH]
        global_logits = global_logits[:, :config.SECRET_LENGTH]

        gate = torch.sigmoid(self.robust_gate)      
        secret = spatial_logits + gate * global_logits

        quantized = (torch.sigmoid(secret) > 0.5).float()

        if return_theta:
            return secret, quantized, theta
        return secret, quantized


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dsconv1 = DepthwiseSeparableConv(3, 64)
        self.dsconv2 = DepthwiseSeparableConv(64, 128)
        self.dsconv3 = DepthwiseSeparableConv(128, 256)
        self.final_conv = nn.Conv2d(256, 1, kernel_size=3, padding=1)

    def forward(self, image):
        x = F.avg_pool2d(self.dsconv1(image), 2)
        x = F.avg_pool2d(self.dsconv2(x), 2)
        x = F.avg_pool2d(self.dsconv3(x), 2)

        return self.final_conv(x).view(x.size(0), -1).mean(dim=1, keepdim=True)


EnhancedEncoder = Encoder
EnhancedDecoder = Decoder
