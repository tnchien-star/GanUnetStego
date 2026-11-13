import numpy as np
import torch
from . import config
from .coding import rs_decode_bits, rs_encode_bits

# Make a random secret
def random_payload(payload_bits=None, rng=None):
    payload_bits = config.PAYLOAD_BITS if payload_bits is None else payload_bits
    rng = np.random if rng is None else rng
    return rng.randint(0, 2, payload_bits).astype(np.float32)


def bytes_to_payload(data, payload_bits=None):
    payload_bits = config.PAYLOAD_BITS if payload_bits is None else payload_bits
    max_bytes = payload_bits // 8
    data = bytes(data)[:max_bytes].ljust(max_bytes, b"\x00")
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return bits[:payload_bits].astype(np.float32)


def payload_to_bytes(payload):
    bits = np.asarray(payload).astype(np.uint8).ravel()
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()

# embedding secret inside image
def embed(encoder, cover, payload, device=None):
    device = cover.device if device is None else device
    coded = rs_encode_bits(np.asarray(payload), config.N_SYMBOLS)
    message = torch.from_numpy(coded).unsqueeze(0).to(device)
    if cover.size(0) > 1:
        message = message.expand(cover.size(0), -1)
    with torch.inference_mode():
        stego = encoder(cover.to(device), message)
    return stego, coded

# extract message from stego image
def extract(decoder, stego, device=None):
    device = stego.device if device is None else device
    with torch.inference_mode():
        _, quantized = decoder(stego.to(device))
    predicted = quantized[0, :config.SECRET_LENGTH].cpu().numpy()
    return rs_decode_bits(predicted, config.N_SYMBOLS, payload_bits=config.PAYLOAD_BITS)


def bit_accuracy(reference, recovered):
    reference = np.asarray(reference).ravel()
    recovered = np.asarray(recovered).ravel()[:reference.size]
    return float((reference == recovered).mean())
