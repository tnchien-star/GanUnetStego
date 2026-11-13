import numpy as np
from . import config
try:
    from reedsolo import RSCodec, ReedSolomonError
except ImportError:
    RSCodec = None

    class ReedSolomonError(Exception):
        """The reedsolo library is not installed."""


def _bits_to_bytes(bits):
    bits = np.asarray(bits).astype(np.uint8).ravel()
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return bytearray(np.packbits(bits).tolist())


def _bytes_to_bits(byte_seq, n_bits):
    arr = np.frombuffer(bytes(byte_seq), dtype=np.uint8)
    bits = np.unpackbits(arr)[:n_bits]
    return bits.astype(np.float32)

# Encode payload with Reed–Solomon 
def rs_encode_bits(payload_bits_arr, n_symbols=None, payload_bits=None):
    n_symbols = config.N_SYMBOLS if n_symbols is None else n_symbols
    payload_bits = config.PAYLOAD_BITS if payload_bits is None else payload_bits
    payload_bits_arr = np.asarray(payload_bits_arr).ravel()[:payload_bits]

    if RSCodec is None or n_symbols <= 0:
        return payload_bits_arr.astype(np.float32)

    data_bytes = _bits_to_bytes(payload_bits_arr)                    
    coded = RSCodec(n_symbols).encode(data_bytes)                    
    return _bytes_to_bits(coded, (len(data_bytes) + n_symbols) * 8)  

# Decode payload with Reed–Solomon 
def rs_decode_bits(coded_bits_arr, n_symbols=None, payload_bits=None):
    n_symbols = config.N_SYMBOLS if n_symbols is None else n_symbols
    payload_bits = config.PAYLOAD_BITS if payload_bits is None else payload_bits
    coded = np.round(np.asarray(coded_bits_arr, dtype=np.float32).ravel()).clip(0, 1)

    if RSCodec is None or n_symbols <= 0:
        return coded[:payload_bits].astype(np.float32), True

    byte_arr = _bits_to_bytes(coded)
    try:
        message = RSCodec(n_symbols).decode(byte_arr)[0]
        return _bytes_to_bits(message, payload_bits), True
    except Exception:
        return coded[:payload_bits].astype(np.float32), False


# Compute the average recovered payload accuracy
def rs_payload_accuracy(coded_pred_bits, payloads):
    pred_np = coded_pred_bits.detach().cpu().numpy()
    pay_np = payloads.detach().cpu().numpy()
    accuracies = []
    successes = 0
    for index in range(pred_np.shape[0]):        # one photo at a time
        decoded, ok = rs_decode_bits(pred_np[index], config.N_SYMBOLS)
        correct = (decoded == pay_np[index][:config.PAYLOAD_BITS])
        accuracies.append(float(correct.mean()))
        successes += int(ok)

    return float(np.mean(accuracies)), successes / max(1, pred_np.shape[0])

# Compute the exact payload recovery rate
def rs_exact_recovery(coded_pred_bits, payloads):
    pred_np = coded_pred_bits.detach().cpu().numpy()
    pay_np = payloads.detach().cpu().numpy()

    exact = 0
    decode_ok = 0
    for index in range(pred_np.shape[0]):
        decoded, ok = rs_decode_bits(pred_np[index], config.N_SYMBOLS)
        decode_ok += int(ok)
        exact += int(np.array_equal(decoded, pay_np[index][:config.PAYLOAD_BITS]))

    denominator = max(1, pred_np.shape[0])
    return exact / denominator, decode_ok / denominator
