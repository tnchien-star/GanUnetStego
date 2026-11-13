import math
import os
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_AMP = torch.cuda.is_available()

BATCH_SIZE = 2
IMAGE_SIZE = 400
PAYLOAD_BITS = 448
RS_ECC_RATIO = 1.0
EPOCHS = 110
SAVE_PATH = "./steganography_model"   
DATA_PATH = "./data/coco/train2017"   

EOT_VIEWS = 3
REAL_JPEG_WEIGHT = 0.25
DEC_CHUNK_BS = 1

JPEG_Q_START = 85.0                 
JPEG_Q_STAGES = (85.0, 70.0, 50.0)  
JPEG_Q90_GATE = 0.990
JPEG_Q70_GATE = 0.980
JPEG_DECODER_WARMUP_EPOCHS = 2

JPEG_ATTACK_WEIGHT_INIT = 0.60
CROP_ATTACK_WEIGHT_INIT = 0.25
ROT_ATTACK_WEIGHT_INIT = 0.15

ATTACK_WEIGHT_EMA = 0.80   
ATTACK_WEIGHT_MIN = 0.10   
ATTACK_WEIGHT_MAX = 0.70   

JPEG_STE_WEIGHT = 0.60
JPEG_DIFF_WEIGHT = 0.40

EMBED_SCALE = 0.05

CURRICULUM_NONE_UNTIL = 2       # epochs 0-1  : no damage 
CURRICULUM_LIGHT_UNTIL = 20     # epochs 2-19 : mild damage
CURRICULUM_MODERATE_UNTIL = 55  # epochs 20-54: medium damage; 55 and up: severe

SEVERE_RAMP_EPOCHS = 10
RAMP_GATE_ATK_ACC = 0.80

# grows from 20 to 30 as the ramp progresses.
ATK_W_MIN, ATK_W_MAX = 20.0, 30.0

MAX_CODED_BITS = 1024

ATTENTION_SIDE = 24

TRANSROBUST_W = 24.0   
CONSISTENCY_W = 2.0    
SYNC_W = 0.5           

CURRICULUM_LR = 1e-4  
LEGACY_LR = 2e-5      
NEW_MODULE_LR = 1e-4  

LOSS_WEIGHTS = {
    "cover": 10.0,     
    "lpips": 50.0,     
    "ssim": 50.0,      
    "message": 15.0,   
    "adv": 1.0,        
}

# Compute Reed–Solomon message geometry
def rs_geometry(payload_bits=PAYLOAD_BITS, ecc_ratio=RS_ECC_RATIO, rs_available=True):
    
    k_bytes = (payload_bits + 7) // 8

    if not rs_available:
        return payload_bits, 0, k_bytes

    n_symbols = max(10, int(k_bytes * ecc_ratio))
    return (k_bytes + n_symbols) * 8, n_symbols, k_bytes

try:
    import reedsolo  
    RS_AVAILABLE = True
except ImportError:
    RS_AVAILABLE = False

SECRET_LENGTH, N_SYMBOLS, K_BYTES = rs_geometry(PAYLOAD_BITS, RS_ECC_RATIO, RS_AVAILABLE)

HEIGHT = int(math.ceil(math.sqrt(SECRET_LENGTH)))
WIDTH = int(math.ceil(SECRET_LENGTH / HEIGHT))


def configure(image_size=None, payload_bits=None, rs_ecc_ratio=None):
    global IMAGE_SIZE, PAYLOAD_BITS, RS_ECC_RATIO
    global SECRET_LENGTH, N_SYMBOLS, K_BYTES, HEIGHT, WIDTH

    if image_size is not None:
        IMAGE_SIZE = int(image_size)
    if payload_bits is not None:
        PAYLOAD_BITS = int(payload_bits)
    if rs_ecc_ratio is not None:
        RS_ECC_RATIO = float(rs_ecc_ratio)

    SECRET_LENGTH, N_SYMBOLS, K_BYTES = rs_geometry(
        PAYLOAD_BITS, RS_ECC_RATIO, RS_AVAILABLE)
    HEIGHT = int(math.ceil(math.sqrt(SECRET_LENGTH)))
    WIDTH = int(math.ceil(SECRET_LENGTH / HEIGHT))

    if SECRET_LENGTH > MAX_CODED_BITS:
        raise ValueError(
            "The coded message needs {} bits but the message are only {} ".format(SECRET_LENGTH, MAX_CODED_BITS)
        )
    return describe_configuration()


def describe_configuration():
    n_bytes = K_BYTES + N_SYMBOLS
    lines = [
        "Payload        : {} bits ({} bytes)".format(PAYLOAD_BITS, K_BYTES),
        "Coded message  : {} bits on a {}x{}".format(SECRET_LENGTH, HEIGHT, WIDTH),
    ]
    if N_SYMBOLS > 0:
        lines.append(
            "Reed-Solomon        : RS(n={}, k={}), "
            "corrects up to t={} symbol errors".format(n_bytes, K_BYTES, N_SYMBOLS // 2)
        )
    else:
        lines.append("install `reedsolo` ")

    lines.append(
        "Payload        : {:.5f} bpp (coded {:.5f} bpp) at {}x{}".format(
            PAYLOAD_BITS / (IMAGE_SIZE * IMAGE_SIZE),
            SECRET_LENGTH / (IMAGE_SIZE * IMAGE_SIZE),
            IMAGE_SIZE,
            IMAGE_SIZE,
        )
    )    
    return "\n".join(lines)


def ensure_save_path():
    os.makedirs(SAVE_PATH, exist_ok=True)
