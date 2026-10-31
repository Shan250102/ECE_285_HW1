# All settings live here. Edit this file, then run train.py / evaluate.py.

DATA_ROOT = "/home/sananth/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray"     
OUT_DIR = "runs/main"        
SYNTHETIC = False            

IMG_SIZE = 128               # images are resized to 128 x 128
LATENT_DIM = 256             # length of the latent vector z
BASE_CHANNELS = 32           # width of the first conv layer 

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
BETA = 0.85                  # weight of the KL term in the loss
WARMUP_EPOCHS = 10          
SEED = 0

N_GENERATED = 5184           
