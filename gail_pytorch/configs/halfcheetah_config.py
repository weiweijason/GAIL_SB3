"""Example configuration for GAIL training."""
import os

# Environment settings
ENV_NAME = "HalfCheetah-v4"  # Gym environment name

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EXPERT_DATA_PATH = os.path.join(DATA_DIR, "expert_trajectories", f"{ENV_NAME}_expert.pkl")
LOG_DIR = os.path.join(DATA_DIR, "logs", ENV_NAME)
MODEL_DIR = os.path.join(DATA_DIR, "models", ENV_NAME)

# Training settings
SEED = 42
TOTAL_TIMESTEPS = 1_000_000
MAX_EPISODE_LENGTH = 1000
POLICY_UPDATE_FREQ = 2048
DISC_UPDATE_FREQ = 1024

# Network settings
POLICY_LR = 3e-4
DISC_LR = 3e-4
GAMMA = 0.99
BATCH_SIZE = 64
ENTROPY_WEIGHT = 0.01
HIDDEN_DIM = 256
N_HIDDEN = 2

# Evaluation settings
EVAL_FREQ = 10000
N_EVAL_EPISODES = 10
SAVE_FREQ = 50000

# Device settings (use 'cuda' for GPU, 'cpu' for CPU)
DEVICE = "cuda"  # Change to 'cpu' if no GPU is available

# Make sure directories exist
os.makedirs(os.path.join(DATA_DIR, "expert_trajectories"), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)