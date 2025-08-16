import os

# Local model configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Default model, can be overridden
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 150
TIMEOUT = 30  # Timeout for generation in seconds

VALID_LABELS = ['Yes', 'No', 'To some extent']

NUM_CONVERSATIONS_TO_PROCESS = 1

# Device configuration
USE_GPU = True  # Set to False to force CPU usage
TORCH_DTYPE = "float16"  # Use "float32" for CPU or if you have memory issues
