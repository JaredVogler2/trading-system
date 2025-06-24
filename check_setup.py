# check_setup.py
import torch
import pandas as pd
import numpy as np

print("Python packages installed successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")