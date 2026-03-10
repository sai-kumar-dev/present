import os
import platform
import subprocess
import torch

print("\n==============================")
print("SYSTEM ENVIRONMENT DETAILS")
print("==============================\n")

# ------------------------------------------------
# OS INFORMATION
# ------------------------------------------------

print("Operating System:")
print(platform.platform())
print()

# ------------------------------------------------
# CPU INFORMATION
# ------------------------------------------------

print("CPU Information:")

try:
    cpu_info = subprocess.check_output("lscpu", shell=True).decode()
    for line in cpu_info.split("\n"):
        if any(k in line for k in [
            "Model name",
            "Socket",
            "Core(s) per socket",
            "CPU(s)"
        ]):
            print(line)
except:
    print("Unable to fetch CPU info")

print()

# ------------------------------------------------
# MEMORY
# ------------------------------------------------

print("Memory Information:")

try:
    mem = subprocess.check_output("free -h", shell=True).decode()
    print(mem)
except:
    print("Unable to fetch memory info")

print()

# ------------------------------------------------
# GPU INFORMATION
# ------------------------------------------------

print("GPU Information:")

try:
    gpu = subprocess.check_output("nvidia-smi", shell=True).decode()
    print(gpu)
except:
    print("No NVIDIA GPU detected")

print()

# ------------------------------------------------
# PYTHON ENVIRONMENT
# ------------------------------------------------

print("Python Environment:")
print("Python version:", platform.python_version())
print()

# ------------------------------------------------
# PYTORCH
# ------------------------------------------------

print("PyTorch Information:")

try:
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU device:", torch.cuda.get_device_name(0))
        print("GPU count:", torch.cuda.device_count())

except:
    print("PyTorch not available")

print()

# ------------------------------------------------
# DISK
# ------------------------------------------------

print("Disk Usage:")

try:
    disk = subprocess.check_output("df -h", shell=True).decode()
    print(disk)
except:
    print("Unable to fetch disk info")

print("\n==============================")
print("END OF SYSTEM REPORT")
print("==============================")