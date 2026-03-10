#!/usr/bin/env python3

import os
import logging
import psutil
import gc
import torch
from datetime import datetime


def system_stats():
    """Return CPU and RAM usage"""


    mem = psutil.virtual_memory()

    cpu = psutil.cpu_percent()
    used = mem.used / 1e9
    total = mem.total / 1e9

    return f"CPU {cpu:.1f}% | RAM {used:.2f}/{total:.2f} GB"


class SystemFormatter(logging.Formatter):
    """Formatter that injects system diagnostics into every log"""


    def format(self, record):

        base = super().format(record)

        stats = system_stats()

        return f"{base} | {stats}"


def create_logger(run_dir=None):


    logger = logging.getLogger("ERA5_PINN")

    # prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = SystemFormatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # console output
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # file output
    if run_dir:

        log_dir = os.path.join(run_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(
            os.path.join(log_dir, "training.log")
        )

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def clear_memory():

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# global logger placeholder

logger = logging.getLogger("ERA5_PINN")
