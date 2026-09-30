#!/usr/bin/env python3
"""Run nnU-Net's normal prediction entry point and print PyTorch GPU peak."""

import torch
from nnunetv2.inference.predict_from_raw_data import predict_entry_point


if __name__ == "__main__":
    torch.cuda.reset_peak_memory_stats()
    predict_entry_point()
    torch.cuda.synchronize()
    print(f"CUDA_PEAK_RESERVED_MIB={torch.cuda.max_memory_reserved() / 2**20:.3f}", flush=True)
