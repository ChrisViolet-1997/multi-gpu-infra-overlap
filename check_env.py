#!/usr/bin/env python3
"""Quick environment check for TP overlap experiments."""

import sys

def check_environment():
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not found")
        return False
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA: {torch.version.cuda}")
        print(f"✓ GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("✗ CUDA not available")
        return False
    
    # Check distributed
    if torch.distributed.is_available():
        print(f"✓ Distributed: Available")
        print(f"✓ NCCL: {torch.distributed.is_nccl_available()}")
    else:
        print("✗ Distributed not available")
        return False
    
    print("=" * 60)
    print("✓ Environment ready for experiments")
    print("=" * 60)
    return True

if __name__ == "__main__":
    if not check_environment():
        sys.exit(1)
