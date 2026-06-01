"""CUDA sanity check. Prints the log lines the spec expects.

    Using device: cuda
    GPU Detected: NVIDIA GeForce RTX 4060 Ti

Run:  python scripts/check_gpu.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402


def main() -> int:
    cuda = torch.cuda.is_available()
    print(f"Using device: {'cuda' if cuda else 'cpu'}")
    print(f"torch: {torch.__version__} | built with CUDA: {torch.version.cuda}")
    if cuda:
        props = torch.cuda.get_device_properties(0)
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {props.total_memory / 1e9:.1f} GB | "
              f"SMs: {props.multi_processor_count} | "
              f"capability: {props.major}.{props.minor}")
        # quick mixed-precision smoke test
        x = torch.randn(1024, 1024, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            _ = x @ x
        torch.cuda.synchronize()
        print("AMP matmul on GPU: OK")
        return 0
    print("WARNING: CUDA not available. Training requires an NVIDIA GPU "
          "(RTX 4060 Ti target). Inference can fall back to CPU with REQUIRE_CUDA=0.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
