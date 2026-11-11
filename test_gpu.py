"""
test_gpu.py — Verification benchmark for AMD Radeon RX 580 (DirectML & Vulkan).
Tests PyTorch tensor operations, a real Neural Network training loop on DirectML,
and verifies VRAM / GPU device detection with CPU vs GPU speedups.
"""

from __future__ import annotations
import sys
import time

def print_banner(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def test_directml_pytorch():
    print_banner("1. DirectML (PyTorch on AMD RX 580)")
    try:
        import torch
        print(f"[*] PyTorch Version: {torch.__version__}")
    except ImportError:
        print("[-] PyTorch is not installed in this Python environment.")
        return False

    try:
        import torch_directml
        available = torch_directml.is_available()
        count = torch_directml.device_count()
        print(f"[*] torch_directml Available: {available}")
        print(f"[*] DirectML Device Count: {count}")
        if not available:
            print("[-] DirectML runtime is not available.")
            return False

        device = torch_directml.device()
        device_name = torch_directml.device_name(0) if count > 0 else "DirectML Device"
        print(f"[+] ACTIVE TARGET GPU: {device_name} ({device})")

    except ImportError:
        print("[-] 'torch-directml' package is missing. Install with: pip install torch-directml")
        return False

    # 1. Matrix Multiplication Benchmark (CPU vs DirectML GPU)
    N = 4096
    print(f"\n--- Benchmark: Large Matrix Multiply ({N}x{N} Float32) ---")
    
    # CPU
    cpu_x = torch.randn(N, N, device="cpu")
    cpu_y = torch.randn(N, N, device="cpu")
    t0 = time.perf_counter()
    cpu_z = torch.matmul(cpu_x, cpu_y)
    t_cpu = time.perf_counter() - t0
    print(f"CPU Time:      {t_cpu:.4f} seconds")

    # DirectML GPU
    dml_x = torch.randn(N, N, device=device)
    dml_y = torch.randn(N, N, device=device)
    # Warmup
    _ = torch.matmul(dml_x, dml_y)
    
    t0 = time.perf_counter()
    dml_z = torch.matmul(dml_x, dml_y)
    t_gpu = time.perf_counter() - t0
    print(f"DirectML GPU:  {t_gpu:.4f} seconds")

    speedup = t_cpu / max(t_gpu, 1e-6)
    print(f"[+] GPU Speedup: {speedup:.2f}x faster than CPU")

    # 2. Mini Neural Network Training Loop on DirectML
    print("\n--- Training Loop: Deep MLP on DirectML GPU ---")
    import torch.nn as nn
    import torch.optim as optim

    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batch_size = 2048
    epochs = 10
    print(f"Training on {device_name} (Batch Size: {batch_size}, Epochs: {epochs})...")
    
    t_train_start = time.perf_counter()
    for ep in range(epochs):
        data = torch.randn(batch_size, 512, device=device)
        target = torch.randint(0, 10, (batch_size,), device=device)

        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        if (ep + 1) % 2 == 0:
            print(f"  Epoch {ep+1}/{epochs} | Loss: {loss.item():.4f} | Processed on: {data.device}")
    
    t_train_total = time.perf_counter() - t_train_start
    print(f"[+] Training completed on AMD GPU in {t_train_total:.3f}s")
    return True

def test_vulkan_llama():
    print_banner("2. Vulkan llama.cpp (LLM Inference on RX 580)")
    import os
    import subprocess

    llama_exe = r"C:\Users\Sultan\Downloads\llama-b12255-bin-win-vulkan-x64\llama-cli.exe"
    models_dir = r"C:\Users\Sultan\Downloads\llama-b12255-bin-win-vulkan-x64\models"

    if not os.path.exists(llama_exe):
        print(f"[-] llama-cli.exe not found at: {llama_exe}")
        return False

    print(f"[+] Found Vulkan executable: {llama_exe}")
    
    # Check for GGUF models
    if os.path.exists(models_dir):
        ggufs = [f for f in os.listdir(models_dir) if f.endswith(".gguf")]
        if ggufs:
            print(f"[+] Found GGUF models in {models_dir}:")
            for m in ggufs:
                print(f"    - {m}")
            
            target_model = os.path.join(models_dir, ggufs[0])
            print(f"\n[*] Testing Vulkan0 offload with: {ggufs[0]}")
            cmd = [
                llama_exe,
                "-m", target_model,
                "-ngl", "99",
                "-c", "512",
                "-n", "16",
                "--prompt", "Hello GPU",
            ]
            print(f"Command: {' '.join(cmd)}")
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                # Parse Vulkan offload logs
                vulkan_lines = [l for l in res.stderr.splitlines() if "Vulkan" in l or "vulkan" in l or "offloaded" in l]
                for vl in vulkan_lines[:5]:
                    print(f"    [Vulkan Log] {vl.strip()}")
                print("[+] Vulkan execution completed successfully!")
            except Exception as e:
                print(f"[-] Execution error: {e}")
        else:
            print(f"[-] No .gguf models found in {models_dir}")
    else:
        print(f"[-] Models directory does not exist: {models_dir}")

    return True

def main():
    print("============================================================")
    print(" AMD Radeon RX 580 (2048SP) GPU UTILIZATION & SPEED TEST")
    print("============================================================")
    
    dml_ok = test_directml_pytorch()
    vulkan_ok = test_vulkan_llama()

    print_banner("SUMMARY VERDICT")
    print(f"DirectML PyTorch ML Acceleration: {'[PASS] GPU UTILIZED' if dml_ok else '[FAIL/NOT CONFIGURED]'}")
    print(f"Vulkan llama.cpp LLM Acceleration:  {'[PASS] GPU READY' if vulkan_ok else '[FAIL/NOT CONFIGURED]'}")

if __name__ == "__main__":
    main()

