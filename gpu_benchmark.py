import torch
import time

def benchmark(device, n=10000):
    """Simple matrix multiply benchmark."""
    torch.manual_seed(0)
    x = torch.randn((1000, 1000), device=device)
    y = torch.randn((1000, 1000), device=device)

    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()

    for _ in range(n):
        z = torch.mm(x, y)
        _ = z.sum()

    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.time()
    return end - start

# Detect devices
cpu = torch.device("cpu")
gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🔹 CPU Benchmark Running...")
cpu_time = benchmark(cpu, n=300)

if gpu.type == "cuda":
    print("🔹 GPU Benchmark Running...")
    gpu_time = benchmark(gpu, n=300)
else:
    gpu_time = None

print("\n📊 Results:")
print(f"CPU time: {cpu_time:.3f} sec")
if gpu_time:
    print(f"GPU time: {gpu_time:.3f} sec")
    print(f"🚀 Speedup: {cpu_time / gpu_time:.2f}x faster on GPU")
else:
    print("⚠️ GPU not available — running on CPU only.")
