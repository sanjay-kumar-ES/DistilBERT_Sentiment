import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 🔹 Check device availability
cpu = torch.device("cpu")
gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🧠 PyTorch version: {torch.__version__}")
print(f"🖥️ GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# 🔹 Load DistilBERT model and tokenizer
print("\n🔹 Loading DistilBERT model and tokenizer...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()  # set to evaluation mode

# 🔹 Create a batch of text inputs
texts = [
    "This movie was absolutely fantastic! I loved it.",
    "The film was terrible, a complete waste of time.",
    "An average movie with some good and bad moments.",
    "The acting was brilliant but the plot was boring.",
    "A masterpiece of storytelling and cinematography!"
] * 100  # replicate 100 times for a fair benchmark

inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)

def run_inference(device):
    """Run inference on given device and measure time."""
    local_model = model.to(device)
    local_inputs = {k: v.to(device) for k, v in inputs.items()}

    # Warmup (ignore first pass)
    with torch.no_grad():
        _ = local_model(**local_inputs)

    # Timed runs
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()
    with torch.no_grad():
        _ = local_model(**local_inputs)
    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.time()
    return end - start

# 🔹 CPU Benchmark
print("\n🔹 Running on CPU...")
cpu_time = run_inference(cpu)
print(f"CPU inference time: {cpu_time:.3f} sec")

# 🔹 GPU Benchmark (if available)
if gpu.type == "cuda":
    print("\n🔹 Running on GPU...")
    gpu_time = run_inference(gpu)
    print(f"GPU inference time: {gpu_time:.3f} sec")
    print(f"\n🚀 Speedup: {cpu_time / gpu_time:.2f}x faster on GPU")
else:
    print("\n⚠️ GPU not available — only CPU benchmarked.")
