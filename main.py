from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch

# 🔹 Check if GPU (CUDA) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# 🔹 Load dataset
print("🔹 Loading IMDb dataset...")
dataset = load_dataset("imdb")

# 🔹 Load model and tokenizer
print("🔹 Loading DistilBERT...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
model.to(device)  # ✅ Move model to GPU if available

# 🔹 Tokenize data
def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=256)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)

# 🔹 Select a single example for quick testing
sample = tokenized_dataset["test"][0]
inputs = {k: torch.tensor([v]).to(device) for k, v in sample.items() if k in tokenizer.model_input_names}

# 🔹 Run inference
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()

print(f"🔹 Predicted sentiment: {'Positive' if prediction == 1 else 'Negative'}")
