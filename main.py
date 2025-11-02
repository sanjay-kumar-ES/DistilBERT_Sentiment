from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from sklearn.metrics import accuracy_score

#  Load IMDb dataset.
print(" Loading IMDb dataset...")
dataset = load_dataset("imdb")

# Use a smaller subset for quick training.
train_dataset = dataset["train"].select(range(2000))
test_dataset = dataset["test"].select(range(500))

#  Load tokenizer and model.
print(" Loading DistilBERT...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

#  Tokenize text data.
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

print(" Tokenizing data...")
train_enc = train_dataset.map(tokenize, batched=True)
test_enc = test_dataset.map(tokenize, batched=True)

train_enc = train_enc.rename_column("label", "labels")
test_enc = test_enc.rename_column("label", "labels")

train_enc.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_enc.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

#  Define evaluation metric.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

#  Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",  #  Works for transformers >= 4.10
    logging_dir="./logs",
    save_strategy="no"
)

#  Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_enc,
    eval_dataset=test_enc,
    compute_metrics=compute_metrics
)

#  Train and evaluate
print(" Training model...")
trainer.train()

print(" Evaluating model...")
results = trainer.evaluate()
print(f"\n✅ Accuracy: {results['eval_accuracy']:.2f}")

#  Custom test.
text = "This movie was absolutely wonderful and the acting was brilliant!"
tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
outputs = model(**tokens)
pred = torch.argmax(outputs.logits, dim=1).item()
print("\n🎬 Sentiment:", "POSITIVE" if pred == 1 else "NEGATIVE")
