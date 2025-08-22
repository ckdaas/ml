from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
print(type(tokenizer))

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True,
                   truncation=True, return_tensors="pt")
print("Tokenized inputs:")
print(inputs)
print("")

# Using a model with a sequence classification head
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print("Shape of the model output:")
print(outputs.logits.shape)
print("")

# Print the raw output logits
print("Raw output logits:")
print(outputs.logits)
print("")

# Applying a SoftMax on the output to get probabilities
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print("Answer")
print(predictions)
print("")

print("Model labels corresponding to IDs:")
print(model.config.id2label)
print("")
