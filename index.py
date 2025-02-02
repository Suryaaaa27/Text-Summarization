import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize
import pickle

# Load dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Load tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Set device (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Preprocessing function
def preprocess_data(example):
    text = example["article"]
    sentences = sent_tokenize(text)
    truncated_text = " ".join(sentences[:10])  # Limit sentence count to fit model constraints
    return {"truncated_text": truncated_text}

# Apply preprocessing
dataset = dataset.map(preprocess_data)

# Select a sample article
sample_text = dataset["train"][0]["truncated_text"]

# Tokenize and truncate input
inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=512).to(device)

# Generate summary
summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=50, num_beams=4, length_penalty=2.0, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("\n🔹 **Extractive Summary:**", dataset["train"][0]["highlights"])
print("\n🔹 **Abstractive Summary:**", summary)
from transformers import pipeline

# Specify the model explicitly (e.g., BART, T5, etc.)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def summarize_text(text):
    # Summarize the text using the specified model
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    # Save the function using pickle

    return summary[0]['summary_text']

with open("summarization.pkl", "wb") as f:
    pickle.dump(summarizer, f)
# Extract the summary from the result
