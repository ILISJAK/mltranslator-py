from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from datasets import load_from_disk
import torch

# Load the model and tokenizer
model_name = "./results"
tokenizer = MBart50Tokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')

# Load the test dataset
test_dataset = load_from_disk("data/test_dataset").select(range(200))

# Set the tokenizer source and target language codes
tokenizer.src_lang = "hr_HR"  # Croatian
tokenizer.tgt_lang = "en_XX"  # English

# Function to generate translations
def generate_translation(batch):
    source_texts = [example['hr'] for example in batch["translation"]]
    inputs = tokenizer(source_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda' if torch.cuda.is_available() else 'cpu')
    translated = model.generate(**inputs)
    batch["predicted_translation"] = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return batch

# Map the generate_translation function to the test dataset
results = test_dataset.map(generate_translation, batched=True, batch_size=4)

# Print some example translations
for i in range(5):
    print(f"Source: {results[i]['translation']['hr']}")
    print(f"Target: {results[i]['translation']['en']}")
    print(f"Predicted: {results[i]['predicted_translation']}")
    print("-----")

# Save the results
results.to_json("results/evaluation.json", orient="records", lines=True)
