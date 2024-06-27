from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from datasets import load_from_disk
import torch
import sacrebleu
import matplotlib.pyplot as plt
import pandas as pd

# Load the tokenizer and model
model_name = "./results/"
tokenizer = MBart50Tokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')

# Load the test dataset
test_dataset = load_from_disk("data/test_dataset").select(range(200))

# Set the tokenizer source and target language codes
tokenizer.src_lang = "hr_HR"
tokenizer.tgt_lang = "en_XX"

def generate_translation(batch):
    source_texts = [example['hr'] for example in batch["translation"]]
    inputs = tokenizer(source_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda' if torch.cuda.is_available() else 'cpu')
    translated = model.generate(**inputs, max_new_tokens=128)
    batch["predicted_translation"] = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return batch

# Generate translations
results = test_dataset.map(generate_translation, batched=True, batch_size=4)

# Extract references and predictions
references = [[example['translation']['en']] for example in test_dataset]  # SacreBLEU expects list of lists for references
predictions = [example['predicted_translation'] for example in results]

# Print some examples to debug
for i in range(5):
    print(f"Reference {i}: {references[i]}")
    print(f"Prediction {i}: {predictions[i]}")

# Calculate BLEU score
bleu = sacrebleu.corpus_bleu(predictions, references)

print(f"BLEU score: {bleu.score}")

# Save the results
results_df = pd.DataFrame({
    'source': [example['translation']['hr'] for example in results],
    'target': [example['translation']['en'] for example in results],
    'predicted': predictions
})
results_df.to_json("results/evaluation.json", orient="records", lines=True)

# Plotting
plt.figure(figsize=(10, 5))
plt.bar(['BLEU score'], [bleu.score])
plt.ylim(0, 100)
plt.ylabel('Score')
plt.title('BLEU Score')
plt.savefig("results/bleu_score.png")
plt.show()

for i in range(5):
    print(f"Source: {results[i]['translation']['hr']}")
    print(f"Target: {results[i]['translation']['en']}")
    print(f"Predicted: {results[i]['predicted_translation']}")
    print("-----")
