from transformers import MBartForConditionalGeneration, MBart50Tokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
import torch
import numpy as np

def main():
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Load the datasets
    train_dataset = load_from_disk("data/train_dataset").select(range(1000))  # Use a smaller subset for quick verification
    test_dataset = load_from_disk("data/test_dataset").select(range(200))    # Use a smaller subset for quick verification

    # Load the tokenizer and model for mBART
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

    # Set the tokenizer source and target language codes
    tokenizer.src_lang = "hr_HR"  # Croatian
    tokenizer.tgt_lang = "en_XX"  # English

    # Encode the data
    def encode_data(examples):
        source_texts = [example['hr'] for example in examples['translation']]
        target_texts = [example['en'] for example in examples['translation']]
        source_encodings = tokenizer(source_texts, truncation=True, padding=True, max_length=128)
        target_encodings = tokenizer(text_target=target_texts, truncation=True, padding=True, max_length=128)
        
        encodings = {
            'input_ids': source_encodings['input_ids'],
            'attention_mask': source_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
        return encodings

    print("Mapping train dataset...")
    train_dataset = train_dataset.map(encode_data, batched=True)

    print("Mapping test dataset...")
    test_dataset = test_dataset.map(encode_data, batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Custom data collator to avoid slow tensor creation from list of numpy arrays
    def custom_data_collator(features):
        batch = {}
        first = features[0]
        for k, v in first.items():
            if k not in ("label", "label_ids", "labels") and torch.is_tensor(v):
                batch[k] = torch.stack([f[k] for f in features])
            elif k in ("label", "label_ids", "labels"):
                batch[k] = torch.tensor(np.array([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        return batch

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,  # Reduced batch size
        per_device_eval_batch_size=4,   # Reduced batch size
        num_train_epochs=1,             # For quick verification
        weight_decay=0.01,
        no_cuda=not torch.cuda.is_available(),  # Set to False to utilize GPU if available
        fp16=True,                     # Enable mixed precision training
        logging_dir='./logs',
        logging_steps=10
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=custom_data_collator
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained("./results")
    tokenizer.save_pretrained("./results")

if __name__ == "__main__":
    main()
