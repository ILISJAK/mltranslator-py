import os
from transformers import (
    MBartForConditionalGeneration,
    MBart50Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_from_disk
import torch
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from analysis import plot_metrics, CustomCallback  # Importing from analysis.py
import numpy as np
import warnings

# Suppress specific deprecation warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`",
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Ensure datasets are on SSD for faster access
    data_dir = "data"  # Path to your data directory
    train_dataset = load_from_disk("data/train_dataset").select(range(1000))  # Use a smaller subset for quick verification
    test_dataset = load_from_disk("data/test_dataset").select(range(200))    # Use a smaller subset for quick verification

    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

    # Set the tokenizer source and target language codes
    tokenizer.src_lang = "hr_HR"  # Croatian
    tokenizer.tgt_lang = "en_XX"  # English

    def encode_data(examples):
        source_texts = [example['hr'] for example in examples['translation']]
        target_texts = [example['en'] for example in examples['translation']]
        source_encodings = tokenizer(source_texts, truncation=True, padding=True, max_length=128)
        tokenizer.tgt_lang = "en_XX"  # Ensure the target language is set before encoding targets
        target_encodings = tokenizer(text_target=target_texts, truncation=True, padding=True, max_length=128)

        encodings = {
            'input_ids': source_encodings['input_ids'],
            'attention_mask': source_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
        return encodings

    print("Mapping train dataset...")
    train_dataset = train_dataset.map(encode_data, batched=True, num_proc=1)  # Disable multiprocessing

    print("Mapping test dataset...")
    test_dataset = test_dataset.map(encode_data, batched=True, num_proc=1)  # Disable multiprocessing

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Using DataCollatorForSeq2Seq to handle padding
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="no",  # Disable evaluation during training
        learning_rate=2e-5,
        per_device_train_batch_size=4,  # Adjusted batch size
        per_device_eval_batch_size=4,
        num_train_epochs=1,  # Adjusted epochs
        weight_decay=0.01,
        no_cuda=not torch.cuda.is_available(),  # Set to False to utilize GPU if available
        fp16=True,  # Enable mixed precision training
        logging_dir='./logs',
        logging_steps=50,
        save_steps=500,  # Save checkpoints more frequently
        save_total_limit=2,  # Limit the total number of checkpoints
        dataloader_num_workers=0,  # Disable multiprocessing for data loading
    )

    custom_callback = CustomCallback()

    # Using PyTorch's AdamW optimizer
    def get_optimizer(model):
        return AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    # Trainer with custom compute_metrics function
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[custom_callback],
        optimizers=(get_optimizer(model), None),
    )

    # Run a small training step to check for issues
    try:
        sanity_check_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="no",  # No evaluation during sanity check
            learning_rate=5e-5,
            per_device_train_batch_size=2,
            num_train_epochs=1,
            weight_decay=0.01,
            save_steps=10,
            max_steps=5,  # Limit to 5 steps for a quick sanity check
            fp16=True,
            logging_dir="./logs",
            logging_steps=10,
            gradient_accumulation_steps=1,
            report_to="none",
        )

        sanity_check_trainer = Trainer(
            model=model,
            args=sanity_check_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            optimizers=(get_optimizer(model), None),
        )

        sanity_check_trainer.train()
        print("Sanity check passed: Training loop and metrics work correctly.")
    except Exception as e:
        print(f"Sanity check failed: {e}")
        return

    # Train the model
    trainer.train()

    print("Saving model and tokenizer...")
    model.save_pretrained("./results")
    tokenizer.save_pretrained("./results")

    # Plot and save the training metrics
    logs = custom_callback.get_logs()
    print("Training logs:", logs)  # Print logs to verify keys
    try:
        plot_metrics(logs)
        print("Metrics plotting successful.")
    except KeyError as e:
        print(f"Metrics plotting failed due to missing key: {e}")

if __name__ == "__main__":
    main()
