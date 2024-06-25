from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import torch

def translate_batch(texts, tokenizer, model, src_lang, tgt_lang):
    # Set the tokenizer source and target language codes
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    # Tokenize the input texts
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Move the inputs to the same device as the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate translations with forced_bos_token_id
    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
    translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    
    return translations

def main():
    # Load the tokenizer and model from the saved directory
    model_name = "./results"
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Example batch of English sentences to translate to Croatian
    texts_en = ["How are you?", "What is your name?", "Where are you from?"]

    # Translate English to Croatian
    translations_hr = translate_batch(texts_en, tokenizer, model, src_lang="en_XX", tgt_lang="hr_HR")
    for text, translation in zip(texts_en, translations_hr):
        print(f"English: {text}")
        print(f"Croatian: {translation}")
    
    # Translate Croatian back to English to verify
    translations_en = translate_batch(translations_hr, tokenizer, model, src_lang="hr_HR", tgt_lang="en_XX")
    for text, translation in zip(translations_hr, translations_en):
        print(f"Croatian: {text}")
        print(f"English: {translation}")

if __name__ == "__main__":
    main()
