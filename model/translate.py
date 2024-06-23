from transformers import MarianMTModel, MarianTokenizer

# Load the tokenizer and model for Croatian to English
tokenizer_hr_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-hr-en")
model_hr_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-hr-en")

# Load the tokenizer and model for English to Croatian
tokenizer_en_hr = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hr")
model_en_hr = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hr")

def translate(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated_tokens = model.generate(**inputs)
    translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translation

# Example translation from Croatian to English
text_hr = "Kako si?"
translation_en = translate(text_hr, tokenizer_hr_en, model_hr_en)
print(f"Croatian: {text_hr}")
print(f"English: {translation_en}")

# Example translation from English to Croatian
text_en = "How are you?"
translation_hr = translate(text_en, tokenizer_en_hr, model_en_hr)
print(f"English: {text_en}")
print(f"Croatian: {translation_hr}")
