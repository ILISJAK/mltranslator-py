from flask import Flask, request, jsonify, render_template
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import torch
import time
import tracemalloc

app = Flask(__name__)

model_name = "./results"
tokenizer = MBart50Tokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')

def translate_text(text, src_lang, tgt_lang, max_length=512):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to('cuda' if torch.cuda.is_available() else 'cpu')
    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang], max_length=max_length)
    translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translation

def translate_long_text(text, src_lang, tgt_lang, max_length=512):
    sentences = text.split('. ')
    translations = []

    for sentence in sentences:
        if sentence.strip():
            translation = translate_text(sentence.strip() + '.', src_lang, tgt_lang, max_length)
            translations.append(translation)
    
    return ' '.join(translations)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    text = data.get('text')
    src_lang = data.get('src_lang')
    tgt_lang = data.get('tgt_lang')

    start_time = time.time()
    tracemalloc.start()

    translation = translate_long_text(text, src_lang, tgt_lang)

    inference_time = time.time() - start_time
    translation_length = len(translation.split())
    current, peak = tracemalloc.get_traced_memory()
    memory_usage = peak / 10**6  # Convert to MB
    tracemalloc.stop()

    print(f"Inference Time: {inference_time:.4f} seconds")
    print(f"Translation Length: {translation_length} words")
    print(f"Memory Usage: {memory_usage:.4f} MB")

    return jsonify({'translation': translation})

if __name__ == "__main__":
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"An error occurred: {e}")
