from flask import Flask, request, jsonify, render_template
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import torch

app = Flask(__name__)

# Load the tokenizer and model
model_name = "./results"
tokenizer = MBart50Tokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')

def translate_text(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda' if torch.cuda.is_available() else 'cpu')
    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translation

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    text = data.get('text')
    src_lang = data.get('src_lang')
    tgt_lang = data.get('tgt_lang')
    translation = translate_text(text, src_lang, tgt_lang)
    return jsonify({'translation': translation})

if __name__ == "__main__":
    app.run(debug=True)
