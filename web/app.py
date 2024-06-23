from flask import Flask, render_template, request
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

# Load the tokenizers and models
tokenizer_hr_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-hr-en")
model_hr_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-hr-en")

tokenizer_en_hr = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hr")
model_en_hr = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hr")

def translate(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated_tokens = model.generate(**inputs)
    translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translation

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        direction = request.form["direction"]
        if direction == "hr-en":
            translation = translate(text, tokenizer_hr_en, model_hr_en)
        else:
            translation = translate(text, tokenizer_en_hr, model_en_hr)
        return render_template("index.html", translation=translation, text=text, direction=direction)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
