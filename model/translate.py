from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import torch

def translate_text(text, tokenizer, model, src_lang, tgt_lang, max_length=512):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang], max_length=max_length)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    return translated_text

def translate_long_text(text, tokenizer, model, src_lang, tgt_lang, max_length=512):
    sentences = text.split('. ')
    translations = []

    for sentence in sentences:
        if sentence.strip():
            translation = translate_text(sentence.strip() + '.', tokenizer, model, src_lang, tgt_lang, max_length)
            translations.append(translation)

    return ' '.join(translations)

def main():
    tokenizer = MBart50Tokenizer.from_pretrained("./results")
    model = MBartForConditionalGeneration.from_pretrained("./results").to('cuda' if torch.cuda.is_available() else 'cpu')

    source_text = "Ako slika vrijedi tisuću riječi, koliko vrijedi ljudima koji ne vide? Bez riječi, osobama s posebnim potrebama u vidu lako je propustiti ključne informacije ili biti frustrirani iskustvom. Zamjenski tekst (zamjenski tekst) opisni je tekst koji prenosi značenje i kontekst vizualne stavke u digitalnoj postaci, npr. na aplikaciji ili web-stranici. Kada čitači zaslona Microsoft pripovjedač, JAWS i NVDA dostignu sadržaj pomoću zamjenskog teksta, zamjenski tekst čita se naglas da bi korisnici mogli bolje razumjeti što se nalazi na zaslonu. Dobro napisan, opisni zamjenski tekst dramatično smanjuje dvosmislenost i poboljšava korisničko iskustvo. U ovoj se temi opisuje razumijevanje, pisanje i korištenje učinkovitog zamjenskog teksta u Microsoft 365 proizvodima."
    src_lang = "hr_HR"
    tgt_lang = "en_XX"

    translation_en = translate_long_text(source_text, tokenizer, model, src_lang, tgt_lang)
    print(f"Croatian: {source_text} -> English: {translation_en}")

if __name__ == "__main__":
    main()
