import neattext.functions as nfx

def clean_text(text):
    text = nfx.remove_stopwords(text)
    text = nfx.remove_special_characters(text)
    text = nfx.remove_emojis(text)
    return text.lower()
