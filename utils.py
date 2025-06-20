# import neattext.functions as nfx

# def clean_text(text):
#     text = nfx.remove_stopwords(text)
#     text = nfx.remove_special_characters(text)
#     text = nfx.remove_emojis(text)
#     return text.lower()

# utils.py
import neattext.functions as nfx

def clean_text(text):
    text = text.lower()
    text = nfx.remove_userhandles(text)
    text = nfx.remove_urls(text)
    text = nfx.remove_punctuations(text)
    text = nfx.remove_stopwords(text)
    return text.strip()
