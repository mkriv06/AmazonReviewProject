import re

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"<.*?>", "", text) # remove html tags

    text = re.sub(r"http\S+|www\S+", "", text) # remove links

    return text.strip()

