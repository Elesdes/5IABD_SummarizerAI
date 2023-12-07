import re


def clean_text(input_text):
    cleaned_text = re.sub(r"<s>", "", input_text)
    cleaned_text = " ".join(cleaned_text.split())
    return re.sub(r"\. (\w)", lambda match: f". {match.group(1).title()}", cleaned_text)
