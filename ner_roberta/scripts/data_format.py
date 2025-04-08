from .data_preprocessing import split_into_sentences, process_sentence

def convert_bc5cdr_to_tag_format_split_sentences(entry):
    """
    entry: a dict with keys ['text', 'annotations'], etc.
    Returns a list of dictionaries, each containing {"tokens": [...], "tags": [...]}.
    """
    text = entry["text"]
    sentences = split_into_sentences(text)
    sentence_outputs = []
    for sentence_text, sent_start, sent_end in sentences:
        sentence_output = process_sentence(sentence_text, sent_start, entry["annotations"])
        sentence_outputs.append(sentence_output)
    return sentence_outputs
