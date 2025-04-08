import re

tag2id = {
    "O": 0,
    "B-Chemical": 1,
    "B-Disease": 2,
    "I-Disease": 3,
    "I-Chemical": 4
}

def read_bc5cdr_file(file_path):
    """
    Reads BC5CDR dataset file and returns data in structured format.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        entry = {"PMID": None, "text": None, "annotations": []}
        for line in file:
            line = line.strip()
            if not line:  # New entry starts
                if entry["PMID"]:
                    data.append(entry)
                    entry = {"PMID": None, "text": None, "annotations": []}
            elif "|t|" in line or "|a|" in line:  # Abstract text lines
                if entry["PMID"] is None:
                    entry["PMID"] = line.split("|")[0]
                if entry["text"] is None:
                    entry["text"] = line.split("|", 2)[2]
                else:
                    entry["text"] += " " + line.split("|", 2)[2]
            else:  # Annotation lines
                parts = line.split("\t")
                if len(parts) == 6:  # Ensure valid annotation
                    entry["annotations"].append({
                        "start": int(parts[1]),
                        "end": int(parts[2]),
                        "entity": parts[3],
                        "type": parts[4],
                    })
        if entry["PMID"]:
            data.append(entry)
    return data

def tokenize(text):
    # Basic regex tokenizer
    return re.findall(r'\w+|[^\w\s]', text)

def get_token_spans(text, tokens):
    spans = []
    offset = 0
    for token in tokens:
        start = text.find(token, offset)
        if start == -1:
            raise ValueError(f"Token {token} not found in text starting at {offset}")
        end = start + len(token)
        spans.append((start, end))
        offset = end
    return spans

def split_into_sentences(text):
    pattern = re.compile(r'(?<=\.)\s+')
    parts = pattern.split(text)
    sentences = []
    current_offset = 0
    for part in parts:
        part = part.strip()
        if not part:
            continue
        start = text.find(part, current_offset)
        end = start + len(part)
        sentences.append((part, start, end))
        current_offset = end
    return sentences

def process_sentence(sentence_text, sentence_offset, annotations):
    tokens = tokenize(sentence_text)
    token_spans = get_token_spans(sentence_text, tokens)
    adjusted_spans = [(start + sentence_offset, end + sentence_offset) for start, end in token_spans]
    tags = ["O"] * len(tokens)
    
    for ann in annotations:
        ann_start = ann["start"]
        ann_end = ann["end"]
        ann_type = ann["type"]  
        # Check if annotation lies fully within this sentence
        if ann_start >= sentence_offset and ann_end <= sentence_offset + len(sentence_text):
            for i, (token_start, token_end) in enumerate(adjusted_spans):
                if token_start >= ann_start and token_end <= ann_end:
                    if token_start == ann_start:
                        tags[i] = "B-" + ann_type
                    else:
                        tags[i] = "I-" + ann_type
    return {"tokens": tokens, "tags": tags}
