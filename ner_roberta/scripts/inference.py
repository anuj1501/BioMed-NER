import re
import torch
from transformers import RobertaForTokenClassification, RobertaTokenizerFast

def predict_entities(text, model_path, label_names=None):
    """
    Load a saved model + tokenizer from model_path, and predict entity tags on input text.
    :param text: raw text to annotate
    :param model_path: directory containing the saved model/tokenizer
    :param label_names: list of labels in the correct order
    :return: List of (word, predicted_label) pairs
    """
    if label_names is None:
        # Default to your label set
        label_names = ["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"]

    # Load model + tokenizer
    model = RobertaForTokenClassification.from_pretrained(model_path)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Basic regex-based word splitting
    words = re.findall(r"\d+\.\d+|\d+|\w+|[^\w\s]", text)

    encoded = tokenizer(
        words,
        is_split_into_words=True,
        return_offsets_mapping=True,
        return_tensors="pt"
    ).to(device)

    # Offset mappings to track subwords
    offset_mapping = encoded.pop('offset_mapping').squeeze().tolist()

    with torch.no_grad():
        outputs = model(**encoded)
        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()

    # Reconstruct predictions for each word
    results = []
    current_word = None
    word_ids = encoded.word_ids(batch_index=0)
    for idx, (word_id, pred) in enumerate(zip(word_ids, predictions)):
        if word_id is None or word_id == current_word:
            # Skip special tokens or subwords
            continue
        current_word = word_id
        results.append((words[word_id], label_names[pred]))

    return results
