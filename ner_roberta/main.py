# main.py
import sys
from scripts.data_preprocessing import read_bc5cdr_file
from scripts.data_format import convert_bc5cdr_to_tag_format_split_sentences
from scripts.training import roberta_training
from scripts.inference import predict_entities

def main():
    # 1. Define file paths for train, dev, and test
    train_file = "/home/spv/ner/bc5cdr_rb/data/bc5cdr/CDR_TrainingSet.PubTator.txt"
    dev_file = "/home/spv/ner/bc5cdr_rb/data/bc5cdr/CDR_DevelopmentSet.PubTator.txt"
    test_file = "/home/spv/ner/bc5cdr_rb/data/bc5cdr/CDR_TestSet.PubTator.txt"
    sys.stdout = open("results/training_log.txt", "w")
    
    # 2. Read each split
    train_data = read_bc5cdr_file(train_file)
    dev_data = read_bc5cdr_file(dev_file)
    test_data = read_bc5cdr_file(test_file)

    # 3. Convert each split into sentence-level tokens/tags
    train_outputs = []
    for entry in train_data:
        train_outputs.extend(convert_bc5cdr_to_tag_format_split_sentences(entry))

    dev_outputs = []
    for entry in dev_data:
        dev_outputs.extend(convert_bc5cdr_to_tag_format_split_sentences(entry))

    test_outputs = []
    for entry in test_data:
        test_outputs.extend(convert_bc5cdr_to_tag_format_split_sentences(entry))

    # 4. Train Roberta model with these splits
    roberta_training(
        train_outputs, 
        dev_outputs, 
        test_outputs, 
        save_directory="models/saved_models/roberta_bc5cdr"
    )

    # 5. Inference on a sample text
    test_text = "Analgesic effect of intravenous ketamine in cancer patients on morphine therapy ..."
    predictions = predict_entities(test_text, "models/saved_models/roberta_bc5cdr")
    print(predictions)

if __name__ == "__main__":
    main()
