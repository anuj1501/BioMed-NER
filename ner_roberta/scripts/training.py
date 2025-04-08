# training.py

import numpy as np
import torch
from transformers import (
    RobertaTokenizerFast,
    RobertaForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report

def tokenize_and_align_labels(examples, tokenizer, label_names):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding=False,  # We'll handle padding via DataCollator
        is_split_into_words=True,
        return_offsets_mapping=True
    )
    
    labels_batch = examples["tags"]
    new_labels = []

    for i, labels in enumerate(labels_batch):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # special token => set label to -100
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_names.index(labels[word_idx]))
            else:
                # subword token => set label to -100
                label_ids.append(-100)
            previous_word_idx = word_idx
        new_labels.append(label_ids)
    
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def compute_metrics(p, label_names):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Convert predicted indices to label strings
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    # Convert label indices to label strings
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Flatten
    flat_predictions = sum(true_predictions, [])
    flat_labels = sum(true_labels, [])

    report = classification_report(
        flat_labels,
        flat_predictions,
        labels=label_names[1:],  # skip "O" if you want only entity metrics
        target_names=label_names[1:],
        output_dict=True,
        zero_division=0
    )

    metrics = {}
    # Add entity-wise precision/recall/f1
    for label in label_names[1:]:
        if label in report:
            metrics[f"{label}_precision"] = report[label]["precision"]
            metrics[f"{label}_recall"] = report[label]["recall"]
            metrics[f"{label}_f1"] = report[label]["f1-score"]

    # Macro averages
    metrics["macro_avg_precision"] = report["macro avg"]["precision"]
    metrics["macro_avg_recall"]    = report["macro avg"]["recall"]
    metrics["macro_avg_f1"]        = report["macro avg"]["f1-score"]

    # Weighted averages
    metrics["weighted_avg_precision"] = report["weighted avg"]["precision"]
    metrics["weighted_avg_recall"]    = report["weighted avg"]["recall"]
    metrics["weighted_avg_f1"]        = report["weighted avg"]["f1-score"]

    return metrics

def roberta_training(train_outputs, dev_outputs, test_outputs, save_directory="model_roberta"):
    """
    Train a RoBERTa model for NER on BC5CDR data.
    :param train_outputs: List of dicts ({"tokens": [...], "tags": [...]}) for training
    :param dev_outputs:   List of dicts ({"tokens": [...], "tags": [...]}) for development/validation
    :param test_outputs:  List of dicts ({"tokens": [...], "tags": [...]}) for testing
    :param save_directory: Where to save the model
    """

    label_names = ["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"]

    # Build separate datasets
    train_ds = Dataset.from_dict({
        'tokens': [x['tokens'] for x in train_outputs],
        'tags': [x['tags'] for x in train_outputs]
    })
    dev_ds = Dataset.from_dict({
        'tokens': [x['tokens'] for x in dev_outputs],
        'tags': [x['tags'] for x in dev_outputs]
    })
    test_ds = Dataset.from_dict({
        'tokens': [x['tokens'] for x in test_outputs],
        'tags': [x['tags'] for x in test_outputs]
    })

    # Combine into a single DatasetDict
    dataset_dict = DatasetDict({
        "train": train_ds,
        "validation": dev_ds,
        "test": test_ds
    })

    # Load tokenizer and model
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True, use_fast=True)
    model = RobertaForTokenClassification.from_pretrained(
        "roberta-base",
        num_labels=len(label_names),
        id2label={i: label for i, label in enumerate(label_names)},
        label2id={label: i for i, label in enumerate(label_names)}
    )

    # Map function for tokenization/alignment
    def map_fn(examples):
        return tokenize_and_align_labels(examples, tokenizer, label_names)

    # Tokenize all splits
    tokenized_data = dataset_dict.map(map_fn, batched=True)

    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        padding=True,
        label_pad_token_id=-100
    )

    # Training arguments

    training_args = TrainingArguments(
        output_dir="./results",                 # Model checkpoints will go here
        logging_dir="./results/logs",           # TensorBoard logs will be written here
        logging_steps=10,                       # Log every N steps
        # evaluation_strategy="epoch",            # Evaluate at each epoch
        save_strategy="epoch",                  # Save model at each epoch
        report_to="tensorboard",                # Enable TensorBoard logging
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        weight_decay=0.01,
        load_best_model_at_end=False,
    )


    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, label_names)
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(save_directory)
    tokenizer.save_pretrained(save_directory)

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=tokenized_data["test"])
    with open("results/test_metrics.txt", "w") as f:
        for k, v in test_results.items():
            f.write(f"{k}: {v:.4f}\n" if isinstance(v, float) else f"{k}: {v}\n")
    for k, v in test_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
