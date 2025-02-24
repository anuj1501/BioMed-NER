import time
import torch
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
from utils import html_parsing_ncbi, html_parsing_n2c2, get_classification_report, get_digit, get_macro_average_f1, label2digit

from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("aaditya/OpenBioLLM-Llama3-8B")
model = AutoModelForCausalLM.from_pretrained("aaditya/OpenBioLLM-Llama3-8B",
                                             torch_dtype=torch.float16,
                                             device_map="auto")
model.gradient_checkpointing_enable()

chat_template = open('../chat_templates/chat_templates/mistral-instruct.jinja').read()
chat_template = chat_template.replace('    ', '').replace('\n', '')
tokenizer.chat_template = chat_template

ncbi_df = pd.read_csv('data/NER/NCBI-disease/test_200.csv')
ncbi_example_df = pd.read_csv('data/NER/NCBI-disease/examples.csv')

system_message = """You are a helpful assistant to perform the following task.
"TASK: the task is to extract disease entities in a sentence."
"INPUT: the input is a sentence."
"OUTPUT: the output is an HTML that highlights all the disease entities in the sentence. The highlighting should only use HTML tags <span style=\"background-color: #FFFF00\"> and </span> and no other tags."
"""

def get_ner_ncbi_disease(sentence: str, shot: int = 0) -> str:
    """
    Get the NER results of NCBI-disease dataset from few-shot prompting.
    Args:
        sentence: the input sentence
        shot: the number of few-shot examples
    Returns:
        the NER results
    """
    messages = [{'role': 'system', 'content': system_message}]
    for i in range(shot):
        messages.append({'role': 'user', 'content': ncbi_example_df.iloc[i]['text']}) 
        messages.append({'role': 'assistant', 'content': ncbi_example_df.iloc[i]['label_text']})
    messages.append({'role': 'user', 'content': sentence})

    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

    time_start = time.time()
    outputs = model.generate(input_ids, max_new_tokens=1024)
    time_end = time.time()
    return tokenizer.decode(outputs[0], skip_special_tokens=True), time_end - time_start

def mistral_parser_ner(text: str):
    """
    Parse the text generation output.
    """
    # find the last "[/INST] " and use the text after it
    cleaned_text = text.split('[/INST] ')[-1]
    # find the first "\n" and use the text before it
    cleaned_text = cleaned_text.split('\n\n')[0]
    return cleaned_text

for i in tqdm(range(0, len(ncbi_df), 1)):
    ncbi_df.loc[i, 'html_mistral_8x7b_instruct_one_shot'], ncbi_df.loc[i, 'mistral_8x7b_instruct_one_shot_time'] = get_ner_ncbi_disease(ncbi_df.loc[i, 'text'], 1)
#     ncbi_df.loc[i, 'html_mistral_8x7b_instruct_five_shot'], ncbi_df.loc[i, 'mistral_8x7b_instruct_five_shot_time'] = get_ner_ncbi_disease(ncbi_df.loc[i, 'text'], 5)
#     ncbi_df.loc[i, 'html_mistral_8x7b_instruct_ten_shot'], ncbi_df.loc[i, 'mistral_8x7b_instruct_ten_shot_time'] = get_ner_ncbi_disease(ncbi_df.loc[i, 'text'], 10)
#     ncbi_df.loc[i, 'html_mistral_8x7b_instruct_twenty_shot'], ncbi_df.loc[i, 'mistral_8x7b_instruct_twenty_shot_time'] = get_ner_ncbi_disease(ncbi_df.loc[i, 'text'], 20)

# print(ncbi_df.iloc[1]['html_mistral_8x7b_instruct_one_shot'])
# print("====================================")
# print(mistral_parser_ner(ncbi_df.iloc[0]['html_mistral_8x7b_instruct_one_shot']))

ncbi_df['mistral_8x7b_instruct_one_shot'] = ncbi_df['html_mistral_8x7b_instruct_one_shot'].apply(mistral_parser_ner)
# ncbi_df['mistral_8x7b_instruct_five_shot'] = ncbi_df['html_mistral_8x7b_instruct_five_shot'].apply(mistral_parser_ner)
# ncbi_df['mistral_8x7b_instruct_ten_shot'] = ncbi_df['html_mistral_8x7b_instruct_ten_shot'].apply(mistral_parser_ner)
# ncbi_df['mistral_8x7b_instruct_twenty_shot'] = ncbi_df['html_mistral_8x7b_instruct_twenty_shot'].apply(mistral_parser_ner)

ncbi_df['gt_labels'], ncbi_df['mistral_8x7b_instruct_one_shot_labels'] = html_parsing_ncbi(ncbi_df, 'html_mistral_8x7b_instruct_one_shot')
# _, ncbi_df['mistral_8x7b_instruct_five_shot_labels'] = html_parsing_ncbi(ncbi_df, 'html_mistral_8x7b_instruct_five_shot')
# _, ncbi_df['mistral_8x7b_instruct_ten_shot_labels'] = html_parsing_ncbi(ncbi_df, 'html_mistral_8x7b_instruct_ten_shot')
# _, ncbi_df['mistral_8x7b_instruct_twenty_shot_labels'] = html_parsing_ncbi(ncbi_df, 'html_mistral_8x7b_instruct_twenty_shot')

print(f"F1-Score One Shot (Lenient): {get_classification_report(ncbi_df, 'gt_labels', 'mistral_8x7b_instruct_one_shot_labels', 'lenient')['default']['f1-score']}")
# print(f"F1-Score Five Shot (Lenient): {get_classification_report(ncbi_df, 'gt_labels', 'mistral_8x7b_instruct_five_shot_labels', 'lenient')['default']['f1-score']}")
# print(f"F1-Score Ten Shot (Lenient): {get_classification_report(ncbi_df, 'gt_labels', 'mistral_8x7b_instruct_ten_shot_labels', 'lenient')['default']['f1-score']}")
# print(f"F1-Score Twenty Shot (Lenient): {get_classification_report(ncbi_df, 'gt_labels', 'mistral_8x7b_instruct_twenty_shot_labels', 'lenient')['default']['f1-score']}")


# # Zero-shot NER using the model

# # Extend loop to evaluate zero-shot
# for i in tqdm(range(0, len(ncbi_df), 1)):
#     ncbi_df.loc[i, 'html_mistral_8x7b_instruct_zero_shot'], ncbi_df.loc[i, 'mistral_8x7b_instruct_zero_shot_time'] = get_ner_ncbi_disease(ncbi_df.loc[i, 'text'], 0)


# ncbi_df['mistral_8x7b_instruct_zero_shot'] = ncbi_df['html_mistral_8x7b_instruct_zero_shot'].apply(mistral_parser_ner)
# _, ncbi_df['mistral_8x7b_instruct_zero_shot_labels'] = html_parsing_ncbi(ncbi_df, 'html_mistral_8x7b_instruct_zero_shot')

# # Calculate Zero-Shot F1 score
# print(f"F1-Score Zero Shot (Lenient): {get_classification_report(ncbi_df, 'gt_labels', 'mistral_8x7b_instruct_zero_shot_labels', 'lenient')['default']['f1-score']}")


import xml.etree.ElementTree as ET

class OntologyRetriever:
    def __init__(self, ontology_type, file_path):
        self.ontology_type = ontology_type
        self.data = []
        if ontology_type.lower() == "mesh":
            self.load_mesh(file_path)
        else:
            raise ValueError("Unsupported ontology type. Only 'mesh' is implemented.")

    def load_mesh(self, file_path):
        """
        Load MeSH ontology from an XML file.
        Args:
            file_path: Path to the MeSH XML file.
        """
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Parse MeSH descriptors
        for descriptor in root.findall(".//DescriptorRecord"):
            descriptor_name = descriptor.findtext(".//DescriptorName/String")
            self.data.append(descriptor_name.lower())  # Store as lowercase for case-insensitive matching

    def retrieve(self, sentence):
        words = sentence.lower().split()
        matched_entities = []
        for term in self.data:
            if any(word in term for word in words):
                matched_entities.append(term)
        return matched_entities

# Example usage:
mesh_file_path = "./desc2024.xml"  # Adjust this to your file's location
ontology_retriever = OntologyRetriever("mesh", mesh_file_path)

def enrich_with_ontology(sentence: str):
    ontology_entities = ontology_retriever.retrieve(sentence)
    enriched_sentence = f"Sentence: {sentence}\nRelevant Medical Terms: {', '.join(ontology_entities)}"
    return enriched_sentence

# Retrieval-Augmented Generation for Enrichment
def retrieval_augmented_ner(sentence: str, shot: int = 0) -> str:
    """
    Perform RAG-style named entity recognition.
    """
    enriched_sentence = enrich_with_ontology(sentence)
    rag_prompt = f"""Use the following relevant medical terms to assist in identifying disease entities in the given sentence. Highlight all disease entities using HTML tags <span style="background-color: #FFFF00"> and </span>.

        {enriched_sentence}

        Output the sentence with disease entities highlighted:"""
    messages = [{'role': 'system', 'content': system_message}]
    
    for i in range(shot):
        messages.append({'role': 'user', 'content': ncbi_example_df.iloc[i]['text']}) 
        messages.append({'role': 'assistant', 'content': ncbi_example_df.iloc[i]['label_text']})
    messages.append({'role': 'user', 'content': rag_prompt})
        
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    
     # Clear cache
    torch.cuda.empty_cache()

    # Use autocast for mixed precision
    with autocast():
        time_start = time.time()
        outputs = model.generate(
            input_ids, 
            max_new_tokens=512,  # Reduce this if possible
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        time_end = time.time()
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True), time_end - time_start

batch_size = 1  # or an even smaller number if needed
accumulation_steps = 4  # adjust as needed

scaler = GradScaler()

for i in tqdm(range(0, len(ncbi_df), batch_size)):
    batch = ncbi_df.iloc[i:i+batch_size]
    
    with autocast():
        for j, row in batch.iterrows():
            enriched_sentence = enrich_with_ontology(row['text'])
            output, time_taken = retrieval_augmented_ner(enriched_sentence, 1)
            
            ncbi_df.loc[j, 'html_rag_mistral_8x7b_instruct'] = output
            ncbi_df.loc[j, 'rag_mistral_8x7b_instruct_time'] = time_taken

    # Clear CUDA cache after each batch
    torch.cuda.empty_cache()

ncbi_df['rag_mistral_8x7b_instruct'] = ncbi_df['html_rag_mistral_8x7b_instruct'].apply(mistral_parser_ner)
_, ncbi_df['rag_mistral_8x7b_instruct_labels'] = html_parsing_ncbi(ncbi_df, 'rag_mistral_8x7b_instruct')

# Calculate RAG F1 score
print(f"F1-Score RAG (Lenient): {get_classification_report(ncbi_df, 'gt_labels', 'rag_mistral_8x7b_instruct_labels', 'lenient')['default']['f1-score']}")
