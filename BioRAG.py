import time
import torch
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
from utils import html_parsing_ncbi, get_classification_report
import xml.etree.ElementTree as ET
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import faiss

ncbi_df = pd.read_csv('data/NER/NCBI-disease/test_200.csv')
ncbi_example_df = pd.read_csv('data/NER/NCBI-disease/examples.csv')

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("aaditya/OpenBioLLM-Llama3-8B")
model = AutoModelForCausalLM.from_pretrained(
    "aaditya/OpenBioLLM-Llama3-8B",
    torch_dtype=torch.float16,
    device_map="auto"
)
model.gradient_checkpointing_enable()

chat_template = open('../chat_templates/chat_templates/mistral-instruct.jinja').read()
chat_template = chat_template.replace('    ', '').replace('\n', '')
tokenizer.chat_template = chat_template

system_message = """You are a helpful assistant to perform the following task.
"TASK: the task is to extract disease entities in a sentence."
"INPUT: the input is a sentence."
"OUTPUT: the output is an HTML that highlights all the disease entities in the sentence. The highlighting should only use HTML tags <span style=\"background-color: #FFFF00\"> and </span> and no other tags."
"""

# Load ontology terms using FAISS
class FAISSRetriever:
    def __init__(self, ontology_file, embed_model="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embed_model)
        self.terms = self.load_terms(ontology_file)
        self.index = self.create_index()

    def load_terms(self, ontology_file):
        """Load MeSH ontology terms."""
        tree = ET.parse(ontology_file)
        root = tree.getroot()
        terms = [
            descriptor.findtext(".//DescriptorName/String").lower()
            for descriptor in root.findall(".//DescriptorRecord")
        ]
        return terms

    def create_index(self):
        """Create FAISS index for ontology terms."""
        term_embeddings = self.embedder.encode(self.terms, convert_to_tensor=True)
        index = faiss.IndexFlatL2(term_embeddings.shape[1])
        term_embeddings_np = term_embeddings.cpu().numpy()
        faiss.normalize_L2(term_embeddings_np)
        index.add(term_embeddings_np)
        return index

    def retrieve(self, sentence, top_k=5):
        """Retrieve top-k relevant terms."""
        sentence_embedding = self.embedder.encode(sentence, convert_to_tensor=True)
        sentence_embedding = sentence_embedding.cpu().numpy().reshape(1, -1)
        faiss.normalize_L2(sentence_embedding)
        distances, indices = self.index.search(sentence_embedding, top_k)
        return [self.terms[idx] for idx in indices[0]]


ontology_retriever = FAISSRetriever("./desc2024.xml")

def enrich_with_ontology(sentence: str):
    terms = ontology_retriever.retrieve(sentence)
    return f"Sentence: {sentence}\nRelevant Terms: {', '.join(terms)}"


def retrieval_augmented_ner(sentence: str, shot: int = 0) -> str:
    enriched_sentence = enrich_with_ontology(sentence)
    
    rag_prompt = f"""
    TASK: Extract disease entities from the following sentence using the provided relevant medical terms.

    INPUT SENTENCE: {sentence}

    INSTRUCTIONS:
    - Highlight disease entities in the input sentence using <span style="background-color: #FFFF00"> and </span>.
    - Use relevant terms as context for entity identification but highlight entities only if present in the sentence.
    - Output the original sentence with entities highlighted.
    - Do not highlight terms that are not entities.

    Begin your output with "Output:" and ensure the extracted entities are correctly highlighted.
    """

    messages = [{'role': 'system', 'content': system_message}]
    for i in range(shot):
        messages.append({'role': 'user', 'content': ncbi_example_df.iloc[i]['text']}) 
        messages.append({'role': 'assistant', 'content': ncbi_example_df.iloc[i]['label_text']})
    messages.append({'role': 'user', 'content': rag_prompt})

    print(type(messages))
    for item in messages:
        print(type(item))
    print(type(messages[0]))
    # Tokenize the input
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

    torch.cuda.empty_cache()

    # Generate output
    with autocast():
        time_start = time.time()
        outputs = model.generate(input_ids, max_new_tokens=1024, do_sample=True)
        time_end = time.time()
    
    # Decode and clean the output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(output_text)
    print()
    if "Output:" in output_text:
        output_text = output_text.split("Output:")[-1].strip()
    
    return output_text, time_end - time_start


# Batch Processing
def process_batch(df, batch_size=1, shots=1):
    scaler = GradScaler()
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        with autocast():
            for j, row in batch.iterrows():
                enriched_sentence = enrich_with_ontology(row['text'])
                output, time_taken = retrieval_augmented_ner(enriched_sentence, shots)
                df.loc[j, 'html_rag'] = output
                df.loc[j, 'time_rag'] = time_taken
        torch.cuda.empty_cache()
    return df



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


ncbi_df = ncbi_df[:5].reset_index(drop=True)

for i in tqdm(range(0, len(ncbi_df), 1)):
    ncbi_df.loc[i, 'html_mistral_8x7b_instruct_one_shot'], ncbi_df.loc[i, 'mistral_8x7b_instruct_one_shot_time'] = get_ner_ncbi_disease(ncbi_df.loc[i, 'text'], 1)


ncbi_df['mistral_8x7b_instruct_one_shot'] = ncbi_df['html_mistral_8x7b_instruct_one_shot'].apply(mistral_parser_ner)

ncbi_df['gt_labels'], ncbi_df['mistral_8x7b_instruct_one_shot_labels'] = html_parsing_ncbi(ncbi_df, 'html_mistral_8x7b_instruct_one_shot')

print(f"F1-Score One Shot (Lenient): {get_classification_report(ncbi_df, 'gt_labels', 'mistral_8x7b_instruct_one_shot_labels', 'lenient')['default']['f1-score']}")

ncbi_df = process_batch(ncbi_df, batch_size=1, shots=1)

ncbi_df['html_rag_instruct'] = ncbi_df['html_rag'].apply(mistral_parser_ner)

_, ncbi_df['html_rag_labels'] = html_parsing_ncbi(ncbi_df, 'html_rag_instruct')

f1_score_rag = get_classification_report(ncbi_df, 'gt_labels', 'html_rag_labels', 'lenient')['default']['f1-score']
# ncbi_df.to_csv("output.csv", index=False)

print(f"RAG F1-Score: {f1_score_rag}")