import time
import torch
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
from utils import html_parsing_ncbi, get_classification_report, label2digit  # other helper functions assumed to be available
# --- UMLS Retriever and Other Helper Functions ---
import requests
import spacy
import logging
from torch.nn.functional import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Unified Prompt and System Message ---
# Now both few-shot and RAG approaches will instruct the model to output HTML with highlighted disease terms.
system_message = (
    "You are a helpful assistant. Your task is to extract disease entities from a clinical sentence. "
    "Return your answer as an HTML string in which all detected disease terms are highlighted using only "
    "the HTML tags <span style=\"background-color: #FFFF00\"> and </span> (for example, if the disease is 'cancer', "
    "the output should contain <span style=\"background-color: #FFFF00\">cancer</span>). "
    "Make sure your answer contains no other formatting or extraneous text."
)

# Read and process chat template if needed (making sure it follows the above instructions)
with open('/home/araghani/chat_templates/chat_templates/openchat-3.5.jinja', 'r') as f:
    chat_template = f.read()
chat_template = chat_template.replace('    ', '').replace('\n', '')
tokenizer = AutoTokenizer.from_pretrained("johnsnowlabs/JSL-MedLlama-3-8B-v2.0")
model = AutoModelForCausalLM.from_pretrained("johnsnowlabs/JSL-MedLlama-3-8B-v2.0",
                                             torch_dtype=torch.float16,
                                             device_map="auto")
model.gradient_checkpointing_enable()
tokenizer.chat_template = chat_template

# Load test data for NER (NCBI-disease)
ncbi_df = pd.read_csv('data/NER/NCBI-disease/test_200.csv')
ncbi_df = ncbi_df[:10]
ncbi_example_df = pd.read_csv('data/NER/NCBI-disease/examples.csv')

# --- Few-Shot NER using Bio LLM ---
def get_ner_ncbi_disease(chat_template, sentence: str, shot: int = 0) -> (str, float):
    """
    Get NER results from few-shot prompting. The prompt now instructs the model to return the disease term 
    highlighted in HTML format.
    """
    messages = [{'role': 'system', 'content': system_message}]
    # Use few-shot examples if available (ensuring they follow the same HTML output format)
    for i in range(shot):
        messages.append({'role': 'user', 'content': ncbi_example_df.iloc[i]['text']})
        messages.append({'role': 'assistant', 'content': ncbi_example_df.iloc[i]['label_text']})
    messages.append({'role': 'user', 'content': sentence})

    input_ids = tokenizer.apply_chat_template(messages, chat_template=chat_template, return_tensors="pt").to("cuda")

    time_start = time.time()
    outputs = model.generate(input_ids, max_new_tokens=1024)
    time_end = time.time()
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded, time_end - time_start

def mistral_parser_ner(text: str):
    """
    Parse the text generation output. Assumes that the answer is the text after the last "[/INST] ".
    """
    cleaned_text = text.split('[/INST] ')[-1]
    cleaned_text = cleaned_text.split('\n\n')[0]
    return cleaned_text

# Run few-shot evaluation (benchmark)
for i in tqdm(range(0, len(ncbi_df), 1), desc="Few-shot evaluation"):
    raw_output, elapsed = get_ner_ncbi_disease(chat_template, ncbi_df.loc[i, 'text'], shot=1)
    ncbi_df.loc[i, 'html_mistral_8x7b_instruct_one_shot'] = raw_output
    print(ncbi_df.loc[i, 'html_mistral_8x7b_instruct_one_shot'])
    ncbi_df.loc[i, 'mistral_8x7b_instruct_one_shot_time'] = elapsed

ncbi_df['mistral_8x7b_instruct_one_shot'] = ncbi_df['html_mistral_8x7b_instruct_one_shot'].apply(mistral_parser_ner)
ncbi_df['gt_labels'], ncbi_df['mistral_8x7b_instruct_one_shot_labels'] = html_parsing_ncbi(ncbi_df, 'html_mistral_8x7b_instruct_one_shot')
print(f"F1-Score Few-Shot (Lenient): {get_classification_report(ncbi_df, 'gt_labels', 'mistral_8x7b_instruct_one_shot_labels', 'lenient')['default']['f1-score']}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm")

try:
    tokenizer = AutoTokenizer.from_pretrained("johnsnowlabs/JSL-MedLlama-3-8B-v2.0")
    model = AutoModelForCausalLM.from_pretrained("johnsnowlabs/JSL-MedLlama-3-8B-v2.0",
                                                torch_dtype=torch.float16,
                                                device_map="auto")
    model.eval()
    EMBEDDING_MODEL_LOADED = True
except Exception as e:
    logger.warning(f"Failed to load embedding model: {e}")
    EMBEDDING_MODEL_LOADED = False

cache = {}

def get_embedding(text: str):
    if not EMBEDDING_MODEL_LOADED:
        logger.warning("Embedding model not loaded, returning None")
        return None
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    embedding = torch.mean(hidden_states, dim=1)
    return embedding.squeeze(0)

class UMLSAPIRetriever:
    def __init__(self, api_key, version="current"):
        self.api_key = api_key
        self.version = version
        self.base_url = "https://uts-ws.nlm.nih.gov/rest"
        self.request_count = 0
        self.last_request_time = 0

    def _respect_rate_limit(self):
        current_time = time.time()
        if current_time - self.last_request_time < 0.2:
            time.sleep(0.2 - (current_time - self.last_request_time))
        self.last_request_time = time.time()
        self.request_count += 1

    def _cached_get(self, url, params):
        key = (url, tuple(sorted(params.items())))
        if key in cache:
            return cache[key]
        self._respect_rate_limit()
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            cache[key] = data
            return data
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                return None
            else:
                logger.error(f"HTTP error: {e}")
                raise e
        except Exception as e:
            logger.error(f"Error in API request: {e}")
            return None

    def search_term(self, term):
        search_url = f"{self.base_url}/search/{self.version}"
        params = {
            "apiKey": self.api_key,
            "string": term,
            "sabs": "SNOMEDCT_US,ICD10CM,MEDLINEPLUS,MSH",
            "returnIdType": "concept",
            "pageSize": 10,
            "language": "ENG"
        }
        data = self._cached_get(search_url, params)
        if data and "result" in data and "results" in data["result"]:
            return [r.get("ui") for r in data["result"]["results"] if r.get("ui") and r.get("ui") != "NONE"]
        return []

    def get_concept_detail(self, cui):
        url = f"{self.base_url}/content/{self.version}/CUI/{cui}"
        params = {"apiKey": self.api_key}
        data = self._cached_get(url, params)
        if data and "result" in data:
            return data["result"]
        return None

    def get_definitions(self, cui):
        url = f"{self.base_url}/content/{self.version}/CUI/{cui}/definitions"
        params = {
            "apiKey": self.api_key,
            "language": "ENG"
        }
        data = self._cached_get(url, params)
        definitions = []
        if data and data.get("result") and data.get("result") != "NONE":
            for d in data["result"]:
                value = d.get("value", "")
                if value:
                    definitions.append(value)
        return definitions
    
    def get_semantic_types(self, cui):
        url = f"{self.base_url}/content/{self.version}/CUI/{cui}/semanticTypes"
        params = {"apiKey": self.api_key}
        data = self._cached_get(url, params)
        semantic_types = []
        if data and data.get("result") and data.get("result") != "NONE":
            for st in data["result"]:
                if "name" in st:
                    semantic_types.append(st["name"])
        return semantic_types
    
    def get_disease_semantic_network(self, term):
        """Query the UMLS Semantic Network specifically for disease concepts"""
        search_url = f"{self.base_url}/search/{self.version}"
        params = {
            "apiKey": self.api_key,
            "string": term,
            "sabs": "SNOMEDCT_US,ICD10CM,MEDLINEPLUS,MSH",
            "searchType": "exact",
            "returnIdType": "concept",
            "sty": "T047,T191",  # Disease or Syndrome, Neoplastic Process
            "pageSize": 5,
            "language": "ENG"
        }
        return self._cached_get(search_url, params)


def extract_biomedical_candidate_phrases(sentence: str):
    # Replace standard spaCy with a biomedical-specific model
    try:
        nlp = spacy.load("en_core_sci_md")  # Scientific/biomedical model
    except:
        logger.warning("Biomedical model not available, using generic model")
        nlp = spacy.load("en_core_web_sm")
    
    doc = nlp(sentence)
    phrases = set()
    
    # Prioritize known medical entities
    for ent in doc.ents:
        if ent.label_ in ["DISEASE", "DISORDER", "SYNDROME", "CONDITION"]:
            phrases.add(ent.text.lower())
    
    # Add noun chunks with medical keywords
    medical_keywords = ["disease", "syndrome", "disorder", "condition", "infection", 
                        "cancer", "illness", "pathology"]
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower()
        if any(keyword in chunk_text for keyword in medical_keywords):
            phrases.add(chunk_text)
    
    logger.info(f"Extracted {len(phrases)} biomedical phrases: {phrases}")
    return list(phrases)


def retrieve_disease_entities(sentence: str, retriever: UMLSAPIRetriever):
    phrases = extract_biomedical_candidate_phrases(sentence)
    disease_entities = {}
    
    # Disease-specific semantic types from UMLS
    disease_semantic_types = ["Disease or Syndrome", "Pathologic Function", 
                             "Neoplastic Process", "Congenital Abnormality"]
    
    for phrase in tqdm(phrases, desc="Processing potential disease terms"):
        try:
            cuis = retriever.search_term(phrase)
            if cuis:
                for cui in cuis[:3]:  # Check top 3 matches
                    semantic_types = retriever.get_semantic_types(cui)
                    
                    # Filter for disease-related semantic types only
                    if any(disease_type in semantic_types for disease_type in disease_semantic_types):
                        definitions = retriever.get_definitions(cui)
                        definition = definitions[0] if definitions else ""
                        if definition:
                            disease_entities[phrase] = {
                                "definition": definition,
                                "semantic_types": semantic_types,
                                "cui": cui
                            }
                            break  # Found a disease match, no need to check other CUIs
        except Exception as e:
            logger.error(f"Error processing phrase '{phrase}': {e}")
    
    logger.info(f"Retrieved {len(disease_entities)} disease entities")
    return disease_entities


# --- New Chain-of-Thought (CoT) Prompt for RAG ---
def create_cot_prompt(sentence: str, meanings: dict) -> str:
    """
    Create a simplified prompt focusing only on disease entity identification
    without complex chain-of-thought reasoning.
    """
    context_text = ""
    for phrase, info in meanings.items():
        semantic_types_str = ", ".join(info["semantic_types"])
        context_text += f"- {phrase}: {info['definition']} (Types: {semantic_types_str})\n"
    
    prompt = (
        "TASK: Extract disease entities from the following clinical sentence. Return the sentence "
        "with disease terms highlighted using <span style=\"background-color: #FFFF00\"> tags.\n\n"
        f"CONTEXT: Here are verified disease entities and their definitions from a medical database:\n"
        f"{context_text}\n"
        f"CLINICAL SENTENCE: \"{sentence}\"\n\n"
        "OUTPUT: Return the clinical sentence with all disease terms wrapped in "
        "<span style=\"background-color: #FFFF00\"> and </span> tags."
    )
    return prompt


def llm_call(prompt: str, chat_template) -> str:
    """
    Call the local OpenBioLLM model with the given prompt.
    This function uses the same chat templating approach as in the few-shot function.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    input_ids = tokenizer.apply_chat_template(messages, chat_template=chat_template, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids, max_new_tokens=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Retrieval-Augmented NER using CoT Prompt ---
def retrieval_augmented_ner_with_rat(chat_template, sentence: str, umls_api_key: str):
    """
    Implement the RAT (Retrieval Augmented Thoughts) approach for NER.
    This iteratively refines entity predictions with retrieved information.
    """
    retriever = UMLSAPIRetriever(umls_api_key)
    
    # Step 1: Get initial entity predictions
    initial_prompt = f"Extract disease entities from this clinical sentence: \"{sentence}\". List each entity on a new line."
    initial_entities = llm_call(initial_prompt, chat_template).split('\n')
    
    # Step 2: Retrieve information for each candidate entity
    refined_entities = {}
    for entity in initial_entities:
        entity = entity.strip()
        if entity:
            meanings = retrieve_disease_entities(entity, retriever)
            if meanings:
                refined_entities[entity] = meanings
    
    # Step 3: Final synthesis with retrieved information
    final_prompt = (
        f"TASK: Identify disease entities in: \"{sentence}\"\n\n"
        f"Based on medical knowledge, these are verified disease concepts:\n"
    )
    
    for entity, info in refined_entities.items():
        final_prompt += f"- {entity}: {info.get(entity, {}).get('definition', 'A disease entity')}\n"
    
    final_prompt += (
        f"\nReturn the clinical sentence with ONLY disease entities highlighted using "
        f"<span style=\"background-color: #FFFF00\"> and </span> tags."
    )
    
    return llm_call(final_prompt, chat_template)


# --- Evaluation Function ---
def run_evaluation(chat_template, ncbi_df, umls_api_key):
    """
    Evaluate each clinical sentence using both few-shot and retrieval-augmented (RAG) approaches.
    """
    
    # --- Retrieval-Augmented (CoT) Evaluation ---
    for i in tqdm(range(len(ncbi_df)), desc="RAG (CoT) evaluation"):
        # Retrieve the HTML output using the chain-of-thought (CoT) method
        rag_html = retrieval_augmented_ner_with_rat(chat_template, ncbi_df.loc[i, 'text'], umls_api_key)
        print(rag_html)
        ncbi_df.loc[i, 'html_rag_cot'] = rag_html

    ncbi_df['rag_cot'] = ncbi_df['html_rag_cot'].apply(mistral_parser_ner)
    ncbi_df['gt_labels'], ncbi_df['rag_cot_labels'] = html_parsing_ncbi(ncbi_df, 'html_rag_cot')

    # Calculate the F1 score by comparing ground truth labels (assumed in 'labels') to predicted labels
    report_rag_cot = get_classification_report(ncbi_df, 'gt_labels', 'rag_cot_labels', 'lenient')
    print(f"F1-Score RAG with CoT (Lenient): {report_rag_cot['default']['f1-score']:.4f}")

    

# --- Example usage ---
if __name__ == "__main__":
    # UMLS API key (ensure this key is valid and that only English content is returned)
    umls_api_key = "a9383cc4-f6e1-442f-bb44-c44935d4f8d8"
    
    # Run evaluation on the dataframe using OpenBioLLM for all tasks
    run_evaluation(chat_template, ncbi_df, umls_api_key)