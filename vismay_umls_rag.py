import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline
import torch

# --- UMLS Data Preparation (Preprocess Once) ---
def preprocess_umls(input_rrf="/home/vivora/BioNER/BioMed-NER/data/2024AB/META/MRCONSO.RRF", output_txt="umls_data.txt"):
    """
    Convert UMLS MRCONSO.RRF file into a text format with relevant NER info:
    [CUI|Concept Name|Definition|Semantic Type]
    """
    with open(input_rrf, "r", encoding="utf-8") as f_in, \
         open(output_txt, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            fields = line.strip().split("|")
            cui = fields[0]        # Concept Unique Identifier
            name = fields[14]      # Preferred term (STR)
            definition = fields[1] # Optional: SAB (Source Vocabulary)
            sem_type = fields[12]  # Semantic Type (e.g., T047, Disease)
            
            f_out.write(f"{cui}|{name}|{definition}|{sem_type}\n")

# --- Core Pipeline ---
class UMLSNERRAG:
    def __init__(self, umls_file="umls_data.txt", index_file="faiss_index.index", entries_file="umls_entries.pkl"):
        # Load or create UMLS entries
        if os.path.exists(entries_file):
            with open(entries_file, "rb") as f:
                self.umls_entries = pickle.load(f)
        else:
            self.umls_entries = []
            with open(umls_file, "r", encoding="utf-8") as f:
                for line in f:
                    cui, name, _, sem_type = line.strip().split("|")
                    self.umls_entries.append(f"{name} ({sem_type})")
            with open(entries_file, "wb") as f:
                pickle.dump(self.umls_entries, f)
        
        # Initialize embedding model
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cuda')
        
        # Load or build FAISS index
        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
        else:
            embeddings = self.embedder.encode(self.umls_entries, convert_to_numpy=True)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
            faiss.write_index(self.index, index_file)
        
        # Initialize LLM
        self.tokenizer = AutoTokenizer.from_pretrained("johnsnowlabs/JSL-MedLlama-3-8B-v2.0")
        self.llm = pipeline(
            "text-generation",
            model="johnsnowlabs/JSL-MedLlama-3-8B-v2.0",
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def retrieve(self, text, top_k=5):
        """Retrieve relevant UMLS concepts"""
        embedding = self.embedder.encode(text, convert_to_numpy=True)
        distances, indices = self.index.search(np.expand_dims(embedding, 0), top_k)
        return [self.umls_entries[i] for i in indices[0]]
    
    def extract_entities(self, medical_text):
        """NER with UMLS-augmented context"""
        # Retrieve UMLS context
        umls_context = self.retrieve(medical_text)
        
        # Construct instruction prompt
        prompt = (
            "Identify medical entities in this text and classify them using UMLS semantic types.\n"
            "UMLS Reference:\n" + "\n".join(umls_context) + "\n\n"
            "Text: " + medical_text + "\n\n"
            "Entities (format: 'Entity [Type]'):"
        )
        
        # Generate response
        outputs = self.llm(
            prompt,
            max_new_tokens=256,
            return_full_text=False,
            do_sample=False
        )
        return outputs[0]['generated_text']

# --- Usage ---
if __name__ == "__main__":
    # First-time setup: Convert MRCONSO.RRF to text format if necessary
    if not os.path.exists("umls_data.txt"):
        preprocess_umls()
    
    # Initialize pipeline (this will load saved index and entries if they exist)
    ner_pipeline = UMLSNERRAG()
    
    # Example medical record
    text = "Patient presented with headache and took 500mg aspirin. History of diabetes. " \
    "What are the properties of Calcimycin?" \
    "Oral and topical steroids were used to induce regression in an inflammatory, obstructing endobronchial polyp caused by a retained foreign body. The FB (a peanut half), which had been present for over six months, was then able to be easily and bloodlessly retrieved with fiberoptic bronchoscopy. " \
    "Recurrent buccal space abscesses: a complication of Crohn's disease. A patient is described with generalized gastrointestinal involvement by Crohn's disease"
    entities = ner_pipeline.extract_entities(text)
    
    print("Input Text:", text)
    print("Identified Entities:\n", entities)
