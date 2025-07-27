import os
import json
import fitz  # PyMuPDF
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from tqdm import tqdm
from collections import Counter
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

# --- CONFIGURATION ---
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"
DOCS_DIR = os.path.join(INPUT_DIR, "docs")
PERSONA_FILE = os.path.join(INPUT_DIR, "persona.json")

# --- MODEL CONFIGURATION ---
# Stage 1: Retriever (a powerful, quantized bi-encoder)
RETRIEVER_MODEL_NAME = 'Optimum/bge-large-en-v1.5-onnx-quantized'
# Stage 2: Re-ranker (a fast and precise cross-encoder)
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L6-v2'
MODEL_CACHE_PATH = "/app/model_cache"
TOP_K_RETRIEVAL = 50 # The number of candidates to pass from Stage 1 to Stage 2

# --- 1. SETUP AND LOADING ---

def initialize_models():
    """Initializes and returns both the retriever and re-ranker models."""
    print("Initializing models...")
    # Load the optimized ONNX model for retrieval
    retriever_tokenizer = AutoTokenizer.from_pretrained(RETRIEVER_MODEL_NAME, cache_dir=MODEL_CACHE_PATH)
    retriever_model = ORTModelForFeatureExtraction.from_pretrained(RETRIEVER_MODEL_NAME, file_name="model_quantized.onnx", cache_dir=MODEL_CACHE_PATH)
    
    # Load the cross-encoder for re-ranking
    reranker_model = CrossEncoder(RERANKER_MODEL_NAME, cache_folder=MODEL_CACHE_PATH)
    print("Models loaded successfully.")
    return (retriever_tokenizer, retriever_model), reranker_model

def load_persona_and_job(persona_path):
    """Loads persona and job details from the structured JSON format."""
    print(f"Loading persona from {persona_path}...")
    with open(persona_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    persona_desc = f"Persona: A {data['persona']['role']}."
    job_desc = f"Task: Find fun activities and useful information for a {data['job_to_be_done']['task']}"
    # For BGE models, it's recommended to add this instruction for retrieval queries
    query = f"Represent this sentence for searching relevant passages: {persona_desc} {job_desc}"
    
    document_filenames = [doc['filename'] for doc in data['documents']]
    data['document_filenames'] = document_filenames

    print("Persona and job query created.")
    return data, query

# --- 2. DOCUMENT PROCESSING (COMBINED HEURISTIC) ---

def chunk_text(text, chunk_size=400, overlap=50):
    """Splits text into smaller, overlapping chunks."""
    tokens = text.split()
    if not tokens:
        return []
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = " ".join(tokens[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_main_font_size(page):
    """Analyzes a page to find the most common font size for body text."""
    sizes = [span['size'] for block in page.get_text("dict")["blocks"] if "lines" in block for line in block["lines"] for span in line["spans"]]
    if not sizes:
        return 10.0
    return Counter(sizes).most_common(1)[0][0]


def extract_content_from_pdfs(docs_dir, persona_data):
    """
    FINAL VERSION: Identifies headings using a combined heuristic of position, font size,
    capitalization, and structure.
    """
    print("Starting PDF extraction with combined super-heuristic...")
    all_sections = []
    
    for doc_name in tqdm(persona_data['document_filenames'], desc="Processing PDFs"):
        pdf_path = os.path.join(docs_dir, doc_name)
        if not os.path.exists(pdf_path):
            print(f"Warning: Document {doc_name} not found. Skipping.")
            continue
            
        try:
            doc = fitz.open(pdf_path)
            current_heading = doc_name
            
            for page_num, page in enumerate(doc):
                page_height = page.rect.height
                header_zone = page_height * 0.10
                footer_zone = page_height * 0.90
                
                body_font_size = get_main_font_size(page)
                current_paragraph = ""

                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" not in block:
                        continue
                    
                    block_bbox = block['bbox']
                    if block_bbox[1] < header_zone or block_bbox[3] > footer_zone:
                        continue

                    text = " ".join([span["text"] for line in block["lines"] for span in line["spans"]]).strip()
                    if not text:
                        continue

                    first_span = block["lines"][0]["spans"][0]
                    font_size = first_span["size"]
                    is_bold = "bold" in first_span["font"].lower()

                    # --- COMBINED SUPER-HEURISTIC ---
                    is_heading_candidate = len(text.split()) < 20 and is_bold
                    
                    if is_heading_candidate:
                        is_larger_font = font_size > (body_font_size * 1.15)
                        
                        letters_only = ''.join(filter(str.isalpha, text))
                        is_all_caps = letters_only.isupper() and len(letters_only) > 1
                        
                        is_single_line_block = len(block["lines"]) == 1

                        if is_larger_font or is_all_caps or is_single_line_block:
                            # This is a confirmed heading
                            if current_paragraph:
                                text_chunks = chunk_text(current_paragraph)
                                for chunk in text_chunks:
                                    all_sections.append({
                                        "document": doc_name, "page_number": page_num + 1,
                                        "section_title": current_heading, "refined_text": chunk
                                    })
                                current_paragraph = ""
                            current_heading = text
                        else:
                            # Just a short bolded phrase, not a heading
                            current_paragraph += " " + text
                    else:
                        # This is a paragraph
                        current_paragraph += " " + text
                
                # After a page, add any remaining paragraph text
                if current_paragraph:
                    text_chunks = chunk_text(current_paragraph)
                    for chunk in text_chunks:
                        all_sections.append({
                            "document": doc_name, "page_number": page_num + 1,
                            "section_title": current_heading, "refined_text": chunk
                        })
                    current_paragraph = "" # Reset for the next page

        except Exception as e:
            print(f"Error processing {doc_name}: {e}")
            
    print(f"Extracted {len(all_sections)} text sections from all documents.")
    return all_sections

# --- 3. AI-DRIVEN ANALYSIS (UPDATED TWO-STAGE RANKING) ---

def rank_sections(query, sections, retriever, reranker):
    """
    Ranks sections using a two-stage retrieve and re-rank architecture.
    """
    if not sections: return []

    retriever_tokenizer, retriever_model = retriever
    section_texts = [sec["refined_text"] for sec in sections]

    # === STAGE 1: RETRIEVAL ===
    print(f"Stage 1: Retrieving top {TOP_K_RETRIEVAL} candidates from {len(sections)} sections...")
    
    # Create embeddings for all sections
    corpus_inputs = retriever_tokenizer(section_texts, padding=True, truncation=True, return_tensors='pt')
    corpus_embeddings = retriever_model(**corpus_inputs).last_hidden_state.mean(dim=1).detach().numpy()
    
    # Create embedding for the query
    query_inputs = retriever_tokenizer([query], padding=True, truncation=True, return_tensors='pt')
    query_embedding = retriever_model(**query_inputs).last_hidden_state.mean(dim=1).detach().numpy()

    # Calculate initial similarity and find top candidates
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
    top_k_indices = np.argsort(similarities)[-TOP_K_RETRIEVAL:]
    top_k_candidates = [sections[i] for i in top_k_indices]
    
    # === STAGE 2: RE-RANKING ===
    print(f"Stage 2: Re-ranking the top {len(top_k_candidates)} candidates for precision...")

    # Create pairs of (query, candidate_text) for the cross-encoder
    reranker_pairs = [(query, cand["refined_text"]) for cand in top_k_candidates]
    
    # Get high-precision scores from the cross-encoder
    reranker_scores = reranker.predict(reranker_pairs, show_progress_bar=True)
    
    # Add the precise scores to the candidates and sort
    for i, cand in enumerate(top_k_candidates):
        cand["importance_rank"] = float(reranker_scores[i])
        
    ranked_sections = sorted(top_k_candidates, key=lambda x: x["importance_rank"], reverse=True)
    
    print("Ranking complete.")
    return ranked_sections

# --- 4. OUTPUT GENERATION ---

def generate_output_json(persona_data, ranked_sections, output_path):
    """Generates the final JSON output file."""
    print("Generating final output JSON...")
    output_data = {
        "metadata": {
            "input_documents": [doc['filename'] for doc in persona_data['documents']],
            "persona": persona_data['persona']['role'],
            "job_to_be_done": persona_data['job_to_be_done']['task'],
            "processing_timestamp": datetime.utcnow().isoformat() + "Z"
        },
        "extracted_sections": [
            {
                "document": sec["document"],
                "page_number": sec["page_number"],
                "section_title": sec["section_title"],
                "importance_rank": i + 1
            } for i, sec in enumerate(ranked_sections)
        ],
        "sub_section_analysis": [
             {
                "document": sec["document"],
                "page_number": sec["page_number"],
                "refined_text": sec["refined_text"]
            } for sec in ranked_sections
        ]
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Output successfully saved to {output_path}")

# --- MAIN EXECUTION (UPDATED) ---

if __name__ == "__main__":
    print("--- 🚀 Starting Persona-Driven Document Intelligence (Two-Stage) 🚀 ---")
    retriever, reranker = initialize_models()
    persona_data, query = load_persona_and_job(PERSONA_FILE)
    all_sections = extract_content_from_pdfs(DOCS_DIR, persona_data)
    
    # Pass both models to the ranking function
    ranked_sections = rank_sections(query, all_sections, retriever, reranker)
    
    output_filename = "challenge1b_output.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    generate_output_json(persona_data, ranked_sections, output_path)
    
    print("--- ✅ Process Complete ✅ ---")
