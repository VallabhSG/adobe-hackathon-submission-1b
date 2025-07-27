# Approach Explanation: Persona-Driven Document Intelligence

### Overview
Our solution is a robust, offline-first system designed to act as an intelligent document analyst. It ingests multiple PDF collections and provides a prioritized list of the most relevant sections based on a specific user persona and their job-to-be-done. The entire application is containerized using Docker and leverages a state-of-the-art, two-stage AI architecture to ensure the highest quality results while adhering to strict performance and size constraints.

### Project Structure
The project is organized into a clean, self-contained directory. The `input` folder holds the source PDFs and the `persona.json` configuration, while the `output` folder stores the final analysis. All core logic, dependencies, and containerization instructions are in the root directory.

/adobe-hackathon/
├── input/
│   ├── docs/
│   │   ├── South of France - Cities.pdf
│   │   └── ... (and other source PDFs)
│   └── persona.json
├── output/
│   └── challenge1b_output.json
├── main.py
├── Dockerfile
├── requirements.txt
└── approach_explanation.md


### Methodology

Our approach was refined through iterative testing across diverse document types, including academic papers, financial reports, and travel guides. The final methodology combines intelligent query formulation with a sophisticated document processing pipeline and a high-precision ranking system.

**1. Persona-Task Synthesis:**
We begin by transforming the structured input into a rich, semantic query. The persona's role and the job-to-be-done are synthesized into a single, detailed prompt. For our retriever model, we prepend a specific instruction ("Represent this sentence for searching relevant passages:") to the query, a best practice that significantly enhances retrieval accuracy.

**2. Advanced Document Ingestion and Heading Detection:**
We use the high-performance `PyMuPDF` library for efficient PDF parsing. To overcome the challenge of varied document structures, our system employs an advanced "super-heuristic" for heading detection. This multi-layered approach includes:
* **Positional Filtering:** Automatically ignoring the top and bottom 10% of each page to eliminate headers and footers.
* **Dynamic Font Analysis:** Calculating the main body text size on each page to adaptively identify headings that are proportionally larger.
* **Structural and Style Checks:** A line of text is confirmed as a heading only if it is bold, short, and meets at least one of three conditions: it is proportionally larger than the body text, written in all caps, or is structurally separate as a single-line block.

**3. Two-Stage "Retrieve and Re-rank" Architecture:**
To achieve state-of-the-art relevance ranking, we implemented a two-stage process:
* **Stage 1 (Retrieval):** We use a powerful, quantized version of the `BAAI/bge-large-en-v1.5` bi-encoder to quickly scan all text chunks and retrieve the top 50 most likely candidates. This model was chosen for its top-tier performance, while quantization ensures it meets the size constraints.
* **Stage 2 (Re-ranking):** The 50 candidates are then passed to a specialized `cross-encoder/ms-marco-MiniLM-L6-v2` model. This cross-encoder analyzes the query and each text chunk *together*, allowing for a much deeper contextual understanding and producing a highly precise final ranking.

### Sample Input JSON Structure
The system takes a structured JSON file as input, which defines the documents to be processed, the user persona, and the task.
```json
{
  "challenge_info": {
    "challenge_id": "round_1b_002",
    "test_case_name": "travel_planner"
  },
  "documents": [
    {
      "filename": "South of France - Cities.pdf",
      "title": "South of France - Cities"
    }
  ],
  "persona": {
    "role": "Travel Planner"
  },
  "job_to_be_done": {
    "task": "Plan a trip of 4 days for a group of 10 college friends."
  }
}

Sample Output JSON Structure
The final output is a structured JSON file containing metadata, a ranked list of the most relevant sections, and the detailed text of those sections. The rank is a clean integer, and the sub-section analysis provides the core text.

{
  "metadata": {
    "input_documents": [
      "South of France - Cities.pdf",
      "South of France - Things to Do.pdf"
    ],
    "persona": "Travel Planner",
    "job_to_be_done": "Plan a trip of 4 days for a group of 10 college friends.",
    "processing_timestamp": "2025-07-23T18:00:00.000000Z"
  },
  "extracted_sections": [
    {
      "document": "South of France - Things to Do.pdf",
      "section_title": "Coastal Adventures",
      "importance_rank": 1,
      "page_number": 2
    }
  ],
  "sub_section_analysis": [
    {
      "document": "South of France - Things to Do.pdf",
      "refined_text": "The South of France is renowned for its beautiful coastline along the Mediterranean Sea. Here are some activities to enjoy by the sea...",
      "page_number": 2
    }
  ]
}
