# Advanced Hybrid RAG | Healthcare Claims & Coverage Assistant

An **educational retrieval-augmented generation (RAG) prototype** built with Python, ChromaDB, sentence transformers, BM25, cross-encoder re-ranking, Groq and Streamlit. It answers questions about health-plan coverage, claims, prior authorization, appeals and related topics from **eight synthetic training PDFs**.

> **Independent educational project. Not affiliated with or endorsed by Aetna.** The included documents are fictional examples—not official plan policies, clinical advice, or actual member records.

## System architecture

```text
Eight synthetic healthcare PDFs (data/raw/)
                  │
                  ▼
pypdf text extraction + filename/page metadata
                  │
                  ▼
RecursiveCharacterTextSplitter (500 chars, overlap 100)
                  │
         ┌────────┴────────┐
         ▼                 ▼
  Dense embeddings     BM25 indexing
all-MiniLM-L6-v2        BM25Okapi
         │                 │
         ▼                 │
  ChromaDB persistent     │
  vector collection       │
         │                 │
         └───────┬─────────┘
                 ▼
    Top-3 dense + top-3 BM25
                 ▼
    Reciprocal Rank Fusion (k=60)
                 ▼
    Top-3 fused candidates
                 ▼
CrossEncoder ms-marco-MiniLM-L-6-v2
                 ▼
     Grounded prompt with source
     filename and page metadata
                 ▼
      Groq: openai/gpt-oss-120b
                 ▼
       Streamlit user interface
```

## Key components

| Component | Implementation |
| --- | --- |
| PDF parsing | `pypdf.PdfReader`, one record per page |
| Chunking | `RecursiveCharacterTextSplitter`, `chunk_size=500`, `chunk_overlap=100` |
| Semantic search | `SentenceTransformer('all-MiniLM-L6-v2')` and ChromaDB |
| Lexical search | `BM25Okapi` |
| Rank fusion | Reciprocal Rank Fusion, constant `k=60` |
| Re-ranking | `CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')` |
| Generation | Groq API with `openai/gpt-oss-120b` |
| Front end | Streamlit |

The prompt instructs the LLM to answer only from retrieved context, give a source file and page number, and state when evidence is insufficient. When the top cross-encoder score is negative, the system **appends a human-review recommendation**. This is a warning, **not** an operational human approval process.

## Repository structure

```text
AI-advanced-rag-project/
├── README.md
├── .gitignore
├── .env.example
├── requirements.txt
├── app.py
├── streamlit_app.py
└── data/
    └── raw/
        ├── 01_member_benefits_and_coverage.pdf
        ├── 02_claims_processing_and_adjudication.pdf
        ├── 03_prior_authorization_policy.pdf
        ├── 04_claim_denials_and_appeals.pdf
        ├── 05_provider_network_and_reimbursement.pdf
        ├── 06_pharmacy_benefits_and_formulary.pdf
        ├── 07_medical_necessity_and_clinical_review.pdf
        └── 08_member_grievances_and_escalation.pdf
```

## Run locally

Requires Python 3.11 and a Groq API key.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env     # On Windows, copy the file manually if needed
```

Add your **own** Groq key to `.env`:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
```

Start the app from the repository root:

```bash
streamlit run streamlit_app.py
```

On import, `app.py` reads PDFs from `data/raw/`, builds embeddings and a BM25 index, and upserts embeddings into the local Chroma collection in `vector_store/`. The initial run may download embedding/reranker model files. That directory is generated locally and is intentionally not committed.

## Example questions

- What are typical reasons a claim might be denied?
- Which services commonly need prior authorization?
- How do copay, coinsurance and deductible differ?
- What information is useful for a claim appeal?
- Which member grievances require escalation?

## Implementation notes and limitations

- The Python source is included exactly as provided; no pipeline behavior has been rewritten.
- Data ingestion and embedding happen at **module import**, rather than through a separate scheduled indexing job.
- Chroma IDs are based on chunk position (`chunk_0`, `chunk_1`, ...). The mapping between IDs and in-memory chunks assumes stable input ordering.
- Citation information is included in the generation prompt, but **automated verification of cited source references is not implemented**.
- Reranking is limited to the top three RRF candidates. The negative-score check is a heuristic, not calibrated confidence or an approval workflow.
- No formal evaluation results, deployment configuration, authentication layer, or production monitoring are supplied.
- The included eight PDFs are synthetic educational data. Do not rely on their statements for actual insurance coverage or medical decisions.

## Author

**Neha Dudhane** — [GitHub](https://github.com/nehadudhane)
