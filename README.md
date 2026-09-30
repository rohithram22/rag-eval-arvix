# rag-eval-arxiv

Question answering over a frozen corpus of ~150 arXiv papers on retrieval-augmented
generation, with an evaluation layer that compares RAG configurations and shows,
with numbers, which one works best.

> Status: 🚧 in progress (Step 1 of 10: scaffold)

## Why this project
Most RAG demos stop at "it answers questions." This one asks *how well*, and *which
design choices matter*: chunk size, dense vs. hybrid retrieval, and reranking,
measured on a manually reviewed golden set.

## Architecture
_Diagram added in Step 10._

## Results
_Results table and write-up added in Step 8._

## Quickstart
```bash
bash scripts/setup.sh           # venv + CPU-only deps
source .venv/bin/activate
python scripts/check_env.py     # verify environment, keys, models
```
Requires `GROQ_API_KEY` and `GOOGLE_API_KEY` (free tiers) as env vars or in `.env`.

## Project structure
```
configs/        experiment configuration (base.yaml + overrides)
scripts/        setup and runnable pipeline steps
src/rag_eval/   library code (config, LLMs, ingestion, retrieval, evaluation)
data/           raw + processed corpus (gitignored, rebuilt by scripts)
eval/           golden Q/A set (committed)
results/        metrics tables (committed)
app/            Streamlit demo
```

## Roadmap
- [x] 1. Repo, environment, config system
- [ ] 2. Corpus download (frozen)
- [ ] 3. Ingestion: parsing, cleaning, chunking
- [ ] 4. Baseline RAG (bge-small + FAISS + Groq)
- [ ] 5. Golden eval set
- [ ] 6. Evaluation harness
- [ ] 7. Experiments
- [ ] 8. Results + write-up
- [ ] 9. Streamlit demo
- [ ] 10. Final README

## License
MIT
