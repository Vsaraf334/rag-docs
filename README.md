# rag-docs

Retrieval-augmented generation (RAG) over my own documents.

## Layout

- `data/`: source documents to index (put your files here)
- `src/ingest.py`: load, chunk, embed, store
- `src/retrieve.py`: find the most relevant chunks for a question
- `src/answer.py`: build the prompt and call the model with citations
- `evals/questions.jsonl`: test questions with expected answers (keep this set stable)
- `evals/run_eval.py`: scores the system and prints accuracy

## Design decisions

_Record chunk size, embedding model, top-k, and why you chose each._

## Eval results

| Date | Change | Accuracy |
|------|--------|----------|
|      |        |          |
