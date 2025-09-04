# LightRAG

LightRAG is a lightweight Retrieval-Augmented Generation (RAG) research/demo repository focused on medical assistance and knowledge-graph backed question answering. The project provides utilities for chunking documents, inserting documents into a local vector/knowledge store, visualizing knowledge graphs, and running simple RAG-style experiments.

This README gives a quick orientation, setup steps, and pointers to the main scripts and data in the repo.

## Key features

- Document chunking and indexing tooling
- Lightweight local vector/knowledge store (JSON-based in `RAG_DataBase/`)
- Knowledge graph generation and interactive visualization
- Example medical assistance script and demo video

## demo Video
  

## Assumptions

- The repository is a local demo/research project (not production-ready).
- Python 3.8+ is assumed. If you use a newer Python, the code should still work in most cases.
- The project uses plain Python scripts; a `requirements.txt` is provided for dependencies.

If these assumptions are incorrect for your environment, adjust the Python version and dependency setup accordingly.

## Quick start

1. Create and activate a virtual environment (Windows PowerShell example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Inspect or run an example script. Two entry points in this repo are:

- `medical_Assistance.py` — a script focused on medical question answering utilities.
- `light_RAG.py` — a central RAG orchestration/demo script.

Run a script with:

```powershell
streamlit run medical_Assistance.py
# or
python light_RAG.py
```

Note: The exact CLI arguments or runtime behaviour depend on the script internals. If a script expects a web UI or additional setup, check that file's top comments for usage notes.

## Project layout

- `chunking.py` — tools to split large documents into chunks suitable for embedding/indexing.
- `insert_doc.py` — utilities to insert documents/chunks into the local stores.
- `light_RAG.py` — main demo/orchestration for RAG experiments.
- `medical_Assistance.py` — example script tailored to medical assistance use-cases.
- `graph_visible.py` — graph generation and visualization helper.
- `RAG_DataBase/` — local JSON graph and vector/kv stores used by the demos (contains graphml and JSON stores).
- `lib/` — front-end assets used by the HTML visualization (`vis-network`, `tom-select`, etc.).
- `requirements.txt` — Python package list for the project.
- `LightRAG demo video.mp4` — a short demo video showing the project in action.

## Data and artifacts

- `RAG_DataBase/graph_chunk_entity_relation.graphml` — example graph exported from the pipeline.
- JSON stores such as `vdb_chunks.json`, `vdb_entities.json`, and `vdb_relationships.json` contain the serialized local indices.
