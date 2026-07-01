# MedVisionEX

```
medvisionex/
├── paths.py                  # relative-path resolver (repo root + env overrides)
│
├── knowledge_graph/          # ── KG module (done) ──
│   ├── __init__.py           #   exports KGAugmentor, KGConfig, DictionaryKB
│   ├── kg_integration.py     #   UMLS/SNOMED/RadLex augmentor
│   └── build_kg_cache.py     #   build the concept cache
│
├── caption_generator/        # ── captioner module (scaffolded) ──
│   └── __init__.py           #   placeholder; see note below
│
│   # ── main pipeline + app (project root) ──
├── vqa_pipeline.py           # single pooled VQA pipeline
├── modality_vqa_pipeline.py  # modality-routed VQA pipeline
├── api.py                    # FastAPI backend
├── streamlit_app.py          # Streamlit frontend
│
│   # retrieval, QA, configs, eval (root, unchanged)
├── config*.py, run_*.py, train_genqa.py, evaluate_*.py, ...

```

## Run from a fresh clone

```bash
git clone <repo-url> medvisionex && cd medvisionex
conda create -n medvqa python=3.11 -y && conda activate medvqa
# install cu128 torch for your GPU, then:
pip install -r requirements-app.txt
huggingface-cli login          # accept the MedGemma license first
# everything caches into ./.hf_cache and writes to ./outputs by default
python run_multimodal_retrieval.py --stage all
```
