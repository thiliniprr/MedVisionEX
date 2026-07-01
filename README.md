# MedVisionEX

The repo is portable: all paths resolve relative to the repo root (see
`paths.py`) and are overridable by environment variables, so a fresh clone runs
without editing any absolute paths.

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
└── Dockerfile, docker-compose.yml, requirements-app.txt, README.md
```

## Path portability (done)

`paths.py` anchors everything to the repo root. Defaults, all overridable:

| Variable | Default | Purpose |
|---|---|---|
| `HF_HOME` | `<repo>/.hf_cache` | HuggingFace model/dataset cache |
| `MEDVQA_DATA` | `<repo>/data` | dataset root |
| `MEDVQA_IMAGES` | `<data>/Images` | local image dir |
| `MEDVQA_OUTPUTS` | `<repo>/outputs` | checkpoints / indices / results |

Importing any config sets `HF_HOME` to the repo-local cache unless you've
already exported one, so HuggingFace caches inside the repo by default.

## Migration status

- **Relative paths (incl. HuggingFace):** done — no absolute paths remain.
- **No AI/tool identifiers in the code:** confirmed — none present.
- **`knowledge_graph/` package:** done and import-verified.
- **`caption_generator/` package:** scaffolded. The captioner currently lives in
  `evaluate_pipeline.py` (`_make_threshold_generator_class`) + `config.py`
  (`PipelineConfig`); extracting it cleanly is a follow-up that refactors
  `evaluate_pipeline.py`.
- **Per-file comment cleanup:** comments in touched files are purposeful; a
  pass over the remaining root modules can be done incrementally.

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
