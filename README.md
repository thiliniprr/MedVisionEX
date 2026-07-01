# MedVisionEx

**A DIKW-Inspired, Retrieval-Augmented and
Knowledge-Graph-Grounded Pipeline for Explainable Medical Visual
Question Answering**

MedVisionEx answers natural-language questions about medical images by combining
cross-modal retrieval, vision-language captioning, biomedical knowledge-graph
grounding, and a fine-tuned generative QA model into a single pipeline, served
through a FastAPI backend and a Streamlit UI.

> Research prototype accompanying an ADBIS 2026 demo paper. Not for clinical use.

```
medvisionex/
├── paths.py                  # relative-path resolver (repo root + env overrides)
│
├── knowledge_graph/          # ── KG module (done) ──
│   ├── __init__.py           #   exports KGAugmentor, KGConfig, DictionaryKB
│   ├── kg_integration.py     #   UMLS/SNOMED/RadLex augmentor
│   └── build_kg_cache.py     #   build the concept cache
│
├── caption_generator/        # ── captioner module (done) ──
│   ├── __init__.py           #   exports _make_threshold_generator_class
│   └── generator.py          #   MedGemma paragraph-caption generator
│
│   # ── main pipeline + app (project root) ──
├── vqa_pipeline.py           # single pooled VQA pipeline
├── modality_vqa_pipeline.py  # modality-routed VQA pipeline
├── api.py                    # FastAPI backend
├── streamlit_app.py          # Streamlit frontend
│
│   # retrieval, QA, configs, eval
├── config*.py, run_*.py, train_genqa.py, ...
```

## Run from a fresh clone

```bash
git clone medvisionex && cd medvisionex
conda create -n medvqa python=3.11 -y && conda activate medvqa
# install cu128 torch for your GPU, then:
pip install -r requirements-app.txt
huggingface-cli login          # accept the MedGemma license first
# everything caches into ./.hf_cache and writes to ./outputs by default
python run_multimodal_retrieval.py --stage all
```

## Citation

If you use this work, please cite the accompanying paper:

```bibtex
@inproceedings{medvisionex2026,
  title     = {NextGen MedVisionEx: Retrieval-Augmented, Knowledge-Grounded
               Visual Question Answering for Radiology},
  author    = {Nadeesha Perera, Alli Raittinen, Jyrki Nummenmaa, Konstantinos Stefanidis},
  booktitle = {ADBIS},
  year      = {2026}
}
```

## Acknowledgements

Built on MedGemma, OpenAI CLIP, FAISS, and the RadImageNet-VQA, MIMIC-CXR, ROCO,
VQA-RAD, SLAKE, and VQA-Med datasets. Respect each dataset's and model's license
and access terms.
