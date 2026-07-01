# config_multimodal.py
"""
Configuration for pooled medical image retrieval + report generation.

Design (simplified — no modality split):
  * Pool ALL radiology image-caption pairs from every dataset into ONE corpus.
  * Fine-tune ONE shared CLIP; validate on the datasets' val splits during
    training (contrastive val loss / in-batch retrieval accuracy).
  * Build ONE FAISS index from the TRAIN images, then APPEND the VAL images
    after training so the final gallery = train + val.
  * Evaluate end-to-end: sample N (=2000) query images, retrieve top-k (=3)
    similar images, generate a report with MedGemma (few-shot, or zero-shot
    if nothing similar is found), and score the report against the query's
    ground-truth caption with BERTScore / BLEU / ROUGE. Save all predictions
    and ground truths for inspection.

Report generation (MedGemma) is unchanged and lives in vlm_report_generator.

NOTE ON DATASET IDS / COLUMNS
  ROCOv2 (eltorio/ROCOv2-radiology) verified: image / image_id / caption / cui,
  splits train|validation|test. Other entries are best-effort defaults —
  verify each hf_id / column mapping on its HF page. A source that fails to
  load or has unmappable columns is skipped with a warning, not a crash.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
from paths import HF_HUB_STR


# ================================================================== #
#  Per-dataset specification
# ================================================================== #

@dataclass
class DatasetSpec:
    name: str
    hf_id: str
    image_column: str = "image"
    caption_columns: List[str] = field(default_factory=lambda: ["caption"])
    # logical split -> this dataset's actual split name(s)
    split_map: Dict[str, str] = field(
        default_factory=lambda: {
            "train": "train", "val": "validation", "test": "test"
        }
    )
    streaming: bool = False
    enabled: bool = True
    # True for inherently-radiology datasets (kept wholesale). For mixed
    # datasets (e.g. MedICaT) set False → a radiology keyword filter applies.
    is_radiology: bool = True
    max_samples: Optional[int] = None
    config_name: Optional[str] = None

    # ── Caption from a LLaVA-style `conversations` field (RadImageNet-VQA) ──
    # When True, the caption is extracted from `conversations_column` via
    # radimagenet_vqa.caption_from_record instead of caption_columns.
    caption_from_conversations: bool = False
    conversations_column: str = "conversations"
    # If `image` is a RELATIVE PATH string (not embedded bytes), resolve it
    # against this directory. None → use the value as-is (works when the Hub
    # stores an Image() feature returning PIL/bytes).
    image_base_dir: Optional[str] = None
    # Prefer this column for the stable image id (RadImageNet uses "id").
    image_id_column: Optional[str] = None


def default_dataset_specs() -> List[DatasetSpec]:
    return [
        DatasetSpec(
            name="rocov2", hf_id="eltorio/ROCOv2-radiology",
            image_column="image", caption_columns=["caption"],
            split_map={"train": "train", "val": "validation", "test": "test"},
            is_radiology=True,
        ),
        DatasetSpec(
            name="roco", hf_id="mdwiratathya/ROCO-radiology",
            image_column="image", caption_columns=["caption"],
            split_map={"train": "train", "val": "validation", "test": "test"},
            is_radiology=True,
        ),
        DatasetSpec(
            name="mimic_cxr", hf_id="itsanmolgupta/mimic-cxr-dataset",
            image_column="image", caption_columns=["findings", "impression"],
            # Verified: this dataset is Parquet with ONLY a 'train' split
            # (30,633 examples). No val/test → train-only; a held-out slice
            # is carved from train at load time (see val_fraction_from_train).
            split_map={"train": "train"},
            is_radiology=True,
        ),
        # ── RadImageNet-VQA (alignment config) — CT/MRI, 1 caption/image ──
        # GATED: accept the Research Use Agreement on HF first; HF_TOKEN must
        # belong to that account. The caption lives in a LLaVA-style
        # `conversations` field, so caption_from_conversations=True. The HF
        # config name is assumed "alignment" (see radimagenet_vqa.py). Large
        # (750k train / 83k val) — consider max_samples for a first run.
        DatasetSpec(
            name="radimagenet_vqa", hf_id="raidium/RadImageNet-VQA",
            config_name="alignment",
            image_column="image", image_id_column="id",
            caption_from_conversations=True, conversations_column="conversations",
            split_map={"train": "train", "val": "validation"},
            is_radiology=True,
            # image_base_dir="/path/to/radimagenet/images",  # if image is a rel path
            # max_samples=100000,  # uncomment to cap the first build
        ),
        # ── IU-CXR (Indiana / Open-I) — all chest X-ray ──
        # NOTE: disabled by default. The previously-guessed id
        # 'Gladiator/IU_XRay' does NOT exist on the Hub. HF mirrors exist
        # (e.g. ayyuce/Indiana_University_Chest_X-ray_Collection,
        # ykumards/open-i) but their column schemas differ and are
        # unverified. Set enabled=True and fix hf_id/caption_columns after
        # confirming the schema on the dataset's HF page.
    ]


# ================================================================== #
#  Radiology filter (only applied to non-radiology datasets)
# ================================================================== #

@dataclass
class RadiologyFilterConfig:
    enabled: bool = True
    keywords: List[str] = field(default_factory=lambda: [
        "x-ray", "x ray", "xray", "radiograph", "chest film", "plain film",
        "cxr", "roentgen", "fluoroscopy", "mammogram", "mammography",
        "ct ", "ct scan", "computed tomography", "cta", "hrct", "cect",
        "mri", "magnetic resonance", "t1-weighted", "t2-weighted", "flair",
        "dwi", "mr imaging", "mra", "ultrasound", "ultrasonography",
        "sonography", "sonographic", "doppler", "echocardiogram",
        "angiography", "angiogram", "pet scan", "pet-ct", "scintigraphy",
        "spect", "myelography", "venography", "radiology", "radiological",
    ])


# Coarse modality guess stored in metadata for inspection only (NOT used for
# indexing or retrieval). "other" when no keyword matches.
MODALITY_GUESS_RULES = {
    "mri": ["mri", "magnetic resonance", "t1-weighted", "t2-weighted",
            "flair", "dwi", "mr imaging", "mra", "t1w", "t2w"],
    "ct": ["ct ", "ct scan", "computed tomography", "cta", "hrct", "cect",
           "ncct"],
    "xray": ["x-ray", "x ray", "xray", "radiograph", "chest film",
             "plain film", "cxr", "roentgen", "mammogra"],
    "ultrasound": ["ultrasound", "ultrasonography", "sonograph", "doppler",
                   "echocardiogram"],
    "angiography": ["angiography", "angiogram", "venography"],
    "pet": ["pet scan", "pet-ct", "scintigraphy", "spect"],
}


def guess_modality(caption: str) -> str:
    text = f" {caption.lower()} "
    for modality, kws in MODALITY_GUESS_RULES.items():
        if any(kw in text for kw in kws):
            return modality
    return "other"


# ================================================================== #
#  CLIP model + fine-tune
# ================================================================== #

@dataclass
class CLIPModelConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    max_text_length: int = 77


@dataclass
class FineTuneConfig:
    num_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    temperature: float = 0.07
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    unfreeze_visual_layers: int = 4
    unfreeze_text_layers: int = 4
    use_projection_head: bool = True
    projection_dim: int = 256
    projection_hidden_dim: int = 1024
    projection_num_layers: int = 3
    projection_dropout: float = 0.1
    label_smoothing: float = 0.1
    save_every_n_epochs: int = 10
    num_workers: int = 4

    # ── Loss / contrastive learning ──
    # loss_type: "infonce" (base CLIP) or "combined" (InfoNCE + hard-neg margin)
    # Hard-negative mining is DISABLED — plain symmetric InfoNCE.
    loss_type: str = "infonce"
    use_hard_negatives: bool = False
    num_hard_negatives: int = 5
    hard_negative_weight: float = 0.5
    margin: float = 0.2
    mining_start_epoch: int = 2
    use_memory_bank: bool = False
    memory_bank_size: int = 4096


# ================================================================== #
#  FAISS (single pooled index)
# ================================================================== #

@dataclass
class FAISSConfig:
    index_type: str = "Flat"     # Flat=exact; IVFFlat/HNSW for large galleries
    nlist: int = 256
    nprobe: int = 16
    use_gpu: bool = False
    normalize_embeddings: bool = True
    index_dir: str = "./faiss_index"
    encode_batch_size: int = 128

    # ── PCA whitening (fit on the gallery at build time, applied to gallery
    #    AND query vectors). Decorrelates embedding dims and downweights
    #    dominant directions, which typically sharpens cosine retrieval. ──
    use_whitening: bool = False
    reduce_dimensions: bool = False   # also project to target_dim if True
    target_dim: int = 256             # output dim when reduce_dimensions

    # ── Neighbor-image reload (so retrieved images can be shown to the VLM) ──
    # Store a reload reference per gallery vector: a lightweight (hf_id, split,
    # row) for map-style datasets, and a small JPEG thumbnail for streamed
    # datasets (which have no random access). Enables cross-process image
    # reload during evaluation.
    store_image_refs: bool = True
    store_streamed_thumbnails: bool = True
    thumbnail_max_size: int = 384


# ================================================================== #
#  End-to-end evaluation (retrieve → generate → score)
# ================================================================== #

@dataclass
class EvaluationConfig:
    num_eval_samples: int = 500

    # ── Retrieval mode ──
    # "threshold": use ALL gallery images with score >= retrieval_threshold
    #              (capped at max_retrieved), then feed the images + captions
    #              to MedGemma. Zero-shot if none qualify.
    # "top_k"    : legacy fixed-k behavior.
    retrieval_mode: str = "threshold"
    retrieval_threshold: float = 0.25
    max_retrieved: int = 20          # safety cap on candidates pulled
    top_k: int = 3                   # used only when retrieval_mode == "top_k"

    # Of the retrieved candidates, how many images to actually SHOW MedGemma
    # (all retrieved captions are always used as text). Bounds VRAM/context.
    max_few_shot_images: int = 3

    report_max_tokens: int = 100   # short single paragraph (no headers)
    eval_source: str = "all"
    exclude_self: bool = False
    bertscore_model_type: str = "roberta-large"
    bertscore_rescale_with_baseline: bool = True
    predictions_filename: str = "pipeline_eval_predictions.csv"
    summary_filename: str = "pipeline_eval_summary.json"
    allow_empty_reports: bool = False


# ================================================================== #
#  Knowledge graph + deployment (single source of truth for the app)
# ================================================================== #

@dataclass
class KGDeploymentConfig:
    """
    Shared by the full pipeline (vqa_pipeline) and the API so the app and the
    offline eval can't drift apart. Env vars in api.py override these.
    """
    use_kg: bool = True
    kg_backend: str = "dictionary"        # "dictionary" (no spaCy/numpy) | "scispacy"
    # Provide ONE concept source for the dictionary backend:
    kg_cache_path: Optional[str] = "./kg_cache.pkl"   # from build_kg_cache.py
    kg_jsonl_path: Optional[str] = None               # scispaCy KB JSONL
    kg_relation_table_path: Optional[str] = None      # local MRREL-style TSV

    # QA adapter: leave None to auto-resolve from config_genqa's output_dir
    # (so it tracks whichever run you trained, e.g. ./outputs_genqa_cat).
    qa_adapter_dir: Optional[str] = None
    qa_max_new_tokens: int = 15


# ================================================================== #
#  Top-level config
# ================================================================== #

@dataclass
class MultiModalConfig:
    model: CLIPModelConfig = field(default_factory=CLIPModelConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    faiss: FAISSConfig = field(default_factory=FAISSConfig)
    radiology: RadiologyFilterConfig = field(
        default_factory=RadiologyFilterConfig
    )
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    kg: KGDeploymentConfig = field(default_factory=KGDeploymentConfig)
    datasets: List[DatasetSpec] = field(default_factory=default_dataset_specs)

    output_dir: str = "./output_retrieval"
    checkpoint_dir: str = "./checkpoints_retrieval"
    cache_dir: str = HF_HUB_STR
    device: str = "cuda"
    preferred_gpu_name: Optional[str] = "RTX PRO 6000"

    # Subsample every split to this fraction for fast smoke tests. None=all.
    sample_frac: Optional[float] = None
    streaming_full_size_hint: int = 10000
    random_seed: int = 42

    # For datasets that expose only a 'train' split (e.g. MIMIC-CXR), carve a
    # held-out validation slice of this fraction from train so they still
    # contribute val images. Applied deterministically by row index.
    val_fraction_from_train: float = 0.05

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.faiss.index_dir, exist_ok=True)
