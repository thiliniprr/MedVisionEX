# config.py
"""
Central configuration — updated with MedGemma 4B-IT support,
cosine matching improvements, VLM-based report generation,
and Ollama remote VLM support.
"""
import os
from paths import HF_HUB_STR
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    embedding_dim: int = 512
    max_text_length: int = 77
    image_size: int = 224


@dataclass
class DatasetConfig:
    dataset_name: str = "itsanmolgupta/mimic-cxr-dataset"
    image_column: str = "image"
    text_columns: List[str] = field(
        default_factory=lambda: ["findings", "impression"]
    )
    text_separator: str = " "
    caption_prefix: str = ""
    caption_column: str = "findings"
    train_split: str = "train"
    val_split: str = "test"
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = 500


@dataclass
class FineTuneConfig:
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    temperature: float = 0.07
    save_every_n_epochs: int = 2
    eval_every_n_steps: int = 500
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    unfreeze_visual_layers: int = 4
    unfreeze_text_layers: int = 4
    use_projection_head: bool = True
    projection_dim: int = 256

    # Improved projection head parameters
    projection_num_layers: int = 3
    projection_hidden_dim: int = 1024
    projection_dropout: float = 0.1
    use_improved_projection: bool = True

    # Hard negative mining parameters
    use_hard_negatives: bool = True
    num_hard_negatives: int = 5
    hard_negative_search_k: int = 50
    mining_start_epoch: int = 2
    hard_negative_weight: float = 0.5

    # Loss configuration
    loss_type: str = "combined"
    margin: float = 0.2
    loss_temperature: float = 0.05


@dataclass
class FAISSConfig:
    index_type: str = "IVFFlat"
    nlist: int = 100
    nprobe: int = 10
    use_gpu: bool = False
    normalize_embeddings: bool = True
    index_save_path: str = "./faiss_index"

    # Embedding post-processing
    use_whitening: bool = True
    reduce_dimensions: bool = False
    target_dim: int = 256
    save_preprocessing: bool = True


@dataclass
class RetrievalConfig:
    top_k: int = 5
    similarity_threshold: float = 0.3
    report_max_length: int = 512
    aggregation_method: str = "weighted"

    # Query expansion parameters
    use_query_expansion: bool = True
    expansion_rounds: int = 1
    expansion_top_k_feedback: int = 3
    expansion_alpha: float = 0.7
    expansion_beta: float = 0.3

    # Re-ranking parameters
    use_reranking: bool = True
    rerank_candidate_pool: int = 50

    # Multi-modal search parameters
    use_multimodal_search: bool = False
    image_weight: float = 0.6
    text_weight: float = 0.4


@dataclass
class VLMConfig:
    """Configuration for local Vision-Language Model report generation."""

    # ── MedGemma 4B-IT as default local model ──
    model_name: str = "google/medgemma-4b-it"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"

    image_size: int = 448
    max_new_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_beams: int = 1

    num_examples: int = 3
    modality_context: str = "chest X-ray radiology"
    system_prompt: Optional[str] = None
    include_scores_in_prompt: bool = True
    include_conditions_context: bool = True
    cache_model: bool = True
    enabled: bool = True

    # MedGemma uses Gemma chat template
    conv_style: str = "gemma"

    load_in_4bit: bool = True
    load_in_8bit: bool = False

    # HuggingFace token for gated model access.
    # Do NOT hardcode tokens. Export HF_TOKEN in the environment instead.
    hf_token: Optional[str] = field(
        default_factory=lambda: os.environ.get("HF_TOKEN")
    )

    # Local cache directory (inside project)
    local_model_dir: str = HF_HUB_STR


# ================================================================== #
#  Ollama Remote VLM Config — llava-llama3:8b
# ================================================================== #

@dataclass
class OllamaVLMConfig:
    """
    Configuration for Ollama-hosted remote VLM report generation.
    """

    # ============================================================
    # Connection settings
    # ============================================================
    host: str = "http://86.50.170.53:11434"

    # ============================================================
    # Model selection
    # ============================================================
    model_name: str = "llava-llama3:8b"

    # ============================================================
    # Generation parameters
    # ============================================================
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    seed: Optional[int] = None

    # ============================================================
    # Few-shot / prompt settings
    # ============================================================
    num_examples: int = 3
    modality_context: str = "chest X-ray radiology"
    system_prompt: Optional[str] = None
    include_scores_in_prompt: bool = True
    include_conditions_context: bool = True

    # ============================================================
    # Connection reliability
    # ============================================================
    timeout: int = 180
    max_retries: int = 3
    retry_delay: float = 2.0

    # ============================================================
    # Image encoding
    # ============================================================
    image_max_size: int = 1024
    image_quality: int = 90

    # ============================================================
    # Enable/disable
    # ============================================================
    enabled: bool = True

    # ============================================================
    # Streaming
    # ============================================================
    stream: bool = False

    # ============================================================
    # Keep-alive
    # ============================================================
    keep_alive: str = "150m"


@dataclass
class PipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    faiss: FAISSConfig = field(default_factory=FAISSConfig)
    retrieval: RetrievalConfig = field(
        default_factory=RetrievalConfig
    )
    vlm: VLMConfig = field(default_factory=VLMConfig)
    ollama_vlm: OllamaVLMConfig = field(
        default_factory=OllamaVLMConfig
    )

    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"
    cache_dir: str = HF_HUB_STR
    device: str = "cuda"

    # Which VLM backend to use: "local" or "ollama"
    vlm_backend: str = "local"

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.faiss.index_save_path, exist_ok=True)
        os.makedirs(self.vlm.local_model_dir, exist_ok=True)