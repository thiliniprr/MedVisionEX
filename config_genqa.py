# config_genqa.py
"""
Configuration for generative QA fine-tuning of MedGemma 4B-IT.

Task: (caption, question) -> short free-text answer (<= ~15 tokens).
Replaces the discriminative MedCaptionVQANet classifier with a
generative model fine-tuned via QLoRA.
"""

import os

# ================================================================
# DATA
# ================================================================
from paths import DATASET_ROOT_STR, HF_HUB_STR
DATASET_ROOT = DATASET_ROOT_STR

data_config = {
    # ── DATA SOURCE ──
    # "categorical_csv" : use the local Train/Val_Categorical.csv + captions CSV
    # "radimagenet"     : pull caption+QA directly from raidium/RadImageNet-VQA
    #                     (alignment caption + instruction QA for train,
    #                      validation split for eval). GATED — accept the RUA on HF.
    'data_source': 'radimagenet',

    # ── RadImageNet-VQA settings (used when data_source == 'radimagenet') ──
    'radimagenet_caption_config': 'alignment',   # captions (1/image)
    'radimagenet_train_qa_config': 'instruct',     # QA for training (train split)
    'radimagenet_eval_qa_config': 'instruct',      # QA for validation (val split)
    'radimagenet_train_split': 'train',            # finetune on train
    'radimagenet_val_split': 'validation',         # validate on val
    'radimagenet_eval_split': 'validation',        # (benchmark set NOT used)
    # Caps for a first manageable run (None = use everything; 7.5M is huge):
    'radimagenet_max_images': None,    # cap unique images for caption table
    'radimagenet_max_pairs': None,     # cap QA pairs
    'radimagenet_caption_source': 'index',  # 'index' (join alignment[i] to
                                            # instruct[i] by row position — no
                                            # id field exists), 'join' (by id),
                                            # or 'self' (in-record description)

    # QA pairs: one "image_id|question|answer" per line
    #'train_qa_path': os.path.join(DATASET_ROOT, "All_QA_Pairs_train.txt"),
    #'val_qa_path':   os.path.join(DATASET_ROOT, "All_QA_Pairs_val.txt"),
    
    'train_qa_path': os.path.join(DATASET_ROOT, "Train_Categorical.csv"),
    'val_qa_path':   os.path.join(DATASET_ROOT, "Val_Categorical.csv"),

    # Pre-generated MedGemma captions (columns: Image_ID, Caption)
    'captions_file': os.path.join(
        DATASET_ROOT, "medgemma_captions_all_images.csv"
    ),

    # Fallback caption when an image id has no generated caption
    'default_caption': "Medical image showing anatomical structures.",

    # Field separator in the QA txt files
    'qa_separator': '|',

    # Token limits
    'max_caption_tokens': 256,    # captions are truncated to this many tokens
    'max_answer_tokens': 32,      # training-time answer truncation
    'max_seq_length': 512,        # hard cap on full (prompt + answer) sequence

    'random_seed': 42,
}

# ================================================================
# MODEL / LoRA
# ================================================================
model_config = {
    'model_name': 'google/medgemma-4b-it',
    'cache_dir': HF_HUB_STR,
    # Gated model: export HF_TOKEN=... before running (do NOT hardcode)
    'hf_token': os.environ.get('HF_TOKEN'),

    'output_dir': './outputs_genqa_cat',

    # ── Hardware: NVIDIA RTX PRO 6000 (Blackwell, 96 GB) ──
    # Substring matched against torch.cuda.get_device_name(i); the first
    # GPU whose name contains this is used. Set to None to use cuda:0.
    'preferred_gpu_name': 'RTX PRO 6000',

    # Quantization — DISABLED: 96 GB fits the 4B model in bf16 with
    # plenty of headroom; bf16 LoRA trains faster than 4-bit QLoRA.
    'load_in_4bit': False,
    'bnb_4bit_quant_type': 'nf4',
    'bnb_4bit_use_double_quant': True,
    'torch_dtype': 'bfloat16',
    'attn_implementation': 'eager',   # recommended for Gemma-3 training

    # LoRA — scoped to the language model only (vision tower unused)
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    # Regex: match q/k/v/o + MLP projections inside the language model
    'lora_target_modules': (
        r".*language_model.*(q_proj|k_proj|v_proj|o_proj|"
        r"gate_proj|up_proj|down_proj)$"
    ),

    # Training — sized for 96 GB VRAM
    'num_epochs': 10,
    'batch_size': 16,
    'gradient_accumulation_steps': 1,    # effective batch = 16
    'learning_rate': 2e-4,
    'weight_decay': 0.01,
    'warmup_ratio': 0.05,
    'max_grad_norm': 1.0,
    'gradient_checkpointing': False,     # not needed at 96 GB; faster off
    'num_workers': 4,
    'log_frequency': 50,

    # Evaluation — batch sizes sized for 96 GB VRAM
    'train_generation_samples': 300,  # train subset for generative metrics
    'eval_generation_samples': 300,   # val subset for generative metrics
    'gen_eval_batch_size': 32,
    'gen_max_new_tokens': 15,         # short-answer cap (user requirement)
    'gen_do_sample': False,           # greedy for deterministic eval
    'early_stopping_patience': 3,     # epochs without improvement

    # Generative metrics
    'bertscore_model_type': 'roberta-large',
    'bertscore_batch_size': 128,
    'bertscore_rescale_with_baseline': False,  # True needs baseline download
    'best_metric': 'bertscore_f1',    # checkpoint selection (on validation)

    # Prompting
    'system_prompt': (
        "You are a medical visual question answering assistant. "
        "You are given a textual description of a medical image and a "
        "question about that image. Answer the question concisely, in "
        "at most 15 words, using only the description. Output only the "
        "answer, with no explanation."
    ),
}

os.makedirs(model_config['output_dir'], exist_ok=True)
