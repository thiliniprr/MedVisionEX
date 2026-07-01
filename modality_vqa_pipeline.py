# modality_vqa_pipeline.py
"""
ModalityVQAPipeline — a drop-in replacement for VQAPipeline that routes each
query to a modality-specific CLIP encoder + FAISS index (CT / MRI / X-ray).

Design (no existing code changed)
---------------------------------
* The heavy, modality-independent components — the VLM captioner, the generative
  QA model and the KG augmentor — are loaded ONCE onto a host VQAPipeline
  instance (we replicate only the shared part of VQAPipeline.load so we don't
  also load the unused default CLIP/FAISS).
* Each modality's (CLIP, FAISS, gallery loader) is loaded from its own
  "<base>_<modality>" directories.
* Per query we classify the modality (ModalityRouter), swap the host's
  _clip/_faiss/_img_loader to that modality's, then reuse the host's own
  caption_for / answer_for — so retrieval is modality-routed while captioning,
  KG and QA stay shared and identical to the single-model pipeline.

Same public surface as VQAPipeline: load(), caption_for(), answer_for(),
answer(); each output also carries the chosen "modality".
"""

import logging
from typing import Optional

from config_multimodal import MultiModalConfig
from config_modality import modality_config
from vqa_pipeline import VQAConfig, VQAPipeline
from modality_utils import ModalityRouter

logger = logging.getLogger("modality_vqa_pipeline")


class ModalityVQAPipeline:
    def __init__(self, config: Optional[VQAConfig] = None,
                 mod_config=modality_config):
        self.config = config or VQAConfig()
        self.mod_config = mod_config
        self.host: Optional[VQAPipeline] = None      # shared VLM/QA/KG + swap slot
        self.enc = {}                                 # modality -> (clip,faiss,loader)
        self.router: Optional[ModalityRouter] = None
        self._classifier_clip = None

    # ------------------------------------------------------------------ #

    def _load_shared(self, device):
        """Replicate ONLY the shared (non-retrieval) part of VQAPipeline.load
        onto a host instance, so the VLM/QA/KG are loaded exactly once."""
        host = VQAPipeline(self.config)

        # ── VLM caption generator (mirror of VQAPipeline.load) ──
        from config import PipelineConfig
        pcfg = PipelineConfig()
        pcfg.vlm.max_new_tokens = self.config.report_max_tokens
        pcfg.ollama_vlm.max_new_tokens = self.config.report_max_tokens
        host._vlm_backend = getattr(pcfg, "vlm_backend", "local")
        if host._vlm_backend == "ollama":
            from vlm_report_generator import VLMReportGenerator
            host._vlm = VLMReportGenerator(pcfg)
        else:
            from caption_generator import _make_threshold_generator_class
            GenCls = _make_threshold_generator_class()
            host._vlm = GenCls(pcfg)
            host._vlm.max_few_shot_images = self.config.max_few_shot_images

        # ── Generative QA ──
        from infer_genqa import GenerativeQAService
        host._qa = GenerativeQAService(self.config.qa_adapter_dir, device=device)
        host._qa.load()

        # ── KG augmentor ──
        if self.config.use_kg:
            from knowledge_graph import KGAugmentor, KGConfig
            kg_cfg = KGConfig(
                backend=self.config.kg_backend,
                kb_cache_path=self.config.kg_cache_path,
                kb_jsonl_path=self.config.kg_jsonl_path,
                relation_table_path=self.config.kg_relation_table_path)
            host._kg = KGAugmentor(kg_cfg)
        return host

    def _load_modality_encoder(self, modality, device):
        from multimodal_clip_finetuner import MultiModalCLIPFineTuner
        from multimodal_faiss_builder import PooledFAISSIndex, GalleryImageLoader
        paths = self.mod_config.paths(modality)
        mm = MultiModalConfig()
        mm.checkpoint_dir = paths.checkpoint_dir
        mm.faiss.index_dir = paths.index_dir
        clip = MultiModalCLIPFineTuner(mm, device=device)
        clip.load_checkpoint(paths.checkpoint_name)
        faiss = PooledFAISSIndex(mm)
        faiss.load()
        img_loader = GalleryImageLoader(mm)
        logger.info(f"[{modality}] loaded CLIP={paths.checkpoint_dir} "
                    f"FAISS={paths.index_dir}")
        return clip, faiss, img_loader

    def load(self):
        import torch
        from multimodal_clip_finetuner import MultiModalCLIPFineTuner

        device = self.config.device or (
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.host = self._load_shared(device)

        for m in self.mod_config.modalities:
            self.enc[m] = self._load_modality_encoder(m, device)

        # Router: unbiased BASE CLIP for zero-shot classification (or reuse a
        # fine-tuned encoder if so configured).
        if self.mod_config.classifier_use_base_clip:
            self._classifier_clip = MultiModalCLIPFineTuner(
                MultiModalConfig(), device=device)   # pretrained, no checkpoint
        else:
            self._classifier_clip = self.enc[self.mod_config.default_modality][0]
        self.router = ModalityRouter(
            clip=self._classifier_clip,
            modalities=self.mod_config.modalities,
            use_text_hint=self.mod_config.use_text_hint,
            use_clip_zeroshot=self.mod_config.use_clip_zeroshot,
            default_modality=self.mod_config.default_modality)
        logger.info("ModalityVQAPipeline loaded "
                    f"({len(self.enc)} modality encoders).")

    # ------------------------------------------------------------------ #

    def _route(self, modality):
        clip, faiss, loader = self.enc[modality]
        self.host._clip = clip
        self.host._faiss = faiss
        self.host._img_loader = loader

    def classify_modality(self, image=None, question: Optional[str] = None) -> str:
        return self.router.classify(image=image, question=question)

    def caption_for(self, query_image, question: Optional[str] = None,
                    include_images: bool = False) -> dict:
        modality = self.classify_modality(image=query_image, question=question)
        self._route(modality)
        out = self.host.caption_for(query_image, include_images=include_images)
        out["modality"] = modality
        return out

    def answer_for(self, caption: str, question: str,
                   use_kg: Optional[bool] = None) -> dict:
        # caption/KG/QA are shared; no routing needed here.
        return self.host.answer_for(caption, question, use_kg=use_kg)

    def answer(self, image_path: str = None, question: str = "",
               use_kg: Optional[bool] = None, **kwargs) -> dict:
        """End-to-end for the app: classify from the loaded image + question,
        route, then run the shared caption→answer path."""
        from multimodal_dataset_loader import load_pil_image
        img = load_pil_image(image_path)
        modality = self.classify_modality(image=img, question=question)
        self._route(modality)
        cap = self.host.caption_for(img)
        ans = self.host.answer_for(cap["caption"], question, use_kg=use_kg)
        return {"modality": modality, "caption": cap["caption"],
                "num_retrieved": cap.get("num_retrieved"),
                "answer": ans["answer"], "kg_used": ans.get("kg_used")}
