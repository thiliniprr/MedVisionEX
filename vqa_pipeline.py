# vqa_pipeline.py
"""
End-to-end VQA inference, KG-augmented at both the caption and QA stages.

    image + question
      → CLIP encode + FAISS retrieve similar cases (images + captions)
      → VLM generates a caption/report from the retrieved cases
            ↑ KG augments this: linked findings passed as detected_conditions
      → generative QA LLM answers from (caption, question)
            ↑ KG augments this: concept grounding injected into the prompt
      → answer

This wires together components already in the project:
  * PooledFAISSIndex / MultiModalCLIPFineTuner   (retrieval)
  * VLMReportGenerator                            (caption, unchanged)
  * GenerativeQAService (infer_genqa)             (QA, KG-aware)
  * KGAugmentor (kg_integration)                  (UMLS/SNOMED/RadLex context)

The KG is built ONCE from the question + the generated caption, then reused
for both stages — the caption side uses the linked findings as conditions, the
QA side uses the full concept/relation text block.

Usage:
    from vqa_pipeline import VQAPipeline, VQAConfig
    pipe = VQAPipeline(VQAConfig(...))
    pipe.load()
    out = pipe.answer(image_path="scan.png",
                      question="Is there evidence of pneumonia?")
    print(out["answer"]); print(out["caption"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vqa")


@dataclass
class VQAConfig:
    """
    All fields default to None and are resolved from config_multimodal
    (retrieval threshold, max images, KG settings) and config_genqa (QA
    adapter) in __post_init__, so config_multimodal is the SINGLE source of
    truth shared with the offline eval. Pass an explicit value (or an env var
    via api.py) to override any one field.
    """
    # Retrieval
    clip_checkpoint: str = "best_model"
    retrieval_threshold: Optional[float] = None
    max_retrieved: Optional[int] = None
    max_few_shot_images: Optional[int] = None
    exclude_self: Optional[bool] = None     # drop the query's own gallery image

    # QA
    qa_adapter_dir: Optional[str] = None
    qa_max_new_tokens: Optional[int] = None
    report_max_tokens: Optional[int] = None

    # KG
    use_kg: Optional[bool] = None
    kg_backend: Optional[str] = None
    kg_cache_path: Optional[str] = None
    kg_jsonl_path: Optional[str] = None
    kg_relation_table_path: Optional[str] = None

    device: Optional[str] = None

    def __post_init__(self):
        from config_multimodal import MultiModalConfig
        mm = MultiModalConfig()
        ev, kg = mm.evaluation, mm.kg

        def pick(v, default):
            return default if v is None else v

        self.retrieval_threshold = pick(self.retrieval_threshold,
                                        ev.retrieval_threshold)
        self.max_retrieved = pick(self.max_retrieved, ev.max_retrieved)
        self.max_few_shot_images = pick(self.max_few_shot_images,
                                        ev.max_few_shot_images)
        self.exclude_self = pick(self.exclude_self, ev.exclude_self)
        self.report_max_tokens = pick(self.report_max_tokens,
                                      ev.report_max_tokens)
        self.use_kg = pick(self.use_kg, kg.use_kg)
        self.kg_backend = pick(self.kg_backend, kg.kg_backend)
        self.kg_cache_path = pick(self.kg_cache_path, kg.kg_cache_path)
        self.kg_jsonl_path = pick(self.kg_jsonl_path, kg.kg_jsonl_path)
        self.kg_relation_table_path = pick(self.kg_relation_table_path,
                                           kg.kg_relation_table_path)
        self.qa_max_new_tokens = pick(self.qa_max_new_tokens,
                                      kg.qa_max_new_tokens)

        # QA adapter: explicit > config_multimodal.kg > config_genqa output_dir
        if self.qa_adapter_dir is None:
            self.qa_adapter_dir = kg.qa_adapter_dir
        if self.qa_adapter_dir is None:
            try:
                from config_genqa import model_config as genqa_cfg
                import os
                self.qa_adapter_dir = os.path.join(
                    genqa_cfg["output_dir"], "checkpoints", "best_adapter")
            except Exception:
                self.qa_adapter_dir = "./outputs_genqa/checkpoints/best_adapter"


class VQAPipeline:
    def __init__(self, config: VQAConfig):
        self.config = config
        self._clip = None
        self._faiss = None
        self._img_loader = None
        self._vlm = None
        self._qa = None
        self._kg = None

    # ------------------------------------------------------------ #

    def load(self):
        import torch
        from config_multimodal import MultiModalConfig
        from multimodal_clip_finetuner import MultiModalCLIPFineTuner
        from multimodal_faiss_builder import PooledFAISSIndex, GalleryImageLoader

        device = self.config.device or (
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # ── Retrieval ──
        mm_cfg = MultiModalConfig()
        self._mm_cfg = mm_cfg
        self._clip = MultiModalCLIPFineTuner(mm_cfg, device=device)
        self._clip.load_checkpoint(self.config.clip_checkpoint)
        self._faiss = PooledFAISSIndex(mm_cfg)
        self._faiss.load()
        self._img_loader = GalleryImageLoader(mm_cfg)

        # ── VLM caption generator (paragraph style, short) ──
        from config import PipelineConfig
        pcfg = PipelineConfig()
        pcfg.vlm.max_new_tokens = self.config.report_max_tokens
        pcfg.ollama_vlm.max_new_tokens = self.config.report_max_tokens
        self._vlm_backend = getattr(pcfg, "vlm_backend", "local")
        if self._vlm_backend == "ollama":
            from vlm_report_generator import VLMReportGenerator
            self._vlm = VLMReportGenerator(pcfg)
        else:
            from caption_generator import _make_threshold_generator_class
            GenCls = _make_threshold_generator_class()
            self._vlm = GenCls(pcfg)
            self._vlm.max_few_shot_images = self.config.max_few_shot_images

        # ── Generative QA (KG-aware) ──
        from infer_genqa import GenerativeQAService
        self._qa = GenerativeQAService(self.config.qa_adapter_dir, device=device)
        self._qa.load()

        # ── KG augmentor ──
        if self.config.use_kg:
            from knowledge_graph import KGAugmentor, KGConfig
            kg_cfg = KGConfig(
                backend=self.config.kg_backend,
                kb_cache_path=self.config.kg_cache_path,
                kb_jsonl_path=self.config.kg_jsonl_path,
                relation_table_path=self.config.kg_relation_table_path)
            self._kg = KGAugmentor(kg_cfg)

        logger.info("VQA pipeline loaded.")

    # ------------------------------------------------------------ #

    def _retrieve(self, query_image):
        proc = self._clip.processor(images=query_image, return_tensors="pt")
        q_emb = self._clip.encode_image(
            proc["pixel_values"].to(self._clip.device)).cpu().numpy()
        results = self._faiss.search(
            q_emb, threshold=self.config.retrieval_threshold,
            max_results=self.config.max_retrieved,
            drop_self=self.config.exclude_self)
        # reload neighbor images for the ones we will show the VLM
        for j, r in enumerate(results):
            r["image_obj"] = (self._img_loader.load(r.get("reload"))
                              if j < self.config.max_few_shot_images else None)
        return results

    def _generate_caption(self, query_image, results, detected_conditions):
        rr = [{"caption": r["caption"], "score": r["score"],
               "modality": r.get("modality_guess"),
               "image_id": r.get("image_id"),
               "image_obj": r.get("image_obj")} for r in results]
        try:
            if self._vlm_backend == "ollama":
                out = (self._vlm.generate_report_ollama_few_shot(
                           query_image=query_image, retrieval_results=rr,
                           detected_conditions=detected_conditions)
                       if rr else
                       self._vlm.generate_report_ollama_zero_shot(
                           query_image=query_image,
                           detected_conditions=detected_conditions))
            else:
                out = (self._vlm.generate_report_few_shot(
                           query_image=query_image, retrieval_results=rr,
                           detected_conditions=detected_conditions)
                       if rr else
                       self._vlm.generate_report_zero_shot(
                           query_image=query_image,
                           detected_conditions=detected_conditions))
            return (out.get("raw_vlm_output") or out.get("report", "")
                    if isinstance(out, dict) else str(out))
        except Exception as e:
            logger.warning(f"caption generation failed: {e}")
            return ""

    # ------------------------------------------------------------ #

    def caption_for(self, query_image, include_images: bool = False) -> dict:
        """
        Generate the paragraph caption for an image. QUESTION-INDEPENDENT —
        retrieval depends only on the image and the KG-derived conditions come
        from the retrieved captions, so this can be cached per image and
        reused across all questions about that image.

        If include_images=True, each retrieved entry carries the loaded PIL
        neighbor image under "image_obj" (used by the API to return evidence
        thumbnails). Off by default so the eval path stays lightweight.
        """
        results = self._retrieve(query_image)
        detected_conditions = None
        if self._kg is not None:
            retrieved_text = " ".join(r["caption"] for r in results)
            # question="" → conditions come purely from the retrieved captions
            seed = self._kg.augment("", retrieved_text)
            detected_conditions = seed["detected_conditions"] or None
        caption = self._generate_caption(query_image, results,
                                         detected_conditions)
        retrieved = []
        for r in results:
            entry = {"caption": r["caption"], "score": r["score"],
                     "source": r.get("source"), "image_id": r.get("image_id")}
            if include_images:
                entry["image_obj"] = r.get("image_obj")
            retrieved.append(entry)
        return {"caption": caption, "num_retrieved": len(results),
                "retrieved": retrieved}

    def answer_for(self, caption: str, question: str,
                   use_kg: bool = True) -> dict:
        """KG pass over (question, caption) → KG-grounded QA answer."""
        kg_text, kg_info = "", {}
        if use_kg and self._kg is not None and caption:
            kg = self._kg.augment(question, caption)
            kg_text = kg["qa_knowledge_text"]
            kg_info = {k: v for k, v in kg.items() if k != "qa_knowledge_text"}
        qa_out = self._qa.answer(
            caption=caption, question=question,
            max_new_tokens=self.config.qa_max_new_tokens,
            knowledge_text=kg_text or None)
        return {"answer": qa_out["answer"], "kg_used": bool(kg_text),
                "kg_info": kg_info, "kg_knowledge_text": kg_text}

    def answer(self, image_path: str = None, question: str = "",
               query_image: Image.Image = None, use_kg: bool = True,
               include_images: bool = False) -> dict:
        if query_image is None:
            query_image = Image.open(image_path).convert("RGB")
        cap = self.caption_for(query_image, include_images=include_images)
        ans = self.answer_for(cap["caption"], question, use_kg=use_kg)
        return {
            "question": question,
            "answer": ans["answer"],
            "caption": cap["caption"],
            "num_retrieved": cap["num_retrieved"],
            "retrieved": cap["retrieved"],
            "kg_used": ans["kg_used"],
            "kg_info": ans["kg_info"],
            "kg_knowledge_text": ans["kg_knowledge_text"],
        }


# ----------------------------------------------------------------- #
#  CLI
# ----------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--clip_checkpoint", default="best_model")
    p.add_argument("--qa_adapter", default="./outputs_genqa/checkpoints/best_adapter")
    p.add_argument("--relation_table", default=None,
                   help="optional local UMLS MRREL-style TSV for relations")
    p.add_argument("--no_kg", action="store_true")
    args = p.parse_args()

    cfg = VQAConfig(
        clip_checkpoint=args.clip_checkpoint,
        qa_adapter_dir=args.qa_adapter,
        use_kg=not args.no_kg,
        kg_relation_table_path=args.relation_table,
    )
    pipe = VQAPipeline(cfg)
    pipe.load()
    out = pipe.answer(image_path=args.image, question=args.question)

    print("\n" + "=" * 64)
    print("QUESTION :", out["question"])
    print("ANSWER   :", out["answer"])
    print("-" * 64)
    print("CAPTION  :", out["caption"])
    print("-" * 64)
    print(f"retrieved {out['num_retrieved']} cases | KG used: {out['kg_used']}")
    if out["kg_used"]:
        print("KG info  :", json.dumps(out["kg_info"]))
        print("KG block :\n" + out["kg_knowledge_text"])
    print("=" * 64)
