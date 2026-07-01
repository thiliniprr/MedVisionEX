# evaluate_pipeline.py
"""
End-to-end evaluation of the complete pipeline.

For N (=2000) randomly sampled query images drawn from all datasets:
    1. encode the query image with the fine-tuned CLIP,
    2. retrieve top-k (=3) similar images from the pooled FAISS gallery
       (excluding the query's own image),
    3. generate a report with MedGemma (UNCHANGED) — few-shot using the
       retrieved captions, or zero-shot if nothing similar was found,
       capped at report_max_tokens (=256),
    4. score the generated report against the query image's ground-truth
       caption with BERTScore (RoBERTa-large), BLEU, and ROUGE.

Saves a per-sample predictions CSV (gold caption, retrieved captions,
generated report, per-sample scores) and a summary JSON.
"""

import csv
import json
import logging
import os
from collections import Counter
from typing import Dict, List

import numpy as np
import torch

from config_multimodal import MultiModalConfig
from multimodal_dataset_loader import MultiModalDataModule
from multimodal_faiss_builder import PooledFAISSIndex
from caption_generator import _make_threshold_generator_class

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate")


# ──────────────────────────────────────────────────────────────────── #
#  MedGemma report bridge
# ──────────────────────────────────────────────────────────────────── #



class ReportBridge:
    """
    Wrapper over the (sub-classed) VLMReportGenerator. Loads it once, sets
    max_new_tokens=report_max_tokens, dispatches few-shot vs zero-shot and
    local vs ollama. For the local backend it feeds the retrieved images
    (attached as 'image_obj') to MedGemma and uses all retrieved captions.
    """

    def __init__(self, eval_config):
        self.eval_config = eval_config
        self.available = False
        self.vlm = None
        self.backend = "local"
        try:
            from config import PipelineConfig
            from vlm_report_generator import VLMReportGenerator
            pcfg = PipelineConfig()
            # enforce the requested 256-token cap on both backends
            pcfg.vlm.max_new_tokens = eval_config.report_max_tokens
            pcfg.ollama_vlm.max_new_tokens = eval_config.report_max_tokens
            self.backend = getattr(pcfg, "vlm_backend", "local")
            if self.backend == "ollama":
                self.vlm = VLMReportGenerator(pcfg)
            else:
                GenCls = _make_threshold_generator_class()
                self.vlm = GenCls(pcfg)
                self.vlm.max_few_shot_images = eval_config.max_few_shot_images
            self.available = True
            logger.info(f"VLMReportGenerator ready (backend={self.backend}, "
                        f"max_new_tokens={eval_config.report_max_tokens}, "
                        f"max_few_shot_images={eval_config.max_few_shot_images})")
        except Exception as e:
            import traceback
            logger.error(
                "Could NOT load VLMReportGenerator — reports would all be "
                "empty and the metrics meaningless. Aborting.\n"
                f"  reason: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )
            if not getattr(eval_config, "allow_empty_reports", False):
                raise RuntimeError(
                    "Report generator unavailable (see traceback above). "
                    "Run from the main project directory so `config` and "
                    "`vlm_report_generator` import, or set "
                    "EvaluationConfig.allow_empty_reports=True to score "
                    "retrieval-only (reports will be empty)."
                ) from e

    def _to_retrieval_results(self, results):
        # Pass through caption/score/image plus the attached PIL image.
        return [{"caption": r["caption"], "score": r["score"],
                 "modality": r.get("modality_guess"),
                 "image_id": r.get("image_id"),
                 "image_obj": r.get("image_obj")} for r in results]
        return [{"caption": r["caption"], "score": r["score"],
                 "modality": r.get("modality_guess"),
                 "image_id": r.get("image_id")} for r in results]

    def generate(self, query_image, results) -> Dict:
        """few-shot if results present, else zero-shot."""
        if not self.available:
            return {"report": "", "success": False, "skipped": True,
                    "mode": "none"}
        try:
            if results:
                rr = self._to_retrieval_results(results)
                if self.backend == "ollama":
                    out = self.vlm.generate_report_ollama_few_shot(
                        query_image=query_image, retrieval_results=rr)
                else:
                    out = self.vlm.generate_report_few_shot(
                        query_image=query_image, retrieval_results=rr)
                mode = "few_shot"
            else:
                if self.backend == "ollama":
                    out = self.vlm.generate_report_ollama_zero_shot(
                        query_image=query_image)
                else:
                    out = self.vlm.generate_report_zero_shot(
                        query_image=query_image)
                mode = "zero_shot"
            # Score the RAW paragraph, not the banner-decorated report (the
            # header/footer boilerplate would pollute the text metrics).
            if isinstance(out, dict):
                report = out.get("raw_vlm_output") or out.get("report", "")
                success = out.get("success", True)
            else:
                report = str(out)
                success = True
            return {"report": report, "success": success, "mode": mode}
        except Exception as e:
            logger.warning(f"generation failed: {e}")
            return {"report": "", "success": False, "error": str(e),
                    "mode": "error"}


# ──────────────────────────────────────────────────────────────────── #
#  Query sampling
# ──────────────────────────────────────────────────────────────────── #

def sample_eval_records(config, data_module) -> List[Dict]:
    src = config.evaluation.eval_source
    if src == "test":
        recs = data_module.records("test")
        if not recs:
            logger.warning("No test records found; falling back to 'all'")
            src = "all"
    if src == "val":
        recs = data_module.records("val")
    elif src == "all":
        recs = list(data_module.records("train")) + \
               list(data_module.records("val"))
    elif src == "test":
        recs = data_module.records("test")

    n = min(config.evaluation.num_eval_samples, len(recs))
    rng = np.random.RandomState(config.random_seed)
    chosen = rng.choice(len(recs), size=n, replace=False)
    sampled = [recs[int(i)] for i in chosen]
    breakdown = Counter((r["source"], r["modality_guess"]) for r in sampled)
    logger.info(f"Eval queries: {n} from source='{config.evaluation.eval_source}'")
    logger.info(f"  source×modality breakdown: {dict(breakdown)}")
    return sampled


# ──────────────────────────────────────────────────────────────────── #
#  Main evaluation
# ──────────────────────────────────────────────────────────────────── #

@torch.no_grad()
def evaluate_pipeline(config: MultiModalConfig, finetuner,
                      faiss_index: PooledFAISSIndex, data_module):
    ev = config.evaluation
    bridge = ReportBridge(ev)

    from multimodal_faiss_builder import GalleryImageLoader
    img_loader = GalleryImageLoader(config)

    eval_records = sample_eval_records(config, data_module)
    eval_ds = data_module.dataset_from_records(eval_records)

    predictions: List[str] = []
    references: List[str] = []
    rows: List[Dict] = []
    mode_counter = Counter()
    n_retrieved_hist = []

    for qi in range(len(eval_ds)):
        rec = eval_records[qi]
        try:
            q_img = eval_ds._load_image(rec)
        except Exception as e:
            logger.warning(f"skip query {qi}: image load failed ({e})")
            continue

        proc = finetuner.processor(images=q_img, return_tensors="pt")
        q_emb = finetuner.encode_image(
            proc["pixel_values"].to(finetuner.device)
        ).cpu().numpy()

        # Threshold retrieval (all >= threshold), or legacy top_k.
        if ev.retrieval_mode == "threshold":
            results = faiss_index.search(
                q_emb, threshold=ev.retrieval_threshold,
                max_results=ev.max_retrieved, drop_self=ev.exclude_self,
                exclude_image_id=rec["image_id"] if ev.exclude_self else None,
                exclude_caption=rec["caption"] if ev.exclude_self else None,
            )
        else:
            results = faiss_index.search(
                q_emb, top_k=ev.top_k, drop_self=ev.exclude_self,
                exclude_image_id=rec["image_id"] if ev.exclude_self else None,
                exclude_caption=rec["caption"] if ev.exclude_self else None,
            )
        n_retrieved_hist.append(len(results))

        # Reload the actual neighbor images for the ones we will SHOW the VLM.
        for j, r in enumerate(results):
            r["image_obj"] = (img_loader.load(r.get("reload"))
                              if j < ev.max_few_shot_images else None)

        gen = bridge.generate(q_img, results)
        mode_counter[gen["mode"]] += 1

        gold = rec["caption"]
        pred = gen["report"] or ""
        predictions.append(pred)
        references.append(gold)

        rows.append({
            "query_index": qi,
            "source": rec["source"],
            "modality_guess": rec["modality_guess"],
            "image_id": rec["image_id"],
            "ground_truth_caption": gold,
            "generation_mode": gen["mode"],
            "num_retrieved": len(results),
            "retrieved_captions": " ||| ".join(r["caption"] for r in results),
            "retrieved_scores": ",".join(f"{r['score']:.4f}" for r in results),
            "generated_report": pred,
        })

        if (qi + 1) % 100 == 0:
            logger.info(f"  processed {qi+1}/{len(eval_ds)} "
                        f"(modes: {dict(mode_counter)})")

    logger.info(f"Generation modes: {dict(mode_counter)}")

    # ── Score predictions vs references ──
    from retrieval_metrics import GenerativeTextMetrics, format_summary
    metrics = GenerativeTextMetrics(
        model_type=ev.bertscore_model_type,
        device=str(finetuner.device),
        batch_size=config.faiss.encode_batch_size,
        rescale_with_baseline=ev.bertscore_rescale_with_baseline,
    )
    scored = metrics.compute(predictions, references)
    summary = scored["summary"]
    per_sample = scored["per_sample"]

    # attach per-sample scores to rows
    f1s = per_sample.get("bertscore_f1", [])
    rls = per_sample.get("rougeL", [])
    for i, row in enumerate(rows):
        row["bertscore_f1"] = round(f1s[i], 4) if i < len(f1s) else ""
        row["rougeL"] = round(rls[i], 4) if rls and i < len(rls) else ""

    # ── Save predictions CSV ──
    csv_path = os.path.join(config.output_dir, ev.predictions_filename)
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ── Save summary JSON ──
    import numpy as _np
    retr = _np.array(n_retrieved_hist) if n_retrieved_hist else _np.array([0])
    summary_out = {
        "num_samples": len(rows),
        "generation_modes": dict(mode_counter),
        "retrieval": {
            "mode": ev.retrieval_mode,
            "threshold": ev.retrieval_threshold,
            "mean_retrieved": float(retr.mean()),
            "median_retrieved": float(_np.median(retr)),
            "max_retrieved_seen": int(retr.max()),
            "zero_retrieved_count": int((retr == 0).sum()),
        },
        "metrics": {
            "bertscore_f1": summary["bertscore_f1"],
            "bertscore_p": summary["bertscore_p"],
            "bertscore_r": summary["bertscore_r"],
            "rouge1": summary["rouge1"],
            "rougeL": summary["rougeL"],
            "bleu": summary["bleu"],
        },
        "config": {
            "eval_source": ev.eval_source,
            "retrieval_mode": ev.retrieval_mode,
            "retrieval_threshold": ev.retrieval_threshold,
            "max_few_shot_images": ev.max_few_shot_images,
            "report_max_tokens": ev.report_max_tokens,
            "bertscore_rescaled": ev.bertscore_rescale_with_baseline,
            "exclude_self": ev.exclude_self,
        },
    }
    summary_path = os.path.join(config.output_dir, ev.summary_filename)
    with open(summary_path, "w") as f:
        json.dump(summary_out, f, indent=2)

    logger.info("=" * 64)
    logger.info("PIPELINE EVALUATION (generated report vs gold caption)")
    logger.info("  " + format_summary(summary))
    logger.info(f"  predictions → {csv_path}")
    logger.info(f"  summary     → {summary_path}")
    logger.info("=" * 64)
    return summary_out
