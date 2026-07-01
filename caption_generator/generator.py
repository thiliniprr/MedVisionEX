"""MedGemma vision-language caption generator.

Builds a VLMReportGenerator subclass that shows the retrieved neighbour images
to the model and prompts for a single-paragraph radiology description (no
section headers, no diagnoses). The base VLMReportGenerator is imported lazily,
so importing this module stays cheap.
"""

import logging

logger = logging.getLogger("caption_generator")


def _make_threshold_generator_class():
    """
    Build a VLMReportGenerator subclass that (a) shows MedGemma the actual
    retrieved images we attach (key 'image_obj') instead of loading them by
    dataset index, (b) removes the hard 3-example cap so ALL retrieved
    neighbors above the threshold are used (images bounded by
    max_few_shot_images, captions unbounded), and (c) uses a SHORT
    PARAGRAPH prompt style — one descriptive paragraph (modality + what is
    imaged + any abnormalities), no FINDINGS/IMPRESSION headers. Defined
    lazily so importing this module never requires vlm_report_generator.
    """
    from vlm_report_generator import VLMReportGenerator

    # ---- paragraph-style instructions (replaces the structured builder) ----
    PARAGRAPH_TASK = (
        "Describe the query image in a SINGLE concise paragraph of plain "
        "prose. State the imaging modality and any modality-specific "
        "specifications, what anatomy/region is imaged, and any abnormalities "
        "observed. Do NOT use section titles or headings such as 'Findings' "
        "or 'Impression'. Do NOT add diagnoses, recommendations, or "
        "follow-up. Keep it brief — a few sentences."
    )
    SYSTEM = (
        "You are an expert radiologist. Write a brief, factual, single-"
        "paragraph description of the image using standard radiology "
        "terminology. Do not hallucinate findings that are not visible."
    )

    def _condition_text(detected_conditions):
        if not detected_conditions:
            return ""
        path = {k: v for k, v in detected_conditions.items()
                if k not in ("normal", "support_devices")}
        if not path:
            return ""
        ordered = sorted(path.items(),
                         key=lambda x: x[1].get("avg_score", 0), reverse=True)
        names = ", ".join(k.replace("_", " ") for k, _ in ordered)
        return (f"\n\nDatabase analysis of visually similar images suggests "
                f"these may be present: {names}. Use as context but rely on "
                f"what you see.")

    def build_paragraph_few_shot(example_captions, example_scores,
                                 num_example_images, modality_context,
                                 include_scores, detected_conditions):
        uc = [{"type": "text", "text": SYSTEM + _condition_text(detected_conditions)}]
        if example_captions:
            uc.append({"type": "text", "text": (
                "\n\nReference descriptions from the most similar cases in the "
                "database:")})
            for i, cap in enumerate(example_captions):
                sc = (f" (similarity {example_scores[i]:.3f})"
                      if include_scores and example_scores
                      and i < len(example_scores) else "")
                if i < num_example_images:
                    uc.append({"type": "image"})
                    uc.append({"type": "text",
                               "text": f"\nSimilar case {i+1}{sc}:\n{cap}"})
                else:
                    uc.append({"type": "text",
                               "text": f"\nSimilar case {i+1}{sc} (text only):\n{cap}"})
        uc.append({"type": "text", "text": (
            f"\n\nNow describe the following query {modality_context} image.")})
        uc.append({"type": "image"})
        uc.append({"type": "text", "text": "\n" + PARAGRAPH_TASK})
        return [{"role": "user", "content": uc}]

    def build_paragraph_zero_shot(modality_context, detected_conditions):
        uc = [{"type": "text", "text": SYSTEM + _condition_text(detected_conditions)},
              {"type": "image"},
              {"type": "text", "text": "\n" + PARAGRAPH_TASK}]
        return [{"role": "user", "content": uc}]

    class ThresholdFewShotGenerator(VLMReportGenerator):
        max_few_shot_images = 6   # set per-instance by the bridge

        def _collect_example_images(self, retrieval_results, max_images=3):
            # Use the PIL images we attached; ignore the dataset-index path.
            imgs = []
            for r in retrieval_results[:max_images]:
                imgs.append(r.get("image_obj"))
            return imgs

        def generate_report_few_shot(self, query_image, retrieval_results,
                                     detected_conditions=None,
                                     num_examples=None):
            # All retrieved captions are used as text; up to
            # max_few_shot_images of them are shown as actual images.
            examples = retrieval_results
            example_captions = [r.get("caption", "") for r in examples]
            example_scores = [r.get("score", 0.0) for r in examples]

            n_imgs = min(self.max_few_shot_images, len(examples))
            example_images = [
                examples[i]["image_obj"] for i in range(n_imgs)
                if examples[i].get("image_obj") is not None
            ]
            num_example_images = len(example_images)

            logger.info(
                f"MedGemma few-shot (paragraph): {len(example_captions)} text "
                f"examples, {num_example_images} images shown (cap "
                f"{self.max_few_shot_images}), query image provided"
            )

            messages = build_paragraph_few_shot(
                example_captions=example_captions,
                example_scores=example_scores,
                num_example_images=num_example_images,
                modality_context=self.vlm_config.modality_context,
                include_scores=self.vlm_config.include_scores_in_prompt,
                detected_conditions=(
                    detected_conditions
                    if self.vlm_config.include_conditions_context else None),
            )
            all_images = example_images + [query_image]
            report_text = self._generate_local(messages, all_images)
            formatted = self._format_report(
                report_text, retrieval_results=examples, mode="few_shot",
                backend="local", num_images=num_example_images + 1)
            return {"report": formatted, "raw_vlm_output": report_text,
                    "method": "vlm_few_shot",
                    "num_examples": len(example_captions),
                    "num_images_sent": num_example_images + 1,
                    "success": True, "backend": "local"}

        def generate_report_zero_shot(self, query_image,
                                      detected_conditions=None):
            logger.info("MedGemma zero-shot (paragraph) generation...")
            messages = build_paragraph_zero_shot(
                modality_context=self.vlm_config.modality_context,
                detected_conditions=(
                    detected_conditions
                    if self.vlm_config.include_conditions_context else None))
            report_text = self._generate_local(messages, [query_image])
            formatted = self._format_report(
                report_text, mode="zero_shot", backend="local", num_images=1)
            return {"report": formatted, "raw_vlm_output": report_text,
                    "method": "vlm_zero_shot", "num_examples": 0,
                    "num_images_sent": 1, "success": True, "backend": "local"}

    return ThresholdFewShotGenerator
