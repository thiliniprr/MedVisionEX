"""Caption generation — the MedGemma vision-language report generator.

The VLM configuration still lives in `config.py` (`PipelineConfig`); this
package holds the generator itself.
"""
from caption_generator.generator import _make_threshold_generator_class

__all__ = ["_make_threshold_generator_class"]