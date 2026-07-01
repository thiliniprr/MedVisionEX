# kg_integration.py
"""
Knowledge-graph augmentation for the VQA pipeline.

Pipeline role
-------------
    image + question
        → FAISS retrieve similar cases
        → VLM generates a caption/report   ← KG augments this (findings context)
        → (caption, question) → QA LLM      ← KG augments this (concept grounding)
        → answer

This module turns free text (the question and the generated caption) into a
compact, structured "knowledge context" by:
  1. linking medical mentions to UMLS concepts with scispaCy
     (UMLS subsumes SNOMED CT [SNOMEDCT_US] and RadLex [RDL], so one linker
      covers all three vocabularies);
  2. attaching each concept's canonical name, semantic type, short definition,
     and a couple of synonyms;
  3. optionally expanding with ontology relations from a LOCAL relation table
     (UMLS MRREL / RadLex / SNOMED subset) via a pluggable RelationProvider;
  4. rendering two views:
       - a `detected_conditions` dict for the VLM prompt (caption side),
       - a compact text block for the QA prompt (QA side).

Dependencies (install once):
    pip install scispacy spacy
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/\
v0.5.4/en_core_sci_sm-0.5.4.tar.gz
  The UMLS linker downloads its KB (~1 GB) on first use.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("kg")


# ================================================================== #
#  Config
# ================================================================== #

@dataclass
class KGConfig:
    # Entity-linker backend:
    #   "dictionary" : pure-Python greedy longest-match against a local concept
    #                  dictionary. NO spaCy/thinc/numpy — immune to the NumPy
    #                  ABI issue. Source the dictionary from a scispaCy KB
    #                  JSONL (free) or your UMLS RRF files (licensed).
    #   "scispacy"   : the scispaCy UMLS linker (needs a working spaCy stack).
    backend: str = "dictionary"

    # ── dictionary backend sources (provide ONE) ──
    # (a) a prebuilt cache from build_kg_cache.py (fastest to load):
    kb_cache_path: Optional[str] = None
    # (b) a scispaCy concept JSONL (cui/canonical_name/aliases/types/definition):
    kb_jsonl_path: Optional[str] = None
    # (c) raw UMLS RRF subset (licensed):
    umls_mrconso_path: Optional[str] = None
    umls_mrsty_path: Optional[str] = None
    umls_mrdef_path: Optional[str] = None
    kb_sabs: Tuple[str, ...] = ()           # restrict RRF/JSONL sources; ()=all
    max_ngram: int = 6                      # longest term span to match
    min_term_chars: int = 3                 # ignore very short dictionary terms

    # ── scispaCy backend ──
    spacy_model: str = "en_core_sci_sm"     # or en_core_sci_md for better NER
    linker_name: str = "umls"               # umls | mesh | rxnorm | go | hpo
    resolve_abbreviations: bool = True
    link_threshold: float = 0.85            # min linker confidence
    max_entities_per_mention: int = 1

    max_concepts: int = 12                  # cap concepts used in context
    definition_max_chars: int = 160
    max_synonyms: int = 2

    # Keep only these UMLS semantic-type groups (radiology-relevant). Empty =
    # keep all. TUIs: T047 disease, T046 patho-function, T184 sign/symptom,
    # T033 finding, T023 body part, T082 spatial, T037 injury, T190 anomaly,
    # T191 neoplastic process, T060 diagnostic procedure.
    keep_tuis: Tuple[str, ...] = (
        "T047", "T046", "T184", "T033", "T023", "T082",
        "T037", "T190", "T191", "T060", "T029", "T030",
    )

    # Optional local relation table (UMLS MRREL-style TSV: CUI1<TAB>REL<TAB>CUI2
    # [<TAB>RELA][<TAB>SAB]). None → no explicit relations (concept metadata
    # and Q∩caption overlap are still used). Pure Python — no numpy.
    relation_table_path: Optional[str] = None
    relation_sabs: Tuple[str, ...] = ("SNOMEDCT_US", "RDL", "RADLEX")
    max_relations: int = 8


# Human-readable labels for common semantic-type TUIs (fallback: the code).
_TUI_LABEL = {
    "T047": "Disease", "T046": "Pathologic Function", "T184": "Sign/Symptom",
    "T033": "Finding", "T023": "Body Part", "T082": "Spatial Concept",
    "T037": "Injury", "T190": "Anatomical Abnormality",
    "T191": "Neoplastic Process", "T060": "Diagnostic Procedure",
    "T029": "Body Location", "T030": "Body Space",
    "T061": "Therapeutic Procedure", "T121": "Pharmacologic Substance",
}


# ================================================================== #
#  Linked-concept record
# ================================================================== #

@dataclass
class Concept:
    cui: str
    name: str
    score: float
    tuis: Tuple[str, ...] = ()
    definition: str = ""
    synonyms: Tuple[str, ...] = ()
    mention: str = ""

    @property
    def type_label(self) -> str:
        for t in self.tuis:
            if t in _TUI_LABEL:
                return _TUI_LABEL[t]
        return self.tuis[0] if self.tuis else "Concept"


# ================================================================== #
#  Relation provider (pluggable)
# ================================================================== #

class RelationProvider:
    """Base: no explicit relations."""
    def relations_between(self, cuis_a, cuis_b) -> List[Tuple[str, str, str]]:
        return []


class UMLSRelationProvider(RelationProvider):
    """
    Loads a local MRREL-style table once into an in-memory adjacency map and
    returns 1-hop relations connecting two CUI sets. Works for UMLS, and
    (filtered by SAB) for SNOMED CT and RadLex, since all three live in the
    UMLS Metathesaurus.
    """
    def __init__(self, path: str, sabs=(), max_relations: int = 8):
        self.max_relations = max_relations
        self._adj: Dict[str, List[Tuple[str, str]]] = {}
        keep_sab = set(s.upper() for s in sabs) if sabs else None
        n = 0
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    parts = line.rstrip("\n").split("|")  # raw MRREL is pipe
                if len(parts) < 3:
                    continue
                c1, rel, c2 = parts[0], parts[1], parts[2]
                rela = parts[3] if len(parts) > 3 else ""
                sab = parts[4] if len(parts) > 4 else ""
                if keep_sab and sab and sab.upper() not in keep_sab:
                    continue
                label = rela or rel
                self._adj.setdefault(c1, []).append((label, c2))
                n += 1
        logger.info(f"UMLSRelationProvider: {n} relations, "
                    f"{len(self._adj)} source concepts from {path}")

    def relations_between(self, cuis_a, cuis_b):
        target = set(cuis_b)
        out = []
        for c1 in cuis_a:
            for label, c2 in self._adj.get(c1, []):
                if c2 in target:
                    out.append((c1, label, c2))
                    if len(out) >= self.max_relations:
                        return out
        return out


# ================================================================== #
#  Dictionary backend (pure Python — no spaCy / thinc / numpy)
# ================================================================== #

import json
import pickle
import re

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


class DictionaryKB:
    """
    In-memory concept dictionary built from a scispaCy KB JSONL or UMLS RRF
    files. Holds:
        term_index : normalized term string → set(cui)
        cui_name   : cui → canonical name
        cui_types  : cui → tuple(TUI)
        cui_def    : cui → definition
        cui_syn    : cui → tuple(synonyms)
    Uses only the standard library, so it never touches the NumPy ABI.
    """

    def __init__(self):
        self.term_index: Dict[str, set] = {}
        self.cui_name: Dict[str, str] = {}
        self.cui_types: Dict[str, Tuple[str, ...]] = {}
        self.cui_def: Dict[str, str] = {}
        self.cui_syn: Dict[str, Tuple[str, ...]] = {}

    # ---- builders ----

    def _add_term(self, term: str, cui: str, min_chars: int):
        norm = _normalize(term)
        if len(norm) < min_chars:
            return
        self.term_index.setdefault(norm, set()).add(cui)

    @classmethod
    def from_jsonl(cls, path: str, keep_tuis=(), sabs=(),
                   min_term_chars: int = 3, max_synonyms: int = 8):
        """scispaCy KB JSONL: one JSON object per line with concept_id/cui,
        canonical_name, aliases, types, definition."""
        kb = cls()
        keep = set(keep_tuis)
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cui = obj.get("concept_id") or obj.get("cui")
                if not cui:
                    continue
                types = tuple(obj.get("types", []) or [])
                if keep and not (set(types) & keep):
                    continue
                name = obj.get("canonical_name") or obj.get("name") or cui
                aliases = obj.get("aliases", []) or []
                definition = (obj.get("definition") or "")
                kb.cui_name[cui] = name
                kb.cui_types[cui] = types
                if definition:
                    kb.cui_def[cui] = definition
                kb.cui_syn[cui] = tuple(aliases[:max_synonyms])
                kb._add_term(name, cui, min_term_chars)
                for a in aliases:
                    kb._add_term(a, cui, min_term_chars)
                n += 1
        logger.info(f"DictionaryKB(JSONL): {n} concepts, "
                    f"{len(kb.term_index)} terms from {path}")
        return kb

    @classmethod
    def from_umls_rrf(cls, mrconso: str, mrsty: str = None, mrdef: str = None,
                      keep_tuis=(), sabs=(), min_term_chars: int = 3,
                      max_synonyms: int = 8):
        """Build from UMLS RRF files (pipe-delimited)."""
        kb = cls()
        keep_sab = set(s.upper() for s in sabs) if sabs else None

        # MRSTY: CUI|TUI|...
        cui_types: Dict[str, set] = {}
        if mrsty:
            with open(mrsty, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    p = line.split("|")
                    if len(p) > 1:
                        cui_types.setdefault(p[0], set()).add(p[1])
        keep = set(keep_tuis)

        # MRDEF: CUI|...|SAB|...|DEF|
        if mrdef:
            with open(mrdef, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    p = line.split("|")
                    if len(p) > 5 and p[0] not in kb.cui_def:
                        kb.cui_def[p[0]] = p[5]

        # MRCONSO: CUI|LAT|TS|LUI|STT|SUI|ISPREF|...|...|...|SAB|...|STR|...
        syn_count: Dict[str, int] = {}
        with open(mrconso, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = line.split("|")
                if len(p) < 15:
                    continue
                cui, lat, ispref, sab, string = (
                    p[0], p[1], p[6], p[11], p[14])
                if lat != "ENG":
                    continue
                if keep_sab and sab.upper() not in keep_sab:
                    continue
                types = tuple(cui_types.get(cui, ()))
                if keep and not (set(types) & keep):
                    continue
                kb.cui_types[cui] = types
                if cui not in kb.cui_name or ispref == "Y":
                    kb.cui_name[cui] = string
                kb._add_term(string, cui, min_term_chars)
                if syn_count.get(cui, 0) < max_synonyms:
                    kb.cui_syn.setdefault(cui, tuple())
                    kb.cui_syn[cui] = kb.cui_syn[cui] + (string,)
                    syn_count[cui] = syn_count.get(cui, 0) + 1
        logger.info(f"DictionaryKB(RRF): {len(kb.cui_name)} concepts, "
                    f"{len(kb.term_index)} terms")
        return kb

    # ---- cache ----

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"term_index": {k: list(v) for k, v
                                        in self.term_index.items()},
                         "cui_name": self.cui_name,
                         "cui_types": self.cui_types,
                         "cui_def": self.cui_def,
                         "cui_syn": self.cui_syn}, f)
        logger.info(f"DictionaryKB cache saved → {path}")

    @classmethod
    def load(cls, path: str):
        kb = cls()
        with open(path, "rb") as f:
            d = pickle.load(f)
        kb.term_index = {k: set(v) for k, v in d["term_index"].items()}
        kb.cui_name = d["cui_name"]
        kb.cui_types = d["cui_types"]
        kb.cui_def = d["cui_def"]
        kb.cui_syn = d["cui_syn"]
        logger.info(f"DictionaryKB cache loaded ({len(kb.cui_name)} concepts, "
                    f"{len(kb.term_index)} terms)")
        return kb


class DictionaryEntityLinker:
    """
    Greedy longest-match linker over a DictionaryKB. Pure Python; same
    .link(text) -> List[Concept] interface as the scispaCy linker, so the rest
    of the module is unchanged.
    """
    def __init__(self, config: KGConfig):
        self.config = config
        self._kb: Optional[DictionaryKB] = None
        self._cache: Dict[str, List[Concept]] = {}

    def _ensure_loaded(self):
        if self._kb is not None:
            return
        c = self.config
        if c.kb_cache_path:
            self._kb = DictionaryKB.load(c.kb_cache_path)
        elif c.kb_jsonl_path:
            self._kb = DictionaryKB.from_jsonl(
                c.kb_jsonl_path, keep_tuis=c.keep_tuis, sabs=c.kb_sabs,
                min_term_chars=c.min_term_chars, max_synonyms=c.max_synonyms)
        elif c.umls_mrconso_path:
            self._kb = DictionaryKB.from_umls_rrf(
                c.umls_mrconso_path, c.umls_mrsty_path, c.umls_mrdef_path,
                keep_tuis=c.keep_tuis, sabs=c.kb_sabs,
                min_term_chars=c.min_term_chars, max_synonyms=c.max_synonyms)
        else:
            raise FileNotFoundError(
                "Dictionary backend needs one of: kb_cache_path, "
                "kb_jsonl_path, or umls_mrconso_path in KGConfig.")

    def link(self, text: str) -> List[Concept]:
        if not text or not text.strip():
            return []
        key = text.strip()
        if key in self._cache:
            return self._cache[key]
        self._ensure_loaded()
        result = self._link_uncached(text)
        if len(self._cache) < 50000:
            self._cache[key] = result
        return result

    def _link_uncached(self, text: str) -> List[Concept]:
        tokens = [(m.group(0), m.start(), m.end())
                  for m in _WORD_RE.finditer(text.lower())]
        n = len(tokens)
        seen: Dict[str, Concept] = {}
        i = 0
        while i < n:
            matched = False
            hi = min(self.config.max_ngram, n - i)
            for L in range(hi, 0, -1):                 # longest match first
                phrase = " ".join(tokens[k][0] for k in range(i, i + L))
                cuis = self._kb.term_index.get(phrase)
                if not cuis:
                    continue
                mention = text[tokens[i][1]:tokens[i + L - 1][2]]
                for cui in cuis:
                    types = self._kb.cui_types.get(cui, ())
                    if self.config.keep_tuis and not (
                        set(types) & set(self.config.keep_tuis)
                    ):
                        continue
                    if cui in seen:
                        continue
                    name = self._kb.cui_name.get(cui, cui)
                    definition = self._kb.cui_def.get(cui, "")[
                        :self.config.definition_max_chars]
                    syn = tuple(
                        s for s in self._kb.cui_syn.get(cui, ())
                        if s.lower() != name.lower()
                    )[:self.config.max_synonyms]
                    # score: longer matched spans are more reliable
                    seen[cui] = Concept(
                        cui=cui, name=name, score=min(1.0, 0.6 + 0.1 * L),
                        tuis=tuple(types), definition=definition,
                        synonyms=syn, mention=mention)
                if seen:
                    i += L
                    matched = True
                    break
            if not matched:
                i += 1
        concepts = sorted(seen.values(), key=lambda c: c.score, reverse=True)
        return concepts[:self.config.max_concepts]


# ================================================================== #
#  scispaCy backend (optional — needs a working spaCy stack)
# ================================================================== #

class MedicalEntityLinker:
    def __init__(self, config: KGConfig):
        self.config = config
        self._nlp = None
        self._linker = None
        self._cache = {}            # text → List[Concept] (repeated captions)

    def _ensure_loaded(self):
        if self._nlp is not None:
            return
        import spacy
        from scispacy.abbreviation import AbbreviationDetector  # noqa: F401
        from scispacy.linking import EntityLinker  # noqa: F401

        logger.info(f"Loading scispaCy model: {self.config.spacy_model}")
        nlp = spacy.load(self.config.spacy_model)
        if self.config.resolve_abbreviations:
            nlp.add_pipe("abbreviation_detector")
        nlp.add_pipe("scispacy_linker", config={
            "resolve_abbreviations": self.config.resolve_abbreviations,
            "linker_name": self.config.linker_name,
            "max_entities_per_mention": self.config.max_entities_per_mention,
            "threshold": self.config.link_threshold,
        })
        self._nlp = nlp
        self._linker = nlp.get_pipe("scispacy_linker")
        logger.info("scispaCy UMLS linker ready.")

    def link(self, text: str) -> List[Concept]:
        if not text or not text.strip():
            return []
        key = text.strip()
        if key in self._cache:
            return self._cache[key]
        self._ensure_loaded()
        result = self._link_uncached(text)
        if len(self._cache) < 50000:    # bound cache size
            self._cache[key] = result
        return result

    def _link_uncached(self, text: str) -> List[Concept]:
        doc = self._nlp(text)
        seen: Dict[str, Concept] = {}
        for ent in doc.ents:
            for cui, score in ent._.kb_ents[:self.config.max_entities_per_mention]:
                if score < self.config.link_threshold:
                    continue
                kb = self._linker.kb.cui_to_entity[cui]
                tuis = tuple(kb.types)
                if self.config.keep_tuis and not (
                    set(tuis) & set(self.config.keep_tuis)
                ):
                    continue
                if cui in seen and seen[cui].score >= score:
                    continue
                definition = (kb.definition or "")[:self.config.definition_max_chars]
                synonyms = tuple(
                    a for a in (kb.aliases or [])
                    if a.lower() != kb.canonical_name.lower()
                )[:self.config.max_synonyms]
                seen[cui] = Concept(
                    cui=cui, name=kb.canonical_name, score=float(score),
                    tuis=tuis, definition=definition, synonyms=synonyms,
                    mention=ent.text,
                )
        concepts = sorted(seen.values(), key=lambda c: c.score, reverse=True)
        return concepts[:self.config.max_concepts]


# ================================================================== #
#  Knowledge context builder
# ================================================================== #

@dataclass
class KnowledgeContext:
    question_concepts: List[Concept] = field(default_factory=list)
    caption_concepts: List[Concept] = field(default_factory=list)
    shared: List[Concept] = field(default_factory=list)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.question_concepts or self.caption_concepts)


class KnowledgeContextBuilder:
    def __init__(self, config: KGConfig,
                 linker=None,
                 relation_provider: Optional[RelationProvider] = None):
        self.config = config
        if linker is not None:
            self.linker = linker
        elif config.backend == "scispacy":
            self.linker = MedicalEntityLinker(config)
        else:
            self.linker = DictionaryEntityLinker(config)
        if relation_provider is not None:
            self.relations = relation_provider
        elif config.relation_table_path:
            self.relations = UMLSRelationProvider(
                config.relation_table_path, config.relation_sabs,
                config.max_relations,
            )
        else:
            self.relations = RelationProvider()  # no-op

    def build(self, question: str, caption: str) -> KnowledgeContext:
        q = self.linker.link(question)
        c = self.linker.link(caption)
        q_cuis = {x.cui for x in q}
        c_cuis = {x.cui for x in c}
        shared_cuis = q_cuis & c_cuis
        shared = [x for x in q if x.cui in shared_cuis]
        rels = self.relations.relations_between(list(q_cuis), list(c_cuis))
        return KnowledgeContext(question_concepts=q, caption_concepts=c,
                                shared=shared, relations=rels)

    # ----- Render for the VLM caption prompt (detected_conditions dict) ----- #

    def to_detected_conditions(self, ctx: KnowledgeContext) -> Dict[str, Dict]:
        """
        Pack caption-side findings/diseases into the dict shape the existing
        VLMReportGenerator prompt builder expects:
            { "<condition name>": {"avg_score": float, "frequency": float} }
        Only finding/disease/abnormality types are surfaced as 'conditions'.
        """
        cond_tuis = {"T047", "T046", "T184", "T033", "T037",
                     "T190", "T191"}
        out: Dict[str, Dict] = {}
        for con in ctx.caption_concepts:
            if set(con.tuis) & cond_tuis:
                key = con.name.lower().replace(" ", "_")
                out[key] = {"avg_score": con.score, "frequency": con.score}
        return out

    # ----- Render for the QA prompt (compact text block) ----- #

    def to_qa_text(self, ctx: KnowledgeContext) -> str:
        if ctx.is_empty():
            return ""
        lines = ["Relevant medical knowledge:"]

        # Prioritise concepts in the question, then shared, then caption.
        rendered = set()
        ordered = (ctx.question_concepts + ctx.shared + ctx.caption_concepts)
        for con in ordered:
            if con.cui in rendered:
                continue
            rendered.add(con.cui)
            piece = f"- {con.name} ({con.type_label})"
            if con.definition:
                piece += f": {con.definition.strip()}"
            if con.synonyms:
                piece += f" [also: {', '.join(con.synonyms)}]"
            lines.append(piece)
            if len(rendered) >= self.config.max_concepts:
                break

        if ctx.shared:
            names = ", ".join(s.name for s in ctx.shared)
            lines.append(f"Concepts in both the question and the findings: {names}.")

        if ctx.relations:
            # Map CUIs back to names for readability
            name_of = {c.cui: c.name for c in
                       (ctx.question_concepts + ctx.caption_concepts)}
            rel_strs = []
            for c1, rel, c2 in ctx.relations:
                r = rel.replace("_", " ")
                rel_strs.append(f"{name_of.get(c1, c1)} {r} {name_of.get(c2, c2)}")
            lines.append("Ontology relations: " + "; ".join(rel_strs) + ".")

        return "\n".join(lines)


# ================================================================== #
#  Convenience: one object that does both renders
# ================================================================== #

class KGAugmentor:
    """
    High-level helper used by the VQA pipeline. Loads the linker lazily on
    first call so importing this module is cheap. FAIL-SOFT: if the linker
    cannot load or a call errors, KG is disabled (empty augmentation) and the
    pipeline continues rather than crashing a long run.
    """
    def __init__(self, config: Optional[KGConfig] = None):
        self.config = config or KGConfig()
        self.builder = KnowledgeContextBuilder(self.config)
        self.disabled = False

    _EMPTY = {"detected_conditions": {}, "qa_knowledge_text": "",
              "num_question_concepts": 0, "num_caption_concepts": 0,
              "shared_concepts": [], "num_relations": 0,
              "graph": {"nodes": [], "edges": []}}

    def augment(self, question: str, caption: str) -> Dict:
        if self.disabled:
            return dict(self._EMPTY)
        try:
            ctx = self.builder.build(question, caption)
        except Exception as e:
            logger.error(f"KG augmentation failed ({type(e).__name__}: {e}). "
                         f"Disabling KG for the rest of this run.")
            self.disabled = True
            return dict(self._EMPTY)
        ctx_graph = self._build_graph(ctx)
        return {
            "detected_conditions": self.builder.to_detected_conditions(ctx),
            "qa_knowledge_text": self.builder.to_qa_text(ctx),
            "num_question_concepts": len(ctx.question_concepts),
            "num_caption_concepts": len(ctx.caption_concepts),
            "shared_concepts": [s.name for s in ctx.shared],
            "num_relations": len(ctx.relations),
            "graph": ctx_graph,
        }

    @staticmethod
    def _build_graph(ctx) -> Dict:
        """
        Build a node/edge graph for visualization.

        Roles: "question" (concept only from the question), "report" (only
        from the caption/report), "shared" (in both). Edges connect the
        question to its shared concepts and the report to its shared concepts,
        plus any explicit ontology relations between concepts.
        """
        shared_cuis = {c.cui for c in ctx.shared}

        def node(c, role):
            return {"id": c.cui or c.name, "label": c.name, "role": role,
                    "type": c.type_label, "definition": c.definition or "",
                    "synonyms": list(c.synonyms)[:4]}

        nodes, seen = [], {}
        # shared first so they win the role assignment
        for c in ctx.shared:
            nid = c.cui or c.name
            if nid not in seen:
                seen[nid] = "shared"
                nodes.append(node(c, "shared"))
        for c in ctx.question_concepts:
            nid = c.cui or c.name
            if nid not in seen:
                seen[nid] = "question"
                nodes.append(node(c, "question"))
        for c in ctx.caption_concepts:
            nid = c.cui or c.name
            if nid not in seen:
                seen[nid] = "report"
                nodes.append(node(c, "report"))

        edges = []
        # anchor hubs: Question and Report, linked to their concepts
        q_ids = [(c.cui or c.name) for c in ctx.question_concepts] + \
                [(c.cui or c.name) for c in ctx.shared]
        r_ids = [(c.cui or c.name) for c in ctx.caption_concepts] + \
                [(c.cui or c.name) for c in ctx.shared]
        for nid in dict.fromkeys(q_ids):
            edges.append({"source": "__QUESTION__", "target": nid,
                          "label": "", "kind": "anchor"})
        for nid in dict.fromkeys(r_ids):
            edges.append({"source": "__REPORT__", "target": nid,
                          "label": "", "kind": "anchor"})
        # explicit ontology relations (cui1, rel, cui2)
        id_set = set(seen)
        for c1, rel, c2 in ctx.relations:
            if c1 in id_set and c2 in id_set:
                edges.append({"source": c1, "target": c2,
                              "label": str(rel).replace("_", " "),
                              "kind": "relation"})

        # the two anchor hub nodes
        hubs = [{"id": "__QUESTION__", "label": "Question", "role": "anchor_q",
                 "type": "", "definition": "", "synonyms": []},
                {"id": "__REPORT__", "label": "Report", "role": "anchor_r",
                 "type": "", "definition": "", "synonyms": []}]
        return {"nodes": hubs + nodes, "edges": edges}
