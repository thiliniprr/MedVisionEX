# build_kg_cache.py
"""
Build a compact dictionary-KB cache ONCE, so the KG linker loads fast at
runtime with no spaCy/thinc/numpy dependency.

Two sources:

  (A) scispaCy concept JSONL (free, no UMLS license needed). Download e.g.:
      https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/kbs/2020-10-09/umls_2020_aa_cat0129.jsonl
      (each line: {"concept_id","canonical_name","aliases","types","definition"})

      python build_kg_cache.py --jsonl umls_2020_aa_cat0129.jsonl \
          --out kg_cache.pkl

  (B) UMLS RRF files (licensed): MRCONSO.RRF (+ MRSTY.RRF, MRDEF.RRF). You can
      restrict to SNOMED CT and RadLex sources to keep it small:

      python build_kg_cache.py --mrconso MRCONSO.RRF --mrsty MRSTY.RRF \
          --mrdef MRDEF.RRF --sabs SNOMEDCT_US RDL --out kg_cache.pkl

By default only radiology-relevant semantic types are kept (see KGConfig
keep_tuis) to bound size; pass --all_types to keep everything.

Then point the evaluation at it:
    python evaluate_qa_pipeline.py --kg_cache kg_cache.pkl
"""

import argparse

from knowledge_graph.kg_integration import DictionaryKB, KGConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default=None, help="scispaCy KB JSONL")
    p.add_argument("--mrconso", default=None, help="UMLS MRCONSO.RRF")
    p.add_argument("--mrsty", default=None, help="UMLS MRSTY.RRF")
    p.add_argument("--mrdef", default=None, help="UMLS MRDEF.RRF")
    p.add_argument("--sabs", nargs="*", default=[],
                   help="restrict to these source vocabularies "
                        "(e.g. SNOMEDCT_US RDL); empty = all")
    p.add_argument("--all_types", action="store_true",
                   help="keep all semantic types (default: radiology-relevant)")
    p.add_argument("--min_term_chars", type=int, default=3)
    p.add_argument("--max_synonyms", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    keep_tuis = () if args.all_types else KGConfig().keep_tuis

    if args.jsonl:
        kb = DictionaryKB.from_jsonl(
            args.jsonl, keep_tuis=keep_tuis, sabs=args.sabs,
            min_term_chars=args.min_term_chars,
            max_synonyms=args.max_synonyms)
    elif args.mrconso:
        kb = DictionaryKB.from_umls_rrf(
            args.mrconso, args.mrsty, args.mrdef,
            keep_tuis=keep_tuis, sabs=args.sabs,
            min_term_chars=args.min_term_chars,
            max_synonyms=args.max_synonyms)
    else:
        raise SystemExit("Provide --jsonl or --mrconso")

    kb.save(args.out)
    print(f"Done: {len(kb.cui_name)} concepts, {len(kb.term_index)} terms "
          f"→ {args.out}")


if __name__ == "__main__":
    main()
