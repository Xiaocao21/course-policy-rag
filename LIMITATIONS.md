# Limitations of this reproduction package

These are stated so the package is not read as claiming more than it supports.

1. **The live corpus cannot be rebuilt.** On 2026-11-21 a fresh crawl matched **1 of 36**
   required documents; 17 were unavailable and 18 had changed content (`data/REBUILD_REPORT.md`).
   Retrieval replay therefore uses the author-provided snapshot and official indexes, not a
   rebuild from the web.

2. **The first test split is a seen split.** It was used once as a pilot before the settings
   were frozen; the formal test numbers are re-evaluations, not first-use holdout results.

3. **test2 has a different composition from the plan.** It has 45 questions / 45 groups, but
   it is integrity-heavy (25 of 45), Glasgow is 33% (above the 25% design cap), and all
   questions are single-clause. These deviations were recorded before evaluation
   (`results/test2/DESIGN_AMENDMENTS.json`); domain- and type-level findings from test2 are
   indicative only. Absolute test2 levels are not comparable with the first test split because
   single-clause evidence makes completeness easier.

4. **test2 was not human-reviewed.** It was drafted by the AI and jointly reviewed by the
   student and the AI; the machine record at lock time says `agent_self_audit`. The attestation
   is in `annotations/TEST2_REVIEW_ATTESTATION.md`. No second human annotator exists, so no
   inter-annotator agreement is claimed.

5. **The two official runs are not a clean comparison of splits.** They used the same frozen
   checkpoint (`547060da…`) but different GPUs (RTX 3060 vs A5000) and different code versions
   (`code_sha256` `6f0bcff6…` vs `da6a039b…`). test-vs-test2 differences mix split, hardware
   and code; `results/artifact_map.csv` records this per row.

6. **Replay is only exact with the published indexes.** Rebuilt FAISS files differ in bytes
   (float32/BLAS); the pre-registration refuses them, by design. `src/preprocess.py` is for
   exploration, not for exact replay.

7. **Retraining is not an exact replay.** Hardware and library versions change floating-point
   results; R3 is a method check, and no canonical claim is made from a retrained model.

8. **Environment provenance is partial.** The test run has no `inputs` block; test2 records
   only torch/CUDA/GPU/precision. Python, transformers, sentence-transformers, FAISS, BLAS
   threads and determinism flags are unknown, so R2 reports per-question deltas and the
   unchanged verdict rather than requiring exactness (`ENVIRONMENT_EVIDENCE.md`).

9. **Some corpus text is redistributed.** The snapshot contains 36 documents; 16 carry
   CC BY-NC-SA 4.0 and 20 were recorded as academic-research use without individual
   redistribution verification. `annotations/QUOTE_PROVENANCE.csv` lists every quotation;
   `NOTICE.md` explains the takedown route.

10. **The `resources` domain is thin** (7 questions in the first test split), so domain-level
    intervals there are wide.
