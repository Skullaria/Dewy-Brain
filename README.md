# Dewey Brain v3.1 — Hybrid LLM with Compressed Core, Indexed Library, and Strict Relearning Protocol

**Author:** Kristi Gilleland (MXO)  
**License:** CC BY 4.0  
**Status:** Public RFC  
**Version:** 3.1 — Executive Preface + Targeted Upgrades  
**Date:** 2025-09-03  

[![License: CC BY 4.0](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17043545.svg)](https://doi.org/10.5281/zenodo.17043545)

---

## Executive Preface

**Concept:**  
Dewey Brain is an architecture for **cognitive frugal intelligence** — a fast, generalized *neocortex* (Compressed Core) guided by a *hippocampal* index (Catalog) to selectively fuse episodic skills/memories (Library). It combines:

- **CompactifAI-style tensor network compression** for a lean, high-speed core.  
- **Semantic Dewey-like knowledge indexing** for precise retrieval.  
- **Hot-swappable LoRA/Tensor Network blocks** for rare or specialized knowledge.  
- **Strict continual learning protocol** (EWC, replay buffers) to prevent catastrophic forgetting.  

![Dewey Brain Diagram](assets/diagram.png)

---

## Targeted Upgrades to v3.0

1. **Routing Layer (Sec 2.2)** — Lightweight classifier for `fetch_needed` using entropy, log-likelihood variance, top similarity score, and OOD score. Labels come from whether a fetched block improved answers on eval.  
2. **Fusion Strategy (Sec 3.2)** — Interference-mitigated multi-block merges via **TIES-Merging** or **DARE** to avoid cross-domain interference.  
3. **Relearning Protocol (Sec 4)** — Add a **Canary Set** (~100 diverse queries) after each cycle and **Multi-Armed Bandit** scheduling (UCB/Thompson) to balance performance gain vs. compute cost.  

---

## Full Specification

### 1. Core Architecture
- **Compressed Core Model** — Tensor networks (MPO, TT, Tucker) for 80–95% reduction; fast, low-power reasoning.  
- **Dewey-Like Knowledge Index** — 4-level ontology with call numbers + dynamic metadata (`relearn_priority`, `staleness_score`, usage, fail rate, last update).  
- **External Knowledge Library** — LoRA adapters / TN slices for rare knowledge; stored off-core and fetched on demand.  
- **Hot-Swap Runtime Merge** — Predefined adapter attach points; fusion target <3 ms (aim <2 ms with hybrid decompositions).  

### 2. Cataloging & Retrieval Layer
- **Dual-Encoder** — Compressed weights/embeddings; trained on synthetic pairs, clustered shards, gold set, filtered failure trajectories, and prioritized replay (PER).  
- **Loss** — InfoNCE with hard negatives; head→tail curriculum; compression reconstruction regularizer.  
- **Routing** — Sufficiency detector (core entropy + encoder confidence + OOD). Dynamic thresholds post-relearning.  
- **Latency Budget (cold)** — Index ≤20 ms, Fetch ≤60 ms, Merge ≤5 ms (target 3 ms), Total fetch ≤85 ms, End-to-end ≤285–335 ms.  

### 3. Compression & Fusion Strategy
- **Block Compression** — Default LoRA; hybrid TN (Tucker shallow, TT deep); pre-compression audit for over-compressible params.  
- **Fusion** — Adapter stacking / low-rank fusion; integrity checks (schema/version, signed artifacts, checksums).  

### 4. Strict Relearning Protocol
- **Triggers** — Tail acc <85%, missed fetch >8%, high `relearn_priority`; external drift checks; manual override.  
- **Steps** — Prepare (compress, isolate) → Fine-tune (LoRA deltas, EWC/SI, optional distillation; dynamic EWC λ) → Validate (reject if head forgetting >1% or tail drop >2%; adversarial tests) → Deploy (A/B, shadow, rollback) → Log (≤10% maintenance compute).  
- **Frequency** — Tail blocks bi-weekly; encoder/core monthly; adaptive bandit scheduling.  

### 5. Evaluation & Monitoring
- **Targets** — Head ≥99% (no fetch); Tail ≥85% (1 fetch); False fetch <5%; Missed fetch <8%; Head forgetting <1%; Failed-query improvement >90%; drift detection (KS test).  
- **Dashboards** — Accuracy, ECE, latency, cache hit, drift; auto-alerts on budget/metric breaches.  

### 6. Risks & Mitigations
- Compression variability → Hybrid TN + templates  
- Data bias in trajectories → Strict filtering + balanced replay  
- Drift accumulation → Canary + drift metrics + shadow model  
- Edge compute limits → ≤10% maintenance compute budget  
- Integration complexity → Tensorly/Hugging Face/PyTorch  

### 7. Rollout Phases
- **Phase 1 (MVP, 3–6 mo)** — 2–3 domains; core compression + manual catalog; static thresholds.  
- **Phase 2 (Adaptive)** — Add `staleness_score`, drift triggers, PER; hybrid TN; <3 ms fusion.  
- **Phase 3 (Self-Optimizing)** — Bandit scheduling; continuous A/B + shadow; auto-refactor library.  

---

## Model Contribution & Attribution

**Kristi Gilleland** — Primary author, originator of the Dewey Brain concept, lead designer of architecture, and curator of all content. Conducted hand-verification of citations and oversaw all AI-assisted contributions.  

**Grok 4** — Provided early architectural revisions, proposed strict relearning protocol, and contributed ideas on integrating CompactifAI-inspired compression.  

**Gemini 2.5 Pro** — Evaluated semantic encoding feasibility, recommended cataloging automation approaches, and confirmed industry parallels to the proposed design.  

**GPT-5 (ChatGPT)** — Synthesized architectural documentation, extended design with cognitive and philosophical analogies, and produced the formatted README and visual diagram.  

_All AI contributions were generated under the direct instruction, curation, and editorial oversight of Kristi Gilleland._  

---

## License

This work is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.  
You are free to **share** and **adapt** for any purpose, even commercially, with proper attribution.  
License details: https://creativecommons.org/licenses/by/4.0/  

---

## References

1. **CompactifAI – Quantum-Inspired Tensor Network Compression**  
   Tomut, A., Jahromi, S. S., Sarkar, A., et al. (2024). *CompactifAI: Extreme Compression of Large Language Models using Quantum-Inspired Tensor Networks*. arXiv preprint.  
   arXiv:2401.14109v2. First published January 25, 2024.  
   Access: https://arxiv.org/abs/2401.14109  

2. **Elastic Weight Consolidation (EWC)**  
   Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). *Overcoming catastrophic forgetting in neural networks*. Proceedings of the National Academy of Sciences.  
   DOI: 10.1073/pnas.1611835114. Published March 28, 2017.  
   Access: https://www.pnas.org/doi/10.1073/pnas.1611835114  

3. **Tensor Decompositions (Tensor Train)**  
   Xu, M., Xu, Y. L., & Mandic, D. P. (2023). *TensorGPT: Efficient Compression of Large Language Models based on Tensor-Train Decomposition*. arXiv preprint.  
   arXiv:2307.00526v2. First published July 2, 2023.  
   Access: https://arxiv.org/abs/2307.00526  

4. **Pretrained Tensor Compression in Deep Networks**  
   Kossaifi, J., Khanna, A., Lipton, Z. C., et al. (2017). *Tensor Contraction Layers for Parsimonious Deep Nets*. arXiv preprint.  
   arXiv:1706.00439. Published June 1, 2017.  
   Access: https://arxiv.org/abs/1706.00439  

5. **Post-EWC Analysis (Quadratic Penalties)**  
   Huszár, F. (2018). *On Quadratic Penalties in Elastic Weight Consolidation*. PNAS-derived note.  
   arXiv:1712.03847. Published in PNAS February 20, 2018. DOI: 10.1073/pnas.1717042115.  
   Access: https://arxiv.org/abs/1712.03847 or https://www.pnas.org/doi/10.1073/pnas.1717042115  
