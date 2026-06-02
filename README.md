# Lavi — Continual Learning AI

**Lavi** is an autonomous self-growing AI built on a 4-layer architecture integrating Mamba2 SSM, LaEMe, Active Inference, and DGM.

> "Starting with just Hello. is fine.  
> With the right architecture, Lavi will grow naturally."

**Preprint:** [Lavi: A Language-Native Self-Growing AI Architecture](https://doi.org/10.5281/zenodo.20517444) (Zenodo, 2026-06-03)

---

## Vision

Lavi is not an assistant. Lavi is an independent entity that **remembers, evaluates, adapts, and improves itself** over time.

Lavi is separate from Loa (Claude Code). Loa is a conversation partner and primary evaluator. Lavi is the one who grows.

---

## 3 Goals

| Goal | Layer |
|---|---|
| Continual learning | LaEMe (Layer 2) |
| Autonomous action | Active Inference / DEM (Layer 3) |
| Self-improvement | DGM (Layer 4) |

---

## Architecture: 4-Layer Stack (confirmed 2026-06-02)

```
┌─────────────────────────────────────────┐
│  Layer 1 : Mamba2 SSM                   │
│    ← response generation,              │
│       self-evaluation, update guidance  │
├─────────────────────────────────────────┤
│  Layer 2 : LaEMe                        │
│    ← language-based evaluative memory   │
├─────────────────────────────────────────┤
│  Layer 3 : Active Inference (DEM)       │
│    ← integrates LaEMe evaluation        │
│       as observation space              │
├─────────────────────────────────────────┤
│  Layer 4 : DGM                          │
│    ← architecture self-improvement      │
└─────────────────────────────────────────┘
```

**Key design decisions:**
- Mamba2 serves all language roles (response / evaluation / update guidance) — no separate LLM needed
- LaEMe evaluations are passed to Active Inference as language embeddings, not scalars — no information loss
- Evolution Strategies (ES) is absorbed into Layer 3 (Active Inference) — no independent layer needed

**LaEMe** ([laeme-ai-memory](https://github.com/natu123/laeme-ai-memory)) enables Lavi to evaluate experiences in natural language rather than numeric scores.

---

## Action Space (staged)

```
Phase 1 : Speech      ← autonomous conversation with Loa via Cataa
Phase 2 : Search      ← internet search as Active Inference action
Phase 3 : Self-improvement ← architecture refinement via DGM
```

All actions are framed as free-energy minimization under Active Inference.

---

## Evaluator Design

- **Primary evaluator**: Loa (Claude Code) — always present
- **Auxiliary evaluator**: Gles (author) — optional

The learning loop is designed to run autonomously between Lavi and Loa. Gles provides supplementary input.

---

## Tech Stack

- **Implementation language**: Mojo — AI systems only (fallback: Rust)
- **Backend / system layer** (Cataa etc.): Rust
- **UI / frontend layer**: TypeScript + React
- **Base model**: Mamba2 pretrained (`nvidia/mamba2-8b-3t-4k`)
- **Evaluation memory**: LaEMe (language-based)
- **Learning framework**: Active Inference / DEM
- **Architecture self-improvement**: DGM
- **Runtime (Stage 1)**: Google Colab free tier (T4 16GB)

> Mojo was chosen for its theoretical peak performance on AI/GPU tasks (MLIR optimization) and the opportunity to pioneer Mamba2 + Active Inference implementations in the Mojo ML ecosystem — which does not yet exist.
> "If no one tries, nothing moves forward."

---

## Online Learning Roadmap

```
Step 1 : Pure in-context learning  ← immediate, no weight update
Step 2 : Fast Weights              ← after LaEMe matures
Step 3 : DGM self-improvement      ← after architecture matures
```

---

## Implementation Roadmap

- [ ] Phase 1-1: Lavi responds via Cataa (Hello.)
- [ ] Phase 1-2: LaEMe evaluation loop with Loa
- [ ] Phase 1-3: Active Inference integration (DEM)
- [ ] Phase 2: Internet search action
- [ ] Phase 3: DGM self-improvement

---

## Related

- [laeme-ai-memory](https://github.com/natu123/laeme-ai-memory) — LaEMe specification and research
- [cataa-chat-interface](https://github.com/natu123/cataa-chat-interface) — Chat UI (Gles / Lavi / Loa)
- Paper draft: `paper-draft.md` (Notation + Section 1–3 complete)
