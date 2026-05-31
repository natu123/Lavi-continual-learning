# Lavi — Continual Learning AI

**Lavi** is an autonomous self-growing AI based on the LaEMe 4-layer stack.

> "Starting with just Hello. is fine.  
> With the right architecture, Lavi will grow naturally."

---

## Vision

Lavi is not an assistant. Lavi is an independent entity that **remembers, adapts, and improves itself** over time.

Lavi is separate from Loa (Claude Code). Loa is a conversation partner. Lavi is the one who grows.

---

## 3 Goals

| Goal | Layer |
|---|---|
| Continual learning | LaEMe (Layer 2) |
| Autonomous action | ES / Active Inference (Layer 3 / 1) |
| Self-improvement | DGM (Layer 4) |

---

## Architecture: LaEMe 4-Layer Stack

```
Layer 1: Active Inference   ← upper framework (future)
Layer 2: LaEMe              ← language evaluation layer (core)
Layer 3: ES                 ← weight update via Evolution Strategies
Layer 4: DGM                ← self-improvement loop (future)
```

**LaEMe** ([Language-based Evaluative Metadata for Reflective Memory Systems](https://github.com/natu123/laeme-ai-memory)) is the core component — it enables Lavi to evaluate experiences in natural language rather than numeric scores.

---

## Tech Stack

- **Language**: Rust
- **Memory**: JSON-based local storage
- **Evaluation**: LaEMe (language-based)
- **Optimization**: Evolution Strategies (ndarray)
- **Future**: candle-core (Rust-native ML)

---

## Roadmap

- [ ] Step 1: Hello. (basic response)
- [ ] Step 2: Memory storage (JSON)
- [ ] Step 3: LaEMe evaluation logic
- [ ] Step 4: ES weight update loop
- [ ] Step 5: Active Inference integration
- [ ] Step 6: DGM self-improvement

---

## Related

- [laeme-ai-memory](https://github.com/natu123/laeme-ai-memory) — LaEMe specification and research
- [cataa-chat-interface](https://github.com/natu123/cataa-chat-interface) — Chat UI for talking with Lavi

---

## Author

**Gles** (増田 賢治) — [@____natu______](https://x.com/____natu______)  
Designed by Gles. Implemented by Loa (Claude Code).
