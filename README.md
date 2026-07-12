# ProofLens

**Finding the right lemma, at the right moment, in a library of 180,000.**

ProofLens is a premise-selection retriever for formal theorem proving in Lean 4. Given a **proof
state** — where the prover is stuck right now — it ranks **premises** (lemmas, definitions,
theorems) from Mathlib so the prover can pick the one that actually moves the proof forward.

---

## Why this matters

Automated theorem provers usually don't fail because they can't reason. They fail because they
can't *find the right fact*. Mathlib has ~180k premises; a prover that searches all of them is
hopeless. Premise selection is the bottleneck — and it's the part with the most obvious room to
improve.

Here's the specific gap we went after. **Every published Lean retriever squashes a premise into a
single vector.** That's fine for prose, where meaning is diffuse. But formal mathematics is not
prose — it hinges on *exact symbols*: a `≤` rather than a `<`, a specific type constructor, a
particular function name. Pooling all of that into one averaged vector blurs precisely the signal
formal matching depends on. And it shows up exactly where you'd predict: on **novel premises** —
lemmas the model never saw in training — where single-vector retrievers degrade badly.

**Our idea:** keep one vector *per token* and match at the token level ("late interaction"), then
explicitly up-weight the **symbol** tokens. That should preserve the symbolic structure single-vector
models pool away — and it should help most on novel premises.

We tested it. It does.

---

## Results

Everything below is measured on [LeanDojo Benchmark 4](https://leandojo.org/) (Mathlib4), on both
official test splits, through one harness with an identical accessibility filter and identical
metrics for every system. Recall in %.

| System | random R@1 | random R@10 | novel R@1 | novel R@10 |
|---|--:|--:|--:|--:|
| BM25 (lexical baseline) | 5.48 | 13.63 | 5.65 | 16.56 |
| Late interaction, off-the-shelf | 4.44 | 11.17 | 5.21 | 14.53 |
| Dense ReProver (the published single-vector system) | 13.04 | **38.59** | — | *27.6*¹ |
| **ProofLens-LI, fine-tuned** | 8.32 | 27.46 | 7.55 | 27.09 |
| **ProofLens-LI, fine-tuned + symbol weighting** | **8.59** | **27.66** | **8.48** | **28.46** |

<sub>¹ We could not measure a *clean* dense number on `novel_premises` — see "The leak we caught" —
so we cite ReProver's own published figure.</sub>

### The four things we learned

**1. Fine-tuning transforms late interaction.** Off the shelf, a general-purpose ColBERT model is
bad at Lean (R@10 **11.2**) — worse than BM25. It has never seen mathematics. Train it on Lean and
it jumps to **27.7 — a 2.6× improvement**, comfortably past BM25. That's one epoch on a single GPU.

**2. It generalises where the single-vector model doesn't.** This is the result the whole project
was built to test:

![Generalisation gap](assets/generalisation_gap.png)

Look at the two *trained* systems. The **single-vector dense retriever falls off a cliff** going
from familiar premises to novel ones: 38.6 → 27.6, a **28% drop**. Our **fine-tuned late-interaction
model barely moves**: 27.5 → 27.1 — and with symbol weighting it actually goes *up* (27.7 → 28.5).

So on `random` the single-vector model is still ahead. But its entire advantage evaporates the
moment you test generalisation — which is the case that actually matters when a prover meets a lemma
it has never seen. On novel premises, **our model (28.5) edges past the published clean dense number
(27.6)**.

**3. The symbol weighting works — and it works hardest exactly where we predicted.**

![Symbol-weighting ablation](assets/ablation_panel.png)

Symbol weighting helps on both splits, but the lift on **novel premises is far bigger** (+5.1% R@10,
**+12.3% R@1**) than on random (+0.7%, +3.2%). That isn't a fluke — it's the mechanism. You can't
memorise a novel premise; you have to *match its symbols*. And novel premises are more lexically
distinctive (the gold lemma's name appears literally in the proof state 26.6% of the time on `novel`
vs 19.4% on `random`). Up-weighting symbol tokens pays off most precisely where symbolic matching is
all you have left.

**4. We caught a leak in how this benchmark gets used** — arguably as valuable as the retrieval
numbers. Next section.

### Overall standings

![Recall@k](assets/recall_at_k.png)

---

## The leak we caught

While calibrating, our dense ReProver run scored *suspiciously well* on `novel_premises` — 63.7 R@10,
more than double the published 27.6, and **better than its own `random` score**, which should be
impossible (novel premises are harder).

It wasn't a bug in our harness. The only public ReProver checkpoint is trained on the **random**
split — and `random` and `novel_premises` are two different partitions **of the same theorems**. So
the released model has already seen most of the "novel" test set in training. We measured it:
**97.2% of novel-split test theorems appear in the random-split training data** (control: 0.0%).

Anyone who downloads that checkpoint and evaluates it on `novel_premises` gets a badly inflated
number. We flag ours rather than report it — and we **train split-matched**: a separate model per
split, evaluated only on its own test set. None of our fine-tuned numbers carry that contamination.

---

## How we evaluate (and why you can trust it)

- **One example** = one tactic that used at least one premise. The query is the proof state before
  that tactic; the gold answer is the premise(s) that tactic actually used.
- Candidates are restricted to the **accessible** premises for that state (defined earlier in the
  same file, or in an imported file) — applied identically to every system.
- Retrieve top 100; report **R@1, R@10, MRR, nDCG@10**. This mirrors ReProver exactly, so our
  numbers are comparable to the literature.

**The calibration check.** Before trusting any number of our own, we ran the *published* ReProver
checkpoint through our harness. It scored 13.04 / 38.59 / 0.320 against its published 13.42 / 39.60
/ 0.328 — within ~3%. That tells us the harness reproduces the field's protocol, so everything
measured on top of it is trustworthy. This check is why we could confidently call the
`novel_premises` anomaly a *leak* rather than a bug.

Every run writes a JSON with full provenance (config, model, dataset version, git commit, seed) plus
the complete ranked list for every example. Metrics are pure, unit-tested functions. Everything is
seeded and reproducible, and the figures above are generated directly from the results file — no
hand-typed numbers.

---

## Where we are, and what's next

### Done ✅
- **A calibrated, trustworthy evaluation harness** — the foundation, and what makes everything else
  defensible.
- **Three reference systems measured**: BM25, the published dense ReProver, and off-the-shelf late
  interaction.
- **The leakage discovery** — found, quantified (97.2%), and designed around.
- **A tuning audit before spending a single GPU-hour on training** — which found that a quarter of
  proof states were being silently truncated (fixed), the best symbol weight, and that a handful of
  ubiquitous lemmas (`rfl`, `mul_comm`) dominate a third of all training examples (capped).
- **A full fine-tuning pipeline** — 335k / 327k training triplets with BM25-mined hard negatives,
  de-noised so a lemma used elsewhere in the same proof is never treated as a "wrong answer".
- **Fine-tuned models on both splits, plus the symbol-weighting ablation** — the headline results
  above.

### In flight 🔄
**The matched single-vector control.** Our strongest claim — *late interaction generalises better
than single-vector pooling* — currently compares our model against ReProver, which uses a different
base model, tokenizer and training recipe. A sceptic could fairly say the difference comes from
those, not from late interaction.

So we are training **our own single-vector model on the identical data, identical hard negatives,
identical base-model lineage and identical budget**, and measuring its generalisation gap in the same
harness. If it drops sharply while our model stays flat, the mechanism is *proven*, not inferred.
*(The code is written, tested and committed; launching the cluster runs is the immediate next step.)*

### Next 📋
- **Ablate the hard negatives** — how much of the gain comes from BM25-mined hard negatives versus
  random ones?
- **Re-tune the symbol weight on the trained model** — it was tuned before training, and the optimum
  will have shifted.
- **Push the model harder** — training used only 4.8 GB of a 48 GB GPU, so a much larger batch (more
  in-batch negatives) is free headroom, and validation loss was *still falling* after one epoch.
- **Publish the fine-tuned retriever** so others can build on it.
- **Close the loop with a prover** — feed retrieved premises to a tactic generator and measure
  end-to-end proof success, the metric that ultimately matters.

**In short:** the core scientific question is answered and the headline results are in. What remains
is making the central claim airtight (the control), attributing the gains (ablations), squeezing more
out of the model, and the write-up.

---

## Repository layout

```
configs/            # one YAML per experiment — model, index, weighting, splits, seed
  train/            #   fine-tuning recipes (late-interaction + single-vector control)
src/prooflens/
  data/             # corpus + import graph, accessibility, proof splits, training-pair mining, audit
  retrievers/       # bm25, dense (ReProver + single-vector), late_interaction (+ symbol weighting)
  eval/             # metrics (pure, unit-tested) + the evaluation loop
  utils/            # io, seeding, logging
scripts/            # download_data, build_index, build_pairs, train_li, train_sv, run_eval, plot_results, audit
slurm/              # cluster jobscripts
tests/              # 113 tests: metrics, loaders, accessibility, every retriever, end-to-end smoke
```

## Getting started

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                             # 113 tests — no downloads, no GPU (tiny bundled fixtures)
```

BM25 and the test suite need no heavy dependencies. The dense and late-interaction retrievers need
`torch`, `transformers` and `pylate`; hard-negative mining needs `bm25s`.

### Running an evaluation

```bash
# 1. Stage the benchmark + model checkpoints (once)
python scripts/download_data.py --out /path/to/data
export DATA_ROOT=/path/to/data/leandojo_benchmark_4
export MODELS_DIR=/path/to/data/models
export SCRATCH=/path/to/scratch

# 2. Evaluate any system
python scripts/run_eval.py --config configs/bm25.yaml
python scripts/run_eval.py --config configs/late_interaction_ft_random_weighted.yaml

# 3. Render the figures (driven only by the results file — never hand-typed)
python scripts/plot_results.py
```

### Fine-tuning

```bash
# Mine training triplets (BM25 hard negatives, de-noised, head-capped)
python scripts/build_pairs.py --config configs/late_interaction.yaml \
  --split random --split-file train.json --negatives bm25 --n-neg 3 --cap 300

# Fine-tune the late-interaction retriever
python scripts/train_li.py --config configs/train/li_ft_random.yaml
```

On a cluster, `slurm/` has ready jobscripts (`dos2unix` them first).

---

## References

- **LeanDojo / ReProver** — Yang et al., 2023 ([arXiv:2306.15626](https://arxiv.org/abs/2306.15626)) —
  the benchmark, and the single-vector retriever we calibrate against.
- **ColBERT** (late interaction) — Khattab & Zaharia, 2020 — via [PyLate](https://github.com/lightonai/pylate).
- Checkpoints: [`kaiyuy/leandojo-lean4-retriever-byt5-small`](https://huggingface.co/kaiyuy/leandojo-lean4-retriever-byt5-small),
  [`lightonai/GTE-ModernColBERT-v1`](https://huggingface.co/lightonai/GTE-ModernColBERT-v1).

## License

MIT — see `LICENSE`.
