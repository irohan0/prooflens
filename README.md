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

We tested it. The generalisation result is real. The absolute performance is not there yet. Both of
those are below, in full.

---

## Results

Everything is measured on [LeanDojo Benchmark 4](https://leandojo.org/) (Mathlib4), on both official
test splits, through **one harness** with an identical accessibility filter and identical metrics for
every system. Recall in %. `random` n=2,811; `novel_premises` n=4,357.

| System | Trained on Lean? | random R@1 | random R@10 | novel R@1 | novel R@10 |
|---|:--:|--:|--:|--:|--:|
| BM25 (lexical baseline) | no | 5.48 | 13.63 | 5.65 | 16.56 |
| Late interaction, off-the-shelf | no | 4.15 | 10.44 | 4.82 | 14.05 |
| Late interaction, off-the-shelf + symbol weighting | no | 4.44 | 11.17 | 5.21 | 14.53 |
| Dense ReProver (published single-vector system) | yes | **13.04** | **38.59** | — | *27.6*¹ |
| **ProofLens-LI, fine-tuned** | yes | 8.32 | 27.46 | 7.55 | 27.09 |
| **ProofLens-LI, fine-tuned + symbol weighting** | yes | 8.59 | 27.66 | **8.48** | **28.46** |

<sub>¹ We could not measure a *clean* dense number on `novel_premises` — see "The leak we caught" —
so we cite ReProver's own published figure. This is the one cell in the table not produced by our
harness, and we flag it as such.</sub>

### Before and after fine-tuning — the single biggest lever

Off the shelf, a general-purpose ColBERT model is genuinely bad at Lean. It has never seen
mathematics, and it shows: it loses to a 1970s lexical baseline. Train it on Lean and it transforms.

| Split | Metric | Off-the-shelf | Fine-tuned | Change |
|---|---|--:|--:|--:|
| random | R@1 | 4.44 | 8.59 | **+94%** |
| random | R@10 | 11.17 | 27.66 | **+148%** |
| random | MRR | 0.107 | 0.236 | **+121%** |
| novel_premises | R@1 | 5.21 | 8.48 | **+63%** |
| novel_premises | R@10 | 14.53 | 28.46 | **+96%** |
| novel_premises | MRR | 0.136 | 0.240 | **+76%** |

*(both rows use symbol weighting ON, so the comparison isolates training)*

**Roughly a 2.5× improvement across the board**, from one epoch on a single GPU. It moves late
interaction from *below* BM25 to roughly **2× BM25** — but, importantly, still **below dense
ReProver on `random`**. More on that in the limitations.

### How we compare to the other methods

| Comparison | random R@10 | novel R@10 | Read |
|---|--:|--:|---|
| Ours vs **BM25** | 27.66 vs 13.63 | 28.46 vs 16.56 | We win decisively — **2.0× / 1.7×** |
| Ours vs **off-the-shelf LI** | 27.66 vs 11.17 | 28.46 vs 14.53 | Fine-tuning is what made it work — **2.5×** |
| Ours vs **dense ReProver** | 27.66 vs **38.59** | **28.46** vs 27.6 | **We lose on `random` by 28%.** We edge ahead on `novel`. |

That last row is the whole story in one line, and it cuts both ways. **On the familiar split we are
clearly worse than the state of the art. On the split that tests generalisation, we are not.**

### The generalisation gap — the result the project was built to test

![Generalisation gap](assets/generalisation_gap.png)

Look only at the two *trained* systems. Going from familiar premises to novel ones:

| System | random R@10 | novel R@10 | Gap |
|---|--:|--:|--:|
| Dense ReProver (single-vector) | 38.59 | 27.6 | **−28.5%** ⬇ |
| ProofLens-LI, fine-tuned | 27.46 | 27.09 | **−1.3%** |
| ProofLens-LI, fine-tuned + symbol weighting | 27.66 | 28.46 | **+2.9%** ⬆ (novel is *better*) |

The single-vector retriever **falls off a cliff**. Ours barely moves — and with symbol weighting it
actually goes *up*. The single-vector model's entire advantage evaporates the moment you test
generalisation, which is the case that matters when a prover meets a lemma it has never seen.

**This is the strongest thing we have. It is also not yet airtight** — see limitations.

### Symbol weighting works, and only where the theory says it should

![Symbol-weighting ablation](assets/ablation_panel.png)

We ran a proper check on whether the symbol-weighting gains are bigger than noise, and the answer is
**split-dependent** — which is the interesting part:

| Split | Metric | OFF | ON | Change | Bigger than noise? |
|---|---|--:|--:|--:|---|
| random | R@1 | 8.32 | 8.59 | +0.27 pts | **No** (~0.5 SE) |
| random | R@10 | 27.46 | 27.66 | +0.20 pts | **No** (~0.2 SE) |
| novel | R@1 | 7.55 | 8.48 | **+0.93 pts** | **Yes** (~2.3 SE) |
| novel | R@10 | 27.09 | 28.46 | **+1.37 pts** | **Yes** (~2.0 SE) |

So we **do not claim a symbol-weighting gain on `random`** — it's inside the error bars. We claim it
on **`novel_premises`**, where it clears the noise floor comfortably.

That's a *better* result than "it helps everywhere," because it's the pattern the theory predicts.
You cannot memorise a novel premise; you have to *match its symbols*. And novel premises are more
lexically distinctive (the gold lemma's name appears literally in the proof state 26.6% of the time
on `novel` vs 19.4% on `random`). Up-weighting symbol tokens should pay off precisely where symbolic
matching is all you have left — and that is exactly, and only, where it pays off.

*(These are marginal standard errors. `scripts/significance.py` runs a proper paired bootstrap +
permutation test over the per-example records; it's written and unit-tested, and runs as soon as the
per-example JSONs are pulled off the cluster.)*

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

## Where we fall short, and is the approach even right?

This is the section to read sceptically. We have gone looking for the holes rather than waiting for
someone else to find them.

### 1. We lose to dense ReProver on `random` — by a lot (27.66 vs 38.59 R@10)

This is the honest headline weakness. **Our retriever is not state of the art in absolute terms on
the standard split.** Three reasons we think it's a *floor* and not a *ceiling*:

- **We are undertrained.** One epoch, batch size 32, learning rate 3e-6 — and **validation loss was
  still falling when we stopped** (1.55 → 1.25). We used **4.8 GB of a 48 GB GPU**. Batch size is
  free quality in contrastive training (more in-batch negatives), and we left almost all of it on the
  table.
- **Our base model has the wrong tokenizer for the job.** We build on an English ColBERT whose BPE
  tokenizer fragments Lean's unicode operators. ReProver deliberately uses **byte-level ByT5** to
  avoid exactly this. That is a known, unaddressed handicap.
- **ReProver's recipe is mature**; ours is a first pass.

None of that is proven yet. Until we run the bigger budget, "we're undertrained" is a hypothesis, not
a result — and we should say so out loud.

### 2. The central claim is not yet a controlled comparison — this is the biggest hole

Our headline is "late interaction generalises better than single-vector pooling." But the comparison
we actually made is **our model vs ReProver**, and those two differ in *everything*: base model,
tokenizer, training data pipeline, negatives, budget, recipe. A sceptic can fairly say the
generalisation difference comes from any of those, not from late interaction.

**So the mechanism is currently inferred, not proven.** The fix is a **matched single-vector control**
— the same triplets, the same hard negatives, the same base lineage, the same budget, the same frozen
harness, with the *only* difference being single-vector cosine vs multi-vector MaxSim. The code is
written, tested and committed; **the cluster runs are the immediate next step.** Until they land, this
claim carries an asterisk, and we will present it with one.

### 3. Are we just rebuilding a neural BM25?

A real concern. BM25 (untrained) *also* does better on `novel` than `random`, because novel premises
are more lexically distinctive. Late interaction is more lexically sensitive than mean-pooled dense —
it's arguably a soft, learned lexical matcher. So some of our "flat generalisation gap" may be
inherited lexical sensitivity rather than a deeper structural advantage.

**Our defence:** we beat BM25 by ~2× on both splits, so we're clearly doing far more than lexical
overlap. **But it isn't measured.** The clean test is to stratify results by whether the gold lemma's
name literally appears in the proof state, and check that our advantage survives on the subset where
it *doesn't*. That's a planned ablation, not a completed one.

### 4. "Novel premises" is not the same as "novel symbols"

The split withholds *premises* the model hasn't seen used in training proofs. But Mathlib names are
highly compositional (`add_comm`, `mul_le_mul_left`), so a model can compose familiar subword units to
match a lemma it has never seen. Late interaction may be winning by **compositional lexical
matching** — which is consistent with our story, but is a weaker and more precise claim than "it
handles genuinely novel symbolic structure." We should state the claim at the strength the evidence
supports, not above it.

### 5. The symbol weighting is a hand-set heuristic

We fix `w = 4.0` for symbol subwords, chosen by a sweep on the **off-the-shelf** model and **never
re-tuned after fine-tuning** — the optimum has almost certainly moved. A *learned* token saliency
would be more principled. The current number is probably a lower bound on what the idea can do.

### 6. Single seed, no variance, no significance tests yet

Every number is one training run at seed 42. We have no run-to-run variance estimate, and the paired
significance test (written, tested) hasn't been run on the real per-example records yet.

### 7. Late interaction is expensive

We store ~12.45M token vectors for 180,973 premises — roughly **69× the index footprint** of a
single-vector system — and MaxSim is far costlier than a dot product. At eval scale (exact scoring
over the accessible set) that's fine. For a deployed prover it is a real cost, and we have not yet
quantified the latency/memory trade-off. "Better generalisation at 69× the index" is a claim that
needs a price tag attached.

### So — is the approach theoretically wrong?

**No.** Late interaction is a well-established IR result, and the claim that pooling destroys
token-level detail is uncontroversial. Our specific hypothesis — that this matters *more* in formal
mathematics because matching hinges on exact symbols — is reasonable and now has real supporting
evidence: the symbol-weighting lift appears **specifically and only on novel premises**, which is the
signature the theory predicts and would be a strange coincidence otherwise.

**But the evidence is suggestive, not conclusive**, and it will stay that way until the matched
control (#2) rules out the confounds. That is the single most valuable remaining experiment, and it's
next.

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

**The harness is frozen.** Metrics, the evaluation loop, and the accessibility filter have not
changed since calibration. A new model enters evaluation *only* as a new checkpoint path in a config
— so every system in the table above is scored by identical code.

Every run writes a JSON with full provenance (config, model, dataset version, git commit, seed) plus
the complete ranked list for every example. Metrics are pure, unit-tested functions. Everything is
seeded and reproducible, and the figures above are generated directly from the results file — no
hand-typed numbers.

---

## Status: where we are

### Part 1 — Evaluation harness and baselines ✅ **COMPLETE**

A calibrated, trustworthy harness; BM25, dense ReProver and off-the-shelf late interaction all
measured on both splits; the cross-split leakage discovered, quantified and designed around; all
figures generated. *This is the foundation that makes everything else defensible.*

### Part 2 — Fine-tuning 🔄 **~70% complete**

| Phase | Status |
|---|---|
| 11 · Pre-training audit & cheap tuning | ✅ Done — found 26% of proof states were being silently truncated (fixed), locked the symbol weight, capped the head of the premise distribution |
| 12 · Training-pair pipeline | ✅ Done — 335k/327k triplets, BM25-mined hard negatives, de-noised, leak-guarded |
| 13 · Late-interaction training loop | ✅ Done — PyLate, validated end-to-end |
| 14 · Fine-tuned LI on `random` | ✅ Done — the 2.5× lift |
| 15 · **Matched single-vector control** | ❌ **Not run** — code written, tested, committed; needs the cluster jobs |
| 16 · Both models on `novel_premises` | 🔄 **Half done** — our LI is done (the generalisation result); the single-vector half waits on Phase 15 |
| 17 · Ablations | 🔄 **Half done** — symbol-weighting on the trained model is done; the hard-vs-random negatives ablation is not |
| 18 · Figures, tables, write-up | 🔄 Provisional — will refresh once 15–17 land |

**What's left in Part 2, in priority order:**

1. **The matched single-vector control** (Phase 15 + 16) — turns the central claim from *inferred*
   into *proven*. Highest value by a distance.
2. **Significance testing** — the paired bootstrap is written and unit-tested; run it on the
   per-example records.
3. **Hard-negative ablation** (Phase 17) — how much of the gain came from BM25-mined negatives?
4. **Re-tune the symbol weight on the trained model** — it was tuned pre-training.
5. **The lexical-overlap stratification** — does our advantage survive on examples where the gold
   name *doesn't* appear in the state? (Answers criticism #3 above.)
6. **Push the training budget** — bigger batch, second epoch. Free headroom, and it directly attacks
   our biggest weakness (the `random` shortfall).

### Part 3 — Make it win outright, not just generalise better 📋 **Planned**

Right now we generalise better but lose in absolute terms on `random`. Part 3 closes that 28% gap.
The levers, roughly in order of expected payoff: a **much larger training budget** (we used 10% of the
GPU), a **Lean-aware or byte-level tokenizer** (the ReProver insight we haven't adopted), **learned
token saliency** instead of our hand-set symbol weight, and possibly **distillation from a
cross-encoder reranker**. This would get its own phase breakdown once Part 2 closes.

### Part 4 — Close the loop with a prover 📋 **Planned**

Retrieval metrics are a proxy. The metric the field actually cares about is **how many theorems get
proved**. Part 4 plugs ProofLens into a tactic generator and measures end-to-end proof success on
LeanDojo's test theorems — showing that better premise selection produces more completed proofs. This
is what turns the work from a retrieval result into a theorem-proving contribution.

### The short version

The **core scientific question has been answered**: fine-tuning transforms late interaction, symbol
weighting works specifically where the theory says it should, and the generalisation gap is
dramatically smaller than the single-vector reference. What remains in Part 2 is **making the central
claim airtight** (the control), **attributing the gains** (ablations), and **being honest about
significance**. Parts 3 and 4 are about turning a validated mechanism into a system that wins
outright and actually proves theorems.

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
scripts/            # download_data, build_index, build_pairs, train_li, train_sv, run_eval,
                    # significance, plot_results, audit
slurm/              # cluster jobscripts
tests/              # 122 tests: metrics, loaders, accessibility, every retriever, significance, smoke
```

## Getting started

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                             # 122 tests — no downloads, no GPU (tiny bundled fixtures)
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

### Is a difference real, or noise?

```bash
python scripts/significance.py \
  --a results/metrics/late_interaction_ft_novel_novel_premises.json \
  --b results/metrics/late_interaction_ft_novel_weighted_novel_premises.json
```

Paired bootstrap + sign-flip permutation test over the per-example records.

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
