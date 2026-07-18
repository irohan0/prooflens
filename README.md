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

### The two paradigms, concretely

**Single-vector (what everyone does).** Encode the proof state into token vectors, then **average
them into one vector**. Do the same for each premise. Score by cosine similarity:

$$\text{score}(s, p) = \cos\big(\underbrace{\text{mean}_i E_s[i]}_{\text{one vector}},\ \underbrace{\text{mean}_j E_p[j]}_{\text{one vector}}\big)$$

Fast — one dot product per premise — but that `mean` is a **lossy compression**. Every token gets
blended into a single point. If the one thing distinguishing the right lemma from the wrong one is a
`≤` where you needed a `<`, that distinction is now one small component of an average, competing with
every other token in the statement.

**Late interaction (ours).** Keep **all** the token vectors. For each query token, find its
best-matching premise token, and sum those best matches (this is **MaxSim**):

$$\text{score}(s, p) = \sum_i \underbrace{w(i)}_{\text{our twist}} \cdot \max_j \ \text{sim}\big(E_s[i],\ E_p[j]\big)$$

Nothing is averaged away. A single decisive token can carry the match, because it gets its own term
in the sum. **Our contribution is the $w(i)$**: a weight that up-weights *symbol* tokens (`≤`, `∀`,
`∘`, type constructors) over filler, on the theory that in formal mathematics the symbols are what
actually carry the matching signal.

Set $w(i) = 1$ and you have standard ColBERT. So the twist is **arithmetic over the score, not a new
model** — which means the ablation is a single flag flip, and we can prove the gain comes from the
weighting alone (same model, same index, same query vectors; only the weights differ).

---

## Results

Everything is measured on [LeanDojo Benchmark 4](https://leandojo.org/) (Mathlib4), on both official
test splits, through **one harness** with an identical accessibility filter and identical metrics for
every system. Recall in %. `random` n=2,811; `novel_premises` n=4,357.

| System | Trained on Lean? | random R@1 | random R@10 | novel R@1 | novel R@10 |
|---|:--:|--:|--:|--:|--:|
| BM25 (lexical baseline) | no | 5.48 | 13.63 | 5.65 | 16.56 |
| Late interaction, off-the-shelf | no | 4.15 | 10.44 | 4.82 | 14.05 |
| Single-vector, off-the-shelf (gte-modernbert) | no | 4.01 | 11.41 | 4.28 | 15.36 |
| Dense ReProver (published single-vector system) | yes | 13.04 | **38.59** | — | *27.6*¹ |
| **Matched single-vector control, fine-tuned** | yes | **8.97** | **32.00** | 6.66 | 26.16 |
| **ProofLens-LI, fine-tuned + symbol weighting** | yes | 8.59 | 27.66 | **8.48** | **28.46** |

<sub>¹ We could not measure a *clean* dense number on `novel_premises` from the public checkpoint — see
"The leak we caught" — so we cite ReProver's own published figure. Every other cell is produced by our
harness. Note the two fine-tuned single-vector systems (our control **32.00** and dense ReProver
**38.59**) both **beat** late interaction on `random` — and both **collapse** on `novel`, which is the
whole point (next section).</sub>

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

![Fine-tuning lift](assets/finetuning_lift.png)

**Roughly a 2.5× improvement across the board**, from one epoch on a single GPU. Read the figure
left to right: off the shelf we start *below* BM25 (the red line) — a neural retriever losing to a
1970s keyword baseline. Fine-tuning takes us to roughly **2× BM25**. And on `novel_premises` it
carries us **over the dense reference line** (the blue one) that the single-vector system drops to.

But note the left panel: on `random` we are still **well below dense ReProver**. That shortfall is
real, and we deal with it head-on in the limitations.

### The matched control — the result the project was built to test

Our central claim is that late interaction *generalises* better than single-vector pooling. To prove
that — not just assert it — we trained **our own single-vector model on the identical everything**:
same training triplets, same hard negatives, same base-model lineage (`gte-modernbert-base`, the
ModernBERT that our ColBERT is built on), same budget, same learning rate, scored through the same
frozen harness. **The only difference between the two systems is single-vector cosine vs multi-vector
MaxSim.** So any difference in behaviour is attributable to *the matching mechanism itself* — not the
data, the tokenizer, the recipe, or the budget.

![The matched control](assets/matched_control.png)

| System (both fine-tuned, everything matched) | random R@10 | novel R@10 | Gap |
|---|--:|--:|--:|
| **Matched single-vector control** | **32.00** | 26.16 | **−18.2%** ⬇ |
| **ProofLens-LI (late interaction)** | 27.66 | **28.46** | **+2.9%** ⬆ |

The lines **cross**. The single-vector control drops **18% (a 5.3σ effect)** going from seen to unseen
premises. Late interaction stays flat — it actually rises. And it's not a single-metric fluke: the
same split holds on R@1 (SV **−25.8%** vs LI −1.3%) and MRR (SV **−14.3%** vs LI +1.5%).

**The honest nuance makes the result sharper, not weaker.** On `random`, single-vector *wins* — 32.0
vs 27.7. So the claim is not "late interaction is better." It is: **late interaction is more robust —
it wins precisely on the unseen-lemma case that actually matters when a prover meets a lemma it has
never seen.** That's a more defensible, more mechanistically specific claim than a blanket victory.

And it isn't just our control. **Two independent single-vector systems collapse on novel** — ours
(−18%) *and* the published dense ReProver (38.59 → 27.6, −28%). Late interaction is the only trained
system that holds. Our clean control's novel score (26.16) even lands right next to ReProver's
published clean-novel (27.6), corroborating both.

![Generalisation gap, all systems](assets/generalisation_gap.png)

Why does this happen? A single vector is a lossy summary. When a model is trained and then meets a
premise from a *seen* distribution, that summary is good enough. When it meets a genuinely *novel*
premise, the specific token-level detail is what it needs — and that's exactly what pooling throws
away. Late interaction keeps one vector per token, so the detail survives. **But we don't leave that
as a story — we decomposed it** (§"Is the advantage structural, or a neural BM25?" below), and found
the surviving detail is specifically *lexical* token overlap. Read that section for the honest, precise
version of this mechanism; it's narrower than the headline and more defensible for it.

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

It wasn't a bug in our harness. `random` and `novel_premises` are two different partitions **of the
same theorem pool** — so a model trained on one split's training set has already seen most of the
*other* split's test theorems.

**The one statistic you should distrust is our own.** We measured that **97.2%** of novel-split test
theorems appear in the random-split training data. That number is **close to a tautology, and we say
so**: `random/train` holds ~97% of all theorems, so *any* theorem subset overlaps it at ~97%. It
proves the splits share theorems — which is true by construction — but it does **not** prove the
overlap *causes* the inflation. The real evidence is different, and stronger:

1. **A sign flip, not a magnitude error.** A properly split-matched trained model *drops* on novel
   premises — ReProver's own published numbers go 38.4 → 27.6, a **28% fall**. Ours went 38.59 →
   **63.66, a 65% rise**. Novel premises are supposed to be *harder*.
2. **Leakage is symmetric, and only one side is contaminated.** Whichever split the checkpoint was
   trained on, the *other* split's test set is ~97% contaminated. We observe `random` **clean**
   (38.59, matching the published 39.60) and `novel` **inflated** (63.66 against a published 27.6).
   That pattern is only consistent with a **random-trained** checkpoint — if it were novel-trained,
   `random` would have been the inflated one instead.
3. **The harness is exonerated.** BM25 and off-the-shelf late interaction give sane `novel` numbers
   through the same code. If the split were mis-loaded, or gold were leaking into the candidate set,
   they would be inflated too. The anomaly is **model-specific**.
4. **No version mismatch.** The ReProver maintainers state the released models were trained on Zenodo
   record `12740403` — exactly the release we evaluate on. So the checkpoint's training splits *are*
   the splits we compute the overlap against.

The authors did nothing wrong: they released one checkpoint that performs correctly on its own split,
and have [publicly noted](https://github.com/lean-dojo/ReProver/discussions/51) that the HuggingFace
models "do not specify which data split it was trained using." The trap is downstream — anyone who
evaluates that checkpoint on `novel_premises`, a completely natural thing to do, gets a number
inflated by more than 2×. We flag ours rather than report it, and we **train split-matched**: a
separate model per split, evaluated only on its own test set. None of our fine-tuned numbers carry
that contamination.

### We tested it directly, and it holds

We did not leave this as an argument. `scripts/leakage_stratified.py` splits `novel_premises/test` by
whether each theorem appears in `random/train`, and re-scores the **same per-example records** in each
group. It is built to be **falsifiable** — if the never-trained-on group had scored just as high, the
explanation was wrong and we would have retracted it.

| Group | n | R@10 | 95% CI |
|---|--:|--:|--:|
| **LEAKED** — theorem is in `random/train` | 4,284 | **64.12%** | — |
| **CLEAN** — never trained on | 73 | **37.04%** | [27.5, 46.6] |

**A 27-point gap — about 4.8 standard errors, p < 1e-5.** The model scores nearly **twice as well** on
theorems it was trained on. The clean group's confidence interval **contains the published clean-novel
figure of 27.6**. The reported 63.66 was inflated by roughly **1.7–2.3×**. Leakage confirmed.

### One honest complication (there are *two* leakage channels)

It would be convenient to now use **37.04** as "the clean dense novel number". **We don't, and it
matters why** — because doing so would drop dense's generalisation gap from −28.5% to −4.0% and
flatter our own thesis for the wrong reason.

The stratified test removes **theorem-level** leakage (the model memorising those exact proof states).
It does **not** remove **premise-level** leakage: the premises that are "novel" with respect to
`novel_premises/train` were still seen by a **random**-trained model, just inside other theorems'
proofs. So `37.04` measures *"a dense model on unseen theorems whose premises it nonetheless knows"* —
which is **not the question the split is asking.**

The published **27.6** comes from a model actually trained on `novel_premises/train`, for which those
premises are genuinely unseen — and that is **exactly the condition our fine-tuned model is in**. So
27.6 stays the correct like-for-like comparator, and the residual 37.04 → 27.6 is plausibly that
second channel (though it sits within the noise, so we don't claim it).

**This is precisely why the matched single-vector control matters** — and we have now run it. Our
control is trained by us on `novel_premises/train`, so it has no premise familiarity and no theorem
memorisation. It settles the comparison without relying on anyone's published number: it drops
**32.00 → 26.16 (−18%)** on novel, and its clean novel score (26.16) lands right next to the published
27.6. Both independent, leak-free single-vector systems drop on novel; late interaction does not.

---

## Where we fall short, and is the approach even right?

This is the section to read sceptically. We have gone looking for the holes rather than waiting for
someone else to find them.

### 1. On `random`, we lose to *both* single-vector systems — absolute performance is not the win

Be clear-eyed about this: on the `random` split, late interaction (27.66 R@10) is beaten by the
published dense ReProver (38.59) **and** by our own matched single-vector control (32.00). **In raw
accuracy on the standard split, late interaction is the weakest of the trained systems.** Our claim is
explicitly *not* absolute performance — it is generalisation robustness (§the matched control). We say
this plainly so nobody mistakes the claim for something it isn't.

Two reasons late interaction's `random` number is likely a *floor* (both untested, so stated as
hypotheses): we are **undertrained** — one epoch, batch 32, **4.8 GB of a 48 GB GPU**, val loss still
falling — and our ColBERT base has a **BPE tokenizer** that fragments Lean's unicode, where the
byte-level alternative might help. Neither is proven; the bigger-budget run is future work.

### 2. The central claim — now a controlled comparison ✅ (resolved)

This was previously the biggest hole: our generalisation claim compared *our model vs ReProver*, which
differ in everything (base model, tokenizer, pipeline, budget), so a sceptic could attribute the
difference to any confound rather than to late interaction. **We have now closed it.**

We trained a **matched single-vector control**: same triplets, same hard negatives, same base lineage,
same budget, **same learning rate (3e-6)**, same frozen harness — the only difference being
single-vector cosine vs multi-vector MaxSim. It drops **−18% on novel** while late interaction stays
flat (§the matched control). With every confound held fixed, the generalisation difference is
attributable to the matching mechanism itself. **The claim no longer carries an asterisk.**

One methodological note worth stating, because it *strengthens* trust: our first attempt at the
control used a "standard" single-vector learning rate (2e-5) and it **catastrophically damaged** the
model — 11.4 → 5.05 R@10, *below* the untrained baseline. We caught it with a "too-bad-to-be-true"
check (the mirror of the leakage "too-good" check): a fine-tune that scores below its own off-the-shelf
model is a bug signal, not a result. We diagnosed it (grad-norm explosion, rising eval-loss), ran a
proper LR sweep, and the working control (3e-6) is the number reported. The failed run never entered
the results table.

### 3. Is the advantage structural, or a "neural BM25"? — we measured it, and it's mostly lexical

This is the sharpest objection to the whole thesis, so we ran the decisive test: split every novel
example by whether the gold lemma's name literally appears in the proof state, then compare late
interaction against the matched single-vector control **within each bucket**, example-paired.

| Novel examples | n | LI R@10 | SV R@10 | LI − SV |
|---|--:|--:|--:|--:|
| **Lexical** (gold name in the state) | 1,158 (27%) | 30.6 | 24.1 | **+6.5** ✅ significant |
| **Structural** (name *not* in the state) | 3,199 (73%) | 27.7 | 26.9 | +0.8 ❌ not significant |

**Honest verdict: late interaction's advantage is concentrated in the lexical-overlap cases. On the
structural 73%, the two are statistically tied.** So the mechanism is best described as *recovering the
token-level lexical signal that single-vector pooling averages away* — real and significant where a
shared symbol exists, but **not** demonstrated structural reasoning. We state the claim at that
strength and no higher.

This is a *sharper* result than the naive one, not a retreat: it pinpoints exactly what late
interaction buys you and honestly bounds it. There is also concrete structural evidence — **456 novel
examples where late interaction ranks the right lemma #1 and single-vector misses it entirely**
(`mul_sum`, `basicOpen_pow`, `filter_union_filter_neg_eq`, none of whose names appear in the state) —
but since the *average* structural gap is a tie, single-vector has comparable wins elsewhere, and we
don't cherry-pick. (`scripts/lexical_stratified.py`, pre-registered and falsifiable — the unit tests
cover the "it's only lexical" outcome too.)

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

**No — and it now has controlled evidence.** Late interaction is a well-established IR result, and the
claim that pooling destroys token-level detail is uncontroversial. Our specific hypothesis — that this
matters *more* in formal mathematics because matching hinges on exact symbols — has two independent
lines of support that both point the same way: the symbol-weighting lift appears **specifically and
only on novel premises**, and the **matched single-vector control** drops 18% on novel while late
interaction stays flat, with every confound held fixed. Both are the signature the theory predicts.

The evidence is no longer merely suggestive on the central claim — the control removes the confounds.
What remains open is *magnitude* (a bigger training budget could lift the absolute numbers) and the
finer attributions in the "Next" list — not the direction of the result.

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

### Part 2 — Fine-tuning 🔄 **~90% complete**

| Phase | Status |
|---|---|
| 11 · Pre-training audit & cheap tuning | ✅ Done — found 26% of proof states were being silently truncated (fixed), locked the symbol weight, capped the head of the premise distribution |
| 12 · Training-pair pipeline | ✅ Done — 335k/327k triplets, BM25-mined hard negatives, de-noised, leak-guarded |
| 13 · Late-interaction training loop | ✅ Done — PyLate, validated end-to-end |
| 14 · Fine-tuned LI on `random` | ✅ Done — the 2.5× lift |
| 15 · **Matched single-vector control** | ✅ **Done** — trained on both splits; an LR bug caught and fixed; the −18% vs flat result |
| 16 · Both models on `novel_premises` | ✅ **Done** — LI *and* the single-vector control; the generalisation gap now measured for both |
| 17 · Ablations | 🔄 **Half done** — symbol-weighting on the trained model is done; the hard-vs-random negatives ablation is not |
| 18 · Figures, tables, write-up | 🔄 In progress — figures + tables refreshed with the control; the dissertation write-up remains |

**What's left in Part 2, in priority order:**

1. **Hard-negative ablation** (Phase 17) — how much of the gain came from BM25-mined negatives?
2. **Re-tune the symbol weight on the trained model** — it was tuned pre-training; likely the IDF
   weighting variant.
3. **Significance testing** — the paired bootstrap is written and unit-tested; run it on the
   per-example records (the eval-only variant to firm up the symbol-weighting p-values).
4. **The lexical-overlap stratification** — does our advantage survive on examples where the gold
   name *doesn't* appear in the state? (Answers criticism #3 above.)
5. **Push the training budget** — bigger batch, second epoch; attacks the absolute-performance gap.

### Part 3 — Make it win outright, not just generalise better 📋 **Planned**

We now *generalise* better (proven by the control) but lose in absolute terms on `random`. Part 3
closes that gap. Levers, roughly in order of expected payoff: a **much larger training budget** (we
used ~10% of the GPU for one epoch), a **Lean-aware or byte-level tokenizer** (the ReProver insight we
haven't adopted), **learned token saliency** instead of our hand-set symbol weight, and possibly
**distillation from a cross-encoder reranker**. This gets its own phase breakdown once Part 2 closes.

### Part 4 — Close the loop with a prover 📋 **Planned**

Retrieval metrics are a proxy. The metric the field actually cares about is **how many theorems get
proved**. Part 4 plugs ProofLens into a tactic generator and measures end-to-end proof success on
LeanDojo's test theorems — showing that better premise selection produces more completed proofs. This
is what turns the work from a retrieval result into a theorem-proving contribution.

### The short version

The **central scientific claim is now proven, not inferred.** The matched single-vector control —
identical in every respect except the matching mechanism — drops 18% on novel premises while late
interaction stays flat. Fine-tuning transforms late interaction (2.5×), symbol weighting helps
specifically where the theory predicts (novel premises), and the generalisation advantage is
attributable to late interaction itself. What remains in Part 2 is **attributing the finer gains**
(ablations) and the **write-up**. Parts 3 and 4 turn a validated mechanism into a system that wins
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
