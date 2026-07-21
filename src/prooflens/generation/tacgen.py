"""ReProver's retrieval-augmented tactic generator — the FIXED component of the Phase-21 study.

Phase 21 isolates the retriever's downstream contribution by holding this generator constant and
varying only the premises in its context. That only works if the generator is driven exactly as
ReProver drives it, so every call below is replicated from ReProver's
`prover/tactic_generator.py::HuggingFaceGenerator` (github.com/lean-dojo/ReProver):

- **Tokenize:** `tokenizer(state, max_length=max_inp_seq_len, truncation=True, return_tensors="pt")`
- **Generate:**::

      generator.generate(
          input_ids=state_ids, attention_mask=state_mask,
          max_length=self.max_oup_seq_len, num_beams=num_samples,
          length_penalty=self.length_penalty, do_sample=False,
          num_return_sequences=num_samples, early_stopping=False,
          output_scores=True, return_dict_in_generate=True,
      )

- **Decode:** `batch_decode(..., skip_special_tokens=True)`, then per candidate::

      t = remove_marks(raw_output_text[j])
      if t not in output_text:            # dedupe, keeping the best-scoring occurrence
          output_text.append(t); output_score.append(raw_scores[j])

  Applying `remove_marks` to the *generated* text is load-bearing: the model can emit `<a>`/`</a>`
  markers, and the reference tactic is mark-free, so skipping it would silently fail real matches.
- **Scores:** `output.sequences_scores` — beam-search sequence log-probabilities (length-penalised
  by `length_penalty`). Used only to order candidates; the metrics never threshold on them.

Defaults come from ReProver's Lean 4 config (`generation/confs/cli_lean4_random.yaml`):
`max_inp_seq_len=2300`, `max_oup_seq_len=512`, `length_penalty=0.0`.

**Deliberate design note — one state at a time, not batched.** ReProver's *proving* path generates
per state, and batching changes padding, which can perturb beam search numerically. Since the
headline number must be trustworthy rather than fast, we keep the faithful unbatched path and get
throughput from running the (independent) conditions as parallel jobs. `generate` logs its own
timing so a `--limit` pilot can extrapolate the full-run cost *before* committing GPU hours — the
Phase-12 lesson about measuring per-example cost first (`results/LEARNINGS.md` [2026-07-10]).

torch/transformers are imported lazily so this module (and its hermetic tests, which inject a fake
generator) can be exercised without a GPU or the ML stack installed.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from prooflens.generation.format import remove_marks
from prooflens.utils.logging import get_logger

log = get_logger("tacgen")

# ReProver generation/confs/cli_lean4_random.yaml
DEFAULT_MAX_INP_SEQ_LEN = 2300
DEFAULT_MAX_OUP_SEQ_LEN = 512
DEFAULT_LENGTH_PENALTY = 0.0


@runtime_checkable
class TacticGenerator(Protocol):
    """Anything that turns an (already augmented) proof state into ranked tactic candidates.

    The eval loop depends only on this, so the hermetic tests can inject a deterministic fake and
    exercise the whole orchestration without torch.
    """

    def generate(self, state: str, num_samples: int) -> list[tuple[str, float]]:
        """Return `[(tactic, score)]`, best first, deduplicated."""
        ...


class ByT5TacticGenerator:
    """ReProver's ByT5 tactic generator (`kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small`).

    Note this is the *retrieval-augmented* checkpoint: it was trained with retrieved premises
    prepended to the state, which is why the "no premises" condition is a meaningful floor rather
    than a different model.
    """

    def __init__(
        self,
        model_path: str,
        max_inp_seq_len: int = DEFAULT_MAX_INP_SEQ_LEN,
        max_oup_seq_len: int = DEFAULT_MAX_OUP_SEQ_LEN,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # ByT5 is a T5 encoder-decoder; AutoModelForSeq2SeqLM resolves to
        # T5ForConditionalGeneration, the class ReProver instantiates directly.
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        self.max_inp_seq_len = max_inp_seq_len
        self.max_oup_seq_len = max_oup_seq_len
        self.length_penalty = length_penalty
        # Truncation bookkeeping — docs/EVALUATION.md asks for the truncation rate in provenance.
        # A high rate here would mean premises are being cut off, which would blunt the very
        # signal we are measuring, so it must be reported, not assumed negligible.
        self.n_generated = 0
        self.n_truncated = 0

    def generate(self, state: str, num_samples: int) -> list[tuple[str, float]]:
        """Generate up to `num_samples` tactic candidates for `state` (already augmented)."""
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        torch = self._torch
        tokenized_state = self.tokenizer(
            state, max_length=self.max_inp_seq_len, truncation=True, return_tensors="pt"
        )
        state_ids = tokenized_state.input_ids.to(self.device)
        state_mask = tokenized_state.attention_mask.to(self.device)

        self.n_generated += 1
        if int(state_mask.sum().item()) >= self.max_inp_seq_len:
            self.n_truncated += 1

        with torch.no_grad():
            output = self.model.generate(
                input_ids=state_ids,
                attention_mask=state_mask,
                max_length=self.max_oup_seq_len,
                num_beams=num_samples,
                length_penalty=self.length_penalty,
                do_sample=False,
                num_return_sequences=num_samples,
                early_stopping=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        raw_output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )
        raw_scores = output.sequences_scores.tolist()
        return dedupe_candidates(raw_output_text, raw_scores)

    def truncation_stats(self) -> dict[str, float | int]:
        """Truncation counters for the run's provenance header."""
        rate = (self.n_truncated / self.n_generated) if self.n_generated else 0.0
        return {
            "n_generated": self.n_generated,
            "n_truncated": self.n_truncated,
            "truncation_rate": rate,
            "max_inp_seq_len": self.max_inp_seq_len,
        }


def dedupe_candidates(
    raw_output_text: list[str],
    raw_scores: list[float],
) -> list[tuple[str, float]]:
    """Strip marks and drop duplicate tactics, keeping each at its best (first) rank.

    Split out from `ByT5TacticGenerator.generate` so this post-processing — the part that decides
    what actually gets compared against the reference tactic — is unit-testable without torch.
    Mirrors ReProver's loop verbatim; beam output is score-ordered, so the first occurrence of a
    duplicated string is its best-scoring one.
    """
    if len(raw_output_text) != len(raw_scores):
        raise ValueError(
            f"decoded {len(raw_output_text)} sequences but got {len(raw_scores)} scores"
        )
    output_text: list[str] = []
    output_score: list[float] = []
    for text, score in zip(raw_output_text, raw_scores, strict=True):
        t = remove_marks(text)
        if t not in output_text:
            output_text.append(t)
            output_score.append(score)
    return list(zip(output_text, output_score, strict=True))
