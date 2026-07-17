"""Dense single-vector retriever — the official ReProver ByT5-small checkpoint (no training).

This is the **harness-correctness calibration anchor** (docs/EVALUATION.md §3): the ReProver
checkpoint run through our harness must land in its published reference range, or the harness is
wrong. So every encoding choice here is replicated **verbatim from ReProver's own source**
(github.com/lean-dojo/ReProver), not approximated:

- **Load:** `AutoTokenizer` + `AutoModelForTextEncoding` — for ByT5 the latter returns the T5
  *encoder*, whose `last_hidden_state` we pool. (ReProver `retrieval/model.py`.)
- **Pooling:** masked mean over tokens — `(hidden * mask).sum(1) / mask.sum(1)`.
- **Normalize:** `F.normalize(features, dim=1)` (unit vectors).
- **Similarity:** dot product of the unit vectors == cosine (ReProver: "cosine similarities for
  unit-norm vectors are just inner products", `torch.mm(context_emb, premise_embs.t())`).
- **max_length = 1024, truncation = True** (ReProver `retrieval/confs/*lean4*.yaml`).
- **Premise text = ReProver `Premise.serialize()`** (common.py): the premise `code` with its own
  fully-qualified name marked by `<a>…</a>`. This is NOT `full_name + code` (that is BM25's own
  representation); the dense retriever must use ReProver's exact serialization.
- **Query text = the proof state** (`state_before`), i.e. ReProver `Context.serialize()` returns
  `self.state`.

**Candidate scoring — exact, matching ReProver's *evaluation*, not ANN.** ReProver's benchmark
eval (`Corpus.get_nearest_premises`) computes the full dot-product matrix and then filters to the
accessible premises; it does not use an approximate FAISS index for the accessibility-restricted
metric. Because cosine is independent per premise, scoring *only* the accessible subset and taking
the top-k is identical to scoring all premises then filtering — the same optimisation BM25 uses
here. We therefore hold the premise matrix in memory and do an exact matmul over the accessible
rows. This is deliberately *exact* (no ANN approximation): a calibration anchor must not introduce
recall error of its own. (Documented deviation from ARCHITECTURE.md §5's "FAISS index" wording;
the numbers are strictly more faithful to ReProver this way. See results/phase_logs/phase7.md.)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from prooflens.retrievers.base import Retriever
from prooflens.retrievers.bm25 import premise_document
from prooflens.utils.logging import get_logger

log = get_logger("dense")

# ReProver common.py: MARK_START_SYMBOL / MARK_END_SYMBOL used by Premise.serialize().
MARK_START_SYMBOL = "<a>"
MARK_END_SYMBOL = "</a>"


def serialize_premise(full_name: str, code: str) -> str:
    """Reproduce ReProver's `Premise.serialize()` **verbatim** (common.py).

    Marks the premise's own fully-qualified name in its source code with `<a>…</a>` so the encoder
    can attend to the self-reference. Two passes, exactly as ReProver:
    1. replace the explicit `_root_.<full_name>` form, then
    2. replace the first (longest) dotted suffix of `<full_name>` that occurs preceded by
       whitespace (optionally wrapped in Lean's `«»` name quotes), and stop.

    NOTE: the regex intentionally leaves the dotted prefix un-escaped (so `.` matches any char),
    exactly as in ReProver — replicating a quirk of the reference implementation is required for
    the calibration to match. Do not "fix" it.
    """
    annot_full_name = f"{MARK_START_SYMBOL}{full_name}{MARK_END_SYMBOL}"
    code = code.replace(f"_root_.{full_name}", annot_full_name)
    fields = full_name.split(".")
    for i in range(len(fields)):
        prefix = ".".join(fields[i:])
        try:
            new_code = re.sub(rf"(?<=\s)«?{prefix}»?", annot_full_name, code)
        except re.error:
            # ReProver's pattern leaves the name un-escaped; on Benchmark 4 it never raises (that
            # code produced the published numbers), so this guard is a no-op for the calibration —
            # it only stops a pathological name from aborting a multi-hour GPU index build.
            break
        if new_code != code:
            code = new_code
            break
    return code


class _ByT5Encoder:
    """Thin wrapper around the ReProver ByT5 retriever checkpoint. Torch/transformers are imported
    lazily so the module (and the hermetic tests, which inject a fake encoder) do not require them.
    `encode(texts)` returns an L2-normalised float32 array `[n, d]`, exactly ReProver's `_encode`.
    """

    def __init__(self, model_path: str, max_length: int = 1024, batch_size: int = 64,
                 device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForTextEncoding, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTextEncoding.from_pretrained(model_path)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.dim = int(self.model.config.d_model)
        # truncation bookkeeping (docs/EVALUATION.md asks us to log the truncation rate)
        self.n_truncated = 0
        self.n_encoded = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                toks = self.tokenizer(
                    batch, max_length=self.max_length, truncation=True,
                    padding=True, return_tensors="pt",
                )
                input_ids = toks["input_ids"].to(self.device)
                attention_mask = toks["attention_mask"].to(self.device)
                hidden = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state
                # masked mean pooling, then L2 normalise (ReProver _encode, verbatim maths)
                mask = attention_mask.unsqueeze(2).to(hidden.dtype)
                lens = attention_mask.sum(dim=1).clamp(min=1)  # every seq has >=1 token (eos)
                feat = (hidden * mask).sum(dim=1) / lens.unsqueeze(1)
                feat = torch.nn.functional.normalize(feat, dim=1)
                out.append(feat.to(torch.float32).cpu().numpy())
                self.n_truncated += int(
                    (attention_mask.sum(dim=1) >= self.max_length).sum().item()
                )
                self.n_encoded += len(batch)
        return np.concatenate(out, axis=0)


class _STEncoder:
    """sentence-transformers single-vector encoder for the **matched control** (the checkpoint's
    own pooling — gte-modernbert-base uses CLS-token pooling, per its ST module config — then
    L2-normalize). Loads a `SentenceTransformer` checkpoint fine-tuned by `scripts/train_sv.py`;
    `encode(texts)` returns an L2-normalised float32 `[n, d]` array — the SAME output contract as
    `_ByT5Encoder`, so `DenseRetriever.retrieve` (exact cosine over the accessible set) is byte-for-
    byte unchanged. torch/sentence-transformers are imported lazily (cluster-only).
    """

    def __init__(self, model_path: str, max_length: int = 512, batch_size: int = 64,
                 device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_path, device=device)
        self.model.max_seq_length = max_length
        self.max_length = max_length
        self.batch_size = batch_size
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.n_encoded = 0
        self.n_truncated = 0            # ST truncates silently; not tracked

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        self.n_encoded += len(texts)
        return np.ascontiguousarray(emb, dtype=np.float32)


class DenseRetriever(Retriever):
    """ReProver ByT5-small dense retriever. Encodes every premise once (persisted to the index
    dir), then at query time encodes the state and scores the accessible premises by exact cosine.
    An `encoder` may be injected (used by the hermetic tests); otherwise a `_ByT5Encoder` is built
    lazily from `model_path`.
    """

    def __init__(self, model_path: str | None = None, max_length: int = 1024,
                 batch_size: int = 64, index_dir: str | None = None,
                 encoder: object | None = None, device: str | None = None,
                 encoder_type: str = "byt5", premise_text: str = "reprover_serialize") -> None:
        self.model_path = model_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.index_dir = index_dir
        self.device = device
        # encoder_type: "byt5" (ReProver, default) | "sentence_transformer" (the matched control).
        # premise_text: "reprover_serialize" (default) | "full_name_code" (matches LI/pairs text).
        self.encoder_type = encoder_type
        self.premise_text = premise_text
        self._encoder = encoder
        self._emb: np.ndarray | None = None      # [N, d] float32, L2-normalised premise vectors
        self._uids: list[str] = []
        self._uid_to_idx: dict[str, int] = {}

    def _get_encoder(self):
        if self._encoder is None:
            if not self.model_path:
                raise RuntimeError("DenseRetriever needs model_path or an injected encoder")
            if self.encoder_type == "sentence_transformer":
                self._encoder = _STEncoder(
                    self.model_path, self.max_length, self.batch_size, self.device
                )
            else:
                self._encoder = _ByT5Encoder(
                    self.model_path, self.max_length, self.batch_size, self.device
                )
        return self._encoder

    def _premise_text(self, full_name: str, code: str) -> str:
        """Serialize a premise for indexing. ReProver's `<a>`-marked form by default; the matched
        single-vector control uses `full_name + code` so its premise text matches the LI/pairs text
        (a like-for-like comparison of the matching mechanism, not the representation)."""
        if self.premise_text == "full_name_code":
            return premise_document(full_name, code)
        return serialize_premise(full_name, code)

    # -- index ---------------------------------------------------------------------------------
    def _paths(self) -> tuple[Path, Path, Path] | None:
        if not self.index_dir:
            return None
        d = Path(self.index_dir)
        return d / "dense_emb.npy", d / "dense_uids.json", d / "dense_meta.json"

    def build_index(self, corpus, rebuild: bool = False) -> None:
        """Encode every premise's ReProver-serialized text into a normalised vector and persist.
        Reload a persisted index if present (unless `rebuild`)."""
        paths = self._paths()
        if paths is not None and not rebuild and paths[0].exists() and paths[1].exists():
            self._load(paths)
            log.info("dense: loaded persisted index (%d premises, dim=%d)",
                     len(self._uids), self._emb.shape[1] if self._emb is not None else -1)
            return

        self._uids = [p.uid for p in corpus.all_premises]
        self._uid_to_idx = {uid: i for i, uid in enumerate(self._uids)}
        texts = [self._premise_text(p.full_name, p.code) for p in corpus.all_premises]
        enc = self._get_encoder()
        log.info("dense: encoding %d premises (batch=%d, max_length=%d)",
                 len(texts), self.batch_size, self.max_length)
        self._emb = np.ascontiguousarray(enc.encode(texts), dtype=np.float32)
        n_enc = getattr(enc, "n_encoded", 0)
        if n_enc:
            n_tr = getattr(enc, "n_truncated", 0)
            log.info("dense: premise truncation %d/%d (%.2f%%) reached max_length=%d",
                     n_tr, n_enc, 100.0 * n_tr / n_enc, self.max_length)
        if paths is not None:
            self._save(paths)

    def _save(self, paths: tuple[Path, Path, Path]) -> None:
        emb_p, uids_p, meta_p = paths
        emb_p.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_p, self._emb)
        uids_p.write_text(json.dumps(self._uids), encoding="utf-8")
        meta_p.write_text(json.dumps({
            "model_path": self.model_path,
            "n": len(self._uids),
            "dim": int(self._emb.shape[1]) if self._emb is not None and self._emb.size else None,
            "max_length": self.max_length,
            "encoder_type": self.encoder_type,
            "premise_text": self.premise_text,
            "dtype": str(self._emb.dtype) if self._emb is not None else None,
        }), encoding="utf-8")

    def _load(self, paths: tuple[Path, Path, Path]) -> None:
        emb_p, uids_p, _meta_p = paths
        self._emb = np.ascontiguousarray(np.load(emb_p), dtype=np.float32)
        self._uids = json.loads(uids_p.read_text(encoding="utf-8"))
        self._uid_to_idx = {uid: i for i, uid in enumerate(self._uids)}
        if len(self._uids) != self._emb.shape[0]:
            raise ValueError(
                f"dense index corrupt: {len(self._uids)} uids != {self._emb.shape[0]} vectors"
            )

    # -- retrieve ------------------------------------------------------------------------------
    def retrieve(self, state: str, accessible: set[str], k: int) -> list[tuple[str, float]]:
        if self._emb is None:
            raise RuntimeError("build_index must be called before retrieve")
        # Score ONLY the accessible premises (the candidate set) — identical to scoring all then
        # filtering (cosine is per-premise independent), matching ReProver's get_nearest_premises.
        acc_uids = [uid for uid in accessible if uid in self._uid_to_idx]
        if not acc_uids:
            return []
        q = self._get_encoder().encode([state])[0]  # [d], unit-norm
        acc_idx = np.fromiter(
            (self._uid_to_idx[uid] for uid in acc_uids), dtype=np.int64, count=len(acc_uids)
        )
        scores = self._emb[acc_idx] @ q  # cosine (both sides unit-norm)
        candidates = list(zip(acc_uids, (float(s) for s in scores), strict=True))
        # Sort by score desc; tie-break by uid for determinism (identical rule to BM25).
        candidates.sort(key=lambda t: (-t[1], t[0]))
        return candidates[:k]
