"""ProofLens-LI — late-interaction (multi-vector) retriever via PyLate. **The contribution.**

A proof state and the lemma that discharges it match on specific symbols (an operator, a type
constructor, a function name). Single-vector retrievers pool those into one averaged vector and
blur them; late interaction keeps one vector per token and matches at the token level, preserving
exactly that signal (docs/ARCHITECTURE.md §6).

**Base mechanism (ColBERT-style, via PyLate):** encode the state and each premise into token-level
multi-vector representations (no pooling), score by **MaxSim** — for each query (state) token take
its maximum cosine similarity to any premise token, then sum over query tokens:

    score(s, p) = Σ_i  w(i) · max_j  sim(E_s[i], E_p[j])

`w(i)` is the symbol-anchored token weight (Phase 9); it is **1 for every token here (OFF)**. The
ON/OFF pair is the headline ablation. Token embeddings are L2-normalized, so the inner product is
cosine — matching PyLate's `colbert_scores` (verified against PyLate source: max over document
tokens, sum over query tokens).

**Candidate scoring — exact over the accessible set, NOT PLAID ANN.** Every retriever in this
harness must rank the *same* accessible candidate set (docs/ARCHITECTURE.md §1, §7). A PLAID/ANN
index retrieves an approximate top-N over the *whole* corpus; filtering that to accessible would
give LI a different, approximate candidate set than BM25/dense saw, invalidating the comparison.
So we compute **exact** MaxSim over the accessible premises (same principle as the dense retriever's
exact matmul). This keeps PyLate purely for GPU *encoding*; the scoring is vectorized numpy
(BLAS matmul + `reduceat` segment-max), which is exact, fast enough at eval scale, and fully unit
-testable without a GPU. (Documented deviation from ARCHITECTURE §6's "PLAID index"; PLAID is a
scaling path for the prover setting, not needed for the accessibility-restricted benchmark.)

**Premise text = `full_name + code`** (BM25's `premise_document`) — the two Lean-untrained methods
share one premise representation, so the sparse-vs-late-interaction comparison is about the
*matching mechanism*, not the text. Query text = the proof state (`state_before`).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from prooflens.retrievers.base import Retriever
from prooflens.retrievers.bm25 import premise_document
from prooflens.retrievers.token_idf import TokenIDF
from prooflens.utils.logging import get_logger

log = get_logger("late_interaction")


def maxsim_score(
    query_emb: np.ndarray, doc_emb: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Reference MaxSim (ColBERT late interaction) between a query and one document.

    `query_emb` is [n_q, dim], `doc_emb` is [n_d, dim], both L2-normalized. Returns
    Σ_i w(i)·max_j (query_emb[i]·doc_emb[j]); `weights` (None → all ones) is the per-query-token
    weighting. This is the obvious, un-vectorised definition — the retriever's batched `reduceat`
    path is unit-tested to agree with it.
    """
    sims = query_emb @ doc_emb.T          # [n_q, n_d] cosine (unit-norm inputs)
    maxs = sims.max(axis=1)               # [n_q] best premise token per query token
    if weights is None:
        return float(maxs.sum())
    return float((weights * maxs).sum())


# -- symbol-anchored token weighting (Phase 9 — the thesis's twist) ----------------------------
# The ColBERT tokenizer emits sub-word tokens (ModernBERT BPE uses "Ġ"/"Ċ" for space/newline).
_SUBWORD_MARKERS = ("Ġ", "Ċ", "ġ", "▁")
_IDENT_CHAR = re.compile(r"[A-Za-z0-9_']")
# Pure syntactic filler: brackets, separators, type-ascription colon, the ⊢ turnstile, whitespace.
_SYNTAX_FILLER = frozenset("()[]{}⟨⟩⦃⦄,;:.|`⊢ \t\n\r")


def is_symbol_subword(token: str, special_tokens: frozenset[str] = frozenset()) -> bool:
    """Two-tier symbol classification of one ColBERT sub-word token, for the MaxSim weighting.

    **Symbol** (up-weighted): identifier/name fragments (contain an identifier char) and
    mathematical operators (non-filler punctuation, e.g. ``≤ → + ∘ ∀``). **Non-symbol** (default
    weight): model special tokens (``[Q] [MASK] [CLS] …``), whitespace, and pure syntactic
    punctuation (brackets, separators, the ascription ``:`` and the ``⊢`` turnstile). Reuses the
    Lean-aware identifier character class; a "simple two-tier weight is enough to start"
    (docs/ARCHITECTURE.md §6).
    """
    if token in special_tokens or (token.startswith("[") and token.endswith("]")):
        return False
    s = token
    for marker in _SUBWORD_MARKERS:
        s = s.replace(marker, "")
    if not s:                                   # was pure whitespace/marker
        return False
    if _IDENT_CHAR.search(s):                    # a name/number fragment -> symbol
        return True
    return not all(c in _SYNTAX_FILLER for c in s)   # operator (not pure filler) -> symbol


def _to_numpy(x) -> np.ndarray:
    """torch.Tensor (possibly on GPU) or array-like -> contiguous numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class _ColBERTEncoder:
    """Wrapper around a PyLate ColBERT checkpoint. PyLate/torch are imported lazily so the module
    and the hermetic tests (which inject a fake encoder) do not need them. `encode_*` returns a
    list of per-text token-embedding arrays [n_tokens, dim], L2-normalized (PyLate does the
    normalization); PyLate runs the model on GPU automatically when one is present.
    """

    def __init__(self, model_path: str, query_length: int = 256, document_length: int = 300,
                 batch_size: int = 32, device: str | None = None) -> None:
        from pylate import models

        self.model = models.ColBERT(
            model_name_or_path=model_path,
            query_length=query_length,
            document_length=document_length,
            device=device,
        )
        self.batch_size = batch_size
        self.special_tokens = frozenset(self.model.tokenizer.all_special_tokens)
        self.n_doc_truncated = 0
        self.n_doc_encoded = 0

    def _encode(self, texts: list[str], is_query: bool) -> list[np.ndarray]:
        embs = self.model.encode(
            texts,
            is_query=is_query,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            output_value="token_embeddings",
            show_progress_bar=False,
        )
        # PyLate returns a single array for a single input; normalise to a list of float32 arrays.
        if isinstance(embs, np.ndarray) and embs.ndim == 2:
            embs = [embs]
        return [np.ascontiguousarray(e, dtype=np.float32) for e in embs]

    def encode_documents(self, texts: list[str]) -> list[np.ndarray]:
        out = self._encode(texts, is_query=False)
        self.n_doc_encoded += len(texts)
        return out

    def encode_queries(self, texts: list[str]) -> list[np.ndarray]:
        return self._encode(texts, is_query=True)

    def encode_query_with_tokens(self, text: str) -> tuple[np.ndarray, list[str]]:
        """Encode one query, returning (token embeddings [n_q, dim], sub-word token strings) that
        are **1:1 aligned** — needed for the symbol weighting. Uses `output_value=None` to recover
        `input_ids` alongside the embeddings, applies the query `masks` (so the result matches the
        filtered embeddings `encode_queries` returns), and L2-normalises (idempotent) so the ON path
        uses the same query vectors as OFF — the ablation then differs ONLY by the token weights.
        """
        out = self.model.encode(
            [text], is_query=True, batch_size=1, convert_to_numpy=False,
            normalize_embeddings=True, output_value=None, show_progress_bar=False,
        )
        # output_value=None returns a dict whose values are per-sentence lists (one input here);
        # [0] extracts this sentence whether the value is a list of tensors or a batched tensor.
        emb = _to_numpy(out["token_embeddings"][0]).astype(np.float32)   # [L, dim]
        ids = _to_numpy(out["input_ids"][0]).astype(np.int64)            # [L]
        # apply the query mask so the result matches encode_queries' filtered embeddings; fall back
        # to attention_mask, or no filtering, if a PyLate version omits "masks".
        mask_key = next((k for k in ("masks", "attention_mask") if k in out), None)
        if mask_key is not None:
            masks = _to_numpy(out[mask_key][0]).astype(bool)             # [L]
            if masks.shape[0] == emb.shape[0]:                          # drop padding/masked tokens
                emb, ids = emb[masks], ids[masks]
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = np.ascontiguousarray(emb / np.clip(norms, 1e-12, None), dtype=np.float32)
        tokens = [self.model.tokenizer.convert_ids_to_tokens(int(t)) for t in ids]
        return emb, tokens


class LateInteractionRetriever(Retriever):
    """ColBERT late-interaction retriever with exact MaxSim over the accessible set.

    Premise token embeddings are stored ragged (CSR): one concatenated `[T, dim]` matrix plus
    per-premise offsets. At query time we gather the accessible premises' tokens, compute all
    query-token↔premise-token cosines in one matmul, take the per-premise max for each query token
    (`reduceat`), and sum (optionally weighted) over query tokens.
    """

    def __init__(self, model_path: str | None = None, query_length: int = 256,
                 document_length: int = 300, batch_size: int = 32, index_dir: str | None = None,
                 weighting: dict | None = None, encoder: object | None = None,
                 device: str | None = None) -> None:
        self.model_path = model_path
        self.query_length = query_length
        self.document_length = document_length
        self.batch_size = batch_size
        self.index_dir = index_dir
        self.weighting = weighting or {"enabled": False}
        self.device = device
        self._encoder = encoder
        self._idf: TokenIDF | None = None        # lazily loaded for idf / symbol_idf modes
        self._tok: np.ndarray | None = None     # [T, dim] float32, all premise tokens concatenated
        self._off: np.ndarray | None = None      # [N+1] int64, CSR offsets into _tok
        self._uids: list[str] = []
        self._uid_to_idx: dict[str, int] = {}
        # per-theorem gather cache (evaluate.py reuses the same accessible-set object per theorem)
        self._cache_key: object | None = None
        self._cache_D: np.ndarray | None = None
        self._cache_seg: np.ndarray | None = None
        self._cache_uids: list[str] = []

    def _get_encoder(self):
        if self._encoder is None:
            if not self.model_path:
                raise RuntimeError("LateInteractionRetriever needs model_path or injected encoder")
            self._encoder = _ColBERTEncoder(
                self.model_path, self.query_length, self.document_length,
                self.batch_size, self.device,
            )
        return self._encoder

    # -- index ---------------------------------------------------------------------------------
    def _paths(self) -> tuple[Path, Path, Path, Path] | None:
        if not self.index_dir:
            return None
        d = Path(self.index_dir)
        return (d / "li_tok.npy", d / "li_off.npy", d / "li_uids.json", d / "li_meta.json")

    def build_index(self, corpus, rebuild: bool = False) -> None:
        """Encode every premise's token embeddings and store them ragged (CSR). Reload if present
        (unless `rebuild`). Symbol weighting does not affect premise encoding, so the ON run reuses
        this index unchanged."""
        paths = self._paths()
        if paths is not None and not rebuild and paths[0].exists() and paths[1].exists():
            self._load(paths)
            log.info("late_interaction: loaded persisted index (%d premises, %d tokens, dim=%d)",
                     len(self._uids), self._tok.shape[0] if self._tok is not None else -1,
                     self._tok.shape[1] if self._tok is not None else -1)
            return

        premises = corpus.all_premises
        self._uids = [p.uid for p in premises]
        self._uid_to_idx = {uid: i for i, uid in enumerate(self._uids)}
        texts = [premise_document(p.full_name, p.code) for p in premises]
        enc = self._get_encoder()
        log.info("late_interaction: encoding %d premises (batch=%d, document_length=%d)",
                 len(texts), self.batch_size, self.document_length)

        blocks: list[np.ndarray] = []
        offsets = [0]
        dim: int | None = None
        for start in range(0, len(texts), self.batch_size):
            batch_embs = enc.encode_documents(texts[start : start + self.batch_size])
            for e in batch_embs:
                if e.shape[0] == 0:                       # guard: never an empty premise block
                    e = np.zeros((1, e.shape[1] if e.ndim == 2 else dim or 1), dtype=np.float32)
                dim = e.shape[1]
                blocks.append(e)
                offsets.append(offsets[-1] + e.shape[0])
        self._tok = np.ascontiguousarray(np.concatenate(blocks, axis=0), dtype=np.float32)
        self._off = np.asarray(offsets, dtype=np.int64)
        log.info("late_interaction: %d premises -> %d tokens (dim=%d), mean %.1f tok/premise",
                 len(self._uids), self._tok.shape[0], self._tok.shape[1],
                 self._tok.shape[0] / max(len(self._uids), 1))
        if paths is not None:
            self._save(paths)

    def _save(self, paths: tuple[Path, Path, Path, Path]) -> None:
        tok_p, off_p, uids_p, meta_p = paths
        tok_p.parent.mkdir(parents=True, exist_ok=True)
        np.save(tok_p, self._tok)
        np.save(off_p, self._off)
        uids_p.write_text(json.dumps(self._uids), encoding="utf-8")
        meta_p.write_text(json.dumps({
            "model_path": self.model_path,
            "n_premises": len(self._uids),
            "n_tokens": int(self._tok.shape[0]) if self._tok is not None else None,
            "dim": int(self._tok.shape[1]) if self._tok is not None else None,
            "query_length": self.query_length,
            "document_length": self.document_length,
        }), encoding="utf-8")

    def _load(self, paths: tuple[Path, Path, Path, Path]) -> None:
        tok_p, off_p, uids_p, _meta_p = paths
        self._tok = np.ascontiguousarray(np.load(tok_p), dtype=np.float32)
        self._off = np.asarray(np.load(off_p), dtype=np.int64)
        self._uids = json.loads(uids_p.read_text(encoding="utf-8"))
        self._uid_to_idx = {uid: i for i, uid in enumerate(self._uids)}
        if len(self._uids) + 1 != self._off.shape[0]:
            raise ValueError(
                f"LI index corrupt: {len(self._uids)} uids but offsets len {self._off.shape[0]}"
            )

    # -- scoring -------------------------------------------------------------------------------
    def _gather(self, acc_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate the token blocks of the accessible premises into `D_acc` [T_acc, dim] and
        return the per-premise start offsets `seg` [m] (reduceat boundaries). Fully vectorised."""
        starts = self._off[acc_idx]
        lengths = self._off[acc_idx + 1] - starts
        total = int(lengths.sum())
        seg = np.empty(len(acc_idx), dtype=np.int64)
        seg[0] = 0
        if len(acc_idx) > 1:
            np.cumsum(lengths[:-1], out=seg[1:])
        prem_of_pos = np.repeat(np.arange(len(acc_idx)), lengths)
        intra = np.arange(total, dtype=np.int64) - seg[prem_of_pos]
        tok_idx = starts[prem_of_pos] + intra
        return self._tok[tok_idx], seg

    def _symbol_weights(self, tokens: list[str], encoder) -> np.ndarray:
        """Per-query-token MaxSim weights (Phase 9). Symbol tokens get `symbol_weight`, everything
        else `default_weight` (both from config, kept mild). Classification reuses the Lean-aware
        character classes via `is_symbol_subword`. This is the thesis's symbol-anchored twist —
        pure arithmetic over the MaxSim sum, no training."""
        sw = float(self.weighting.get("symbol_weight", 2.0))
        dw = float(self.weighting.get("default_weight", 1.0))
        specials = getattr(encoder, "special_tokens", frozenset())
        return np.array(
            [sw if is_symbol_subword(t, specials) else dw for t in tokens], dtype=np.float32
        )

    def _get_idf(self) -> TokenIDF:
        """Lazily load the corpus IDF table (Phase 22). Path is `weighting['idf_path']` if set,
        else `<index_dir>/token_idf.json`. Injectable for the hermetic tests (set `self._idf`)."""
        if self._idf is not None:
            return self._idf
        raw = self.weighting.get("idf_path")
        if raw:
            path = Path(os.path.expandvars(raw))
        elif self.index_dir:
            path = Path(os.path.expandvars(self.index_dir)) / "token_idf.json"
        else:
            raise RuntimeError(
                "idf weighting needs weighting['idf_path'] or an index_dir holding token_idf.json"
            )
        if not path.exists():
            raise FileNotFoundError(
                f"IDF table not found at {path} — run scripts/build_token_idf.py first"
            )
        self._idf = TokenIDF.load(path)
        log.info("late_interaction: loaded IDF table (%d tokens, N=%d) from %s",
                 len(self._idf), self._idf.n_docs, path)
        return self._idf

    def _token_weights(self, tokens: list[str], encoder) -> np.ndarray:
        """Per-query-token MaxSim weights, dispatching on `weighting['mode']`:

        - ``"symbol"`` (default) — Phase-9 flat weighting (`symbol_weight` vs `default_weight`).
        - ``"idf"`` — Phase-22 **data-derived**: every token weighted by its corpus IDF, so a rare
          symbol counts more than a common one. Scale-invariant (only relative weights matter).
        - ``"symbol_idf"`` — the direct upgrade of Phase 9: **symbol** tokens weighted by their IDF
          (instead of a flat 4.0), non-symbol tokens kept at `default_weight`. Here the IDF scale
          relative to the default matters, so `idf_scale` is available (default 1.0).

        `idf_fallback` (default 1.0) is used for tokens absent from the table (special tokens; any
        sub-word unseen in premises) — deliberately neutral so they neither help nor hurt.
        """
        mode = self.weighting.get("mode", "symbol")
        if mode == "symbol":
            return self._symbol_weights(tokens, encoder)
        idf = self._get_idf()
        fb = float(self.weighting.get("idf_fallback", 1.0))
        scale = float(self.weighting.get("idf_scale", 1.0))
        if mode == "idf":
            return np.array([scale * idf.weight(t, fb) for t in tokens], dtype=np.float32)
        if mode == "symbol_idf":
            dw = float(self.weighting.get("default_weight", 1.0))
            specials = getattr(encoder, "special_tokens", frozenset())
            return np.array(
                [scale * idf.weight(t, fb) if is_symbol_subword(t, specials) else dw
                 for t in tokens],
                dtype=np.float32,
            )
        raise ValueError(
            f"unknown weighting mode {mode!r}; expected 'symbol', 'idf', or 'symbol_idf'"
        )

    def retrieve(self, state: str, accessible: set[str], k: int) -> list[tuple[str, float]]:
        if self._tok is None or self._off is None:
            raise RuntimeError("build_index must be called before retrieve")
        # Rebuild the accessible-token gather only when the accessible set changes (per theorem).
        if accessible is not self._cache_key:
            acc_uids = [uid for uid in accessible if uid in self._uid_to_idx]
            if not acc_uids:
                self._cache_key = accessible
                self._cache_D, self._cache_seg, self._cache_uids = None, None, []
            else:
                acc_idx = np.fromiter(
                    (self._uid_to_idx[u] for u in acc_uids), dtype=np.int64, count=len(acc_uids)
                )
                self._cache_D, self._cache_seg = self._gather(acc_idx)
                self._cache_uids = acc_uids
                self._cache_key = accessible
        if not self._cache_uids:
            return []

        enc = self._get_encoder()
        if self.weighting.get("enabled", False):
            # ON: token-aligned query encode so symbol tokens can be up-weighted (same query
            # vectors as OFF — the ablation differs only by the weights).
            e_q, q_tokens = enc.encode_query_with_tokens(state)     # [n_q, dim] + aligned strings
            weights = self._token_weights(q_tokens, enc)            # [n_q] (mode-dispatched)
        else:                                                       # OFF: unchanged from Phase 8
            e_q = enc.encode_queries([state])[0]                    # [n_q, dim], unit-norm
            weights = np.ones(e_q.shape[0], dtype=np.float32)       # [n_q]
        sims = e_q @ self._cache_D.T                                # [n_q, T_acc]
        maxs = np.maximum.reduceat(sims, self._cache_seg, axis=1)   # [n_q, m] per-premise max
        scores = weights @ maxs                                     # [m] weighted sum over q tokens
        candidates = list(zip(self._cache_uids, (float(s) for s in scores), strict=True))
        candidates.sort(key=lambda t: (-t[1], t[0]))               # score desc, uid tie-break
        return candidates[:k]
