"""Hermetic tests for the Phase-21 tactic-generation eval.

No torch, no GPU, no model download: a deterministic fake generator is injected, so the whole
orchestration (premise condition -> ReProver input assembly -> generation -> scoring -> JSON/CSV)
is exercised end to end on the mini fixture.

The load-bearing assertions here are the ones that would otherwise fail *silently* and still
produce a plausible table:
  * premises from the retrieval JSON really do reach the generator's input, in rank order;
  * records are joined to split examples with an `eid` checksum, and a mismatch aborts the run;
  * the `none` and `retrieval_json` conditions genuinely differ in what the generator sees.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from prooflens.data.corpus import load_corpus
from prooflens.data.proofs import load_split
from prooflens.eval.generate_eval import (
    align_retrieved,
    generate_eval,
    load_retrieval_records,
)

FIX = Path(__file__).parent / "fixtures" / "mini_benchmark"


# -- helpers --------------------------------------------------------------------------------------

class RecordingGenerator:
    """Fake generator: records every state it is asked about, returns fixed candidates."""

    def __init__(self, candidates: list[tuple[str, float]] | None = None) -> None:
        self.seen_states: list[str] = []
        default = [("simp", -0.1), ("ring", -0.9)]
        self._candidates = candidates if candidates is not None else default

    def generate(self, state: str, num_samples: int) -> list[tuple[str, float]]:
        self.seen_states.append(state)
        return list(self._candidates[:num_samples])


class EchoReferenceGenerator:
    """Returns the correct tactic ONLY when a given marker appears in its input context.

    This is how we prove the retrieval->generation wiring is real: with premises supplied the
    generator can "see" the marker and score 1.0; with no premises it cannot and scores 0.0. A
    broken pipeline that dropped the premises would fail this test rather than quietly reporting
    equal accuracy for every condition.
    """

    def __init__(self, marker: str, reference: str) -> None:
        self.marker = marker
        self.reference = reference

    def generate(self, state: str, num_samples: int) -> list[tuple[str, float]]:
        if self.marker in state:
            return [(self.reference, -0.01)]
        return [("sorry", -5.0)]


def _examples():
    corpus = load_corpus(str(FIX / "corpus.jsonl"))
    return corpus, list(load_split(str(FIX), "random", corpus, "test.json"))


def _base_config(name: str, tmp_path: Path, **overrides) -> dict:
    config = {
        "name": name,
        "generator": {
            "hf_id": "kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small",
            "max_inp_seq_len": 2300,
            "max_oup_seq_len": 512,
            "length_penalty": 0.0,
        },
        "premises": {"source": "none"},
        "data": {
            "corpus_path": str(FIX / "corpus.jsonl"),
            "splits_dir": str(FIX),
            "split_file": "test.json",
        },
        "eval": {"split": "random", "num_samples": 2, "k_list": [1, 2]},
        "seed": 42,
    }
    config.update(overrides)
    return config


def _write_retrieval_json(path: Path, examples, retrieved_for) -> None:
    """Write a minimal retrieval results file shaped like evaluate.py's output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "provenance": {"config_name": "fake_retriever", "retriever": "bm25"},
        "metrics": {},
        "examples": [
            {"eid": ex.eid, "gold": sorted(ex.gold), "retrieved": retrieved_for(ex)}
            for ex in examples
        ],
    }), encoding="utf-8")


# -- the loader now carries the ground-truth tactic ------------------------------------------------

def test_examples_carry_the_raw_tactic():
    _, examples = _examples()
    assert examples, "fixture must yield at least one example"
    assert all(ex.tactic for ex in examples)
    assert examples[0].tactic == "exact continuous_id"


# -- alignment ------------------------------------------------------------------------------------

def test_align_retrieved_happy_path(tmp_path):
    _, examples = _examples()
    p = tmp_path / "ret.json"
    _write_retrieval_json(p, examples, lambda ex: ["A::b@1,1"])
    _, records = load_retrieval_records(str(p))
    aligned = align_retrieved(examples, records)
    assert len(aligned) == len(examples)
    assert aligned[0] == ["A::b@1,1"]


def test_align_retrieved_rejects_length_mismatch(tmp_path):
    _, examples = _examples()
    p = tmp_path / "ret.json"
    _write_retrieval_json(p, examples[:-1], lambda ex: [])
    _, records = load_retrieval_records(str(p))
    with pytest.raises(ValueError, match="do not match split examples"):
        align_retrieved(examples, records)


def test_align_retrieved_rejects_eid_mismatch(tmp_path):
    _, examples = _examples()
    p = tmp_path / "ret.json"
    _write_retrieval_json(p, examples, lambda ex: [])
    _, records = load_retrieval_records(str(p))
    records[0]["eid"] = "some.other.theorem#0"
    with pytest.raises(ValueError, match="alignment broken at index 0"):
        align_retrieved(examples, records)


def test_load_retrieval_records_rejects_a_non_results_file(tmp_path):
    p = tmp_path / "junk.json"
    p.write_text(json.dumps({"nope": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a retrieval results file"):
        load_retrieval_records(str(p))


# -- the `none` condition -------------------------------------------------------------------------

def test_none_condition_feeds_the_bare_state(tmp_path):
    _, examples = _examples()
    gen = RecordingGenerator()
    config = _base_config("gen_none", tmp_path)
    generate_eval(config, results_dir=str(tmp_path), generator=gen)
    assert gen.seen_states == [ex.state for ex in examples], (
        "with no premises the generator must receive exactly the proof state"
    )


# -- the `retrieval_json` condition ---------------------------------------------------------------

def test_retrieved_premises_reach_the_generator_in_rank_order(tmp_path):
    corpus, examples = _examples()
    # two real premises from the fixture corpus, in a deliberate rank order
    uids = [p.uid for p in corpus.all_premises][:2]
    ret = tmp_path / "ret.json"
    _write_retrieval_json(ret, examples, lambda ex: uids)

    gen = RecordingGenerator()
    config = _base_config("gen_ret", tmp_path)
    config["premises"] = {
        "source": "retrieval_json",
        "retrieval_results": str(ret),
        "num_retrieved": 2,
    }
    generate_eval(config, results_dir=str(tmp_path), generator=gen)

    state_seen = gen.seen_states[0]
    p0 = corpus.premise_by_uid(uids[0])
    p1 = corpus.premise_by_uid(uids[1])
    assert p0 is not None and p1 is not None
    # ReProver prepends, so rank-1 ends up CLOSEST to the state and rank-2 before it.
    assert state_seen.endswith(examples[0].state)
    assert state_seen.index(p1.full_name) < state_seen.index(p0.full_name), (
        "rank-1 premise must sit nearer the state than rank-2 (ReProver prepend order)"
    )


def test_num_retrieved_caps_the_context(tmp_path):
    corpus, examples = _examples()
    uids = [p.uid for p in corpus.all_premises][:3]
    ret = tmp_path / "ret.json"
    _write_retrieval_json(ret, examples, lambda ex: uids)

    gen = RecordingGenerator()
    config = _base_config("gen_cap", tmp_path)
    config["premises"] = {
        "source": "retrieval_json", "retrieval_results": str(ret), "num_retrieved": 1,
    }
    generate_eval(config, results_dir=str(tmp_path), generator=gen)

    out = json.loads((tmp_path / "metrics" / "gen_cap_random.json").read_text(encoding="utf-8"))
    assert all(r["n_premises_offered"] == 1 for r in out["examples"])
    # tiny fixture premises, so the one offered also fits the budget
    assert all(r["n_premises_in_context"] == 1 for r in out["examples"])


def test_reports_fitted_premises_not_offered_when_the_budget_bites(tmp_path):
    """The reporting bug caught by the Phase-21 pilot: `offered` != `in_context`."""
    corpus, examples = _examples()
    uids = [p.uid for p in corpus.all_premises][:3]
    ret = tmp_path / "ret.json"
    _write_retrieval_json(ret, examples, lambda ex: uids)
    config = _base_config("gen_budget", tmp_path)
    config["premises"] = {
        "source": "retrieval_json", "retrieval_results": str(ret), "num_retrieved": 3,
    }
    # a budget so small that no premise can fit, but three are still offered
    config["generator"]["max_inp_seq_len"] = 1
    generate_eval(config, results_dir=str(tmp_path), generator=RecordingGenerator())

    out = json.loads((tmp_path / "metrics" / "gen_budget_random.json").read_text(encoding="utf-8"))
    rec = out["examples"][0]
    assert rec["n_premises_offered"] == 3
    assert rec["n_premises_in_context"] == 0, "nothing fits a 1-byte budget"


def test_missing_uids_are_counted_not_swallowed(tmp_path):
    _, examples = _examples()
    ret = tmp_path / "ret.json"
    _write_retrieval_json(ret, examples, lambda ex: ["Not/A/Real.lean::nope@1,1"])
    config = _base_config("gen_missing", tmp_path)
    config["premises"] = {
        "source": "retrieval_json", "retrieval_results": str(ret), "num_retrieved": 10,
    }
    generate_eval(config, results_dir=str(tmp_path), generator=RecordingGenerator())
    out = json.loads((tmp_path / "metrics" / "gen_missing_random.json").read_text(encoding="utf-8"))
    assert out["provenance"]["premise_condition"]["uids_missing_from_corpus"] == len(examples)


# -- the wiring invariant: premises must be able to change the outcome -----------------------------

def test_premises_change_the_score_relative_to_no_premises(tmp_path):
    """The whole study rests on this: supplying premises must be able to alter accuracy."""
    corpus, examples = _examples()
    ref = examples[0].tactic
    # marker = a premise name that only appears once premises are injected into the context
    uid = next(p.uid for p in corpus.all_premises if p.full_name == "le_refl")
    marker = "le_refl"
    ret = tmp_path / "ret.json"
    _write_retrieval_json(ret, examples, lambda ex: [uid])

    with_prem = _base_config("gen_with", tmp_path)
    with_prem["premises"] = {
        "source": "retrieval_json", "retrieval_results": str(ret), "num_retrieved": 1,
    }
    without = _base_config("gen_without", tmp_path)

    gen = EchoReferenceGenerator(marker, ref)
    a = generate_eval(with_prem, results_dir=str(tmp_path), generator=gen)
    b = generate_eval(without, results_dir=str(tmp_path), generator=gen)
    assert a["match@1"] > b["match@1"], (
        "injecting a premise that the generator can use must raise match@1 above the "
        "no-premise floor; if these are equal the premises are not reaching the model"
    )


# -- outputs --------------------------------------------------------------------------------------

def test_writes_json_with_provenance_and_per_example_candidates(tmp_path):
    _, examples = _examples()
    config = _base_config("gen_out", tmp_path)
    generate_eval(config, results_dir=str(tmp_path), generator=RecordingGenerator())

    out = json.loads((tmp_path / "metrics" / "gen_out_random.json").read_text(encoding="utf-8"))
    prov = out["provenance"]
    assert prov["task"] == "retrieval_augmented_tactic_generation"
    assert prov["n_examples"] == len(examples)
    assert prov["is_full_run"] is True
    assert prov["target"].startswith("remove_marks")
    assert "PREPENDED" in prov["input_format"]
    assert "NOT a proof-success rate" in prov["metric_definition"]
    # every candidate persisted -> match@k recomputable without the GPU
    assert out["examples"][0]["candidates"] == ["simp", "ring"]
    assert out["examples"][0]["scores"] == [-0.1, -0.9]
    assert set(out["metrics"]) == {
        "match@1", "match@2", "match@1_strict", "premise_name@1", "premise_name@2",
    }
    assert out["examples"][0]["gold_premise_names"], "gold short names must be persisted"


def test_appends_a_generation_summary_row(tmp_path):
    config = _base_config("gen_csv", tmp_path)
    generate_eval(config, results_dir=str(tmp_path), generator=RecordingGenerator())
    rows = list(csv.DictReader(
        (tmp_path / "metrics" / "generation_summary.csv").open(encoding="utf-8")
    ))
    assert len(rows) == 1
    assert rows[0]["config_name"] == "gen_csv"
    assert rows[0]["premise_source"] == "none"


def test_limit_marks_the_run_as_a_subset(tmp_path):
    config = _base_config("gen_limit", tmp_path)
    generate_eval(config, results_dir=str(tmp_path), generator=RecordingGenerator(), limit=1)
    out = json.loads((tmp_path / "metrics" / "gen_limit_random.json").read_text(encoding="utf-8"))
    assert out["provenance"]["n_examples"] == 1
    assert out["provenance"]["subset_limit"] == 1
    assert out["provenance"]["is_full_run"] is False


def test_scoring_is_correct_when_the_generator_is_right(tmp_path):
    _, examples = _examples()
    ref = examples[0].tactic
    config = _base_config("gen_hit", tmp_path)
    # top-1 is wrong, top-2 is the reference -> match@1 == 0, match@2 == 1 for example 0
    gen = RecordingGenerator([("definitely_wrong", -0.1), (ref, -0.2)])
    generate_eval(config, results_dir=str(tmp_path), generator=gen)
    out = json.loads((tmp_path / "metrics" / "gen_hit_random.json").read_text(encoding="utf-8"))
    rec0 = out["examples"][0]
    assert rec0["metrics"]["match@1"] == 0.0
    assert rec0["metrics"]["match@2"] == 1.0
    assert rec0["first_match_rank"] == 2


def test_unknown_premise_source_raises(tmp_path):
    config = _base_config("gen_bad", tmp_path)
    config["premises"] = {"source": "telepathy"}
    with pytest.raises(ValueError, match="unknown premises.source"):
        generate_eval(config, results_dir=str(tmp_path), generator=RecordingGenerator())


def test_retrieval_json_source_requires_a_path(tmp_path):
    config = _base_config("gen_bad2", tmp_path)
    config["premises"] = {"source": "retrieval_json"}
    with pytest.raises(ValueError, match="needs premises.retrieval_results"):
        generate_eval(config, results_dir=str(tmp_path), generator=RecordingGenerator())
